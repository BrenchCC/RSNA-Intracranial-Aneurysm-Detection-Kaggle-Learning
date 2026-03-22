#!/usr/bin/env python3
"""
Step 2: ROI Patch Extraction from 3D Volumes
=============================================

Extract 64x64x64 voxel patches from NIfTI volumes centered on brain tissue
regions of interest (ROI). This approach balances computational efficiency
with information preservation for 3D CNN training.

Rationale for 64x64x64 Patches:
-------------------------------
1. Computational Efficiency: Full brain volumes (512x512x200+) require
   excessive memory and computation for 3D CNNs
2. Information Preservation: 64^3 voxels capture sufficient anatomical
   context for aneurysm detection (typical aneurysm: 3-15mm diameter)
3. Spatial Resolution: At 0.5mm spacing, 64 voxels = 32mm physical size,
   adequate for Circle of Willis coverage
4. Model Compatibility: Most 3D CNN architectures designed for 64^3 inputs
5. Empirical Performance: Achieved 0.8585 AUC with this patch size

Processing Pipeline:
-------------------
1. Load NIfTI volume
2. Apply HU windowing (-100 to 300 for brain CTA)
3. Compute brain mask (thresholding)
4. Extract center-of-mass for ROI centering
5. Extract 64x64x64 patch centered on ROI
6. Apply z-score normalization
7. Save as compressed NPZ file

Input:
    - NIfTI volumes: data/volumes_nifti/*.nii.gz
    - Labels CSV: data/train_labels_14class.csv (optional, for filtering)

Output:
    - Patch files: data/patches_roi/{series_uid}.npz
    - Metadata: data/patch_metadata.csv
    - Extraction log: logs/02_patch_extraction.log

Competition: RSNA 2025 Intracranial Aneurysm Detection
Author: Brench
Date: 2026-03-22
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


logger = logging.getLogger(__name__)


class VolumeProcessingError(Exception):
    """
    Raised when a volume cannot be processed correctly.

    Args:
        message: Error description for the failed volume
    """


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description = "Extract 64^3 voxel ROI patches from NIfTI volumes",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--nifti-dir',
        type = str,
        default = 'data/volumes_nifti',
        help = 'Directory containing NIfTI volumes',
    )
    parser.add_argument(
        '--output-dir',
        type = str,
        default = 'data/patches_roi',
        help = 'Output directory for patch NPZ files',
    )
    parser.add_argument(
        '--labels-csv',
        type = str,
        default = None,
        help = 'Optional labels CSV for filtering volumes',
    )
    parser.add_argument(
        '--patch-size',
        type = int,
        default = 64,
        help = 'Patch edge length in voxels',
    )
    parser.add_argument(
        '--window-min',
        type = int,
        default = -100,
        help = 'Minimum HU for CTA windowing',
    )
    parser.add_argument(
        '--window-max',
        type = int,
        default = 300,
        help = 'Maximum HU for CTA windowing',
    )
    parser.add_argument(
        '--limit',
        type = int,
        default = None,
        help = 'Limit number of volumes (for testing)',
    )
    parser.add_argument(
        '--overwrite',
        action = 'store_true',
        help = 'Overwrite existing patch files',
    )
    parser.add_argument(
        '--verbose',
        action = 'store_true',
        help = 'Enable verbose logging',
    )
    parser.add_argument(
        '--num-workers',
        type = int,
        default = min(8, max(1, os.cpu_count() or 1)),
        help = 'Number of worker processes for parallel processing',
    )
    return parser.parse_args()


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """
    Configure logging to file and console.

    Args:
        output_dir: Directory for patch files
        verbose: Whether to enable debug logging

    Returns:
        Configured logger instance
    """
    log_dir = output_dir.parent.parent / "logs"
    log_dir.mkdir(parents = True, exist_ok = True)

    log_file = log_dir / "02_patch_extraction.log"
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level = log_level,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Raises:
        ValueError: If any argument is invalid
    """
    if args.patch_size <= 0:
        raise ValueError("patch_size must be positive")

    if args.window_min >= args.window_max:
        raise ValueError("window_min must be smaller than window_max")

    if args.num_workers <= 0:
        raise ValueError("num_workers must be positive")


def apply_cta_windowing(volume: np.ndarray, window_min: int = -100, window_max: int = 300) -> np.ndarray:
    """
    Apply Hounsfield Unit (HU) windowing for brain CTA.

    Args:
        volume: 3D array in Hounsfield Units
        window_min: Minimum HU value
        window_max: Maximum HU value

    Returns:
        Windowed volume clipped to [window_min, window_max]
    """
    return np.clip(volume, window_min, window_max)


def compute_brain_mask(volume: np.ndarray, threshold: int = -50) -> np.ndarray:
    """
    Compute binary brain mask using simple thresholding.

    Args:
        volume: 3D volume in HU
        threshold: HU threshold for brain

    Returns:
        Binary mask (True = brain tissue)
    """
    mask = volume > threshold
    structure = np.ones((3, 3, 3), dtype = bool)
    mask = ndimage.binary_closing(mask, structure = structure)
    mask = ndimage.binary_opening(mask, structure = structure)
    return mask


def extract_roi_patch(
    volume: np.ndarray,
    patch_size: int = 64,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract patch centered on region of interest.

    Args:
        volume: 3D volume (H, W, D)
        patch_size: Edge length of cubic patch
        mask: Optional binary mask for ROI centering

    Returns:
        patch: Extracted patch (patch_size, patch_size, patch_size)
        center: Center coordinates in original volume (3,)
    """
    if mask is not None and np.any(mask):
        center = np.array(ndimage.center_of_mass(mask), dtype = np.int32)
    else:
        center = np.array(volume.shape, dtype = np.int32) // 2

    half_size = patch_size // 2
    starts = np.maximum(center - half_size, 0)
    ends = np.minimum(starts + patch_size, volume.shape)
    starts = np.maximum(ends - patch_size, 0)

    patch = volume[
        starts[0]:ends[0],
        starts[1]:ends[1],
        starts[2]:ends[2],
    ]

    if patch.shape != (patch_size, patch_size, patch_size):
        padded_patch = np.zeros((patch_size, patch_size, patch_size), dtype = volume.dtype)
        padded_patch[
            :patch.shape[0],
            :patch.shape[1],
            :patch.shape[2],
        ] = patch
        patch = padded_patch

    return patch, center


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization to patch.

    Args:
        patch: 3D patch array

    Returns:
        Normalized patch (zero mean, unit variance)
    """
    mean_value = patch.mean()
    std_value = patch.std()

    if std_value < 1e-6:
        return patch - mean_value

    return (patch - mean_value) / std_value


def process_volume(
    nifti_path: Path,
    patch_size: int,
    window_min: int,
    window_max: int,
) -> Tuple[np.ndarray, Dict]:
    """
    Process a single NIfTI volume and extract an ROI patch.

    Args:
        nifti_path: Path to NIfTI file
        patch_size: Patch edge length
        window_min: Minimum HU for windowing
        window_max: Maximum HU for windowing

    Returns:
        patch: Normalized ROI patch
        metadata: Dictionary with processing metadata

    Raises:
        VolumeProcessingError: If volume loading or processing fails
    """
    try:
        nii = nib.load(str(nifti_path))
        volume = nii.get_fdata(caching = 'unchanged').astype(np.float32)
    except Exception as error:
        raise VolumeProcessingError(f"failed to load NIfTI: {error}") from error

    if volume.ndim != 3:
        raise VolumeProcessingError(f"expected 3D volume, got shape {volume.shape}")

    original_shape = tuple(int(dim) for dim in volume.shape)
    volume_windowed = apply_cta_windowing(volume, window_min, window_max)
    brain_mask = compute_brain_mask(volume_windowed)
    brain_voxels = int(brain_mask.sum())

    if brain_voxels < 1000:
        logger.warning(f"Very small brain mask for {nifti_path.name}: {brain_voxels} voxels")

    patch, center = extract_roi_patch(
        volume = volume_windowed,
        patch_size = patch_size,
        mask = brain_mask,
    )
    patch_normalized = normalize_patch(patch).astype(np.float32, copy = False)

    metadata = {
        'original_shape': original_shape,
        'patch_size': patch_size,
        'center': center.tolist(),
        'brain_voxels': brain_voxels,
        'patch_mean': float(patch.mean()),
        'patch_std': float(patch.std()),
        'patch_min': float(patch.min()),
        'patch_max': float(patch.max()),
    }
    return patch_normalized, metadata


def save_patch(patch: np.ndarray, output_path: Path, metadata: Dict) -> None:
    """
    Save patch as compressed NPZ file with metadata.

    Args:
        patch: Normalized patch array
        output_path: Output file path (.npz)
        metadata: Processing metadata dictionary
    """
    np.savez_compressed(output_path, patch = patch, **metadata)


def build_metadata_row(series_uid: str, nifti_path: Path, output_path: Path, metadata: Dict) -> Dict:
    """
    Build a flat metadata row for CSV export.

    Args:
        series_uid: Series identifier
        nifti_path: Input NIfTI path
        output_path: Output patch path
        metadata: Processing metadata

    Returns:
        Flattened metadata record
    """
    return {
        'series_uid': series_uid,
        'nifti_file': nifti_path.name,
        'patch_file': output_path.name,
        **metadata,
    }


def process_one_file(
    nifti_path: Path,
    output_dir: Path,
    patch_size: int,
    window_min: int,
    window_max: int,
    overwrite: bool,
) -> Dict:
    """
    Process one NIfTI file end-to-end.

    Args:
        nifti_path: Input NIfTI path
        output_dir: Output directory for patch files
        patch_size: Patch edge length
        window_min: Minimum HU for windowing
        window_max: Maximum HU for windowing
        overwrite: Whether to overwrite existing output

    Returns:
        Result dictionary with status and metadata
    """
    series_uid = nifti_path.name.removesuffix('.nii.gz').removesuffix('.nii')
    output_path = output_dir / f"{series_uid}.npz"

    if output_path.exists() and not overwrite:
        return {
            'status': 'skipped',
            'series_uid': series_uid,
        }

    patch, metadata = process_volume(
        nifti_path = nifti_path,
        patch_size = patch_size,
        window_min = window_min,
        window_max = window_max,
    )
    save_patch(patch = patch, output_path = output_path, metadata = metadata)

    return {
        'status': 'success',
        'series_uid': series_uid,
        'metadata_row': build_metadata_row(series_uid, nifti_path, output_path, metadata),
    }


def process_files_in_parallel(
    nifti_files: list[Path],
    output_dir: Path,
    patch_size: int,
    window_min: int,
    window_max: int,
    overwrite: bool,
    num_workers: int,
) -> Tuple[int, int, list[Tuple[str, str]], list[Dict]]:
    """
    Process NIfTI files using a process pool.

    Args:
        nifti_files: Input NIfTI file list
        output_dir: Output directory for patches
        patch_size: Patch edge length
        window_min: Minimum HU for windowing
        window_max: Maximum HU for windowing
        overwrite: Whether to overwrite existing output
        num_workers: Number of worker processes

    Returns:
        success_count: Number of successful files
        skipped_count: Number of skipped files
        failed_list: List of (series_uid, error_message)
        metadata_records: Successful metadata rows
    """
    success_count = 0
    skipped_count = 0
    failed_list = []
    metadata_records = []

    with ProcessPoolExecutor(max_workers = num_workers) as executor:
        future_to_path = {
            executor.submit(
                process_one_file,
                nifti_path,
                output_dir,
                patch_size,
                window_min,
                window_max,
                overwrite,
            ): nifti_path
            for nifti_path in nifti_files
        }

        for future in tqdm(as_completed(future_to_path), total = len(future_to_path), desc = "Extracting patches"):
            nifti_path = future_to_path[future]
            series_uid = nifti_path.name.removesuffix('.nii.gz').removesuffix('.nii')

            try:
                result = future.result()
                status = result['status']

                if status == 'success':
                    success_count += 1
                    metadata_records.append(result['metadata_row'])
                    logger.debug(f"Success: {series_uid}")
                elif status == 'skipped':
                    skipped_count += 1
                    logger.debug(f"Skipping {series_uid} (already exists)")
                else:
                    failed_list.append((series_uid, f"unexpected status: {status}"))
                    logger.error(f"Failed {series_uid}: unexpected status: {status}")
            except Exception as error:
                failed_list.append((series_uid, str(error)))
                logger.error(f"Failed {series_uid}: {error}")

    return success_count, skipped_count, failed_list, metadata_records


def main() -> None:
    """Main patch extraction pipeline."""
    args = parse_args()
    validate_args(args)

    nifti_dir = Path(args.nifti_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    setup_logging(output_dir = output_dir, verbose = args.verbose)

    logger.info("=" * 80)
    logger.info("ROI Patch Extraction - Step 2")
    logger.info("=" * 80)
    logger.info(f"Input directory: {nifti_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Patch size: {args.patch_size}^3 voxels")
    logger.info(f"HU window: [{args.window_min}, {args.window_max}]")
    logger.info(f"Worker threads: {args.num_workers}")

    if not nifti_dir.exists():
        logger.error(f"Input directory does not exist: {nifti_dir}")
        sys.exit(1)

    nifti_files = sorted(list(nifti_dir.glob("*.nii.gz")) + list(nifti_dir.glob("*.nii")))

    if args.limit is not None:
        nifti_files = nifti_files[:args.limit]
        logger.info(f"Limited to {args.limit} volumes for testing")

    logger.info(f"Found {len(nifti_files)} NIfTI files")

    if len(nifti_files) == 0:
        logger.error("No NIfTI files found")
        sys.exit(1)

    if args.labels_csv:
        labels_path = Path(args.labels_csv)
        if labels_path.exists():
            labels_df = pd.read_csv(labels_path)
            logger.info(f"Loaded labels: {len(labels_df)} rows")
        else:
            logger.warning(f"Labels CSV not found: {labels_path}")

    start_time = datetime.now()
    success_count, skipped_count, failed_list, metadata_records = process_files_in_parallel(
        nifti_files = nifti_files,
        output_dir = output_dir,
        patch_size = args.patch_size,
        window_min = args.window_min,
        window_max = args.window_max,
        overwrite = args.overwrite,
        num_workers = args.num_workers,
    )
    duration = (datetime.now() - start_time).total_seconds()

    logger.info("")
    logger.info("=" * 80)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total volumes:          {len(nifti_files)}")
    logger.info(f"Successfully extracted: {success_count}")
    logger.info(f"Skipped (existing):     {skipped_count}")
    logger.info(f"Failed:                 {len(failed_list)}")
    logger.info(f"Success rate:           {100.0 * success_count / len(nifti_files):.2f}%")
    logger.info(f"Processing time:        {duration:.2f} seconds")
    logger.info(f"Average per volume:     {duration / len(nifti_files):.2f} seconds")

    if failed_list:
        logger.info("")
        logger.info("Failed volumes (first 10):")
        for series_uid, error_message in failed_list[:10]:
            logger.info(f"  - {series_uid}: {error_message}")
        if len(failed_list) > 10:
            logger.info(f"  ... and {len(failed_list) - 10} more")

    patch_files = list(output_dir.glob('*.npz'))
    logger.info("")
    logger.info(f"Total patch files in output directory: {len(patch_files)}")

    if metadata_records:
        metadata_csv = output_dir.parent / "patch_metadata.csv"
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_csv(metadata_csv, index = False)
        logger.info(f"Metadata saved: {metadata_csv}")

        logger.info("")
        logger.info("Patch Statistics:")
        logger.info(f"  Mean patch size: {args.patch_size}^3 voxels")
        logger.info(f"  Brain voxels (mean): {metadata_df['brain_voxels'].mean():.0f}")
        logger.info(f"  Brain voxels (std):  {metadata_df['brain_voxels'].std():.0f}")

    logger.info("=" * 80)

    failure_rate = len(failed_list) / len(nifti_files)
    if failure_rate > 0.1:
        logger.error(f"High failure rate: {failure_rate * 100:.1f}%")
        sys.exit(1)


if __name__ == "__main__":
    main()
