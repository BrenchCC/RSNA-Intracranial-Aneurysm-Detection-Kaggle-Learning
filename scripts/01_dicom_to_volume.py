#!/usr/bin/env python3
"""
Step 1: DICOM to NIfTI Volume Conversion
==========================================

Convert raw DICOM series from the RSNA 2025 Intracranial Aneurysm Detection
competition to standardized NIfTI format volumes.

This script:
1. Reads DICOM series from organized directories
2. Sorts slices by spatial position (ImagePositionPatient)
3. Converts to Hounsfield Units (HU)
4. Preserves spatial metadata (spacing, affine transform)
5. Saves as compressed NIfTI (.nii.gz) for efficient storage

Input:
    - Raw DICOM series directories (default: data/raw/series/)
    - Each series in format: series/{series_uid}/*.dcm

Output:
    - NIfTI volumes: data/volumes_nifti/{series_uid}.nii.gz
    - Conversion log: logs/01_dicom_conversion.log
    - Success/failure report

Competition: RSNA 2025 Intracranial Aneurysm Detection
Link: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection
Author: Brench
Date: 2026-03-17
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from multiprocessing import Queue, Event

import nibabel as nib
import numpy as np
import pydicom
from tqdm import tqdm


logger = logging.getLogger(__name__)


class StopRequestedError(RuntimeError):
    """Raised when a stop request is detected in the main process."""


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description = "Convert DICOM series to NIfTI volumes",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--series-dir",
        type = str,
        default = "data/raw/series",
        help = "Directory containing DICOM series folders"
    )
    parser.add_argument(
        "--output-dir",
        type = str,
        default = "data/volumes_nifti",
        help = "Output directory for NIfTI volumes"
    )
    parser.add_argument(
        "--limit",
        type = int,
        default = None,
        help = "Limit number of series (for testing)"
    )
    parser.add_argument(
        "--overwrite",
        action = "store_true",
        help = "Overwrite existing NIfTI files"
    )
    parser.add_argument(
        "--verbose",
        action = "store_true",
        help = "Enable verbose logging (DEBUG level)"
    )
    parser.add_argument(
        "--num-workers",
        type = int,
        default = max(1, os.cpu_count() or 1),
        help = "Number of worker processes for parallel series conversion"
    )
    parser.add_argument(
        "--queue-size",
        type = int,
        default = 0,
        help = "Task queue size, 0 means auto (2 x workers)"
    )
    parser.add_argument(
        "--stop-file",
        type = str,
        default = "",
        help = "Optional file path; if this file appears, processing stops gracefully"
    )
    parser.add_argument(
        "--graceful-stop-timeout",
        type = float,
        default = 5.0,
        help = "Seconds to wait for workers to exit after stop request before terminate()"
    )
    return parser.parse_args()


def setup_logging(output_dir: Path, verbose: bool) -> logging.Logger:
    """
    Configure logging to file and console.

    Args:
        output_dir: Directory for output files.
        verbose: Whether to enable DEBUG level logging.

    Returns:
        Configured logger instance.
    """
    log_dir = output_dir.parent.parent / "logs"
    log_dir.mkdir(parents = True, exist_ok = True)

    log_file = log_dir / "01_dicom_conversion.log"
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level = log_level,
        format = "%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def sort_dicom_slices(slices: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
    """
    Sort DICOM slices by ImagePositionPatient Z-coordinate.

    Args:
        slices: List of pydicom Dataset objects.

    Returns:
        Sorted list of DICOM slices.

    Raises:
        ValueError: If no slices have valid ImagePositionPatient.
    """
    slices_with_pos = []

    for slice_ds in slices:
        try:
            z_pos = float(slice_ds.ImagePositionPatient[2])
            slices_with_pos.append((z_pos, slice_ds))
        except (AttributeError, IndexError, ValueError):
            continue

    if not slices_with_pos:
        raise ValueError("No DICOM slices contain valid ImagePositionPatient")

    slices_with_pos.sort(key = lambda item: item[0])
    return [slice_ds for _, slice_ds in slices_with_pos]


def process_dicom_series(series_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load DICOM series and convert to 3D volume with spatial metadata.

    Args:
        series_dir: Path to DICOM series directory.

    Returns:
        Tuple of volume array and affine matrix.

    Raises:
        ValueError: If no valid DICOM files are found or sorting fails.
    """
    dicom_files = list(series_dir.glob("*.dcm"))

    if not dicom_files:
        raise ValueError(f"No .dcm files found in {series_dir}")

    slices = []
    read_errors = 0

    for dcm_file in dicom_files:
        try:
            slices.append(pydicom.dcmread(str(dcm_file), force = True))
        except Exception:
            read_errors += 1

    if not slices:
        raise ValueError(f"No valid DICOM slices (read errors: {read_errors})")

    try:
        slices = sort_dicom_slices(slices)
    except ValueError as exc:
        raise ValueError(f"Failed to sort slices: {exc}") from exc

    volume = np.stack([slice_ds.pixel_array for slice_ds in slices], axis = -1)

    try:
        slope = float(slices[0].RescaleSlope)
        intercept = float(slices[0].RescaleIntercept)
        volume = volume.astype(np.float32) * slope + intercept
    except (AttributeError, ValueError):
        volume = volume.astype(np.float32, copy = False)

    try:
        pixel_spacing = slices[0].PixelSpacing
        px_x, px_y = float(pixel_spacing[0]), float(pixel_spacing[1])

        if len(slices) > 1:
            z1 = float(slices[0].ImagePositionPatient[2])
            z2 = float(slices[1].ImagePositionPatient[2])
            px_z = abs(z2 - z1)
        else:
            px_z = float(slices[0].get("SliceThickness", 1.0))

        affine = np.diag([px_x, px_y, px_z, 1.0])
    except (AttributeError, IndexError, ValueError):
        affine = np.eye(4)

    return volume, affine


def convert_one_series(
    series_folder: Path,
    output_dir: Path,
    overwrite: bool
) -> Tuple[str, str, Optional[str]]:
    """
    Convert a single DICOM series to NIfTI.

    Args:
        series_folder: Input DICOM series directory.
        output_dir: Output directory for NIfTI volumes.
        overwrite: Whether to overwrite existing output files.

    Returns:
        Tuple of status, series UID, and optional error message.
    """
    series_uid = series_folder.name
    output_path = output_dir / f"{series_uid}.nii.gz"

    if output_path.exists() and not overwrite:
        return "skipped", series_uid, None

    try:
        volume, affine = process_dicom_series(series_folder)
        nifti_img = nib.Nifti1Image(volume, affine)
        nib.save(nifti_img, str(output_path))
        return "success", series_uid, None
    except Exception as exc:
        return "failed", series_uid, str(exc)


def worker_main(
    task_queue: Queue,
    result_queue: Queue,
    stop_event: Event,
    output_dir_str: str,
    overwrite: bool
) -> None:
    """
    Consume series tasks from the queue and write results to the result queue.

    Args:
        task_queue: Queue containing Path-like series folder strings or None sentinel.
        result_queue: Queue used to return conversion results.
        stop_event: Shared event that signals stop request.
        output_dir_str: Output directory string.
        overwrite: Whether to overwrite existing files.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    output_dir = Path(output_dir_str)

    while not stop_event.is_set():
        try:
            task_item = task_queue.get(timeout = 0.5)
        except queue.Empty:
            continue

        if task_item is None:
            break

        series_folder = Path(task_item)
        result = convert_one_series(series_folder, output_dir, overwrite)
        result_queue.put(result)



def save_conversion_report(
    output_file: Path,
    total: int,
    success: int,
    skipped: int,
    failed: List[Tuple[str, str]],
    duration_sec: float,
    stopped_early: bool
) -> None:
    """
    Save conversion summary to JSON file.

    Args:
        output_file: Path to JSON report file.
        total: Total series discovered.
        success: Successfully converted series count.
        skipped: Already existing series count.
        failed: Failed series with error messages.
        duration_sec: Processing duration in seconds.
        stopped_early: Whether processing stopped before all tasks completed.
    """
    processed_total = success + skipped + len(failed)
    report = {
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "total_series": total,
            "processed_series": processed_total,
            "successful": success,
            "skipped_existing": skipped,
            "failed": len(failed),
            "stopped_early": stopped_early
        },
        "success_rate": f"{100.0 * success / processed_total:.2f}%" if processed_total > 0 else "N/A",
        "processing_time_sec": round(duration_sec, 2),
        "failed_series": [
            {"series_uid": uid, "error": error}
            for uid, error in failed[:50]
        ]
    }

    with open(output_file, "w") as file_obj:
        json.dump(report, file_obj, indent = 2)



def stop_requested(stop_file: Optional[Path]) -> bool:
    """
    Check whether an external stop request has been issued.

    Args:
        stop_file: Optional file path used as an external stop signal.

    Returns:
        True if stop file exists, otherwise False.
    """
    return bool(stop_file and stop_file.exists())



def request_stop(
    stop_event: Event,
    workers: List[mp.Process],
    task_queue: Queue,
    graceful_stop_timeout: float,
    active_logger: logging.Logger
) -> None:
    """
    Request workers to stop and clean them up.

    Args:
        stop_event: Shared stop event.
        workers: Worker process list.
        task_queue: Task queue to unblock workers.
        graceful_stop_timeout: Seconds to wait before terminate().
        active_logger: Logger instance.
    """
    stop_event.set()

    for _ in workers:
        try:
            task_queue.put_nowait(None)
        except queue.Full:
            break

    deadline = time.time() + max(0.0, graceful_stop_timeout)
    for worker in workers:
        remaining = deadline - time.time()
        worker.join(timeout = max(0.0, remaining))

    for worker in workers:
        if worker.is_alive():
            active_logger.warning("Force terminating worker: %s", worker.name)
            worker.terminate()

    for worker in workers:
        worker.join(timeout = 1.0)



def main() -> None:
    """Run the DICOM to NIfTI conversion pipeline."""
    args = parse_args()

    series_dir = Path(args.series_dir)
    output_dir = Path(args.output_dir)
    stop_file_path = Path(args.stop_file) if args.stop_file else None
    output_dir.mkdir(parents = True, exist_ok = True)

    active_logger = setup_logging(output_dir, args.verbose)

    active_logger.info("=" * 80)
    active_logger.info("DICOM to NIfTI Conversion - Step 1")
    active_logger.info("=" * 80)
    active_logger.info("Input directory: %s", series_dir)
    active_logger.info("Output directory: %s", output_dir)
    active_logger.info("Overwrite existing: %s", args.overwrite)
    active_logger.info("Number of worker processes: %s", args.num_workers)
    active_logger.info("Stop file: %s", stop_file_path or "disabled")

    if not series_dir.exists():
        active_logger.error("Input directory does not exist: %s", series_dir)
        sys.exit(1)

    series_folders = sorted([item for item in series_dir.iterdir() if item.is_dir()])

    if args.limit:
        series_folders = series_folders[:args.limit]
        active_logger.info("Limited to %s series for testing", args.limit)

    active_logger.info("Found %s series directories", len(series_folders))

    if len(series_folders) == 0:
        active_logger.error("No series directories found")
        sys.exit(1)

    start_time = datetime.now()
    success_count = 0
    skipped_count = 0
    failed_list: List[Tuple[str, str]] = []
    stopped_early = False

    worker_count = min(max(1, args.num_workers), len(series_folders))
    queue_size = args.queue_size if args.queue_size > 0 else worker_count * 2

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize = queue_size)
    result_queue = ctx.Queue()
    stop_event = ctx.Event()
    workers: List[mp.Process] = []

    try:
        for worker_index in range(worker_count):
            worker = ctx.Process(
                target = worker_main,
                args = (
                    task_queue,
                    result_queue,
                    stop_event,
                    str(output_dir),
                    args.overwrite,
                ),
                name = f"converter-{worker_index:02d}"
            )
            worker.start()
            workers.append(worker)

        submitted_count = 0
        completed_count = 0
        series_iter = iter(series_folders)

        progress_bar = tqdm(total = len(series_folders), desc = "Converting DICOM to NIfTI")
        try:
            while completed_count < len(series_folders):
                while not stop_event.is_set() and submitted_count < len(series_folders):
                    if stop_requested(stop_file_path):
                        raise StopRequestedError(f"Stop file detected: {stop_file_path}")

                    try:
                        next_series = next(series_iter)
                    except StopIteration:
                        break

                    try:
                        task_queue.put(str(next_series), timeout = 0.2)
                        submitted_count += 1
                    except queue.Full:
                        break

                try:
                    status, series_uid, error_msg = result_queue.get(timeout = 0.5)
                except queue.Empty:
                    if stop_requested(stop_file_path):
                        raise StopRequestedError(f"Stop file detected: {stop_file_path}")

                    if all(not worker.is_alive() for worker in workers) and completed_count >= submitted_count:
                        break
                    continue

                completed_count += 1
                progress_bar.update(1)

                if status == "success":
                    success_count += 1
                    active_logger.debug("Success: %s", series_uid)
                elif status == "skipped":
                    skipped_count += 1
                    active_logger.debug("Skipping %s (already exists)", series_uid)
                else:
                    failed_list.append((series_uid, error_msg or "Unknown error"))
                    active_logger.error("Failed %s: %s", series_uid, error_msg)

        finally:
            progress_bar.close()

        for _ in workers:
            task_queue.put(None)

        for worker in workers:
            worker.join()

    except KeyboardInterrupt:
        stopped_early = True
        active_logger.warning("KeyboardInterrupt received, stopping workers...")
        request_stop(stop_event, workers, task_queue, args.graceful_stop_timeout, active_logger)
    except StopRequestedError as exc:
        stopped_early = True
        active_logger.warning("%s", exc)
        request_stop(stop_event, workers, task_queue, args.graceful_stop_timeout, active_logger)
    finally:
        for queue_obj in (task_queue, result_queue):
            try:
                queue_obj.close()
            except Exception:
                pass

    duration = (datetime.now() - start_time).total_seconds()
    processed_count = success_count + skipped_count + len(failed_list)

    active_logger.info("")
    active_logger.info("=" * 80)
    active_logger.info("CONVERSION SUMMARY")
    active_logger.info("=" * 80)
    active_logger.info("Total series:           %s", len(series_folders))
    active_logger.info("Processed series:       %s", processed_count)
    active_logger.info("Successfully converted: %s", success_count)
    active_logger.info("Skipped (existing):     %s", skipped_count)
    active_logger.info("Failed:                 %s", len(failed_list))
    active_logger.info("Stopped early:          %s", stopped_early)
    active_logger.info(
        "Success rate:           %s",
        f"{100.0 * success_count / processed_count:.2f}%" if processed_count > 0 else "N/A"
    )
    active_logger.info("Processing time:        %.2f seconds", duration)
    if processed_count > 0:
        active_logger.info("Average per series:     %.2f seconds", duration / processed_count)

    if failed_list:
        active_logger.info("")
        active_logger.info("Failed series (first 10):")
        for series_uid, error_msg in failed_list[:10]:
            active_logger.info("  - %s: %s", series_uid, error_msg)
        if len(failed_list) > 10:
            active_logger.info("  ... and %s more (see log file)", len(failed_list) - 10)

    nifti_files = list(output_dir.glob("*.nii.gz"))
    active_logger.info("")
    active_logger.info("Total NIfTI files in output directory: %s", len(nifti_files))

    report_file = output_dir.parent / "conversion_report.json"
    save_conversion_report(
        report_file,
        len(series_folders),
        success_count,
        skipped_count,
        failed_list,
        duration,
        stopped_early
    )
    active_logger.info("Detailed report saved: %s", report_file)
    active_logger.info("=" * 80)

    if not stopped_early and len(series_folders) > 0:
        failure_rate = len(failed_list) / len(series_folders)
        if failure_rate > 0.1:
            active_logger.error("High failure rate: %.1f%%", failure_rate * 100)
            sys.exit(1)


if __name__ == "__main__":
    main()
