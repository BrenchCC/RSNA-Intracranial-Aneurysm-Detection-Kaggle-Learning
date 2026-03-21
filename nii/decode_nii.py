import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = 'Decode one NIfTI file and export preview images.')
    parser.add_argument('--nii-path', type = str, required = True, help = 'Path to a .nii or .nii.gz file.')
    parser.add_argument('--output-dir', type = str, default = 'nii/decoded', help = 'Output root directory.')
    parser.add_argument('--num-grid-slices', type = int, default = 9, help = 'Number of axial slices in the grid image.')
    return parser.parse_args()


def _normalize_slice(slice_2d):
    """Normalize a 2D slice into [0, 1] for visualization.

    Args:
        slice_2d (np.ndarray): Input 2D array.

    Returns:
        np.ndarray: Normalized 2D image.
    """
    low = float(np.percentile(slice_2d, 1.0))
    high = float(np.percentile(slice_2d, 99.0))
    clipped = np.clip(slice_2d, low, high)
    normalized = (clipped - low) / (high - low + 1e-6)
    return normalized.astype(np.float32)


def _save_png(image_2d, out_path, title):
    """Save one 2D image as PNG.

    Args:
        image_2d (np.ndarray): 2D image.
        out_path (Path): Output image path.
        title (str): Figure title.

    Returns:
        None
    """
    out_path.parent.mkdir(parents = True, exist_ok = True)

    fig, ax = plt.subplots(figsize = (6, 6))
    ax.imshow(image_2d, cmap = 'gray')
    ax.set_title(title)
    ax.axis('off')
    fig.savefig(out_path, dpi = 160, bbox_inches = 'tight')
    plt.close(fig)


def _save_grid(volume, out_path, num_slices):
    """Save an axial slice grid from 3D volume.

    Args:
        volume (np.ndarray): 3D volume in x-y-z layout.
        out_path (Path): Output image path.
        num_slices (int): Number of slices in grid.

    Returns:
        None
    """
    out_path.parent.mkdir(parents = True, exist_ok = True)
    num_slices = max(1, int(num_slices))

    z = volume.shape[2]
    indices = np.linspace(0, z - 1, num = num_slices, dtype = int)

    cols = 3
    rows = int(np.ceil(num_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize = (cols * 3.2, rows * 3.2))
    axes = np.array(axes).reshape(-1)

    for idx, ax in enumerate(axes):
        if idx < num_slices:
            z_idx = int(indices[idx])
            image = _normalize_slice(volume[:, :, z_idx].T)
            ax.imshow(image, cmap = 'gray')
            ax.set_title(f'axial z={z_idx}', fontsize = 9)
        ax.axis('off')

    fig.suptitle('Axial Slice Grid', fontsize = 12)
    fig.savefig(out_path, dpi = 160, bbox_inches = 'tight')
    plt.close(fig)


def decode_nii(nii_path, output_dir, num_grid_slices):
    """Decode a NIfTI file and export metadata and images.

    Args:
        nii_path (str): Path to input NIfTI file.
        output_dir (str): Output root directory.
        num_grid_slices (int): Number of slices for axial grid.

    Returns:
        dict: Output summary.
    """
    nii_file = Path(nii_path)
    if not nii_file.exists():
        raise FileNotFoundError(f'NIfTI not found: {nii_file}')

    case_name = nii_file.name.replace('.nii.gz', '').replace('.nii', '')
    case_out_dir = Path(output_dir) / case_name
    case_out_dir.mkdir(parents = True, exist_ok = True)

    img = nib.load(str(nii_file))
    volume = img.get_fdata(dtype = np.float32)

    x_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    z_mid = volume.shape[2] // 2

    axial = _normalize_slice(volume[:, :, z_mid].T)
    coronal = _normalize_slice(volume[:, y_mid, :].T)
    sagittal = _normalize_slice(volume[x_mid, :, :].T)

    _save_png(axial, case_out_dir / 'axial_mid.png', 'Axial Mid Slice')
    _save_png(coronal, case_out_dir / 'coronal_mid.png', 'Coronal Mid Slice')
    _save_png(sagittal, case_out_dir / 'sagittal_mid.png', 'Sagittal Mid Slice')
    _save_grid(volume = volume, out_path = case_out_dir / 'axial_grid.png', num_slices = num_grid_slices)

    voxel_spacing = [float(v) for v in img.header.get_zooms()[:3]]

    meta = {
        'nii_path': str(nii_file),
        'shape_xyz': list(volume.shape),
        'dtype': str(volume.dtype),
        'voxel_spacing': voxel_spacing,
        'affine': img.affine.tolist(),
        'intensity_stats': {
            'min': float(np.min(volume)),
            'max': float(np.max(volume)),
            'mean': float(np.mean(volume)),
            'std': float(np.std(volume))
        },
        'images': {
            'axial_mid': str(case_out_dir / 'axial_mid.png'),
            'coronal_mid': str(case_out_dir / 'coronal_mid.png'),
            'sagittal_mid': str(case_out_dir / 'sagittal_mid.png'),
            'axial_grid': str(case_out_dir / 'axial_grid.png')
        }
    }

    with open(case_out_dir / 'meta.json', 'w', encoding = 'utf-8') as file:
        json.dump(meta, file, indent = 2, ensure_ascii = False)

    logger.info('=' * 80)
    logger.info('Decoded NIfTI: %s', nii_file)
    logger.info('=' * 80)
    logger.info('Output directory: %s', case_out_dir)

    return meta


def main():
    """Program entry point.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    decode_nii(
        nii_path = args.nii_path,
        output_dir = args.output_dir,
        num_grid_slices = args.num_grid_slices
    )


if __name__ == '__main__':
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
