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
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = 'Generate vessel-focused visualization from one NIfTI file.')
    parser.add_argument('--nii-path', type = str, required = True, help = 'Path to .nii or .nii.gz file.')
    parser.add_argument('--output-dir', type = str, default = 'nii/decoded', help = 'Output root directory.')
    parser.add_argument('--vessel-low-hu', type = float, default = 120.0, help = 'Lower HU threshold for vessel candidates.')
    parser.add_argument('--vessel-high-hu', type = float, default = 700.0, help = 'Upper HU threshold for vessel candidates.')
    parser.add_argument('--num-grid-slices', type = int, default = 9, help = 'Number of axial slices in overlay grid.')
    return parser.parse_args()


def _norm_2d(image_2d):
    """Normalize a 2D image into [0, 1].

    Args:
        image_2d (np.ndarray): Input 2D image.

    Returns:
        np.ndarray: Normalized image.
    """
    low = float(np.percentile(image_2d, 1.0))
    high = float(np.percentile(image_2d, 99.0))
    clipped = np.clip(image_2d, low, high)
    return ((clipped - low) / (high - low + 1e-6)).astype(np.float32)


def _save_overlay(base_2d, mask_2d, out_path, title):
    """Save base image and mask overlay.

    Args:
        base_2d (np.ndarray): Base image.
        mask_2d (np.ndarray): Binary mask.
        out_path (Path): Output image path.
        title (str): Figure title.

    Returns:
        None
    """
    out_path.parent.mkdir(parents = True, exist_ok = True)

    fig, ax = plt.subplots(figsize = (6, 6))
    ax.imshow(base_2d, cmap = 'gray')
    mask_alpha = np.where(mask_2d > 0, 0.45, 0.0)
    ax.imshow(mask_2d, cmap = 'autumn', alpha = mask_alpha)
    ax.set_title(title)
    ax.axis('off')
    fig.savefig(out_path, dpi = 170, bbox_inches = 'tight')
    plt.close(fig)


def _save_overlay_grid(volume_xyz, mask_xyz, out_path, num_slices):
    """Save axial slice grid with mask overlay.

    Args:
        volume_xyz (np.ndarray): 3D image in x-y-z.
        mask_xyz (np.ndarray): 3D mask in x-y-z.
        out_path (Path): Output image path.
        num_slices (int): Number of grid slices.

    Returns:
        None
    """
    out_path.parent.mkdir(parents = True, exist_ok = True)
    num_slices = max(1, int(num_slices))

    z = volume_xyz.shape[2]
    indices = np.linspace(0, z - 1, num = num_slices, dtype = int)

    cols = 3
    rows = int(np.ceil(num_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize = (cols * 3.3, rows * 3.3))
    axes = np.array(axes).reshape(-1)

    for idx, ax in enumerate(axes):
        if idx < num_slices:
            zi = int(indices[idx])
            base = _norm_2d(volume_xyz[:, :, zi].T)
            mask = mask_xyz[:, :, zi].T.astype(np.float32)
            ax.imshow(base, cmap = 'gray')
            ax.imshow(mask, cmap = 'autumn', alpha = np.where(mask > 0, 0.4, 0.0))
            ax.set_title(f'axial z={zi}', fontsize = 8)
        ax.axis('off')

    fig.suptitle('Vessel Overlay Axial Grid', fontsize = 12)
    fig.savefig(out_path, dpi = 170, bbox_inches = 'tight')
    plt.close(fig)


def _largest_component(mask_xyz):
    """Keep the largest connected component in a binary mask.

    Args:
        mask_xyz (np.ndarray): Binary mask.

    Returns:
        np.ndarray: Largest-component mask.
    """
    labeled, num = ndi.label(mask_xyz)
    if num == 0:
        return np.zeros_like(mask_xyz, dtype = bool)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep_label = int(np.argmax(counts))
    return labeled == keep_label


def _build_vessel_mask(volume_xyz, low_hu, high_hu):
    """Build a coarse vessel mask using intensity and morphology rules.

    Args:
        volume_xyz (np.ndarray): Input CT-like volume.
        low_hu (float): Lower HU threshold.
        high_hu (float): Upper HU threshold.

    Returns:
        np.ndarray: Binary vessel mask in x-y-z.
    """
    body_mask = volume_xyz > -300.0
    body_mask = _largest_component(mask_xyz = body_mask)

    vessel_raw = (volume_xyz >= low_hu) & (volume_xyz <= high_hu) & body_mask

    vessel_clean = ndi.binary_opening(vessel_raw, structure = np.ones((2, 2, 2), dtype = bool))
    vessel_clean = ndi.binary_closing(vessel_clean, structure = np.ones((2, 2, 2), dtype = bool))

    labeled, num = ndi.label(vessel_clean)
    if num == 0:
        return np.zeros_like(vessel_clean, dtype = np.uint8)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    min_voxel = 80
    keep_labels = np.where(counts >= min_voxel)[0]
    vessel_mask = np.isin(labeled, keep_labels)

    return vessel_mask.astype(np.uint8)


def view_vessels(nii_path, output_dir, vessel_low_hu, vessel_high_hu, num_grid_slices):
    """Run vessel visualization pipeline for one NIfTI file.

    Args:
        nii_path (str): Path to NIfTI file.
        output_dir (str): Output root directory.
        vessel_low_hu (float): Lower HU threshold.
        vessel_high_hu (float): Upper HU threshold.
        num_grid_slices (int): Grid slice count.

    Returns:
        dict: Summary metadata.
    """
    nii_file = Path(nii_path)
    if not nii_file.exists():
        raise FileNotFoundError(f'NIfTI file not found: {nii_file}')

    case_name = nii_file.name.replace('.nii.gz', '').replace('.nii', '')
    case_out_dir = Path(output_dir) / case_name / 'vessel_view'
    case_out_dir.mkdir(parents = True, exist_ok = True)

    img = nib.load(str(nii_file))
    volume_xyz = img.get_fdata(dtype = np.float32)

    vessel_mask = _build_vessel_mask(
        volume_xyz = volume_xyz,
        low_hu = vessel_low_hu,
        high_hu = vessel_high_hu
    )

    x_mid = volume_xyz.shape[0] // 2
    y_mid = volume_xyz.shape[1] // 2
    z_mid = volume_xyz.shape[2] // 2

    axial_base = _norm_2d(volume_xyz[:, :, z_mid].T)
    coronal_base = _norm_2d(volume_xyz[:, y_mid, :].T)
    sagittal_base = _norm_2d(volume_xyz[x_mid, :, :].T)

    axial_mask = vessel_mask[:, :, z_mid].T
    coronal_mask = vessel_mask[:, y_mid, :].T
    sagittal_mask = vessel_mask[x_mid, :, :].T

    _save_overlay(
        base_2d = axial_base,
        mask_2d = axial_mask,
        out_path = case_out_dir / 'axial_vessel_overlay.png',
        title = 'Axial Vessel Overlay'
    )
    _save_overlay(
        base_2d = coronal_base,
        mask_2d = coronal_mask,
        out_path = case_out_dir / 'coronal_vessel_overlay.png',
        title = 'Coronal Vessel Overlay'
    )
    _save_overlay(
        base_2d = sagittal_base,
        mask_2d = sagittal_mask,
        out_path = case_out_dir / 'sagittal_vessel_overlay.png',
        title = 'Sagittal Vessel Overlay'
    )

    _save_overlay_grid(
        volume_xyz = volume_xyz,
        mask_xyz = vessel_mask,
        out_path = case_out_dir / 'axial_vessel_overlay_grid.png',
        num_slices = num_grid_slices
    )

    mip = np.max(np.clip(volume_xyz, vessel_low_hu, vessel_high_hu), axis = 2)
    mip_norm = _norm_2d(mip.T)
    _save_overlay(
        base_2d = mip_norm,
        mask_2d = np.max(vessel_mask, axis = 2).T,
        out_path = case_out_dir / 'axial_mip_vessel_overlay.png',
        title = 'Axial MIP Vessel Overlay'
    )

    np.save(case_out_dir / 'vessel_mask.npy', vessel_mask.astype(np.uint8))

    vessel_xyz = np.argwhere(vessel_mask > 0).astype(np.int32)
    spacing = [float(v) for v in img.header.get_zooms()[:3]]
    affine = img.affine.astype(np.float64)

    homogeneous = np.concatenate(
        [vessel_xyz.astype(np.float64), np.ones((vessel_xyz.shape[0], 1), dtype = np.float64)],
        axis = 1
    )
    world_xyz = homogeneous @ affine.T
    world_xyz = world_xyz[:, :3]

    coord_payload = {
        'nii_path': str(nii_file),
        'num_points': int(vessel_xyz.shape[0]),
        'coordinate_definition': {
            'voxel_xyz': 'Index in NIfTI array order (x, y, z).',
            'world_xyz_mm': 'Physical coordinate in mm after affine transform.'
        },
        'spacing_xyz_mm': spacing,
        'vessel_coordinates': [
            {
                'voxel_xyz': [int(x), int(y), int(z)],
                'world_xyz_mm': [float(wx), float(wy), float(wz)]
            }
            for (x, y, z), (wx, wy, wz) in zip(vessel_xyz, world_xyz)
        ]
    }

    with open(case_out_dir / 'vessel_coordinates.json', 'w', encoding = 'utf-8') as file:
        json.dump(coord_payload, file, indent = 2, ensure_ascii = False)

    vessel_voxels = int(vessel_mask.sum())
    total_voxels = int(vessel_mask.size)
    ratio = float(vessel_voxels / max(1, total_voxels))

    summary = {
        'nii_path': str(nii_file),
        'shape_xyz': [int(v) for v in volume_xyz.shape],
        'voxel_spacing': [float(v) for v in img.header.get_zooms()[:3]],
        'threshold_hu': {
            'low': float(vessel_low_hu),
            'high': float(vessel_high_hu)
        },
        'vessel_voxels': vessel_voxels,
        'total_voxels': total_voxels,
        'vessel_ratio': ratio,
        'outputs': {
            'mask_npy': str(case_out_dir / 'vessel_mask.npy'),
            'coordinates_json': str(case_out_dir / 'vessel_coordinates.json'),
            'axial_overlay': str(case_out_dir / 'axial_vessel_overlay.png'),
            'coronal_overlay': str(case_out_dir / 'coronal_vessel_overlay.png'),
            'sagittal_overlay': str(case_out_dir / 'sagittal_vessel_overlay.png'),
            'axial_grid_overlay': str(case_out_dir / 'axial_vessel_overlay_grid.png'),
            'axial_mip_overlay': str(case_out_dir / 'axial_mip_vessel_overlay.png')
        }
    }

    with open(case_out_dir / 'vessel_summary.json', 'w', encoding = 'utf-8') as file:
        json.dump(summary, file, indent = 2, ensure_ascii = False)

    logger.info('=' * 80)
    logger.info('Vessel view generated for: %s', nii_file)
    logger.info('=' * 80)
    logger.info('Output: %s', case_out_dir)
    logger.info('Vessel ratio: %.6f', ratio)

    return summary


def main():
    """Program entry point.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    view_vessels(
        nii_path = args.nii_path,
        output_dir = args.output_dir,
        vessel_low_hu = args.vessel_low_hu,
        vessel_high_hu = args.vessel_high_hu,
        num_grid_slices = args.num_grid_slices
    )


if __name__ == '__main__':
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
