import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np

sys.path.append(os.getcwd())
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from visualize import dump_json, save_mid_slice, save_overlay

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = 'Stage1 mock preprocessing with vessel prior generation.')
    parser.add_argument('--case-id', type = str, required = True, help = 'Case identifier.')
    parser.add_argument('--input-dir', type = str, default = 'mock/input', help = 'Input root directory.')
    parser.add_argument('--output-dir', type = str, default = 'mock/output', help = 'Output root directory.')
    parser.add_argument('--seed', type = int, default = 42, help = 'Random seed for deterministic mock behavior.')
    return parser.parse_args()


def _sigmoid(array):
    """Compute element-wise sigmoid.

    Args:
        array (np.ndarray): Input array.

    Returns:
        np.ndarray: Sigmoid values.
    """
    return 1.0 / (1.0 + np.exp(-array))


def normalize_volume(volume):
    """Normalize intensity with clipping and z-score.

    Args:
        volume (np.ndarray): Raw volume in z-y-x layout.

    Returns:
        tuple[np.ndarray, dict]: Normalized volume and statistics.
    """
    lower = float(np.percentile(volume, 1.0))
    upper = float(np.percentile(volume, 99.0))
    clipped = np.clip(volume, lower, upper)

    valid_mask = clipped > float(np.percentile(clipped, 35.0))
    if valid_mask.sum() < 20:
        valid_mask = np.ones_like(clipped, dtype = bool)

    mean = float(clipped[valid_mask].mean())
    std = float(clipped[valid_mask].std()) + 1e-6
    volume_norm = ((clipped - mean) / std).astype(np.float32)

    stats = {
        'clip_lower': lower,
        'clip_upper': upper,
        'zscore_mean': mean,
        'zscore_std': std,
        'valid_voxel_count': int(valid_mask.sum())
    }
    return volume_norm, stats


def create_vessel_prior(volume_norm, seed):
    """Generate mock vessel prior probability map.

    Args:
        volume_norm (np.ndarray): Normalized volume.
        seed (int): Deterministic random seed.

    Returns:
        np.ndarray: Vessel prior map in [0, 1].
    """
    rng = np.random.default_rng(seed)
    z, y, x = volume_norm.shape

    zz = np.linspace(-1.0, 1.0, z, dtype = np.float32)[:, None, None]
    yy = np.linspace(-1.0, 1.0, y, dtype = np.float32)[None, :, None]
    xx = np.linspace(-1.0, 1.0, x, dtype = np.float32)[None, None, :]

    center_bias = 1.0 - np.clip(np.sqrt(xx ** 2 + yy ** 2 + (0.75 * zz) ** 2), 0.0, 1.5) / 1.5
    texture_signal = _sigmoid(volume_norm * 1.4)

    vessel_prior = 0.60 * texture_signal + 0.40 * center_bias

    for _ in range(5):
        cy = rng.integers(low = y // 4, high = y * 3 // 4)
        cx = rng.integers(low = x // 4, high = x * 3 // 4)
        width = rng.uniform(6.0, 13.0)
        amp = rng.uniform(0.08, 0.16)
        y_grid = np.arange(y, dtype = np.float32)[None, :, None]
        x_grid = np.arange(x, dtype = np.float32)[None, None, :]
        line = np.exp(-((y_grid - cy) ** 2 + (x_grid - cx) ** 2) / (2 * width ** 2))
        vessel_prior += amp * line

    vessel_prior = np.clip(vessel_prior, 0.0, 1.0).astype(np.float32)
    return vessel_prior


def run_stage1(case_id, input_dir, output_dir, seed):
    """Run stage1 preprocessing pipeline.

    Args:
        case_id (str): Case identifier.
        input_dir (str): Input root directory.
        output_dir (str): Output root directory.
        seed (int): Random seed.

    Returns:
        dict: Stage summary metadata.
    """
    case_input_dir = Path(input_dir) / case_id
    volume_path = case_input_dir / 'volume.npy'
    meta_path = case_input_dir / 'case_meta.json'

    if not volume_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f'Missing input files under {case_input_dir}')

    with open(meta_path, 'r', encoding = 'utf-8') as file:
        case_meta = json.load(file)

    volume = np.load(volume_path)
    volume_norm, norm_stats = normalize_volume(volume = volume)
    vessel_prior = create_vessel_prior(volume_norm = volume_norm, seed = seed)

    stage_root = Path(output_dir) / case_id / 'stage1'
    artifacts_dir = stage_root / 'artifacts'
    debug_fig_dir = stage_root / 'figs' / 'debug'
    report_fig_dir = stage_root / 'figs' / 'report'

    artifacts_dir.mkdir(parents = True, exist_ok = True)
    debug_fig_dir.mkdir(parents = True, exist_ok = True)
    report_fig_dir.mkdir(parents = True, exist_ok = True)

    volume_norm_path = artifacts_dir / 'volume_norm.npy'
    vessel_prior_path = artifacts_dir / 'vessel_prior.npy'
    stage_meta_path = artifacts_dir / 'stage1_meta.json'

    np.save(volume_norm_path, volume_norm)
    np.save(vessel_prior_path, vessel_prior)

    stage_meta = {
        'stage': 'stage1_preprocess',
        'case_id': case_id,
        'seed': int(seed),
        'input': {
            'volume_path': str(volume_path),
            'case_meta_path': str(meta_path)
        },
        'output': {
            'volume_norm_path': str(volume_norm_path),
            'vessel_prior_path': str(vessel_prior_path)
        },
        'shape_zyx': list(volume.shape),
        'spacing_zyx_mm': case_meta.get('spacing_zyx_mm', [1.0, 1.0, 1.0]),
        'trick_mapping': {
            'trick1_segmentation_to_detection': 'Generated vessel prior to reduce search space for later stages.'
        },
        'normalization': norm_stats
    }

    dump_json(stage_meta, stage_meta_path)

    for style in ['debug', 'report']:
        save_mid_slice(
            volume = volume,
            out_path = stage_root / 'figs' / style / 'stage1_raw_mid.png',
            style = style,
            title = 'Stage1 Raw Volume Mid Slice'
        )
        save_mid_slice(
            volume = volume_norm,
            out_path = stage_root / 'figs' / style / 'stage1_norm_mid.png',
            style = style,
            title = 'Stage1 Normalized Mid Slice'
        )
        save_overlay(
            base_volume = volume_norm,
            overlay_volume = vessel_prior,
            out_path = stage_root / 'figs' / style / 'stage1_vessel_prior_overlay.png',
            style = style,
            title = 'Stage1 Vessel Prior Overlay (Trick1)'
        )

    logger.info('=' * 80)
    logger.info('Stage1 completed: %s', case_id)
    logger.info('=' * 80)
    logger.info('Saved artifacts to %s', artifacts_dir)

    return stage_meta


def main():
    """Program entry point.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    run_stage1(
        case_id = args.case_id,
        input_dir = args.input_dir,
        output_dir = args.output_dir,
        seed = args.seed
    )


if __name__ == '__main__':
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
