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

from visualize import dump_json, save_mid_slice, save_slice_grid

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = 'Decode mock input volume into viewable PNG images.')
    parser.add_argument('--case-id', type = str, required = True, help = 'Case identifier.')
    parser.add_argument('--input-dir', type = str, default = 'mock/input', help = 'Input root directory.')
    parser.add_argument('--output-dir', type = str, default = 'mock/output', help = 'Output root directory.')
    parser.add_argument('--num-slices', type = int, default = 9, help = 'Number of slices in grid preview.')
    return parser.parse_args()


def _normalize_for_view(volume):
    """Normalize volume to [0, 1] for visualization.

    Args:
        volume (np.ndarray): Input volume.

    Returns:
        np.ndarray: Normalized volume.
    """
    low = float(np.percentile(volume, 1.0))
    high = float(np.percentile(volume, 99.0))
    clipped = np.clip(volume, low, high)
    scaled = (clipped - low) / (high - low + 1e-6)
    return scaled.astype(np.float32)


def run_decode(case_id, input_dir, output_dir, num_slices):
    """Run input decoding and save preview images.

    Args:
        case_id (str): Case identifier.
        input_dir (str): Input root directory.
        output_dir (str): Output root directory.
        num_slices (int): Number of slices in grid preview.

    Returns:
        dict: Output summary.
    """
    case_dir = Path(input_dir) / case_id
    volume_path = case_dir / 'volume.npy'
    meta_path = case_dir / 'case_meta.json'

    if not volume_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f'Missing input files under {case_dir}')

    with open(meta_path, 'r', encoding = 'utf-8') as file:
        case_meta = json.load(file)

    volume = np.load(volume_path)
    view_volume = _normalize_for_view(volume = volume)

    out_root = Path(output_dir) / case_id / 'input_preview'
    artifacts_dir = out_root / 'artifacts'
    artifacts_dir.mkdir(parents = True, exist_ok = True)

    for style in ['debug', 'report']:
        save_mid_slice(
            volume = view_volume,
            out_path = out_root / 'figs' / style / 'input_mid_slice.png',
            style = style,
            title = 'Input Volume Mid Slice'
        )
        save_slice_grid(
            volume = view_volume,
            out_path = out_root / 'figs' / style / 'input_slice_grid.png',
            style = style,
            title = 'Input Volume Slice Grid',
            num_slices = num_slices
        )

    summary = {
        'case_id': case_id,
        'input_volume_path': str(volume_path),
        'meta_path': str(meta_path),
        'shape_zyx': list(volume.shape),
        'spacing_zyx_mm': case_meta.get('spacing_zyx_mm', [1.0, 1.0, 1.0]),
        'preview_files': {
            'debug_mid': str(out_root / 'figs' / 'debug' / 'input_mid_slice.png'),
            'debug_grid': str(out_root / 'figs' / 'debug' / 'input_slice_grid.png'),
            'report_mid': str(out_root / 'figs' / 'report' / 'input_mid_slice.png'),
            'report_grid': str(out_root / 'figs' / 'report' / 'input_slice_grid.png')
        }
    }

    dump_json(summary, artifacts_dir / 'input_preview_meta.json')

    logger.info('=' * 80)
    logger.info('Input decode completed: %s', case_id)
    logger.info('=' * 80)
    logger.info('Saved preview images to %s', out_root)

    return summary


def main():
    """Program entry point.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    run_decode(
        case_id = args.case_id,
        input_dir = args.input_dir,
        output_dir = args.output_dir,
        num_slices = args.num_slices
    )


if __name__ == '__main__':
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
