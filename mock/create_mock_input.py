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

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = 'Create mock volume inputs for RSNA pipeline.')
    parser.add_argument('--input-dir', type = str, default = 'mock/input', help = 'Root input directory.')
    parser.add_argument('--case-id', type = str, default = 'case_001', help = 'Case identifier.')
    parser.add_argument('--seed', type = int, default = 42, help = 'Random seed.')
    parser.add_argument('--shape', type = int, nargs = 3, default = [64, 128, 128], help = 'Volume shape z y x.')
    return parser.parse_args()


def create_volume(shape, seed):
    """Create a synthetic 3D volume with vessel-like intensity patterns.

    Args:
        shape (tuple[int, int, int]): Volume shape in z, y, x order.
        seed (int): Random seed.

    Returns:
        np.ndarray: Generated volume as float32.
    """
    rng = np.random.default_rng(seed)
    z, y, x = shape

    volume = rng.normal(loc = 0.0, scale = 0.15, size = shape).astype(np.float32)

    zz = np.linspace(-1.0, 1.0, z, dtype = np.float32)[:, None, None]
    yy = np.linspace(-1.0, 1.0, y, dtype = np.float32)[None, :, None]
    xx = np.linspace(-1.0, 1.0, x, dtype = np.float32)[None, None, :]

    brain_mask = ((xx ** 2) / 0.95 + (yy ** 2) / 0.85 + (zz ** 2) / 1.20) <= 1.0
    volume += brain_mask.astype(np.float32) * 0.45

    for _ in range(5):
        base_y = rng.integers(low = y // 4, high = y * 3 // 4)
        base_x = rng.integers(low = x // 4, high = x * 3 // 4)
        amp_y = rng.uniform(5, 12)
        amp_x = rng.uniform(5, 12)
        phase = rng.uniform(0, np.pi)
        radius = rng.uniform(1.8, 3.2)

        for zi in range(z):
            cy = int(base_y + amp_y * np.sin(zi / 8.0 + phase))
            cx = int(base_x + amp_x * np.cos(zi / 10.0 + phase))

            y0 = max(0, cy - 4)
            y1 = min(y, cy + 5)
            x0 = max(0, cx - 4)
            x1 = min(x, cx + 5)

            yy_patch, xx_patch = np.ogrid[y0:y1, x0:x1]
            circle = ((yy_patch - cy) ** 2 + (xx_patch - cx) ** 2) <= radius ** 2
            volume[zi, y0:y1, x0:x1][circle] += rng.uniform(0.35, 0.65)

    return volume.astype(np.float32)


def create_case(input_dir, case_id, seed, shape):
    """Create one mock case files.

    Args:
        input_dir (str): Input root directory.
        case_id (str): Case identifier.
        seed (int): Random seed.
        shape (tuple[int, int, int]): Volume shape.

    Returns:
        dict: Created file paths.
    """
    case_dir = Path(input_dir) / case_id
    case_dir.mkdir(parents = True, exist_ok = True)

    volume = create_volume(shape = tuple(shape), seed = seed)
    volume_path = case_dir / 'volume.npy'
    np.save(volume_path, volume)

    meta = {
        'case_id': case_id,
        'shape_zyx': list(volume.shape),
        'spacing_zyx_mm': [1.0, 0.6, 0.6],
        'orientation': 'RAS',
        'origin_xyz_mm': [0.0, 0.0, 0.0],
        'mock_note': 'Synthetic volume for pipeline interface validation.'
    }

    meta_path = case_dir / 'case_meta.json'
    with open(meta_path, 'w', encoding = 'utf-8') as file:
        json.dump(meta, file, indent = 2, ensure_ascii = False)

    return {'volume_path': str(volume_path), 'meta_path': str(meta_path)}


def main():
    """Program entry point.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    created = create_case(
        input_dir = args.input_dir,
        case_id = args.case_id,
        seed = args.seed,
        shape = args.shape
    )
    logger.info('Created mock case: %s', created)
    print(json.dumps(created, indent = 2, ensure_ascii = False))


if __name__ == '__main__':
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
