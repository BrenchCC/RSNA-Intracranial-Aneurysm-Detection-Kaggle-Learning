import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.append(os.getcwd())
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from create_mock_input import create_case
from stage1_preprocess import run_stage1
from stage2_candidates import run_stage2
from stage3_roi_classifier import run_stage3
from stage4_aggregate import run_stage4

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = 'Run full mock 4-stage RSNA pipeline.')
    parser.add_argument('--case-id', type = str, required = True, help = 'Case identifier.')
    parser.add_argument('--input-dir', type = str, default = 'mock/input', help = 'Input root directory.')
    parser.add_argument('--output-dir', type = str, default = 'mock/output', help = 'Output root directory.')
    parser.add_argument('--seed', type = int, default = 42, help = 'Random seed.')
    parser.add_argument('--num-roi', type = int, default = 64, help = 'Target ROI count for stage2.')
    parser.add_argument('--k-top', type = int, default = 5, help = 'Top-k used in stage4.')
    parser.add_argument('--num-slices', type = int, default = 8, help = '2.5D slices used in stage3, must be 6-10.')
    parser.add_argument('--distance-threshold-mm', type = float, default = 40.0, help = 'Distance filter threshold used in stage2.')
    parser.add_argument('--patch-size', type = int, default = 24, help = 'ROI patch side length.')
    return parser.parse_args()


def _ensure_input(case_id, input_dir, seed):
    """Create mock input if case files are missing.

    Args:
        case_id (str): Case identifier.
        input_dir (str): Input root directory.
        seed (int): Random seed.

    Returns:
        None
    """
    case_dir = Path(input_dir) / case_id
    volume_path = case_dir / 'volume.npy'
    meta_path = case_dir / 'case_meta.json'

    if volume_path.exists() and meta_path.exists():
        logger.info('Input exists: %s', case_dir)
        return

    logger.info('-' * 60)
    logger.info('Input missing, creating mock case: %s', case_id)
    logger.info('-' * 60)
    create_case(
        input_dir = input_dir,
        case_id = case_id,
        seed = seed,
        shape = [64, 128, 128]
    )


def run_all(case_id, input_dir, output_dir, seed, num_roi, k_top, num_slices, distance_threshold_mm, patch_size):
    """Run the entire 4-stage mock pipeline.

    Args:
        case_id (str): Case identifier.
        input_dir (str): Input root directory.
        output_dir (str): Output root directory.
        seed (int): Random seed.
        num_roi (int): Number of ROI candidates.
        k_top (int): Top-k count.
        num_slices (int): Number of 2.5D slices.
        distance_threshold_mm (float): Distance threshold.
        patch_size (int): ROI patch size.

    Returns:
        dict: Final stage metadata.
    """
    if num_slices < 6 or num_slices > 10:
        raise ValueError('num_slices must be in [6, 10].')

    _ensure_input(case_id = case_id, input_dir = input_dir, seed = seed)

    logger.info('=' * 80)
    logger.info('Running full mock pipeline for case: %s', case_id)
    logger.info('=' * 80)

    run_stage1(
        case_id = case_id,
        input_dir = input_dir,
        output_dir = output_dir,
        seed = seed
    )

    run_stage2(
        case_id = case_id,
        output_dir = output_dir,
        seed = seed,
        num_roi = num_roi,
        distance_threshold_mm = distance_threshold_mm,
        patch_size = patch_size
    )

    run_stage3(
        case_id = case_id,
        output_dir = output_dir,
        seed = seed,
        num_slices = num_slices
    )

    stage4_meta = run_stage4(
        case_id = case_id,
        output_dir = output_dir,
        k_top = k_top
    )

    logger.info('*' * 50)
    logger.info('Pipeline completed. Final artifacts: %s', Path(output_dir) / case_id / 'stage4' / 'artifacts')
    logger.info('*' * 50)

    return stage4_meta


def main():
    """Program entry point.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    run_all(
        case_id = args.case_id,
        input_dir = args.input_dir,
        output_dir = args.output_dir,
        seed = args.seed,
        num_roi = args.num_roi,
        k_top = args.k_top,
        num_slices = args.num_slices,
        distance_threshold_mm = args.distance_threshold_mm,
        patch_size = args.patch_size
    )


if __name__ == '__main__':
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
