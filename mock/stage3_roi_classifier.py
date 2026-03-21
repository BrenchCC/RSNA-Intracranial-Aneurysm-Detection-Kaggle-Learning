import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from constants import clamp_prob, sigmoid
from visualize import dump_json, save_bar, save_patch_grid

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = 'Stage3 mock ROI classifier with 2.5D input simulation.')
    parser.add_argument('--case-id', type = str, required = True, help = 'Case identifier.')
    parser.add_argument('--output-dir', type = str, default = 'mock/output', help = 'Output root directory.')
    parser.add_argument('--seed', type = int, default = 42, help = 'Random seed.')
    parser.add_argument('--num-slices', type = int, default = 8, help = '2.5D slice count for feature extraction, must be 6-10.')
    return parser.parse_args()


def _validate_num_slices(num_slices):
    """Validate 2.5D slice count range.

    Args:
        num_slices (int): Slice count.

    Returns:
        int: Validated slice count.
    """
    if num_slices < 6 or num_slices > 10:
        raise ValueError('num_slices must be in [6, 10].')
    return int(num_slices)


def _select_slice_indices(depth, num_slices):
    """Select evenly distributed slice indices.

    Args:
        depth (int): Patch depth.
        num_slices (int): Number of slices.

    Returns:
        np.ndarray: Selected indices.
    """
    start = max(0, depth // 2 - num_slices // 2)
    end = min(depth, start + num_slices)
    idx = np.arange(start, end)

    if len(idx) < num_slices:
        idx = np.linspace(0, depth - 1, num = num_slices, dtype = int)

    return idx.astype(int)


def _compute_roi_scores(patch, prior_score, seed_bias):
    """Compute mock probabilities for a single ROI.

    Args:
        patch (np.ndarray): ROI patch.
        prior_score (float): Prior score from stage2.
        seed_bias (float): Seed-based deterministic perturbation.

    Returns:
        tuple[float, float, float]: p_a, p_c, fused score.
    """
    mean_v = float(patch.mean())
    std_v = float(patch.std())
    max_v = float(patch.max())

    p_a_raw = 1.15 * mean_v + 0.55 * std_v + 0.85 * max_v + 0.75 * prior_score + seed_bias
    p_c_raw = 1.05 * prior_score + 0.35 * std_v + 0.25 * mean_v - 0.15 + 0.5 * seed_bias

    p_a = clamp_prob(sigmoid(p_a_raw))
    p_c = clamp_prob(sigmoid(p_c_raw))
    fused = clamp_prob(p_a * p_c)

    return p_a, p_c, fused


def run_stage3(case_id, output_dir, seed, num_slices):
    """Run stage3 ROI classifier simulation.

    Args:
        case_id (str): Case identifier.
        output_dir (str): Output root directory.
        seed (int): Random seed.
        num_slices (int): Number of 2.5D slices.

    Returns:
        dict: Stage summary metadata.
    """
    num_slices = _validate_num_slices(num_slices = num_slices)

    rng = np.random.default_rng(seed)
    stage2_dir = Path(output_dir) / case_id / 'stage2' / 'artifacts'
    stage3_root = Path(output_dir) / case_id / 'stage3'
    stage3_artifacts = stage3_root / 'artifacts'
    stage3_artifacts.mkdir(parents = True, exist_ok = True)

    roi_patches = np.load(stage2_dir / 'roi_patches.npy')
    with open(stage2_dir / 'candidates.json', 'r', encoding = 'utf-8') as file:
        candidates_payload = json.load(file)

    candidates = candidates_payload['candidates']

    roi_scores = []
    top_scores = []
    top_patches = []

    for idx, (patch, candidate) in enumerate(tqdm(list(zip(roi_patches, candidates)), desc = 'Scoring ROI')):
        selected_idx = _select_slice_indices(depth = patch.shape[0], num_slices = num_slices)
        patch_2p5d = patch[selected_idx]

        seed_bias = float(rng.normal(loc = 0.0, scale = 0.05))
        p_a, p_c, fused = _compute_roi_scores(
            patch = patch_2p5d,
            prior_score = candidate['prior_score'],
            seed_bias = seed_bias
        )

        score_item = {
            'id': int(candidate['id']),
            'vessel_class': candidate['vessel_class'],
            'p_a': float(p_a),
            'p_c': float(p_c),
            'fused_score': float(fused)
        }
        roi_scores.append(score_item)

        top_scores.append(float(fused))
        top_patches.append(patch)

    ranked = sorted(
        list(zip(roi_scores, top_patches)),
        key = lambda item: item[0]['fused_score'],
        reverse = True
    )

    roi_payload = {
        'case_id': case_id,
        'roi_scores': [item[0] for item in ranked]
    }

    roi_scores_path = stage3_artifacts / 'roi_scores.json'
    dump_json(roi_payload, roi_scores_path)

    stage3_meta = {
        'stage': 'stage3_roi_classifier',
        'case_id': case_id,
        'seed': int(seed),
        'input': {
            'roi_patches_path': str(stage2_dir / 'roi_patches.npy'),
            'candidates_path': str(stage2_dir / 'candidates.json')
        },
        'output': {
            'roi_scores_path': str(roi_scores_path)
        },
        'num_roi': int(len(roi_scores)),
        'num_slices_for_2p5d': int(num_slices),
        'trick_mapping': {
            'trick5_input_2p5d': f'Used {num_slices} slices for 2.5D scoring simulation.'
        }
    }

    meta_path = stage3_artifacts / 'stage3_meta.json'
    dump_json(stage3_meta, meta_path)

    stage3_debug = stage3_root / 'figs' / 'debug'
    stage3_report = stage3_root / 'figs' / 'report'
    stage3_debug.mkdir(parents = True, exist_ok = True)
    stage3_report.mkdir(parents = True, exist_ok = True)

    ranked_scores = roi_payload['roi_scores']
    labels = [f"roi_{item['id']}" for item in ranked_scores[:20]]
    values = [item['fused_score'] for item in ranked_scores[:20]]

    top_patch_array = np.stack([item[1] for item in ranked[:9]], axis = 0).astype(np.float32)
    top_patch_scores = [item[0]['fused_score'] for item in ranked[:9]]

    for style in ['debug', 'report']:
        save_bar(
            labels = labels,
            values = values,
            out_path = stage3_root / 'figs' / style / 'stage3_top_roi_scores.png',
            style = style,
            title = 'Stage3 ROI Fused Scores (Top 20)',
            rotate = 70
        )
        save_patch_grid(
            patches = top_patch_array,
            scores = top_patch_scores,
            out_path = stage3_root / 'figs' / style / 'stage3_top_evidence_grid.png',
            style = style,
            title = 'Stage3 Top Evidence ROI Grid',
            n_show = 9
        )

    logger.info('=' * 80)
    logger.info('Stage3 completed: %s', case_id)
    logger.info('=' * 80)
    logger.info('Produced %d ROI scores.', len(roi_scores))

    return stage3_meta


def main():
    """Program entry point.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    run_stage3(
        case_id = args.case_id,
        output_dir = args.output_dir,
        seed = args.seed,
        num_slices = args.num_slices
    )


if __name__ == '__main__':
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
