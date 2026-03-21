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

from constants import VESSEL_CLASSES
from visualize import dump_json, save_bar, save_hist, save_patch_grid, save_candidate_scatter

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = 'Stage2 mock candidate generation with distance-aware filtering.')
    parser.add_argument('--case-id', type = str, required = True, help = 'Case identifier.')
    parser.add_argument('--output-dir', type = str, default = 'mock/output', help = 'Output root directory.')
    parser.add_argument('--seed', type = int, default = 42, help = 'Random seed.')
    parser.add_argument('--num-roi', type = int, default = 64, help = 'Target number of kept ROI candidates.')
    parser.add_argument('--distance-threshold-mm', type = float, default = 40.0, help = 'Distance threshold for trick3 simulation.')
    parser.add_argument('--patch-size', type = int, default = 24, help = 'ROI patch cubic side length.')
    return parser.parse_args()


def _pick_vessel_class(z_idx, y_idx, x_idx, shape):
    """Assign vessel class based on coarse spatial bins.

    Args:
        z_idx (int): Z index.
        y_idx (int): Y index.
        x_idx (int): X index.
        shape (tuple[int, int, int]): Volume shape.

    Returns:
        str: Vessel class name.
    """
    z, y, x = shape
    z_bin = min(2, int(3 * z_idx / max(1, z)))
    y_bin = min(1, int(2 * y_idx / max(1, y)))
    x_bin = min(1, int(2 * x_idx / max(1, x)))
    idx = (z_bin * 4 + y_bin * 2 + x_bin) % len(VESSEL_CLASSES)
    return VESSEL_CLASSES[idx]


def _crop_patch(volume, center, patch_size):
    """Extract a cubic patch with edge padding if needed.

    Args:
        volume (np.ndarray): Input volume.
        center (tuple[int, int, int]): Center index in z-y-x.
        patch_size (int): Cube side length.

    Returns:
        np.ndarray: Patch array with shape (patch_size, patch_size, patch_size).
    """
    half = patch_size // 2
    z, y, x = volume.shape
    cz, cy, cx = center

    z0 = max(0, cz - half)
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    z1 = min(z, cz + half)
    y1 = min(y, cy + half)
    x1 = min(x, cx + half)

    patch = volume[z0:z1, y0:y1, x0:x1]

    pad_z = patch_size - patch.shape[0]
    pad_y = patch_size - patch.shape[1]
    pad_x = patch_size - patch.shape[2]

    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        pad_width = (
            (0, pad_z),
            (0, pad_y),
            (0, pad_x)
        )
        patch = np.pad(patch, pad_width = pad_width, mode = 'edge')

    return patch.astype(np.float32)


def _build_pseudo_lesions(shape):
    """Create deterministic pseudo lesion centers for trick3 simulation.

    Args:
        shape (tuple[int, int, int]): Volume shape.

    Returns:
        list[list[int]]: Pseudo lesion centers in z-y-x.
    """
    z, y, x = shape
    centers = [
        [int(z * 0.45), int(y * 0.35), int(x * 0.42)],
        [int(z * 0.62), int(y * 0.58), int(x * 0.68)]
    ]
    return centers


def run_stage2(case_id, output_dir, seed, num_roi, distance_threshold_mm, patch_size):
    """Run stage2 candidate generation and filtering.

    Args:
        case_id (str): Case identifier.
        output_dir (str): Output root directory.
        seed (int): Random seed.
        num_roi (int): Target number of kept candidates.
        distance_threshold_mm (float): Distance filter threshold.
        patch_size (int): Patch side length.

    Returns:
        dict: Stage summary metadata.
    """
    rng = np.random.default_rng(seed)

    stage1_dir = Path(output_dir) / case_id / 'stage1' / 'artifacts'
    stage2_root = Path(output_dir) / case_id / 'stage2'
    stage2_artifacts = stage2_root / 'artifacts'
    stage2_artifacts.mkdir(parents = True, exist_ok = True)

    with open(stage1_dir / 'stage1_meta.json', 'r', encoding = 'utf-8') as file:
        stage1_meta = json.load(file)

    volume_norm = np.load(stage1_dir / 'volume_norm.npy')
    vessel_prior = np.load(stage1_dir / 'vessel_prior.npy')

    spacing = stage1_meta.get('spacing_zyx_mm', [1.0, 1.0, 1.0])
    shape = volume_norm.shape

    pseudo_lesions = _build_pseudo_lesions(shape = shape)

    sample_pool = int(max(num_roi * 3, 180))
    high_prior_indices = np.argwhere(vessel_prior > float(np.quantile(vessel_prior, 0.87)))

    if len(high_prior_indices) < sample_pool:
        high_prior_indices = np.argwhere(vessel_prior > float(np.quantile(vessel_prior, 0.75)))

    chosen_idx = rng.choice(len(high_prior_indices), size = min(sample_pool, len(high_prior_indices)), replace = False)
    sampled_points = high_prior_indices[chosen_idx]

    candidates = []
    for idx, point in enumerate(tqdm(sampled_points, desc = 'Building candidates')):
        cz, cy, cx = [int(v) for v in point]

        dists = []
        for lesion_center in pseudo_lesions:
            dz = (cz - lesion_center[0]) * spacing[0]
            dy = (cy - lesion_center[1]) * spacing[1]
            dx = (cx - lesion_center[2]) * spacing[2]
            dists.append(float(np.sqrt(dz ** 2 + dy ** 2 + dx ** 2)))

        dist_to_lesion = float(min(dists))
        keep = bool(dist_to_lesion > distance_threshold_mm)

        vessel_class = _pick_vessel_class(
            z_idx = cz,
            y_idx = cy,
            x_idx = cx,
            shape = shape
        )

        candidate = {
            'id': int(idx),
            'center_xyz': [cz, cy, cx],
            'vessel_class': vessel_class,
            'prior_score': float(vessel_prior[cz, cy, cx]),
            'dist_to_pseudo_lesion_mm': dist_to_lesion,
            'keep_by_distance_rule': keep
        }
        candidates.append(candidate)

    candidates_sorted = sorted(
        candidates,
        key = lambda item: (
            item['keep_by_distance_rule'],
            item['prior_score'],
            item['dist_to_pseudo_lesion_mm']
        ),
        reverse = True
    )

    kept = [item for item in candidates_sorted if item['keep_by_distance_rule']]
    if len(kept) < num_roi:
        fallback = [item for item in candidates_sorted if not item['keep_by_distance_rule']]
        kept = kept + fallback[:max(0, num_roi - len(kept))]

    kept = kept[:num_roi]

    patches = []
    for candidate in tqdm(kept, desc = 'Cropping ROI patches'):
        center = tuple(candidate['center_xyz'])
        patch = _crop_patch(volume = volume_norm, center = center, patch_size = patch_size)
        patches.append(patch)

    roi_patches = np.stack(patches).astype(np.float32)

    candidates_path = stage2_artifacts / 'candidates.json'
    roi_path = stage2_artifacts / 'roi_patches.npy'
    meta_path = stage2_artifacts / 'stage2_meta.json'

    candidates_payload = {
        'case_id': case_id,
        'candidates': kept
    }
    dump_json(candidates_payload, candidates_path)
    np.save(roi_path, roi_patches)

    all_distances = [item['dist_to_pseudo_lesion_mm'] for item in candidates]
    kept_distances = [item['dist_to_pseudo_lesion_mm'] for item in kept]

    per_class_counts = {name: 0 for name in VESSEL_CLASSES}
    for item in kept:
        per_class_counts[item['vessel_class']] += 1

    stage2_meta = {
        'stage': 'stage2_candidates',
        'case_id': case_id,
        'seed': int(seed),
        'input': {
            'vessel_prior_path': str(stage1_dir / 'vessel_prior.npy'),
            'stage1_meta_path': str(stage1_dir / 'stage1_meta.json')
        },
        'output': {
            'candidates_path': str(candidates_path),
            'roi_patches_path': str(roi_path)
        },
        'num_candidates_sampled': int(len(candidates)),
        'num_candidates_kept': int(len(kept)),
        'patch_size': int(patch_size),
        'distance_threshold_mm': float(distance_threshold_mm),
        'pseudo_lesion_centers_zyx': pseudo_lesions,
        'distance_stats_mm': {
            'all_mean': float(np.mean(all_distances)),
            'kept_mean': float(np.mean(kept_distances)),
            'all_min': float(np.min(all_distances)),
            'all_max': float(np.max(all_distances))
        },
        'trick_mapping': {
            'trick2_vessel_class_roi': 'Each candidate includes vessel_class and class-wise statistics.',
            'trick3_distance_aware_negative_sampling_sim': 'Candidates filtered by distance to pseudo lesions for demonstration.'
        }
    }
    dump_json(stage2_meta, meta_path)

    stage2_debug = stage2_root / 'figs' / 'debug'
    stage2_report = stage2_root / 'figs' / 'report'
    stage2_debug.mkdir(parents = True, exist_ok = True)
    stage2_report.mkdir(parents = True, exist_ok = True)

    class_labels = list(per_class_counts.keys())
    class_values = [per_class_counts[key] for key in class_labels]

    for style in ['debug', 'report']:
        save_candidate_scatter(
            base_volume = volume_norm,
            candidates = kept,
            out_path = stage2_root / 'figs' / style / 'stage2_candidate_scatter.png',
            style = style,
            title = 'Stage2 Candidate Scatter (Trick2)'
        )
        save_bar(
            labels = class_labels,
            values = class_values,
            out_path = stage2_root / 'figs' / style / 'stage2_candidate_per_class.png',
            style = style,
            title = 'Stage2 Candidate Counts per Vessel Class',
            rotate = 80
        )
        save_bar(
            labels = ['sampled_total', 'kept_after_distance'],
            values = [len(candidates), len(kept)],
            out_path = stage2_root / 'figs' / style / 'stage2_distance_filter_counts.png',
            style = style,
            title = 'Stage2 Distance Filter Result (Trick3)',
            rotate = 0
        )
        save_hist(
            values = all_distances,
            bins = 24,
            out_path = stage2_root / 'figs' / style / 'stage2_distance_hist_all.png',
            style = style,
            title = 'Stage2 Distance Distribution to Pseudo Lesions (mm)'
        )
        prior_scores = [item['prior_score'] for item in kept]
        save_patch_grid(
            patches = roi_patches,
            scores = prior_scores,
            out_path = stage2_root / 'figs' / style / 'stage2_roi_patch_grid.png',
            style = style,
            title = 'Stage2 ROI Patch Grid by Prior Score',
            n_show = 9
        )

    logger.info('=' * 80)
    logger.info('Stage2 completed: %s', case_id)
    logger.info('=' * 80)
    logger.info('Kept %d candidates after distance filtering.', len(kept))

    return stage2_meta


def main():
    """Program entry point.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    run_stage2(
        case_id = args.case_id,
        output_dir = args.output_dir,
        seed = args.seed,
        num_roi = args.num_roi,
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
