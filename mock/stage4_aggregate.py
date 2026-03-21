import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from constants import LABEL_COLS, VESSEL_CLASS_TO_LABEL, clamp_prob
from visualize import dump_json, save_bar, save_compare_bars

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = 'Stage4 mock aggregation with top-k mean.')
    parser.add_argument('--case-id', type = str, required = True, help = 'Case identifier.')
    parser.add_argument('--output-dir', type = str, default = 'mock/output', help = 'Output root directory.')
    parser.add_argument('--k-top', type = int, default = 5, help = 'Top-k used in aggregation.')
    return parser.parse_args()


def _topk_mean(values, k_top):
    """Compute top-k mean with safe fallback.

    Args:
        values (list[float]): Input values.
        k_top (int): Top-k count.

    Returns:
        float: Top-k mean score.
    """
    if len(values) == 0:
        return 0.0

    sorted_values = sorted(values, reverse = True)
    k_use = min(len(sorted_values), max(1, int(k_top)))
    return float(np.mean(sorted_values[:k_use]))


def run_stage4(case_id, output_dir, k_top):
    """Run stage4 aggregation.

    Args:
        case_id (str): Case identifier.
        output_dir (str): Output root directory.
        k_top (int): Top-k count.

    Returns:
        dict: Stage summary metadata.
    """
    stage3_dir = Path(output_dir) / case_id / 'stage3' / 'artifacts'
    stage4_root = Path(output_dir) / case_id / 'stage4'
    stage4_artifacts = stage4_root / 'artifacts'
    stage4_artifacts.mkdir(parents = True, exist_ok = True)

    with open(stage3_dir / 'roi_scores.json', 'r', encoding = 'utf-8') as file:
        roi_payload = json.load(file)

    roi_scores = roi_payload['roi_scores']

    class_to_scores = {key: [] for key in VESSEL_CLASS_TO_LABEL.keys()}
    for item in roi_scores:
        vessel_class = item['vessel_class']
        if vessel_class in class_to_scores:
            class_to_scores[vessel_class].append(float(item['fused_score']))

    pred_13_topk = {}
    pred_13_max = {}

    for vessel_class, label_name in VESSEL_CLASS_TO_LABEL.items():
        values = class_to_scores.get(vessel_class, [])
        score_topk = clamp_prob(_topk_mean(values = values, k_top = k_top))
        score_max = clamp_prob(max(values) if len(values) > 0 else 0.0)
        pred_13_topk[label_name] = float(score_topk)
        pred_13_max[label_name] = float(score_max)

    aneurysm_present = float(max(pred_13_topk.values()))

    pred_14 = {}
    for label in LABEL_COLS[:-1]:
        pred_14[label] = pred_13_topk[label]
    pred_14['Aneurysm Present'] = aneurysm_present

    case_pred = {
        'case_id': case_id,
        'method': 'vessel-wise top-k mean on fused_score',
        'k': int(k_top),
        'pred_14': pred_14
    }

    case_pred_path = stage4_artifacts / 'case_pred.json'
    submission_path = stage4_artifacts / 'submission_like.csv'
    stage4_meta_path = stage4_artifacts / 'stage4_meta.json'

    dump_json(case_pred, case_pred_path)

    row = {'series_uid': case_id}
    for label in LABEL_COLS:
        row[label] = pred_14[label]

    pd.DataFrame([row]).to_csv(submission_path, index = False)

    stage4_meta = {
        'stage': 'stage4_aggregate',
        'case_id': case_id,
        'input': {
            'roi_scores_path': str(stage3_dir / 'roi_scores.json')
        },
        'output': {
            'case_pred_path': str(case_pred_path),
            'submission_like_csv_path': str(submission_path)
        },
        'k_top': int(k_top),
        'aggregation_rule': {
            'per_vessel': 'top-k mean on fused_score grouped by vessel_class',
            'aneurysm_present': 'max over 13 location probabilities'
        },
        'trick_mapping': {
            'trick4_topk_mean_aggregation': 'Case-level vector derived from vessel-wise top-k mean.'
        },
        'class_roi_counts': {
            key: int(len(values)) for key, values in class_to_scores.items()
        }
    }
    dump_json(stage4_meta, stage4_meta_path)

    stage4_debug = stage4_root / 'figs' / 'debug'
    stage4_report = stage4_root / 'figs' / 'report'
    stage4_debug.mkdir(parents = True, exist_ok = True)
    stage4_report.mkdir(parents = True, exist_ok = True)

    loc_labels = LABEL_COLS[:-1]
    topk_values = [pred_13_topk[label] for label in loc_labels]
    max_values = [pred_13_max[label] for label in loc_labels]

    pred14_values = [pred_14[label] for label in LABEL_COLS]

    for style in ['debug', 'report']:
        save_bar(
            labels = LABEL_COLS,
            values = pred14_values,
            out_path = stage4_root / 'figs' / style / 'stage4_pred14_bar.png',
            style = style,
            title = 'Stage4 Case-Level 14-Dim Prediction',
            rotate = 80
        )
        save_compare_bars(
            labels = loc_labels,
            values_a = max_values,
            values_b = topk_values,
            out_path = stage4_root / 'figs' / style / 'stage4_max_vs_topk.png',
            style = style,
            title = 'Stage4 Max vs Top-k Mean (Trick4)',
            name_a = 'max',
            name_b = 'top-k mean'
        )

    logger.info('=' * 80)
    logger.info('Stage4 completed: %s', case_id)
    logger.info('=' * 80)
    logger.info('Saved case prediction to %s', case_pred_path)

    return stage4_meta


def main():
    """Program entry point.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    run_stage4(
        case_id = args.case_id,
        output_dir = args.output_dir,
        k_top = args.k_top
    )


if __name__ == '__main__':
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
