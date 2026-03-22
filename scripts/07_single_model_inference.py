#!/usr/bin/env python3
"""
Single model inference script for Eric3D aneurysm detection.

Features:
- Supports val/test inference
- Loads model architecture consistent with training
- Saves predictions to CSV/NPY
- Saves sample visualization images for 3D patches
- Optionally saves labels for validation mode
"""

import os
import sys
import argparse
import json
import logging
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------
# Replace this import path with your actual training script module name.
# ---------------------------------------------------------------------
sys.path.append(os.getcwd())
from scripts.train_model import (
    LABEL_COLS,
    ResNet3D,
    DenseNet3D,
    EfficientNet3D,
    VisionTransformer3D,
    UNet3D,
    MobileNetV2_3D,
    MobileNetV3_3D,
    MobileNetV4_3D,
    SwinTransformer3D,
    ConvNeXt3D,
    Inception3D,
    SEResNet3D,
)

logger = logging.getLogger(__name__)


class Eric3DInferenceDataset(Dataset):
    """Dataset for inference on Eric3D ROI patches.

    Args:
        patch_files: List of NPZ patch file paths.
        labels_df: Optional labels dataframe for validation mode.
        label_cols: List of label column names.
    """

    def __init__(
        self,
        patch_files: list[Path],
        labels_df: pd.DataFrame | None,
        label_cols: list[str],
    ):
        self.patch_files = patch_files
        self.labels_df = labels_df
        self.label_cols = label_cols

        self.uid_to_labels = {}
        if labels_df is not None:
            if "series_uid" not in labels_df.columns:
                raise ValueError("labels_df must contain 'series_uid' column.")

            for _, row in labels_df.iterrows():
                series_uid = row["series_uid"]
                labels = row[label_cols].values.astype(np.float32)
                self.uid_to_labels[series_uid] = labels

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.patch_files)

    def __getitem__(self, index: int):
        """Load one patch and optional labels.

        Args:
            index: Sample index.

        Returns:
            Tuple of:
                patch_tensor: Tensor with shape (1, D, H, W)
                labels_tensor: Tensor with shape (num_classes,)
                series_uid: UID string
                patch_array: Numpy array with shape (D, H, W) for visualization
        """
        patch_path = self.patch_files[index]
        series_uid = patch_path.stem

        try:
            npz_data = np.load(patch_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load NPZ file: {patch_path}") from exc

        if "patch" not in npz_data:
            raise KeyError(f"'patch' key not found in file: {patch_path}")

        patch_array = np.asarray(npz_data["patch"], dtype=np.float32)

        if patch_array.ndim != 3:
            raise ValueError(
                f"Patch must be 3D, but got shape {patch_array.shape} in {patch_path}"
            )

        patch_tensor = torch.from_numpy(patch_array[np.newaxis, ...]).float()

        if series_uid in self.uid_to_labels:
            labels_array = self.uid_to_labels[series_uid].copy()
        else:
            labels_array = np.zeros(len(self.label_cols), dtype=np.float32)

        labels_tensor = torch.from_numpy(labels_array).float()

        return patch_tensor, labels_tensor, series_uid, patch_array


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["val", "test"],
        help="Inference mode.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing ROI NPZ patch files.",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="CSV with series_uid and 14 labels. Required for val mode.",
    )
    parser.add_argument(
        "--cv-dir",
        type=str,
        default=None,
        help="Cross-validation split directory. Required for val mode.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index for validation split.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=[
            "resnet18", "resnet34", "resnet50", "resnet101",
            "densenet121", "densenet169",
            "efficientnet_b0", "efficientnet_b2", "efficientnet_b3",
            "efficientnet_b4", "efficientnet_b7",
            "vit", "unet3d", "mobilenetv2", "mobilenetv3", "mobilenetv4",
            "swin", "convnext", "inception",
            "seresnet10", "seresnet14", "seresnet18", "seresnet34",
            "seresnet50", "seresnet101",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--save-samples",
        type=int,
        default=16,
        help="Number of sample images to save.",
    )
    parser.add_argument(
        "--sample-strategy",
        type=str,
        default="even",
        choices=["even", "random", "head"],
        help="Strategy for selecting visualization samples.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k predictions to display on sample images.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for positive prediction summary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_args(args) -> None:
    """Validate input arguments.

    Args:
        args: Parsed CLI arguments.
    """
    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.mode == "val":
        if args.labels_csv is None:
            raise ValueError("--labels-csv is required when --mode val")
        if args.cv_dir is None:
            raise ValueError("--cv-dir is required when --mode val")

        labels_csv = Path(args.labels_csv)
        cv_dir = Path(args.cv_dir) / f"fold_{args.fold}"

        if not labels_csv.exists():
            raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
        if not cv_dir.exists():
            raise FileNotFoundError(f"CV fold directory not found: {cv_dir}")
        if not (cv_dir / "val_indices.npy").exists():
            raise FileNotFoundError(f"Missing val_indices.npy in {cv_dir}")


def build_model(arch: str, device: torch.device) -> torch.nn.Module:
    """Build model by architecture name.

    Args:
        arch: Architecture name.
        device: Torch device.

    Returns:
        Instantiated model on target device.
    """
    if arch == "densenet121":
        model = DenseNet3D(num_classes = 14, block_config = (6, 12, 24, 16))
    elif arch == "densenet169":
        model = DenseNet3D(num_classes = 14, block_config = (6, 12, 32, 32))
    elif arch == "efficientnet_b0":
        model = EfficientNet3D(num_classes = 14, variant = "b0")
    elif arch == "efficientnet_b2":
        model = EfficientNet3D(num_classes = 14, variant = "b2")
    elif arch == "efficientnet_b3":
        model = EfficientNet3D(num_classes = 14, variant = "b2")
    elif arch == "efficientnet_b4":
        model = EfficientNet3D(num_classes = 14, variant = "b2")
    elif arch == "efficientnet_b7":
        model = EfficientNet3D(num_classes = 14, variant = "b2")
    elif arch == "vit":
        model = VisionTransformer3D(num_classes = 14)
    elif arch == "unet3d":
        model = UNet3D(num_classes = 14)
    elif arch == "mobilenetv2":
        model = MobileNetV2_3D(num_classes = 14)
    elif arch == "mobilenetv3":
        model = MobileNetV3_3D(num_classes = 14)
    elif arch == "mobilenetv4":
        model = MobileNetV4_3D(num_classes = 14, variant = "medium")
    elif arch == "swin":
        model = SwinTransformer3D(num_classes = 14)
    elif arch == "convnext":
        model = ConvNeXt3D(num_classes = 14)
    elif arch == "inception":
        model = Inception3D(num_classes = 14)
    elif arch == "seresnet10":
        model = SEResNet3D(num_classes = 14, depth = 10)
    elif arch == "seresnet14":
        model = SEResNet3D(num_classes = 14, depth = 14)
    elif arch == "seresnet18":
        model = SEResNet3D(num_classes = 14, depth = 18)
    elif arch == "seresnet34":
        model = SEResNet3D(num_classes = 14, depth = 34)
    elif arch == "seresnet50":
        model = SEResNet3D(num_classes = 14, depth = 50)
    elif arch == "seresnet101":
        model = SEResNet3D(num_classes = 14, depth = 101)
    else:
        depth_map = {
            "resnet18": 18,
            "resnet34": 34,
            "resnet50": 50,
            "resnet101": 101,
        }
        if arch not in depth_map:
            raise ValueError(f"Unsupported architecture: {arch}")
        model = ResNet3D(num_classes = 14, depth = depth_map[arch])

    return model.to(device)


def load_labels_df(labels_csv: str | None) -> pd.DataFrame | None:
    """Load labels CSV if provided.

    Args:
        labels_csv: Labels CSV path.

    Returns:
        Labels dataframe or None.
    """
    if labels_csv is None:
        return None

    labels_df = pd.read_csv(labels_csv)

    required_cols = {"series_uid", *LABEL_COLS}
    missing_cols = required_cols.difference(labels_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in labels CSV: {sorted(missing_cols)}")

    return labels_df


def load_patch_files(data_dir: str) -> tuple[list[Path], dict[str, Path]]:
    """Load patch files from directory.

    Args:
        data_dir: Directory containing NPZ files.

    Returns:
        Tuple of patch file list and UID-to-path mapping.
    """
    patch_dir = Path(data_dir)
    patch_files = sorted(patch_dir.glob("*.npz"))

    if not patch_files:
        raise FileNotFoundError(f"No NPZ files found in: {patch_dir}")

    uid_to_patch = {patch_file.stem: patch_file for patch_file in patch_files}
    return patch_files, uid_to_patch


def build_inference_file_list(
    mode: str,
    patch_files: list[Path],
    uid_to_patch: dict[str, Path],
    labels_df: pd.DataFrame | None,
    cv_dir: str | None,
    fold: int,
) -> tuple[list[Path], pd.DataFrame | None]:
    """Build file list for inference.

    Args:
        mode: Inference mode.
        patch_files: All patch files.
        uid_to_patch: Mapping from series_uid to patch path.
        labels_df: Labels dataframe for validation mode.
        cv_dir: CV split directory.
        fold: Fold index.

    Returns:
        Tuple of selected patch files and matching labels dataframe.
    """
    if mode == "test":
        return patch_files, None

    if labels_df is None or cv_dir is None:
        raise ValueError("labels_df and cv_dir are required for val mode.")

    fold_dir = Path(cv_dir) / f"fold_{fold}"
    val_indices = np.load(fold_dir / "val_indices.npy")

    selected_rows = labels_df.iloc[val_indices].copy()

    selected_files = []
    missing_uids = []

    for _, row in selected_rows.iterrows():
        series_uid = row["series_uid"]
        if series_uid in uid_to_patch:
            selected_files.append(uid_to_patch[series_uid])
        else:
            missing_uids.append(series_uid)

    if missing_uids:
        logger.warning("Missing %d validation patches.", len(missing_uids))
        logger.warning("First 10 missing series_uids: %s", missing_uids[:10])

    selected_uids = {patch_path.stem for patch_path in selected_files}
    selected_rows = selected_rows[selected_rows["series_uid"].isin(selected_uids)].copy()
    selected_rows = selected_rows.reset_index(drop = True)

    return selected_files, selected_rows


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """Load checkpoint into model.

    Args:
        model: Model instance.
        checkpoint_path: Path to checkpoint file.
        device: Torch device.
    """
    checkpoint = torch.load(checkpoint_path, map_location = device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[len("module."):]] = value
        else:
            cleaned_state_dict[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict = False)

    if missing_keys:
        logger.warning("Missing keys when loading checkpoint: %d", len(missing_keys))
        logger.warning("First 10 missing keys: %s", missing_keys[:10])

    if unexpected_keys:
        logger.warning("Unexpected keys when loading checkpoint: %d", len(unexpected_keys))
        logger.warning("First 10 unexpected keys: %s", unexpected_keys[:10])


def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[np.ndarray]]:
    """Run model inference.

    Args:
        model: Model instance.
        dataloader: Inference dataloader.
        device: Torch device.

    Returns:
        Tuple of:
            all_probs: Prediction probabilities, shape (N, C)
            all_labels: Labels, shape (N, C)
            all_series_uids: List of series_uids
            all_patches: List of patch arrays for visualization
    """
    model.eval()

    prob_batches = []
    label_batches = []
    series_uids = []
    patch_arrays = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc = "Inference"):
            patches, labels, batch_series_uids, batch_patch_arrays = batch

            patches = patches.to(device, non_blocking = True)

            logits = model(patches)
            probs = torch.sigmoid(logits)

            prob_batches.append(probs.cpu().numpy())
            label_batches.append(labels.numpy())
            series_uids.extend(list(batch_series_uids))

            for patch_array in batch_patch_arrays:
                patch_arrays.append(np.asarray(patch_array))

    all_probs = np.vstack(prob_batches)
    all_labels = np.vstack(label_batches)

    return all_probs, all_labels, series_uids, patch_arrays


def get_positive_label_names(label_vector: np.ndarray, threshold: float = 0.5) -> list[str]:
    """Convert label vector to positive label names.

    Args:
        label_vector: Label vector.
        threshold: Positive threshold.

    Returns:
        List of positive label names.
    """
    positive_names = []
    for index, value in enumerate(label_vector):
        if value >= threshold:
            positive_names.append(LABEL_COLS[index])
    return positive_names


def get_top_k_predictions(prob_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
    """Get top-k prediction classes.

    Args:
        prob_vector: Probability vector.
        top_k: Number of top classes.

    Returns:
        List of (label_name, score).
    """
    top_indices = np.argsort(prob_vector)[::-1][:top_k]
    top_items = [(LABEL_COLS[index], float(prob_vector[index])) for index in top_indices]
    return top_items


def normalize_for_display(image_2d: np.ndarray) -> np.ndarray:
    """Normalize 2D image for display.

    Args:
        image_2d: Input 2D image.

    Returns:
        Normalized image in [0, 1].
    """
    image_2d = np.asarray(image_2d, dtype = np.float32)

    min_value = float(image_2d.min())
    max_value = float(image_2d.max())

    if math.isclose(max_value, min_value):
        return np.zeros_like(image_2d, dtype = np.float32)

    return (image_2d - min_value) / (max_value - min_value)


def save_sample_visualization(
    patch_array: np.ndarray,
    series_uid: str,
    prob_vector: np.ndarray,
    output_path: Path,
    top_k: int,
    label_vector: np.ndarray | None = None,
) -> None:
    """Save 3-view sample visualization image.

    Args:
        patch_array: 3D patch array with shape (D, H, W).
        series_uid: Series UID.
        prob_vector: Probability vector.
        output_path: PNG output path.
        top_k: Number of top predictions to show.
        label_vector: Optional ground-truth label vector.
    """
    depth, height, width = patch_array.shape

    axial_index = depth // 2
    coronal_index = height // 2
    sagittal_index = width // 2

    axial_slice = normalize_for_display(patch_array[axial_index, :, :])
    coronal_slice = normalize_for_display(patch_array[:, coronal_index, :])
    sagittal_slice = normalize_for_display(patch_array[:, :, sagittal_index])

    top_items = get_top_k_predictions(prob_vector, top_k = top_k)
    top_text = "\n".join([f"{name}: {score:.4f}" for name, score in top_items])

    if label_vector is not None and np.any(label_vector > 0.5):
        gt_names = get_positive_label_names(label_vector, threshold = 0.5)
        gt_text = "\n".join(gt_names[:6])
        if len(gt_names) > 6:
            gt_text += "\n..."
    else:
        gt_text = "None"

    figure = plt.figure(figsize = (15, 5))

    axis_1 = figure.add_subplot(1, 3, 1)
    axis_1.imshow(axial_slice, cmap = "gray")
    axis_1.set_title(f"Axial @ z={axial_index}")
    axis_1.axis("off")

    axis_2 = figure.add_subplot(1, 3, 2)
    axis_2.imshow(coronal_slice, cmap = "gray")
    axis_2.set_title(f"Coronal @ y={coronal_index}")
    axis_2.axis("off")

    axis_3 = figure.add_subplot(1, 3, 3)
    axis_3.imshow(sagittal_slice, cmap = "gray")
    axis_3.set_title(f"Sagittal @ x={sagittal_index}")
    axis_3.axis("off")

    figure.suptitle(f"series_uid: {series_uid}", fontsize = 12)

    figure.text(
        0.02,
        0.02,
        f"Top-{top_k} predictions:\n{top_text}",
        fontsize = 10,
        va = "bottom",
        ha = "left",
        family = "monospace",
    )

    figure.text(
        0.72,
        0.02,
        f"GT labels:\n{gt_text}",
        fontsize = 10,
        va = "bottom",
        ha = "left",
        family = "monospace",
    )

    plt.tight_layout(rect = [0, 0.08, 1, 0.95])
    figure.savefig(output_path, dpi = 150, bbox_inches = "tight")
    plt.close(figure)


def select_sample_indices(
    num_samples_total: int,
    save_samples: int,
    strategy: str,
    seed: int,
) -> list[int]:
    """Select sample indices for visualization.

    Args:
        num_samples_total: Total number of samples.
        save_samples: Number of samples to save.
        strategy: Sampling strategy.
        seed: Random seed.

    Returns:
        List of selected indices.
    """
    if num_samples_total <= 0 or save_samples <= 0:
        return []

    save_samples = min(save_samples, num_samples_total)

    if strategy == "head":
        return list(range(save_samples))

    if strategy == "random":
        rng = random.Random(seed)
        indices = list(range(num_samples_total))
        rng.shuffle(indices)
        return sorted(indices[:save_samples])

    if save_samples == 1:
        return [0]

    step = (num_samples_total - 1) / (save_samples - 1)
    indices = [round(index * step) for index in range(save_samples)]
    return sorted(set(indices))


def build_prediction_dataframe(
    series_uids: list[str],
    probs: np.ndarray,
) -> pd.DataFrame:
    """Build prediction dataframe.

    Args:
        series_uids: List of series_uids.
        probs: Prediction probabilities.

    Returns:
        Prediction dataframe.
    """
    prediction_df = pd.DataFrame(probs, columns = LABEL_COLS)
    prediction_df.insert(0, "series_uid", series_uids)
    return prediction_df


def build_validation_dataframe(
    series_uids: list[str],
    probs: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Build validation dataframe with predictions and labels.

    Args:
        series_uids: List of series_uids.
        probs: Prediction probabilities.
        labels: Ground-truth labels.

    Returns:
        Validation dataframe.
    """
    prediction_df = pd.DataFrame(probs, columns = [f"pred_{name}" for name in LABEL_COLS])
    label_df = pd.DataFrame(labels, columns = [f"label_{name}" for name in LABEL_COLS])

    merged_df = pd.concat([prediction_df, label_df], axis = 1)
    merged_df.insert(0, "series_uid", series_uids)

    return merged_df


def build_summary(
    args,
    num_samples: int,
    probs: np.ndarray,
    labels: np.ndarray | None,
) -> dict:
    """Build summary dictionary.

    Args:
        args: Parsed CLI arguments.
        num_samples: Number of inferred samples.
        probs: Prediction probabilities.
        labels: Ground-truth labels or None.

    Returns:
        Summary dictionary.
    """
    summary = {
        "mode": args.mode,
        "arch": args.arch,
        "checkpoint": str(args.checkpoint),
        "num_samples": int(num_samples),
        "num_classes": int(probs.shape[1]),
        "threshold": float(args.threshold),
        "mean_prob_per_class": {
            label_name: float(probs[:, index].mean())
            for index, label_name in enumerate(LABEL_COLS)
        },
        "positive_count_per_class_at_threshold": {
            label_name: int((probs[:, index] >= args.threshold).sum())
            for index, label_name in enumerate(LABEL_COLS)
        },
    }

    if labels is not None:
        summary["ground_truth_positive_count_per_class"] = {
            label_name: int((labels[:, index] >= 0.5).sum())
            for index, label_name in enumerate(LABEL_COLS)
        }

    return summary


def save_outputs(
    args,
    output_dir: Path,
    probs: np.ndarray,
    labels: np.ndarray,
    series_uids: list[str],
    patch_arrays: list[np.ndarray],
) -> None:
    """Save inference outputs.

    Args:
        args: Parsed CLI arguments.
        output_dir: Output directory.
        probs: Prediction probabilities.
        labels: Ground-truth labels.
        series_uids: Series UIDs.
        patch_arrays: Patch arrays for visualization.
    """
    prediction_df = build_prediction_dataframe(series_uids, probs)
    prediction_csv_path = output_dir / "predictions.csv"
    prediction_npy_path = output_dir / "predictions.npy"

    prediction_df.to_csv(prediction_csv_path, index = False)
    np.save(prediction_npy_path, probs)

    logger.info("Saved prediction CSV: %s", prediction_csv_path)
    logger.info("Saved prediction NPY: %s", prediction_npy_path)

    if args.mode == "val":
        validation_df = build_validation_dataframe(series_uids, probs, labels)
        validation_csv_path = output_dir / "val_with_labels.csv"
        validation_df.to_csv(validation_csv_path, index = False)
        logger.info("Saved validation CSV: %s", validation_csv_path)

    sample_dir = output_dir / "sample_images"
    sample_dir.mkdir(parents = True, exist_ok = True)

    selected_indices = select_sample_indices(
        num_samples_total = len(series_uids),
        save_samples = args.save_samples,
        strategy = args.sample_strategy,
        seed = args.seed,
    )

    logger.info("Saving %d sample visualization images.", len(selected_indices))

    for sample_rank, sample_index in enumerate(selected_indices):
        series_uid = series_uids[sample_index]
        patch_array = patch_arrays[sample_index]
        prob_vector = probs[sample_index]

        label_vector = None
        if args.mode == "val":
            label_vector = labels[sample_index]

        image_output_path = sample_dir / f"{sample_rank:03d}_{series_uid}.png"

        save_sample_visualization(
            patch_array = patch_array,
            series_uid = series_uid,
            prob_vector = prob_vector,
            output_path = image_output_path,
            top_k = args.top_k,
            label_vector = label_vector,
        )

    summary = build_summary(
        args = args,
        num_samples = len(series_uids),
        probs = probs,
        labels = labels if args.mode == "val" else None,
    )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding = "utf-8") as file:
        json.dump(summary, file, indent = 2, ensure_ascii = False)

    logger.info("Saved summary JSON: %s", summary_path)


def build_dataloader(
    selected_files: list[Path],
    labels_df: pd.DataFrame | None,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Build inference dataloader.

    Args:
        selected_files: Selected NPZ patch files.
        labels_df: Optional labels dataframe.
        batch_size: Batch size.
        num_workers: Number of dataloader workers.

    Returns:
        Inference dataloader.
    """
    dataset = Eric3DInferenceDataset(
        patch_files = selected_files,
        labels_df = labels_df,
        label_cols = LABEL_COLS,
    )

    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = torch.cuda.is_available(),
    )

    return dataloader


def main():
    """Main entry."""
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output)
    output_dir.mkdir(parents = True, exist_ok = True)

    logger.info("Using device: %s", device)
    logger.info("Inference mode: %s", args.mode)
    logger.info("Architecture: %s", args.arch)

    labels_df = load_labels_df(args.labels_csv)
    patch_files, uid_to_patch = load_patch_files(args.data_dir)

    selected_files, selected_labels_df = build_inference_file_list(
        mode = args.mode,
        patch_files = patch_files,
        uid_to_patch = uid_to_patch,
        labels_df = labels_df,
        cv_dir = args.cv_dir,
        fold = args.fold,
    )

    logger.info("Total patch files found: %d", len(patch_files))
    logger.info("Selected files for inference: %d", len(selected_files))

    dataloader = build_dataloader(
        selected_files = selected_files,
        labels_df = selected_labels_df,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
    )

    model = build_model(args.arch, device)
    load_checkpoint(model, args.checkpoint, device)

    probs, labels, series_uids, patch_arrays = run_inference(
        model = model,
        dataloader = dataloader,
        device = device,
    )

    save_outputs(
        args = args,
        output_dir = output_dir,
        probs = probs,
        labels = labels,
        series_uids = series_uids,
        patch_arrays = patch_arrays,
    )

    logger.info("=" * 80)
    logger.info("Inference completed successfully.")
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [
            logging.StreamHandler()
        ],
    )
    main()