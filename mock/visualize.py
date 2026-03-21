import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_STYLE = {
    'debug': {
        'cmap': 'gray',
        'dpi': 140,
        'figsize': (8, 6),
        'show_axes': True,
        'title_size': 12,
        'bg': '#111111'
    },
    'report': {
        'cmap': 'bone',
        'dpi': 170,
        'figsize': (9, 6),
        'show_axes': False,
        'title_size': 14,
        'bg': '#f7f7f7'
    }
}


def _ensure_parent(path):
    """Create parent directory for output file.

    Args:
        path (Path | str): Output file path.

    Returns:
        None
    """
    Path(path).parent.mkdir(parents = True, exist_ok = True)


def _apply_axes(ax, style, title):
    """Apply figure style.

    Args:
        ax (plt.Axes): Matplotlib axes.
        style (str): Style key.
        title (str): Title text.

    Returns:
        None
    """
    cfg = _STYLE[style]
    ax.set_title(title, fontsize = cfg['title_size'])
    if not cfg['show_axes']:
        ax.axis('off')


def save_mid_slice(volume, out_path, style, title):
    """Save middle slice image.

    Args:
        volume (np.ndarray): 3D array.
        out_path (str): Output image path.
        style (str): Style key.
        title (str): Title text.

    Returns:
        None
    """
    cfg = _STYLE[style]
    z_mid = volume.shape[0] // 2

    fig, ax = plt.subplots(figsize = cfg['figsize'])
    fig.patch.set_facecolor(cfg['bg'])
    ax.imshow(volume[z_mid], cmap = cfg['cmap'])
    _apply_axes(ax = ax, style = style, title = title)

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi = cfg['dpi'], bbox_inches = 'tight')
    plt.close(fig)


def save_overlay(base_volume, overlay_volume, out_path, style, title):
    """Save overlay figure for one middle slice.

    Args:
        base_volume (np.ndarray): Base 3D array.
        overlay_volume (np.ndarray): Overlay 3D array.
        out_path (str): Output image path.
        style (str): Style key.
        title (str): Title text.

    Returns:
        None
    """
    cfg = _STYLE[style]
    z_mid = base_volume.shape[0] // 2

    fig, ax = plt.subplots(figsize = cfg['figsize'])
    fig.patch.set_facecolor(cfg['bg'])
    ax.imshow(base_volume[z_mid], cmap = cfg['cmap'])
    ov = overlay_volume[z_mid]
    ax.imshow(ov, cmap = 'turbo', alpha = np.clip(ov, 0.0, 1.0) * 0.55)
    _apply_axes(ax = ax, style = style, title = title)

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi = cfg['dpi'], bbox_inches = 'tight')
    plt.close(fig)


def save_candidate_scatter(base_volume, candidates, out_path, style, title):
    """Save candidate points on middle slice.

    Args:
        base_volume (np.ndarray): Base 3D array.
        candidates (list[dict]): Candidate records.
        out_path (str): Output image path.
        style (str): Style key.
        title (str): Title text.

    Returns:
        None
    """
    cfg = _STYLE[style]
    z_mid = base_volume.shape[0] // 2

    fig, ax = plt.subplots(figsize = cfg['figsize'])
    fig.patch.set_facecolor(cfg['bg'])
    ax.imshow(base_volume[z_mid], cmap = cfg['cmap'])

    xs = []
    ys = []
    for candidate in candidates:
        cz, cy, cx = candidate['center_xyz']
        if abs(cz - z_mid) <= 2:
            xs.append(cx)
            ys.append(cy)

    ax.scatter(xs, ys, s = 18, c = '#ff3366', marker = 'o', alpha = 0.8)
    _apply_axes(ax = ax, style = style, title = title)

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi = cfg['dpi'], bbox_inches = 'tight')
    plt.close(fig)


def save_patch_grid(patches, scores, out_path, style, title, n_show = 9):
    """Save patch center slice grid.

    Args:
        patches (np.ndarray): ROI patches, shape (N, D, H, W).
        scores (list[float]): Score list for displayed patches.
        out_path (str): Output image path.
        style (str): Style key.
        title (str): Figure title.
        n_show (int): Number of patches to show.

    Returns:
        None
    """
    cfg = _STYLE[style]
    n_show = min(n_show, len(patches))
    cols = 3
    rows = int(np.ceil(n_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize = (cols * 3.0, rows * 3.0))
    fig.patch.set_facecolor(cfg['bg'])
    axes = np.array(axes).reshape(-1)

    for idx, ax in enumerate(axes):
        if idx < n_show:
            patch = patches[idx]
            mid = patch.shape[0] // 2
            ax.imshow(patch[mid], cmap = cfg['cmap'])
            ax.set_title(f's={scores[idx]:.3f}', fontsize = 9)
        if not cfg['show_axes']:
            ax.axis('off')

    fig.suptitle(title, fontsize = cfg['title_size'])
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi = cfg['dpi'], bbox_inches = 'tight')
    plt.close(fig)


def save_bar(labels, values, out_path, style, title, rotate = 80):
    """Save bar chart.

    Args:
        labels (list[str]): Label names.
        values (list[float]): Values.
        out_path (str): Output image path.
        style (str): Style key.
        title (str): Figure title.
        rotate (int): X tick rotation.

    Returns:
        None
    """
    cfg = _STYLE[style]

    fig, ax = plt.subplots(figsize = (12, 5.5))
    fig.patch.set_facecolor(cfg['bg'])
    ax.bar(range(len(labels)), values, color = '#2f7ed8', alpha = 0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation = rotate, ha = 'right', fontsize = 8)
    if len(values) > 0:
        max_v = float(max(values))
        ax.set_ylim(0.0, max(1.0, max_v * 1.2))
    _apply_axes(ax = ax, style = style, title = title)

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi = cfg['dpi'], bbox_inches = 'tight')
    plt.close(fig)


def save_compare_bars(labels, values_a, values_b, out_path, style, title, name_a = 'max', name_b = 'top-k mean'):
    """Save side-by-side comparison bars.

    Args:
        labels (list[str]): Label names.
        values_a (list[float]): First values.
        values_b (list[float]): Second values.
        out_path (str): Output image path.
        style (str): Style key.
        title (str): Figure title.
        name_a (str): Name for first bars.
        name_b (str): Name for second bars.

    Returns:
        None
    """
    cfg = _STYLE[style]
    idx = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize = (12, 5.5))
    fig.patch.set_facecolor(cfg['bg'])
    ax.bar(idx - width / 2, values_a, width = width, label = name_a, alpha = 0.82)
    ax.bar(idx + width / 2, values_b, width = width, label = name_b, alpha = 0.82)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation = 80, ha = 'right', fontsize = 8)

    merged = list(values_a) + list(values_b)
    if len(merged) > 0:
        max_v = float(max(merged))
        ax.set_ylim(0.0, max(1.0, max_v * 1.2))

    ax.legend(loc = 'upper right')
    _apply_axes(ax = ax, style = style, title = title)

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi = cfg['dpi'], bbox_inches = 'tight')
    plt.close(fig)


def save_hist(values, bins, out_path, style, title):
    """Save histogram figure.

    Args:
        values (list[float] | np.ndarray): Numeric values.
        bins (int): Number of bins.
        out_path (str): Output image path.
        style (str): Style key.
        title (str): Figure title.

    Returns:
        None
    """
    cfg = _STYLE[style]

    fig, ax = plt.subplots(figsize = cfg['figsize'])
    fig.patch.set_facecolor(cfg['bg'])
    ax.hist(values, bins = bins, color = '#f06d4f', alpha = 0.86)
    _apply_axes(ax = ax, style = style, title = title)

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi = cfg['dpi'], bbox_inches = 'tight')
    plt.close(fig)


def dump_json(data, out_path):
    """Write JSON file.

    Args:
        data (dict): Serializable object.
        out_path (str): Output file path.

    Returns:
        None
    """
    _ensure_parent(out_path)
    with open(out_path, 'w', encoding = 'utf-8') as file:
        json.dump(data, file, indent = 2, ensure_ascii = False)


def save_slice_grid(volume, out_path, style, title, num_slices = 9):
    """Save evenly sampled slice grid from a 3D volume.

    Args:
        volume (np.ndarray): 3D volume in z-y-x.
        out_path (str): Output image path.
        style (str): Style key.
        title (str): Figure title.
        num_slices (int): Number of slices to display.

    Returns:
        None
    """
    cfg = _STYLE[style]
    num_slices = max(1, int(num_slices))

    z = volume.shape[0]
    idx = np.linspace(0, z - 1, num = num_slices, dtype = int)

    cols = 3
    rows = int(np.ceil(num_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize = (cols * 3.0, rows * 3.0))
    fig.patch.set_facecolor(cfg['bg'])
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < num_slices:
            zi = int(idx[i])
            ax.imshow(volume[zi], cmap = cfg['cmap'])
            ax.set_title(f'z={zi}', fontsize = 9)
        if not cfg['show_axes']:
            ax.axis('off')

    fig.suptitle(title, fontsize = cfg['title_size'])
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi = cfg['dpi'], bbox_inches = 'tight')
    plt.close(fig)
