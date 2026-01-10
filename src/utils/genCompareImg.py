from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.tools import visual

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}


def has_image_file(folder: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in folder.iterdir())


def select_feature(arr: np.ndarray, feature_index: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return arr[..., 0]
        return arr[..., feature_index]
    raise ValueError(f"Unexpected array shape: {arr.shape}")


def mae_per_sample(pred: np.ndarray, true: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    n_samples = pred.shape[0]
    maes = np.empty(n_samples, dtype=np.float64)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        diff = np.abs(pred[start:end] - true[start:end])
        maes[start:end] = diff.mean(axis=1)
    return maes


def pick_min_max_error_samples(
    pred: np.ndarray,
    true: np.ndarray,
    feature_index: int,
) -> tuple[
    int, np.ndarray, np.ndarray, float,
    int, np.ndarray, np.ndarray, float,
]:
    pred_2d = select_feature(pred, feature_index)
    true_2d = select_feature(true, feature_index)
    if pred_2d.shape != true_2d.shape:
        raise ValueError(f"Pred/true shape mismatch: {pred_2d.shape} vs {true_2d.shape}")
    if pred_2d.ndim != 2:
        raise ValueError(f"Expected 2D array after feature selection, got {pred_2d.ndim}D")
    maes = mae_per_sample(pred_2d, true_2d)
    worst_idx = int(np.argmax(maes))
    best_idx = int(np.argmin(maes))
    return (
        best_idx, pred_2d[best_idx], true_2d[best_idx], float(maes[best_idx]),
        worst_idx, pred_2d[worst_idx], true_2d[worst_idx], float(maes[worst_idx]),
    )


def downsample_series(pred_line: np.ndarray, true_line: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray, int]:
    if max_points <= 0:
        return pred_line, true_line, 1
    n = pred_line.shape[0]
    if n <= max_points:
        return pred_line, true_line, 1
    stride = int(np.ceil(n / max_points))
    return pred_line[::stride], true_line[::stride], stride


def build_title(folder_name: str, label: str) -> str:
    parts = folder_name.split("_")
    if len(parts) >= 2:
        return f"{parts[0]} | {parts[1]} | {label}"
    return f"{folder_name} | {label}"


def build_output_names(output_name: str) -> tuple[str, str, str]:
    name_path = Path(output_name)
    stem = name_path.stem or output_name
    suffix = name_path.suffix
    if not suffix:
        return f"{output_name}_min", f"{output_name}_max", f"{output_name}_last"
    return f"{stem}_min{suffix}", f"{stem}_max{suffix}", f"{stem}_last{suffix}"


def generate_for_folder(
    folder: Path,
    output_name: str,
    feature_index: int,
    max_points: int,
    force: bool,
) -> bool:
    pred_path = folder / "pred.npy"
    true_path = folder / "true.npy"
    if not pred_path.exists() or not true_path.exists():
        return False
    if not force and has_image_file(folder):
        return False

    pred = np.load(pred_path, mmap_mode="r")
    true = np.load(true_path, mmap_mode="r")
    try:
        (
            best_idx,
            best_pred_line,
            best_true_line,
            best_mae,
            worst_idx,
            worst_pred_line,
            worst_true_line,
            worst_mae,
        ) = pick_min_max_error_samples(pred, true, feature_index)
    except ValueError as exc:
        print(f"[skip] {folder.name}: {exc}")
        return False

    pred_2d = select_feature(pred, feature_index)
    true_2d = select_feature(true, feature_index)
    last_idx = pred_2d.shape[0] - 1
    last_pred_line = pred_2d[last_idx]
    last_true_line = true_2d[last_idx]

    min_name, max_name, last_name = build_output_names(output_name)
    best_pred_line, best_true_line, best_stride = downsample_series(
        best_pred_line, best_true_line, max_points
    )
    worst_pred_line, worst_true_line, worst_stride = downsample_series(
        worst_pred_line, worst_true_line, max_points
    )
    last_pred_line, last_true_line, last_stride = downsample_series(
        last_pred_line, last_true_line, max_points
    )

    min_output_path = folder / min_name
    max_output_path = folder / max_name
    last_output_path = folder / last_name
    min_title = build_title(folder.name, "min MAE")
    max_title = build_title(folder.name, "max MAE")
    last_title = build_title(folder.name, "last")
    visual(true=best_true_line, preds=best_pred_line, name=str(min_output_path), title=min_title)
    visual(true=worst_true_line, preds=worst_pred_line, name=str(max_output_path), title=max_title)
    visual(true=last_true_line, preds=last_pred_line, name=str(last_output_path), title=last_title)
    print(
        f"[ok] {folder.name}: min_sample={best_idx}, min_mae={best_mae:.6f}, "
        f"max_sample={worst_idx}, max_mae={worst_mae:.6f}, "
        f"last_sample={last_idx}, "
        f"stride_min={best_stride}, stride_max={worst_stride}, stride_last={last_stride}, "
        f"saved={min_output_path.name},{max_output_path.name},{last_output_path.name}"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate pred/true comparison plots from results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results",
        help="Root results directory.",
    )
    parser.add_argument(
        "--output-name",
        default="pred_true_compare.png",
        help="Base output image filename for each folder (min/max suffix added).",
    )
    parser.add_argument(
        "--feature-index",
        type=int,
        default=-1,
        help="Feature index to plot when data is multivariate.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Max points to plot per line (downsample if larger).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Generate image even if the folder already contains images.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"[skip] results dir not found: {results_dir}")
        return 0

    generated = 0
    for folder in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if generate_for_folder(folder, args.output_name, args.feature_index, args.max_points, args.force):
            generated += 1
    print(f"[done] generated: {generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
