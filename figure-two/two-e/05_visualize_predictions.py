"""
05_visualize_predictions.py
===========================

Step 5 of the Figure 2E gesture-classification pipeline: render the
paper-figure confusion matrices from the per-gesture-block 5-fold CV.

For each participant, writes to ``results/``:

    <P>_confusion_matrix_paper.svg  -- compact paper-figure confusion matrix
        (``CM_FIG_WIDTH_IN`` wide); only the diagonal is labelled with the
        integer per-class accuracy %, cells are coloured by the row-normalised
        %, and the background is transparent for assembly in the manuscript.

plus one shared ``confusion_matrix_colorbar.svg`` for all three matrices.

(The original script also wrote a DIN-A4 confusion matrix and a predictions PNG
per participant; those are intentionally omitted in this two-e variant.)

Like step 4, this runs on the RMS features produced by step 3 and reproduces the
published Figure 2E confusion-matrix panels. The features are not distributed
with this repository; they are available from the authors on reasonable request.

Input / Output
--------------
    in : ``data/<P>_rms_features.npz``          (RMS features, from step 3)
    out: ``results/<P>_confusion_matrix_paper.svg``
         ``results/confusion_matrix_colorbar.svg``

Dependencies
------------
    numpy, matplotlib, scikit-learn, catboost
        (pip install numpy matplotlib scikit-learn catboost)

Usage
-----
    uv run python figure-two/two-e/05_visualize_predictions.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# ============================================================================
# Confusion-matrix appearance — change a value here to restyle every CM plot.
# Cell color encodes the sample count (sequential colormap, like the paper
# figure). Gestures are always shown in the canonical GESTURES order
# (Rest, Power Grasp, Pinch, Tripod Pinch).
# ============================================================================
CM_CMAP = "Blues"            # any matplotlib colormap name
CM_TEXT_ON_LIGHT = "black"   # annotation color on light (low-count) cells
CM_TEXT_ON_DARK = "white"    # annotation color on dark (high-count) cells
CM_GRID_COLOR = "white"      # thin separator drawn between cells
CM_NA_TEXT = "N/A"           # shown for a ground-truth class with no samples
CM_ANNOT_FONTSIZE = 9
CM_TICK_FONTSIZE = 9
CM_FIG_WIDTH_IN = 1.55       # paper-figure width in inches (SVG)
CM_FIG_WIDTH_A4_IN = 210.0 / 25.4  # DIN A4 width (210 mm) in inches
CM_CBAR_TICKS = [0, 20, 40, 60, 80, 100]  # % ticks for the 0-100 color scale
# ============================================================================

from _cv import CATBOOST_PARAMS, RESULTS_DIR, cross_validate
from _pipeline import load_step

_labeler = load_step("labeler")  # -> 01_labeler.py
GESTURE_COLORS = _labeler.GESTURE_COLORS
GESTURES = _labeler.GESTURES

_rms = load_step("rms_classification")  # -> 04_classification.py
RMS_DATASET_PATHS = _rms.RMS_DATASET_PATHS
keep_mask_from_bad_spec = _rms.keep_mask_from_bad_spec
load_rms_dataset = _rms.load_rms_dataset


def _gesture_cmap() -> ListedColormap:
    return ListedColormap([GESTURE_COLORS[g] for g in GESTURES])


def _legend_handles() -> list[Patch]:
    return [Patch(color=GESTURE_COLORS[g], label=g) for g in GESTURES]


def plot_predictions_vs_gt(
    participant: str,
    t: np.ndarray,
    signal_envelope: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_boundaries_t: np.ndarray,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(
        4, 1, figsize=(18, 7), sharex=True,
        gridspec_kw={"height_ratios": [4, 0.5, 0.5, 0.4], "hspace": 0.05},
    )

    axes[0].plot(t, signal_envelope, color="black", linewidth=0.5)
    axes[0].set_ylabel("RMS\n(mean over channels)", fontsize=9)
    axes[0].set_title(
        f"{participant} — out-of-fold prediction vs ground truth "
        f"(per-region 5-fold CV, CatBoost MyoGestic defaults)"
    )
    for tb in region_boundaries_t:
        axes[0].axvline(tb, color="lightgrey", linewidth=0.5, zorder=0)
    axes[0].legend(handles=_legend_handles(), loc="upper right",
                   fontsize=8, framealpha=0.9, ncol=4)

    cmap = _gesture_cmap()
    norm = BoundaryNorm(np.arange(-0.5, len(GESTURES) + 0.5), cmap.N)
    extent = (float(t[0]), float(t[-1]), 0.0, 1.0)

    axes[1].imshow(y_true[None, :], aspect="auto", cmap=cmap, norm=norm,
                   extent=extent, interpolation="nearest")
    axes[1].set_yticks([0.5])
    axes[1].set_yticklabels(["Ground truth"], fontsize=9)

    axes[2].imshow(y_pred[None, :], aspect="auto", cmap=cmap, norm=norm,
                   extent=extent, interpolation="nearest")
    axes[2].set_yticks([0.5])
    axes[2].set_yticklabels(["Predicted"], fontsize=9)

    errors = (y_true != y_pred).astype(np.int8)
    axes[3].imshow(errors[None, :], aspect="auto",
                   cmap=ListedColormap(["white", "red"]),
                   vmin=0, vmax=1, extent=extent, interpolation="nearest")
    axes[3].set_yticks([0.5])
    axes[3].set_yticklabels(["Errors"], fontsize=9)

    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(t[0], t[-1])

    overall_acc = accuracy_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    fig.text(
        0.99, 0.005,
        f"acc={overall_acc:.3f}  f1_macro={overall_f1:.3f}  "
        f"errors={int(errors.sum())}/{len(errors)}",
        ha="right", fontsize=8, style="italic",
    )

    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    participant: str, y_true: np.ndarray, y_pred: np.ndarray, output_path: str,
    *, width_in: float = CM_FIG_WIDTH_IN, annot: str = "diag_pct",
) -> None:
    """Confusion matrix in the PlayAgain paper format (cf. PlayAgain_abb3).

    - Cell color encodes the row-normalized percentage of that ground-truth
      class (sequential ``CM_CMAP``), on a fixed 0-100% scale.
    - ``width_in``: exact saved figure width in inches; the SVG has a
      transparent background.
    - ``annot`` controls both the cell labels and the figure chrome:
        - ``"diag_pct"``: bare square paper panel — only the diagonal is
          labelled with the per-class accuracy as an integer percent (no
          decimals), off-diagonal blank, and NO title/ticks/axis-labels/
          colorbar (the grid fills the whole width_in x width_in canvas).
        - ``"full"``: 6:5 figure with full chrome (title, class ticks on
          both axes, "Prediction"/"Ground Truth" labels, 0-100% colorbar);
          count on every cell and count + one-decimal accuracy percent on
          the diagonal.
      Either way a ground-truth class with no samples shows ``CM_NA_TEXT``
      on its diagonal. Rows/cols follow the canonical GESTURES order.
    """
    n = len(GESTURES)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    row_sums = cm.sum(axis=1)
    # Row-normalized percentage (per ground-truth class); empty rows -> 0%.
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    cm_pct = 100.0 * cm / safe_sums[:, None]

    # "diag_pct" is the bare paper panel: a square, full-bleed grid with no
    # title/ticks/axis-labels/colorbar (those go in the paper caption), so
    # the matrix fills the exact width_in canvas and stays legible at 1.55".
    # "full" keeps the 6:5 aspect and the chrome; constrained_layout packs
    # it inside the exact width_in canvas.
    bare = annot == "diag_pct"
    if bare:
        fig, ax = plt.subplots(figsize=(width_in, width_in))
        ax.set_position([0.0, 0.0, 1.0, 1.0])
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        fig, ax = plt.subplots(
            figsize=(width_in, width_in * 5 / 6),
            constrained_layout=True,
        )
    im = ax.imshow(cm_pct, cmap=CM_CMAP, vmin=0, vmax=100)

    if not bare:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(GESTURES, fontsize=CM_TICK_FONTSIZE)
        ax.set_yticklabels(GESTURES, fontsize=CM_TICK_FONTSIZE)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground Truth")
        ax.set_title(participant)

    # Thin separators between cells (reference-style boxed look).
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=CM_GRID_COLOR, linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n):
        for j in range(n):
            count = int(cm[i, j])
            on_dark = cm_pct[i, j] > 50.0
            color = CM_TEXT_ON_DARK if on_dark else CM_TEXT_ON_LIGHT
            if annot == "diag_pct":
                if i != j:
                    continue
                if row_sums[i] == 0:
                    ax.text(j, i, CM_NA_TEXT, ha="center", va="center",
                            color=CM_TEXT_ON_LIGHT, fontsize=CM_ANNOT_FONTSIZE)
                else:
                    ax.text(j, i, f"{cm_pct[i, j]:.0f}%", ha="center",
                            va="center", color=color,
                            fontsize=CM_ANNOT_FONTSIZE)
                continue
            # annot == "full"
            if row_sums[i] == 0:
                ax.text(j, i, CM_NA_TEXT, ha="center", va="center",
                        color=CM_TEXT_ON_LIGHT, fontsize=CM_ANNOT_FONTSIZE)
                continue
            if i == j:
                acc = 100.0 * count / row_sums[i]
                text = f"{count}\n{acc:.1f}%"
            else:
                text = f"{count}"
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=CM_ANNOT_FONTSIZE)

    if not bare:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                            ticks=CM_CBAR_TICKS)
        cbar.ax.set_yticklabels([f"{v}%" for v in CM_CBAR_TICKS])
        cbar.ax.tick_params(length=2)
        cbar.outline.set_visible(False)

    # No bbox_inches="tight": that would crop to content and change the
    # width; the layout already fits everything in the fixed canvas.
    fig.savefig(output_path, format="svg", transparent=True)
    plt.close(fig)


def plot_cm_colorbar(
    output_path: str, height_in: float = CM_FIG_WIDTH_IN,
) -> None:
    """Standalone vertical colorbar matching the confusion-matrix color scale.

    The colored bar is exactly ``height_in`` tall — i.e. the same height as
    the (square) paper confusion matrix — so the two SVGs line up when placed
    side by side. Same 0-100% scale, 20% ticks and ``CM_CMAP`` as the matrix;
    transparent background, no outline. ``bbox_inches="tight"`` only crops
    surrounding whitespace (and keeps the % tick labels), it does not rescale,
    so the bar stays exactly ``height_in``.
    """
    fig = plt.figure(figsize=(0.5, height_in))
    # Full figure height -> bar height == height_in == matrix height.
    cax = fig.add_axes([0.0, 0.0, 0.34, 1.0])
    sm = plt.cm.ScalarMappable(cmap=CM_CMAP, norm=Normalize(vmin=0, vmax=100))
    cbar = fig.colorbar(sm, cax=cax, ticks=CM_CBAR_TICKS)
    cbar.ax.set_yticklabels([f"{v}%" for v in CM_CBAR_TICKS],
                            fontsize=CM_TICK_FONTSIZE)
    cbar.ax.tick_params(length=2)
    cbar.outline.set_visible(False)
    fig.savefig(output_path, format="svg", transparent=True,
                bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def run_and_visualize(participant: str, path: str, n_splits: int = 5) -> dict:
    print(f"\n=== {participant} ===")
    print(f"  loading {path}")
    data = load_rms_dataset(path)

    bad_mask = keep_mask_from_bad_spec(
        data["kept_channels_1idx"], data["bad_channels"]
    )
    features = data["features"][bad_mask]
    frame_mask = data["region_ids_per_frame"] >= 0
    X = features[:, frame_mask].T.astype(np.float32)
    y = data["labels"][frame_mask].astype(np.int64)
    region_ids = data["region_ids_per_frame"][frame_mask]

    counts = {GESTURES[i]: int((y == i).sum()) for i in range(len(GESTURES))}
    classes_with_data = sum(c > 0 for c in counts.values())
    if classes_with_data < 2:
        print(f"  only {classes_with_data} class(es) — skipping")
        return {"participant": participant, "skipped": "single_class"}

    print(f"  feature matrix: {X.shape}; class counts: {counts}")
    print(f"  running {n_splits}-fold per-gesture-block CV with CatBoost "
          f"(task_type={CATBOOST_PARAMS['task_type']})...")
    fold_results, oof_pred, fold_assignments = cross_validate(
        X, y, region_ids, n_splits=n_splits, group_by=y,
    )

    valid = oof_pred >= 0
    n_skipped = int((~valid).sum())
    if n_skipped:
        print(f"  {n_skipped} frames not predicted (region too short for "
              f"{n_splits}-fold) — excluded from plots")

    # two-e keeps only the paper-figure confusion matrix (+ the shared colour
    # bar, written in main()). The DIN-A4 CM and the predictions PNG that the
    # original script also produced are intentionally not generated here.
    cm_paper = os.path.join(
        RESULTS_DIR, f"{participant}_confusion_matrix_paper.svg")
    plot_confusion_matrix(participant, y[valid], oof_pred[valid], cm_paper,
                          width_in=CM_FIG_WIDTH_IN, annot="diag_pct")
    print(f"  saved {cm_paper}")

    return {"participant": participant, "fold_results": fold_results,
            "n_frames_predicted": int(valid.sum())}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for participant, path in RMS_DATASET_PATHS.items():
        if not os.path.isfile(path):
            print(f"\n=== {participant} === missing: {path}")
            continue
        run_and_visualize(participant, path)

    cbar_svg = os.path.join(RESULTS_DIR, "confusion_matrix_colorbar.svg")
    plot_cm_colorbar(cbar_svg)
    print(f"\nsaved {cbar_svg}")
    print(f"All plots saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
