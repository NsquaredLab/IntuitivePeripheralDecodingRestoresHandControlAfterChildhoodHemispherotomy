"""
06_supplementary_figure.py
==========================

Step 6 of the Figure 2E gesture-classification pipeline: generate the
supplementary classification figure as a single combined SVG (the A/B/C/D panel
letters are added later in Illustrator).

Panels:
  A) Example Power Grasp recording (P01): the selected EMG channels stacked,
     bad/discarded channels drawn in the discard colour, the labelled Power
     Grasp span tinted, the rest of the recording on a light-grey background.
  B) Concatenated labelled EMG per participant, one row each, a few
     representative channels per row, gesture-coloured signals. Row width is
     normalised to the longest recording so shorter participants occupy
     proportionally less of the page width.
  C) The per-gesture-block chronological 5-fold split shown on P01's RMS
     feature signal: the feature envelope on top, then one strip per fold with
     the validation chunks coloured (5 graded colours) and the training chunks
     light grey.
  D) Out-of-fold prediction result per participant, one row each, a few
     representative RMS channels, gesture-coloured ground truth, and a red strip
     marking misclassified frames. Same width normalisation as B.

Input / Output
--------------
    in : ``data/<P>.npz`` (panels A/B) + ``data/<P>_rms_features.npz`` (C/D)
         and one raw NAS recording (panel A)
    out: ``results/supplementary/supplementary_combined.svg``

This figure needs the participant data (panels A and B draw the raw EMG traces;
panels C and D use the RMS features), which is not distributed with this
repository because it derives from identifiable human EMG of a vulnerable
participant group. All data are available from the authors on reasonable request
(see ``data/RAW_DATA_ACCESS.md``).

Dependencies
------------
    numpy, scipy, matplotlib, scikit-learn, catboost, pysynclient

Usage
-----
    uv run python figure-two/two-e/06_supplementary_figure.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from pysynclient import get_server_paths

# ---------------------------------------------------------------------------
# Global style: Arial everywhere, min font size 10, editable SVG text.
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "svg.fonttype": "none",   # keep text as editable text (Arial) in Illustrator
})

# Page sizing: fit the usable width of a DIN A4 page (21.0 cm) with 2.0 cm
# margins on the left and right -> 17.0 cm usable width.
CM_PER_IN = 2.54
A4_WIDTH_CM = 21.0
PAGE_MARGIN_CM = 2.0
FIG_W_IN = (A4_WIDTH_CM - 2 * PAGE_MARGIN_CM) / CM_PER_IN   # = 6.693 in (17.0 cm)

from _cv import cross_validate, per_gesture_kfold_splits
from _pipeline import load_step, require_raw_signal

OUTPUT_DIR = load_step("dataset_creation").OUTPUT_DIR  # -> 02_dataset_creation.py
apply_filters = load_step("feature_extraction").apply_filters  # -> 03_feature_extraction.py

_labeler = load_step("labeler")  # -> 01_labeler.py
GESTURE_COLORS = _labeler.GESTURE_COLORS
GESTURES = _labeler.GESTURES
N_BIOSIGNAL_CHANNELS = _labeler.N_BIOSIGNAL_CHANNELS
flatten_biosignal = _labeler.flatten_biosignal
normalize_label = _labeler.normalize_label
parse_channel_spec = _labeler.parse_channel_spec

_rms = load_step("rms_classification")  # -> 04_classification.py
RMS_DATASET_PATHS = _rms.RMS_DATASET_PATHS
keep_mask_from_bad_spec = _rms.keep_mask_from_bad_spec
load_rms_dataset = _rms.load_rms_dataset

# ---------------------------------------------------------------------------
# Central color/style config — tweak here to restyle every panel.
# ---------------------------------------------------------------------------
COL_SIGNAL = "#333333"        # EMG / RMS traces (good channels)
COL_BAD = "#C44E52"           # discarded bad channels / prediction errors
COL_UNLABELLED = "#EFEFEF"    # light-grey background for unlabelled signal
COL_UNLAB_SIGNAL = "#B5B5B5"  # Panel A: trace color in non-extracted regions
COL_TRAIN = "#F2F2F2"         # training chunks in the CV split strip (subtle)
GESTURE_BG_ALPHA = 0.40       # gesture region background tint
HIGHLIGHT_ALPHA = 0.22        # Panel A labelled-span tint

# A more saturated Tripod-Pinch yellow with better contrast on a white
# background than the scheme's pale #FEFABC (display only; the canonical
# GESTURE_COLORS scheme is left untouched).
TRIPOD_YELLOW = "#E0A500"
GESTURE_DISPLAY_COLORS = {**GESTURE_COLORS, "Tripod Pinch": TRIPOD_YELLOW}

# 5 graded fold colors, interpolated between the two scheme anchors
# (#4371CB blue -> #FF868B pink) so they stay on-brand while being distinct.
FOLD_COLORS = ["#4371CB", "#7A6FBE", "#A95FA8", "#D85896", "#FF868B"]

N_REP_CHANNELS = 4            # representative channels per row in B & D
PARTICIPANTS = ["P01", "P01_2", "P02"]
DISPLAY_NAMES = {"P01": "P01 (Session 1)", "P01_2": "P01 (Session 2)",
                 "P02": "P02"}
# Display labels for gestures (lower-case; canonical names stay in
# labeler.GESTURES). Power Grasp is also shortened to "grasp".
GESTURE_LABELS = {"Rest": "rest", "Power Grasp": "grasp", "Pinch": "pinch",
                  "Tripod Pinch": "tripod pinch"}
OUT_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "results", "supplementary")

# File-server-relative dir + the example Power Grasp recording for panel A.
# De-identified here (the real path/filename encoded participant initials and a
# recording date/time); the authors substitute the actual values from a private
# mapping when running this step.
NAS_DIRS = {
    "P01": r"<RAW_DATA_ROOT>\P01\session_01",
}
PANEL_A_FILE = "<panel_a_power_grasp_recording>.pkl"
PANEL_A_PARTICIPANT = "P01"
PANEL_A_CHANNEL_SPEC = "1-64"      # Grid 1 only (first 64-channel array)
PANEL_A_GRID_LABEL = "Grid 1"
PANEL_A_SPREAD_K = 6.5             # vertical gap between channels (higher = more)

# Panel B: filter the raw EMG before plotting (mirrors the pipeline's
# pre-feature filtering step, with the band the user asked for).
PANEL_B_BANDPASS_HZ = (10.0, 500.0)
PANEL_B_NOTCH_HZ = (48.0, 52.0)    # 50 Hz power-line notch (band-stop)
PANEL_B_BUTTER_ORDER = 2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index runs where a boolean mask is True."""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return list(zip(starts, ends))


def _pick_representative_rows(mat: np.ndarray, n: int) -> np.ndarray:
    """Indices of the `n` most active rows (highest std), returned sorted."""
    activity = mat.std(axis=1)
    n = min(n, mat.shape[0])
    idx = np.argsort(activity)[::-1][:n]
    return np.sort(idx)


def _load_concat(participant: str) -> dict:
    raw_path = require_raw_signal(  # raw EMG: request from authors if absent
        os.path.join(OUTPUT_DIR, f"{participant}.npz"))
    z = np.load(raw_path, allow_pickle=False)
    return {
        "signal": z["signal"],
        "fs": int(z["fs"]),
        "bad_channels": str(z["bad_channels"]),
        "original_channels_1idx": z["original_channels_1idx"].astype(np.int64),
        "region_starts": z["region_starts"].astype(np.int64),
        "region_ends": z["region_ends"].astype(np.int64),
        "region_labels": np.array(
            [normalize_label(str(l)) for l in z["region_labels"]]),
    }


def _save(fig, out_path: str, **kw) -> None:
    """Save the figure as SVG (the deliverable). The original script also wrote
    a PNG preview next to it; this two-e variant keeps only the SVG."""
    fig.savefig(out_path, format="svg", transparent=True, **kw)


def _stacked_lines(traces: np.ndarray, t: np.ndarray, spread: float):
    """Build LineCollection segments for vertically stacked traces."""
    centered = traces - traces.mean(axis=1, keepdims=True)
    return [np.column_stack([t, centered[i] + i * spread])
            for i in range(traces.shape[0])]


def _gesture_label(g: str) -> str:
    return GESTURE_LABELS.get(g, g)


def _gesture_signal_color(g: str) -> str:
    """Per-gesture trace color. Rest uses Panel A's 'not extracted' grey for
    visibility/consistency; the others use the team gesture colors (with the
    higher-contrast Tripod-Pinch yellow)."""
    return COL_UNLAB_SIGNAL if g == "Rest" else GESTURE_DISPLAY_COLORS[g]


def _draw_colored_traces(ax, traces_centered: np.ndarray, t: np.ndarray,
                         spread: float, labels_per_sample: np.ndarray,
                         lw: float = 0.5) -> None:
    """Stack centered traces and color each by the per-sample gesture label
    (color travels with the signal instead of a background tint)."""
    for g in GESTURES:
        m = labels_per_sample == g
        if not m.any():
            continue
        segs = [np.column_stack([t, np.where(m, traces_centered[i] + i * spread,
                                             np.nan)])
                for i in range(traces_centered.shape[0])]
        lc = LineCollection(segs, colors=_gesture_signal_color(g),
                            linewidths=lw)
        lc.set_rasterized(True)
        ax.add_collection(lc)


# ---------------------------------------------------------------------------
# Panel A
# ---------------------------------------------------------------------------
def panel_a(out_path: str) -> None:
    base = get_server_paths()["nsquared-nas"]["datasets"]
    rec = os.path.join(base, NAS_DIRS[PANEL_A_PARTICIPANT], PANEL_A_FILE)
    with open(rec, "rb") as f:
        d = pickle.load(f)
    fs = int(d["device_information"]["sampling_frequency"])
    sig = flatten_biosignal(d["biosignal"])  # (384, n)

    kept_zero = sorted(parse_channel_spec(PANEL_A_CHANNEL_SPEC,
                                          N_BIOSIGNAL_CHANNELS))
    ch_1idx = np.array([g + 1 for g in kept_zero])
    sig = sig[np.array(kept_zero)]
    bad_spec = _load_concat(PANEL_A_PARTICIPANT)["bad_channels"]
    keep_mask = keep_mask_from_bad_spec(ch_1idx, bad_spec)  # True = good
    bad_idx = np.where(~keep_mask)[0]

    n_ch, n = sig.shape
    # labelled Power Grasp union mask over time, from labels.json spans
    import json
    labels = json.load(open(os.path.join(
        base, NAS_DIRS[PANEL_A_PARTICIPANT], "labels.json"), encoding="utf-8"))
    entry = next(e for e in labels if e["file"] == PANEL_A_FILE)
    lab = np.zeros(n, dtype=bool)
    for sp in entry["spans"]:
        if normalize_label(sp["label"]) == "Power Grasp":
            lab[max(0, int(sp["start"] * fs)):min(n, int(sp["end"] * fs))] = True

    ds = max(1, n // 6000)
    t = np.arange(n)[::ds] / fs
    lab_ds = lab[::ds]                       # labelled mask on the display grid
    disp = sig[:, ::ds]
    centered = disp - disp.mean(axis=1, keepdims=True)
    spread = float(np.median(np.std(centered, axis=1))) * PANEL_A_SPREAD_K or 1.0
    y_all = centered + np.arange(n_ch)[:, None] * spread   # stacked y per channel

    # Wide, short landscape that fits the usable width of an A4 page; the
    # background stays transparent and the *signals* carry the color instead:
    # grey where the signal is not extracted, blue where it is (Power Grasp).
    fig, ax = plt.subplots(figsize=(FIG_W_IN, 1.3))

    good_idx = np.where(keep_mask)[0]
    grey_segs, blue_segs = [], []
    for i in good_idx:
        y = y_all[i]
        grey_segs.append(np.column_stack([t, np.where(lab_ds, np.nan, y)]))
        blue_segs.append(np.column_stack([t, np.where(lab_ds, y, np.nan)]))
    bad_segs = [np.column_stack([t, y_all[i]]) for i in bad_idx]

    for segs, col, w in (
        (grey_segs, COL_UNLAB_SIGNAL, 0.35),
        (blue_segs, GESTURE_COLORS["Power Grasp"], 0.35),
        (bad_segs, COL_BAD, 0.7),
    ):
        if not segs:
            continue
        lc = LineCollection(segs, colors=col, linewidths=w)
        lc.set_rasterized(True)
        ax.add_collection(lc)

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-spread, n_ch * spread)
    # left axis: clean channel numbering (few ticks so they fit the short panel)
    left_ticks = np.linspace(0, n_ch - 1, 5, dtype=int)
    ax.set_yticks(left_ticks * spread)
    ax.set_yticklabels([str(ch_1idx[i]) for i in left_ticks])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"EMG channel ({PANEL_A_GRID_LABEL})")
    for sp in ax.spines.values():        # no plot borders — cleaner look
        sp.set_visible(False)

    # right axis: discarded channels, tick numbers in red
    if len(bad_idx):
        ax_r = ax.twinx()
        ax_r.set_ylim(ax.get_ylim())
        ax_r.set_yticks(bad_idx * spread)
        ax_r.set_yticklabels([str(ch_1idx[i]) for i in bad_idx])
        ax_r.tick_params(axis="y", colors=COL_BAD)
        for sp in ax_r.spines.values():
            sp.set_visible(False)

    # legend above the plot, one row: extracted (left), discarded (right)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=GESTURE_COLORS["Power Grasp"], lw=1.8,
               label="extracted area"),
        Line2D([0], [0], color=COL_BAD, lw=1.8, label="discarded channel"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, frameon=False, handlelength=1.3, columnspacing=2.5)

    fig.tight_layout(pad=0.4)
    _save(fig, out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Panel B
# ---------------------------------------------------------------------------
def panel_b(out_path: str) -> None:
    label_to_idx = {g: i for i, g in enumerate(GESTURES)}
    data = {p: _load_concat(p) for p in PARTICIPANTS}
    durs = {p: data[p]["signal"].shape[1] / data[p]["fs"] for p in PARTICIPANTS}
    max_dur = max(durs.values())

    n_rows = len(PARTICIPANTS)
    # height ~ Panel A; legend sits above the rows, a single full-width time
    # axis (P01's scale, 0..max_dur) sits below them.
    fig = plt.figure(figsize=(FIG_W_IN, 1.45))
    left, label_w, usable = 0.015, 0.135, 0.83
    top, bot_margin = 0.88, 0.30
    stride = (top - bot_margin) / n_rows
    ax_h = stride * 0.80
    for r, p in enumerate(PARTICIPANTS):
        c = data[p]
        fs = c["fs"]
        keep_mask = keep_mask_from_bad_spec(c["original_channels_1idx"],
                                            c["bad_channels"])
        # Filter the good channels like the pipeline's pre-feature step
        # (band-pass 10-500 Hz + 50 Hz notch), then reorder labelled regions
        # into the canonical Rest -> Grasp -> Pinch -> Tripod Pinch order
        # (stable sort keeps chronological order within each gesture).
        good_f = apply_filters(c["signal"][keep_mask].astype(np.float32), fs,
                               PANEL_B_BANDPASS_HZ, PANEL_B_NOTCH_HZ,
                               PANEL_B_BUTTER_ORDER)
        order = sorted(
            (k for k in range(len(c["region_labels"]))
             if c["region_labels"][k] in label_to_idx),
            key=lambda k: label_to_idx[c["region_labels"][k]],
        )
        seg_sig, seg_lab = [], []
        for k in order:
            s, e = int(c["region_starts"][k]), int(c["region_ends"][k])
            seg_sig.append(good_f[:, s:e])
            seg_lab.append(np.full(e - s, c["region_labels"][k]))
        disp = np.concatenate(seg_sig, axis=1)
        lab_per_sample = np.concatenate(seg_lab)

        n = disp.shape[1]
        ds = max(1, n // 4000)
        t = np.arange(n)[::ds] / fs
        dd = disp[:, ::ds]
        lab_ds = lab_per_sample[::ds]
        rep = _pick_representative_rows(dd, N_REP_CHANNELS)
        traces = dd[rep] - dd[rep].mean(axis=1, keepdims=True)

        width = usable * (durs[p] / max_dur)
        bottom = top - r * stride - ax_h
        ax = fig.add_axes([left + label_w, bottom, width, ax_h])

        spread = float(np.median(np.std(traces, axis=1))) * 4 or 1.0
        _draw_colored_traces(ax, traces, t, spread, lab_ds, lw=0.5)
        ax.set_xlim(0, durs[p])
        ax.set_ylim(-spread, len(rep) * spread)
        ax.set_yticks([])
        ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.text(left, bottom + ax_h / 2,
                 DISPLAY_NAMES[p].replace(" (", "\n("),
                 ha="left", va="center")

    # single shared time axis spanning P01's full width (0 .. max_dur), placed
    # just under the last row (tick distance similar to Panel A)
    last_bottom = top - (n_rows - 1) * stride - ax_h
    ax_t = fig.add_axes([left + label_w, last_bottom - 0.012, usable, 0.001])
    ax_t.set_xlim(0, max_dur)
    ax_t.set_yticks([])
    for sp in ax_t.spines.values():       # no border, just ticks + label
        sp.set_visible(False)
    ax_t.set_xlabel("Time (s)")

    _gesture_line_legend(fig, y=1.0)
    _save(fig, out_path)
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Panel C
# ---------------------------------------------------------------------------
def panel_c(out_path: str, participant: str = "P01", n_splits: int = 5) -> None:
    data = load_rms_dataset(RMS_DATASET_PATHS[participant])
    keep_mask = keep_mask_from_bad_spec(data["kept_channels_1idx"],
                                        data["bad_channels"])
    feats = data["features"][keep_mask]
    frame_mask = data["region_ids_per_frame"] >= 0
    feats = feats[:, frame_mask]
    labels = data["labels"][frame_mask]          # gesture per frame (block-ordered)
    rate = data["feature_rate_hz"]
    n_frames = feats.shape[1]
    t = np.arange(n_frames) / rate

    envelope = feats.mean(axis=0)
    # fold over gesture-class blocks (Rest/Grasp/Pinch/Tripod), not regions
    splits = per_gesture_kfold_splits(labels, n_splits=n_splits)
    block_bounds = np.where(np.diff(labels) != 0)[0] + 1   # 3 gesture-block edges

    fig = plt.figure(figsize=(FIG_W_IN, 1.9))
    lx, wx = 0.12, 0.85
    # top: feature envelope, colored by gesture block (~1/3 of former height);
    # kept low so there is clear space below the legend.
    ax0 = fig.add_axes([lx, 0.70, wx, 0.12])
    for i, g in enumerate(GESTURES):       # labels are integer class indices
        m = labels == i
        if not m.any():
            continue
        ax0.plot(t, np.where(m, envelope, np.nan),
                 color=_gesture_signal_color(g), linewidth=0.7)
    for b in block_bounds:                # 3 separators between gesture blocks
        ax0.axvline(t[b], color="#9A9A9A", linewidth=0.7, zorder=3)
    ax0.set_xlim(t[0], t[-1])
    ax0.set_ylabel("RMS (mean)\n[µV]")
    # align the RMS label's left edge with the 'Fold k' strip labels below
    ax0.yaxis.set_label_coords(-0.075, 0.5)
    ax0.set_xticks([])
    for sp in ax0.spines.values():        # no borders
        sp.set_visible(False)

    # gesture legend above the colored envelope
    _gesture_line_legend(fig, y=1.0)

    # training / validation legend: grey = training, color gradient = validation
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple
    train_h = Patch(facecolor=COL_TRAIN, edgecolor="none")
    val_h = tuple(Patch(facecolor=c, edgecolor="none") for c in FOLD_COLORS)
    fig.legend([train_h, val_h], ["training", "validation"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
               loc="upper right", bbox_to_anchor=(0.98, 0.645), ncol=2,
               frameon=False, handlelength=2.8, handletextpad=0.5,
               columnspacing=1.4)

    # below: one strip per fold (validation colored, training light grey).
    # Bars are kept thin (<= the ~10 pt 'Fold k' label height) and packed
    # close together.
    strip_top, slot, bar_h = 0.58, 0.073, 0.040
    for k, (train_idx, test_idx) in enumerate(splits):
        bottom = strip_top - (k + 1) * slot
        ax = fig.add_axes([lx, bottom, wx, bar_h])
        val = np.zeros(n_frames, dtype=bool)
        val[test_idx] = True
        ax.axhspan(0, 1, color=COL_TRAIN, lw=0)
        for s, e in _contiguous_runs(val):
            ax.axvspan(t[s], t[min(e, n_frames - 1)], color=FOLD_COLORS[k],
                       lw=0)
        for b in block_bounds:            # delimit the gesture blocks
            ax.axvline(t[b], color="#9A9A9A", linewidth=0.7, zorder=3)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5])
        ax.set_yticklabels([f"Fold {k + 1}"])
        ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if k == n_splits - 1:
            ax.set_xlabel("Time (s)")
            xticks = np.arange(0, t[-1] + 1, 50)   # 50 s steps, like Panel B
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{v:.0f}" for v in xticks])
    _save(fig, out_path)
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Panel D
# ---------------------------------------------------------------------------
def panel_d(out_path: str, n_splits: int = 5) -> None:
    # compute oof predictions + assemble per-participant display data
    pdata = {}
    for p in PARTICIPANTS:
        data = load_rms_dataset(RMS_DATASET_PATHS[p])
        keep_mask = keep_mask_from_bad_spec(data["kept_channels_1idx"],
                                            data["bad_channels"])
        feats = data["features"][keep_mask]
        frame_mask = data["region_ids_per_frame"] >= 0
        X = feats[:, frame_mask].T.astype(np.float32)
        y = data["labels"][frame_mask].astype(np.int64)
        region_ids = data["region_ids_per_frame"][frame_mask]
        print(f"  [D] {p}: {X.shape[0]} frames — running {n_splits}-fold CV")
        _, oof, _ = cross_validate(X, y, region_ids, n_splits=n_splits,
                                   group_by=y)
        valid = oof >= 0
        rep = _pick_representative_rows(feats[:, frame_mask][:, valid],
                                        N_REP_CHANNELS)
        pdata[p] = {
            "rate": data["feature_rate_hz"],
            "traces": feats[:, frame_mask][:, valid][rep],
            "y": y[valid],
            "oof": oof[valid],
            "region_ids": region_ids[valid],
            "n": int(valid.sum()),
        }
    durs = {p: pdata[p]["n"] / pdata[p]["rate"] for p in PARTICIPANTS}
    max_dur = max(durs.values())

    n_rows = len(PARTICIPANTS)
    # styled like Panel B: gesture-colored signals (no background tint), legend
    # on top, stacked names on the left, one shared full-width time axis below.
    fig = plt.figure(figsize=(FIG_W_IN, 2.3))
    left, label_w, usable = 0.015, 0.135, 0.83
    top, stride = 0.84, 0.23
    ax_h, err_gap, err_h = 0.135, 0.012, 0.028
    last_err_bottom = 0.0
    for r, p in enumerate(PARTICIPANTS):
        c = pdata[p]
        n = c["n"]
        t_full = np.arange(n) / c["rate"]
        ds = max(1, n // 4000)
        t = t_full[::ds]
        width = usable * (durs[p] / max_dur)
        row_top = top - r * stride
        bottom = row_top - ax_h
        ax = fig.add_axes([left + label_w, bottom, width, ax_h])

        # signal colored by ground-truth gesture (like Panel B), no bg tint
        traces = c["traces"]
        tc = (traces - traces.mean(axis=1, keepdims=True))[:, ::ds]
        y_names = np.array(GESTURES)[c["y"]][::ds]
        spread = float(np.median(np.std(tc, axis=1))) * 4 or 1.0
        _draw_colored_traces(ax, tc, t, spread, y_names, lw=0.5)
        ax.set_xlim(0, durs[p])
        ax.set_ylim(-spread, len(traces) * spread)
        ax.set_yticks([])
        ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        # misprediction strip below the traces (full resolution)
        err = (c["oof"] != c["y"]).astype(bool)
        err_bottom = bottom - err_gap - err_h
        ax_e = fig.add_axes([left + label_w, err_bottom, width, err_h])
        for s, e in _contiguous_runs(err):
            ax_e.axvspan(t_full[s], t_full[min(e, n - 1)], color=COL_BAD, lw=0)
        ax_e.set_xlim(0, durs[p])
        ax_e.set_ylim(0, 1)
        ax_e.set_xticks([])
        ax_e.set_yticks([])
        for sp in ax_e.spines.values():
            sp.set_visible(False)
        last_err_bottom = err_bottom

        acc = float((c["oof"] == c["y"]).mean())
        name_str = DISPLAY_NAMES[p].replace(" (", chr(10) + "(")
        fig.text(left, bottom + ax_h / 2 + 0.006, name_str,
                 ha="left", va="bottom")                 # name: normal weight
        fig.text(left, bottom + ax_h / 2 - 0.006, f"acc {acc*100:.1f}%",
                 ha="left", va="top", fontweight="bold")  # acc: bold

    # one shared full-width time axis (P01 scale, 50 s steps), like Panel B
    ax_t = fig.add_axes([left + label_w, last_err_bottom - 0.02, usable, 0.001])
    ax_t.set_xlim(0, max_dur)
    ax_t.set_yticks([])
    for sp in ax_t.spines.values():
        sp.set_visible(False)
    xticks = np.arange(0, max_dur + 1, 50)
    ax_t.set_xticks(xticks)
    ax_t.set_xticklabels([f"{v:.0f}" for v in xticks])
    ax_t.set_xlabel("Time (s)")

    # legend on top: gesture colors + misprediction
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=_gesture_signal_color(g), lw=2.2,
                      label=_gesture_label(g)) for g in GESTURES]
    handles.append(Line2D([0], [0], color=COL_BAD, lw=2.2,
                          label="misprediction"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    _save(fig, out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def _gesture_legend(fig) -> None:
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=GESTURE_DISPLAY_COLORS[g], alpha=GESTURE_BG_ALPHA,
                     label=_gesture_label(g)) for g in GESTURES]
    fig.legend(handles=handles, loc="lower center", ncol=len(GESTURES),
               frameon=False, bbox_to_anchor=(0.5, 0.02))


def _gesture_line_legend(fig, y: float = 1.0) -> None:
    """Solid-line legend matching the colored signals in Panel B, placed above
    the plots by default."""
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=_gesture_signal_color(g), lw=2.2,
                      label=_gesture_label(g)) for g in GESTURES]
    fig.legend(handles=handles, loc="upper center", ncol=len(GESTURES),
               frameon=False, bbox_to_anchor=(0.5, y))


PANEL_B_ROWS_1IDX = [10, 31, 52]   # fixed kept-channel rows shown in panel B


def combined_figure(out_path: str, n_splits: int = 5) -> None:
    """All four panels on a single figure (no A/B/C/D letters — added later in
    Illustrator). Panels B, C and D share one time axis, shown only under D;
    one gesture legend (above B) applies to B, C and D."""
    import json
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    label_to_idx = {g: i for i, g in enumerate(GESTURES)}

    # ---- RMS data + per-gesture-block CV (panels C & D) --------------------
    rms = {}
    for p in PARTICIPANTS:
        data = load_rms_dataset(RMS_DATASET_PATHS[p])
        kmask = keep_mask_from_bad_spec(data["kept_channels_1idx"],
                                        data["bad_channels"])
        feats = data["features"][kmask]
        fmask = data["region_ids_per_frame"] >= 0
        F = feats[:, fmask]
        y = data["labels"][fmask].astype(np.int64)
        region_ids = data["region_ids_per_frame"][fmask]
        print(f"  [combined] {p}: {F.shape[1]} frames — {n_splits}-fold CV")
        _, oof, _ = cross_validate(F.T.astype(np.float32), y, region_ids,
                                   n_splits=n_splits, group_by=y)
        valid = oof >= 0
        r = float(data["feature_rate_hz"])
        rms[p] = {"rate": r, "env": F[:, valid].mean(axis=0), "y": y[valid],
                  "oof": oof[valid], "n": int(valid.sum()),
                  "dur": int(valid.sum()) / r}
    dur = {p: rms[p]["dur"] for p in PARTICIPANTS}
    MAX = max(dur.values())

    # panel C uses P01's RMS envelope + per-gesture split
    cy = rms["P01"]["y"]
    c_env = rms["P01"]["env"]
    c_rate = rms["P01"]["rate"]
    c_n = rms["P01"]["n"]
    c_t = np.arange(c_n) / c_rate
    c_splits = per_gesture_kfold_splits(cy, n_splits=n_splits)
    c_bounds = np.where(np.diff(cy) != 0)[0] + 1

    # ---- panel B: 3 fixed channels, band-pass + notch, gesture order -------
    b_rows0 = [i - 1 for i in PANEL_B_ROWS_1IDX]
    bdat = {}
    for p in PARTICIPANTS:
        c = _load_concat(p)
        fs = c["fs"]
        kmask = keep_mask_from_bad_spec(c["original_channels_1idx"],
                                        c["bad_channels"])
        gf = apply_filters(c["signal"][kmask].astype(np.float32), fs,
                           PANEL_B_BANDPASS_HZ, PANEL_B_NOTCH_HZ,
                           PANEL_B_BUTTER_ORDER)
        order = sorted((k for k in range(len(c["region_labels"]))
                        if c["region_labels"][k] in label_to_idx),
                       key=lambda k: label_to_idx[c["region_labels"][k]])
        seg_sig, seg_lab = [], []
        for k in order:
            s, e = int(c["region_starts"][k]), int(c["region_ends"][k])
            seg_sig.append(gf[:, s:e])
            seg_lab.append(np.full(e - s, c["region_labels"][k]))
        disp = np.concatenate(seg_sig, axis=1)
        labps = np.concatenate(seg_lab)
        nB = disp.shape[1]
        ds = max(1, nB // 4000)
        rows = [i for i in b_rows0 if i < disp.shape[0]]
        sub = disp[rows][:, ::ds]
        bdat[p] = {"t": np.linspace(0, dur[p], sub.shape[1]),
                   "traces": sub - sub.mean(axis=1, keepdims=True),
                   "lab": labps[::ds]}

    # ---- panel A: example power-grasp recording (unchanged content) --------
    # Panel A reads one raw recording from the lab NAS. It is not distributed
    # with two-e — request it from the authors (or mount the NAS). If the NAS
    # is unreachable, fall back to a path that fails the guard below with the
    # explanatory message.
    try:
        base = get_server_paths()["nsquared-nas"]["datasets"]
        panelA_pkl = os.path.join(base, NAS_DIRS[PANEL_A_PARTICIPANT],
                                  PANEL_A_FILE)
    except Exception:
        base = OUTPUT_DIR
        panelA_pkl = os.path.join(OUTPUT_DIR, PANEL_A_FILE)
    require_raw_signal(panelA_pkl)
    with open(panelA_pkl, "rb") as f:
        dA = pickle.load(f)
    fsA = int(dA["device_information"]["sampling_frequency"])
    sigA = flatten_biosignal(dA["biosignal"])
    keptA = sorted(parse_channel_spec(PANEL_A_CHANNEL_SPEC, N_BIOSIGNAL_CHANNELS))
    chA = np.array([g + 1 for g in keptA])
    sigA = sigA[np.array(keptA)]
    keepA = keep_mask_from_bad_spec(
        chA, _load_concat(PANEL_A_PARTICIPANT)["bad_channels"])
    badA = np.where(~keepA)[0]
    nchA, nA = sigA.shape
    labsA = json.load(open(os.path.join(base, NAS_DIRS[PANEL_A_PARTICIPANT],
                                        "labels.json"), encoding="utf-8"))
    entryA = next(e for e in labsA if e["file"] == PANEL_A_FILE)
    labA = np.zeros(nA, dtype=bool)
    for sp in entryA["spans"]:
        if normalize_label(sp["label"]) == "Power Grasp":
            labA[max(0, int(sp["start"] * fsA)):
                 min(nA, int(sp["end"] * fsA))] = True
    dsA = max(1, nA // 6000)
    tA = np.arange(nA)[::dsA] / fsA
    labA_ds = labA[::dsA]
    cenA = sigA[:, ::dsA] - sigA[:, ::dsA].mean(axis=1, keepdims=True)
    sprA = float(np.median(np.std(cenA, axis=1))) * PANEL_A_SPREAD_K or 1.0
    yA = cenA + np.arange(nchA)[:, None] * sprA

    # ---- layout (generous gaps between the panel blocks) -------------------
    H = 8.0
    fig = plt.figure(figsize=(FIG_W_IN, H))
    PL, PW, LABX = 0.14, 0.82, 0.012

    def width(p):
        return PW * dur[p] / MAX

    # Panel A
    axA = fig.add_axes([PL, 0.8375, PW, 0.125])
    goodA = np.where(keepA)[0]
    for segs, col, w in (
        ([np.column_stack([tA, np.where(labA_ds, np.nan, yA[i])]) for i in goodA],
         COL_UNLAB_SIGNAL, 0.35),
        ([np.column_stack([tA, np.where(labA_ds, yA[i], np.nan)]) for i in goodA],
         GESTURE_DISPLAY_COLORS["Power Grasp"], 0.35),
        ([np.column_stack([tA, yA[i]]) for i in badA], COL_BAD, 0.7),
    ):
        if segs:
            lc = LineCollection(segs, colors=col, linewidths=w)
            lc.set_rasterized(True)
            axA.add_collection(lc)
    axA.set_xlim(tA[0], tA[-1])
    axA.set_ylim(-sprA, nchA * sprA)
    ltA = np.linspace(0, nchA - 1, 5, dtype=int)
    axA.set_yticks(ltA * sprA)
    axA.set_yticklabels([str(chA[i]) for i in ltA])
    axA.set_xlabel("Time (s)")
    axA.set_ylabel(f"EMG channel ({PANEL_A_GRID_LABEL})")
    for sp in axA.spines.values():
        sp.set_visible(False)
    if len(badA):
        axAr = axA.twinx()
        axAr.set_ylim(axA.get_ylim())
        axAr.set_yticks(badA * sprA)
        axAr.set_yticklabels([str(chA[i]) for i in badA])
        axAr.tick_params(axis="y", colors=COL_BAD)
        for sp in axAr.spines.values():
            sp.set_visible(False)
    axA.legend(handles=[
        Line2D([0], [0], color=GESTURE_DISPLAY_COLORS["Power Grasp"], lw=1.8,
               label="extracted area"),
        Line2D([0], [0], color=COL_BAD, lw=1.8, label="discarded channel")],
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
        handlelength=1.3)

    # shared gesture legend (applies to B, C, D)
    fig.legend(handles=[Line2D([0], [0], color=_gesture_signal_color(g), lw=2.2,
                               label=_gesture_label(g)) for g in GESTURES],
               loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.735))

    # Panel B (3 channels, no x-axis)
    b_top, b_stride, b_h = 0.701, 0.0375, 0.030
    for r, p in enumerate(PARTICIPANTS):
        bt = b_top - r * b_stride - b_h
        ax = fig.add_axes([PL, bt, width(p), b_h])
        tr = bdat[p]["traces"]
        spread = float(np.median(np.std(tr, axis=1))) * 4 or 1.0
        _draw_colored_traces(ax, tr, bdat[p]["t"], spread, bdat[p]["lab"],
                             lw=0.3)
        ax.set_xlim(0, dur[p])
        ax.set_ylim(-spread, len(tr) * spread)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.text(LABX, bt + b_h / 2, DISPLAY_NAMES[p].replace(" (", "\n("),
                 ha="left", va="center")

    # Panel C (envelope + fold strips, no x-axis)
    axc0 = fig.add_axes([PL, 0.474, PW, 0.0525])
    for i, g in enumerate(GESTURES):
        m = cy == i
        if m.any():
            axc0.plot(c_t, np.where(m, c_env, np.nan),
                      color=_gesture_signal_color(g), linewidth=0.7)
    for b in c_bounds:
        axc0.axvline(c_t[b], color="#9A9A9A", linewidth=0.7, zorder=3)
    axc0.set_xlim(0, MAX)
    axc0.set_ylabel("RMS (mean)\n[µV]")
    axc0.yaxis.set_label_coords(-0.075, 0.5)
    axc0.set_xticks([])
    for sp in axc0.spines.values():
        sp.set_visible(False)
    train_h = Patch(facecolor=COL_TRAIN, edgecolor="none")
    val_h = tuple(Patch(facecolor=c, edgecolor="none") for c in FOLD_COLORS)
    fig.legend([train_h, val_h], ["training", "validation"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
               loc="upper right", bbox_to_anchor=(0.98, 0.472), ncol=2,
               frameon=False, handlelength=2.8, handletextpad=0.5,
               columnspacing=1.4)
    cs_top, cslot, cbar = 0.432, 0.01875, 0.0112
    for k, (_, te_idx) in enumerate(c_splits):
        bt = cs_top - (k + 1) * cslot
        ax = fig.add_axes([PL, bt, PW, cbar])
        val = np.zeros(c_n, dtype=bool)
        val[te_idx] = True
        ax.axhspan(0, 1, color=COL_TRAIN, lw=0)
        for s, e in _contiguous_runs(val):
            ax.axvspan(c_t[s], c_t[min(e, c_n - 1)], color=FOLD_COLORS[k], lw=0)
        for b in c_bounds:
            ax.axvline(c_t[b], color="#9A9A9A", linewidth=0.7, zorder=3)
        ax.set_xlim(0, MAX)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([0.5])
        ax.set_yticklabels([f"Fold {k + 1}"])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Panel D (mean RMS over all channels + misprediction strip, shared x-axis)
    d_top, d_stride, d_h, d_eg, d_eh = 0.295, 0.068, 0.025, 0.007, 0.014
    last_eb = 0.0
    for r, p in enumerate(PARTICIPANTS):
        c = rms[p]
        n, rt = c["n"], c["rate"]
        tD = np.arange(n) / rt
        ds = max(1, n // 4000)
        env, yv, oofv = c["env"], c["y"], c["oof"]
        bt = d_top - r * d_stride - d_h
        ax = fig.add_axes([PL, bt, width(p), d_h])
        for i, g in enumerate(GESTURES):
            m = yv == i
            if m.any():
                ax.plot(tD[::ds], np.where(m[::ds], env[::ds], np.nan),
                        color=_gesture_signal_color(g), linewidth=0.6)
        ax.set_xlim(0, dur[p])
        ax.set_ylim(float(env.min()), float(env.max()))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        err = oofv != yv
        eb = bt - d_eg - d_eh
        axe = fig.add_axes([PL, eb, width(p), d_eh])
        xerr = tD[err]
        if xerr.size:
            axe.vlines(xerr, 0, 1, color=COL_BAD, linewidth=0.6)
        axe.set_xlim(0, dur[p])
        axe.set_ylim(0, 1)
        axe.set_xticks([])
        axe.set_yticks([])
        for sp in axe.spines.values():
            sp.set_visible(False)
        last_eb = eb
        acc = float((oofv == yv).mean())
        fig.text(LABX, bt + d_h / 2 + 0.004,
                 DISPLAY_NAMES[p].replace(" (", "\n("), ha="left", va="bottom")
        fig.text(LABX, bt + d_h / 2 - 0.004, f"acc {acc*100:.1f}%",
                 ha="left", va="top", fontweight="bold")

    # shared time axis (only under D), 50 s steps
    axt = fig.add_axes([PL, last_eb - 0.02, PW, 0.001])
    axt.set_xlim(0, MAX)
    axt.set_yticks([])
    for sp in axt.spines.values():
        sp.set_visible(False)
    xticks = np.arange(0, MAX + 1, 50)
    axt.set_xticks(xticks)
    axt.set_xticklabels([f"{v:.0f}" for v in xticks])
    axt.set_xlabel("Time (s)")

    # misprediction note for panel D (red), placed at the right of its band
    fig.legend(handles=[Line2D([0], [0], color=COL_BAD, lw=2.2,
                               label="misprediction")],
               loc="upper right", bbox_to_anchor=(0.98, d_top + 0.03),
               frameon=False, handlelength=1.3)

    _save(fig, out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    combined_figure(os.path.join(OUT_DIR, "supplementary_combined.svg"))
    print(f"\nCombined figure saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
