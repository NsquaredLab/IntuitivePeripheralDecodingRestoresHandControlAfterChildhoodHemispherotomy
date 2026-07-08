"""
figure3BC.py
============

Figure 3B shows, for a single exergame recording:

    (left)   the RMS envelope of the surface-EMG signal (dark blue),
             the ground-truth cue windows the participant was asked to react to
             (shaded red), the binary myocontrol prediction (black, right axis);
    (right)  a box showing the mean reaction latency +/- one standard deviation
             across all cues.

Figure 3C shows the **event-based confusion matrix** for a single exergame
recording.

    - a *burst* is a contiguous run of active predictions in the
      (frame-downsampled) prediction stream;
    - bursts shorter than ``min_burst_seconds`` are treated as noise and ignored;
    - **TP** = number of ground-truth cue windows overlapped by at least one
      valid burst (correctly detected activations);
    - **FN** = number of ground-truth cue windows with no overlapping valid
      burst (missed activations);
    - **FP** = number of valid bursts that do not overlap any ground-truth
      window (false alarms);
    - **TN** is undefined for event-based metrics (there is no natural notion of
      a "true negative event"), so the corresponding cell is left blank / N/A.

    From these counts the event-based precision, recall and F1 are derived and the
    2x2 matrix is drawn as a heatmap (top row = ground-truth Active: TP | FN;
    bottom row = ground-truth Rest: FP | N/A).

Dependencies
------------
    numpy, scipy, matplotlib          (pip install numpy scipy matplotlib)

Usage
-----
    python figure3BC.py
    edit the ``RECORDING_PATH`` and ``GROUND_TRUTH`` constants at the bottom of the file.

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Experiment / hardware constants
# ---------------------------------------------------------------------------
# FS      : EMG sampling frequency of the recording device in Hz (samples/s).
# FRAME   : Decimation factor of the real-time control loop. The myocontrol
#           pipeline produces one prediction / rms value per FRAME EMG samples,
#           i.e. the prediction stream runs at FS / FRAME Hz. FRAME is needed to
#           convert prediction indices back to seconds (and cue times into
#           prediction-frame indices).
FS = 2048       # [Hz]  EMG sampling frequency
FRAME = 18      # [-]   EMG samples per control-loop frame (prediction cadence)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def get_pkl_data(path):
    """
    Load one exergame recording stored as a pickled dictionary.

    Parameters
    ----------
    path : str
        Path to the ``myocontrol_data*.pkl`` file written by the exergame
        software.

    Returns
    -------
    rms : ndarray, shape (n_frames,)
        RMS envelope of the EMG signal, one value per control-loop frame.
        This is the signal that is plotted on the left y-axis of Figure 3B.
    predictions : ndarray, shape (n_frames,)
        Binary control signal (0 = rest, 1 = active) produced by the
        myocontrol classifier, one value per frame.
    prediction_threshold : float
        Dimensionless scaling factor applied to ``rms_reference`` to obtain the
        activation threshold (empirically ~2x the mean resting activation).
    rms_reference : float
        Reference RMS level (baseline) measured during a calibration phase.
        The absolute activation threshold shown in the plot is
        ``rms_reference * prediction_threshold``.
    """
    with open(path, "rb") as f:
        recording = pickle.load(f)

    rms = recording["rms"]                                   # RMS envelope [frames]
    predictions = recording["predictions"]                  # binary control signal
    prediction_threshold = recording["prediction_threshold"]  # threshold scaling factor
    rms_reference = recording["rms_reference"]              # baseline RMS reference

    return rms, predictions, prediction_threshold, rms_reference


# ---------------------------------------------------------------------------
# 2. Signal helpers
# ---------------------------------------------------------------------------
def upsample(reference, var):
    """
    Linearly resample ``var`` so that it has the same number of samples as
    ``reference``. Necessary to align the decimated myocontrol prediction with the RMS envelope.

    Parameters
    ----------
    reference : array_like
        Signal whose length defines the target number of samples.
    var : array_like
        Signal to be resampled.

    Returns
    -------
    ndarray, shape (len(reference),)
        ``var`` resampled to the length of ``reference``.
    """
    var = np.asarray(var, dtype=float)
    reference = np.asarray(reference)

    x_old = np.linspace(0, 1, len(var))         # normalised original axis
    x_new = np.linspace(0, 1, len(reference))   # normalised target axis
    interpolator = interp1d(x_old, var, kind="linear")
    return interpolator(x_new)


def find_onsets(prediction):
    """
    Return the frame indices at which the binary ``prediction`` rises from 0
    to 1 (i.e. the moments the classifier switched from rest to active).

    Parameters
    ----------
    prediction : array_like
        Binary prediction signal (0/1) at control cadence.

    Returns
    -------
    ndarray
        Indices (in frames) of every rising edge.
    """
    prediction = np.asarray(prediction)
    # np.diff with prepend=0 yields +1 exactly at a 0->1 transition, also at index 0.
    diff = np.diff(prediction, prepend=0)
    onsets = np.where(diff == 1)[0]
    return onsets


# ---------------------------------------------------------------------------
# 3. Latency computation (Figure 3B)
# ---------------------------------------------------------------------------
def compute_latency(ground_truth, prediction, fs, frame):
    """
    Compute, for every cue window, the reaction latency between the cue onset
    and the first prediction onset that follows it.

    Parameters
    ----------
    ground_truth : list of (float, float)
        Cue windows ``(start, end)`` in **seconds**. Each tuple marks the time
        span during which the participant was instructed to activate.
    prediction : array_like
        Binary prediction signal (0/1) at control cadence.
    fs : float
        EMG sampling frequency in Hz (see ``FS``).
    frame : int
        EMG samples per control frame (see ``FRAME``).

    Returns
    -------
    list of float
        One latency per correctly performed cue, in **seconds**.

    Notes
    -----
    * ``f = frame / fs`` is the duration of one control frame in seconds and is
      used to convert between frame indices and seconds.
    * A latency is only accepted if it is <= 2 s. Larger values are assumed to
      correspond to a missed/incorrectly performed activation and are discarded,
      so the reported latencies reflect genuine reactions only.
    """
    latencies = []
    onset_pred = find_onsets(prediction)
    f = frame / fs  # seconds per control frame

    for (start, end) in ground_truth:
        gt_start = int(start / f)  # cue onset expressed in frame indices
        for i in range(len(onset_pred)):
            if onset_pred[i] >= gt_start:
                latency = onset_pred[i] - gt_start          # latency in frames
                if latency * f <= 2:                        # keep only plausible reactions
                    latencies.append(latency * f)           # store in seconds
                break

    return latencies


# ---------------------------------------------------------------------------
# 4. Event-based confusion matrix (core computation, Figure 3C)
# ---------------------------------------------------------------------------
def compute_confusion_matrix_event(ground_truth, prediction, fs, frame,
                                   min_burst_seconds=0.5):
    """
    Compute the event-based confusion-matrix counts and metrics for a single
    exergame recording.

    Parameters
    ----------
    ground_truth : list of (float, float)
        Cue windows ``(start, end)`` in **seconds**. Each tuple marks the time
        span during which the participant was instructed to activate.
    prediction : array_like
        Binary prediction signal (0/1) at control cadence (one value per frame).
    fs : float
        EMG sampling frequency in Hz (see ``FS``).
    frame : int
        EMG samples per control frame (see ``FRAME``).
    min_burst_seconds : float, optional
        Minimum duration (in seconds) for a contiguous active run ("burst") to
        count as a genuine activation. Shorter bursts are treated as noise and
        ignored (default 0.5 s).

    Returns
    -------
    counts : dict
        ``{"tp": int, "fn": int, "fp": int}`` -- the event-based confusion
        counts. TN is undefined for event-based metrics and is not returned.
    metrics : dict
        ``{"precision": float, "recall": float, "f1": float}``.
    detected : list of bool
        Per-cue detection flag (``True`` if the cue was detected by a valid
        burst), one entry per ground-truth window.

    Notes
    -----
    * A *burst* is a maximal contiguous run of active (``True``) frames in the
      prediction stream.
    * A ground-truth window counts as detected (**TP**) if any valid burst
      overlaps it; otherwise it is a miss (**FN**).
    * A valid burst that overlaps no ground-truth window is a false alarm (**FP**).
    """
    prediction = np.asarray(prediction).astype(bool)
    max_frame = len(prediction)

    # Convert ground truth from seconds to frame indices.
    gt_frames = [(int(start * fs / frame), int(end * fs / frame))
                 for (start, end) in ground_truth]

    # Minimum burst length expressed in (frame-downsampled) prediction samples.
    min_burst_samples = round(min_burst_seconds * fs / frame)

    # Ground-truth activation vector (used for the burst-overlap / false-alarm check).
    gt_vector = np.zeros(max_frame, dtype=bool)
    for (start, end) in gt_frames:
        if start < max_frame:
            gt_vector[start:min(end, max_frame)] = True

    # Find all bursts (maximal contiguous runs of active frames).
    bursts = []
    i = 0
    while i < max_frame:
        if prediction[i]:
            j = i
            while j < max_frame and prediction[j]:
                j += 1
            bursts.append((i, j))
            i = j
        else:
            i += 1

    # Keep only bursts long enough to count as a genuine activation.
    valid_bursts = [(s, e) for (s, e) in bursts if (e - s) >= min_burst_samples]
    n_short = len(bursts) - len(valid_bursts)

    # TP / FN: a GT window is detected if any valid burst overlaps it.
    detected = []
    for (start, end) in gt_frames:
        end_clipped = min(end, max_frame)
        if start >= max_frame:
            detected.append(False)
            continue
        is_detected = any(bs < end_clipped and be > start
                          for (bs, be) in valid_bursts)
        detected.append(is_detected)
    tp = int(sum(detected))
    fn = int(len(detected) - tp)

    # FP: valid bursts that don't overlap any ground-truth window (false alarms).
    fp = int(sum(1 for (bs, be) in valid_bursts if not np.any(gt_vector[bs:be])))

    # Precision / Recall / F1 with safe division.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    print(f"Event-based: TP={tp}/{len(gt_frames)} GT detected, FN={fn}, "
          f"FP={fp} (false-alarm bursts of {len(valid_bursts)} valid; "
          f"{n_short} short bursts ignored)")
    print(f"  Per-GT detection: {detected}")

    counts = {"tp": tp, "fn": fn, "fp": fp}
    metrics = {"precision": precision, "recall": recall, "f1": f1}
    return counts, metrics, detected


# ---------------------------------------------------------------------------
# 5. Figure 3B
# ---------------------------------------------------------------------------
def plot_exergame(rms, ground_truth, prediction, fs, latencies, threshold,
                  frame=FRAME, cutX=None, savepath=None):
    """
    Draw Figure 3B.

    Parameters
    ----------
    rms : ndarray, shape (n_frames,)
        RMS envelope of the EMG (left y-axis, converted to microvolts for display).
    ground_truth : list of (float, float)
        Cue windows ``(start, end)`` in seconds, drawn as shaded areas.
    prediction : array_like
        Binary prediction signal (0/1) at control cadence; resampled onto the
        RMS length and drawn on the right y-axis.
    fs : float
        EMG sampling frequency in Hz.
    latencies : list of float or None
        Per-cue reaction latencies in seconds (from :func:`compute_latency`).
        If not ``None`` a box panel (mean +/- SD) is added to the right.
    threshold : float or ndarray or None
        Activation threshold in the same units as ``rms``
        (typically ``rms_reference * prediction_threshold``). Drawn as a dashed
        line. If ``None`` no threshold is plotted.
    frame : int, optional
        EMG samples per control frame, used to align the prediction with the RMS
        time axis (default ``FRAME``).


    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    # ---- Journal-style typography -------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 22
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 22

    # ---- Time axis and prediction alignment ---------------------------------------
    # time : seconds for every RMS sample (RMS defines the plot's time base).
    time = np.arange(len(rms)) / fs
    # Resample the (decimated) prediction onto the RMS length so both share one x-axis.
    prediction = upsample(rms, prediction)


    # ---- Figure layout: main panel + latency box side by side ---------------------
    if latencies is not None:
        fig, (ax1, ax3) = plt.subplots(
            1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 0.25]})
        ax2 = ax1.twinx()
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

    # ---- Top panel: RMS envelope (left axis) --------------------------------------
    # rms * 1000 converts millivolts to microvolts for the display.
    ax1.plot(time, rms * 1000, label="RMS (EMG)",
             color="#001C55", alpha=0.7, linewidth=2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("RMS Amplitude (µV)", color="#001C55")
    ax1.tick_params(axis="y", labelcolor="#001C55", width=2, length=6)
    ax1.tick_params(axis="x", width=2, length=6)


    ax1.set_xticks(np.arange(0, len(time) / fs, 4))

    # Shade the ground-truth cue windows.
    for (start, end) in ground_truth:
        ax1.axvspan(start, end, color="#FC7C86", alpha=0.3,
                    label="Cue" if (start, end) == ground_truth[0] else "")

    # ---- Top panel: binary prediction (right axis) --------------------------------
    ax2.plot(time, prediction, label="Prediction",
             color="black", alpha=0.8, linewidth=2)
    ax2.set_ylabel("Prediction", color="black")
    ax2.tick_params(axis="y", labelcolor="black", width=2, length=6)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_yticks([0, 1])

    # ---- Combined legend for both y-axes ------------------------------------------
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc="upper right", fontsize=14)
    ax1.set_title("Exergame: EMG with Ground Truth and Predictions",
                  fontsize=20, pad=20)

    # ---- Right panel: latency box (mean +/- SD), toolbox style --------------------
    if latencies is not None:
        # mean_lat : average reaction latency across all cues [s]
        # std_lat  : standard deviation of the reaction latency [s]
        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)

        # Shaded rectangle spanning mean +/-  SD
        ax3.add_patch(plt.Rectangle((-0.3, mean_lat - std_lat), 0.6, 2 * std_lat,
                                    facecolor="#FC7C86", alpha=0.5,
                                    edgecolor="black", label="SD"))
        # Horizontal line at the mean.
        ax3.hlines(mean_lat, -0.3, 0.3, colors="#001C55", linewidth=2,
                   label=f"Mean: {mean_lat:.3f}s")

        ax3.set_xlim(-0.6, 0.6)
        ax3.set_xticks([0])
        ax3.set_ylabel("Latency (s)")
        ax3.set_ylim(0, mean_lat + std_lat + 0.5)
        ax3.set_title("Mean ± SD", fontsize=20, pad=20)
        ax3.grid(True, alpha=0.7, axis="y")
        ax3.tick_params(axis="both", which="major", width=2, length=6)
        ax3.legend(loc="upper right", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)  # restore global style
    return fig


# ---------------------------------------------------------------------------
# 6. Figure 3C
# ---------------------------------------------------------------------------
def plot_confusion_matrix_event(counts, n_gt, title=None):
    """
    Draw Figure 3C: the event-based confusion matrix as a 2x2 heatmap.

    Layout (rows = ground truth, columns = prediction):

                    Pred Active     Pred Rest
        GT Active       TP              FN
        GT Rest         FP             N/A

    The bottom-right cell (event-based "true negative") is left blank and
    labelled "N/A" because it is undefined for event-based metrics.

    Parameters
    ----------
    counts : dict
        ``{"tp": int, "fn": int, "fp": int}`` from
        :func:`compute_confusion_matrix_event`.
    n_gt : int
        Number of ground-truth cue windows. Used to scale the colour axis so
        that a fully saturated cell corresponds to "all cues" (vmax = n_gt).
    title : str or None, optional
        Optional title; " (Event-based)" is appended.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    tp, fn, fp = counts["tp"], counts["fn"], counts["fp"]

    # 2x2 matrix; the bottom-right (TN) cell is undefined -> masked.
    cm = np.array([[tp, fn],
                   [fp, np.nan]], dtype=float)

    # ---- Journal-style typography -------------------------------------------------
    scale = 1.2
    size = 32 * scale
    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots(figsize=(6 * scale, 5 * scale))

    # Heatmap via imshow (masked array hides the N/A cell). Blues colormap,
    # colour axis 0..n_gt so the shading is comparable across recordings.
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="white")
    masked = np.ma.masked_invalid(cm)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=n_gt)

    # Annotate every defined cell with its integer count; label the N/A cell.
    for (r, c), value in np.ndenumerate(cm):
        if np.isnan(value):
            ax.text(c, r, "N/A", ha="center", va="center",
                    fontsize=size, color="gray")
        else:
            # White text on dark cells, dark text on light cells for contrast.
            txt_color = "white" if value > 0.6 * n_gt else "black"
            ax.text(c, r, f"{int(value)}", ha="center", va="center",
                    fontsize=size, color=txt_color)

    # Axis ticks / labels (sklearn-style: rows = GT, columns = prediction).
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Active", "Rest"], fontsize=size - 4)
    ax.set_yticklabels(["Active", "Rest"], fontsize=size - 4)
    ax.set_xlabel("Prediction", fontsize=size)
    ax.set_ylabel("Ground Truth", fontsize=size)

    full_title = "Event-based" if title is None else f"{title} (Event-based)"
    ax.set_title(full_title, fontsize=size)

    # Colour bar with integer ticks 0..n_gt.
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks(np.arange(0, n_gt + 1))
    cbar.ax.tick_params(labelsize=size - 6)

    plt.tight_layout()

    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)  # restore global style
    return fig


# ---------------------------------------------------------------------------
# 7. Reproduce Figure 3B
# ---------------------------------------------------------------------------
def make_figure3B(recording_path, ground_truth):
    """
    Parameters
    ----------
    recording_path : str
        Path to the ``myocontrol_data*.pkl`` recording.
    ground_truth : list of (float, float)
        Cue windows ``(start, end)`` in seconds, extracted from the video/protocol.
    """
    # 1. Load the signals required for the figure.
    rms, predictions, prediction_threshold, rms_reference = get_pkl_data(recording_path)

    # 2. Absolute activation threshold in RMS units.
    threshold = rms_reference * prediction_threshold

    # 3. Reaction latency for each cue.
    latencies = compute_latency(ground_truth, predictions, FS, FRAME)

    # 4. Draw the figure.
    plot_exergame(rms, ground_truth, predictions, FS, latencies, threshold,
                  frame=FRAME)


# ---------------------------------------------------------------------------
# 8. Reproduce Figure 3C
# ---------------------------------------------------------------------------
def make_figure3C(recording_path, ground_truth, title=None,
                  min_burst_seconds=0.5):
    """
    End-to-end reproduction of Figure 3C from a single recording.

    Parameters
    ----------
    recording_path : str
        Path to the ``myocontrol_data*.pkl`` recording.
    ground_truth : list of (float, float)
        Cue windows ``(start, end)`` in seconds, extracted from the video/protocol.
    title : str or None
        Optional plot title.
    min_burst_seconds : float, optional
        Minimum burst duration in seconds (default 0.5 s).
    """
    # 1. Load the binary prediction stream (only predictions are needed here).
    _, predictions, _, _ = get_pkl_data(recording_path)

    # 2. Event-based confusion counts and metrics.
    counts, metrics, _ = compute_confusion_matrix_event(
        ground_truth, predictions, FS, FRAME,
        min_burst_seconds=min_burst_seconds)
    print(f"Event-based   -> Precision: {metrics['precision']:.3f}, "
          f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")

    # 3. Draw the event-based confusion matrix.
    plot_confusion_matrix_event(counts, n_gt=len(ground_truth),
                                title=title)


if __name__ == "__main__":
    import sys

    RECORDING_PATH = (
        r"<PATH_TO_DATASET>\myocontrol_data.pkl"
    )

    # Ground-truth cue windows (start, end) in seconds, from the recording protocol.
    GROUND_TRUTH = []

    make_figure3B(RECORDING_PATH, GROUND_TRUTH)
    make_figure3C(RECORDING_PATH, GROUND_TRUTH, title=None)
