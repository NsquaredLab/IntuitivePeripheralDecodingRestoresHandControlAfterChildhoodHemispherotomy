"""
02_dataset_creation.py
======================

Step 2 of the Figure 2E gesture-classification pipeline: build one labelled
raw-EMG dataset per participant.

Walks the ``labels.json`` produced by step 1, loads every kept recording,
slices it to the requested channels, cuts out each labelled gesture span and
concatenates the spans into a single ``(channels x samples)`` signal. An
interactive window then lets the user mark bad channels on the concatenated
signal. The result is written to ``data/<P>.npz`` together with the per-sample
gesture labels and region bookkeeping, and the bad-channel spec to
``data/<P>_bad_channels.json``.

Input / Output
--------------
    in : raw ``*.pkl`` recordings + ``labels.json`` on the lab NAS
    out: ``data/<P>.npz``               (raw labelled signal)
         ``data/<P>_bad_channels.json`` (bad-channel spec)

The raw recordings and the resulting ``data/<P>.npz`` contain identifiable
human EMG and are not distributed with this repository; they are available from
the authors on reasonable request. This step
therefore only runs with access to the raw data.

Dependencies
------------
    numpy, matplotlib, pysynclient    (pip install numpy matplotlib pysynclient)

Usage
-----
    uv run python figure-two/two-e/02_dataset_creation.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import json
import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, TextBox
from pysynclient import get_server_paths

from _pipeline import load_step

_labeler = load_step("labeler")  # -> 01_labeler.py
GESTURE_COLORS = _labeler.GESTURE_COLORS
GESTURES = _labeler.GESTURES
LABELS_FILENAME = _labeler.LABELS_FILENAME
N_BIOSIGNAL_CHANNELS = _labeler.N_BIOSIGNAL_CHANNELS
channels_to_spec = _labeler.channels_to_spec
flatten_biosignal = _labeler.flatten_biosignal
normalize_label = _labeler.normalize_label
parse_channel_spec = _labeler.parse_channel_spec

# Per-participant: (relative dir on the file server, channel-spec for kept EMG
# channels using the global 1-indexed EMG numbering 1..384). The directory paths
# are de-identified here (they encoded participant initials and recording dates);
# the authors substitute the actual paths from a private mapping when running
# these steps. "P01"/"P01_2" are two sessions of the same participant.
PARTICIPANTS = {
    "P01":   (r"<RAW_DATA_ROOT>\P01\session_01", "1-384"),
    "P01_2": (r"<RAW_DATA_ROOT>\P01\session_02", "1-192"),
    "P02":   (r"<RAW_DATA_ROOT>\P02\session_01", "1-192"),
}

# two-e is self-contained: raw + feature datasets live in ./data next to the
# step scripts (original: <repo>/data). Every other step reads OUTPUT_DIR from
# here, and RESULTS_DIR is derived from it, so this single line points the
# whole pipeline at the two-e folder.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def extract_labelled_segments(participant_dir: str, kept_global_zero_idx: list[int]):
    """
    Walk labels.json in `participant_dir`, load each kept .pkl, slice to the
    requested channels, and cut out every labelled span.

    Returns:
        segments: list of (n_kept_channels, n_samples) float32 arrays
        seg_labels: list of gesture names, one per segment
        seg_sources: list of (file_name, span_index) tuples for traceability
        fs: sampling frequency
    """
    labels_path = os.path.join(participant_dir, LABELS_FILENAME)
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"No labels.json in {participant_dir}")
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    kept_idx = np.asarray(sorted(kept_global_zero_idx), dtype=np.int64)
    segments: list[np.ndarray] = []
    seg_labels: list[str] = []
    seg_sources: list[tuple[str, int]] = []
    fs: int | None = None

    for entry in labels:
        if not entry.get("keep", True):
            continue
        spans = entry.get("spans") or []
        if not spans:
            continue
        file_path = os.path.join(participant_dir, entry["file"])
        if not os.path.isfile(file_path):
            print(f"  missing file (skipped): {entry['file']}")
            continue
        with open(file_path, "rb") as f:
            d = pickle.load(f)
        cur_fs = int(d["device_information"]["sampling_frequency"])
        if fs is None:
            fs = cur_fs
        elif cur_fs != fs:
            raise ValueError(
                f"Sampling-frequency mismatch in {entry['file']}: {cur_fs} vs {fs}"
            )
        biosignal = flatten_biosignal(d["biosignal"])[kept_idx]  # (n_kept, n_samples)
        n_samples = biosignal.shape[1]
        for i, span in enumerate(spans):
            label = normalize_label(span.get("label"))
            if label not in GESTURE_COLORS:
                continue
            start = max(0, int(round(span["start"] * fs)))
            end = min(n_samples, int(round(span["end"] * fs)))
            if end <= start:
                continue
            segments.append(biosignal[:, start:end].copy())
            seg_labels.append(label)
            seg_sources.append((entry["file"], i))

    if fs is None:
        fs = 0
    return segments, seg_labels, seg_sources, fs


def concatenate(segments: list[np.ndarray]):
    if not segments:
        return None, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    full = np.concatenate(segments, axis=1)
    lengths = np.array([s.shape[1] for s in segments], dtype=np.int64)
    ends = np.cumsum(lengths)
    starts = np.concatenate(([0], ends[:-1]))
    return full, starts, ends


def per_sample_label_array(seg_labels: list[str], seg_lengths: np.ndarray) -> np.ndarray:
    label_to_idx = {g: i for i, g in enumerate(GESTURES)}
    out = np.empty(int(seg_lengths.sum()), dtype=np.int8)
    cursor = 0
    for label, length in zip(seg_labels, seg_lengths):
        out[cursor:cursor + int(length)] = label_to_idx[label]
        cursor += int(length)
    return out


def plot_and_mark_bad_channels(
    participant: str,
    signal: np.ndarray,
    region_starts: np.ndarray,
    region_ends: np.ndarray,
    region_labels: list[str],
    fs: int,
    original_channels_1idx: np.ndarray,
    prior_bad: str = "",
) -> str:
    """
    Plot the concatenated signal stacked vertically (one trace per kept EMG
    channel) and let the user mark bad channels via a TextBox. Channel numbers
    in the TextBox refer to the GLOBAL 1..384 EMG numbering.

    Returns the bad-channel spec as a string, also in global 1..384 numbering.
    """
    n_channels, n_samples = signal.shape
    t = np.arange(n_samples) / fs
    centered = signal - signal.mean(axis=1, keepdims=True)
    spread = float(np.median(np.std(centered, axis=1))) * 4 or 1.0

    ds = max(1, n_samples // 5000)
    t_disp = t[::ds]
    centered_disp = centered[:, ::ds]

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_axes([0.10, 0.13, 0.87, 0.80])

    # Tinted background per region (gesture color, low alpha)
    for s, e, label in zip(region_starts, region_ends, region_labels):
        ax.axvspan(s / fs, e / fs, color=GESTURE_COLORS[label], alpha=0.12, lw=0)

    segments_lines = [
        np.column_stack([t_disp, centered_disp[c] + c * spread])
        for c in range(n_channels)
    ]
    lc = LineCollection(segments_lines, linewidths=0.4, colors=["black"] * n_channels)
    lc.set_rasterized(True)
    ax.add_collection(lc)

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-spread, n_channels * spread)
    tick_positions = np.arange(0, n_channels, max(1, n_channels // 12))
    ax.set_yticks(tick_positions * spread)
    ax.set_yticklabels([str(original_channels_1idx[i]) for i in tick_positions])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EMG channel (global numbering)")
    ax.set_title(
        f"{participant} — concatenated labelled signal "
        f"({n_channels} ch, {n_samples / fs:.1f} s) — mark bad channels"
    )

    # Legend for gesture colors
    handles = [plt.Rectangle((0, 0), 1, 1, color=GESTURE_COLORS[g], alpha=0.4)
               for g in GESTURES]
    ax.legend(handles, GESTURES, loc="upper right", fontsize=8, framealpha=0.9)

    # Mapping global 0-indexed -> local 0-indexed
    global_to_local = {int(g) - 1: i for i, g in enumerate(original_channels_1idx)}

    def parse_bad_to_local(spec: str) -> set[int]:
        if not spec.strip():
            return set()
        return {
            global_to_local[g]
            for g in parse_channel_spec(spec, N_BIOSIGNAL_CHANNELS)
            if g in global_to_local
        }

    def local_to_global_spec(local_set: set[int]) -> str:
        global_zero = {int(original_channels_1idx[i]) - 1 for i in local_set}
        return channels_to_spec(global_zero, N_BIOSIGNAL_CHANNELS)

    state = {"bad_local": parse_bad_to_local(prior_bad)}

    def apply_colors():
        bad = state["bad_local"]
        lc.set_color(["red" if c in bad else "black" for c in range(n_channels)])
        fig.canvas.draw_idle()

    apply_colors()

    tax = fig.add_axes([0.10, 0.04, 0.65, 0.05])
    text_box = TextBox(tax, "Bad channels  ", initial=prior_bad)

    def update_from_text(text: str, *, log_errors: bool):
        try:
            state["bad_local"] = parse_bad_to_local(text)
        except ValueError:
            if log_errors:
                print(f"  could not parse bad-channel spec: {text!r}")
            return
        apply_colors()

    text_box.on_text_change(lambda text: update_from_text(text, log_errors=False))
    text_box.on_submit(lambda text: update_from_text(text, log_errors=True))

    bax_done = fig.add_axes([0.85, 0.04, 0.10, 0.05])
    btn_done = Button(bax_done, "Done")
    btn_done.on_clicked(lambda _: plt.close(fig))

    plt.show()
    return local_to_global_spec(state["bad_local"])


def main():
    server_path = get_server_paths()["nsquared-nas"]["datasets"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary: dict[str, str] = {}
    for participant, (rel_dir, channel_spec) in PARTICIPANTS.items():
        print(f"\n=== {participant}: channels {channel_spec} ===")
        participant_dir = os.path.join(server_path, rel_dir)

        kept_global_zero = sorted(parse_channel_spec(channel_spec, N_BIOSIGNAL_CHANNELS))
        original_channels_1idx = np.array([g + 1 for g in kept_global_zero])

        segments, seg_labels, seg_sources, fs = extract_labelled_segments(
            participant_dir, kept_global_zero
        )
        if not segments:
            print(f"  no labelled spans (or no kept files), skipping")
            continue

        signal, region_starts, region_ends = concatenate(segments)
        per_sample_labels = per_sample_label_array(seg_labels, region_ends - region_starts)
        print(f"  {len(segments)} segments → ({signal.shape[0]} ch × "
              f"{signal.shape[1]} samples = {signal.shape[1] / fs:.1f}s)")
        gesture_counts = {g: seg_labels.count(g) for g in GESTURES}
        print(f"  gestures per segment count: {gesture_counts}")

        bad_path = os.path.join(OUTPUT_DIR, f"{participant}_bad_channels.json")
        prior_bad = ""
        if os.path.isfile(bad_path):
            with open(bad_path, "r", encoding="utf-8") as f:
                prior_bad = json.load(f).get("bad_channels", "")

        bad_spec = plot_and_mark_bad_channels(
            participant=participant,
            signal=signal,
            region_starts=region_starts,
            region_ends=region_ends,
            region_labels=seg_labels,
            fs=fs,
            original_channels_1idx=original_channels_1idx,
            prior_bad=prior_bad,
        )
        summary[participant] = bad_spec
        print(f"  bad channels: {bad_spec or '(none)'}")

        npz_path = os.path.join(OUTPUT_DIR, f"{participant}.npz")
        np.savez_compressed(
            npz_path,
            signal=signal,
            labels=per_sample_labels,
            fs=fs,
            channel_spec=channel_spec,
            original_channels_1idx=original_channels_1idx,
            bad_channels=bad_spec,
            region_starts=region_starts,
            region_ends=region_ends,
            region_labels=np.array(seg_labels),
            source_files=np.array([s[0] for s in seg_sources]),
            source_span_idx=np.array([s[1] for s in seg_sources]),
        )
        with open(bad_path, "w", encoding="utf-8") as f:
            json.dump({"bad_channels": bad_spec}, f, indent=2)
        print(f"  saved {npz_path}")

    print("\nBad channels per participant:")
    for p, spec in summary.items():
        print(f"  {p}: {spec or '(none)'}")


if __name__ == "__main__":
    main()
