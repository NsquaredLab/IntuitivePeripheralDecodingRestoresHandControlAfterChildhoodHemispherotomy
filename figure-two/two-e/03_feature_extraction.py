"""
03_feature_extraction.py
========================

Step 3 of the Figure 2E gesture-classification pipeline: recompute the RMS
features from the raw labelled signal, following the MyoGestic (Science
Advances) software-side implementation:

    - 2nd-order Butterworth band-pass 20-500 Hz
    - 2nd-order Butterworth band-stop 45-55 Hz (power-line notch)
    - RMS over a 360-sample moving window with 64-sample stride
    - per-region windowing (no window crosses a region boundary, so every
      feature carries a single, unambiguous gesture label)
    - no spatial smoothing kernel

Input / Output
--------------
    in : ``data/<P>.npz``               (raw labelled signal, from step 2)
    out: ``data/<P>_rms_features.npz``  (RMS features, consumed by steps 4-6)

Neither the raw ``data/<P>.npz`` nor the RMS features this step produces are
distributed with this repository, because they derive from identifiable human
EMG of a vulnerable participant group; all data are available from the authors
on reasonable request.

Dependencies
------------
    numpy, scipy                        (pip install numpy scipy)

Usage
-----
    uv run python figure-two/two-e/03_feature_extraction.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt

from _pipeline import load_step, require_raw_signal

OUTPUT_DIR = load_step("dataset_creation").OUTPUT_DIR  # -> 02_dataset_creation.py
_labeler = load_step("labeler")  # -> 01_labeler.py
GESTURES = _labeler.GESTURES
N_BIOSIGNAL_CHANNELS = _labeler.N_BIOSIGNAL_CHANNELS
normalize_label = _labeler.normalize_label
parse_channel_spec = _labeler.parse_channel_spec

WINDOW_SAMPLES = 360
STRIDE_SAMPLES = 64
BANDPASS_HZ = (20.0, 500.0)
BANDSTOP_HZ = (45.0, 55.0)
BUTTER_ORDER = 2


def keep_mask_from_bad_spec(original_channels_1idx: np.ndarray,
                            bad_spec: str) -> np.ndarray:
    if not bad_spec.strip():
        return np.ones(len(original_channels_1idx), dtype=bool)
    bad_global_zero = parse_channel_spec(bad_spec, N_BIOSIGNAL_CHANNELS)
    bad_global_1idx = {g + 1 for g in bad_global_zero}
    return np.array(
        [int(c) not in bad_global_1idx for c in original_channels_1idx],
        dtype=bool,
    )


def apply_filters(signal: np.ndarray, fs: int, bandpass_hz: tuple[float, float],
                  bandstop_hz: tuple[float, float], order: int) -> np.ndarray:
    sos_bp = butter(order, bandpass_hz, btype="bandpass", fs=fs, output="sos")
    out = sosfiltfilt(sos_bp, signal, axis=-1)
    sos_bs = butter(order, bandstop_hz, btype="bandstop", fs=fs, output="sos")
    out = sosfiltfilt(sos_bs, out, axis=-1)
    return out.astype(np.float32)


def rms_moving_window(signal: np.ndarray, window: int,
                      stride: int) -> np.ndarray:
    """signal: (n_channels, n_samples). Returns (n_channels, n_features) where
    n_features = max(0, (n_samples - window) // stride + 1)."""
    if signal.shape[1] < window:
        return np.zeros((signal.shape[0], 0), dtype=np.float32)
    windows = sliding_window_view(signal, window_shape=window, axis=-1)
    windows = windows[:, ::stride]  # (n_channels, n_features, window)
    return np.sqrt((windows ** 2).mean(axis=-1)).astype(np.float32)


def recompute_for_participant(participant: str) -> None:
    in_path = os.path.join(OUTPUT_DIR, f"{participant}.npz")
    out_path = os.path.join(OUTPUT_DIR, f"{participant}_rms_features.npz")
    print(f"\n{participant}: {in_path}")
    require_raw_signal(in_path)  # raw EMG: request from authors if absent
    z = np.load(in_path, allow_pickle=False)
    signal = z["signal"]
    region_starts_s = z["region_starts"].astype(np.int64)
    region_ends_s = z["region_ends"].astype(np.int64)
    region_labels = z["region_labels"].astype(str)
    fs = int(z["fs"])
    channel_spec = str(z["channel_spec"])
    bad_spec = str(z["bad_channels"])
    original_channels_1idx = z["original_channels_1idx"].astype(np.int32)
    source_files_in = z["source_files"]
    source_span_idx_in = z["source_span_idx"]

    bad_mask = keep_mask_from_bad_spec(original_channels_1idx, bad_spec)
    signal = signal[bad_mask].astype(np.float32)
    kept_channels_1idx = original_channels_1idx[bad_mask]
    print(f"  channels: {bad_mask.sum()}/{len(bad_mask)} kept")

    print(f"  filtering: order {BUTTER_ORDER} Butterworth — "
          f"BP {BANDPASS_HZ[0]}-{BANDPASS_HZ[1]} Hz, "
          f"BS {BANDSTOP_HZ[0]}-{BANDSTOP_HZ[1]} Hz...")
    signal_f = apply_filters(signal, fs, BANDPASS_HZ, BANDSTOP_HZ,
                             BUTTER_ORDER)

    print(f"  RMS: {WINDOW_SAMPLES}-sample moving window, "
          f"{STRIDE_SAMPLES}-sample stride per region...")
    label_to_idx = {g: i for i, g in enumerate(GESTURES)}
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    new_starts, new_ends = [], []
    new_labels, new_files, new_span_idx = [], [], []
    cursor = 0
    # Emit regions grouped by the canonical GESTURES order (Rest, Power
    # Grasp, Pinch, Tripod Pinch) instead of recording order, so every
    # participant's dataset has the same deterministic class layout. The
    # sort is stable, so regions keep their chronological order within a
    # class. Unknown/legacy labels are normalized first and dropped here.
    norm_labels = [normalize_label(str(l)) for l in region_labels]
    region_order = sorted(
        (i for i in range(len(norm_labels)) if norm_labels[i] in label_to_idx),
        key=lambda i: label_to_idx[norm_labels[i]],
    )
    for i in region_order:
        s, e, lbl = region_starts_s[i], region_ends_s[i], norm_labels[i]
        region = signal_f[:, int(s):int(e)]
        feats = rms_moving_window(region, WINDOW_SAMPLES, STRIDE_SAMPLES)
        n = feats.shape[1]
        if n == 0:
            continue
        feature_blocks.append(feats)
        label_blocks.append(np.full(n, label_to_idx[lbl], dtype=np.int8))
        new_starts.append(cursor)
        new_ends.append(cursor + n)
        new_labels.append(lbl)
        new_files.append(source_files_in[i])
        new_span_idx.append(source_span_idx_in[i])
        cursor += n

    features = (np.concatenate(feature_blocks, axis=1) if feature_blocks
                else np.zeros((bad_mask.sum(), 0), dtype=np.float32))
    labels = (np.concatenate(label_blocks) if label_blocks
              else np.zeros(0, dtype=np.int8))
    feature_rate = fs / STRIDE_SAMPLES
    print(f"  -> features {features.shape}, "
          f"{features.shape[1] / feature_rate:.1f}s @ {feature_rate:.1f} Hz, "
          f"{len(new_starts)} regions")

    np.savez_compressed(
        out_path,
        features=features,
        labels=labels,
        fs=np.int32(fs),
        samples_per_frame=np.int32(STRIDE_SAMPLES),
        feature_rate=np.float64(feature_rate),
        feature_window_samples=np.int32(WINDOW_SAMPLES),
        kept_channels_1idx=kept_channels_1idx,
        channel_spec=channel_spec,
        bad_channels=bad_spec,
        region_starts=np.array(new_starts, dtype=np.int64),
        region_ends=np.array(new_ends, dtype=np.int64),
        region_labels=np.array(new_labels),
        source_files=np.array(new_files),
        source_span_idx=np.array(new_span_idx, dtype=np.int32),
        bandpass_hz=np.array(BANDPASS_HZ, dtype=np.float64),
        bandstop_hz=np.array(BANDSTOP_HZ, dtype=np.float64),
        butter_order=np.int32(BUTTER_ORDER),
    )
    print(f"  saved {out_path}")


def main():
    for p in ["P01", "P01_2", "P02"]:
        recompute_for_participant(p)


if __name__ == "__main__":
    main()
