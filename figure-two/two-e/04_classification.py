"""
04_classification.py
====================

Step 4 of the Figure 2E gesture-classification pipeline. This is the classifier
behind the published Figure 2E: CatBoost gesture classification on the
pre-computed RMS-feature datasets, with per-gesture-block 5-fold
cross-validation. Uses the shared CV + CatBoost engine (``_cv.py``); the
``*_rms_features.npz`` files already contain one RMS value per (channel, frame)
at the 32 Hz frame rate.

This step runs on the RMS features produced by step 3 and reproduces the
published Figure 2E cross-validation scores. The features are not distributed
with this repository (they derive from identifiable-participant data); they are
available from the authors on reasonable request (see
``data/RAW_DATA_ACCESS.md``).

Input / Output
--------------
    in : ``data/<P>_rms_features.npz``          (RMS features, from step 3)
    out: ``results/rms_classification_results.json``

Dependencies
------------
    numpy, scikit-learn, catboost
        (pip install numpy scikit-learn catboost)

Usage
-----
    uv run python figure-two/two-e/04_classification.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import json
import os

import numpy as np

from _cv import CATBOOST_PARAMS, RESULTS_DIR, cross_validate, summarize_folds
from _pipeline import load_step

_labeler = load_step("labeler")  # -> 01_labeler.py
GESTURES = _labeler.GESTURES
N_BIOSIGNAL_CHANNELS = _labeler.N_BIOSIGNAL_CHANNELS
normalize_label = _labeler.normalize_label
parse_channel_spec = _labeler.parse_channel_spec

OUTPUT_DIR = load_step("dataset_creation").OUTPUT_DIR  # -> 02_dataset_creation.py

# Local feature files produced by feature_extraction.py with the paper-aligned
# preprocessing (2nd-order Butterworth BP 20-500 Hz + BS 45-55 Hz, RMS over a
# 360-sample moving window with 64-sample stride). The previous NAS files used
# 5th-order filters and a different RMS scheme; they are no longer used.
RMS_DATASET_PATHS = {
    "P01":   os.path.join(OUTPUT_DIR, "P01_rms_features.npz"),
    "P01_2": os.path.join(OUTPUT_DIR, "P01_2_rms_features.npz"),
    "P02":   os.path.join(OUTPUT_DIR, "P02_rms_features.npz"),
}


def load_rms_dataset(path: str) -> dict:
    """Load a precomputed RMS-feature .npz and derive a per-frame region id.

    Per-frame integer labels are reconstructed from the order-independent
    string ``region_labels`` (run through ``normalize_label`` to translate
    legacy names) plus the region spans, rather than the integer ``labels``
    array stored in the file. The stored integers were baked with whatever
    GESTURES order was current at dataset-creation time; rebuilding from the
    strings keeps results correct after a reorder/rename without having to
    regenerate the .npz files.

    Frames not falling inside any of the labelled regions — or inside a
    region whose label is not a known gesture — get region_id=-1 and label
    =-1, and are excluded from training/eval downstream.
    """
    npz = np.load(path, allow_pickle=False)
    n_frames = int(npz["features"].shape[1])
    region_starts = npz["region_starts"].astype(np.int64)
    region_ends = npz["region_ends"].astype(np.int64)
    region_labels = np.array(
        [normalize_label(str(l)) for l in npz["region_labels"]]
    )

    label_to_idx = {g: i for i, g in enumerate(GESTURES)}
    region_ids_per_frame = np.full(n_frames, -1, dtype=np.int64)
    labels = np.full(n_frames, -1, dtype=np.int64)
    for region_idx, (s, e, lbl) in enumerate(
        zip(region_starts, region_ends, region_labels)
    ):
        if lbl not in label_to_idx:
            continue
        region_ids_per_frame[int(s):int(e)] = region_idx
        labels[int(s):int(e)] = label_to_idx[lbl]

    return {
        "features": npz["features"],  # (n_channels, n_frames)
        "labels": labels,
        "fs": int(npz["fs"]),
        "feature_rate_hz": float(npz["feature_rate"]),
        "kept_channels_1idx": npz["kept_channels_1idx"].astype(np.int64),
        "channel_spec": str(npz["channel_spec"]),
        "bad_channels": str(npz["bad_channels"]),
        "region_starts": region_starts,
        "region_ends": region_ends,
        "region_labels": region_labels,
        "region_ids_per_frame": region_ids_per_frame,
        "n_frames": n_frames,
    }


def keep_mask_from_bad_spec(kept_channels_1idx: np.ndarray,
                            bad_spec: str) -> np.ndarray:
    """`bad_spec` uses global 1..384 EMG numbering. Returns a boolean mask
    over the rows of `features` (which already only contain the
    participant's selected channels)."""
    if not bad_spec.strip():
        return np.ones(len(kept_channels_1idx), dtype=bool)
    bad_global_zero = parse_channel_spec(bad_spec, N_BIOSIGNAL_CHANNELS)
    bad_global_1idx = {g + 1 for g in bad_global_zero}
    return np.array(
        [int(c) not in bad_global_1idx for c in kept_channels_1idx],
        dtype=bool,
    )


def run_rms_dataset(participant: str, path: str, n_splits: int = 5) -> dict:
    print(f"\n=== {participant} ===")
    print(f"  loading {path}")
    data = load_rms_dataset(path)

    bad_mask = keep_mask_from_bad_spec(data["kept_channels_1idx"],
                                       data["bad_channels"])
    n_dropped = int((~bad_mask).sum())
    print(f"  channels: {bad_mask.sum()}/{len(bad_mask)} kept "
          f"(dropped {n_dropped}: '{data['bad_channels'] or 'none'}')")

    # Drop bad-channel rows, then transpose so each frame is one feature row.
    features = data["features"][bad_mask]                # (n_kept, n_frames)
    frame_mask = data["region_ids_per_frame"] >= 0       # only labelled frames
    X = features[:, frame_mask].T.astype(np.float32)     # (n_used_frames, n_kept)
    y = data["labels"][frame_mask].astype(np.int64)
    region_ids = data["region_ids_per_frame"][frame_mask]

    n_total = int(data["n_frames"])
    n_used = int(X.shape[0])
    duration_s = n_used / data["feature_rate_hz"]
    print(f"  frames: {n_used}/{n_total} labelled "
          f"(~{duration_s:.1f}s @ {data['feature_rate_hz']:.0f} Hz)")
    print(f"  feature matrix: {X.shape}")
    counts = {GESTURES[i]: int((y == i).sum()) for i in range(len(GESTURES))}
    print(f"  class counts: {counts}")
    n_regions = int(len(np.unique(region_ids)))
    print(f"  regions: {n_regions}")

    classes_with_data = sum(c > 0 for c in counts.values())
    if classes_with_data < 2:
        print(f"  only {classes_with_data} class(es) present — skipping")
        return {"participant": participant, "path": path,
                "skipped": "single_class", "class_counts": counts}

    print(f"  running {n_splits}-fold per-gesture-block CV with CatBoost "
          f"(task_type={CATBOOST_PARAMS['task_type']}, MyoGestic defaults)...")
    # Fold over whole gesture-class blocks (Rest/Grasp/Pinch/Tripod), not each
    # individual region: split each gesture's frames into n chronological
    # chunks, fold k = chunk k of every gesture.
    fold_results, _, _ = cross_validate(X, y, region_ids, n_splits=n_splits,
                                        group_by=y)
    summary = summarize_folds(fold_results)
    print(f"  summary: "
          f"acc={summary['accuracy']['mean']:.3f}+/-{summary['accuracy']['std']:.3f}  "
          f"f1_macro={summary['f1_macro']['mean']:.3f}+/-{summary['f1_macro']['std']:.3f}")

    return {
        "participant": participant,
        "path": path,
        "n_channels_used": int(bad_mask.sum()),
        "n_channels_dropped": n_dropped,
        "n_frames": n_used,
        "feature_rate_hz": data["feature_rate_hz"],
        "class_counts": counts,
        "n_regions": n_regions,
        "n_splits": n_splits,
        "catboost_params": {k: v for k, v in CATBOOST_PARAMS.items()
                            if k != "train_dir"},
        "fold_results": fold_results,
        "summary": summary,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results: dict[str, dict] = {}
    for participant, path in RMS_DATASET_PATHS.items():
        if not os.path.isfile(path):
            print(f"\n=== {participant} ===\n  missing: {path}")
            all_results[participant] = {"participant": participant,
                                        "path": path, "skipped": "missing"}
            continue
        all_results[participant] = run_rms_dataset(participant, path)

    out_path = os.path.join(RESULTS_DIR, "rms_classification_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {out_path}")

    print("\n=== Overall summary ===")
    for p, r in all_results.items():
        if "summary" in r:
            s = r["summary"]
            print(f"  {p}: "
                  f"acc={s['accuracy']['mean']:.3f}+/-{s['accuracy']['std']:.3f}  "
                  f"f1_macro={s['f1_macro']['mean']:.3f}+/-{s['f1_macro']['std']:.3f}")
        else:
            print(f"  {p}: skipped ({r.get('skipped')})")


if __name__ == "__main__":
    main()
