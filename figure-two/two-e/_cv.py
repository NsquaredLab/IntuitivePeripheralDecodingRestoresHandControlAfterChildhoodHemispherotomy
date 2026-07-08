"""
_cv.py
======

Shared CatBoost cross-validation engine for the Figure 2E pipeline.

Provides the cross-validation routine and CatBoost configuration imported by the
classifier (step 4), the confusion-matrix figures (step 5) and the supplementary
figure (step 6):

    - ``CATBOOST_PARAMS``          reproducible CatBoost settings (CPU, fixed seed)
    - ``cross_validate``           per-region / per-gesture-block chronological
                                   K-fold CV with z-scoring on the train fold
    - ``per_region_kfold_splits``  chronological K-fold over individual regions
    - ``per_gesture_kfold_splits`` chronological K-fold over gesture-class blocks
    - ``summarize_folds``          mean ± std across folds
    - ``RESULTS_DIR``              two-e/results output directory

This module holds no analysis of its own — it is the common engine so the RMS
computation and the classification live in exactly one place each.

Dependencies
------------
    numpy, scikit-learn, catboost

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import os

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from _pipeline import load_step

_labeler = load_step("labeler")  # -> 01_labeler.py
GESTURES = _labeler.GESTURES

OUTPUT_DIR = load_step("dataset_creation").OUTPUT_DIR  # -> 02_dataset_creation.py
RESULTS_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "results")

N_FOLDS = 5
RANDOM_SEED = 42

# CPU + a fixed random_seed makes CatBoost bit-for-bit reproducible. GPU training
# is non-deterministic (floating-point reduction ordering) even with a seed set.
CATBOOST_PARAMS = {
    "iterations": 1000,
    "l2_leaf_reg": 5,
    "border_count": 254,
    "task_type": "CPU",
    "random_seed": RANDOM_SEED,
    "train_dir": None,
    "verbose": False,
    "allow_writing_files": False,
}


def per_region_kfold_splits(
    region_ids: np.ndarray, n_splits: int = N_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-region chronological K-fold.

    Each region's windows are split chronologically into `n_splits` equal
    chunks (np.array_split handles uneven divisions). Fold k uses chunk k of
    every region as test, and the remaining chunks as train. This way each
    fold is a contiguous 1/k slice of every labelled region rather than a
    shuffled mix, which avoids overlapping windows leaking across folds.

    Returns a list of (train_idx, test_idx) tuples — one entry per fold.
    """
    n_total = len(region_ids)
    if n_total == 0:
        return []

    chunks_per_region: dict[int, list[np.ndarray]] = {}
    for r in np.unique(region_ids):
        idxs = np.where(region_ids == r)[0]  # already chronological
        chunks_per_region[int(r)] = np.array_split(idxs, n_splits)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        test_parts = [chunks[k] for chunks in chunks_per_region.values()
                      if len(chunks[k]) > 0]
        if not test_parts:
            continue
        test_idx = np.sort(np.concatenate(test_parts))
        mask = np.ones(n_total, dtype=bool)
        mask[test_idx] = False
        train_idx = np.where(mask)[0]
        splits.append((train_idx, test_idx))
    return splits


def per_gesture_kfold_splits(
    labels: np.ndarray, n_splits: int = N_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Like `per_region_kfold_splits`, but the grouping unit is the gesture
    CLASS block (Rest / Grasp / Pinch / Tripod Pinch) instead of each
    individual extracted region. Each gesture's frames (chronological, as
    concatenated by the feature pipeline) are split into `n_splits` chunks;
    fold k uses chunk k of every gesture as test. With 4 gestures this yields
    exactly one contiguous validation chunk per gesture per fold, and far
    fewer chunk boundaries than the per-region scheme.

    Implemented by reusing the per-region splitter with the label array as the
    grouping key.
    """
    return per_region_kfold_splits(labels, n_splits=n_splits)


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    region_ids: np.ndarray,
    n_splits: int = N_FOLDS,
    catboost_params: dict | None = None,
    group_by: np.ndarray | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """5-fold per-region chronological CV. Each fold: z-score normalize on
    train, apply the same normalization to test, fit CatBoost, evaluate.

    Returns:
        fold_results: list of per-fold metric dicts
        oof_predictions: per-frame out-of-fold class prediction (-1 means
            the frame was never in any fold's test set, e.g. a region too
            short to contribute to all 5 chunks)
        fold_assignments: per-frame fold index that produced the prediction
            (1-indexed; -1 for never-tested frames)
    """
    catboost_params = {**CATBOOST_PARAMS, **(catboost_params or {})}
    # group_by lets callers fold over gesture-class blocks (pass the labels)
    # instead of individual regions; default keeps per-region behaviour.
    group = region_ids if group_by is None else group_by
    splits = per_region_kfold_splits(group, n_splits=n_splits)
    fold_results: list[dict] = []
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    fold_assignments = np.full(len(y), -1, dtype=np.int64)
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        X_train_n = (X_train - mean) / std
        X_test_n = (X_test - mean) / std

        model = CatBoostClassifier(**catboost_params)
        model.fit(X_train_n, y_train)
        y_pred = model.predict(X_test_n).reshape(-1).astype(int)
        oof_pred[test_idx] = y_pred
        fold_assignments[test_idx] = fold_idx

        f1_per_class = f1_score(
            y_test, y_pred, labels=list(range(len(GESTURES))),
            average=None, zero_division=0,
        )
        result = {
            "fold": fold_idx,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro",
                                       zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted",
                                          zero_division=0)),
            "f1_per_class": {GESTURES[i]: float(f1_per_class[i])
                             for i in range(len(GESTURES))},
            "confusion_matrix": confusion_matrix(
                y_test, y_pred, labels=list(range(len(GESTURES))),
            ).tolist(),
        }
        fold_results.append(result)
        print(f"  fold {fold_idx}/{len(splits)}: "
              f"acc={result['accuracy']:.3f}  "
              f"f1_macro={result['f1_macro']:.3f}  "
              f"f1_weighted={result['f1_weighted']:.3f}  "
              f"(train={result['n_train']}, test={result['n_test']})")
    return fold_results, oof_pred, fold_assignments


def summarize_folds(fold_results: list[dict]) -> dict:
    accs = np.array([f["accuracy"] for f in fold_results])
    f1m = np.array([f["f1_macro"] for f in fold_results])
    f1w = np.array([f["f1_weighted"] for f in fold_results])
    return {
        "accuracy": {"mean": float(accs.mean()), "std": float(accs.std())},
        "f1_macro": {"mean": float(f1m.mean()), "std": float(f1m.std())},
        "f1_weighted": {"mean": float(f1w.mean()), "std": float(f1w.std())},
    }
