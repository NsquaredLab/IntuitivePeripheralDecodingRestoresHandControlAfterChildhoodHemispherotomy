"""
_pipeline.py
============

Import shim + shared raw-data guard for the numbered Figure 2E pipeline steps.

The step scripts are named ``01_labeler.py`` … ``06_supplementary_figure.py``
so their run order is obvious on disk. A filename that starts with a digit is
not a valid Python module name, so the steps cannot import each other with a
normal ``import`` statement. ``load_step`` loads a sibling step by its logical
name (``"labeler"``, ``"rms_classification"``, …) and caches it in
``sys.modules`` so every step sees the exact same module object.
``require_raw_signal`` is the shared guard that turns a missing data file into
an explanatory message instead of a bare traceback. (The shared CatBoost
cross-validation engine lives in ``_cv.py``, a normal module that steps import
directly.)

Usage inside a step file:

    from _pipeline import load_step
    _labeler = load_step("labeler")
    GESTURES = _labeler.GESTURES

This is the ONLY behavioural change relative to the original
``playagain/classification/*.py`` modules — the analysis code itself is copied
verbatim, so results are identical.

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import importlib.util
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))

# logical name -> filename on disk (usage order encoded in the numeric prefix)
_STEP_FILES = {
    "labeler": "01_labeler.py",
    "dataset_creation": "02_dataset_creation.py",
    "feature_extraction": "03_feature_extraction.py",
    "rms_classification": "04_classification.py",
    "visualize_predictions": "05_visualize_predictions.py",
    "supplementary_figure": "06_supplementary_figure.py",
}


def require_raw_signal(path: str) -> str:
    """Guard for inputs that derive from identifiable participant EMG.

    No participant data are distributed with this repository — neither the raw
    recordings (the ``.pkl`` + ``labels.json`` and the ``data/<P>.npz``
    datasets) nor the derived RMS features — because they come from
    identifiable EMG of a vulnerable participant group. All data are available
    from the authors on reasonable request.
    Steps that need such an input call this first so a missing file yields an
    explanatory message instead of a bare traceback.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required data input is not available:\n    {path}\n"
            "No participant data are distributed with this repository — the raw "
            "recordings, the labelled datasets (data/<P>.npz) and the RMS "
            "features (data/<P>_rms_features.npz) all derive from identifiable "
            "EMG of a vulnerable participant group. Request the data from the "
            "authors, then place it in the "
            "data/ folder and re-run this step."
        )
    return path


def load_step(name: str):
    """Return the pipeline step module registered under `name`, importing it
    from its numbered file on first use and caching it in ``sys.modules``."""
    if name in sys.modules:
        return sys.modules[name]
    try:
        filename = _STEP_FILES[name]
    except KeyError:
        raise KeyError(
            f"unknown pipeline step {name!r}; known: {sorted(_STEP_FILES)}"
        ) from None
    path = os.path.join(_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so any (non-cyclic) cross-imports resolve to this
    # same object rather than triggering a second load.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
