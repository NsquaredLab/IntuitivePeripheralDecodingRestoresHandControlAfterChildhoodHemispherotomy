"""
01_labeler.py
=============

Step 1 of the Figure 2E gesture-classification pipeline: an interactive
labelling tool.

Opens a single long-lived Matplotlib window that lets the user step through the
per-trial high-density surface-EMG recordings of every participant, mark the
time spans in which each gesture (Rest, Power Grasp, Pinch, Tripod Pinch) was
performed, flag whole recordings or individual channels as bad, and have every
change persisted to a ``labels.json`` next to the recordings. All later steps
consume that ``labels.json``.

The canonical gesture order and the per-gesture colours defined here
(``GESTURES``, ``GESTURE_COLORS``) are imported by every other step, so this
file is the single source of truth for the class layout used in the label
indices, confusion-matrix rows/columns and plot legends.

Input / Output
--------------
    in : raw Quattrocento ``*.pkl`` recordings on the lab NAS (per participant)
    out: ``labels.json`` written next to each participant's recordings

The raw recordings are not distributed with this repository because they
contain identifiable human EMG; they are available from the authors on
reasonable request. This step therefore only
runs with access to the raw data.

Dependencies
------------
    numpy, matplotlib, pysynclient    (pip install numpy matplotlib pysynclient)

Usage
-----
    uv run python figure-two/two-e/01_labeler.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import json
import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.widgets import (
    Button,
    CheckButtons,
    RadioButtons,
    SpanSelector,
    TextBox,
)
from pysynclient import get_server_paths

# Canonical gesture order. This order is used everywhere (label indices,
# confusion-matrix rows/cols, plot legends): Rest, Power Grasp, Pinch,
# Tripod Pinch.
GESTURES = ["Rest", "Power Grasp", "Pinch", "Tripod Pinch"]

# Per-gesture colors — adjust a single value here to recolor that gesture in
# every plot (label strips, legends, region tints).
GESTURE_COLORS = {
    "Rest": "lightgray",
    "Power Grasp": "#4371CB",
    "Pinch": "#FF868B",
    "Tripod Pinch": "#FEFABC",
}

# Gesture labels stored on disk before the rename/reorder. Old labels.json
# files and previously-built .npz datasets contain "Faust"; map it to the
# current name so existing data still loads correctly.
LEGACY_LABEL_ALIASES = {"Faust": "Power Grasp"}


def normalize_label(label: str) -> str:
    """Map a stored gesture label to its current canonical name.

    Any label written before the Faust -> Power Grasp rename is translated;
    current names pass through unchanged.
    """
    return LEGACY_LABEL_ALIASES.get(label, label)
LABELS_FILENAME = "labels.json"
N_BIOSIGNAL_CHANNELS = 384

# Directories on the lab file server that hold each participant's raw
# recordings. The real paths are de-identified here (they encoded participant
# initials and recording dates) to protect the participants; the authors
# substitute the actual file-server paths from a private mapping when running
# these steps. "P01"/"P01_2" are two sessions of the same participant.
PARTICIPANTS = {
    "P01":   r"<RAW_DATA_ROOT>\P01\session_01",
    "P01_2": r"<RAW_DATA_ROOT>\P01\session_02",
    "P02":   r"<RAW_DATA_ROOT>\P02\session_01",
}


def flatten_biosignal(biosignal_3d: np.ndarray) -> np.ndarray:
    # (channels, samples_per_frame, frames) -> (channels, samples), then drop
    # the 16 auxiliary channels at the tail so only EMG remains.
    channels = biosignal_3d.shape[0]
    flat = biosignal_3d.transpose(0, 2, 1).reshape(channels, -1)
    return flat[:N_BIOSIGNAL_CHANNELS].astype(np.float32)


def parse_channel_spec(spec: str, n_channels: int) -> set[int]:
    # Empty spec means "keep everything". Channels in the spec are 1-indexed,
    # ranges are inclusive on both ends, separators are ; or ,.
    if not spec.strip():
        return set(range(n_channels))
    kept: set[int] = set()
    for part in spec.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a.strip()) - 1
            end = int(b.strip())
            kept.update(range(max(0, start), min(n_channels, end)))
        else:
            idx = int(part) - 1
            if 0 <= idx < n_channels:
                kept.add(idx)
    return kept


def channels_to_spec(channels: set[int], n_channels: int) -> str:
    if not channels or len(channels) == n_channels:
        return ""
    sorted_ch = sorted(channels)
    parts: list[str] = []
    run_start = prev = sorted_ch[0]
    for c in sorted_ch[1:]:
        if c == prev + 1:
            prev = c
            continue
        parts.append(f"{run_start + 1}" if run_start == prev
                     else f"{run_start + 1}-{prev + 1}")
        run_start = prev = c
    parts.append(f"{run_start + 1}" if run_start == prev
                 else f"{run_start + 1}-{prev + 1}")
    return "; ".join(parts)


def load_existing_labels(directory: str) -> dict:
    path = os.path.join(directory, LABELS_FILENAME)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {entry["file"]: entry for entry in json.load(f)}


def save_labels(directory: str, labels: dict) -> None:
    path = os.path.join(directory, LABELS_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(labels.values()), f, indent=2)


def short_name(file_name: str) -> str:
    s = file_name
    if s.startswith("VHI_Recording_"):
        s = s[len("VHI_Recording_"):]
    if s.endswith(".pkl"):
        s = s[:-len(".pkl")]
    return s


class LabelerApp:
    """Single long-lived figure that lets the user pick any participant and
    any file via sidebars on the left, edit labels, and have changes persisted
    automatically when navigating away or closing the window."""

    def __init__(self, server_path: str):
        self.server_path = server_path
        self.participant_names = list(PARTICIPANTS.keys())

        self.current_participant: str | None = None
        self.current_dir: str | None = None
        self.files: list[str] = []
        self.labels: dict[str, dict] = {}
        self.current_file: str | None = None

        # Per-file display state (rebuilt on every load_file)
        self.n_channels = N_BIOSIGNAL_CHANNELS
        self.fs = 0
        self.lc: LineCollection | None = None

        # Mutable label state for current file
        self.gesture = GESTURES[0]
        self.keep = True
        self.spans: list[dict] = []
        self.patches: list = []
        self.texts: list = []
        self.kept_channels: set[int] = set(range(self.n_channels))

        # Suppress widget callbacks during programmatic state restores
        self._suppress = False
        self.span_selector: SpanSelector | None = None

        self._build_figure()
        self.load_participant(self.participant_names[0])
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        plt.show()

    # ---------- layout ----------

    def _build_figure(self):
        self.fig = plt.figure(figsize=(18, 10))

        # Main plot area
        self.ax = self.fig.add_axes([0.25, 0.13, 0.73, 0.82])

        # Participant selector (top-left)
        self.fig.text(0.01, 0.945, "Participant", fontsize=9, weight="bold")
        self.participant_ax = self.fig.add_axes([0.01, 0.81, 0.22, 0.13])
        self.participant_radio = RadioButtons(
            self.participant_ax, self.participant_names, active=0
        )
        for t in self.participant_radio.labels:
            t.set_fontsize(9)
        self.participant_radio.on_clicked(self._on_participant)

        # File navbar (left middle) — created/replaced per participant
        self.fig.text(0.01, 0.795, "Files", fontsize=9, weight="bold")
        self.navbar_ax = self.fig.add_axes([0.01, 0.34, 0.22, 0.45])
        self.nav_radio: RadioButtons | None = None
        self._nav_label_to_file: dict[str, str] = {}

        # Gesture selector (left lower)
        self.fig.text(0.01, 0.325, "Gesture", fontsize=9, weight="bold")
        self.gesture_ax = self.fig.add_axes([0.01, 0.18, 0.22, 0.14])
        self.gesture_radio = RadioButtons(self.gesture_ax, GESTURES, active=0)
        self.gesture_radio.set_radio_props(
            {"facecolor": [GESTURE_COLORS[g] for g in GESTURES]}
        )
        for t, g in zip(self.gesture_radio.labels, GESTURES):
            t.set_color(GESTURE_COLORS[g])
            t.set_fontsize(9)
        self.gesture_radio.on_clicked(self._on_gesture)

        # Keep checkbox
        self.check_ax = self.fig.add_axes([0.01, 0.115, 0.22, 0.05])
        self.check = CheckButtons(self.check_ax, ["Keep file"], [True])
        self.check.on_clicked(self._on_keep_toggle)

        # Undo button
        self.undo_ax = self.fig.add_axes([0.01, 0.05, 0.10, 0.05])
        self.undo_btn = Button(self.undo_ax, "Undo span")
        self.undo_btn.on_clicked(self._on_undo)

        # Channel textbox (below plot)
        self.tax = self.fig.add_axes([0.30, 0.05, 0.55, 0.05])
        self.text_box = TextBox(self.tax, "Channels  ", initial="")
        self.text_box.on_text_change(
            lambda t: self._on_channels(t, log_errors=False)
        )
        self.text_box.on_submit(
            lambda t: self._on_channels(t, log_errors=True)
        )

    def _build_navbar(self):
        if self.nav_radio is not None:
            self.nav_radio.disconnect_events()
            self.nav_radio = None
        self.navbar_ax.clear()
        self.navbar_ax.set_xticks([])
        self.navbar_ax.set_yticks([])
        if not self.files:
            self._nav_label_to_file = {}
            return
        labels = []
        self._nav_label_to_file = {}
        for f in self.files:
            label = short_name(f)
            # Disambiguate if collisions
            if label in self._nav_label_to_file:
                label = f
            labels.append(label)
            self._nav_label_to_file[label] = f
        active = 0
        if self.current_file in self.files:
            active = self.files.index(self.current_file)
        self.nav_radio = RadioButtons(self.navbar_ax, labels, active=active)
        for t in self.nav_radio.labels:
            t.set_fontsize(7)
        self._update_navbar_colors()
        self.nav_radio.on_clicked(self._on_nav_select)

    def _update_navbar_colors(self):
        if self.nav_radio is None:
            return
        for i, file_name in enumerate(self.files):
            entry = self.labels.get(file_name)
            if entry is None:
                color = "black"
            elif not entry.get("keep", True):
                color = "tab:red"
            elif entry.get("spans"):
                color = "tab:green"
            else:
                color = "tab:orange"
            self.nav_radio.labels[i].set_color(color)

    # ---------- data loading + persistence ----------

    def load_participant(self, name: str):
        self._snapshot_and_persist()

        self.current_participant = name
        self.current_dir = os.path.join(self.server_path, PARTICIPANTS[name])
        self.files = sorted(
            os.path.basename(f)
            for f in glob(os.path.join(self.current_dir, "*.pkl"))
            if os.path.getsize(f) > 0
        )
        self.labels = load_existing_labels(self.current_dir)
        self.current_file = None
        self._build_navbar()
        if self.files:
            self.load_file(self.files[0])
        else:
            self.ax.clear()
            self.ax.set_title(f"[{name}] no .pkl files in this directory")
            self.fig.canvas.draw_idle()

    def load_file(self, file_name: str):
        self._snapshot_and_persist()

        self.current_file = file_name
        file_path = os.path.join(self.current_dir, file_name)
        with open(file_path, "rb") as f:
            d = pickle.load(f)
        self.fs = int(d["device_information"]["sampling_frequency"])
        biosignal = flatten_biosignal(d["biosignal"])
        self.n_channels, n_samples = biosignal.shape
        t = np.arange(n_samples) / self.fs
        centered = biosignal - biosignal.mean(axis=1, keepdims=True)
        spread = float(np.median(np.std(centered, axis=1))) * 4 or 1.0

        ds = max(1, n_samples // 5000)
        t_disp = t[::ds]
        c_disp = centered[:, ::ds]

        self.ax.clear()
        segments = [
            np.column_stack([t_disp, c_disp[c] + c * spread])
            for c in range(self.n_channels)
        ]
        self.lc = LineCollection(segments, linewidths=0.4,
                                 colors=["black"] * self.n_channels)
        self.lc.set_rasterized(True)
        self.ax.add_collection(self.lc)
        self.ax.set_xlim(t[0], t[-1])
        self.ax.set_ylim(-spread, self.n_channels * spread)
        tick_step = 32
        self.ax.set_yticks(np.arange(0, self.n_channels, tick_step) * spread)
        self.ax.set_yticklabels(
            [str(i + 1) for i in np.arange(0, self.n_channels, tick_step)]
        )
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Channel")
        prior = self.labels.get(file_name)
        title_prefix = "(edit) " if prior else ""
        idx = self.files.index(file_name) + 1
        self.ax.set_title(
            f"[{self.current_participant}] [{idx}/{len(self.files)}] "
            f"{title_prefix}{file_name}"
        )

        # Restore label state from prior (or defaults)
        self.keep = bool(prior["keep"]) if prior else True
        spec = (prior.get("channels", "") if prior else "")
        self.kept_channels = parse_channel_spec(spec, self.n_channels)
        self.spans = []
        self.patches = []
        self.texts = []
        if prior:
            for span in prior.get("spans", []):
                if span.get("label") in GESTURE_COLORS:
                    self._add_span_visual(span["start"], span["end"], span["label"])

        # Sync widgets without firing callbacks
        self._suppress = True
        textbox_value = (
            channels_to_spec(self.kept_channels, self.n_channels)
            or f"1-{self.n_channels}"
        )
        self.text_box.set_val(textbox_value)
        if self.check.get_status()[0] != self.keep:
            self.check.set_active(0)
        self._suppress = False

        self._apply_channel_colors()

        # ax.clear() removed the SpanSelector's internal artist — re-create.
        if self.span_selector is not None:
            self.span_selector.disconnect_events()
        self.span_selector = SpanSelector(
            self.ax, self._on_select, "horizontal", useblit=True,
            props=dict(alpha=0.3, facecolor="orange"),
        )

        self.fig.canvas.draw_idle()

    def _snapshot_and_persist(self):
        if self.current_file is None or self.current_dir is None:
            return
        self.labels[self.current_file] = {
            "file": self.current_file,
            "keep": self.keep,
            "spans": list(self.spans),
            "channels": channels_to_spec(self.kept_channels, self.n_channels),
        }
        save_labels(self.current_dir, self.labels)
        self._update_navbar_colors()

    # ---------- visual helpers ----------

    def _add_span_visual(self, xmin: float, xmax: float, label: str):
        color = GESTURE_COLORS[label]
        patch = self.ax.axvspan(xmin, xmax, alpha=0.25, color=color)
        text = self.ax.text(
            (xmin + xmax) / 2, self.ax.get_ylim()[1] * 0.99, label,
            ha="center", va="top", fontsize=8, color=color,
        )
        self.spans.append({"start": float(xmin), "end": float(xmax),
                           "label": label})
        self.patches.append(patch)
        self.texts.append(text)

    def _apply_channel_colors(self):
        if self.lc is None:
            return
        kept = self.kept_channels
        self.lc.set_color(["black" if c in kept else "lightgrey"
                           for c in range(self.n_channels)])
        self.fig.canvas.draw_idle()

    # ---------- widget callbacks ----------

    def _on_participant(self, name: str):
        if self._suppress or name == self.current_participant:
            return
        self.load_participant(name)

    def _on_nav_select(self, label: str):
        if self._suppress:
            return
        target = self._nav_label_to_file.get(label)
        if target is None or target == self.current_file:
            return
        self.load_file(target)

    def _on_gesture(self, label: str):
        if self._suppress:
            return
        self.gesture = label

    def _on_keep_toggle(self, _):
        if self._suppress:
            return
        self.keep = not self.keep

    def _on_undo(self, _):
        if not self.spans:
            return
        self.spans.pop()
        self.patches.pop().remove()
        self.texts.pop().remove()
        self.fig.canvas.draw_idle()

    def _on_channels(self, text: str, *, log_errors: bool):
        if self._suppress:
            return
        try:
            self.kept_channels = parse_channel_spec(text, self.n_channels)
        except ValueError:
            if log_errors:
                print(f"Could not parse channel spec: {text!r}")
            return
        self._apply_channel_colors()

    def _on_select(self, xmin: float, xmax: float):
        if xmax - xmin < 1e-3:
            return
        self._add_span_visual(xmin, xmax, self.gesture)
        self.fig.canvas.draw_idle()

    def _on_close(self, _event):
        self._snapshot_and_persist()


def main():
    server_path = get_server_paths()["nsquared-nas"]["datasets"]
    LabelerApp(server_path)


if __name__ == "__main__":
    main()
