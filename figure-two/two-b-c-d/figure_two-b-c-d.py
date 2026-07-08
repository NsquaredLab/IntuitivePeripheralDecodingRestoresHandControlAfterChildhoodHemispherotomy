"""
figure2.py
==========

Figure 2 has three data sub-panels, each written to its own SVG so they can be
assembled freely in the manuscript:

    2B  Exemplary raw high-density surface-EMG: 32 channels, band-pass +
        notch filtered, normalised and stacked with a vertical offset.

    2C  Root-mean-square (RMS) envelope of the EMG together with the motor-unit
        (MU) spike trains and their instantaneous firing rates during the
        guided gesture attempts. Gray shading marks the hold periods.

    2D  Spatial map of one selected motor unit: a heatmap of the peak-to-peak
        amplitude of the spike-triggered-average (STA) MUAP across the
        electrode grid, next to the STA MUAP waveform of the strongest channel.
        Together these show the spatial distribution and the shape of that
        motor unit's action potential.

Two input files are required (both come from the same recording):

    * a Quattrocento ``*.pkl`` recording  -> raw EMG channels + kinematic
      ground truth (used for 2B and for the RMS / hold periods of 2C);
    * a decomposition ``*_finish.mat`` file -> ``SIG`` (per-channel EMG grid),
      ``MUPulses`` (per-MU spike-sample indices) and ``discardChannelsVec``
      (used for the spike trains of 2C and the STA map of 2D).

Every method required to compute the figure is included here (only the standard
scientific-Python stack is imported); 

Dependencies
------------
    numpy, scipy, matplotlib          (pip install numpy scipy matplotlib)

Usage
-----
    python figure2.py  <recording.pkl>  <decomposition_finish.mat>

or edit the ``PKL_PATH`` / ``MAT_PATH`` constants at the bottom of the file.

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import pickle

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import butter, iirnotch, filtfilt


# ---------------------------------------------------------------------------
# Experiment / hardware constants
# ---------------------------------------------------------------------------
FS = 2048  # [Hz]  EMG sampling frequency of the Quattrocento recording device

# Electrode layout of the high-density grid used for the STA heatmap (1D).
# The decomposition stores channels on a 13 x 17 raster made of three 13 x 5
# grids; columns 6 and 12 are the (empty) gaps between the three physical grids.
# Rows / cols are 1-indexed here to match the toolbox convention.
GRIDS = [
    {"rows": range(1, 14), "cols": range(1, 6)},    # grid 1: cols 1-5
    {"rows": range(1, 14), "cols": range(7, 12)},   # grid 2: cols 7-11
    {"rows": range(1, 14), "cols": range(13, 18)},  # grid 3: cols 13-17
]

# ---- Toolbox house colours -----------------------------------------------------
DARK_BLUE = "#001C55"
BLUE = "#4371CB"
ROSA = "#FF8088"
HELLBEIGE = "#FCF9BB"
# Custom colormap used for the stacked raw channels of 1B.
CHANNEL_CMAP = LinearSegmentedColormap.from_list(
    "blau_rosa_hellbeige", [BLUE, ROSA, HELLBEIGE])


# ===========================================================================
# 1. Data loading
# ===========================================================================
def load_raw_recording(pkl_path):
    """
    Load the raw high-density EMG and the kinematic ground truth from a
    Quattrocento ``*.pkl`` recording.

    Reproduces ``import_pickle.importPkl_4cento_path`` /
    ``groundtruth_Myogestic``.

    Parameters
    ----------
    pkl_path : str
        Path to the recording pickle.

    Returns
    -------
    emg : ndarray, shape (n_channels, n_samples)
        Raw EMG, one row per channel (the last 24 auxiliary channels are
        dropped, as in the toolbox).
    trigger : ndarray, shape (n_gt_samples,)
        Kinematic ground-truth trace (mean across the recorded degrees of
        freedom); non-zero during gesture holds. Used to shade hold periods.
    fsamp : float
        Sampling frequency in Hz.
    task : str
        Task label stored in the recording (e.g. "Power Grasp").
    """
    with open(pkl_path, "rb") as f:
        recording = pickle.load(f)

    # biosignal: (channels, n_windows, window_len). Drop the last 24 auxiliary
    # channels, then flatten the windowed signal into one continuous trace.
    biosignal = recording["biosignal"][:-24]
    biosignal = np.transpose(biosignal, (0, 2, 1))         # (ch, window_len, n_windows)
    emg = np.array([np.concatenate(channel) for channel in biosignal])  # (ch, samples)

    # Kinematic ground truth -> single 1-D trigger (mean over DOFs).
    kinematics = recording["ground_truth"]
    trigger = np.mean(kinematics, axis=0)

    fsamp = recording["device_information"]["sampling_frequency"]
    task = recording.get("task", "")

    print(f"Raw EMG: {emg.shape[0]} channels x {emg.shape[1]} samples @ {fsamp} Hz")
    return emg, trigger, fsamp, task


def load_decomposition(mat_path):
    """
    Load the motor-unit decomposition from a ``*_finish.mat`` file.

    Parameters
    ----------
    mat_path : str
        Path to the decomposition MAT-file.

    Returns
    -------
    SIG : ndarray of object, shape (rows, cols)
        Per-channel EMG arranged on the electrode grid; each cell is the 1-D
        EMG time series of that electrode (empty for gap columns).
    MUPulses : ndarray of object, shape (n_mu,)
        For every detected motor unit, the sample indices of its discharges.
    discardChannelsVec : ndarray, shape (rows, cols)
        1 = channel discarded (bad), 0 = channel kept.
    fsamp : float
        Sampling frequency stored in the decomposition.
    """
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    SIG = mat["SIG"]
    MUPulses = mat["MUPulses"]
    discardChannelsVec = mat["discardChannelsVec"]
    fsamp = float(mat["fsamp"])

    # Make MUPulses uniformly indexable even when only one MU was found.
    if MUPulses.dtype != "object":
        MUPulses = np.array([MUPulses], dtype=object)

    print(f"Decomposition: {MUPulses.shape[0]} MUs, "
          f"SIG grid {SIG.shape}, fsamp {fsamp}")
    return SIG, MUPulses, discardChannelsVec, fsamp


# ===========================================================================
# 2. Signal helpers 
# ===========================================================================
def compute_rms(signal, fs, window):
    """
    Mean RMS envelope across all channels of a (channels x samples) EMG array.

    A moving-average of the squared signal (window ``window`` seconds) is taken
    per channel; the per-channel RMS traces are then averaged into one 1-D
    reference signal.

    Returns
    -------
    ndarray, shape (n_samples,)
    """
    window_size = int(window * fs)
    signal = np.atleast_2d(signal)
    if signal.shape[0] > signal.shape[1]:
        signal = signal.T

    num_channels, num_samples = signal.shape
    rms_signal = np.zeros((num_channels, num_samples))
    for ch in range(num_channels):
        squared = signal[ch, :] ** 2
        cumsum = np.cumsum(np.insert(squared, 0, 0))
        mov_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        pad_left = window_size // 2
        pad_right = num_samples - len(mov_avg) - pad_left
        mov_avg = np.pad(mov_avg, (pad_left, pad_right), mode="edge")
        rms_signal[ch, :] = np.sqrt(mov_avg)

    return np.mean(rms_signal, axis=0)


def extract_trigger(trigger):
    """
    Return the start/stop sample indices (in the trigger's own time base) of
    every contiguous run where the trigger is above a small threshold.
    """
    thr = 0.001
    above = trigger > thr
    d = np.diff(np.concatenate(([0], above.astype(int), [0])))
    start = np.where(d == 1)[0]
    stop = np.where(d == -1)[0] - 1
    return start, stop


def compute_trigger_timings(ref_signal, trigger, fsamp):
    """
    Convert the trigger's active runs into (start, stop) times **in seconds** on
    the reference-signal (RMS) time base, so they can be drawn with ``axvspan``.
    """
    conversion = len(ref_signal) / trigger.shape[0]
    start, stop = extract_trigger(trigger)
    start = np.ceil(start * conversion / fsamp).astype(int)
    stop = np.ceil(stop * conversion / fsamp).astype(int)
    return start, stop


def sta(signal, spike_times, window):
    """
    Spike-triggered average of ``signal`` at the given ``spike_times``.

    Parameters
    ----------
    signal : 1-D array
        EMG of a single channel.
    spike_times : array of int
        Discharge sample indices of one motor unit.
    window : int
        Total window length (samples) centred on each spike.

    Returns
    -------
    ndarray
        Averaged snippet (the channel's contribution to the MUAP); empty array
        if no spike had a full window inside the signal.
    """
    snippets = []
    half_window = window // 2
    signal_length = len(signal)
    for t in spike_times:
        if t - half_window >= 0 and t + half_window <= signal_length:
            snippets.append(signal[t - half_window:t + half_window])
    if len(snippets) == 0:
        return np.array([])
    return np.mean(np.array(snippets), axis=0)


def get_MU_shapes_single(SIG, mu_pulses, fsamp, window_seconds):
    """
    Compute the STA MUAP shape of one motor unit for every electrode of the grid.

    Returns
    -------
    list of list of ndarray
        ``mu_shapes[i][j]`` is the STA waveform at grid position (i, j).
    """
    rows = len(SIG)
    cols = len(SIG[0]) if rows > 0 else 0
    window_length = round(fsamp * window_seconds)

    mu_shapes = [[None for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            mu_shapes[i][j] = sta(SIG[i][j], mu_pulses, window_length)
    return mu_shapes


def compute_mapP2P(mu_shapes, discardChannelsVec):
    """
    Peak-to-peak amplitude of every channel's STA MUAP -> spatial amplitude map.
    Discarded / empty channels are left as NaN.
    """
    m = len(mu_shapes)
    n = len(mu_shapes[0]) if m > 0 else 0
    mapP2P = np.full((m, n), np.nan)
    for i in range(m):
        for j in range(n):
            if discardChannelsVec[i][j] == 0 and mu_shapes[i][j].shape[0] > 0:
                mapP2P[i, j] = np.max(mu_shapes[i][j]) - np.min(mu_shapes[i][j])
    return mapP2P


def fill_heatmap_NaNs(amp_map, discardChannelsVec, grids):
    """
    Fill the NaNs of discarded channels *inside* the physical grids with a
    neighbouring channel's value, so the heatmap has no holes within a grid
    (gap columns between grids stay NaN).
    """
    m, n = amp_map.shape
    is_grid = np.zeros(amp_map.shape)
    for g in grids:
        rows = np.array(g["rows"]) - 1
        cols = np.array(g["cols"]) - 1
        is_grid[np.ix_(rows, cols)] = 1

    new_map = amp_map.copy()
    for i in range(m):
        for j in range(n):
            if discardChannelsVec[i][j] == 1 and is_grid[i][j] == 1:
                if j > 0 and not np.isnan(amp_map[i][j - 1]):
                    new_map[i, j] = amp_map[i, j - 1]
                elif j == 0 and not np.isnan(amp_map[i][j + 1]):
                    new_map[i, j] = amp_map[i, j + 1]
                elif j > 0 and j + 1 < n and not np.isnan(amp_map[i][j + 1]):
                    new_map[i, j] = amp_map[i, j + 1]
                elif j > 1 and not np.isnan(amp_map[i][j - 2]):
                    new_map[i, j] = amp_map[i, j - 2]
                else:
                    print(f"Could not fill discarded channel at ({i}, {j})")
    return new_map


# ===========================================================================
# 3. Sub-panel 1B: exemplary raw EMG channels
# ===========================================================================
def plot_raw_channels(emg, fsamp, n_channels=32):
    """
    Draw sub-panel 1B: ``n_channels`` exemplary raw-EMG channels, band-pass
    (20-500 Hz) + 50 Hz notch filtered, each normalised to [-1, 1] and stacked
    with a vertical offset.

    Parameters
    ----------
    emg : ndarray, shape (n_channels_total, n_samples)
        Raw EMG (channels x samples).
    fsamp : float
        Sampling frequency in Hz.
    n_channels : int, optional
        Number of channels to display (default 32).
    """
    plt.rcParams["font.family"] = "Arial"

    # 50 Hz notch + 20-500 Hz Butterworth band-pass.
    b_notch, a_notch = iirnotch(50, 30, fsamp)
    nyq = fsamp / 2
    b_bp, a_bp = butter(4, [20 / nyq, 500 / nyq], btype="band")

    num_channels = emg.shape[0]
    time = np.arange(emg.shape[1]) / fsamp
    selected = list(range(min(n_channels, num_channels)))
    colors = CHANNEL_CMAP(np.linspace(0, 1, len(selected)))

    fig = plt.figure(figsize=(10, 7))
    for i, ch in enumerate(selected):
        channel_data = emg[ch, :]
        filtered = filtfilt(b_notch, a_notch, channel_data)
        filtered = filtfilt(b_bp, a_bp, filtered)
        ch_min, ch_max = np.min(filtered), np.max(filtered)
        if ch_max - ch_min > 0:
            normalized = 2 * (filtered - ch_min) / (ch_max - ch_min) - 1
        else:
            normalized = np.zeros_like(filtered)
        plt.plot(time, normalized + i * 1.25, color=colors[i], linewidth=0.5)

    plt.yticks([])
    plt.xlabel("Time (s)", fontsize=18)
    plt.xticks(np.arange(0, time[-1] + 10, 10), fontsize=18)
    plt.title(f"Raw EMG ({len(selected)} channels)", fontsize=20, pad=12)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)
    return fig


# ===========================================================================
# 4. Sub-panel 1C: RMS envelope + MU spike trains + firing rates
# ===========================================================================
def plot_rms_spikes_discharge(emg, MUPulses, trigger, fsamp):
    """
    Draw sub-panel 1C: the RMS envelope of the EMG (top) and, below it, one row
    per motor unit showing its spike train (dark blue) and instantaneous firing
    rate in pulses-per-second (rosa, right axis). Gray shading marks the guided
    gesture *hold* periods derived from the trigger.

    Parameters
    ----------
    emg : ndarray, shape (n_channels, n_samples)
        EMG used for the RMS envelope.
    MUPulses : ndarray of object, shape (n_mu,)
        Per-MU discharge sample indices.
    trigger : ndarray
        Kinematic ground-truth trace (defines the hold periods).
    fsamp : float
        Sampling frequency in Hz.
    """
    scale = 1.2
    size = 18 * scale
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = size
    plt.rcParams["axes.titlesize"] = size
    plt.rcParams["axes.labelsize"] = size
    plt.rcParams["xtick.labelsize"] = size - 1
    plt.rcParams["ytick.labelsize"] = size - 1
    plt.rcParams["legend.fontsize"] = size

    window = 1  # [s]  firing-rate smoothing window
    time = np.arange(emg.shape[1]) / fsamp
    rms = compute_rms(emg, fsamp, window)

    num_mus = MUPulses.shape[0]
    height_ratios = [1.5] + [1] * num_mus
    fig, axes = plt.subplots(
        num_mus + 1, 1, figsize=(14 * scale, 2 + num_mus * 1.5 * scale),
        sharex=True, gridspec_kw={"height_ratios": height_ratios})

    # ---- Top row: RMS envelope (smoothed for display) -----------------------------
    smooth_window_size = int(0.1 * fsamp)  # 100 ms
    if smooth_window_size > 1:
        kernel = np.ones(smooth_window_size) / smooth_window_size
        rms_smooth = np.convolve(rms, kernel, mode="same")
    else:
        rms_smooth = rms
    axes[0].plot(time, rms_smooth, color=DARK_BLUE, alpha=0.7, linewidth=1.5 * scale)
    axes[0].set_ylabel("RMS (mV)", color=DARK_BLUE)
    axes[0].tick_params(axis="y", labelcolor=DARK_BLUE)
    axes[0].grid(True, alpha=0.3)

    start, stop = compute_trigger_timings(rms, trigger, fsamp)
    for x1, x2 in zip(start, stop):
        axes[0].axvspan(x1, x2, color=(0.8, 0.8, 0.8), alpha=0.3, edgecolor="none")

    # ---- Gaussian kernel used to turn spike trains into firing rates --------------
    smooth_window = int(window * fsamp)
    t = np.linspace(-3, 3, smooth_window)
    gauss_kernel = np.exp(-(t ** 2))
    gauss_kernel = gauss_kernel / np.sum(gauss_kernel)

    # ---- One row per motor unit: spike train + firing rate ------------------------
    for i in range(num_mus):
        spike_times = np.zeros(emg.shape[1])
        spike_times[MUPulses[i]] = 1
        if len(MUPulses[i]) < 20:
            continue  # skip units with too few spikes

        smoothed = np.convolve(spike_times, gauss_kernel, "same")
        firing_rate_Hz = (smoothed * fsamp) / (smooth_window / fsamp)

        for x1, x2 in zip(start, stop):
            axes[i + 1].axvspan(x1, x2, color=(0.8, 0.8, 0.8), alpha=0.3,
                                edgecolor="none")

        axes[i + 1].plot(time, spike_times, color=DARK_BLUE, linewidth=0.5 * scale)
        ax_right = axes[i + 1].twinx()
        ax_right.plot(time, firing_rate_Hz, color=ROSA, linewidth=2 * scale)
        ax_right.set_ylabel("Firing Rate\n (pps)", color=ROSA, rotation=0,
                            ha="left", va="center")
        ax_right.tick_params(axis="y", labelcolor=ROSA)
        axes[i + 1].set_ylabel(f"MU {i + 1}")
        axes[i + 1].set_yticks([])
        axes[i + 1].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)
    return fig


# ===========================================================================
# 5. Sub-panel 1D: STA heatmap + MUAP shape of one selected motor unit
# ===========================================================================
def plot_mu_heatmap(SIG, mu_pulses, discardChannelsVec, fsamp, grids=GRIDS,
                    window_seconds=0.05, mu_label=1):
    """
    Draw sub-panel 1D for one selected motor unit:

        (left)  a heatmap of the peak-to-peak amplitude of the STA MUAP over the
                electrode grid -> the *spatial distribution* of the motor unit;
        (right) the STA MUAP waveform of the strongest (largest peak-to-peak)
                channel -> the *MUAP shape* of that motor unit.

    Parameters
    ----------
    SIG : ndarray of object, shape (rows, cols)
        Per-channel EMG grid from the decomposition.
    mu_pulses : array of int
        Discharge sample indices of the selected motor unit.
    discardChannelsVec : ndarray, shape (rows, cols)
        1 = discarded channel.
    fsamp : float
        Sampling frequency in Hz.
    grids : list of dict, optional
        Electrode-grid layout (see ``GRIDS``).
    window_seconds : float, optional
        STA window length in seconds (default 0.05 s).
    mu_label : int, optional
        1-based label of the motor unit, used only for the title.
    """
    plt.rcParams["font.family"] = "Arial"

    # STA MUAP per channel -> peak-to-peak amplitude map (holes filled within grids).
    mu_shapes = get_MU_shapes_single(SIG, mu_pulses, fsamp, window_seconds)
    mapP2P = compute_mapP2P(mu_shapes, discardChannelsVec)
    mapP2P = fill_heatmap_NaNs(mapP2P, discardChannelsVec, grids)

    # Strongest channel -> its STA waveform is the representative MUAP shape.
    if np.all(np.isnan(mapP2P)):
        raise ValueError("STA amplitude map is empty (no valid channels).")
    peak_i, peak_j = np.unravel_index(np.nanargmax(mapP2P), mapP2P.shape)
    peak_waveform = mu_shapes[peak_i][peak_j]
    wf_time = np.arange(len(peak_waveform)) / fsamp * 1000.0  # ms

    fig, (ax_map, ax_wave) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 0.7]})

    # ---- Left: spatial P2P heatmap ------------------------------------------------
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")   # gap columns / unfillable channels stay white
    masked = np.ma.masked_invalid(mapP2P)
    im = ax_map.imshow(masked, cmap=cmap, aspect="equal")
    ax_map.set_title(f"MU {mu_label}: P2P amplitude map", fontsize=18, pad=12)
    ax_map.set_xlabel("Grid column")
    ax_map.set_ylabel("Grid row")
    ax_map.set_xticks(np.arange(mapP2P.shape[1]))
    ax_map.set_xticklabels(np.arange(1, mapP2P.shape[1] + 1), fontsize=8)
    ax_map.set_yticks(np.arange(mapP2P.shape[0]))
    ax_map.set_yticklabels(np.arange(1, mapP2P.shape[0] + 1), fontsize=8)
    # Mark the strongest channel (source of the waveform on the right).
    ax_map.plot(peak_j, peak_i, marker="o", markersize=12, markerfacecolor="none",
                markeredgecolor=ROSA, markeredgewidth=2.5)
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label("Peak-to-peak (µV)", fontsize=14)

    # ---- Right: STA MUAP waveform of the strongest channel ------------------------
    ax_wave.plot(wf_time, peak_waveform, color=DARK_BLUE, linewidth=2)
    ax_wave.set_title(f"MUAP shape (ch. row {peak_i + 1}, col {peak_j + 1})",
                      fontsize=18, pad=12)
    ax_wave.set_xlabel("Time (ms)")
    ax_wave.set_ylabel("Amplitude (µV)")
    ax_wave.grid(True, alpha=0.3)
    for spine in ("top", "right"):
        ax_wave.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)
    return fig


# ===========================================================================
# 6. Reproduce Figure 1
# ===========================================================================
def make_figure1(pkl_path, mat_path, selected_mu=0, n_channels_1B=32):
    """
    End-to-end reproduction of Figure 1 (sub-panels 1B, 1C, 1D).

    Parameters
    ----------
    pkl_path : str
        Path to the Quattrocento ``*.pkl`` recording.
    mat_path : str
        Path to the decomposition ``*_finish.mat`` file.
    selected_mu : int, optional
        0-based index of the motor unit shown in sub-panel 1D (default 0).
    n_channels_1B : int, optional
        Number of raw channels drawn in sub-panel 1B (default 32).
    """
    # ---- Load both data sources ---------------------------------------------------
    emg, trigger, fsamp, task = load_raw_recording(pkl_path)
    SIG, MUPulses, discardChannelsVec, fsamp_dec = load_decomposition(mat_path)

    # ---- 1B: exemplary raw channels -----------------------------------------------
    plot_raw_channels(emg, fsamp, n_channels=n_channels_1B)

    # ---- 1C: RMS envelope + spike trains + firing rates ---------------------------
    plot_rms_spikes_discharge(emg, MUPulses, trigger, fsamp)

    # ---- 1D: STA heatmap + MUAP shape of the selected MU --------------------------
    plot_mu_heatmap(SIG, MUPulses[selected_mu], discardChannelsVec, fsamp_dec,
                    grids=GRIDS, mu_label=selected_mu + 1)


if __name__ == "__main__":
    import sys

    # Default recording used for Figure 1 (override via command-line arguments).
    PKL_PATH = ( ) # PATH to the Quattrocento recording pickle (*.pkl)
    MAT_PATH = ( ) # PATH to the decomposition finish file (*_finish.mat)


    make_figure1(PKL_PATH, MAT_PATH, selected_mu=0, n_channels_1B=32) #selected_mu=index of MU to be plotted, n_channels_1B=32 for the first 32 channels
