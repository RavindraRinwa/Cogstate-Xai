"""EEG -> Differential Entropy (DE) features per (channel, band) for each 3 s window.
DE of a band-passed Gaussian signal = 0.5*log(2*pi*e*variance)  (standard in EEG emotion work)."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from config import *


def bandpass(x, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def window_de(w):
    """w: (..., channels, WIN) -> (..., channels, n_bands)"""
    de = [0.5 * np.log(2 * np.pi * np.e * np.var(bandpass(w, lo, hi), axis=-1) + 1e-12)
          for lo, hi in BANDS.values()]
    return np.stack(de, axis=-1)


def trial_de_windows(trial):
    """trial: (channels, T) -> (n_windows, channels, n_bands)"""
    n = (trial.shape[1] - WIN) // STEP + 1
    w = np.stack([trial[:, i * STEP:i * STEP + WIN] for i in range(n)])
    return window_de(w).astype(np.float32)
