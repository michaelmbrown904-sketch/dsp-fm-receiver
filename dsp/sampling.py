"""Sampling, aliasing, and sample-rate reduction."""
import numpy as np

from .filters import fir_filter, windowed_sinc_lowpass


def alias_frequency(f, fs):
    """Apparent frequency (0 .. fs/2) of a real tone at f after sampling at fs.

    Sampling cannot tell f apart from f + k*fs or from -f + k*fs, so the tone
    folds into the first Nyquist zone [0, fs/2].
    """
    f_mod = f % fs
    return min(f_mod, fs - f_mod)


def downsample(x, factor):
    """Keep every `factor`-th sample. No filtering, so aliasing can occur."""
    return np.asarray(x)[::factor]


def decimate(x, factor, fs, cutoff=None, num_taps=101):
    """Low-pass filter, then downsample by an integer factor.

    The new rate is fs / factor, so anything above fs / (2 factor) must be
    removed first or it aliases. Default cutoff is 80% of the new Nyquist
    frequency, leaving room for the filter's transition band.
    """
    new_nyquist = fs / (2 * factor)
    cutoff = 0.8 * new_nyquist if cutoff is None else cutoff
    h = windowed_sinc_lowpass(cutoff, fs, num_taps)
    return downsample(fir_filter(x, h), factor)
