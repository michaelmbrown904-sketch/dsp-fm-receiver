"""AM and FM modulation and demodulation."""
import numpy as np
from scipy.signal import hilbert


def am_modulate(message, fc, fs, mod_index=0.5):
    """Real AM signal: s[n] = (1 + mu * m[n]) cos(2 pi fc n / fs), with |m| <= 1."""
    n = np.arange(len(message))
    return (1 + mod_index * message) * np.cos(2 * np.pi * fc * n / fs)


def am_demodulate(s, mod_index=0.5):
    """Envelope detector using the analytic signal.

    hilbert(s) = s + j*H{s} has magnitude equal to the envelope 1 + mu*m[n].
    """
    envelope = np.abs(hilbert(s))
    return (envelope - np.mean(envelope)) / mod_index


def fm_modulate(message, fs, freq_dev):
    """Complex-baseband FM: s[n] = exp(j phi[n]).

    Instantaneous frequency is f[n] = freq_dev * m[n], and phase is the running
    integral of frequency:  phi[n] = 2 pi / fs * sum_{k<=n} freq_dev * m[k].
    """
    phase = 2 * np.pi * freq_dev * np.cumsum(message) / fs
    return np.exp(1j * phase)


def fm_demodulate(iq, fs, freq_dev):
    """Polar discriminator: recover frequency as phase change per sample.

    s[n] * conj(s[n-1]) = |.|^2 exp(j (phi[n] - phi[n-1])), so its angle is
    the phase step dphi[n]. Frequency = dphi * fs / (2 pi). Dividing by
    freq_dev returns the original message scale.
    """
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    inst_freq = dphi * fs / (2 * np.pi)
    out = inst_freq / freq_dev
    return np.concatenate(([out[0]], out))  # keep the same length as the input
