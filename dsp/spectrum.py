"""Frequency-domain analysis: DFT, FFT, windowing."""
import numpy as np
from scipy.signal import get_window, welch


def dft(x):
    """Direct DFT from the definition, O(N^2).

        X[k] = sum_{n=0}^{N-1} x[n] * exp(-j 2 pi k n / N)

    Written out for learning; np.fft.fft computes the same thing in O(N log N).
    """
    x = np.asarray(x)
    n = np.arange(len(x))
    k = n.reshape(-1, 1)
    W = np.exp(-2j * np.pi * k * n / len(x))
    return W @ x


def spectrum(x, fs, window="hann", nfft=None):
    """Windowed, centered spectrum.

    Returns (freqs, X) with freqs running from -fs/2 to +fs/2.
    Scaled by sum(window) so a complex exponential of amplitude A peaks at A
    (a real cosine of amplitude A shows two peaks of A/2 at +f and -f).
    """
    x = np.asarray(x)
    nfft = nfft or len(x)
    w = get_window(window, len(x))
    X = np.fft.fftshift(np.fft.fft(x * w, nfft)) / np.sum(w)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / fs))
    return freqs, X


def averaged_psd_db(x, fs, nperseg=2048):
    """Welch power spectral density, centered, in dB relative to its peak.

    Splits x into overlapping segments, windows each, and averages their
    |FFT|^2. Averaging trades frequency resolution for a much smoother,
    lower-variance estimate, which is what you want for a long noisy capture.
    """
    freqs, P = welch(x, fs, nperseg=nperseg, return_onesided=False)
    freqs, P = np.fft.fftshift(freqs), np.fft.fftshift(P)
    return freqs, 10 * np.log10(P / P.max())


def to_db(x, floor=1e-12):
    """Magnitude in decibels: 20 log10 |x|."""
    return 20 * np.log10(np.maximum(np.abs(x), floor))


def peak_frequencies(x, fs, n_peaks=1, min_separation_hz=50.0):
    """Frequencies of the strongest spectral peaks of a real signal (positive side)."""
    freqs, X = spectrum(x, fs)
    keep = freqs >= 0
    freqs, mag = freqs[keep], np.abs(X[keep])
    found = []
    for idx in np.argsort(mag)[::-1]:
        if all(abs(freqs[idx] - f) > min_separation_hz for f in found):
            found.append(freqs[idx])
        if len(found) == n_peaks:
            break
    return sorted(found)
