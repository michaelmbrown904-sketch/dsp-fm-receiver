"""FIR filter design and filtering."""
import numpy as np
from scipy.signal import get_window


def windowed_sinc_lowpass(cutoff, fs, num_taps=101, window="hamming"):
    """Linear-phase FIR low-pass filter by the windowed-sinc method.

    The ideal low-pass filter with cutoff fc (cycles/sample) has impulse response
        h_ideal[n] = 2 fc sinc(2 fc n),     -inf < n < inf
    which is infinitely long. We keep `num_taps` samples centered on n = 0 and
    taper them with a window to reduce ripple (Gibbs phenomenon).

    num_taps must be odd so the filter is symmetric about an integer sample,
    giving an integer group delay of (num_taps - 1) / 2 samples.
    """
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd")
    if not 0 < cutoff < fs / 2:
        raise ValueError("cutoff must be between 0 and fs/2")
    fc = cutoff / fs
    n = np.arange(num_taps) - (num_taps - 1) / 2
    h = 2 * fc * np.sinc(2 * fc * n)
    h *= get_window(window, num_taps, fftbins=False)
    return h / np.sum(h)  # unity gain at DC


def fir_filter(x, h):
    """Apply FIR filter h to x by convolution: y[n] = sum_k h[k] x[n-k].

    mode="same" returns len(x) samples and, for an odd symmetric h, removes the
    (num_taps-1)/2 sample group delay so input and output line up in time.
    """
    return np.convolve(x, h, mode="same")


def frequency_response(h, fs, n_points=4096):
    """Frequency response H(f) on 0 .. fs/2 via a zero-padded FFT of h."""
    H = np.fft.rfft(h, 2 * n_points)
    freqs = np.fft.rfftfreq(2 * n_points, d=1 / fs)
    return freqs, H
