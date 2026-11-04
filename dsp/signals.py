"""Test-signal generation and noise."""
import numpy as np


def time_vector(fs, duration):
    """Sample times t[n] = n / fs for n = 0 .. N-1, where N = fs * duration."""
    n_samples = int(round(fs * duration))
    return np.arange(n_samples) / fs


def tone(freq, fs, duration, amplitude=1.0, phase=0.0):
    """Real sinusoid x[n] = A cos(2*pi*f*n/fs + phase)."""
    t = time_vector(fs, duration)
    return amplitude * np.cos(2 * np.pi * freq * t + phase)


def multitone(freqs, fs, duration, amplitudes=None):
    """Sum of cosines, normalized so the peak magnitude is 1."""
    amplitudes = np.ones(len(freqs)) if amplitudes is None else amplitudes
    x = sum(tone(f, fs, duration, a) for f, a in zip(freqs, amplitudes))
    return x / np.max(np.abs(x))


def signal_power(x):
    """Average power: mean of |x[n]|^2."""
    return np.mean(np.abs(x) ** 2)


def add_awgn(x, snr_db, rng=None):
    """Add white Gaussian noise so that power(x) / power(noise) = snr_db.

    For complex signals the noise power is split evenly between I and Q.
    """
    rng = np.random.default_rng(rng)
    noise_power = signal_power(x) / 10 ** (snr_db / 10)
    if np.iscomplexobj(x):
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
        )
    else:
        noise = np.sqrt(noise_power) * rng.standard_normal(x.shape)
    return x + noise


def measure_snr_db(reference, estimate, trim=0):
    """SNR of `estimate` relative to `reference`, in dB.

    A least-squares gain g is fitted first, so a pure scale difference is not
    counted as error:  SNR = power(g * ref) / power(estimate - g * ref).
    `trim` drops samples at each end (filter start-up transients).
    """
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    if trim:
        ref, est = ref[trim:-trim], est[trim:-trim]
    g = np.dot(est, ref) / np.dot(ref, ref)
    error = est - g * ref
    return 10 * np.log10(signal_power(g * ref) / signal_power(error))
