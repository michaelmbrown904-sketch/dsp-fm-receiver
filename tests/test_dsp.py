"""Unit tests: each building block is checked against theory or a reference library."""
import numpy as np
import pytest
from scipy.signal import firwin

from dsp.filters import fir_filter, frequency_response, windowed_sinc_lowpass
from dsp.modulation import am_demodulate, am_modulate, fm_demodulate, fm_modulate
from dsp.receiver import Station, expected_audio, fm_receive, synthesize_band
from dsp.sampling import alias_frequency, decimate, downsample
from dsp.signals import add_awgn, measure_snr_db, signal_power, tone
from dsp.spectrum import dft, peak_frequencies, spectrum, to_db


# ---------- signals ----------
def test_awgn_hits_requested_snr():
    x = tone(100, 10_000, 5.0)
    noisy = add_awgn(x, snr_db=10, rng=0)
    measured = 10 * np.log10(signal_power(x) / signal_power(noisy - x))
    assert measured == pytest.approx(10, abs=0.1)


def test_snr_ignores_pure_gain():
    x = tone(100, 10_000, 1.0)
    assert measure_snr_db(x, 3 * x + 1e-6 * np.sin(np.arange(len(x)))) > 100


# ---------- sampling ----------
@pytest.mark.parametrize("f, expected", [(1_000, 1_000), (6_000, 4_000), (7_000, 3_000),
                                         (12_000, 2_000), (19_000, 1_000)])
def test_alias_frequency(f, expected):
    assert alias_frequency(f, 10_000) == pytest.approx(expected)


def test_decimate_removes_tone_that_would_alias():
    fs = 8_000
    x = tone(300, fs, 1.0) + tone(3_000, fs, 1.0)  # 3 kHz is above new Nyquist (1 kHz)
    naive = downsample(x, 4)           # 3 kHz aliases to 2 kHz - 3 kHz -> 1 kHz
    proper = decimate(x, 4, fs)
    assert peak_frequencies(naive, fs / 4, n_peaks=2) == pytest.approx([300, 1_000], abs=5)
    _, X = spectrum(proper, fs / 4)
    assert np.max(np.abs(X)) == pytest.approx(0.5, abs=0.02)  # only the 300 Hz tone left
    assert peak_frequencies(proper, fs / 4) == pytest.approx([300], abs=5)


# ---------- spectrum ----------
def test_dft_matches_fft():
    x = np.random.default_rng(0).standard_normal(128) + 1j
    np.testing.assert_allclose(dft(x), np.fft.fft(x), atol=1e-9)


def test_spectrum_peak_amplitude_and_frequency():
    fs, f0 = 1_000, 125.0                  # bin-centered tone
    x = 2.0 * np.exp(2j * np.pi * f0 * np.arange(1_000) / fs)
    f, X = spectrum(x, fs)
    assert f[np.argmax(np.abs(X))] == pytest.approx(f0)
    assert np.max(np.abs(X)) == pytest.approx(2.0, rel=1e-3)


# ---------- filters ----------
def test_lowpass_matches_scipy_firwin():
    ours = windowed_sinc_lowpass(1_000, 8_000, 101, window="hamming")
    ref = firwin(101, 1_000, fs=8_000, window="hamming")
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_lowpass_passband_and_stopband():
    h = windowed_sinc_lowpass(1_000, 8_000, 101)
    f, H = frequency_response(h, 8_000)
    assert np.all(np.abs(to_db(H[f < 800])) < 0.1)    # flat passband
    assert np.all(to_db(H[f > 1_200]) < -50)          # >50 dB stopband
    np.testing.assert_allclose(h, h[::-1])             # symmetric -> linear phase


def test_filter_output_is_time_aligned():
    fs = 8_000
    x = tone(200, fs, 0.1)
    y = fir_filter(x, windowed_sinc_lowpass(1_000, fs, 101))
    np.testing.assert_allclose(y[200:-200], x[200:-200], atol=1e-3)


# ---------- modulation ----------
def test_fm_round_trip_is_exact_without_noise():
    fs = 100_000
    m = tone(1_000, fs, 0.01)
    np.testing.assert_allclose(fm_demodulate(fm_modulate(m, fs, 10e3), fs, 10e3)[1:], m[1:],
                               atol=1e-9)


def test_fm_has_constant_envelope():
    iq = fm_modulate(tone(500, 50_000, 0.01), 50_000, 5e3)
    np.testing.assert_allclose(np.abs(iq), 1.0)


def test_am_round_trip():
    fs = 50_000
    m = tone(200, fs, 0.02)
    out = am_demodulate(am_modulate(m, 5_000, fs, 0.5), 0.5)
    np.testing.assert_allclose(out[100:-100], m[100:-100], atol=1e-6)


# ---------- receiver ----------
STATIONS = [Station(-300e3, [1_000], name="A"),
            Station(0.0, [600, 2_400], name="B"),
            Station(300e3, [3_000], name="C")]


@pytest.mark.parametrize("station", STATIONS, ids=lambda s: s.name)
def test_receiver_recovers_each_station(station):
    iq, messages = synthesize_band(STATIONS, 1e6, 0.2, snr_db=20, rng=0)
    audio, audio_fs, _ = fm_receive(iq, 1e6, station)
    assert audio_fs == 50_000
    detected = peak_frequencies(audio, audio_fs, n_peaks=len(station.tones_hz))
    assert detected == pytest.approx(station.tones_hz, abs=10)
    assert measure_snr_db(expected_audio(messages[station.name]), audio, trim=200) > 40


def test_receiver_degrades_below_threshold():
    iq, messages = synthesize_band(STATIONS, 1e6, 0.2, snr_db=-5, rng=0)
    audio, _, _ = fm_receive(iq, 1e6, STATIONS[0])
    assert measure_snr_db(expected_audio(messages["A"]), audio, trim=200) < 5
