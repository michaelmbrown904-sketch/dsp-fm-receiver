"""03 - FIR low-pass filter design (windowed sinc) and filtering.

Design a 1 kHz low-pass filter, compare window choices, then use it to remove a
3 kHz interferer from a 500 Hz signal.
"""
import numpy as np

from _common import BLUE, GRAY, ORANGE, plt, save
from dsp.filters import fir_filter, frequency_response, windowed_sinc_lowpass
from dsp.signals import add_awgn, measure_snr_db, tone
from dsp.spectrum import spectrum, to_db

fs, cutoff, taps = 8_000, 1_000, 101

h_ham = windowed_sinc_lowpass(cutoff, fs, taps, window="hamming")
h_rect = windowed_sinc_lowpass(cutoff, fs, taps, window="boxcar")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.stem(np.arange(taps) - taps // 2, h_ham, basefmt=" ")
ax1.set(xlabel="n (samples from center)", ylabel="h[n]",
        title="Impulse response: windowed sinc, 101 taps")
for h, color, label in [(h_rect, ORANGE, "rectangular window"), (h_ham, BLUE, "Hamming window")]:
    f, H = frequency_response(h, fs)
    ax2.plot(f, to_db(H), color=color, label=label)
ax2.axvline(cutoff, color=GRAY, ls="--", lw=1)
ax2.set(ylim=(-100, 5), xlabel="frequency (Hz)", ylabel="|H(f)| (dB)",
        title="Frequency response")
ax2.legend()
save(fig, "03_filter_design")

f, H = frequency_response(h_ham, fs)
worst_stopband_db = to_db(H[f > 1_200]).max()   # highest sidelobe in the stopband
print(f"Hamming design: at least {-worst_stopband_db:.1f} dB attenuation above 1.2 kHz")

# Use the filter: keep 500 Hz, remove a 3 kHz interferer and some noise.
duration = 0.05
wanted = tone(500, fs, duration)
received = add_awgn(wanted + 0.8 * tone(3_000, fs, duration), snr_db=10, rng=1)
filtered = fir_filter(received, h_ham)
print(f"SNR before filtering: {measure_snr_db(wanted, received):5.1f} dB")
print(f"SNR after  filtering: {measure_snr_db(wanted, filtered, trim=taps):5.1f} dB")

t_ms = np.arange(len(wanted)) / fs * 1e3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.plot(t_ms, received, color=GRAY, lw=0.8, label="received")
ax1.plot(t_ms, filtered, color=BLUE, lw=1.5, label="filtered")
ax1.set(xlim=(10, 20), xlabel="time (ms)", ylabel="amplitude", title="Time domain")
ax1.legend(loc="upper right")
for sig, color, label in [(received, GRAY, "received"), (filtered, BLUE, "filtered")]:
    f, X = spectrum(sig, fs)
    keep = f >= 0
    ax2.plot(f[keep], to_db(2 * X[keep]), color=color, label=label)
ax2.set(ylim=(-90, 5), xlabel="frequency (Hz)", ylabel="magnitude (dB)",
        title="Frequency domain: 3 kHz interferer removed")
ax2.legend()
save(fig, "03_filtering")
