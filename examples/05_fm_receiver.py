"""05 - Practical application: a multi-station FM receiver.

Scenario: an SDR centered on some RF frequency delivers 1 MHz of complex (I/Q)
samples. Three FM stations sit in that band at -300, 0 and +300 kHz, each
carrying different audio tones. The receiver tunes to each station, isolates
it, demodulates it, and identifies the tones it was carrying.

Then the noise is swept to measure the FM threshold effect.
"""
import numpy as np
from scipy.signal import spectrogram

from _common import BLUE, GRAY, GREEN, ORANGE, plt, save
from dsp.receiver import Station, expected_audio, fm_receive, synthesize_band
from dsp.signals import measure_snr_db
from dsp.spectrum import averaged_psd_db, peak_frequencies

FS = 1e6           # SDR sample rate: 1 MHz of bandwidth
DURATION = 0.5     # seconds
SNR_DB = 20        # signal-to-noise ratio over the whole 1 MHz capture
STATIONS = [
    Station(-300e3, [1_000], name="Station A"),
    Station(0.0, [600, 2_400], name="Station B"),
    Station(+300e3, [3_000], name="Station C"),
]
COLORS = [BLUE, ORANGE, GREEN]

iq, messages = synthesize_band(STATIONS, FS, DURATION, SNR_DB, rng=1)

# ---- Figure 1: what the SDR sees ----
f, t, Sxx = spectrogram(iq, FS, nperseg=1024, return_onesided=False)
f, Sxx = np.fft.fftshift(f), np.fft.fftshift(Sxx, axes=0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.pcolormesh(f / 1e3, t * 1e3, 10 * np.log10(Sxx.T + 1e-15), shading="auto", cmap="viridis")
ax1.set(xlabel="frequency offset (kHz)", ylabel="time (ms)", title="Spectrogram of 1 MHz capture")
fw, P = averaged_psd_db(iq, FS)
ax2.plot(fw / 1e3, P, color=GRAY, lw=0.8)
for st, color in zip(STATIONS, COLORS):
    ax2.axvspan((st.offset_hz - 78e3) / 1e3, (st.offset_hz + 78e3) / 1e3, color=color, alpha=0.15)
    ax2.text(st.offset_hz / 1e3, 2, st.name, ha="center", color=color, fontweight="bold")
ax2.set(ylim=(-50, 6), xlabel="frequency offset (kHz)", ylabel="power (dB, relative)",
        title="Averaged spectrum: three FM stations in noise")
save(fig, "05_wideband_capture")

# ---- Figure 2: the receiver chain, step by step, for Station A ----
target = STATIONS[0]
audio, audio_fs, stages = fm_receive(iq, FS, target)
titles = {
    "wideband": "1. Wideband input (fs = 1 MHz)",
    "tuned": "2. Mixed: Station A moved to 0 Hz",
    "channel": "3. Low-pass + decimate by 5 (fs = 200 kHz)",
    "audio": "4. Demodulated, filtered, decimate by 4 (fs = 50 kHz)",
}
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
for ax, key in zip(axes.flat, titles):
    sig, rate = stages[key]
    fk, Pk = averaged_psd_db(sig, rate)
    ax.plot(fk / 1e3, Pk, color=BLUE, lw=0.8)
    ax.set(title=titles[key], xlabel="frequency (kHz)", ylabel="power (dB, relative)",
           ylim=(-90, 5))
axes.flat[3].set_xlim(0, 10)
save(fig, "05_receiver_chain")

# ---- Recover every station and identify its tones ----
print(f"Input SNR over the 1 MHz capture: {SNR_DB} dB\n")
print(f"{'station':<10}{'offset':>10}{'sent tones (Hz)':>20}{'detected (Hz)':>20}{'audio SNR':>12}")
fig, axes = plt.subplots(len(STATIONS), 1, figsize=(9, 6.5), sharex=True)
for ax, st, color in zip(axes, STATIONS, COLORS):
    audio, audio_fs, _ = fm_receive(iq, FS, st)
    ideal = expected_audio(messages[st.name])
    detected = peak_frequencies(audio, audio_fs, n_peaks=len(st.tones_hz))
    snr = measure_snr_db(ideal, audio, trim=200)
    print(f"{st.name:<10}{st.offset_hz / 1e3:>8.0f} k{str(st.tones_hz):>20}"
          f"{str([round(d) for d in detected]):>20}{snr:>10.1f} dB")
    t_ms = np.arange(len(audio)) / audio_fs * 1e3
    ax.plot(t_ms, ideal, color=GRAY, lw=3, alpha=0.5, label="transmitted audio")
    ax.plot(t_ms, audio, color=color, lw=1, label=f"recovered ({snr:.0f} dB SNR)")
    ax.set(xlim=(100, 105), ylabel=st.name)
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)
axes[-1].set_xlabel("time (ms)")
axes[0].set_title("Recovered audio for each station")
save(fig, "05_recovered_audio")

# ---- Figure 4: FM threshold effect ----
input_snrs = np.arange(-15, 31, 2.5)
fig, ax = plt.subplots(figsize=(9, 4))
for st, color in zip(STATIONS, COLORS):
    out = []
    for snr_in in input_snrs:
        trials = []
        for seed in range(3):
            iq_n, msgs = synthesize_band(STATIONS, FS, 0.2, snr_in, rng=seed)
            audio, _, _ = fm_receive(iq_n, FS, st)
            trials.append(measure_snr_db(expected_audio(msgs[st.name]), audio, trim=200))
        out.append(np.mean(trials))
    ax.plot(input_snrs, out, "o-", color=color, ms=4, label=st.name)
ax.axvspan(0, 10, color=GRAY, alpha=0.12)
ax.text(5, 50, "threshold\nregion", ha="center", color=GRAY)
ax.set(xlabel="input SNR over 1 MHz capture (dB)", ylabel="recovered audio SNR (dB)",
       title="FM threshold: audio quality collapses once input SNR drops below ~10 dB")
ax.legend(loc="upper left")
save(fig, "05_fm_threshold")
