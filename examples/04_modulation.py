"""04 - AM and FM modulation.

AM puts the message in the carrier's amplitude; FM puts it in the carrier's
instantaneous frequency. FM's bandwidth follows Carson's rule:
    B ~ 2 (freq_dev + f_message)
"""
import numpy as np

from _common import BLUE, GRAY, ORANGE, plt, save
from dsp.modulation import am_demodulate, am_modulate, fm_demodulate, fm_modulate
from dsp.signals import tone
from dsp.spectrum import spectrum, to_db

# --- AM: 200 Hz message on a 5 kHz carrier ---
fs, duration = 50_000, 0.02
message = tone(200, fs, duration)
am = am_modulate(message, fc=5_000, fs=fs, mod_index=0.5)
recovered = am_demodulate(am, mod_index=0.5)
print(f"AM noiseless demod max error: {np.max(np.abs(recovered - message)[100:-100]):.1e}")

t_ms = np.arange(len(message)) / fs * 1e3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(t_ms, am, color=GRAY, lw=0.6, label="AM signal")
ax.plot(t_ms, 1 + 0.5 * message, color=BLUE, lw=2, label="envelope = 1 + 0.5 m(t)")
ax.set(xlabel="time (ms)", ylabel="amplitude",
       title="AM: the message rides on the carrier's envelope")
ax.legend(loc="upper right")
save(fig, "04_am")

# --- FM: 1 kHz message, two deviations, complex baseband ---
fs, duration, f_msg = 500_000, 0.05, 1_000
message = tone(f_msg, fs, duration)
fig, ax = plt.subplots(figsize=(9, 3.8))
for dev, color in [(5e3, ORANGE), (50e3, BLUE)]:
    iq = fm_modulate(message, fs, dev)
    f, X = spectrum(iq, fs)
    carson = 2 * (dev + f_msg)
    ax.plot(f / 1e3, to_db(X), color=color, lw=0.9,
            label=f"deviation {dev / 1e3:.0f} kHz  (Carson BW = {carson / 1e3:.0f} kHz)")
    ax.axvspan(-carson / 2e3, carson / 2e3, color=color, alpha=0.08)
    max_err = np.max(np.abs(fm_demodulate(iq, fs, dev)[1:] - message[1:]))
    print(f"FM deviation {dev / 1e3:>4.0f} kHz: Carson BW {carson / 1e3:5.0f} kHz, "
          f"noiseless demod max error {max_err:.1e}")
ax.set(xlim=(-80, 80), ylim=(-80, 0), xlabel="frequency (kHz)", ylabel="magnitude (dB)",
       title="FM spectrum: larger deviation spreads power over more bandwidth")
ax.legend(loc="lower center")
save(fig, "04_fm_spectrum")
