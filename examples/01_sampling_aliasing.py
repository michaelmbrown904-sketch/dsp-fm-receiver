"""01 - Sampling and aliasing.

A 7 kHz tone sampled at 10 kHz produces exactly the same samples as a 3 kHz
tone. Once sampled, the two cannot be told apart: 7 kHz has aliased to
fs - 7 kHz = 3 kHz because it is above the Nyquist frequency fs/2 = 5 kHz.
"""
import numpy as np

from _common import BLUE, ORANGE, plt, save
from dsp.sampling import alias_frequency

fs = 10_000            # sample rate, Hz
f_low, f_high = 3_000, 7_000
duration = 2e-3        # 2 ms

# A very finely sampled signal stands in for "continuous time".
t_fine = np.linspace(0, duration, 4000)
n = np.arange(int(fs * duration))
t_samples = n / fs

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(t_fine * 1e3, np.cos(2 * np.pi * f_low * t_fine), color=BLUE, label="3 kHz tone")
ax.plot(t_fine * 1e3, np.cos(2 * np.pi * f_high * t_fine), color=ORANGE, alpha=0.6,
        label="7 kHz tone")
ax.plot(t_samples * 1e3, np.cos(2 * np.pi * f_high * t_samples), "ko", ms=5,
        label="samples at fs = 10 kHz (identical for both)")
ax.set(xlabel="time (ms)", ylabel="amplitude",
       title="Aliasing: 7 kHz sampled at 10 kHz is indistinguishable from 3 kHz")
ax.legend(loc="upper right")
save(fig, "01_aliasing_time")

# Apparent frequency as the true frequency sweeps past Nyquist.
f_true = np.linspace(0, 3 * fs, 1000)
f_apparent = [alias_frequency(f, fs) for f in f_true]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(f_true / 1e3, np.array(f_apparent) / 1e3, color=BLUE)
for k in range(1, 6):
    ax.axvline(k * fs / 2e3, color="k", lw=0.6, ls=":")
ax.plot([f_high / 1e3], [alias_frequency(f_high, fs) / 1e3], "o", color=ORANGE)
ax.annotate("7 kHz appears at 3 kHz", (7, 3), (9, 4.2), arrowprops=dict(arrowstyle="->"))
ax.set(xlabel="true frequency (kHz)", ylabel="apparent frequency (kHz)",
       title="Frequency folding: every tone lands in 0 .. fs/2 (dotted lines = multiples of fs/2)")
save(fig, "01_aliasing_folding")

print("True f -> apparent f at fs = 10 kHz")
for f in [1_000, 4_000, 6_000, 7_000, 12_000, 19_000]:
    print(f"  {f:>6} Hz -> {alias_frequency(f, fs):>6.0f} Hz")
