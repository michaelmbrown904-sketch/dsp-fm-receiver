"""02 - The DFT/FFT, frequency resolution, and windowing.

Part 1 checks that the direct DFT formula matches np.fft.fft.
Part 2 shows spectral leakage: a strong tone and a weak tone 50 dB below it.
With no window (rectangular), leakage from the strong tone buries the weak one.
A Hann window trades a wider main lobe for much lower sidelobes, revealing it.
"""
import numpy as np

from _common import BLUE, GRAY, ORANGE, plt, save
from dsp.signals import tone
from dsp.spectrum import dft, spectrum, to_db

# Part 1: DFT from the definition equals the FFT.
x = np.random.default_rng(0).standard_normal(64)
err = np.max(np.abs(dft(x) - np.fft.fft(x)))
print(f"max |DFT - FFT| on 64 random samples: {err:.2e}")

# Part 2: leakage.
fs, N = 8_000, 256
duration = N / fs
print(f"bin spacing fs/N = {fs / N:.2f} Hz")
f_strong, f_weak = 1_015.0, 1_300.0   # not centered on FFT bins -> leakage
x = tone(f_strong, fs, duration) + tone(f_weak, fs, duration, amplitude=10 ** (-50 / 20))

fig, ax = plt.subplots(figsize=(9, 4))
for window, color, label in [("boxcar", ORANGE, "rectangular (no window)"),
                             ("hann", BLUE, "Hann window")]:
    f, X = spectrum(x, fs, window=window, nfft=8 * N)   # zero-pad to see the shape
    keep = f >= 0
    ax.plot(f[keep], to_db(2 * X[keep]), color=color, label=label)  # x2: one-sided
ax.axvline(f_weak, color=GRAY, ls="--", lw=1)
ax.annotate("weak tone, -50 dB", (f_weak, -50), (1_900, -25), arrowprops=dict(arrowstyle="->"))
ax.set(xlim=(0, 4_000), ylim=(-110, 5), xlabel="frequency (Hz)", ylabel="magnitude (dB)",
       title="Spectral leakage: windowing reveals a tone 50 dB below a strong neighbour")
ax.legend()
save(fig, "02_windowing")
