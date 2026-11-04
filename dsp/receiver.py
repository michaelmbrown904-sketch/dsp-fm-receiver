"""Practical application: a simulated multi-station FM receiver.

An SDR delivers a wideband stream of complex (I/Q) samples containing several
signals at once. To listen to one station, the receiver must

    1. tune     : shift the wanted station to 0 Hz (complex mixing)
    2. select   : low-pass filter away the neighbours, then decimate
    3. demodulate: turn phase changes back into the audio message
    4. clean up : filter to the audio band and decimate to an audio rate

This module synthesizes such a wideband capture and implements that chain.
"""
from dataclasses import dataclass, field

import numpy as np

from .modulation import fm_demodulate, fm_modulate
from .sampling import decimate
from .signals import add_awgn, multitone, time_vector


@dataclass
class Station:
    offset_hz: float              # carrier offset from the SDR's center frequency
    tones_hz: list                # audio content: a sum of tones
    freq_dev: float = 75e3        # peak frequency deviation (broadcast FM: 75 kHz)
    name: str = field(default="")


def synthesize_band(stations, fs, duration, snr_db, rng=None):
    """Build a wideband complex-baseband capture of several FM stations plus noise.

    Returns (iq, messages), where messages[name] is each station's audio at rate fs.
    """
    t = time_vector(fs, duration)
    iq = np.zeros(len(t), dtype=complex)
    messages = {}
    for st in stations:
        msg = multitone(st.tones_hz, fs, duration)
        messages[st.name] = msg
        # Modulate at baseband, then mix up to the station's offset frequency.
        iq += fm_modulate(msg, fs, st.freq_dev) * np.exp(2j * np.pi * st.offset_hz * t)
    return add_awgn(iq, snr_db, rng), messages


def frequency_shift(x, shift_hz, fs):
    """Multiply by exp(-j 2 pi shift n / fs): moves content at +shift_hz to 0 Hz."""
    n = np.arange(len(x))
    return x * np.exp(-2j * np.pi * shift_hz * n / fs)


def expected_audio(message, channel_decim=5, audio_decim=4):
    """What a perfect receiver should output, for measuring performance.

    After decimating by D, the discriminator measures the phase step over D
    original samples, i.e. the AVERAGE frequency over m[n-D+1] .. m[n]. So the
    ideal output is a causal D-sample moving average of the message, sampled
    at the audio rate.
    """
    D = channel_decim
    averaged = np.convolve(message, np.ones(D) / D)[: len(message)]
    return averaged[:: D * audio_decim]


def fm_receive(iq, fs, station, channel_cutoff=90e3, audio_cutoff=15e3,
               channel_decim=5, audio_decim=4):
    """Tune to `station` and recover its audio.

    Returns (audio, audio_fs, stages), where `stages` holds the intermediate
    signals and rates so every step can be plotted and inspected.
    """
    stages = {"wideband": (iq, fs)}

    tuned = frequency_shift(iq, station.offset_hz, fs)
    stages["tuned"] = (tuned, fs)

    channel = decimate(tuned, channel_decim, fs, cutoff=channel_cutoff)
    channel_fs = fs / channel_decim
    stages["channel"] = (channel, channel_fs)

    demod = fm_demodulate(channel, channel_fs, station.freq_dev)
    stages["demod"] = (demod, channel_fs)

    audio = decimate(demod, audio_decim, channel_fs, cutoff=audio_cutoff)
    audio_fs = channel_fs / audio_decim
    stages["audio"] = (audio, audio_fs)
    return audio, audio_fs, stages
