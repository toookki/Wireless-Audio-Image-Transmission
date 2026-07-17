"""Lectura y escritura de archivos WAV mono.

El receptor acepta WAV a distintas tasas de muestreo y los remuestrea a la
tasa interna del módem. Esto facilita usar grabaciones de celulares o de otros
computadores, que con frecuencia trabajan a 48 kHz.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def _to_float_mono(data: np.ndarray) -> np.ndarray:
    """Convierte audio entero/flotante y estéreo/mono a float64 mono."""

    if data.ndim == 2:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), info.max)
        audio = data.astype(np.float64) / scale
    else:
        audio = data.astype(np.float64)

    if not np.all(np.isfinite(audio)):
        raise ValueError("El archivo contiene NaN o valores infinitos.")
    return audio


def read_wav_mono(path: str | Path, target_sample_rate: int) -> np.ndarray:
    """Lee WAV, convierte a mono y remuestrea cuando es necesario."""

    sample_rate, data = wavfile.read(path)
    audio = _to_float_mono(data)

    if sample_rate != target_sample_rate:
        divisor = gcd(sample_rate, target_sample_rate)
        up = target_sample_rate // divisor
        down = sample_rate // divisor
        audio = resample_poly(audio, up, down)

    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak
    return audio


def write_wav_mono(path: str | Path, sample_rate: int, audio: np.ndarray) -> None:
    """Guarda audio mono en PCM de 16 bits."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float64)
    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak

    pcm = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, pcm)