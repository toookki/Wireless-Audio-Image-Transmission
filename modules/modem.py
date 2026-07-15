"""Módem 4-FSK y multiplexación en frecuencia (FDM).

Flujo de transmisión:
1. Se agrega una secuencia de preámbulo a cada canal.
2. Cada símbolo 0..3 selecciona uno de cuatro tonos.
3. El símbolo se repite para obtener decisión por energía acumulada.
4. Los cuatro subcanales se suman en una única señal audible.

Flujo de recepción:
1. Se localiza un chirp de sincronización por correlación normalizada.
2. Se divide la señal en ventanas de símbolo.
3. Para cada tono se calcula energía de correlación I/Q.
4. Se decide el tono de mayor energía y se valida el preámbulo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from config.settings import (
    CHANNEL_AMPLITUDE,
    CHANNELS,
    GUARD_DURATION_S,
    LEADING_SILENCE_S,
    OUTPUT_PEAK,
    PREAMBLE_SYMBOLS,
    REPETITION_FACTOR,
    SAMPLE_RATE,
    SAMPLES_PER_SLOT,
    SYNC_DURATION_S,
    SYNC_END_HZ,
    SYNC_MIN_CORRELATION,
    SYNC_START_HZ,
    TRAILING_SILENCE_S,
    ChannelConfig,
)


@dataclass(frozen=True)
class SyncResult:
    """Resultado de la detección del chirp inicial."""

    sync_start_sample: int
    data_start_sample: int
    correlation: float


@dataclass(frozen=True)
class ChannelDemodulation:
    """Símbolos recuperados y diagnóstico básico de un subcanal."""

    symbols: np.ndarray
    preamble_score: float
    mean_decision_margin: float


def generate_sync_chirp() -> np.ndarray:
    """Genera el chirp conocido usado para localizar el inicio de la trama."""

    sample_count = int(round(SYNC_DURATION_S * SAMPLE_RATE))
    t = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    sweep_rate = (SYNC_END_HZ - SYNC_START_HZ) / SYNC_DURATION_S
    phase = 2.0 * np.pi * (
        SYNC_START_HZ * t + 0.5 * sweep_rate * t * t
    )
    chirp = np.sin(phase)

    # Pequeño fade para evitar clicks al reproducir.
    fade_samples = min(int(0.02 * SAMPLE_RATE), sample_count // 2)
    if fade_samples > 0:
        fade = np.sin(np.linspace(0.0, np.pi / 2.0, fade_samples)) ** 2
        chirp[:fade_samples] *= fade
        chirp[-fade_samples:] *= fade[::-1]
    return chirp


def _modulate_channel(symbols: np.ndarray, channel: ChannelConfig) -> np.ndarray:
    """Modula un subcanal con fase continua."""

    logical = np.concatenate(
        [np.asarray(PREAMBLE_SYMBOLS, dtype=np.uint8), np.asarray(symbols, dtype=np.uint8)]
    )
    repeated = np.repeat(logical, REPETITION_FACTOR)
    frequencies = np.asarray(channel.frequencies_hz, dtype=np.float64)[repeated]
    sample_frequencies = np.repeat(frequencies, SAMPLES_PER_SLOT)

    phase_steps = 2.0 * np.pi * sample_frequencies / SAMPLE_RATE
    phase = np.cumsum(phase_steps)
    return CHANNEL_AMPLITUDE * np.sin(phase)


def build_fdm_waveform(channel_symbols: dict[int, np.ndarray]) -> np.ndarray:
    """Construye la señal completa: silencio, sincronía, guarda y datos FDM."""

    channel_waves: list[np.ndarray] = []
    for channel in CHANNELS:
        if channel.channel_id not in channel_symbols:
            raise ValueError(f"Faltan símbolos para el canal {channel.channel_id}.")
        channel_waves.append(_modulate_channel(channel_symbols[channel.channel_id], channel))

    max_length = max(len(wave) for wave in channel_waves)
    data_wave = np.zeros(max_length, dtype=np.float64)
    for wave in channel_waves:
        data_wave[: len(wave)] += wave

    leading = np.zeros(int(round(LEADING_SILENCE_S * SAMPLE_RATE)))
    guard = np.zeros(int(round(GUARD_DURATION_S * SAMPLE_RATE)))
    trailing = np.zeros(int(round(TRAILING_SILENCE_S * SAMPLE_RATE)))
    sync = generate_sync_chirp() * 0.75

    complete = np.concatenate([leading, sync, guard, data_wave, trailing])
    peak = np.max(np.abs(complete))
    if peak > 0:
        complete = complete * (OUTPUT_PEAK / peak)
    return complete


def find_sync(audio: np.ndarray) -> SyncResult:
    """Localiza el chirp usando correlación normalizada de filtro adaptado."""

    sync = generate_sync_chirp()
    if len(audio) < len(sync):
        raise ValueError("El audio es más corto que la señal de sincronización.")

    correlation = fftconvolve(audio, sync[::-1], mode="valid")
    local_energy = fftconvolve(audio * audio, np.ones(len(sync)), mode="valid")
    denominator = np.sqrt(np.maximum(local_energy * np.sum(sync * sync), 1e-15))
    normalized = np.abs(correlation) / denominator

    sync_start = int(np.argmax(normalized))
    score = float(normalized[sync_start])
    if score < SYNC_MIN_CORRELATION:
        raise ValueError(
            f"No se encontró sincronización confiable: correlación={score:.3f}."
        )

    data_start = sync_start + len(sync) + int(round(GUARD_DURATION_S * SAMPLE_RATE))
    return SyncResult(sync_start, data_start, score)


def _slot_energies(slots: np.ndarray, channel: ChannelConfig) -> np.ndarray:
    """Calcula energía de correlación para los cuatro tonos de un canal."""

    n = np.arange(SAMPLES_PER_SLOT, dtype=np.float64)
    window = np.hanning(SAMPLES_PER_SLOT)
    windowed_slots = slots * window

    energies = np.empty((len(slots), 4), dtype=np.float64)
    for tone_index, frequency in enumerate(channel.frequencies_hz):
        angle = 2.0 * np.pi * frequency * n / SAMPLE_RATE
        cosine = np.cos(angle)
        sine = np.sin(angle)
        i_component = windowed_slots @ cosine
        q_component = windowed_slots @ sine
        energies[:, tone_index] = i_component * i_component + q_component * q_component
    return energies


def demodulate_channel(
    data_audio: np.ndarray,
    channel: ChannelConfig,
) -> ChannelDemodulation:
    """Demodula todos los símbolos disponibles de un subcanal."""

    slot_count = len(data_audio) // SAMPLES_PER_SLOT
    slot_count -= slot_count % REPETITION_FACTOR
    if slot_count <= 0:
        raise ValueError("No hay suficientes muestras para demodular.")

    slots = data_audio[: slot_count * SAMPLES_PER_SLOT].reshape(
        slot_count, SAMPLES_PER_SLOT
    )
    energies = _slot_energies(slots, channel)

    logical_count = slot_count // REPETITION_FACTOR
    combined = energies.reshape(logical_count, REPETITION_FACTOR, 4).sum(axis=1)
    decisions = np.argmax(combined, axis=1).astype(np.uint8)

    ordered = np.sort(combined, axis=1)
    margins = (ordered[:, -1] - ordered[:, -2]) / np.maximum(ordered[:, -1], 1e-15)

    preamble_length = len(PREAMBLE_SYMBOLS)
    if len(decisions) < preamble_length:
        raise ValueError("La grabación no contiene el preámbulo completo.")

    expected = np.asarray(PREAMBLE_SYMBOLS, dtype=np.uint8)
    score = float(np.mean(decisions[:preamble_length] == expected))
    payload_symbols = decisions[preamble_length:]

    return ChannelDemodulation(
        symbols=payload_symbols,
        preamble_score=score,
        mean_decision_margin=float(np.mean(margins)),
    )
