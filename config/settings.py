"""Parámetros centrales del sistema de comunicación.

Este archivo concentra las decisiones de diseño. Durante el desarrollo conviene
modificar parámetros aquí, en vez de buscar constantes repartidas por el código.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelConfig:
    """Describe un subcanal 4-FSK del sistema FDM."""

    channel_id: int
    name: str
    frequencies_hz: tuple[float, float, float, float]


# Audio y temporización -------------------------------------------------------
SAMPLE_RATE = 44_100
SLOT_DURATION_S = 0.010  # Cada observación elemental dura 10 ms.
SAMPLES_PER_SLOT = int(SAMPLE_RATE * SLOT_DURATION_S)
REPETITION_FACTOR = 5  # Cada símbolo 4-FSK se transmite cinco veces.

LEADING_SILENCE_S = 0.50
SYNC_DURATION_S = 0.70
GUARD_DURATION_S = 0.20
TRAILING_SILENCE_S = 0.50

SYNC_START_HZ = 500.0
SYNC_END_HZ = 4_800.0
SYNC_MIN_CORRELATION = 0.12

# Imagen usada por la alternativa M-FSK del enunciado.
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20

# La separación de 200 Hz facilita distinguir tonos en ventanas de 10 ms.
CHANNELS: tuple[ChannelConfig, ...] = (
    ChannelConfig(0, "red", (900.0, 1_100.0, 1_300.0, 1_500.0)),
    ChannelConfig(1, "green", (1_900.0, 2_100.0, 2_300.0, 2_500.0)),
    ChannelConfig(2, "blue", (2_900.0, 3_100.0, 3_300.0, 3_500.0)),
    ChannelConfig(3, "text", (3_900.0, 4_100.0, 4_300.0, 4_500.0)),
)

CHANNEL_BY_ID = {channel.channel_id: channel for channel in CHANNELS}
CHANNEL_BY_NAME = {channel.name: channel for channel in CHANNELS}

# Secuencia conocida que permite verificar la alineación de símbolos.
PREAMBLE_SYMBOLS: tuple[int, ...] = tuple(
    [0, 1, 2, 3, 3, 2, 1, 0] * 4
)

# Nivel de cada subcanal antes de normalizar la mezcla final.
CHANNEL_AMPLITUDE = 0.22
OUTPUT_PEAK = 0.92

# Formato de paquete.
PACKET_MAGIC = b"PC"
PACKET_VERSION = 1
