"""Lógica del transmisor: imagen + texto -> archivo WAV audible.

Este módulo no interpreta argumentos de terminal. La interfaz de usuario está en
``main.py`` para mantener un único punto de entrada fácil de depurar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.settings import CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, SAMPLE_RATE
from modules.audio_io import write_wav_mono
from modules.modem import build_fdm_waveform
from modules.payload import bytes_to_symbols, image_to_channel_packets, text_to_packet


@dataclass(frozen=True)
class TransmissionInfo:
    """Resumen de la transmisión generado para consola y reporte."""

    output_path: Path
    duration_s: float
    payload_bits: int
    effective_payload_rate_bps: float


def create_transmission(image_path: str | Path, text: str, output_path: str | Path,) -> TransmissionInfo:
    """Genera el WAV 4-FSK/FDM que contiene la imagen y el texto.

    Pasos:
    1. La imagen se separa en paquetes R, G y B.
    2. El texto se almacena en un cuarto paquete.
    3. Cada paquete se convierte en símbolos 4-FSK.
    4. Los cuatro canales se modulan y se suman mediante FDM.
    5. La señal final se guarda como WAV mono de 16 bits.
    """

    packets = image_to_channel_packets(str(image_path))
    packets[3] = text_to_packet(text)

    channel_symbols = {channel.channel_id: bytes_to_symbols(packets[channel.channel_id]) for channel in CHANNELS}
    waveform = build_fdm_waveform(channel_symbols)
    write_wav_mono(output_path, SAMPLE_RATE, waveform)

    # Bits útiles, sin contar cabeceras, preámbulo, CRC ni silencios.
    payload_bits = IMAGE_WIDTH * IMAGE_HEIGHT * 3 * 8 + len(text.encode("utf-8")) * 8
    duration_s = len(waveform) / SAMPLE_RATE

    return TransmissionInfo(
        output_path=Path(output_path),
        duration_s=duration_s,
        payload_bits=payload_bits,
        effective_payload_rate_bps=payload_bits / duration_s,
    )