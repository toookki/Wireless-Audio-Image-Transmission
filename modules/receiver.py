"""Lógica del receptor: grabación WAV -> imagen y texto reconstruidos.

Este módulo contiene solamente funciones reutilizables. ``main.py`` es el único
punto de entrada para ejecutar una recepción normal desde la terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from config.settings import CHANNELS, SAMPLE_RATE
from modules.audio_io import read_wav_mono
from modules.modem import ChannelDemodulation, demodulate_channel, find_sync
from modules.payload import Packet, PacketError, packets_to_image, parse_packet, symbols_to_bytes


@dataclass(frozen=True)
class DecodedTransmission:
    """Datos reconstruidos y diagnósticos de la recepción."""

    image: Image.Image
    text: str
    packets: dict[int, Packet]
    diagnostics: dict[int, ChannelDemodulation]
    sync_correlation: float


def decode_transmission(
    input_wav: str | Path,
    *,
    strict_crc: bool = True,
) -> DecodedTransmission:
    """Decodifica una grabación completa.

    Por defecto exige CRC correcto. ``strict_crc=False`` se reserva para los
    experimentos, porque permite conservar datos dañados y medir el error real.
    """

    audio = read_wav_mono(input_wav, SAMPLE_RATE)
    sync = find_sync(audio)
    data_audio = audio[sync.data_start_sample :]

    packets: dict[int, Packet] = {}
    diagnostics: dict[int, ChannelDemodulation] = {}
    errors: list[str] = []

    for channel in CHANNELS:
        demodulated = demodulate_channel(data_audio, channel)
        diagnostics[channel.channel_id] = demodulated
        raw_bytes = symbols_to_bytes(demodulated.symbols)

        try:
            packets[channel.channel_id] = parse_packet(
                raw_bytes,
                expected_channel_id=channel.channel_id,
                require_crc=strict_crc,
            )
        except PacketError as error:
            errors.append(f"Canal {channel.name}: {error}")

    if errors:
        details = "\n".join(errors)
        raise PacketError(f"No fue posible validar todos los canales:\n{details}")

    image = packets_to_image(packets)
    text = packets[3].payload.decode(
        "utf-8", errors="strict" if strict_crc else "replace"
    )

    return DecodedTransmission(
        image=image,
        text=text,
        packets=packets,
        diagnostics=diagnostics,
        sync_correlation=sync.correlation,
    )


def save_decoded(decoded: DecodedTransmission, output_dir: str | Path) -> None:
    """Guarda la imagen y el texto reconstruidos en ``output_dir``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decoded.image.save(output_dir / "reconstructed_image.png")
    (output_dir / "reconstructed_text.txt").write_text(
        decoded.text + "\n", encoding="utf-8"
    )
