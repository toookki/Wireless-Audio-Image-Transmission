"""Lógica del receptor: grabación WAV -> imagen y texto reconstruidos.

Este módulo contiene solamente funciones reutilizables. ``main.py`` es el único
punto de entrada para ejecutar una recepción normal desde la terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from config.settings import CHANNELS, SAMPLE_RATE, IMAGE_HEIGHT, IMAGE_WIDTH
from modules.audio_io import read_wav_mono
from modules.modem import ChannelDemodulation, demodulate_channel, find_sync
from modules.payload import Packet, PacketError, packets_to_image, parse_packet, parse_packet_experimental, symbols_to_bytes


@dataclass(frozen=True)
class DecodedTransmission:
    """Datos reconstruidos y diagnósticos de la recepción."""

    image: Image.Image
    text: str
    packets: dict[int, Packet]
    diagnostics: dict[int, ChannelDemodulation]
    sync_correlation: float


def decode_transmission(input_wav: str | Path, *, strict_crc: bool = True, allow_bad_header: bool = False, expected_text_byte_length: int | None = None
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
            if allow_bad_header:
                # Los canales RGB siempre contienen 20 × 20 = 400 bytes.
                if channel.channel_id < 3:
                    expected_payload_length = IMAGE_WIDTH * IMAGE_HEIGHT
                    expected_height = IMAGE_HEIGHT
                    expected_width = IMAGE_WIDTH

                # La longitud del texto se obtiene desde el texto de referencia.
                else:
                    if expected_text_byte_length is None:
                        raise ValueError(
                            "Debe indicarse expected_text_byte_length cuando "
                            "allow_bad_header=True."
                        )

                    expected_payload_length = expected_text_byte_length
                    expected_height = 0
                    expected_width = 0

                packets[channel.channel_id] = parse_packet_experimental(
                    raw_bytes,
                    expected_channel_id=channel.channel_id,
                    expected_payload_length=expected_payload_length,
                    expected_height=expected_height,
                    expected_width=expected_width,
                )

            else:
                # Comportamiento normal: se valida toda la cabecera.
                packets[channel.channel_id] = parse_packet(
                    raw_bytes,
                    expected_channel_id=channel.channel_id,
                    require_crc=strict_crc,
                )

        except (PacketError, ValueError) as error:
            errors.append(f"Canal {channel.name}: {error}")

    if errors:
        details = "\n".join(errors)
        raise PacketError(f"No fue posible validar todos los canales:\n{details}")

    image = packets_to_image(packets)
    text = packets[3].payload.decode("utf-8", errors="strict" if strict_crc else "replace")

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
    (output_dir / "reconstructed_text.txt").write_text(decoded.text + "\n", encoding="utf-8")