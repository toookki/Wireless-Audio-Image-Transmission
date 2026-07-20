"""Conversión de imagen/texto a paquetes binarios y viceversa.

Cada canal transporta un paquete independiente con cabecera y CRC-32. El CRC
no corrige errores, pero permite detectar inmediatamente si un canal fue
recuperado de forma incorrecta.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image

from config.settings import (IMAGE_HEIGHT, IMAGE_WIDTH, PACKET_MAGIC, PACKET_VERSION)

# magic, version, channel_id, payload_length, height, width, crc32
_HEADER = struct.Struct(">2sBBIHHI")
HEADER_SIZE = _HEADER.size


class PacketError(ValueError):
    """Indica que un paquete está incompleto o no supera sus verificaciones."""


@dataclass(frozen=True)
class Packet:
    """Paquete recuperado y diagnósticos de integridad."""

    channel_id: int
    payload: bytes
    height: int = 0
    width: int = 0
    crc_ok: bool = True
    header_ok: bool = True
    payload_complete: bool = True


def build_packet(packet: Packet) -> bytes:
    """Serializa un paquete y agrega CRC-32 sobre el payload."""

    crc = zlib.crc32(packet.payload) & 0xFFFFFFFF
    header = _HEADER.pack(
        PACKET_MAGIC,
        PACKET_VERSION,
        packet.channel_id,
        len(packet.payload),
        packet.height,
        packet.width,
        crc,
    )
    return header + packet.payload


def parse_packet(raw: bytes, expected_channel_id: int | None = None, require_crc: bool = True) -> Packet:
    """Interpreta un paquete y opcionalmente exige que su CRC sea correcto.

    ``require_crc=False`` es útil en los experimentos: permite reconstruir un
    payload corrupto y medir su tasa de error real, en vez de clasificar todo
    fallo de CRC automáticamente como 100 % de error.
    """

    if len(raw) < HEADER_SIZE:
        raise PacketError(f"Paquete demasiado corto: {len(raw)} bytes; se requieren al menos {HEADER_SIZE}.")

    magic, version, channel_id, payload_length, height, width, expected_crc = (_HEADER.unpack(raw[:HEADER_SIZE]))

    if magic != PACKET_MAGIC:
        raise PacketError(f"Magic incorrecto: {magic!r}.")
    if version != PACKET_VERSION:
        raise PacketError(f"Versión no soportada: {version}.")
    if expected_channel_id is not None and channel_id != expected_channel_id:
        raise PacketError(f"Canal recibido {channel_id}; se esperaba {expected_channel_id}.")

    total_length = HEADER_SIZE + payload_length
    if len(raw) < total_length:
        raise PacketError(f"Payload incompleto: llegaron {len(raw)} bytes y se requieren {total_length}.")

    payload = raw[HEADER_SIZE:total_length]
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    crc_ok = actual_crc == expected_crc
    if require_crc and not crc_ok:
        raise PacketError(f"CRC incorrecto: calculado 0x{actual_crc:08X}, esperado 0x{expected_crc:08X}.")

    return Packet(channel_id=channel_id, payload=payload, height=height, width=width, crc_ok=crc_ok)


def parse_packet_experimental(raw: bytes, *, expected_channel_id: int, expected_payload_length: int, expected_height: int = 0, expected_width: int = 0,
) -> Packet:
    """Extrae un payload usando la estructura fija conocida del experimento.

    La cabecera se revisa para diagnóstico, pero sus campos no se utilizan
    para decidir dónde comienza o cuánto mide el payload.

    Esto permite medir el error aunque magic, versión, canal, longitud o
    dimensiones hayan sido recibidos incorrectamente.
    """

    # Se toman los bytes disponibles de la cabecera.
    header_bytes = raw[:HEADER_SIZE]
    header_complete = len(header_bytes) == HEADER_SIZE

    # Si la cabecera llegó incompleta, se completa temporalmente con ceros
    # para que struct pueda desempaquetarla sin detener el experimento.
    padded_header = header_bytes.ljust(HEADER_SIZE, b"\x00")

    (magic, version, channel_id, payload_length, height, width, expected_crc) = _HEADER.unpack(padded_header)

    # La cabecera se evalúa, pero un resultado False no detiene el proceso.
    header_ok = (
        header_complete
        and magic == PACKET_MAGIC
        and version == PACKET_VERSION
        and channel_id == expected_channel_id
        and payload_length == expected_payload_length
        and height == expected_height
        and width == expected_width
    )

    # En el experimento sabemos que el payload siempre comienza justo
    # después de los HEADER_SIZE bytes.
    payload_start = HEADER_SIZE
    payload_end = payload_start + expected_payload_length
    payload = raw[payload_start:payload_end]

    payload_complete = len(payload) == expected_payload_length

    # Si la grabación terminó antes, los bytes faltantes se completan con
    # cero. Al comparar con la referencia aparecerán como errores reales.
    if not payload_complete:
        payload = payload.ljust(expected_payload_length, b"\x00")

    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    crc_ok = payload_complete and actual_crc == expected_crc

    return Packet(
        channel_id=expected_channel_id,
        payload=payload,
        height=expected_height,
        width=expected_width,
        crc_ok=crc_ok,
        header_ok=header_ok,
        payload_complete=payload_complete,
    )


def bytes_to_symbols(data: bytes) -> np.ndarray:
    """Convierte bytes en símbolos 4-FSK (dos bits por símbolo)."""

    byte_array = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(byte_array)
    pairs = bits.reshape(-1, 2)
    return (2 * pairs[:, 0] + pairs[:, 1]).astype(np.uint8)


def symbols_to_bytes(symbols: np.ndarray) -> bytes:
    """Convierte símbolos 4-FSK en bytes; ignora símbolos sobrantes."""

    symbols = np.asarray(symbols, dtype=np.uint8)
    usable_symbols = (len(symbols) // 4) * 4  # 4 símbolos = 1 byte.
    symbols = symbols[:usable_symbols]
    if len(symbols) == 0:
        return b""

    bits = np.empty(len(symbols) * 2, dtype=np.uint8)
    bits[0::2] = (symbols >> 1) & 1
    bits[1::2] = symbols & 1
    return np.packbits(bits).tobytes()


def image_to_channel_packets(image_path: str) -> dict[int, bytes]:
    """Lee una imagen, la lleva a 20x20 RGB y crea paquetes R, G y B."""

    image = Image.open(image_path).convert("RGB")
    if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.NEAREST)

    pixels = np.asarray(image, dtype=np.uint8)
    packets: dict[int, bytes] = {}
    for channel_id in range(3):
        channel_payload = pixels[:, :, channel_id].reshape(-1).tobytes()
        packets[channel_id] = build_packet(
            Packet(
                channel_id=channel_id,
                payload=channel_payload,
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
            )
        )
    return packets


def text_to_packet(text: str) -> bytes:
    """Codifica texto UTF-8 en el canal 3."""

    return build_packet(Packet(channel_id=3, payload=text.encode("utf-8")))


def packets_to_image(packets: dict[int, Packet]) -> Image.Image:
    """Reconstruye una imagen RGB a partir de los paquetes 0, 1 y 2."""

    missing = [channel_id for channel_id in range(3) if channel_id not in packets]
    if missing:
        raise PacketError(f"Faltan canales RGB: {missing}.")

    heights = {packets[channel_id].height for channel_id in range(3)}
    widths = {packets[channel_id].width for channel_id in range(3)}
    if len(heights) != 1 or len(widths) != 1:
        raise PacketError("Los tres canales RGB informan dimensiones distintas.")

    height = heights.pop()
    width = widths.pop()
    expected_values = height * width

    rgb_channels = []
    for channel_id in range(3):
        values = np.frombuffer(packets[channel_id].payload, dtype=np.uint8)
        if len(values) != expected_values:
            raise PacketError(f"Canal {channel_id}: {len(values)} valores; se esperaban {expected_values}.")
        rgb_channels.append(values.reshape(height, width))

    rgb = np.stack(rgb_channels, axis=2)
    return Image.fromarray(rgb, mode="RGB")