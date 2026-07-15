import numpy as np

from modules.payload import Packet, build_packet, bytes_to_symbols, parse_packet, symbols_to_bytes


def test_bytes_symbols_roundtrip() -> None:
    original = bytes(range(256))
    symbols = bytes_to_symbols(original)
    recovered = symbols_to_bytes(symbols)
    assert recovered == original


def test_packet_crc_roundtrip() -> None:
    original = Packet(channel_id=3, payload="hola".encode("utf-8"))
    encoded = build_packet(original)
    decoded = parse_packet(encoded, expected_channel_id=3)
    assert decoded == original


def test_symbol_values_are_valid() -> None:
    symbols = bytes_to_symbols(b"abc")
    assert np.all((symbols >= 0) & (symbols <= 3))
