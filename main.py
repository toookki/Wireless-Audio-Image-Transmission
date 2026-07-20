"""Punto de entrada principal para la demostración física del proyecto.

Comandos disponibles:

    python main.py transmitter
    python main.py receiver

El transmisor crea el WAV que se reproduce desde el computador emisor. El
receptor procesa offline el WAV grabado por el computador o teléfono receptor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config.settings import CHANNELS
from modules.receiver import decode_transmission, save_decoded
from modules.transmitter import create_transmission

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = PROJECT_ROOT / "data" / "input_image.png"
DEFAULT_TEXT_FILE = PROJECT_ROOT / "data" / "input_text.txt"
DEFAULT_TRANSMISSION = PROJECT_ROOT / "outputs" / "physical" / "transmission.wav"
DEFAULT_RECORDING = PROJECT_ROOT / "recordings" / "received.wav"
DEFAULT_DECODED_DIR = PROJECT_ROOT / "outputs" / "physical" / "decoded"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transmisor y receptor 4-FSK/FDM mediante tonos audibles."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    transmitter = commands.add_parser(
        "transmitter",
        aliases=["tx"],
        help="Genera el WAV que debe reproducirse en el computador transmisor.",
    )
    transmitter.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Imagen de entrada. Por defecto: {DEFAULT_IMAGE}",
    )
    text_source = transmitter.add_mutually_exclusive_group()
    text_source.add_argument("--text", help="Texto literal que se transmitirá.")
    text_source.add_argument(
        "--text-file",
        type=Path,
        default=DEFAULT_TEXT_FILE,
        help=f"Archivo UTF-8. Por defecto: {DEFAULT_TEXT_FILE}",
    )
    transmitter.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TRANSMISSION,
        help=f"WAV de salida. Por defecto: {DEFAULT_TRANSMISSION}",
    )

    receiver = commands.add_parser(
        "receiver",
        aliases=["rx"],
        help="Decodifica el WAV grabado por el computador receptor.",
    )
    receiver.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RECORDING,
        help=f"Grabación WAV. Por defecto: {DEFAULT_RECORDING}",
    )
    receiver.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DECODED_DIR,
        help=f"Carpeta de resultados. Por defecto: {DEFAULT_DECODED_DIR}",
    )

    return parser


def _run_transmitter(args: argparse.Namespace) -> None:
    if not args.image.exists():
        raise FileNotFoundError(f"No existe la imagen: {args.image}")

    if args.text is not None:
        text = args.text
    else:
        if not args.text_file.exists():
            raise FileNotFoundError(f"No existe el archivo de texto: {args.text_file}")
        text = args.text_file.read_text(encoding="utf-8").strip()

    info = create_transmission(args.image, text, args.output)

    print("=== TRANSMISOR ===")
    print(f"WAV creado: {info.output_path}")
    print(f"Duración: {info.duration_s:.2f} s")
    print(f"Bits útiles: {info.payload_bits}")
    print(f"Tasa útil aproximada: {info.effective_payload_rate_bps:.2f} bit/s")
    print("Reproduce este WAV y grábalo sin cancelación de ruido ni formato MP3.")


def _run_receiver(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(
            f"No existe la grabación: {args.input}\n"
            "Guarda el audio como recordings/received.wav o usa --input."
        )

    decoded = decode_transmission(args.input, strict_crc=False)
    save_decoded(decoded, args.output_dir)

    print("=== RECEPTOR ===")
    print(f"Correlación de sincronía: {decoded.sync_correlation:.3f}")
    for channel in CHANNELS:
        diagnostic = decoded.diagnostics[channel.channel_id]
        packet = decoded.packets[channel.channel_id]
        print(
            f"Canal {channel.name:>5}: "
            f"preámbulo={diagnostic.preamble_score:.1%}, "
            f"margen={diagnostic.mean_decision_margin:.1%}, "
            f"CRC={'OK' if packet.crc_ok else 'FALLÓ'}"
        )
    print(f"Texto recuperado: {decoded.text}")
    print(f"Resultados guardados en: {args.output_dir}")


def main() -> None:
    args = _build_parser().parse_args()

    if args.command in {"transmitter", "tx"}:
        _run_transmitter(args)
    elif args.command in {"receiver", "rx"}:
        _run_receiver(args)


if __name__ == "__main__":
    main()
