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
from modules.visualization import save_receiver_diagnostics
from modules.metrics import channel_value_error_rates, load_reference, pixel_error_rate, text_symbol_error_rate

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


def _run_receiver(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(
            f"No existe la grabación: {args.input}\n"
            "Guarda el audio como recordings/received.wav o usa --input."
        )
    
    reference_image_path = DEFAULT_IMAGE
    reference_text_path = DEFAULT_TEXT_FILE

    if not reference_image_path.exists():
        raise FileNotFoundError(
            f"No existe la imagen de referencia: {reference_image_path}"
        )
    
    if not reference_text_path.exists():
        raise FileNotFoundError(
            f"No existe el texto de referencia: {reference_text_path}"
        )
    
    reference_image, reference_text = load_reference(reference_image_path, reference_text_path)
    expected_text_byte_length = len(reference_text.encode("utf-8"))

    decoded = decode_transmission(args.input, strict_crc=False, allow_bad_header=True, expected_text_byte_length=expected_text_byte_length)
    save_decoded(decoded, args.output_dir)

    diagnostics_path = save_receiver_diagnostics(reference_image=reference_image, reconstructed_image=decoded.image, output_dir=args.output_dir)
    image_error = pixel_error_rate(reference_image, decoded.image)
    (red_error, green_error, blue_error) = channel_value_error_rates(reference_image, decoded.image)
    text_error = text_symbol_error_rate(reference_text, decoded.text)

    print("=== RECEPTOR ===")
    print(f"Archivo procesado: {args.input}")
    print(f"Correlación de sincronía: {decoded.sync_correlation:.3f}")
    for channel in CHANNELS:
        diagnostic = decoded.diagnostics[channel.channel_id]
        packet = decoded.packets[channel.channel_id]
        print(
            f"Canal {channel.name:>5}: "
            f"preámbulo={diagnostic.preamble_score:.1%}, "
            f"margen={diagnostic.mean_decision_margin:.1%}, "
            f"cabecera={'OK' if packet.header_ok else 'FALLÓ'}, "
            f"payload={'COMPLETO' if packet.payload_complete else 'INCOMPLETO'}, "
            f"CRC={'OK' if packet.crc_ok else 'FALLÓ'}"
        )

    print()
    print("=== ERRORES ===")
    print(f"Error por píxel: {image_error:.3%}")
    print(f"Error de valores R: {red_error:.3%}")
    print(f"Error de valores G: {green_error:.3%}")
    print(f"Error de valores B: {blue_error:.3%}")
    print(f"Error de texto: {text_error:.3%}")
    print()
    print(f"Texto original: {reference_text}")
    print(f"Texto recuperado: {decoded.text}")
    print()
    print(f"Resultados guardados en: {args.output_dir}")
    print(f"Figura de diagnóstico: {diagnostics_path}")


def main() -> None:
    args = _build_parser().parse_args()

    if args.command in {"transmitter", "tx"}:
        _run_transmitter(args)
    elif args.command in {"receiver", "rx"}:
        _run_receiver(args)


if __name__ == "__main__":
    main()
