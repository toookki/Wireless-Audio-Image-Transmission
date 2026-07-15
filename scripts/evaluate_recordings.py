"""Procesa múltiples grabaciones y crea CSV e histogramas de error.

Uso típico para la entrega:
1. Reproducir la misma transmisión aproximadamente 21 veces.
2. Guardar cada grabación WAV dentro de ``recordings/``.
3. Ejecutar este programa para obtener las distribuciones de error.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from modules.metrics import (
    channel_value_error_rates,
    load_reference,
    pixel_error_rate,
    text_symbol_error_rate,
)
from modules.receiver import decode_transmission


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_folder(
    recordings_dir: str | Path,
    reference_image_path: str | Path,
    reference_text_path: str | Path,
    output_dir: str | Path,
) -> None:
    recordings_dir = Path(recordings_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_image, reference_text = load_reference(
        reference_image_path, reference_text_path
    )
    wav_files = sorted(recordings_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No hay archivos WAV en {recordings_dir}.")

    rows: list[dict[str, object]] = []
    for wav_path in wav_files:
        row: dict[str, object] = {"recording": wav_path.name}
        try:
            decoded = decode_transmission(wav_path, strict_crc=False)
            red_error, green_error, blue_error = channel_value_error_rates(
                reference_image, decoded.image
            )
            row.update(
                {
                    "decoded": True,
                    "pixel_error_rate": pixel_error_rate(
                        reference_image, decoded.image
                    ),
                    "red_value_error_rate": red_error,
                    "green_value_error_rate": green_error,
                    "blue_value_error_rate": blue_error,
                    "text_symbol_error_rate": text_symbol_error_rate(
                        reference_text, decoded.text
                    ),
                    "sync_correlation": decoded.sync_correlation,
                    "crc_red_ok": decoded.packets[0].crc_ok,
                    "crc_green_ok": decoded.packets[1].crc_ok,
                    "crc_blue_ok": decoded.packets[2].crc_ok,
                    "crc_text_ok": decoded.packets[3].crc_ok,
                    "error": "",
                }
            )
        except Exception as error:  # El CSV debe registrar incluso fallos completos.
            row.update(
                {
                    "decoded": False,
                    "pixel_error_rate": 1.0,
                    "red_value_error_rate": 1.0,
                    "green_value_error_rate": 1.0,
                    "blue_value_error_rate": 1.0,
                    "text_symbol_error_rate": 1.0,
                    "sync_correlation": 0.0,
                    "crc_red_ok": False,
                    "crc_green_ok": False,
                    "crc_blue_ok": False,
                    "crc_text_ok": False,
                    "error": str(error).replace("\n", " | "),
                }
            )
        rows.append(row)

    csv_path = output_dir / "experiment_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pixel_errors = [float(row["pixel_error_rate"]) for row in rows]
    text_errors = [float(row["text_symbol_error_rate"]) for row in rows]

    plt.figure()
    plt.hist(pixel_errors, bins=10)
    plt.xlabel("Tasa de error por píxel")
    plt.ylabel("Número de transmisiones")
    plt.title("Histograma del error de imagen")
    plt.tight_layout()
    plt.savefig(output_dir / "pixel_error_histogram.png", dpi=160)
    plt.close()

    plt.figure()
    plt.hist(text_errors, bins=10)
    plt.xlabel("Tasa de error por símbolo de texto")
    plt.ylabel("Número de transmisiones")
    plt.title("Histograma del error de texto")
    plt.tight_layout()
    plt.savefig(output_dir / "text_error_histogram.png", dpi=160)
    plt.close()

    print(f"Grabaciones procesadas: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Histogramas: {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa múltiples grabaciones WAV.")
    parser.add_argument("--recordings", type=Path, default=PROJECT_ROOT / "recordings")
    parser.add_argument("--reference-image", type=Path, default=PROJECT_ROOT / "data" / "input_image.png")
    parser.add_argument("--reference-text", type=Path, default=PROJECT_ROOT / "data" / "input_text.txt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "experiment")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evaluate_folder(
        args.recordings,
        args.reference_image,
        args.reference_text,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
