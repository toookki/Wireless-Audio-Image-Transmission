"""Procesa múltiples grabaciones y crea CSV e histogramas de error.

El modo experimental permite extraer el payload aunque la cabecera o el CRC
hayan sido recibidos incorrectamente. Para ello utiliza las longitudes conocidas
de la imagen y del texto de referencia.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter

from modules.metrics import (
    channel_value_error_rates,
    load_reference,
    pixel_error_rate,
    text_symbol_error_rate,
)
from modules.receiver import decode_transmission


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _save_error_histogram(errors: list[float], output_path: str | Path, *, title: str, x_label: str, bar_color: str) -> None:
    """Guarda un histograma de tasas de error entre 0 % y 100 %.

    Parameters
    ----------
    errors:
        Tasas de error expresadas como valores entre 0 y 1.

    output_path:
        Ruta del archivo PNG que se generará.

    title:
        Título del histograma.

    x_label:
        Descripción del eje horizontal.

    bar_color:
        Color utilizado para las barras.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(12, 6.5),
        facecolor="white",
    )

    # Veinte intervalos de 5 %:
    #
    # 0–5 %, 5–10 %, ..., 95–100 %.
    bins = np.linspace(
        0.0,
        1.0,
        21,
    )

    if errors:
        error_array = np.asarray(
            errors,
            dtype=np.float64,
        )

        axis.hist(
            error_array,
            bins=bins,
            color=bar_color,
            alpha=0.82,
            edgecolor="black",
            linewidth=0.75,
            rwidth=0.94,
        )

    else:
        axis.text(
            0.5,
            0.5,
            "No hubo transmisiones medibles",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axis.transAxes,
            fontsize=13,
        )

    axis.set_title(
        title,
        fontsize=17,
        fontweight="normal",
        pad=14,
    )

    axis.set_xlabel(
        x_label,
        fontsize=13,
        labelpad=9,
    )

    axis.set_ylabel(
        "Número de transmisiones",
        fontsize=13,
        labelpad=9,
    )

    # El eje horizontal siempre llega de 0 % a 100 %.
    axis.set_xlim(
        0.0,
        1.0,
    )

    # Marcas cada 5 %.
    axis.set_xticks(
        np.linspace(
            0.0,
            1.0,
            21,
        )
    )

    axis.xaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    # Se rotan ligeramente las etiquetas para evitar superposición.
    axis.tick_params(
        axis="x",
        labelsize=9,
        rotation=45,
    )

    axis.tick_params(
        axis="y",
        labelsize=11,
    )

    # El eje Y representa cantidades enteras de transmisiones.
    axis.yaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    axis.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.30,
    )

    axis.set_axisbelow(True)

    # Apariencia más limpia.
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)


def evaluate_folder(
    recordings_dir: str | Path,
    reference_image_path: str | Path,
    reference_text_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Evalúa todos los archivos WAV encontrados en una carpeta.

    Para cada grabación intenta:

    1. Encontrar el chirp de sincronización.
    2. Demodular los cuatro subcanales.
    3. Extraer los payloads usando las longitudes conocidas.
    4. Comparar la imagen y el texto reconstruidos con las referencias.
    5. Registrar el estado de cabecera, payload y CRC.

    Una cabecera o CRC incorrectos no detienen el procesamiento experimental.
    Los fallos que impiden demodular completamente la transmisión se registran
    en la columna ``error``.
    """

    recordings_dir = Path(recordings_dir)
    reference_image_path = Path(
        reference_image_path
    )
    reference_text_path = Path(
        reference_text_path
    )
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_image, reference_text = load_reference(
        reference_image_path,
        reference_text_path,
    )

    # Se necesita la cantidad de bytes, no solamente la cantidad
    # de caracteres. Por ejemplo, "¡" ocupa más de un byte en UTF-8.
    expected_text_byte_length = len(
        reference_text.encode("utf-8")
    )

    wav_files = sorted(
        recordings_dir.glob("*.wav")
    )

    if not wav_files:
        raise FileNotFoundError(
            f"No hay archivos WAV en la carpeta: {recordings_dir}"
        )

    rows: list[dict[str, object]] = []

    for wav_path in wav_files:
        row: dict[str, object] = {
            "recording": wav_path.name,
        }

        try:
            decoded = decode_transmission(
                wav_path,
                strict_crc=False,
                allow_bad_header=True,
                expected_text_byte_length=(
                    expected_text_byte_length
                ),
            )

            (
                red_error,
                green_error,
                blue_error,
            ) = channel_value_error_rates(
                reference_image,
                decoded.image,
            )

            row.update(
                {
                    "decoded": True,
                    "pixel_error_rate": pixel_error_rate(
                        reference_image,
                        decoded.image,
                    ),
                    "red_value_error_rate": red_error,
                    "green_value_error_rate": green_error,
                    "blue_value_error_rate": blue_error,
                    "text_symbol_error_rate": (
                        text_symbol_error_rate(
                            reference_text,
                            decoded.text,
                        )
                    ),
                    "sync_correlation": (
                        decoded.sync_correlation
                    ),

                    # Estado del CRC recibido en cada subcanal.
                    "crc_red_ok": (
                        decoded.packets[0].crc_ok
                    ),
                    "crc_green_ok": (
                        decoded.packets[1].crc_ok
                    ),
                    "crc_blue_ok": (
                        decoded.packets[2].crc_ok
                    ),
                    "crc_text_ok": (
                        decoded.packets[3].crc_ok
                    ),

                    # Indica si la cabecera recibida coincidió
                    # con la esperada.
                    "header_red_ok": (
                        decoded.packets[0].header_ok
                    ),
                    "header_green_ok": (
                        decoded.packets[1].header_ok
                    ),
                    "header_blue_ok": (
                        decoded.packets[2].header_ok
                    ),
                    "header_text_ok": (
                        decoded.packets[3].header_ok
                    ),

                    # Indica si llegaron todos los bytes esperados
                    # del payload.
                    "payload_red_complete": (
                        decoded.packets[0].payload_complete
                    ),
                    "payload_green_complete": (
                        decoded.packets[1].payload_complete
                    ),
                    "payload_blue_complete": (
                        decoded.packets[2].payload_complete
                    ),
                    "payload_text_complete": (
                        decoded.packets[3].payload_complete
                    ),

                    "error": "",
                }
            )

        except Exception as error:
            # La grabación se conserva en el CSV aunque no haya sido
            # posible demodularla. Se utiliza NaN porque el error no
            # pudo medirse realmente.
            row.update(
                {
                    "decoded": False,
                    "pixel_error_rate": float("nan"),
                    "red_value_error_rate": float("nan"),
                    "green_value_error_rate": float("nan"),
                    "blue_value_error_rate": float("nan"),
                    "text_symbol_error_rate": float("nan"),
                    "sync_correlation": float("nan"),

                    "crc_red_ok": False,
                    "crc_green_ok": False,
                    "crc_blue_ok": False,
                    "crc_text_ok": False,

                    "header_red_ok": False,
                    "header_green_ok": False,
                    "header_blue_ok": False,
                    "header_text_ok": False,

                    "payload_red_complete": False,
                    "payload_green_complete": False,
                    "payload_blue_complete": False,
                    "payload_text_complete": False,

                    "error": str(error).replace(
                        "\n",
                        " | ",
                    ),
                }
            )

        rows.append(row)

    csv_path = (
        output_dir / "experiment_results.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    # Solo se incluyen en los histogramas las transmisiones
    # para las cuales se pudo obtener una medición real.
    decoded_rows = [
        row
        for row in rows
        if bool(row["decoded"])
    ]

    pixel_errors = [
        float(row["pixel_error_rate"])
        for row in decoded_rows
    ]

    text_errors = [
        float(row["text_symbol_error_rate"])
        for row in decoded_rows
    ]

    _save_error_histogram(
        pixel_errors,
        output_dir / "pixel_error_histogram.png",
        title="Histograma del error de imagen",
        x_label="Tasa de error por píxel",
        bar_color="#4dabf7",
    )

    _save_error_histogram(
        text_errors,
        output_dir / "text_error_histogram.png",
        title="Histograma del error de texto",
        x_label="Tasa de error por símbolo de texto",
        bar_color="#9775fa",
    )

    total_recordings = len(rows)
    decoded_recordings = len(decoded_rows)
    failed_recordings = (
        total_recordings - decoded_recordings
    )

    print("=== EVALUACIÓN EXPERIMENTAL ===")
    print(
        f"Grabaciones encontradas: "
        f"{total_recordings}"
    )
    print(
        f"Grabaciones medibles: "
        f"{decoded_recordings}"
    )
    print(
        f"Fallos completos: "
        f"{failed_recordings}"
    )
    print(f"CSV: {csv_path}")
    print(f"Histogramas: {output_dir}")


def _parse_args() -> argparse.Namespace:
    """Lee los argumentos entregados desde la terminal."""

    parser = argparse.ArgumentParser(
        description="Evalúa múltiples grabaciones WAV.",
    )

    parser.add_argument(
        "--recordings",
        type=Path,
        default=(
            PROJECT_ROOT
            / "recordings"
            / "experiment"
        ),
        help="Carpeta que contiene las grabaciones WAV.",
    )

    parser.add_argument(
        "--reference-image",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "input_image.png"
        ),
        help="Imagen original utilizada como referencia.",
    )

    parser.add_argument(
        "--reference-text",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "input_text.txt"
        ),
        help="Texto original utilizado como referencia.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "experiment"
        ),
        help=(
            "Carpeta en la que se guardarán "
            "el CSV y los histogramas."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Punto de entrada del programa de evaluación."""

    args = _parse_args()

    evaluate_folder(
        args.recordings,
        args.reference_image,
        args.reference_text,
        args.output_dir,
    )


if __name__ == "__main__":
    main()