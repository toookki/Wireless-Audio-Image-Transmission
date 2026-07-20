"""Visualización de los resultados obtenidos por el receptor.

Este módulo genera una figura que contiene:

- Imagen original.
- Imagen reconstruida.
- Mapa de píxeles erróneos.
- Histograma del canal rojo.
- Histograma del canal verde.
- Histograma del canal azul.

La figura se guarda como ``receiver_diagnostics.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_receiver_diagnostics(reference_image: np.ndarray, reconstructed_image: np.ndarray, output_dir: str | Path) -> Path:
    """Guarda una comparación visual entre la imagen original y la recibida.

    Un píxel se considera erróneo cuando al menos uno de sus componentes
    rojo, verde o azul es diferente del píxel original.

    En el mapa de píxeles erróneos:

    - Negro: píxel correcto.
    - Blanco: píxel incorrecto.

    Parameters
    ----------
    reference_image:
        Imagen RGB original con forma ``(alto, ancho, 3)``.

    reconstructed_image:
        Imagen RGB reconstruida por el receptor.

    output_dir:
        Carpeta donde se guardará la figura.

    Returns
    -------
    Path
        Ruta del archivo ``receiver_diagnostics.png``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = np.asarray(reference_image, dtype=np.uint8)

    reconstructed = np.asarray(reconstructed_image, dtype=np.uint8)

    if original.shape != reconstructed.shape:
        raise ValueError(
            "La imagen original y la reconstruida deben tener "
            "las mismas dimensiones. "
            f"Original={original.shape}, "
            f"reconstruida={reconstructed.shape}."
        )

    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError(
            "Las imágenes deben ser RGB y tener forma "
            "(alto, ancho, 3)."
        )

    # True cuando al menos uno de los componentes RGB es diferente.
    error_mask = np.any(original != reconstructed, axis=2)

    # Figura de dos filas y tres columnas.
    figure, axes = plt.subplots(nrows=2, ncols=3, figsize=(13.5, 8.65), facecolor="white")

    # Imagen original
    axes[0, 0].imshow(original, interpolation="nearest")
    axes[0, 0].set_title("Original", fontsize=12, fontweight="normal")
    axes[0, 0].axis("off")

    # Imagen reconstruida
    axes[0, 1].imshow(reconstructed, interpolation="nearest")
    axes[0, 1].set_title("Reconstruida", fontsize=12, fontweight="normal")
    axes[0, 1].axis("off")

    # Mapa de píxeles erróneos
    axes[0, 2].imshow(error_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 2].set_title("Mapa de píxeles erróneos", fontsize=12, fontweight="normal",)
    axes[0, 2].axis("off")

    # Configuración de los histogramas RGB
    channel_information = ((0, "R", "#ff6b6b"), (1, "G", "#66bb6a"), (2, "B", "#6666ff"))
    bins = np.arange(-0.5, 256.5, 4)
    for column, (channel_index, channel_name, channel_color) in enumerate(channel_information):
        axis = axes[1, column]
        original_values = original[:, :, channel_index].ravel()
        reconstructed_values = reconstructed[:, :, channel_index].ravel()

        axis.hist(original_values, bins=bins, color=channel_color, alpha=0.75, edgecolor="black", linewidth=0.55, rwidth=0.95, label="Original")
        axis.hist(reconstructed_values, bins=bins, histtype="step", color="black", linewidth=1.25, label="Reconstruida")
        axis.set_title(f"Histograma {channel_name}", fontsize=12, fontweight="normal")
        axis.set_xlabel("Valor", fontsize=10, fontweight="normal")
        axis.set_ylabel("Conteo", fontsize=10, fontweight="normal")
        axis.set_xlim(-10, 265)
        axis.set_xticks([0, 50, 100, 150, 200, 250])
        axis.tick_params(axis="both", labelsize=9)
        axis.legend(loc="upper right", fontsize=8, frameon=True,)

        # Se conservan los cuatro bordes de cada gráfico.
        axis.spines["top"].set_visible(True)
        axis.spines["right"].set_visible(True)
        axis.spines["bottom"].set_visible(True)
        axis.spines["left"].set_visible(True)

        # La referencia no utiliza cuadrícula.
        axis.grid(False)

    figure.tight_layout(pad=1.0, h_pad=0.45, w_pad=1.35)
    figure_path = (output_dir / "receiver_diagnostics.png")
    figure.savefig(figure_path, dpi=160, bbox_inches="tight", facecolor="white",)
    plt.close(figure)
    return figure_path