"""Métricas solicitadas por el enunciado."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def pixel_error_rate(reference: Image.Image, received: Image.Image) -> float:
    """Fracción de píxeles cuyo RGB completo difiere de la referencia."""

    reference_array = np.asarray(reference.convert("RGB"), dtype=np.uint8)
    received_array = np.asarray(received.convert("RGB"), dtype=np.uint8)
    if reference_array.shape != received_array.shape:
        raise ValueError(f"Dimensiones distintas: {reference_array.shape} vs {received_array.shape}.")
    wrong_pixels = np.any(reference_array != received_array, axis=2)
    return float(np.mean(wrong_pixels))


def channel_value_error_rates(reference: Image.Image, received: Image.Image) -> tuple[float, float, float]:
    """Error de valores R, G y B por separado."""

    reference_array = np.asarray(reference.convert("RGB"), dtype=np.uint8)
    received_array = np.asarray(received.convert("RGB"), dtype=np.uint8)
    if reference_array.shape != received_array.shape:
        raise ValueError("Las imágenes deben tener las mismas dimensiones.")
    return tuple(
        float(np.mean(reference_array[:, :, index] != received_array[:, :, index]))
        for index in range(3)
    )


def levenshtein_distance(reference: str, received: str) -> int:
    """Número mínimo de inserciones, eliminaciones y sustituciones."""

    previous = list(range(len(received) + 1))
    for row, reference_char in enumerate(reference, start=1):
        current = [row]
        for column, received_char in enumerate(received, start=1):
            substitution_cost = 0 if reference_char == received_char else 1
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + substitution_cost))
        previous = current
    return previous[-1]


def text_symbol_error_rate(reference: str, received: str) -> float:
    """SER de texto basada en distancia de edición normalizada."""

    denominator = max(len(reference), 1)
    return levenshtein_distance(reference, received) / denominator


def load_reference(image_path: str | Path, text_path: str | Path) -> tuple[Image.Image, str]:
    """Carga las referencias usadas en los experimentos."""

    image = Image.open(image_path).convert("RGB")
    text = Path(text_path).read_text(encoding="utf-8").strip()
    return image, text