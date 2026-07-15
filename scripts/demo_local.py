"""Demostración completa sin hardware: transmite y decodifica el mismo WAV."""

from pathlib import Path

from modules.metrics import pixel_error_rate, text_symbol_error_rate
from modules.receiver import decode_transmission, save_decoded
from modules.transmitter import create_transmission


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    image_path = root / "data" / "input_image.png"
    text_path = root / "data" / "input_text.txt"
    wav_path = root / "outputs" / "transmission.wav"
    decoded_dir = root / "outputs" / "decoded"

    reference_text = text_path.read_text(encoding="utf-8").strip()
    info = create_transmission(image_path, reference_text, wav_path)
    decoded = decode_transmission(wav_path)
    save_decoded(decoded, decoded_dir)

    from PIL import Image

    reference_image = Image.open(image_path).convert("RGB")
    image_error = pixel_error_rate(reference_image, decoded.image)
    text_error = text_symbol_error_rate(reference_text, decoded.text)

    print("=== DEMOSTRACIÓN COMPLETA ===")
    print(f"WAV: {wav_path}")
    print(f"Duración: {info.duration_s:.2f} s")
    print(f"Tasa útil: {info.effective_payload_rate_bps:.2f} bit/s")
    print(f"Error de píxel: {image_error:.3%}")
    print(f"Error de texto: {text_error:.3%}")
    print(f"Salida reconstruida: {decoded_dir}")


if __name__ == "__main__":
    main()
