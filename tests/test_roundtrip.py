from pathlib import Path

from modules.metrics import pixel_error_rate, text_symbol_error_rate
from modules.receiver import decode_transmission
from modules.transmitter import create_transmission


def test_clean_audio_roundtrip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    image_path = root / "data" / "input_image.png"
    text = (root / "data" / "input_text.txt").read_text(encoding="utf-8").strip()
    wav_path = tmp_path / "transmission.wav"

    create_transmission(image_path, text, wav_path)
    decoded = decode_transmission(wav_path)

    from PIL import Image

    reference_image = Image.open(image_path).convert("RGB")
    assert pixel_error_rate(reference_image, decoded.image) == 0.0
    assert text_symbol_error_rate(text, decoded.text) == 0.0