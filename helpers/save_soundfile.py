import io
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = PROJECT_ROOT / "generated"
SUPPORTED_FORMATS = ("wav", "mp3")


def save_soundfile(
    audio,
    name: str,
    sample_rate: int | None = None,
    audio_format: str = "wav",
) -> Path:
    """Save audio into ``generated/``.

    Samples are stored as WAV, same as before. Pass ``audio_format="mp3"`` to
    encode those samples as MP3.

    ``audio`` may also be WAV or MP3 bytes, for example the MP3 returned by an
    online TTS service. Bytes are stored unchanged when the requested format
    matches. MP3 bytes requested as WAV are decoded first, so later processing
    keeps a PCM file instead of the lossy service file.
    """
    audio_format = _normalize_format(audio_format)
    file_path = _output_path(name, audio_format)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio, (bytes, bytearray)):
        _write_encoded_bytes(bytes(audio), file_path, audio_format, sample_rate)
    else:
        _write_samples(audio, file_path, audio_format, sample_rate)

    if not file_path.is_file() or file_path.stat().st_size == 0:
        raise OSError(f"Audio file was not created: {file_path}")

    print(f"Saved {audio_format.upper()} file: {file_path}")
    return file_path


def _normalize_format(audio_format: str) -> str:
    if not isinstance(audio_format, str):
        raise TypeError("audio_format must be 'wav' or 'mp3'.")

    normalized = audio_format.strip().lower().lstrip(".")
    if normalized not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format {audio_format!r}. Use 'wav' or 'mp3'."
        )
    return normalized


def _output_path(name: str, audio_format: str) -> Path:
    if not isinstance(name, str):
        raise TypeError("A filename must be provided.")

    filename = Path(name.strip()).name
    if filename in {"", ".", ".."}:
        raise ValueError("A filename must be provided.")

    suffix = Path(filename).suffix.lower()
    expected = f".{audio_format}"
    if suffix == expected:
        filename = f"{Path(filename).stem}{expected}"
    elif suffix in {f".{fmt}" for fmt in SUPPORTED_FORMATS}:
        raise ValueError(
            f"Filename {filename!r} does not match the audio format {audio_format!r}."
        )
    else:
        filename += expected

    return GENERATED_DIR / filename


def _require_sample_rate(sample_rate) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise TypeError("sample_rate must be a positive integer.")

    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    return sample_rate


def _write_samples(audio, file_path: Path, audio_format: str, sample_rate) -> None:
    sample_rate = _require_sample_rate(sample_rate)
    samples = np.asarray(audio)
    if samples.size == 0:
        raise ValueError("Audio must not be empty.")

    if audio_format == "wav":
        sf.write(str(file_path), samples, sample_rate)
        return

    try:
        sf.write(str(file_path), samples, sample_rate, format="MP3")
    except sf.SoundFileError as error:
        raise RuntimeError(
            "Could not write MP3. This libsndfile build needs MP3 encoding support."
        ) from error


def _write_encoded_bytes(
    data: bytes,
    file_path: Path,
    audio_format: str,
    sample_rate,
) -> None:
    if not data:
        raise ValueError("Audio bytes must not be empty.")

    source_format = _detect_encoded_format(data)
    if source_format == audio_format:
        file_path.write_bytes(data)
        return

    samples, decoded_rate = _decode_encoded_bytes(data)
    if sample_rate is not None and _require_sample_rate(sample_rate) != decoded_rate:
        raise ValueError(
            f"sample_rate {sample_rate} does not match the encoded audio "
            f"({decoded_rate} Hz)."
        )
    _write_samples(samples, file_path, audio_format, decoded_rate)


def _detect_encoded_format(data: bytes) -> str:
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "wav"
    if data.startswith(b"ID3"):
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mp3"
    raise ValueError("Encoded audio must be WAV or MP3 bytes.")


def _decode_encoded_bytes(data: bytes) -> tuple[np.ndarray, int]:
    try:
        samples, decoded_rate = sf.read(
            io.BytesIO(data),
            dtype="float32",
            always_2d=False,
        )
    except sf.SoundFileError as error:
        raise RuntimeError("Could not decode the encoded audio.") from error

    if samples.size == 0:
        raise RuntimeError("Decoded audio is empty.")

    # MPEG decoding can slightly overshoot the [-1, 1] range. Clip before
    # PCM conversion so the WAV does not wrap.
    return np.clip(samples, -1.0, 1.0), int(decoded_rate)
