from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = PROJECT_ROOT / "generated"


def save_soundfile(audio, name: str, sample_rate: int) -> Path:
    filename = Path(name).name  # Prevent names from writing outside generated/.
    if not filename:
        raise ValueError("A filename must be provided.")

    if Path(filename).suffix.lower() != ".wav":
        filename += ".wav"

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    file_path = GENERATED_DIR / filename

    sf.write(str(file_path), audio, sample_rate)

    if not file_path.is_file():
        raise OSError(f"Audio file was not created: {file_path}")

    print(f"Saved WAV file: {file_path}")
    return file_path

