"""
Pitch-shift each TTS word onto one melody note.

The spoken duration of every word stays as the TTS engine produced it.
One word receives one note. A word that is sung on several notes in the
original song is flattened to that single note. Phoneme alignment is a
separate method.

The melody file is a local JSON list, one note per word, in the same order
as the word timings. Boulevard of Broken Dreams is copyrighted, so this
repository does not contain its vocal melody. If the melody file is missing,
this script writes a short stand-in contour so the method can be heard.
Replace those notes locally before judging melodic accuracy.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf

from helpers.save_soundfile import save_soundfile

SONG_DIR = PROJECT_ROOT / "songs" / "boulevard_of_broken_dreams"
DEFAULT_WORDS = SONG_DIR / "words.json"
DEFAULT_MELODY = SONG_DIR / "melody.json"
DEFAULT_NAME = "03_02_word_pitch_shift"
PAD_SECONDS = 0.25
FADE_SECONDS = 0.015
MAX_SEMITONES = 24.0
STAND_IN_NOTES = ("A3", "C4", "D4", "E4", "D4", "C4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shift each spoken word onto the note listed in a melody file."
    )
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--melody", type=Path, default=DEFAULT_MELODY)
    parser.add_argument(
        "--audio",
        type=Path,
        help="WAV file. Defaults to the path stored in the word-timing file.",
    )
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument(
        "--transpose",
        type=float,
        default=0.0,
        help="Extra semitones added to every melody note.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=0,
        help="Shift only the first N words. 0 uses the whole file.",
    )
    parser.add_argument("--play", action="store_true")
    return parser.parse_args()


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_helpers():
    pitch = _load_module(
        "shift_by_semitones",
        PROJECT_ROOT / "experiments" / "03_pitch_shifting" / "shift_by_semitones.py",
    )
    notes = _load_module(
        "playing_notes",
        PROJECT_ROOT / "experiments" / "01_basic_audio" / "playing_notes.py",
    )
    return pitch, notes


def load_words(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Word timings not found: {path}. "
            "Run experiments/02_text_to_speech/tts_with_word_boundaries.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    words = payload.get("words")
    if not isinstance(words, list) or not words:
        raise ValueError(f"No words in {path}")
    return payload


def load_notes(path: Path, word_count: int) -> list[str]:
    if not path.is_file():
        write_stand_in_melody(path, word_count)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "notes" in payload:
        notes = payload["notes"]
    elif "words" in payload:
        notes = [item["note"] for item in payload["words"]]
    else:
        raise ValueError(f"{path} needs a 'notes' list or a 'words' list with 'note'.")
    if len(notes) != word_count:
        raise ValueError(
            f"Melody has {len(notes)} notes but the TTS file has {word_count} words. "
            "One note is required for each word, in the same order."
        )
    return [str(note) for note in notes]


def write_stand_in_melody(path: Path, word_count: int) -> None:
    notes = [STAND_IN_NOTES[index % len(STAND_IN_NOTES)] for index in range(word_count)]
    payload = {
        "bpm": 84,
        "description": (
            "Stand-in contour for the word-level pitch experiment. "
            "One note per TTS word, in order. This is not a transcription "
            "of Boulevard of Broken Dreams. Replace the notes locally "
            "before judging whether the melody is correct."
        ),
        "notes": notes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote stand-in melody: {path}")
    print("These notes are not the vocal melody of Boulevard of Broken Dreams.")


def resolve_audio_path(args: argparse.Namespace, words_payload: dict) -> Path:
    if args.audio is not None:
        return args.audio
    stored = words_payload.get("audio")
    if not stored:
        raise ValueError("Pass --audio or store an audio path in the word-timing file.")
    path = Path(stored)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def sample_at(seconds: float, sample_rate: int, limit: int) -> int:
    return min(limit, max(0, int(round(seconds * sample_rate))))


def spoken_f0(pitch, audio: np.ndarray, sample_rate: int) -> float | None:
    if audio.size < int(0.03 * sample_rate):
        return None
    track, voiced = pitch.f0_track(audio, sample_rate)
    return pitch.median_voiced_f0(track, voiced)


def shift_word(
    pitch,
    audio: np.ndarray,
    sample_rate: int,
    start: int,
    end: int,
    semitones: float,
) -> np.ndarray:
    if abs(semitones) < 0.05:
        return audio[start:end].copy()

    pad = int(PAD_SECONDS * sample_rate)
    pad_start = max(0, start - pad)
    pad_end = min(audio.size, end + pad)
    shifted = pitch.pitch_shift_semitones(audio[pad_start:pad_end], sample_rate, semitones)
    rel_start = start - pad_start
    rel_end = rel_start + (end - start)
    segment = np.asarray(shifted[rel_start:rel_end], dtype=np.float32)
    if segment.size < end - start:
        segment = np.pad(segment, (0, end - start - segment.size))
    return segment[: end - start]


def place_segment(
    output: np.ndarray,
    original: np.ndarray,
    start: int,
    segment: np.ndarray,
    sample_rate: int,
    fade_out: bool,
) -> None:
    end = start + segment.size
    fade = min(int(FADE_SECONDS * sample_rate), segment.size // 3)
    body = segment.copy()
    if fade > 1:
        fade_in = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        body[:fade] = output[start:start + fade] * (1.0 - fade_in) + segment[:fade] * fade_in
        if fade_out:
            fade_out_ramp = np.linspace(1.0, 0.0, fade, endpoint=False, dtype=np.float32)
            body[-fade:] = (
                segment[-fade:] * fade_out_ramp
                + original[end - fade:end] * (1.0 - fade_out_ramp)
            )
    output[start:end] = body


def shift_to_melody(args: argparse.Namespace) -> Path:
    pitch, notes_module = load_helpers()
    words_payload = load_words(args.words)
    words = words_payload["words"]
    if args.max_words > 0:
        words = words[: args.max_words]
    notes = load_notes(args.melody, len(words_payload["words"]))
    notes = notes[: len(words)]

    audio_path = resolve_audio_path(args, words_payload)
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) != 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    sample_rate = int(sample_rate)
    original = np.asarray(audio, dtype=np.float32)
    output = original.copy()

    reference = spoken_f0(pitch, original, sample_rate)
    if reference is None:
        raise RuntimeError("No voiced pitch found in the recording.")

    print(f"Audio: {audio_path}")
    print(f"Reference F0: {reference:.2f} Hz")
    print(f"Words to shift: {len(words)}")
    print(
        f"{'index':>5} {'note':>6} {'spoken':>8} {'shift':>8} "
        f"{'after':>8} {'error':>8} {'f0 from':>10}"
    )

    report = []
    errors = []
    for index, (word, note_name) in enumerate(zip(words, notes)):
        start = sample_at(float(word["start"]), sample_rate, original.size)
        end = sample_at(float(word["end"]), sample_rate, original.size)
        if end <= start:
            continue

        target_hz = notes_module.note_to_frequency(note_name) * (
            2.0 ** (args.transpose / 12.0)
        )
        measured = spoken_f0(pitch, original[start:end], sample_rate)
        f0_source = "word"
        if measured is None:
            measured = reference
            f0_source = "fallback"

        semitones = float(12.0 * np.log2(target_hz / measured))
        clamped = abs(semitones) > MAX_SEMITONES
        semitones = float(np.clip(semitones, -MAX_SEMITONES, MAX_SEMITONES))
        segment = shift_word(pitch, original, sample_rate, start, end, semitones)

        next_start = None
        if index + 1 < len(words):
            next_start = float(words[index + 1]["start"])
        fade_out = next_start is None or abs(next_start - float(word["end"])) > 0.03
        place_segment(output, original, start, segment, sample_rate, fade_out)

        after = spoken_f0(pitch, output[start:end], sample_rate)
        error = None if after is None else float(12.0 * np.log2(after / target_hz))
        if error is not None:
            errors.append(abs(error))

        spoken_text = f"{measured:.1f}"
        after_text = "-" if after is None else f"{after:.1f}"
        error_text = "-" if error is None else f"{error:+.2f}"
        print(
            f"{index:5d} {note_name:>6} {spoken_text:>8} {semitones:+8.2f} "
            f"{after_text:>8} {error_text:>8} {f0_source:>10}"
        )
        report.append(
            {
                "index": index,
                "note": note_name,
                "target_hz": round(target_hz, 2),
                "spoken_hz": round(measured, 2),
                "f0_source": f0_source,
                "semitones": round(semitones, 3),
                "clamped": clamped,
                "measured_hz": None if after is None else round(after, 2),
                "error_semitones": None if error is None else round(error, 3),
            }
        )

    wav_path = save_soundfile(output, args.name, sample_rate)
    report_path = args.melody.with_name("word_pitch_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved report: {report_path}")
    if errors:
        print(
            f"Median absolute pitch error: {float(np.median(errors)):.2f} semitones "
            f"across {len(errors)} measured words."
        )
    return wav_path


def main() -> None:
    args = parse_args()
    wav_path = shift_to_melody(args)
    if not args.play:
        return
    import sounddevice as sd

    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    print("Playing word-pitched speech...")
    sd.play(audio, sample_rate)
    sd.wait()
    print("Done.")


if __name__ == "__main__":
    main()
