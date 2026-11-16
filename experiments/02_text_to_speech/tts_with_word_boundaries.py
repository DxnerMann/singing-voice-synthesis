"""
Synthesize speech and save one timing entry per word.

This is separate from tts_with_online_service.py. That script only stores
audio. This one asks the Edge TTS service for WordBoundary events and writes
them next to the song data.

Offsets from the service are in 100-nanosecond ticks. Divide by 10_000_000
to get seconds. The word list is the segmentation used by the word-level
pitch-shift experiment. It is not a phoneme alignment.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import aiohttp
import edge_tts
import sounddevice as sd
import soundfile as sf
from edge_tts.exceptions import EdgeTTSException

from helpers.save_soundfile import save_soundfile

SONG_DIR = PROJECT_ROOT / "songs" / "boulevard_of_broken_dreams"
DEFAULT_LYRICS_FILE = SONG_DIR / "lyrics.txt"
DEFAULT_WORDS_FILE = SONG_DIR / "words.json"
DEFAULT_VOICE = "en-US-GuyNeural"
DEFAULT_NAME = "02_01_online_tts_words"
DEFAULT_TEXT = (
    "Hello. This is a spoken test sentence for the singing voice synthesis experiments."
)
TICKS_PER_SECOND = 10_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online TTS that also stores word start and end times."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to synthesize.")
    source.add_argument(
        "--text-file",
        type=Path,
        help="UTF-8 text file to synthesize. Copyrighted lyrics must stay local.",
    )
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Edge TTS voice name.")
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help="Audio file name without a directory. Saved under generated/.",
    )
    parser.add_argument(
        "--words-out",
        type=Path,
        help="Where to write the word timings. Defaults to the song folder.",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Save the audio without playing it.",
    )
    return parser.parse_args()


def resolve_text(args: argparse.Namespace) -> tuple[str, str, bool]:
    if args.text is not None:
        text = args.text.strip()
        return text, "command line", False
    if args.text_file is not None:
        return _read_text_file(args.text_file), str(args.text_file), False
    if DEFAULT_LYRICS_FILE.is_file():
        return _read_text_file(DEFAULT_LYRICS_FILE), str(DEFAULT_LYRICS_FILE), True

    print(f"No local lyrics file found at {DEFAULT_LYRICS_FILE.relative_to(PROJECT_ROOT)}.")
    print("Using a short original test sentence.")
    return DEFAULT_TEXT, "built-in test sentence", False


def _read_text_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Text file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Text file is empty: {path}")
    return text


def words_output_path(args: argparse.Namespace, using_song_lyrics: bool) -> Path:
    if args.words_out is not None:
        return args.words_out
    if using_song_lyrics:
        return DEFAULT_WORDS_FILE
    return PROJECT_ROOT / "generated" / f"{args.name}.words.json"


async def _collect(text: str, voice: str) -> tuple[bytes, list[dict]]:
    communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    audio = bytearray()
    words = []
    async for message in communicate.stream():
        if message["type"] == "audio":
            audio.extend(message["data"])
        elif message["type"] == "WordBoundary":
            start = float(message["offset"]) / TICKS_PER_SECOND
            end = float(message["offset"] + message["duration"]) / TICKS_PER_SECOND
            words.append(
                {
                    "text": message["text"],
                    "start": round(start, 4),
                    "end": round(end, 4),
                }
            )
    return bytes(audio), words


def synthesize(text: str, voice: str) -> tuple[bytes, list[dict]]:
    try:
        audio, words = asyncio.run(_collect(text, voice))
    except EdgeTTSException as error:
        raise RuntimeError(
            f"The online TTS service failed for voice {voice!r}: {error}"
        ) from error
    except aiohttp.ClientError as error:
        raise RuntimeError(
            "Could not reach the online TTS service. Check the internet connection."
        ) from error

    if not audio:
        raise RuntimeError("The online TTS service returned no audio.")
    if not words:
        raise RuntimeError("The online TTS service returned no word boundaries.")
    return audio, words


def save_words(path: Path, voice: str, audio_path: Path, words: list[dict], duration: float, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "voice": voice,
        "audio": str(audio_path.relative_to(PROJECT_ROOT)),
        "sample_rate": sample_rate,
        "duration": round(duration, 4),
        "words": words,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved word timings: {path}")
    print(f"Words: {len(words)}")


def main() -> None:
    args = parse_args()
    text, source, using_song_lyrics = resolve_text(args)
    print(f"Text source: {source}")
    print(f"Voice: {args.voice}")

    mp3_bytes, words = synthesize(text, args.voice)
    save_soundfile(mp3_bytes, args.name, audio_format="mp3")
    wav_path = save_soundfile(mp3_bytes, args.name, audio_format="wav")

    info = sf.info(str(wav_path))
    print(f"Sample rate: {info.samplerate} Hz")
    print(f"Duration: {info.duration:.2f} s")
    save_words(words_output_path(args, using_song_lyrics), args.voice, wav_path, words, info.duration, info.samplerate)

    last_end = words[-1]["end"]
    if last_end > info.duration + 0.05:
        print(
            f"Warning: last word ends at {last_end:.3f} s, "
            f"after the WAV duration of {info.duration:.3f} s."
        )

    if args.no_play:
        return

    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    print("Playing synthesized speech...")
    sd.play(audio, sample_rate)
    sd.wait()
    print("Done.")


if __name__ == "__main__":
    main()
