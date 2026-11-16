"""Synthesize speech with Microsoft Edge's online TTS and store WAV plus MP3.

The service returns MP3. The MP3 is kept as the original service output.
A WAV copy is written for later pitch and timing experiments.

Boulevard of Broken Dreams by Green Day is the planned musical test piece.
Its lyrics are copyrighted and are not stored in this repository. Put a local
lyrics file at songs/boulevard_of_broken_dreams/lyrics.txt, or pass --text-file.

Run from the project root:

    python experiments/02_text_to_speech/tts_with_online_service.py
    python experiments/02_text_to_speech/tts_with_online_service.py --no-play
    python experiments/02_text_to_speech/tts_with_online_service.py \\
        --text-file songs/boulevard_of_broken_dreams/lyrics.txt
"""

import argparse
import asyncio
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

DEFAULT_LYRICS_FILE = PROJECT_ROOT / "songs" / "boulevard_of_broken_dreams" / "lyrics.txt"
DEFAULT_VOICE = "en-US-GuyNeural"
DEFAULT_NAME = "02_00_online_tts"
DEFAULT_TEXT = (
    "Hello. This is a spoken test sentence for the singing voice synthesis experiments."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert text to speech with the online Edge TTS service."
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
        help="File name without directory. Saved under generated/.",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Save the audio without playing it.",
    )
    return parser.parse_args()


def resolve_text(args: argparse.Namespace) -> tuple[str, str]:
    if args.text is not None:
        text = args.text.strip()
        source = "command line"
    elif args.text_file is not None:
        text = _read_text_file(args.text_file)
        source = str(args.text_file)
    elif DEFAULT_LYRICS_FILE.is_file():
        text = _read_text_file(DEFAULT_LYRICS_FILE)
        source = str(DEFAULT_LYRICS_FILE)
    else:
        text = DEFAULT_TEXT
        source = "built-in test sentence"
        print(
            "No local lyrics file found at "
            f"{DEFAULT_LYRICS_FILE.relative_to(PROJECT_ROOT)}."
        )
        print(
            "Using a short original test sentence. Boulevard of Broken Dreams "
            "is the planned test song; keep its lyrics outside the repository."
        )

    if not text:
        raise ValueError("Text must not be empty.")
    return text, source


def _read_text_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Text file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


async def _collect_mp3(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    buffer = bytearray()
    async for message in communicate.stream():
        if message["type"] == "audio":
            buffer.extend(message["data"])
    return bytes(buffer)


def synthesize_mp3(text: str, voice: str) -> bytes:
    try:
        mp3_bytes = asyncio.run(_collect_mp3(text, voice))
    except EdgeTTSException as error:
        raise RuntimeError(
            f"The online TTS service failed for voice {voice!r}: {error}"
        ) from error
    except aiohttp.ClientError as error:
        raise RuntimeError(
            "Could not reach the online TTS service. Check the internet connection."
        ) from error

    if not mp3_bytes:
        raise RuntimeError("The online TTS service returned no audio.")
    return mp3_bytes


def main() -> None:
    args = parse_args()
    text, source = resolve_text(args)
    print(f"Text source: {source}")
    print(f"Voice: {args.voice}")

    mp3_bytes = synthesize_mp3(text, args.voice)
    save_soundfile(mp3_bytes, args.name, audio_format="mp3")
    wav_path = save_soundfile(mp3_bytes, args.name, audio_format="wav")

    info = sf.info(str(wav_path))
    print(f"Sample rate: {info.samplerate} Hz")
    print(f"Duration: {info.duration:.2f} s")

    if args.no_play:
        return

    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    print("Playing synthesized speech...")
    sd.play(audio, sample_rate)
    sd.wait()
    print("Done.")


if __name__ == "__main__":
    main()
