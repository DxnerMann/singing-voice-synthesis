"""
Shift pitch in semitones without changing the duration.

The shift uses a phase vocoder: the signal is time-stretched, then resampled
so the duration returns to the original length while the frequencies move.
One step is one semitone. n semitones multiply every frequency by 2^(n/12).

The whole spectrum moves together, including the vocal formants. Large shifts
therefore change the voice character as well as the pitch. This experiment
records that result; it does not correct the formants.

The sine tone is the control. Its frequency is known, so the measured shift
can be checked exactly. The speech part uses the online TTS recording.
"""

import argparse
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import numpy as np
import soundfile as sf

from helpers.save_soundfile import save_soundfile

DEFAULT_TTS_FILE = PROJECT_ROOT / "generated" / "02_00_online_tts.wav"
DEFAULT_SEMITONES = (-5.0, 7.0, 12.0)
SINE_FREQUENCY = 440.0
SINE_DURATION = 2.0
SINE_SAMPLE_RATE = 44100
ANALYSIS_SECONDS = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pitch-shift a sine control tone and the online TTS recording."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_TTS_FILE,
        help="Speech WAV file. Defaults to the online TTS result.",
    )
    parser.add_argument(
        "--semitones",
        type=float,
        nargs="+",
        default=list(DEFAULT_SEMITONES),
        help="Pitch shifts in semitones. Positive values move the pitch up.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Use only the first N seconds of the speech file. 0 uses the full file.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play each shifted speech file after saving it.",
    )
    return parser.parse_args()


def pitch_shift_semitones(audio, sample_rate: int, semitones: float) -> np.ndarray:
    """Return a copy of ``audio`` shifted by ``semitones`` at the same duration."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("Audio must be mono.")
    if samples.size == 0:
        raise ValueError("Audio must not be empty.")
    if semitones == 0:
        return samples.copy()

    with warnings.catch_warnings():
        # librosa/numba can emit a harmless cast warning during the STFT.
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in cast",
            category=RuntimeWarning,
        )
        shifted = librosa.effects.pitch_shift(
            samples,
            sr=sample_rate,
            n_steps=float(semitones),
        )
    shifted = np.asarray(shifted, dtype=np.float32)
    if shifted.shape != samples.shape:
        raise RuntimeError(
            f"Pitch shift changed the length from {samples.shape[-1]} "
            f"to {shifted.shape[-1]} samples."
        )
    return _match_peak(shifted, samples)


def _match_peak(shifted: np.ndarray, original: np.ndarray) -> np.ndarray:
    original_peak = float(np.max(np.abs(original)))
    shifted_peak = float(np.max(np.abs(shifted)))
    if original_peak == 0.0 or shifted_peak == 0.0:
        return shifted
    return shifted * (original_peak / shifted_peak)


def peak_frequency(audio, sample_rate: int) -> float:
    """Frequency of the strongest spectral peak, in Hz."""
    samples = np.asarray(audio, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    bin_index = int(np.argmax(spectrum[1:]) + 1)
    if 0 < bin_index < spectrum.size - 1:
        alpha = spectrum[bin_index - 1]
        beta = spectrum[bin_index]
        gamma = spectrum[bin_index + 1]
        denominator = alpha - 2.0 * beta + gamma
        delta = 0.5 * (alpha - gamma) / denominator if denominator != 0.0 else 0.0
    else:
        delta = 0.0
    return (bin_index + delta) * sample_rate / samples.size


def f0_track(audio, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Voiced fundamental-frequency track for the shared analysis window."""
    window = _analysis_window(audio, sample_rate)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in cast",
            category=RuntimeWarning,
        )
        f0, voiced_flag, _voiced_probability = librosa.pyin(
            window,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C6")),
            sr=sample_rate,
        )
    return np.asarray(f0, dtype=np.float64), np.asarray(voiced_flag, dtype=bool)


def median_voiced_f0(f0: np.ndarray, voiced: np.ndarray) -> float | None:
    values = f0[voiced & np.isfinite(f0) & (f0 > 0)]
    if values.size == 0:
        return None
    return float(np.median(values))


def frame_shift_semitones(
    source_f0: np.ndarray,
    source_voiced: np.ndarray,
    shifted_f0: np.ndarray,
    shifted_voiced: np.ndarray,
) -> float | None:
    """Median semitone shift on frames that are voiced before and after."""
    both = (
        source_voiced
        & shifted_voiced
        & np.isfinite(source_f0)
        & np.isfinite(shifted_f0)
        & (source_f0 > 0)
        & (shifted_f0 > 0)
    )
    if int(np.count_nonzero(both)) < 10:
        return None
    shifts = 12.0 * np.log2(shifted_f0[both] / source_f0[both])
    return float(np.median(shifts))


def _analysis_window(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    window_samples = min(audio.size, int(ANALYSIS_SECONDS * sample_rate))
    start = max(0, (audio.size - window_samples) // 2)
    return np.ascontiguousarray(audio[start:start + window_samples])


def semitone_error(measured_hz: float, reference_hz: float, requested: float) -> float:
    measured_shift = 12.0 * np.log2(measured_hz / reference_hz)
    return float(measured_shift - requested)


def format_shift(semitones: float) -> str:
    if semitones == int(semitones):
        return f"{int(semitones):+d}"
    return f"{semitones:+.2f}"


def shift_name(prefix: str, semitones: float) -> str:
    if semitones == int(semitones):
        amount = str(abs(int(semitones)))
    else:
        amount = f"{abs(semitones):.2f}".replace(".", "p")
    direction = "plus" if semitones > 0 else "minus" if semitones < 0 else "zero"
    return f"{prefix}_{direction}_{amount}_semitones"


def load_speech(path: Path, seconds: float) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Speech file not found: {path}")
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim != 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    if seconds < 0:
        raise ValueError("--seconds must be 0 or positive.")
    if seconds > 0:
        audio = audio[:int(seconds * sample_rate)]
    if audio.size < int(0.5 * sample_rate):
        raise ValueError("Need at least 0.5 seconds of speech to pitch-shift.")
    return audio, int(sample_rate)


def run_sine_control(semitones: list[float]) -> None:
    sample_count = int(SINE_SAMPLE_RATE * SINE_DURATION)
    time = np.linspace(0, SINE_DURATION, sample_count, endpoint=False)
    sine = np.sin(2 * np.pi * SINE_FREQUENCY * time).astype(np.float32)

    print()
    print(
        f"Sine control: {SINE_FREQUENCY:.2f} Hz, "
        f"{SINE_DURATION:.1f} s, {SINE_SAMPLE_RATE} Hz"
    )
    print(
        f"{'shift':>8} {'target Hz':>12} {'measured Hz':>12} "
        f"{'error st':>10} {'duration s':>12}"
    )
    for steps in semitones:
        shifted = pitch_shift_semitones(sine, SINE_SAMPLE_RATE, steps)
        measured_hz = peak_frequency(shifted, SINE_SAMPLE_RATE)
        target_hz = SINE_FREQUENCY * (2.0 ** (steps / 12.0))
        error = semitone_error(measured_hz, SINE_FREQUENCY, steps)
        duration = shifted.size / SINE_SAMPLE_RATE
        print(
            f"{format_shift(steps):>8} {target_hz:12.2f} {measured_hz:12.2f} "
            f"{error:10.3f} {duration:12.3f}"
        )
        if steps != 0:
            save_soundfile(shifted, shift_name("03_00_sine", steps), SINE_SAMPLE_RATE)


def run_speech(path: Path, semitones: list[float], seconds: float, play: bool) -> None:
    audio, sample_rate = load_speech(path, seconds)
    source_track, source_voiced = f0_track(audio, sample_rate)
    source_f0 = median_voiced_f0(source_track, source_voiced)
    duration = audio.size / sample_rate

    print()
    print(f"Speech: {path}")
    print(f"Duration: {duration:.2f} s, sample rate: {sample_rate} Hz")
    if source_f0 is None:
        print("No voiced fundamental frequency found in the analysis window.")
    else:
        print(f"Median F0 in the analysis window: {source_f0:.2f} Hz")
    print("Measured shift compares the same voiced frames before and after.")
    print(
        f"{'shift':>8} {'F0 Hz':>10} {'measured st':>12} "
        f"{'error st':>10} {'duration s':>12}"
    )

    for steps in semitones:
        shifted = pitch_shift_semitones(audio, sample_rate, steps)
        shifted_track, shifted_voiced = f0_track(shifted, sample_rate)
        shifted_f0 = median_voiced_f0(shifted_track, shifted_voiced)
        shifted_duration = shifted.size / sample_rate
        measured = frame_shift_semitones(
            source_track,
            source_voiced,
            shifted_track,
            shifted_voiced,
        )
        if measured is None or steps == 0:
            error = float("nan")
        else:
            error = measured - steps
        f0_text = "-" if shifted_f0 is None else f"{shifted_f0:.2f}"
        measured_text = "-" if measured is None else f"{measured:.3f}"
        error_text = "-" if not np.isfinite(error) else f"{error:.3f}"
        print(
            f"{format_shift(steps):>8} {f0_text:>10} {measured_text:>12} "
            f"{error_text:>10} {shifted_duration:12.3f}"
        )
        if steps == 0:
            continue
        save_soundfile(shifted, shift_name("03_01_tts", steps), sample_rate)
        if play:
            import sounddevice as sd

            print(f"Playing {format_shift(steps)} semitones...")
            sd.play(shifted, sample_rate)
            sd.wait()


def main() -> None:
    args = parse_args()
    print("Pitch shift by phase vocoder. Duration stays the same.")
    print("n semitones multiply frequency by 2^(n/12).")
    run_sine_control(args.semitones)
    run_speech(args.input, args.semitones, args.seconds, args.play)


if __name__ == "__main__":
    main()
