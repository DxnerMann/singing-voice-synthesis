import re
import numpy as np
import sounddevice as sd
from helpers.save_soundfile import save_soundfile

SAMPLE_RATE = 44100

def note_to_frequency(note_name):
    """
    Converts a musical note string (e.g., 'A4', 'C#5', 'Bb3')
    to its frequency in Hz using the standard formula:
    f = 440 * 2^(n / 12)
    """
    match = re.match(r"^([A-Ga-g][#b]?)(-?\d+)$", note_name.strip())
    if not match:
        raise ValueError(f"Invalid note format: '{note_name}'. Use format like 'A4', 'C#5', or 'Bb3'.")

    note, octave = match.groups()
    octave = int(octave)

    # Semitone offsets from C
    note_offsets = {
        'C': 0, 'C#': 1, 'Db': 1,
        'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4,
        'F': 5, 'F#': 6, 'Gb': 6,
        'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10,
        'B': 11
    }

    semitone_from_c = note_offsets[note.capitalize()]

    # Calculate semitone distance from A4
    semitones_from_c0 = octave * 12 + semitone_from_c
    a4_semitones_from_c0 = 4 * 12 + 9
    semitones_from_a4 = semitones_from_c0 - a4_semitones_from_c0

    # Frequency formula
    return 440.0 * (2.0 ** (semitones_from_a4 / 12.0))


def note_length_to_seconds(note_length, bpm):
    """
    Converts a relative note length (1.0 = whole, 0.5 = half, 0.25 = quarter)
    into seconds based on BPM (where 1 beat = 1 quarter note).
    """
    seconds_per_beat = 60.0 / bpm
    seconds_per_whole_note = seconds_per_beat * 4.0
    return note_length * seconds_per_whole_note


def generate_note(note_name, duration_sec, sample_rate):
    freq = note_to_frequency(note_name)
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Generate sine wave
    audio = np.sin(2 * np.pi * freq * t)

    # Apply a tiny fade-in/fade-out envelope (10ms) to avoid clicking between repeated notes
    fade_len = min(int(sample_rate * 0.01), num_samples // 2)
    if fade_len > 0:
        fade = np.linspace(0, 1, fade_len)
        audio[:fade_len] *= fade
        audio[-fade_len:] *= fade[::-1]

    return audio


def generate_melody(sequence, bpm, sample_rate):
    audio_pieces = []
    for note_name, note_length in sequence:
        duration_sec = note_length_to_seconds(note_length, bpm)
        print(f"Generating: {note_name:<3} | Length: {note_length:<4} ({duration_sec:.2f}s) | {note_to_frequency(note_name):.2f} Hz")

        note_audio = generate_note(note_name, duration_sec, sample_rate)
        audio_pieces.append(note_audio)

    return np.concatenate(audio_pieces)


def main():
    bpm = 150

    melody_sequence = [
        ("C4", 0.25), ("D4", 0.25), ("E4", 0.25), ("F4", 0.25),
        ("G4", 0.5),  ("G4", 0.5),

        ("A4", 0.25), ("A4", 0.25), ("A4", 0.25), ("A4", 0.25),
        ("G4", 1.0),

        ("A4", 0.25), ("A4", 0.25), ("A4", 0.25), ("A4", 0.25),
        ("G4", 1.0),

        ("F4", 0.25), ("F4", 0.25), ("F4", 0.25), ("F4", 0.25),
        ("E4", 0.5),  ("E4", 0.5),

        ("D4", 0.25), ("D4", 0.25), ("D4", 0.25), ("D4", 0.25),
        ("C4", 1.0),
    ]

    audio = generate_melody(melody_sequence, bpm, SAMPLE_RATE)

    save_soundfile(audio, "01_01_alle_meine_entchen", SAMPLE_RATE)

    print("Playing melody...")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
    print("Done.")


if __name__ == "__main__":
    main()