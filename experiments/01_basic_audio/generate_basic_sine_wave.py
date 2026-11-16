import numpy as np
import sounddevice as sd
from helpers.save_soundfile import save_soundfile

SAMPLE_RATE = 44100
FREQUENCY = 440
DURATION = 5

def generate_sine_wave(frequency, duration, sample_rate):
    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False
    )

    return np.sin(2 * np.pi * frequency * t)


def main():
    audio = generate_sine_wave(
        FREQUENCY,
        DURATION,
        SAMPLE_RATE
    )

    save_soundfile(audio, "01_00_basic_sine_wave", SAMPLE_RATE)

    print("Playing 440 Hz...")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
    print("Done.")


if __name__ == "__main__":
    main()