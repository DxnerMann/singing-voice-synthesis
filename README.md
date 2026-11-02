# Singing Voice Synthesis

This repository contains the software, hardware, experiments, and documentation developed in connection with a study thesis on the synthesis of singing voices from text and melody information.

The project investigates different approaches for generating synthetic singing voices and compares them with regard to pitch accuracy, timing, intelligibility, sound quality, implementation effort, and controllability.

The main focus is on two approaches:

1. **Speech-based synthesis**  
   Text is converted into speech using a Text-to-Speech (TTS) system. The resulting audio is then adapted to the given melody, for example by modifying pitch and timing.

2. **Phoneme-based synthesis**  
   Lyrics are decomposed into phonetic units and these units are explicitly assigned to musical notes and their durations.

An optional hardware demonstrator may also be developed. It can provide a simple interface for selecting and playing songs and may control a small robot synchronized with the synthesized singing.

## Project Goals

The main goals of the study project are:

- Investigate the fundamentals of speech and singing voice synthesis.
- Research existing approaches for synthetic singing voices.
- Implement and investigate suitable synthesis approaches.
- Generate singing voices from text and melody information.
- Evaluate the implemented approaches qualitatively.
- Compare the approaches regarding:
    - Pitch accuracy
    - Timing
    - Intelligibility
    - Sound quality
    - Implementation effort
    - Controllability
- Optionally demonstrate the results using a small robotic hardware system.

## Project Structure

```text
singing-voice-synthesis/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── docs/
│   ├── thesis/
│   │   ├── thesis_de.pdf
│   │   └── thesis_en.pdf
│   │
│   ├── research/
│   │   ├── sources.md
│   │
│   └── images/
│
├── src/
│   └── singing_synthesis/
│
├── songs/
│   └── ...
│
├── hardware/
│
└── experiments/
```

### `docs/`

Contains documentation related to the study project.

- `thesis/` contains the current version of the study thesis and other relevant documents.
- `research/` contains research notes, literature references, and collected information.
- `images/` contains diagrams, photographs, and other documentation images.

Two public versions of the study thesis are provided:
```text
docs/
└── thesis/
├── thesis_de/
└── thesis_en/
```

> [!NOTE]
> The German version is the original version of the study thesis. However, some information may have been removed from the public version.

> [!WARNING]
> The English version was automatically translated and may therefore contain translation errors or inaccuracies.


### `src/`

Contains the implementation of the singing voice synthesis system.

### `songs/`

Contains song-specific input data and generated results.

A song may contain:

- Lyrics
- Melody and note information
- Generated audio
- Configuration and metadata

### `hardware/`

Contains all components related to the optional hardware demonstrator.

This may include:

- Electronic components and wiring
- Controller software
- Display and button control
- Servo and motor control
- 3D models and CAD files
- Circuit diagrams
- Assembly documentation

The hardware demonstrator is optional and is not the primary focus of the study project.

### `experiments/`

Contains experimental implementations and evaluation material.

Experiments may include:

- TTS experiments
- Pitch-shifting experiments
- Timing and duration experiments
- Phoneme-based synthesis experiments
- Evaluation results and comparisons

Experimental code does not necessarily represent the final implementation.

## Hardware Demonstrator

An optional hardware demonstrator may consist of a small controller with a display and buttons for selecting and starting songs.

```text
+-------------------+
|      Display      |
|   Song Selection  |
+---------+---------+
          |
    +-----+-----+
    |  Buttons  |
    +-----+-----+
          |
          v
+-------------------+
|     Controller    |
|   Raspberry Pi    |
+--------+----------+
         |
    +----+----+
    |         |
    v         v
 Speaker    Robot
            Movement
```

The intended interaction is to select a stored song using the display and buttons. Starting the song should trigger both audio playback and synchronized robot movement.

The hardware implementation may be extended or reduced depending on the progress of the core singing voice synthesis project.

## Development

The main software components are expected to be implemented in **Python**, particularly for audio processing, speech synthesis, experimentation, and evaluation.

The hardware demonstrator may also use Python for controller and hardware-related software, depending on the selected hardware platform.

The exact technologies and libraries will be determined during the implementation phase.

## License

The source code developed as part of this project is licensed under the MIT License.

Third-party software, libraries, models, datasets, and other external materials are subject to their respective licenses.