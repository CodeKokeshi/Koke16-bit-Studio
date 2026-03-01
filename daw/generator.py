"""Procedural music generator – genre-aware, multi-track, loop-friendly.

Generates 4 tracks (Lead, Bass, Harmony, Drums) following music-theory
conventions for each supported genre.  Every generation is randomised
so no two outputs are the same, yet the results always loop cleanly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from daw.models import NoteEvent

# ── Note / scale helpers ────────────────────────────────────────────

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_SCALE_INTERVALS: dict[str, list[int]] = {
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "natural_minor":    [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":   [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":    [0, 2, 3, 5, 7, 9, 11],
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":       [0, 2, 4, 5, 7, 9, 10],
    "phrygian":         [0, 1, 3, 5, 7, 8, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
    "chromatic":        list(range(12)),
    "whole_tone":       [0, 2, 4, 6, 8, 10],
    "diminished":       [0, 2, 3, 5, 6, 8, 9, 11],
}

# Chord quality intervals (from root)
_CHORD_TYPES: dict[str, list[int]] = {
    "maj":   [0, 4, 7],
    "min":   [0, 3, 7],
    "dim":   [0, 3, 6],
    "aug":   [0, 4, 8],
    "sus4":  [0, 5, 7],
    "7":     [0, 4, 7, 10],
    "m7":    [0, 3, 7, 10],
    "maj7":  [0, 4, 7, 11],
    "dim7":  [0, 3, 6, 9],
}


def _build_scale(root: int, scale_name: str) -> list[int]:
    """Return all MIDI notes in a scale across the playable range."""
    intervals = _SCALE_INTERVALS[scale_name]
    notes: list[int] = []
    for octave in range(11):
        for iv in intervals:
            midi = root + octave * 12 + iv
            if 0 <= midi <= 127:
                notes.append(midi)
    return sorted(set(notes))


def _snap_to_scale(midi_note: int, scale_notes: list[int]) -> int:
    """Snap a MIDI note to the nearest note in the scale."""
    best = min(scale_notes, key=lambda s: abs(s - midi_note))
    return best


def _chord_notes(root_midi: int, quality: str) -> list[int]:
    """Return MIDI notes for a chord voicing."""
    return [root_midi + iv for iv in _CHORD_TYPES[quality]]


# ── Genre configurations ────────────────────────────────────────────

@dataclass
class GenreConfig:
    """Full spec for how a genre should sound."""
    name: str
    bpm_range: tuple[int, int]
    bars: int                        # number of bars per loop
    ticks_per_beat: int = 4
    beats_per_bar: int = 4

    # Tonality
    root_choices: list[int] = field(default_factory=lambda: list(range(12)))
    scale_choices: list[str] = field(default_factory=lambda: ["major"])

    # Chord progressions (list of alternative progressions; each is a list of
    # (scale_degree, chord_quality) per bar).  Degree is 0-based.
    progressions: list[list[tuple[int, str]]] = field(default_factory=list)

    # Rhythm patterns per role: list of (tick_offset_in_bar, length_ticks)
    # If empty, generator uses a default pattern.
    lead_rhythm: list[list[tuple[int, int]]] = field(default_factory=list)
    bass_rhythm: list[list[tuple[int, int]]] = field(default_factory=list)
    drum_rhythm: list[list[tuple[int, int]]] = field(default_factory=list)

    # Instrument assignments  (instrument_name, waveform)
    lead_instrument: tuple[str, str] = ("Generic Saw", "sawtooth")
    bass_instrument: tuple[str, str] = ("Generic Triangle", "triangle")
    harmony_instrument: tuple[str, str] = ("Generic Square", "square")
    drum_instrument: tuple[str, str] = ("Generic Noise Drum", "noise")

    # Velocity ranges
    lead_vel: tuple[int, int] = (85, 110)
    bass_vel: tuple[int, int] = (80, 100)
    harmony_vel: tuple[int, int] = (60, 85)
    drum_vel: tuple[int, int] = (90, 120)

    # Track mix-volume ranges (0.0 - 1.0)
    lead_mix_range: tuple[float, float] = (0.70, 0.92)
    bass_mix_range: tuple[float, float] = (0.62, 0.82)
    harmony_mix_range: tuple[float, float] = (0.45, 0.70)
    drum_mix_range: tuple[float, float] = (0.58, 0.86)

    # Pitch ranges  (MIDI)
    lead_range: tuple[int, int] = (60, 84)
    bass_range: tuple[int, int] = (36, 55)
    harmony_range: tuple[int, int] = (48, 72)
    drum_pitches: list[int] = field(default_factory=lambda: [36, 38, 42, 46])

    # Melody behaviour
    lead_step_max: int = 4          # max scale-step jump per note
    lead_rest_prob: float = 0.1     # probability of a rest instead of a note


# Helper to build tick-offset patterns
def _simple_rhythm(tpb: int, bpb: int, divisions: list[float],
                   lengths: list[float]) -> list[tuple[int, int]]:
    """Build a bar-length rhythm pattern from beat-relative divisions.

    *divisions* is a list of beat-positions (0-based, can be fractional).
    *lengths* each note's length in beats.
    """
    pattern: list[tuple[int, int]] = []
    for pos, dur in zip(divisions, lengths):
        tick_start = int(pos * tpb)
        tick_len = max(1, int(dur * tpb))
        pattern.append((tick_start, tick_len))
    return pattern


# ---------- genre definitions ----------

_TPB = 4  # default ticks per beat
_BPB = 4  # 4/4 time

def _genre_happy() -> GenreConfig:
    return GenreConfig(
        name="Happy",
        bpm_range=(115, 140),
        bars=8,
        scale_choices=["major", "mixolydian", "pentatonic_major"],
        progressions=[
            # I – V – vi – IV  (most popular pop progression)
            [(0, "maj"), (4, "maj"), (5, "min"), (3, "maj")] * 2,
            # I – IV – V – V
            [(0, "maj"), (3, "maj"), (4, "maj"), (4, "maj")] * 2,
            # I – vi – IV – V
            [(0, "maj"), (5, "min"), (3, "maj"), (4, "maj")] * 2,
        ],
        lead_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 3, 3.5],
                           [0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5]),
            _simple_rhythm(_TPB, _BPB, [0, 1, 2, 2.5, 3],
                           [1, 1, 0.5, 0.5, 1]),
        ],
        bass_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 2], [2, 2]),
            _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3], [1, 1, 1, 1]),
        ],
        drum_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25]),
            _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                           [0.25] * 8),
        ],
        lead_instrument=("NES Square", "square"),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Generic Pulse 25%", "pulse25"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(90, 115),
        lead_range=(60, 84),
        bass_range=(36, 55),
        lead_step_max=3,
        lead_rest_prob=0.08,
    )


def _genre_calm() -> GenreConfig:
    return GenreConfig(
        name="Calm",
        bpm_range=(70, 95),
        bars=8,
        scale_choices=["major", "pentatonic_major", "mixolydian"],
        progressions=[
            # I – IV – I – V
            [(0, "maj"), (3, "maj"), (0, "maj"), (4, "maj")] * 2,
            # I – vi – IV – V
            [(0, "maj"), (5, "min"), (3, "maj"), (4, "maj")] * 2,
            # I – iii – vi – IV
            [(0, "maj"), (2, "min"), (5, "min"), (3, "maj")] * 2,
        ],
        lead_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 2, 3], [2, 1, 1]),
            _simple_rhythm(_TPB, _BPB, [0, 1.5, 3], [1.5, 1.5, 1]),
        ],
        bass_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0], [4]),
            _simple_rhythm(_TPB, _BPB, [0, 2], [2, 2]),
        ],
        drum_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 2], [0.25, 0.25]),
        ],
        lead_instrument=("Generic Sine", "sine"),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Generic Sine", "sine"),
        drum_instrument=("Generic Noise Drum", "noise"),
        lead_vel=(60, 85),
        bass_vel=(55, 75),
        harmony_vel=(45, 65),
        drum_vel=(40, 60),
        lead_range=(60, 79),
        bass_range=(40, 55),
        lead_step_max=2,
        lead_rest_prob=0.2,
    )


def _genre_sad() -> GenreConfig:
    return GenreConfig(
        name="Sad",
        bpm_range=(60, 85),
        bars=8,
        scale_choices=["natural_minor", "harmonic_minor", "dorian"],
        progressions=[
            # i – iv – v – i
            [(0, "min"), (3, "min"), (4, "min"), (0, "min")] * 2,
            # i – VI – III – VII
            [(0, "min"), (5, "maj"), (2, "maj"), (6, "maj")] * 2,
            # i – iv – VI – V
            [(0, "min"), (3, "min"), (5, "maj"), (4, "maj")] * 2,
        ],
        lead_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3], [1, 1, 1, 1]),
            _simple_rhythm(_TPB, _BPB, [0, 2], [2, 2]),
        ],
        bass_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0], [4]),
            _simple_rhythm(_TPB, _BPB, [0, 2], [2, 2]),
        ],
        drum_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 2], [0.25, 0.25]),
        ],
        lead_instrument=("Generic Sine", "sine"),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Generic Saw", "sawtooth"),
        drum_instrument=("Generic Noise Drum", "noise"),
        lead_vel=(65, 90),
        bass_vel=(60, 80),
        harmony_vel=(50, 70),
        drum_vel=(45, 65),
        lead_range=(55, 79),
        bass_range=(36, 52),
        lead_step_max=3,
        lead_rest_prob=0.15,
    )


def _genre_horror() -> GenreConfig:
    return GenreConfig(
        name="Horror",
        bpm_range=(75, 110),
        bars=8,
        scale_choices=["phrygian", "harmonic_minor", "diminished", "chromatic"],
        progressions=[
            # i – bII – i – v
            [(0, "min"), (1, "maj"), (0, "min"), (4, "dim")] * 2,
            # i – bVI – bVII – i
            [(0, "min"), (5, "maj"), (6, "maj"), (0, "min")] * 2,
            # i – iv – bII – i
            [(0, "min"), (3, "min"), (1, "maj"), (0, "min")] * 2,
            # tritone-heavy: i – bV – iv – i
            [(0, "dim"), (6, "dim"), (3, "min"), (0, "dim")] * 2,
        ],
        lead_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 1.5, 3], [1.5, 1.5, 1]),
            _simple_rhythm(_TPB, _BPB, [0, 2, 3.5], [2, 1.5, 0.5]),
        ],
        bass_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0], [4]),
            _simple_rhythm(_TPB, _BPB, [0, 3], [3, 1]),
        ],
        drum_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 1.5, 3], [0.25, 0.25, 0.25]),
            _simple_rhythm(_TPB, _BPB, [0], [0.25]),  # sparse single hit
        ],
        lead_instrument=("Generic Saw", "sawtooth"),
        bass_instrument=("Generic Sine", "sine"),
        harmony_instrument=("Generic Pulse 25%", "pulse25"),
        drum_instrument=("Generic Noise Drum", "noise"),
        lead_vel=(70, 100),
        bass_vel=(60, 85),
        harmony_vel=(40, 65),
        drum_vel=(50, 80),
        lead_range=(48, 76),
        bass_range=(30, 50),
        lead_step_max=5,
        lead_rest_prob=0.25,
    )


def _genre_epic() -> GenreConfig:
    return GenreConfig(
        name="Epic",
        bpm_range=(125, 160),
        bars=8,
        scale_choices=["natural_minor", "harmonic_minor", "dorian"],
        progressions=[
            # i – III – VII – VI  (epic cinematic)
            [(0, "min"), (2, "maj"), (6, "maj"), (5, "maj")] * 2,
            # i – iv – VII – III
            [(0, "min"), (3, "min"), (6, "maj"), (2, "maj")] * 2,
            # i – VI – III – VII
            [(0, "min"), (5, "maj"), (2, "maj"), (6, "maj")] * 2,
        ],
        lead_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 2, 2.5, 3],
                           [0.5, 0.5, 1, 0.5, 0.5, 1]),
            _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3],
                           [1, 1, 1, 1]),
        ],
        bass_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3], [1, 1, 1, 1]),
            _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                           [0.5] * 8),
        ],
        drum_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                           [0.25] * 8),
            _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25]),
        ],
        lead_instrument=("Generic Saw", "sawtooth"),
        bass_instrument=("Generic Square", "square"),
        harmony_instrument=("NES Pulse 25%", "pulse25"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(95, 120),
        bass_vel=(85, 110),
        harmony_vel=(70, 95),
        drum_vel=(95, 127),
        lead_range=(55, 84),
        bass_range=(33, 52),
        lead_step_max=4,
        lead_rest_prob=0.05,
    )


def _genre_action() -> GenreConfig:
    return GenreConfig(
        name="Action",
        bpm_range=(135, 170),
        bars=8,
        scale_choices=["natural_minor", "dorian", "blues", "pentatonic_minor"],
        progressions=[
            # i – iv – i – VII
            [(0, "min"), (3, "min"), (0, "min"), (6, "maj")] * 2,
            # i – III – VII – iv
            [(0, "min"), (2, "maj"), (6, "maj"), (3, "min")] * 2,
            # power-chord feel: i – v – iv – i
            [(0, "min"), (4, "min"), (3, "min"), (0, "min")] * 2,
        ],
        lead_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                           [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 2, 2.5, 3],
                           [0.5, 0.5, 1, 0.5, 0.5, 1]),
        ],
        bass_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                           [0.5] * 8),
            _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3], [1, 1, 1, 1]),
        ],
        drum_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                           [0.25] * 8),
        ],
        lead_instrument=("NES Square", "square"),
        bass_instrument=("Generic Saw", "sawtooth"),
        harmony_instrument=("Generic Square", "square"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(90, 120),
        bass_vel=(85, 110),
        harmony_vel=(65, 90),
        drum_vel=(100, 127),
        lead_range=(55, 84),
        bass_range=(33, 52),
        lead_step_max=4,
        lead_rest_prob=0.03,
    )


def _genre_retro() -> GenreConfig:
    return GenreConfig(
        name="Retro / Chiptune",
        bpm_range=(120, 155),
        bars=8,
        scale_choices=["major", "pentatonic_major", "pentatonic_minor", "blues"],
        progressions=[
            # I – IV – V – I  (classic 8-bit)
            [(0, "maj"), (3, "maj"), (4, "maj"), (0, "maj")] * 2,
            # I – vi – IV – V
            [(0, "maj"), (5, "min"), (3, "maj"), (4, "maj")] * 2,
            # I – III – IV – iv  (Mario-esque)
            [(0, "maj"), (2, "maj"), (3, "maj"), (3, "min")] * 2,
        ],
        lead_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 3, 3.5],
                           [0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5]),
            _simple_rhythm(_TPB, _BPB, [0, 1, 1.5, 2, 3],
                           [1, 0.5, 0.5, 1, 1]),
        ],
        bass_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3], [1, 1, 1, 1]),
            _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                           [0.5] * 8),
        ],
        drum_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25]),
            _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                           [0.25] * 8),
        ],
        lead_instrument=("NES Square", "square"),
        bass_instrument=("NES Triangle", "triangle"),
        harmony_instrument=("NES Pulse 25%", "pulse25"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(90, 115),
        bass_vel=(80, 100),
        harmony_vel=(65, 85),
        drum_vel=(85, 110),
        lead_range=(60, 84),
        bass_range=(36, 55),
        lead_step_max=3,
        lead_rest_prob=0.07,
    )


def _genre_mystery() -> GenreConfig:
    return GenreConfig(
        name="Mystery",
        bpm_range=(80, 110),
        bars=8,
        scale_choices=["whole_tone", "diminished", "harmonic_minor", "phrygian"],
        progressions=[
            # ambiguous movement
            [(0, "maj7"), (1, "min"), (5, "dim"), (0, "maj7")] * 2,
            [(0, "min"), (2, "dim"), (5, "maj"), (6, "dim")] * 2,
            [(0, "sus4"), (3, "min"), (1, "maj"), (0, "sus4")] * 2,
        ],
        lead_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 1.5, 3], [1.5, 1.5, 1]),
            _simple_rhythm(_TPB, _BPB, [0, 2, 3], [2, 1, 1]),
        ],
        bass_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 2], [2, 2]),
            _simple_rhythm(_TPB, _BPB, [0], [4]),
        ],
        drum_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 2.5], [0.25, 0.25]),
            _simple_rhythm(_TPB, _BPB, [0], [0.25]),
        ],
        lead_instrument=("Generic Sine", "sine"),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Gameboy Pulse 12.5%", "pulse12"),
        drum_instrument=("Generic Noise Drum", "noise"),
        lead_vel=(55, 80),
        bass_vel=(50, 70),
        harmony_vel=(40, 60),
        drum_vel=(40, 65),
        lead_range=(52, 79),
        bass_range=(36, 52),
        lead_step_max=5,
        lead_rest_prob=0.3,
    )


def _genre_boss_battle() -> GenreConfig:
    return GenreConfig(
        name="Boss Battle",
        bpm_range=(150, 180),
        bars=8,
        scale_choices=["harmonic_minor", "phrygian", "natural_minor"],
        progressions=[
            # i – bII – iv – V  (tension-heavy)
            [(0, "min"), (1, "maj"), (3, "min"), (4, "maj")] * 2,
            # i – bVII – bVI – V  (Andalusian cadence)
            [(0, "min"), (6, "maj"), (5, "maj"), (4, "maj")] * 2,
            # i – iv – bVI – bVII
            [(0, "min"), (3, "min"), (5, "maj"), (6, "maj")] * 2,
        ],
        lead_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                           [0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                           [0.5] * 8),
        ],
        bass_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                           [0.5] * 8),
        ],
        drum_rhythm=[
            _simple_rhythm(_TPB, _BPB, [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75,
                                         2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75],
                           [0.25] * 16),
        ],
        lead_instrument=("Generic Saw", "sawtooth"),
        bass_instrument=("NES Square", "square"),
        harmony_instrument=("Generic Pulse 25%", "pulse25"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(100, 127),
        bass_vel=(90, 115),
        harmony_vel=(75, 100),
        drum_vel=(100, 127),
        lead_range=(55, 88),
        bass_range=(30, 52),
        lead_step_max=5,
        lead_rest_prob=0.02,
    )


# ── Registry of all genres ──────────────────────────────────────────

GENRE_BUILDERS: dict[str, type(lambda: None)] = {
    "Happy":            _genre_happy,
    "Calm":             _genre_calm,
    "Sad":              _genre_sad,
    "Horror":           _genre_horror,
    "Epic":             _genre_epic,
    "Action":           _genre_action,
    "Retro / Chiptune": _genre_retro,
    "Mystery":          _genre_mystery,
    "Boss Battle":      _genre_boss_battle,
}

GENRE_NAMES: list[str] = list(GENRE_BUILDERS.keys())


# ── Track generation ────────────────────────────────────────────────

@dataclass
class GeneratedTrack:
    role: str                   # "Lead", "Bass", "Harmony", "Drums"
    instrument_name: str
    waveform: str
    volume: float
    notes: list[NoteEvent]


@dataclass
class GeneratedMusic:
    genre: str
    bpm: int
    root_note: int              # 0-11
    scale_name: str
    tracks: list[GeneratedTrack]
    total_ticks: int


def generate_music(genre_name: str, *,
                   loop_friendly: bool = True,
                   bars_override: int | None = None) -> GeneratedMusic:
    """Generate a full multi-track piece for the given genre.

    Parameters
    ----------
    genre_name : str
        One of the keys in ``GENRE_BUILDERS``.
    loop_friendly : bool
        If *True* (default), the generated music is post-processed for
        seamless looping: the last bar's melody leads back naturally
        into the first bar, velocity is crossfaded at the boundaries,
        and all tracks align to a clean bar boundary.
    bars_override : int or None
        Override the genre's default bar count.  Useful for "one-time"
        mode where the user wants a longer piece.

    Returns a ``GeneratedMusic`` object containing 4 tracks
    (Lead, Bass, Harmony, Drums) plus metadata.
    """
    if genre_name not in GENRE_BUILDERS:
        raise ValueError(f"Unknown genre: {genre_name!r}")

    cfg = GENRE_BUILDERS[genre_name]()

    # Apply bars override if provided
    if bars_override is not None and bars_override > 0:
        cfg.bars = bars_override

    # Random BPM within range
    bpm = random.randint(*cfg.bpm_range)

    # Random root and scale
    root = random.choice(cfg.root_choices)
    scale_name = random.choice(cfg.scale_choices)
    scale = _build_scale(root, scale_name)

    # Choose a random chord progression
    if not cfg.progressions:
        # Fallback: simple I-IV-V-I * bars/4
        chord_prog = [(0, "maj"), (3, "maj"), (4, "maj"), (0, "maj")] * (cfg.bars // 4 or 1)
    else:
        chord_prog = list(random.choice(cfg.progressions))

    # Ensure progression covers all bars
    while len(chord_prog) < cfg.bars:
        chord_prog += chord_prog
    chord_prog = chord_prog[:cfg.bars]

    tpb = cfg.ticks_per_beat
    bpb = cfg.beats_per_bar
    bar_ticks = tpb * bpb
    total_ticks = bar_ticks * cfg.bars

    # Build scale intervals for chord root calculation
    intervals = _SCALE_INTERVALS.get(scale_name, _SCALE_INTERVALS["major"])

    def _rand_mix(level_range: tuple[float, float]) -> float:
        lo, hi = level_range
        lo = max(0.0, min(1.0, lo))
        hi = max(0.0, min(1.0, hi))
        if hi < lo:
            lo, hi = hi, lo
        return round(random.uniform(lo, hi), 2)

    # ── Generate LEAD melody ────────────────────────────────────────
    lead_notes = _generate_lead(cfg, scale, chord_prog, intervals, root,
                                bar_ticks, tpb, bpb)

    # ── Generate BASS line ──────────────────────────────────────────
    bass_notes = _generate_bass(cfg, scale, chord_prog, intervals, root,
                                bar_ticks, tpb, bpb)

    # ── Generate HARMONY (chords) ───────────────────────────────────
    harmony_notes = _generate_harmony(cfg, scale, chord_prog, intervals, root,
                                      bar_ticks, tpb, bpb)

    # ── Generate DRUMS ──────────────────────────────────────────────
    drum_notes = _generate_drums(cfg, bar_ticks, tpb, bpb)

    tracks = [
        GeneratedTrack(
            "Lead",
            cfg.lead_instrument[0],
            cfg.lead_instrument[1],
            _rand_mix(cfg.lead_mix_range),
            lead_notes,
        ),
        GeneratedTrack(
            "Bass",
            cfg.bass_instrument[0],
            cfg.bass_instrument[1],
            _rand_mix(cfg.bass_mix_range),
            bass_notes,
        ),
        GeneratedTrack(
            "Harmony",
            cfg.harmony_instrument[0],
            cfg.harmony_instrument[1],
            _rand_mix(cfg.harmony_mix_range),
            harmony_notes,
        ),
        GeneratedTrack(
            "Drums",
            cfg.drum_instrument[0],
            cfg.drum_instrument[1],
            _rand_mix(cfg.drum_mix_range),
            drum_notes,
        ),
    ]

    # ── Loop-friendly post-processing ───────────────────────────────
    if loop_friendly:
        from daw.theory import fix_loops as _fix_loops

        all_note_lists = [t.notes for t in tracks]
        fixed = _fix_loops(all_note_lists, tpb)
        for i, t in enumerate(tracks):
            t.notes = fixed[i]

        # Recalculate total_ticks after loop-fixing (may have been
        # extended to a bar boundary by fix_loops)
        total_ticks = 0
        for t in tracks:
            if t.notes:
                end = max(n.start_tick + n.length_tick for n in t.notes)
                total_ticks = max(total_ticks, end)
        # Snap to bar boundary
        if total_ticks % bar_ticks != 0:
            total_ticks += bar_ticks - (total_ticks % bar_ticks)

    return GeneratedMusic(
        genre=genre_name,
        bpm=bpm,
        root_note=root,
        scale_name=scale_name,
        tracks=tracks,
        total_ticks=total_ticks,
    )


# ── Individual track generators ─────────────────────────────────────

def _generate_lead(cfg: GenreConfig, scale: list[int],
                   chord_prog: list[tuple[int, str]],
                   intervals: list[int], root: int,
                   bar_ticks: int, tpb: int, bpb: int) -> list[NoteEvent]:
    """Generate a lead melody following the chord progression."""
    notes: list[NoteEvent] = []
    lo, hi = cfg.lead_range

    # Filter scale to lead range
    lead_scale = [s for s in scale if lo <= s <= hi]
    if not lead_scale:
        lead_scale = list(range(lo, hi + 1))

    # Start near the middle
    current_idx = len(lead_scale) // 2

    rhythm_patterns = cfg.lead_rhythm or [
        _simple_rhythm(tpb, bpb, [0, 1, 2, 3], [1, 1, 1, 1])
    ]

    for bar_idx, (degree, _quality) in enumerate(chord_prog):
        bar_start = bar_idx * bar_ticks
        pattern = random.choice(rhythm_patterns)

        # Get chord tones for guidance
        idx_in_scale = degree % len(intervals)
        chord_root_semitone = root + intervals[idx_in_scale]
        chord_tones = set()
        for ct_iv in _CHORD_TYPES.get(_quality, [0, 4, 7]):
            chord_tones.add((chord_root_semitone + ct_iv) % 12)

        for tick_off, length in pattern:
            # Decide rest
            if random.random() < cfg.lead_rest_prob:
                continue

            # Walk with bias toward chord tones
            max_step = cfg.lead_step_max
            step = random.randint(-max_step, max_step)

            # Bias: if current note's pitch class is not a chord tone, try to step toward one
            candidate_idx = max(0, min(len(lead_scale) - 1, current_idx + step))
            candidate_note = lead_scale[candidate_idx]

            # Extra bias: with 40% chance, snap to nearest chord tone
            if random.random() < 0.4:
                # Find nearest lead_scale note whose pc is in chord_tones
                chord_scale = [i for i, n in enumerate(lead_scale)
                               if n % 12 in chord_tones]
                if chord_scale:
                    candidate_idx = min(chord_scale,
                                        key=lambda i: abs(i - current_idx))
                    candidate_note = lead_scale[candidate_idx]

            current_idx = candidate_idx
            vel = random.randint(*cfg.lead_vel)
            notes.append(NoteEvent(
                start_tick=bar_start + tick_off,
                length_tick=length,
                midi_note=candidate_note,
                velocity=min(127, vel),
            ))

    return notes


def _generate_bass(cfg: GenreConfig, scale: list[int],
                   chord_prog: list[tuple[int, str]],
                   intervals: list[int], root: int,
                   bar_ticks: int, tpb: int, bpb: int) -> list[NoteEvent]:
    """Generate a bass line: root/fifth patterns following chord progression."""
    notes: list[NoteEvent] = []
    lo, hi = cfg.bass_range
    bass_scale = [s for s in scale if lo <= s <= hi]
    if not bass_scale:
        bass_scale = list(range(lo, hi + 1))

    rhythm_patterns = cfg.bass_rhythm or [
        _simple_rhythm(tpb, bpb, [0, 2], [2, 2])
    ]

    for bar_idx, (degree, quality) in enumerate(chord_prog):
        bar_start = bar_idx * bar_ticks
        pattern = random.choice(rhythm_patterns)

        # Find the chord root in bass range
        idx_in_scale = degree % len(intervals)
        chord_root_pc = (root + intervals[idx_in_scale]) % 12

        # Find closest bass scale note matching chord_root_pc
        root_candidates = [n for n in bass_scale if n % 12 == chord_root_pc]
        if not root_candidates:
            root_candidates = [_snap_to_scale((lo + hi) // 2, bass_scale)]
        bass_root = random.choice(root_candidates)

        # Also get the fifth
        fifth_pc = (chord_root_pc + 7) % 12
        fifth_candidates = [n for n in bass_scale if n % 12 == fifth_pc]
        bass_fifth = random.choice(fifth_candidates) if fifth_candidates else bass_root

        for note_idx, (tick_off, length) in enumerate(pattern):
            # Alternate root and fifth with some randomness
            if note_idx == 0 or random.random() < 0.6:
                midi = bass_root
            else:
                midi = bass_fifth if random.random() < 0.7 else bass_root

            # Occasional octave variation
            if random.random() < 0.15 and lo <= midi + 12 <= hi:
                midi += 12

            vel = random.randint(*cfg.bass_vel)
            notes.append(NoteEvent(
                start_tick=bar_start + tick_off,
                length_tick=length,
                midi_note=midi,
                velocity=min(127, vel),
            ))

    return notes


def _generate_harmony(cfg: GenreConfig, scale: list[int],
                      chord_prog: list[tuple[int, str]],
                      intervals: list[int], root: int,
                      bar_ticks: int, tpb: int, bpb: int) -> list[NoteEvent]:
    """Generate harmony: chord voicings, one per bar (or broken chords)."""
    notes: list[NoteEvent] = []
    lo, hi = cfg.harmony_range

    # Choose style: block chords or arpeggiated
    use_arpeggio = random.random() < 0.4

    for bar_idx, (degree, quality) in enumerate(chord_prog):
        bar_start = bar_idx * bar_ticks

        # Chord root in harmony range
        idx_in_scale = degree % len(intervals)
        chord_root_pc = (root + intervals[idx_in_scale]) % 12

        # Find octave for root in range
        chord_root = None
        for octave in range(11):
            candidate = chord_root_pc + octave * 12
            if lo <= candidate <= hi:
                chord_root = candidate
                break
        if chord_root is None:
            chord_root = (lo + hi) // 2

        # Build chord voicing
        chord_ivs = _CHORD_TYPES.get(quality, [0, 4, 7])
        voicing = []
        for iv in chord_ivs:
            note = chord_root + iv
            # Ensure within range
            while note > hi and note > 12:
                note -= 12
            while note < lo:
                note += 12
            if lo <= note <= hi:
                voicing.append(note)
        if not voicing:
            voicing = [chord_root]

        vel = random.randint(*cfg.harmony_vel)

        if use_arpeggio:
            # Arpeggiate: spread notes across the bar
            num_notes = len(voicing)
            arp_len = bar_ticks // max(num_notes * 2, 1)
            arp_len = max(1, arp_len)

            # Arpeggio pattern: up, down, or up-down
            arp_style = random.choice(["up", "down", "updown"])
            if arp_style == "down":
                voicing = list(reversed(voicing))
            elif arp_style == "updown":
                voicing = voicing + list(reversed(voicing[1:-1] if len(voicing) > 2 else voicing))

            tick = 0
            for i, midi in enumerate(voicing):
                if bar_start + tick >= bar_start + bar_ticks:
                    break
                length = min(arp_len, bar_ticks - tick)
                notes.append(NoteEvent(
                    start_tick=bar_start + tick,
                    length_tick=length,
                    midi_note=midi,
                    velocity=min(127, vel + random.randint(-5, 5)),
                ))
                tick += arp_len

            # Repeat arpeggio to fill bar
            if tick < bar_ticks:
                remaining = bar_ticks - tick
                for i in range(remaining // arp_len + 1):
                    idx = i % len(voicing)
                    if bar_start + tick >= bar_start + bar_ticks:
                        break
                    length = min(arp_len, bar_ticks - tick)
                    notes.append(NoteEvent(
                        start_tick=bar_start + tick,
                        length_tick=length,
                        midi_note=voicing[idx],
                        velocity=min(127, vel + random.randint(-5, 5)),
                    ))
                    tick += arp_len
        else:
            # Block chord: whole bar sustained
            for midi in voicing:
                notes.append(NoteEvent(
                    start_tick=bar_start,
                    length_tick=bar_ticks,
                    midi_note=midi,
                    velocity=min(127, vel),
                ))

    return notes


def _generate_drums(cfg: GenreConfig, bar_ticks: int,
                    tpb: int, bpb: int) -> list[NoteEvent]:
    """Generate drum patterns using available drum pitches."""
    notes: list[NoteEvent] = []
    pitches = cfg.drum_pitches
    if not pitches:
        pitches = [36, 38, 42, 46]

    # Assign roles to pitches: kick, snare, hihat, accent
    kick = pitches[0]
    snare = pitches[1] if len(pitches) > 1 else pitches[0]
    hihat = pitches[2] if len(pitches) > 2 else pitches[0]
    accent = pitches[3] if len(pitches) > 3 else pitches[0]

    rhythm_patterns = cfg.drum_rhythm or [
        _simple_rhythm(tpb, bpb, [0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25])
    ]

    for bar_idx in range(cfg.bars):
        bar_start = bar_idx * bar_ticks
        pattern = random.choice(rhythm_patterns)

        for note_idx, (tick_off, length) in enumerate(pattern):
            vel = random.randint(*cfg.drum_vel)

            # Assign drum sound based on beat position
            beat_pos = tick_off / tpb  # beat number (float)

            if beat_pos % 2 < 0.01:
                # Downbeats: kick
                midi = kick
            elif abs(beat_pos % 2 - 1.0) < 0.01:
                # Backbeats: snare
                midi = snare
            elif beat_pos % 0.5 < 0.01:
                # Offbeats: hihat
                midi = hihat
            else:
                # Sub-divisions: random between hihat and accent
                midi = random.choice([hihat, accent])

            # Random ghost notes / accents
            if random.random() < 0.15:
                vel = max(30, vel - 30)  # ghost note

            # Occasional fill (randomise an extra hit)
            notes.append(NoteEvent(
                start_tick=bar_start + tick_off,
                length_tick=length,
                midi_note=midi,
                velocity=min(127, vel),
            ))

        # Occasional fill on last 2 bars
        if bar_idx >= cfg.bars - 2 and random.random() < 0.4:
            fill_tick = bar_start + bar_ticks - tpb
            for fi in range(random.randint(2, 4)):
                offset = fi * max(1, tpb // 4)
                if fill_tick + offset < bar_start + bar_ticks:
                    notes.append(NoteEvent(
                        start_tick=fill_tick + offset,
                        length_tick=max(1, tpb // 4),
                        midi_note=random.choice([snare, kick]),
                        velocity=min(127, random.randint(*cfg.drum_vel)),
                    ))

    return notes
