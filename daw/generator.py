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
    "lydian":           [0, 2, 4, 6, 7, 9, 11],   # dreamy, bright (#4)
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
    "sus2":  [0, 2, 7],
    "7":     [0, 4, 7, 10],
    "m7":    [0, 3, 7, 10],
    "maj7":  [0, 4, 7, 11],
    "dim7":  [0, 3, 6, 9],
    "add9":  [0, 4, 7, 14],
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
class ExtraTrackConfig:
    """Configuration for an extra track beyond Lead/Bass/Harmony/Drums."""
    role: str
    gen_type: str               # "counter_melody", "pad", "arpeggio"
    instrument: tuple[str, str]
    vel: tuple[int, int] = (60, 85)
    pitch_range: tuple[int, int] = (48, 72)
    mix_range: tuple[float, float] = (0.45, 0.70)
    rest_prob: float = 0.2
    step_max: int = 3
    rhythm: list[list[tuple[int, int]]] = field(default_factory=list)
    pan: float = 0.0            # -1.0 (left) .. +1.0 (right)


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

    # Style variants
    lead_style: str = "default"       # "default", "arpeggio_peaks", "question_answer", "gba_town"
    bass_style: str = "default"       # "default", "waltz", "walking", "root_fifth"
    harmony_style: str = "default"    # "default", "staccato"
    drum_style: str = "default"       # "default", "gba_town"

    # Harmony rhythm patterns (used when harmony_style != "default")
    harmony_rhythm: list[list[tuple[int, int]]] = field(default_factory=list)

    # Per-track panning  (-1.0 left .. +1.0 right)
    lead_pan: float = 0.0
    bass_pan: float = 0.0
    harmony_pan: float = 0.0
    drum_pan: float = 0.0

    # Lead doubling: create a "sparkle" copy at low volume  (instrument_name, waveform, volume)
    lead_doubling: tuple[str, str, float] | None = None

    # Swing: 0.0 = straight, 0.1 = 10% shuffle on 8th notes
    swing: float = 0.0

    # Town vibe: stored so the progression can be built *after* bars_override
    vibe: str = ""                    # "inland" or "island"; empty = use progressions list
    # Progression family: determines which chord builder to use (nes/gba/snes)
    progression_family: str = "gba"   # "nes", "gba", "snes"

    # Custom progression builder: (vibe, bars, family) -> list[tuple[int,str]]
    # When set, generate_music uses this instead of the default town/cave dispatch.
    progression_builder: object = None
    # Custom section boundary builder: (total_bars) -> dict[str, int]
    # When set, _generate_lead uses this instead of _town_sections/_cave_sections.
    section_builder: object = None

    # Extra tracks beyond the standard Lead / Bass / Harmony / Drums
    extra_tracks: list[ExtraTrackConfig] = field(default_factory=list)


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


# ── Shared Town progression builder (adapts to bar count) ───────────

def _nes_town_progression(vibe: str, bars: int) -> list[tuple[int, str]]:
    """NES-era Town progression — simple diatonic triads only.

    Warm, simple, nostalgic 8-bit character.
    No 7th chords — just major/minor triads + occasional sus4.
    """
    if vibe == "island":
        phrases = [
            [(0, "maj"), (5, "maj"), (3, "maj"), (0, "maj")],   # bVII bounce
            [(0, "maj"), (3, "maj"), (5, "maj"), (0, "maj")],   # IV-bVII-I
            [(0, "maj"), (3, "maj"), (5, "maj"), (3, "maj")],   # open island
            [(0, "maj"), (5, "maj"), (0, "maj"), (3, "maj")],   # I-bVII-I-IV
            [(3, "maj"), (5, "maj"), (0, "maj"), (0, "maj")],   # IV-bVII-I-I
        ]
        dev_pool = [
            [(5, "maj"), (3, "maj"), (0, "maj"), (3, "maj")],
            [(3, "maj"), (0, "maj"), (5, "maj"), (0, "maj")],
            [(0, "maj"), (5, "maj"), (3, "maj"), (3, "maj")],
        ]
    else:  # inland
        phrases = [
            [(0, "maj"), (3, "maj"), (4, "maj"), (0, "maj")],   # I-IV-V-I
            [(0, "maj"), (5, "min"), (3, "maj"), (4, "maj")],   # I-vi-IV-V
            [(0, "maj"), (3, "maj"), (5, "min"), (4, "maj")],   # I-IV-vi-V
            [(0, "maj"), (4, "maj"), (3, "maj"), (0, "maj")],   # I-V-IV-I
            [(0, "maj"), (2, "min"), (3, "maj"), (4, "maj")],   # I-iii-IV-V
            [(0, "maj"), (5, "min"), (4, "maj"), (0, "maj")],   # I-vi-V-I
            [(3, "maj"), (0, "maj"), (4, "maj"), (0, "maj")],   # IV-I-V-I
            [(0, "maj"), (2, "min"), (5, "min"), (4, "maj")],   # I-iii-vi-V
        ]
        dev_pool = [
            [(5, "min"), (3, "maj"), (4, "maj"), (0, "maj")],   # vi-IV-V-I
            [(3, "maj"), (4, "maj"), (5, "min"), (4, "maj")],   # IV-V-vi-V
        ]

    if bars <= 8:
        return random.choice(phrases) + random.choice(phrases)
    if bars <= 16:
        p1 = random.choice(phrases)
        p2 = random.choice(phrases)
        p3 = random.choice(phrases)
        cadence = random.choice(phrases)
        form = random.choice(["AABA", "AABA", "ABAC", "ABBA"])
        if form == "AABA":
            return p1 + p1 + p2 + cadence
        if form == "ABAC":
            return p1 + p2 + p1 + cadence
        if form == "ABBA":
            return p1 + p2 + p2 + cadence
        return p1 + p2 + p3 + cadence
    # 32+
    intro = [(0, "maj")] * 2
    head  = random.choice(phrases) + random.choice(phrases)       # 8
    dev   = random.choice(dev_pool) + random.choice(dev_pool)     # 8
    clim  = random.choice(phrases) + random.choice(dev_pool)      # 8
    reset = random.choice(phrases)                                # 4
    cadence = [(0, "maj"), (0, "maj")]                            # 2
    prog = intro + head + dev + clim + reset + cadence
    while len(prog) < bars:
        prog.append((0, "maj"))
    return prog[:bars]


def _gba_town_progression(vibe: str, bars: int) -> list[tuple[int, str]]:
    """GBA-era Town progression — jazz-influenced with 7th chords.

    Bouncy, jazzy, secondary dominants.
    Uses II7, V7, Imaj7, IVmaj7 for a "floating" quality.
    """
    if vibe == "island":
        phrases = [
            [(0, "maj"), (6, "maj7"), (3, "maj"), (0, "maj")],   # sea
            [(0, "maj7"), (3, "maj"), (6, "7"), (0, "maj")],     # island float
            [(0, "maj"), (4, "7"), (3, "maj"), (0, "maj")],      # bounce
            [(0, "maj"), (3, "7"), (6, "maj"), (4, "7")],        # island drift
            [(6, "maj7"), (0, "maj"), (3, "maj"), (4, "7")],     # reverse sea
        ]
        dev_a = [(3, "maj7"), (6, "7"), (0, "maj"), (4, "7")]
        dev_b = [(0, "maj"), (6, "maj"), (3, "7"), (4, "7")]
        clim_pool = phrases + [[(6, "maj7"), (3, "7"), (4, "7"), (0, "maj")]]
    else:  # inland
        phrases = [
            [(0, "maj7"), (5, "m7"), (1, "m7"), (4, "7")],       # I-vi-ii-V
            [(0, "maj"), (3, "maj"), (1, "m7"), (4, "7")],       # I-IV-ii-V
            [(0, "maj"), (5, "min"), (3, "maj"), (4, "7")],      # I-vi-IV-V
            [(0, "maj7"), (1, "m7"), (5, "m7"), (4, "7")],       # I-ii-vi-V
            [(0, "maj"), (2, "min"), (3, "maj7"), (4, "7")],     # I-iii-IV-V
            [(3, "maj7"), (0, "maj"), (1, "m7"), (4, "7")],      # IV-I-ii-V
        ]
        dev_a = [(5, "m7"), (3, "maj7"), (1, "m7"), (4, "7")]
        dev_b = [(0, "maj7"), (5, "m7"), (3, "maj"), (4, "7")]
        clim_pool = phrases + [[(5, "m7"), (3, "maj7"), (1, "7"), (4, "7")]]

    cadence = random.choice([[(4, "7")], [(4, "sus4")]])

    if bars <= 8:
        return random.choice(phrases) + random.choice(phrases)
    if bars <= 16:
        intro = [(0, "maj")]
        p1 = random.choice(phrases)
        p2 = random.choice(phrases)
        p3 = random.choice(phrases)
        cad = [(0, "maj")] + cadence * 2
        form = random.choice(["AABC", "AABC", "ABAC", "ABBC"])
        if form == "AABC":
            return intro + p1 + p1 + p2 + cad
        if form == "ABAC":
            return intro + p1 + p2 + p1 + cad
        return intro + p1 + p2 + p2 + cad
    # 32+
    intro  = [(0, "maj")] * 2
    head_a = random.choice(phrases)
    head_b = random.choice(phrases)
    clim_a = random.choice(clim_pool)
    clim_b = random.choice(clim_pool)
    reset  = [(0, "maj"), (0, "maj")] + cadence * 2
    prog = intro + head_a + head_b + dev_a + dev_b + clim_a + clim_b + reset
    while len(prog) < bars:
        prog.append((0, "maj"))
    return prog[:bars]


def _snes_town_progression(vibe: str, bars: int) -> list[tuple[int, str]]:
    """SNES-era Town progression — dreamy suspended/extended chords.

    Orchestral, lush, ethereal character.
    Heavy use of sus2, sus4, maj7, add9 for a "floating dream" quality.
    """
    if vibe == "island":
        phrases = [
            [(0, "maj7"), (6, "maj7"), (3, "sus2"), (0, "maj")],   # dreamy sea
            [(0, "add9"), (3, "maj"), (6, "sus4"), (0, "maj7")],   # floating
            [(0, "maj"), (3, "add9"), (6, "maj7"), (4, "sus4")],   # ethereal
        ]
        dev_pool = [
            [(3, "sus2"), (6, "maj7"), (0, "add9"), (4, "sus4")],
            [(6, "maj7"), (3, "add9"), (4, "sus4"), (0, "maj")],
        ]
    else:  # inland
        phrases = [
            [(0, "sus2"), (3, "maj7"), (5, "min"), (4, "sus4")],   # dream walk
            [(0, "maj7"), (3, "add9"), (5, "min"), (4, "maj")],    # soft glow
            [(0, "add9"), (3, "maj"), (5, "m7"), (3, "maj7")],     # gentle
            [(0, "maj"), (3, "sus2"), (1, "m7"), (4, "sus4")],     # wander
            [(0, "maj7"), (2, "min"), (3, "sus2"), (4, "add9")],   # float
            [(0, "sus2"), (5, "m7"), (3, "add9"), (0, "maj7")],    # drift
            [(3, "maj7"), (0, "add9"), (5, "min"), (4, "sus4")],   # bloom
        ]
        dev_pool = [
            [(5, "m7"), (3, "maj7"), (0, "sus2"), (4, "sus4")],
            [(3, "add9"), (5, "min"), (0, "maj7"), (4, "sus4")],
            [(0, "add9"), (2, "min"), (3, "maj7"), (4, "sus4")],
        ]

    if bars <= 8:
        return random.choice(phrases) + random.choice(phrases)
    if bars <= 16:
        p1 = random.choice(phrases)
        p2 = random.choice(phrases)
        p3 = random.choice(phrases)
        cad = [(0, "sus2"), (4, "sus4"), (0, "maj7"), (0, "maj")]
        form = random.choice(["AABC", "AABC", "ABAC", "ABBC"])
        if form == "AABC":
            return p1 + p1 + p2 + cad
        if form == "ABAC":
            return p1 + p2 + p1 + cad
        return p1 + p2 + p2 + cad
    # 32+
    intro  = [(0, "maj7")] * 2
    head   = random.choice(phrases) + random.choice(phrases)       # 8
    dev    = random.choice(dev_pool) + random.choice(dev_pool)     # 8
    clim   = random.choice(phrases) + random.choice(dev_pool)      # 8
    reset  = [(0, "sus2"), (3, "maj7"), (4, "sus4"), (0, "maj")]   # 4
    tail   = [(0, "maj7"), (0, "maj")]                             # 2
    prog = intro + head + dev + clim + reset + tail
    while len(prog) < bars:
        prog.append((0, "maj"))
    return prog[:bars]


def _town_progression(vibe: str, bars: int,
                      family: str = "gba") -> list[tuple[int, str]]:
    """Dispatch to the correct Town progression builder."""
    if family == "nes":
        return _nes_town_progression(vibe, bars)
    if family == "snes":
        return _snes_town_progression(vibe, bars)
    return _gba_town_progression(vibe, bars)


# ── Cave progression builders ───────────────────────────────────────

def _generic_cave_progression(vibe: str, bars: int) -> list[tuple[int, str]]:
    """Generic Cave — dark, echoey 8-bit dungeon.

    Natural minor / phrygian triads, dim chords, pedal-tone drones.
    Sparse, atmospheric, tension-building.
    """
    if vibe == "deep":
        phrases = [
            [(0, "min"), (5, "dim"), (3, "min"), (0, "min")],
            [(0, "min"), (2, "min"), (5, "dim"), (0, "min")],
            [(0, "min"), (3, "min"), (4, "min"), (0, "min")],
            [(0, "min"), (0, "min"), (5, "dim"), (4, "min")],
            [(3, "min"), (5, "dim"), (0, "min"), (0, "min")],
        ]
        dev_pool = [
            [(5, "dim"), (3, "min"), (2, "min"), (0, "min")],
            [(0, "min"), (4, "min"), (5, "dim"), (0, "min")],
        ]
    else:  # shallow
        phrases = [
            [(0, "min"), (3, "maj"), (4, "min"), (0, "min")],
            [(0, "min"), (5, "min"), (3, "maj"), (4, "min")],
            [(0, "min"), (2, "dim"), (3, "maj"), (0, "min")],
            [(0, "min"), (4, "min"), (3, "maj"), (0, "min")],
            [(3, "maj"), (4, "min"), (5, "min"), (0, "min")],
        ]
        dev_pool = [
            [(5, "min"), (3, "maj"), (4, "min"), (0, "min")],
            [(4, "min"), (2, "dim"), (3, "maj"), (0, "min")],
        ]

    if bars <= 8:
        return random.choice(phrases) + random.choice(phrases)
    if bars <= 16:
        p1 = random.choice(phrases)
        p2 = random.choice(phrases)
        cad = random.choice(phrases)
        form = random.choice(["AABA", "ABAC", "AABB"])
        if form == "AABA":
            return p1 + p1 + p2 + cad
        if form == "ABAC":
            return p1 + p2 + p1 + cad
        return p1 + p1 + p2 + p2
    # 32+
    drone = [(0, "min")] * 2
    head  = random.choice(phrases) + random.choice(phrases)
    dev   = random.choice(dev_pool) + random.choice(dev_pool)
    clim  = random.choice(phrases) + random.choice(dev_pool)
    reset = random.choice(phrases)
    tail  = [(0, "min"), (0, "min")]
    prog = drone + head + dev + clim + reset + tail
    while len(prog) < bars:
        prog.append((0, "min"))
    return prog[:bars]


def _gba_cave_progression(vibe: str, bars: int) -> list[tuple[int, str]]:
    """GBA Cave — tense m7 / dim7 jazz-minor atmosphere.

    Uses m7, dim7, half-diminished sounds for GBA-era dungeon feel.
    More harmonic movement than generic cave but still moody.
    """
    if vibe == "deep":
        phrases = [
            [(0, "m7"), (5, "dim7"), (3, "m7"), (0, "min")],
            [(0, "m7"), (2, "dim"), (5, "dim7"), (4, "m7")],
            [(0, "min"), (3, "m7"), (5, "dim7"), (0, "m7")],
            [(0, "m7"), (1, "m7"), (5, "dim7"), (0, "min")],
            [(3, "m7"), (5, "dim7"), (4, "7"), (0, "min")],
        ]
        dev_pool = [
            [(5, "dim7"), (3, "m7"), (1, "m7"), (0, "min")],
            [(0, "m7"), (4, "7"), (5, "dim7"), (0, "min")],
        ]
    else:  # shallow
        phrases = [
            [(0, "m7"), (3, "maj7"), (5, "min"), (0, "m7")],
            [(0, "min"), (5, "m7"), (3, "maj7"), (4, "m7")],
            [(0, "m7"), (2, "dim"), (3, "maj"), (0, "m7")],
            [(0, "m7"), (4, "m7"), (5, "min"), (0, "min")],
            [(3, "maj7"), (0, "m7"), (5, "min"), (4, "m7")],
        ]
        dev_pool = [
            [(5, "m7"), (3, "maj7"), (4, "7"), (0, "m7")],
            [(0, "m7"), (2, "dim"), (5, "m7"), (0, "min")],
        ]

    if bars <= 8:
        return random.choice(phrases) + random.choice(phrases)
    if bars <= 16:
        intro = [(0, "m7")]
        p1 = random.choice(phrases)
        p2 = random.choice(phrases)
        cad = [(0, "min")] + [(5, "dim7")] * 2
        form = random.choice(["AABC", "ABAC", "ABBC"])
        if form == "AABC":
            return intro + p1 + p1 + p2 + cad
        if form == "ABAC":
            return intro + p1 + p2 + p1 + cad
        return intro + p1 + p2 + p2 + cad
    # 32+
    drone  = [(0, "m7")] * 2
    head_a = random.choice(phrases)
    head_b = random.choice(phrases)
    dev_a  = random.choice(dev_pool)
    dev_b  = random.choice(dev_pool)
    clim_a = random.choice(phrases)
    clim_b = random.choice(phrases)
    reset  = [(0, "m7"), (0, "min")] + [(5, "dim7")] * 2
    prog = drone + head_a + head_b + dev_a + dev_b + clim_a + clim_b + reset
    while len(prog) < bars:
        prog.append((0, "min"))
    return prog[:bars]


def _snes_cave_progression(vibe: str, bars: int) -> list[tuple[int, str]]:
    """SNES Cave — ethereal, haunted, suspended atmosphere.

    Uses sus2, sus4, m7, dim with dreamy/uneasy suspension.
    Slow-moving harmony with colour tones.
    """
    if vibe == "deep":
        phrases = [
            [(0, "sus2"), (5, "dim"), (3, "m7"), (0, "min")],
            [(0, "m7"), (3, "sus4"), (5, "dim"), (0, "sus2")],
            [(0, "sus4"), (2, "dim"), (3, "m7"), (0, "min")],
            [(0, "m7"), (5, "dim"), (4, "sus4"), (0, "sus2")],
            [(3, "m7"), (0, "sus2"), (5, "dim"), (0, "min")],
        ]
        dev_pool = [
            [(5, "dim"), (3, "sus4"), (0, "sus2"), (4, "m7")],
            [(0, "sus4"), (5, "dim"), (3, "m7"), (0, "sus2")],
        ]
    else:  # shallow
        phrases = [
            [(0, "sus2"), (3, "m7"), (5, "min"), (0, "sus4")],
            [(0, "m7"), (3, "sus2"), (5, "min"), (4, "sus4")],
            [(0, "min"), (2, "m7"), (3, "sus4"), (0, "sus2")],
            [(0, "sus4"), (3, "m7"), (5, "min"), (0, "m7")],
            [(3, "sus2"), (0, "m7"), (5, "min"), (4, "sus4")],
        ]
        dev_pool = [
            [(5, "min"), (3, "sus2"), (0, "m7"), (4, "sus4")],
            [(0, "sus2"), (2, "m7"), (3, "sus4"), (0, "min")],
        ]

    if bars <= 8:
        return random.choice(phrases) + random.choice(phrases)
    if bars <= 16:
        p1 = random.choice(phrases)
        p2 = random.choice(phrases)
        cad = [(0, "sus2"), (5, "dim"), (0, "m7"), (0, "min")]
        form = random.choice(["AABC", "ABAC", "ABBC"])
        if form == "AABC":
            return p1 + p1 + p2 + cad
        if form == "ABAC":
            return p1 + p2 + p1 + cad
        return p1 + p2 + p2 + cad
    # 32+
    drone  = [(0, "sus2")] * 2
    head   = random.choice(phrases) + random.choice(phrases)
    dev    = random.choice(dev_pool) + random.choice(dev_pool)
    clim   = random.choice(phrases) + random.choice(dev_pool)
    reset  = [(0, "sus4"), (5, "dim"), (0, "m7"), (0, "min")]
    tail   = [(0, "sus2"), (0, "min")]
    prog = drone + head + dev + clim + reset + tail
    while len(prog) < bars:
        prog.append((0, "min"))
    return prog[:bars]


def _cave_progression(vibe: str, bars: int,
                      family: str = "nes") -> list[tuple[int, str]]:
    """Dispatch to the correct Cave progression builder."""
    if family == "gba":
        return _gba_cave_progression(vibe, bars)
    if family == "snes":
        return _snes_cave_progression(vibe, bars)
    return _generic_cave_progression(vibe, bars)


_MINORISH_SCALES = {
    "natural_minor", "harmonic_minor", "melodic_minor",
    "dorian", "phrygian", "pentatonic_minor", "blues",
}


def _inject_secondary_dominants(
    chord_prog: list[tuple[int, str]],
    *,
    strength: float,
) -> list[tuple[int, str]]:
    """Inject occasional dominant pivots (V/V and V/I) for lift.

    Degrees are 0-based. In a major-like frame:
    - V is degree 4
    - V/V is degree 1 dominant (D7 in key of C)
    """
    if not chord_prog:
        return chord_prog

    out = list(chord_prog)
    for i in range(1, len(out)):
        next_degree, _next_quality = out[i]
        if random.random() > strength:
            continue

        if next_degree == 4:
            out[i - 1] = (1, "7")   # V/V -> V
        elif next_degree == 0 and random.random() < 0.45:
            out[i - 1] = (4, "7")   # V -> I
    return out


def _apply_picardy_third(
    chord_prog: list[tuple[int, str]],
    *,
    scale_name: str,
    chance: float,
) -> list[tuple[int, str]]:
    """End minor-ish loops with a tonic major chord (Picardy third)."""
    if not chord_prog or scale_name not in _MINORISH_SCALES:
        return chord_prog
    if random.random() > chance:
        return chord_prog

    out = list(chord_prog)
    out[-1] = (0, "maj")
    return out


# ── Dynamic section boundaries for Town themes ─────────────────────

def _town_sections(total_bars: int) -> dict[str, int]:
    """Return section boundary indices for Town generation.

    For 8 bars:  no intro silence, melody all the way through.
    For 16 bars: 1-bar intro, rest is melody.
    For 32 bars: 2-bar intro (was 4), 2-bar silent tail.
    """
    if total_bars <= 8:
        return {
            "intro_end": 0,            # no intro
            "head_end": total_bars // 2,
            "dev_end": total_bars - 1,
            "climax_end": total_bars - 1,
            "reset_start": total_bars - 1,
            "silent_tail": total_bars,  # no silence
        }
    if total_bars <= 16:
        # Add ±1 bar randomness to section boundaries
        _he = random.choice([5, 6, 7])
        _de = random.choice([10, 11, 12])
        return {
            "intro_end": random.choice([1, 1, 2]),
            "head_end": _he,
            "dev_end": _de,
            "climax_end": 14,
            "reset_start": 14,
            "silent_tail": total_bars,  # no silence
        }
    # 32+
    _intro = random.choice([1, 2, 2, 3])
    _head  = _intro + random.choice([7, 8, 9])
    _dev   = _head + random.choice([7, 8, 9])
    return {
        "intro_end": _intro,
        "head_end": _head,
        "dev_end": _dev,
        "climax_end": total_bars - random.choice([3, 4, 5]),
        "reset_start": total_bars - random.choice([3, 4, 5]),
        "silent_tail": total_bars - random.choice([1, 2, 2]),
    }


# ── Genre-specific rhythm presets ────────────────────────────────────

# -- NES Town: simple, singable, pentatonic-friendly --
_NES_LEAD_RHYTHMS = [
    _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3],
                   [1, 1, 1, 1]),                    # straight quarter notes
    _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 2, 3],
                   [0.5, 0.5, 1, 1, 1]),             # pickup 8th feel
    _simple_rhythm(_TPB, _BPB, [0, 1, 2, 2.5, 3],
                   [1, 1, 0.5, 0.5, 1]),             # slight syncopation
    _simple_rhythm(_TPB, _BPB, [0, 0.5, 1.5, 2.5, 3.5],
                   [0.5, 1, 1, 0.5, 0.5]),           # walking pickup
    _simple_rhythm(_TPB, _BPB, [0, 1.5, 2, 3],
                   [1.5, 0.5, 1, 1]),                # long-short answer
]
_NES_BASS_PUMP = [
    _simple_rhythm(_TPB, _BPB, [0, 2], [1.5, 1.5]),           # beats 1 & 3
]
_NES_DRUM_TICK = [
    _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3],
                   [0.5, 0.5, 0.5, 0.5]),           # quarter note ticking
]
_NES_HARMONY_ARP = [
    # 8th-note arpeggiation — simulates NES "chord" effect (less dense than 16ths)
    _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                   [0.5] * 8),
]

# -- GBA Town: bouncy, call-response, 3+3+2 feel --
_GBA_LEAD_RHYTHMS = [
    _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 2, 2.5, 3],
                   [0.5, 0.5, 1, 0.5, 0.5, 1]),
    _simple_rhythm(_TPB, _BPB, [0, 1, 1.5, 2, 3],
                   [1, 0.5, 0.5, 1, 1]),
    _simple_rhythm(_TPB, _BPB, [0, 0.75, 1.5, 2.5, 3.25],
                   [0.5, 0.5, 0.75, 0.5, 0.75]),     # off-grid bounce
    _simple_rhythm(_TPB, _BPB, [0, 0.5, 1.5, 2, 2.5, 3.5],
                   [0.5, 0.75, 0.5, 0.5, 0.75, 0.5]),
]
_GBA_BASS_332 = [
    _simple_rhythm(_TPB, _BPB, [0, 1.5, 3], [1.5, 1.5, 1]),
]
_GBA_DRUM_8THS = [
    _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                   [0.25] * 8),
]
_GBA_HARMONY_STACCATO = [
    _simple_rhythm(_TPB, _BPB, [0, 1.5, 3], [0.5, 0.5, 0.5]),
]

# -- SNES Town: legato, flowing, orchestral --
_SNES_LEAD_RHYTHMS = [
    _simple_rhythm(_TPB, _BPB, [0, 1.5, 3],
                   [1.5, 1.5, 1]),                   # dotted quarter feel
    _simple_rhythm(_TPB, _BPB, [0, 2, 3],
                   [2, 1, 1]),                        # long-short-short
    _simple_rhythm(_TPB, _BPB, [0, 2],
                   [2, 2]),                           # half notes (legato)
    _simple_rhythm(_TPB, _BPB, [0, 1, 2.5],
                   [1, 1.5, 1.5]),                    # lyrical arch
    _simple_rhythm(_TPB, _BPB, [0, 1.5, 2.5, 3.5],
                   [1.5, 1, 1, 0.5]),                 # phrase tail
]
_SNES_BASS_WALK = [
    _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3],
                   [1, 1, 1, 1]),                     # walking quarter notes
]
_SNES_DRUM_BRUSH = [
    _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3],
                   [0.25, 0.25, 0.25, 0.25]),         # gentle quarter ticks
]

# (Keep old aliases for backward compat — point to GBA variants)
_TOWN_LEAD_RHYTHMS    = _GBA_LEAD_RHYTHMS
_TOWN_BASS_332        = _GBA_BASS_332
_TOWN_DRUM_8THS       = _GBA_DRUM_8THS
_TOWN_HARMONY_STACCATO = _GBA_HARMONY_STACCATO


# ── Cave rhythm presets ──────────────────────────────────────────────

# -- Generic Cave: sparse, echoey, lots of space --
_CAVE_LEAD_SPARSE = [
    _simple_rhythm(_TPB, _BPB, [0, 2],
                   [1.5, 1.5]),                       # half-note echo
    _simple_rhythm(_TPB, _BPB, [0, 1.5, 3],
                   [1.5, 1, 0.5]),                    # drip pattern
    _simple_rhythm(_TPB, _BPB, [0, 2.5],
                   [2, 1.5]),                         # wide breath
    _simple_rhythm(_TPB, _BPB, [0, 1, 3],
                   [1, 1.5, 0.5]),                    # echo bounce
]
_CAVE_BASS_DRONE = [
    _simple_rhythm(_TPB, _BPB, [0], [4]),             # whole-note drone
    _simple_rhythm(_TPB, _BPB, [0, 2], [2, 2]),       # half-note pedal
]
_CAVE_DRUM_DRIP = [
    _simple_rhythm(_TPB, _BPB, [0, 2.5],
                   [0.25, 0.25]),                     # sparse water drips
    _simple_rhythm(_TPB, _BPB, [0, 1.5, 3],
                   [0.25, 0.25, 0.25]),               # triple drip
]
_CAVE_HARMONY_DRONE = [
    _simple_rhythm(_TPB, _BPB, [0], [4]),             # whole-bar sustain
]

# -- GBA Cave: tense, rhythmic undercurrent --
_GBA_CAVE_LEAD = [
    _simple_rhythm(_TPB, _BPB, [0, 0.5, 2, 3],
                   [0.5, 1, 0.5, 0.5]),              # nervous stutter
    _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3.5],
                   [0.75, 0.75, 1, 0.5]),             # uneasy walk
    _simple_rhythm(_TPB, _BPB, [0, 1.5, 2.5, 3.5],
                   [1, 0.75, 0.75, 0.5]),             # creeping
]
_GBA_CAVE_BASS = [
    _simple_rhythm(_TPB, _BPB, [0, 1.5, 3],
                   [1, 1, 1]),                        # stalking 3+3+2
    _simple_rhythm(_TPB, _BPB, [0, 2],
                   [1.5, 1.5]),                       # half-note pulse
]
_GBA_CAVE_DRUM = [
    _simple_rhythm(_TPB, _BPB, [0, 1, 2, 3],
                   [0.25] * 4),                       # quarter ticks
    _simple_rhythm(_TPB, _BPB, [0, 0.5, 1.5, 2.5, 3.5],
                   [0.25] * 5),                       # off-beat tension
]

# -- SNES Cave: atmospheric, reverb-soaked pads --
_SNES_CAVE_LEAD = [
    _simple_rhythm(_TPB, _BPB, [0, 2],
                   [2, 2]),                           # half notes
    _simple_rhythm(_TPB, _BPB, [0, 1.5, 3],
                   [1.5, 1, 0.5]),                    # lyrical echo
    _simple_rhythm(_TPB, _BPB, [0, 3],
                   [2.5, 1.5]),                       # wide sustain
]
_SNES_CAVE_BASS = [
    _simple_rhythm(_TPB, _BPB, [0], [4]),             # whole-note drone
    _simple_rhythm(_TPB, _BPB, [0, 2], [2, 2]),       # half-note walk
]
_SNES_CAVE_DRUM = [
    _simple_rhythm(_TPB, _BPB, [0, 2],
                   [0.25, 0.25]),                     # minimal percussion
]


# ── Cave section boundaries ─────────────────────────────────────────

def _cave_sections(total_bars: int) -> dict[str, int]:
    """Return section boundary indices for Cave generation.

    Caves have longer drone intros and atmospheric tails.
    """
    if total_bars <= 8:
        return {
            "intro_end": 1,
            "head_end": total_bars // 2 + 1,
            "dev_end": total_bars - 1,
            "climax_end": total_bars - 1,
            "reset_start": total_bars - 1,
            "silent_tail": total_bars,
        }
    if total_bars <= 16:
        _he = random.choice([5, 6, 7])
        _de = random.choice([10, 11, 12])
        return {
            "intro_end": random.choice([2, 2, 3]),
            "head_end": _he,
            "dev_end": _de,
            "climax_end": 14,
            "reset_start": 14,
            "silent_tail": total_bars,
        }
    # 32+
    _intro = random.choice([2, 3, 3, 4])
    _head  = _intro + random.choice([6, 7, 8])
    _dev   = _head + random.choice([7, 8, 9])
    return {
        "intro_end": _intro,
        "head_end": _head,
        "dev_end": _dev,
        "climax_end": total_bars - random.choice([4, 5, 6]),
        "reset_start": total_bars - random.choice([4, 5, 6]),
        "silent_tail": total_bars - random.choice([2, 3, 3]),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generic Town — 8-bit Hometown (NES / GB era)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_generic_town() -> GenreConfig:
    """Generic Town — warm 8-bit NES era.

    Simple diatonic triads, pentatonic-biased melodies,
    chip-arpeggio harmony, minimal drums.
    Nostalgic and singable — the simplest of the town variants.
    """
    vibe = random.choice(["inland", "inland", "inland", "island"])
    bpm_range = (115, 132) if vibe == "inland" else (120, 138)
    scale_choices = ["major", "pentatonic_major"] if vibe == "inland" else ["mixolydian", "major"]
    root_choices = [0, 2, 4, 5, 7, 9] if vibe == "inland" else [0, 2, 5, 7, 10]

    return GenreConfig(
        name="Generic Town",
        bpm_range=bpm_range,
        bars=32,
        root_choices=root_choices,
        scale_choices=scale_choices,
        vibe=vibe,
        progression_family="nes",

        lead_rhythm=_NES_LEAD_RHYTHMS,
        bass_rhythm=_NES_BASS_PUMP,
        drum_rhythm=_NES_DRUM_TICK,

        lead_instrument=random.choice([
            ("Generic Sine", "sine"),
            ("Generic Triangle", "triangle"),
        ]),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Generic Pulse 25%", "pulse25"),
        drum_instrument=("Generic Noise Drum", "noise"),

        lead_vel=(82, 102),
        bass_vel=(82, 94),
        harmony_vel=(40, 60),
        drum_vel=(18, 28),
        lead_mix_range=(0.78, 0.92),
        bass_mix_range=(0.55, 0.70),
        harmony_mix_range=(0.18, 0.30),
        drum_mix_range=(0.08, 0.16),
        lead_range=(56, 74),
        bass_range=(36, 55),
        harmony_range=(54, 72),
        drum_pitches=[42],                   # hi-hat only (minimal NES)
        lead_step_max=2,                     # smaller steps — pentatonic feel
        lead_rest_prob=0.15,                 # more rests — breathing space
        lead_style="nes_town",               # ★ NEW pentatonic melody style
        bass_style="simple_root",            # ★ NEW simple root-note bass
        harmony_style="arpeggio_chip",       # ★ NEW chiptune arpeggio
        drum_style="simple_tick",            # ★ NEW minimal ticking
        harmony_rhythm=_NES_HARMONY_ARP,
        swing=0.0,                           # straight feel (no shuffle)

        # Lead doubling with NES square sparkle
        lead_doubling=("NES Square", "square", 0.10),

        extra_tracks=[
            ExtraTrackConfig(
                role="Pad",
                gen_type="pad",
                instrument=("Generic Saw", "sawtooth"),
                vel=(30, 45),
                pitch_range=(48, 72),
                mix_range=(0.15, 0.28),
                rest_prob=0.25,
            ),
        ],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GBA Town — jazz-bounce Sappy engine era
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_gba_town() -> GenreConfig:
    """GBA Town — jazzy bounce (Sappy / m4a engine era).

    Jazz 7th chords, secondary dominants, call-response melodies,
    3+3+2 staccato harmony stabs, shuffle feel, velocity ramps.
    """
    vibe = random.choice(["inland", "inland", "inland", "island"])
    bpm_range = (118, 132) if vibe == "inland" else (122, 140)
    scale_choices = ["major", "dorian", "mixolydian"] if vibe == "inland" else ["mixolydian", "dorian"]
    root_choices = [0, 1, 2, 4, 5, 7, 9, 10] if vibe == "inland" else [0, 2, 3, 5, 7, 10]

    if vibe == "inland":
        lead_inst = random.choice([("GBA Flute", "sine"), ("GBA Ocarina", "sine")])
        counter_inst = random.choice([("GBA Vibraphone", "triangle"),
                                       ("GBA Glockenspiel", "triangle")])
    else:
        lead_inst = random.choice([("GBA Muted Trumpet", "sawtooth"),
                                    ("GBA Steel Drums", "triangle")])
        counter_inst = ("GBA Vibraphone", "triangle")

    extras = [
        ExtraTrackConfig(
            role="Strings Pad",
            gen_type="pad",
            instrument=("GBA Strings", "sawtooth"),
            vel=(40, 60),
            pitch_range=(48, 72),
            mix_range=(0.20, 0.35),
            rest_prob=0.20,
            pan=0.30,
        ),
        ExtraTrackConfig(
            role="Counter Melody",
            gen_type="counter_melody",
            instrument=counter_inst,
            vel=(45, 65),
            pitch_range=(60, 84),
            mix_range=(0.22, 0.38),
            rest_prob=0.50,
            step_max=2,
            pan=0.15 if vibe == "inland" else 0.20,
        ),
    ]

    if vibe == "inland":
        extras.append(ExtraTrackConfig(
            role="Guitar Arpeggio",
            gen_type="arpeggio",
            instrument=("GBA Acoustic Guitar", "triangle"),
            vel=(40, 58),
            pitch_range=(48, 72),
            mix_range=(0.20, 0.35),
            rest_prob=0.30,
            pan=-0.15,
        ))

    return GenreConfig(
        name="GBA Town",
        bpm_range=bpm_range,
        bars=32,
        root_choices=root_choices,
        scale_choices=scale_choices,
        vibe=vibe,
        progression_family="gba",

        lead_rhythm=_GBA_LEAD_RHYTHMS,
        bass_rhythm=_GBA_BASS_332,
        drum_rhythm=_GBA_DRUM_8THS,

        lead_instrument=lead_inst,
        bass_instrument=random.choice([("GBA Fretless Bass", "triangle"),
                                       ("GBA Slap Bass", "square")]),
        harmony_instrument=("GBA Piano", "pulse25"),
        drum_instrument=("GBA Light Kit", "noise"),

        lead_vel=(88, 108),
        bass_vel=(85, 95),
        harmony_vel=(50, 72) if vibe == "inland" else (48, 68),
        drum_vel=(20, 32) if vibe == "inland" else (22, 35),
        lead_mix_range=(0.80, 0.94),
        bass_mix_range=(0.56, 0.72),
        harmony_mix_range=(0.28, 0.42),
        drum_mix_range=(0.12, 0.22),
        lead_range=(58, 76),
        bass_range=(36, 55),
        harmony_range=(54, 72),
        drum_pitches=[38, 42],
        lead_step_max=3,
        lead_rest_prob=0.08 if vibe == "inland" else 0.06,
        lead_style="gba_town",
        bass_style="root_fifth",
        harmony_style="staccato",
        drum_style="gba_town",
        harmony_rhythm=_GBA_HARMONY_STACCATO,

        lead_pan=0.0,
        bass_pan=0.0,
        harmony_pan=-0.30,
        drum_pan=0.0,

        lead_doubling=("NES Square", "square", 0.12),
        swing=0.10,

        extra_tracks=extras,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SNES Town — dreamy 16-bit orchestral
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_snes_town() -> GenreConfig:
    """SNES Town — dreamy 16-bit orchestral.

    Lush suspended chords, walking bass, legato melodies with Lydian
    touches, sustained pads, brush-style drums.
    The most "orchestral" town variant.
    """
    vibe = random.choice(["inland", "inland", "inland", "island"])
    bpm_range = (110, 126) if vibe == "inland" else (118, 132)
    scale_choices = ["major", "lydian", "melodic_minor"] if vibe == "inland" else ["mixolydian", "lydian"]
    root_choices = [0, 2, 3, 5, 7, 8, 10] if vibe == "inland" else [0, 2, 5, 7, 9, 10]

    lead_inst = random.choice([
        ("SNES Flute", "sine"),
        ("SNES Trumpet", "sawtooth"),
    ])

    extras = [
        # Strings pad (warm sustain)
        ExtraTrackConfig(
            role="Strings Pad",
            gen_type="pad",
            instrument=("SNES Strings", "sawtooth"),
            vel=(35, 52),
            pitch_range=(48, 72),
            mix_range=(0.22, 0.38),
            rest_prob=0.15,
            pan=0.25,
        ),
        # Harp arpeggios
        ExtraTrackConfig(
            role="Harp Arpeggio",
            gen_type="arpeggio",
            instrument=("SNES Harp", "sine"),
            vel=(38, 55),
            pitch_range=(48, 72),
            mix_range=(0.18, 0.32),
            rest_prob=0.25,
            pan=-0.20,
        ),
        # Marimba counter / twinkle
        ExtraTrackConfig(
            role="Marimba Counter",
            gen_type="counter_melody",
            instrument=("SNES Marimba", "triangle"),
            vel=(40, 58),
            pitch_range=(60, 84),
            mix_range=(0.18, 0.32),
            rest_prob=0.55,
            step_max=2,
            pan=0.15,
        ),
    ]

    return GenreConfig(
        name="SNES Town",
        bpm_range=bpm_range,
        bars=32,
        root_choices=root_choices,
        scale_choices=scale_choices,
        vibe=vibe,
        progression_family="snes",

        lead_rhythm=_SNES_LEAD_RHYTHMS,
        bass_rhythm=_SNES_BASS_WALK,
        drum_rhythm=_SNES_DRUM_BRUSH,

        lead_instrument=lead_inst,
        bass_instrument=("SNES Slap Bass", "square"),
        harmony_instrument=("SNES Piano", "pulse25"),
        drum_instrument=("SNES Kit", "noise"),

        lead_vel=(85, 108),
        bass_vel=(82, 94),
        harmony_vel=(42, 62),
        drum_vel=(14, 26),
        lead_mix_range=(0.78, 0.92),
        bass_mix_range=(0.52, 0.68),
        harmony_mix_range=(0.25, 0.40),
        drum_mix_range=(0.06, 0.15),
        lead_range=(56, 74),
        bass_range=(36, 55),
        harmony_range=(54, 72),
        drum_pitches=[42],                   # hi-hat only (brush)
        lead_step_max=5,                     # wider intervals — legato leap
        lead_rest_prob=0.06,                 # fewer rests — flowing melody
        lead_style="snes_town",              # ★ NEW legato/Lydian melody
        bass_style="walking",                # ★ walking bass (reuse existing)
        harmony_style="sustained_pad",       # ★ NEW held sus/maj7 pads
        drum_style="snes_brush",             # ★ NEW gentle brush
        harmony_rhythm=[],                   # not needed for sustained pad

        lead_pan=0.0,
        bass_pan=0.0,
        harmony_pan=-0.25,
        drum_pan=0.0,

        # SNES doubling: flute + triangle sparkle
        lead_doubling=("SNES Acoustic", "triangle", 0.10),
        swing=0.05,                          # very light swing

        extra_tracks=extras,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mix Town — cross-family 16-bit mashup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_mix_town() -> GenreConfig:
    """Mix Town — instruments from any family, unrestricted.

    Randomly picks lead/bass/harmony/drum instruments from ALL families.
    Uses a random progression family. Every generation is a surprise.
    """
    vibe = random.choice(["inland", "inland", "inland", "island"])

    # Random progression family
    prog_family = random.choice(["nes", "gba", "snes"])

    if prog_family == "nes":
        bpm_range = (112, 136)
        scale_choices = ["major", "pentatonic_major", "mixolydian"]
        lead_rhythm = _NES_LEAD_RHYTHMS
        bass_rhythm = _NES_BASS_PUMP
        drum_rhythm = _NES_DRUM_TICK
        harmony_rhythm = _NES_HARMONY_ARP
    elif prog_family == "gba":
        bpm_range = (116, 138)
        scale_choices = ["major", "dorian", "mixolydian"]
        lead_rhythm = _GBA_LEAD_RHYTHMS
        bass_rhythm = _GBA_BASS_332
        drum_rhythm = _GBA_DRUM_8THS
        harmony_rhythm = _GBA_HARMONY_STACCATO
    else:  # snes
        bpm_range = (108, 130)
        scale_choices = ["major", "lydian", "melodic_minor"]
        lead_rhythm = _SNES_LEAD_RHYTHMS
        bass_rhythm = _SNES_BASS_WALK
        drum_rhythm = _SNES_DRUM_BRUSH
        harmony_rhythm = []

    root_choices = list(range(12))

    # Pick instruments from ANY family
    _ALL_LEADS = [
        ("Generic Sine", "sine"), ("Generic Triangle", "triangle"),
        ("NES Square", "square"), ("NES Pulse 25%", "pulse25"),
        ("Gameboy Square", "square"),
        ("SNES Flute", "sine"), ("SNES Trumpet", "sawtooth"),
        ("GBA Flute", "sine"), ("GBA Ocarina", "sine"),
        ("GBA Muted Trumpet", "sawtooth"),
    ]
    _ALL_BASS = [
        ("Generic Triangle", "triangle"),
        ("SNES Slap Bass", "square"),
        ("GBA Fretless Bass", "triangle"), ("GBA Slap Bass", "square"),
    ]
    _ALL_HARMONY = [
        ("Generic Pulse 25%", "pulse25"), ("Generic Square", "square"),
        ("SNES Piano", "pulse25"), ("GBA Piano", "pulse25"),
    ]
    _ALL_DRUMS = [
        ("Generic Noise Drum", "noise"),
        ("NES Noise", "noise"), ("SNES Kit", "noise"),
        ("GBA Light Kit", "noise"),
    ]

    lead_inst = random.choice(_ALL_LEADS)
    bass_inst = random.choice(_ALL_BASS)
    harm_inst = random.choice(_ALL_HARMONY)
    drum_inst = random.choice(_ALL_DRUMS)

    # Lead style follows progression family
    lead_style = {"nes": "nes_town", "gba": "gba_town", "snes": "snes_town"}[prog_family]
    bass_style = {"nes": "simple_root", "gba": "root_fifth", "snes": "walking"}[prog_family]
    harmony_style_val = {"nes": "arpeggio_chip", "gba": "staccato", "snes": "sustained_pad"}[prog_family]
    drum_style = {"nes": "simple_tick", "gba": "gba_town", "snes": "snes_brush"}[prog_family]

    # Sparkle from a random different family
    _SPARKLE_CHOICES = [
        ("NES Square", "square"), ("SNES Acoustic", "triangle"),
        ("GBA Glockenspiel", "triangle"), ("Gameboy Square", "square"),
    ]

    extras = [
        ExtraTrackConfig(
            role="Pad",
            gen_type="pad",
            instrument=random.choice([
                ("Generic Saw", "sawtooth"), ("SNES Strings", "sawtooth"),
                ("GBA Strings", "sawtooth"),
            ]),
            vel=(32, 48),
            pitch_range=(48, 72),
            mix_range=(0.15, 0.30),
            rest_prob=0.22,
        ),
    ]

    return GenreConfig(
        name="Mix Town",
        bpm_range=bpm_range,
        bars=32,
        root_choices=root_choices,
        scale_choices=scale_choices,
        vibe=vibe,
        progression_family=prog_family,

        lead_rhythm=lead_rhythm,
        bass_rhythm=bass_rhythm,
        drum_rhythm=drum_rhythm,
        harmony_rhythm=harmony_rhythm,

        lead_instrument=lead_inst,
        bass_instrument=bass_inst,
        harmony_instrument=harm_inst,
        drum_instrument=drum_inst,

        lead_vel=(84, 105),
        bass_vel=(82, 95),
        harmony_vel=(42, 62),
        drum_vel=(16, 30),
        lead_mix_range=(0.78, 0.93),
        bass_mix_range=(0.54, 0.70),
        harmony_mix_range=(0.20, 0.35),
        drum_mix_range=(0.08, 0.18),
        lead_range=(56, 76),
        bass_range=(36, 55),
        harmony_range=(54, 72),
        drum_pitches=[38, 42] if prog_family == "gba" else [42],
        lead_step_max={"nes": 2, "gba": 3, "snes": 5}[prog_family],
        lead_rest_prob={"nes": 0.14, "gba": 0.08, "snes": 0.06}[prog_family],
        lead_style=lead_style,
        bass_style=bass_style,
        harmony_style=harmony_style_val,
        drum_style=drum_style,
        swing={"nes": 0.0, "gba": 0.10, "snes": 0.05}[prog_family],

        lead_doubling=random.choice(_SPARKLE_CHOICES) + (0.10,),

        extra_tracks=extras,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generic Cave — dark 8-bit dungeon
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_generic_cave() -> GenreConfig:
    """Generic Cave — dark, echoey 8-bit dungeon.

    Natural minor, sparse melodies, drone bass, water-drip drums.
    Tense and atmospheric.
    """
    vibe = random.choice(["shallow", "shallow", "deep"])
    bpm_range = (75, 100) if vibe == "deep" else (85, 108)
    scale_choices = (["natural_minor", "phrygian"] if vibe == "deep"
                     else ["natural_minor", "harmonic_minor", "dorian"])
    root_choices = [0, 1, 3, 5, 7, 8, 10]

    return GenreConfig(
        name="Generic Cave",
        bpm_range=bpm_range,
        bars=32,
        root_choices=root_choices,
        scale_choices=scale_choices,
        vibe=vibe,
        progression_family="nes",

        lead_rhythm=_CAVE_LEAD_SPARSE,
        bass_rhythm=_CAVE_BASS_DRONE,
        drum_rhythm=_CAVE_DRUM_DRIP,
        harmony_rhythm=_CAVE_HARMONY_DRONE,

        lead_instrument=random.choice([
            ("Generic Sine", "sine"),
            ("Generic Triangle", "triangle"),
        ]),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Generic Pulse 25%", "pulse25"),
        drum_instrument=("Generic Noise Drum", "noise"),

        lead_vel=(60, 82),
        bass_vel=(65, 80),
        harmony_vel=(30, 48),
        drum_vel=(10, 22),
        lead_mix_range=(0.65, 0.82),
        bass_mix_range=(0.50, 0.65),
        harmony_mix_range=(0.15, 0.28),
        drum_mix_range=(0.05, 0.12),
        lead_range=(50, 70),
        bass_range=(32, 50),
        harmony_range=(48, 66),
        drum_pitches=[42],
        lead_step_max=3,
        lead_rest_prob=0.30,
        lead_style="cave_generic",
        bass_style="drone_pedal",
        harmony_style="dark_drone",
        drum_style="cave_drip",
        swing=0.0,

        extra_tracks=[],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GBA Cave — tense jazz-minor dungeon
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_gba_cave() -> GenreConfig:
    """GBA Cave — tense, dim7-flavoured dungeon atmosphere.

    m7/dim7 jazz-minor harmony, nervous lead patterns, stalking bass,
    sparse percussion with tension.
    """
    vibe = random.choice(["shallow", "shallow", "deep"])
    bpm_range = (80, 105) if vibe == "deep" else (90, 115)
    scale_choices = (["harmonic_minor", "phrygian"] if vibe == "deep"
                     else ["natural_minor", "dorian", "harmonic_minor"])
    root_choices = [0, 1, 3, 5, 6, 8, 10]

    lead_inst = random.choice([
        ("GBA Flute", "sine"), ("GBA Ocarina", "sine"),
    ])

    extras = [
        ExtraTrackConfig(
            role="Strings Pad",
            gen_type="pad",
            instrument=("GBA Strings", "sawtooth"),
            vel=(30, 48),
            pitch_range=(42, 66),
            mix_range=(0.15, 0.28),
            rest_prob=0.35,
            pan=0.20,
        ),
    ]

    return GenreConfig(
        name="GBA Cave",
        bpm_range=bpm_range,
        bars=32,
        root_choices=root_choices,
        scale_choices=scale_choices,
        vibe=vibe,
        progression_family="gba",

        lead_rhythm=_GBA_CAVE_LEAD,
        bass_rhythm=_GBA_CAVE_BASS,
        drum_rhythm=_GBA_CAVE_DRUM,
        harmony_rhythm=_CAVE_HARMONY_DRONE,

        lead_instrument=lead_inst,
        bass_instrument=random.choice([
            ("GBA Fretless Bass", "triangle"),
            ("GBA Slap Bass", "square"),
        ]),
        harmony_instrument=("GBA Piano", "pulse25"),
        drum_instrument=("GBA Light Kit", "noise"),

        lead_vel=(65, 88),
        bass_vel=(68, 84),
        harmony_vel=(32, 52),
        drum_vel=(12, 24),
        lead_mix_range=(0.68, 0.85),
        bass_mix_range=(0.52, 0.68),
        harmony_mix_range=(0.18, 0.32),
        drum_mix_range=(0.06, 0.14),
        lead_range=(52, 72),
        bass_range=(34, 52),
        harmony_range=(48, 68),
        drum_pitches=[38, 42],
        lead_step_max=3,
        lead_rest_prob=0.22,
        lead_style="cave_gba",
        bass_style="drone_pedal",
        harmony_style="dark_drone",
        drum_style="cave_drip",

        lead_pan=0.0,
        bass_pan=0.0,
        harmony_pan=-0.20,
        drum_pan=0.0,

        swing=0.05,

        extra_tracks=extras,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SNES Cave — ethereal haunted atmosphere
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_snes_cave() -> GenreConfig:
    """SNES Cave — haunted, suspended, atmospheric.

    Ethereal pads, sparse legato melody, heavy reverb feel,
    minimal percussion. The most atmospheric cave variant.
    """
    vibe = random.choice(["shallow", "deep", "deep"])
    bpm_range = (70, 95) if vibe == "deep" else (80, 105)
    scale_choices = (["phrygian", "harmonic_minor"] if vibe == "deep"
                     else ["natural_minor", "dorian", "melodic_minor"])
    root_choices = [0, 1, 3, 5, 7, 8, 10]

    lead_inst = random.choice([
        ("SNES Flute", "sine"),
        ("SNES Harp", "sine"),
    ])

    extras = [
        ExtraTrackConfig(
            role="Strings Pad",
            gen_type="pad",
            instrument=("SNES Strings", "sawtooth"),
            vel=(28, 45),
            pitch_range=(42, 66),
            mix_range=(0.18, 0.32),
            rest_prob=0.30,
            pan=0.25,
        ),
        ExtraTrackConfig(
            role="Harp Echo",
            gen_type="arpeggio",
            instrument=("SNES Harp", "sine"),
            vel=(25, 42),
            pitch_range=(48, 72),
            mix_range=(0.12, 0.25),
            rest_prob=0.45,
            pan=-0.20,
        ),
    ]

    return GenreConfig(
        name="SNES Cave",
        bpm_range=bpm_range,
        bars=32,
        root_choices=root_choices,
        scale_choices=scale_choices,
        vibe=vibe,
        progression_family="snes",

        lead_rhythm=_SNES_CAVE_LEAD,
        bass_rhythm=_SNES_CAVE_BASS,
        drum_rhythm=_SNES_CAVE_DRUM,
        harmony_rhythm=[],

        lead_instrument=lead_inst,
        bass_instrument=("SNES Slap Bass", "square"),
        harmony_instrument=("SNES Piano", "pulse25"),
        drum_instrument=("SNES Kit", "noise"),

        lead_vel=(58, 82),
        bass_vel=(60, 78),
        harmony_vel=(28, 45),
        drum_vel=(8, 18),
        lead_mix_range=(0.62, 0.80),
        bass_mix_range=(0.48, 0.62),
        harmony_mix_range=(0.18, 0.32),
        drum_mix_range=(0.04, 0.10),
        lead_range=(50, 72),
        bass_range=(32, 50),
        harmony_range=(46, 66),
        drum_pitches=[42],
        lead_step_max=4,
        lead_rest_prob=0.25,
        lead_style="cave_snes",
        bass_style="drone_pedal",
        harmony_style="sustained_pad",
        drum_style="cave_drip",

        lead_pan=0.0,
        bass_pan=0.0,
        harmony_pan=-0.25,
        drum_pan=0.0,

        lead_doubling=("SNES Acoustic", "triangle", 0.08),
        swing=0.0,

        extra_tracks=extras,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mix Cave — cross-family dark dungeon
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_mix_cave() -> GenreConfig:
    """Mix Cave — instruments from any family, cave atmosphere.

    Dark, atmospheric, with randomly picked instruments.
    Combines the surprise of Mix Town with cave mood.
    """
    vibe = random.choice(["shallow", "shallow", "deep"])
    prog_family = random.choice(["nes", "gba", "snes"])

    bpm_range = (72, 105)
    scale_choices = ["natural_minor", "harmonic_minor", "phrygian", "dorian"]
    root_choices = [0, 1, 3, 5, 7, 8, 10]

    # Pick from ALL families
    _CAVE_LEADS = [
        ("Generic Sine", "sine"), ("Generic Triangle", "triangle"),
        ("SNES Flute", "sine"), ("SNES Harp", "sine"),
        ("GBA Flute", "sine"), ("GBA Ocarina", "sine"),
        ("NES Pulse 25%", "pulse25"),
    ]
    _CAVE_BASS = [
        ("Generic Triangle", "triangle"),
        ("SNES Slap Bass", "square"),
        ("GBA Fretless Bass", "triangle"),
    ]
    _CAVE_HARMONY = [
        ("Generic Pulse 25%", "pulse25"),
        ("SNES Piano", "pulse25"), ("GBA Piano", "pulse25"),
    ]
    _CAVE_DRUMS = [
        ("Generic Noise Drum", "noise"),
        ("SNES Kit", "noise"), ("GBA Light Kit", "noise"),
    ]

    lead_inst = random.choice(_CAVE_LEADS)
    bass_inst = random.choice(_CAVE_BASS)
    harm_inst = random.choice(_CAVE_HARMONY)
    drum_inst = random.choice(_CAVE_DRUMS)

    cave_style = {"nes": "cave_generic", "gba": "cave_gba", "snes": "cave_snes"}[prog_family]

    if prog_family == "nes":
        lead_rhythm = _CAVE_LEAD_SPARSE
        bass_rhythm = _CAVE_BASS_DRONE
        drum_rhythm = _CAVE_DRUM_DRIP
    elif prog_family == "gba":
        lead_rhythm = _GBA_CAVE_LEAD
        bass_rhythm = _GBA_CAVE_BASS
        drum_rhythm = _GBA_CAVE_DRUM
    else:
        lead_rhythm = _SNES_CAVE_LEAD
        bass_rhythm = _SNES_CAVE_BASS
        drum_rhythm = _SNES_CAVE_DRUM

    return GenreConfig(
        name="Mix Cave",
        bpm_range=bpm_range,
        bars=32,
        root_choices=root_choices,
        scale_choices=scale_choices,
        vibe=vibe,
        progression_family=prog_family,

        lead_rhythm=lead_rhythm,
        bass_rhythm=bass_rhythm,
        drum_rhythm=drum_rhythm,
        harmony_rhythm=_CAVE_HARMONY_DRONE,

        lead_instrument=lead_inst,
        bass_instrument=bass_inst,
        harmony_instrument=harm_inst,
        drum_instrument=drum_inst,

        lead_vel=(60, 85),
        bass_vel=(62, 80),
        harmony_vel=(28, 48),
        drum_vel=(10, 22),
        lead_mix_range=(0.65, 0.82),
        bass_mix_range=(0.50, 0.66),
        harmony_mix_range=(0.15, 0.30),
        drum_mix_range=(0.05, 0.12),
        lead_range=(50, 72),
        bass_range=(32, 52),
        harmony_range=(46, 68),
        drum_pitches=[42],
        lead_step_max=3,
        lead_rest_prob=0.26,
        lead_style=cave_style,
        bass_style="drone_pedal",
        harmony_style="dark_drone",
        drum_style="cave_drip",
        swing=0.0,

        extra_tracks=[],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NES Town — authentic Famicom 2A03 sound
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_nes_town() -> GenreConfig:
    """NES Town — strict 2A03 chip: 2× pulse, triangle, noise.

    Pentatonic-biased, simple progressions, punchy chip arps,
    very limited channel count but full of character.
    """
    vibe = random.choice(["inland", "inland", "inland", "island"])
    bpm_range = (118, 135) if vibe == "inland" else (122, 140)
    scale_choices = ["major", "pentatonic_major"] if vibe == "inland" else ["mixolydian", "major"]
    root_choices = [0, 2, 4, 5, 7, 9] if vibe == "inland" else [0, 2, 5, 7, 10]

    return GenreConfig(
        name="NES Town",
        bpm_range=bpm_range, bars=32,
        root_choices=root_choices, scale_choices=scale_choices,
        vibe=vibe, progression_family="nes",
        lead_rhythm=_NES_LEAD_RHYTHMS, bass_rhythm=_NES_BASS_PUMP,
        drum_rhythm=_NES_DRUM_TICK, harmony_rhythm=_NES_HARMONY_ARP,
        lead_instrument=random.choice([
            ("NES Square", "square"), ("NES Pulse 25%", "pulse25"),
        ]),
        bass_instrument=("NES Triangle", "triangle"),
        harmony_instrument=("NES Pulse 25%", "pulse25"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(84, 104), bass_vel=(82, 94),
        harmony_vel=(42, 60), drum_vel=(18, 28),
        lead_mix_range=(0.80, 0.94), bass_mix_range=(0.56, 0.72),
        harmony_mix_range=(0.18, 0.30), drum_mix_range=(0.08, 0.16),
        lead_range=(56, 74), bass_range=(36, 55),
        harmony_range=(54, 72), drum_pitches=[42],
        lead_step_max=2, lead_rest_prob=0.14,
        lead_style="nes_town", bass_style="simple_root",
        harmony_style="arpeggio_chip", drum_style="simple_tick",
        swing=0.0,
        lead_doubling=("NES Pulse 25%", "pulse25", 0.08),
        extra_tracks=[],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gameboy Town — DMG-01 lo-fi charm
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_gameboy_town() -> GenreConfig:
    """Gameboy Town — original DMG chip: 2× pulse, wave, noise.

    Very constrained palette. Charming, lo-fi, pentatonic melodies,
    wave-channel bass, minimal noise percussion.
    """
    vibe = random.choice(["inland", "inland", "island"])
    bpm_range = (112, 130) if vibe == "inland" else (116, 135)
    scale_choices = ["major", "pentatonic_major"] if vibe == "inland" else ["mixolydian", "major"]
    root_choices = [0, 2, 4, 5, 7, 9]

    return GenreConfig(
        name="Gameboy Town",
        bpm_range=bpm_range, bars=32,
        root_choices=root_choices, scale_choices=scale_choices,
        vibe=vibe, progression_family="nes",
        lead_rhythm=_NES_LEAD_RHYTHMS, bass_rhythm=_NES_BASS_PUMP,
        drum_rhythm=_NES_DRUM_TICK, harmony_rhythm=_NES_HARMONY_ARP,
        lead_instrument=("Gameboy Square", "square"),
        bass_instrument=("Gameboy Wave", "triangle"),
        harmony_instrument=("Gameboy Pulse 12.5%", "pulse12"),
        drum_instrument=("Gameboy Noise", "noise"),
        lead_vel=(80, 100), bass_vel=(78, 92),
        harmony_vel=(38, 56), drum_vel=(14, 24),
        lead_mix_range=(0.78, 0.92), bass_mix_range=(0.54, 0.70),
        harmony_mix_range=(0.16, 0.28), drum_mix_range=(0.06, 0.14),
        lead_range=(56, 74), bass_range=(36, 55),
        harmony_range=(54, 72), drum_pitches=[42],
        lead_step_max=2, lead_rest_prob=0.16,
        lead_style="nes_town", bass_style="simple_root",
        harmony_style="arpeggio_chip", drum_style="simple_tick",
        swing=0.0,
        lead_doubling=("Gameboy Pulse 12.5%", "pulse12", 0.06),
        extra_tracks=[],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NES Cave — eerie Famicom dungeon
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_nes_cave() -> GenreConfig:
    """NES Cave — dark chip-tune dungeon with 2A03 hardware.

    Minor keys, sparse pulse leads, triangle drone bass,
    noise drip percussion. Authentic NES horror feel.
    """
    vibe = random.choice(["shallow", "shallow", "deep"])
    bpm_range = (78, 102) if vibe == "deep" else (88, 110)
    scale_choices = (["natural_minor", "phrygian"] if vibe == "deep"
                     else ["natural_minor", "harmonic_minor"])
    root_choices = [0, 1, 3, 5, 7, 8, 10]

    return GenreConfig(
        name="NES Cave",
        bpm_range=bpm_range, bars=32,
        root_choices=root_choices, scale_choices=scale_choices,
        vibe=vibe, progression_family="nes",
        lead_rhythm=_CAVE_LEAD_SPARSE, bass_rhythm=_CAVE_BASS_DRONE,
        drum_rhythm=_CAVE_DRUM_DRIP, harmony_rhythm=_CAVE_HARMONY_DRONE,
        lead_instrument=random.choice([
            ("NES Pulse 25%", "pulse25"), ("NES Square", "square"),
        ]),
        bass_instrument=("NES Triangle", "triangle"),
        harmony_instrument=("NES Square", "square"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(62, 84), bass_vel=(66, 82),
        harmony_vel=(30, 48), drum_vel=(10, 22),
        lead_mix_range=(0.66, 0.84), bass_mix_range=(0.52, 0.66),
        harmony_mix_range=(0.14, 0.26), drum_mix_range=(0.05, 0.12),
        lead_range=(50, 70), bass_range=(32, 50),
        harmony_range=(48, 66), drum_pitches=[42],
        lead_step_max=3, lead_rest_prob=0.28,
        lead_style="cave_generic", bass_style="drone_pedal",
        harmony_style="dark_drone", drum_style="cave_drip",
        swing=0.0,
        extra_tracks=[],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gameboy Cave — lo-fi dungeon crawl
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_gameboy_cave() -> GenreConfig:
    """Gameboy Cave — eerie DMG dungeon.

    4-channel lo-fi horror. Wave-channel drone bass, pulse leads
    with wide rest gaps, noise drips. Minimal and unsettling.
    """
    vibe = random.choice(["shallow", "shallow", "deep"])
    bpm_range = (74, 98) if vibe == "deep" else (84, 106)
    scale_choices = (["natural_minor", "phrygian"] if vibe == "deep"
                     else ["natural_minor", "harmonic_minor", "dorian"])
    root_choices = [0, 1, 3, 5, 7, 8, 10]

    return GenreConfig(
        name="Gameboy Cave",
        bpm_range=bpm_range, bars=32,
        root_choices=root_choices, scale_choices=scale_choices,
        vibe=vibe, progression_family="nes",
        lead_rhythm=_CAVE_LEAD_SPARSE, bass_rhythm=_CAVE_BASS_DRONE,
        drum_rhythm=_CAVE_DRUM_DRIP, harmony_rhythm=_CAVE_HARMONY_DRONE,
        lead_instrument=("Gameboy Square", "square"),
        bass_instrument=("Gameboy Wave", "triangle"),
        harmony_instrument=("Gameboy Pulse 12.5%", "pulse12"),
        drum_instrument=("Gameboy Noise", "noise"),
        lead_vel=(58, 80), bass_vel=(62, 78),
        harmony_vel=(28, 45), drum_vel=(8, 20),
        lead_mix_range=(0.62, 0.80), bass_mix_range=(0.50, 0.64),
        harmony_mix_range=(0.12, 0.24), drum_mix_range=(0.04, 0.10),
        lead_range=(50, 70), bass_range=(32, 50),
        harmony_range=(48, 66), drum_pitches=[42],
        lead_step_max=3, lead_rest_prob=0.32,
        lead_style="cave_generic", bass_style="drone_pedal",
        harmony_style="dark_drone", drum_style="cave_drip",
        swing=0.0,
        extra_tracks=[],
    )


# ── Registry of all genres ──────────────────────────────────────────

GENRE_BUILDERS: dict[str, callable] = {
    "Generic Town":  _genre_generic_town,
    "NES Town":      _genre_nes_town,
    "Gameboy Town":  _genre_gameboy_town,
    "GBA Town":      _genre_gba_town,
    "SNES Town":     _genre_snes_town,
    "Mix Town":      _genre_mix_town,
    "Generic Cave":  _genre_generic_cave,
    "NES Cave":      _genre_nes_cave,
    "Gameboy Cave":  _genre_gameboy_cave,
    "GBA Cave":      _genre_gba_cave,
    "SNES Cave":     _genre_snes_cave,
    "Mix Cave":      _genre_mix_cave,
}

# Merge new genre presets from genre_presets.py
from daw.genre_presets import NEW_GENRE_BUILDERS as _NEW_BUILDERS
GENRE_BUILDERS.update(_NEW_BUILDERS)

GENRE_NAMES: list[str] = list(GENRE_BUILDERS.keys())


# ── Track generation ────────────────────────────────────────────────

@dataclass
class GeneratedTrack:
    role: str                   # "Lead", "Bass", "Harmony", "Drums"
    instrument_name: str
    waveform: str
    volume: float
    notes: list[NoteEvent]
    pan: float = 0.0            # -1.0 (left) .. +1.0 (right)


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
                   bars_override: int | None = None,
                   track_density: str = "standard") -> GeneratedMusic:
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
    track_density : str
        Controls how many tracks are generated:
        - ``"minimal"`` — 3 tracks: Lead, Bass, Drums.
        - ``"standard"`` — 4 tracks: Lead, Bass, Harmony, Drums (default).
        - ``"rich"`` — 5-6 tracks: core 4 + select extras.
        - ``"full"`` — all tracks (6-8), the old behaviour.

    Returns a ``GeneratedMusic`` object containing the requested tracks
    plus metadata.
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

    # Choose a chord progression — if vibe is set, build dynamically to match bar count
    if cfg.vibe:
        if cfg.progression_builder is not None:
            chord_prog = cfg.progression_builder(cfg.vibe, cfg.bars, cfg.progression_family)
        elif cfg.lead_style.startswith("cave_"):
            chord_prog = _cave_progression(cfg.vibe, cfg.bars, cfg.progression_family)
        else:
            chord_prog = _town_progression(cfg.vibe, cfg.bars, cfg.progression_family)
    elif cfg.progressions:
        chord_prog = list(random.choice(cfg.progressions))
    else:
        # Fallback: simple I-IV-V-I * bars/4
        chord_prog = [(0, "maj"), (3, "maj"), (4, "maj"), (0, "maj")] * (cfg.bars // 4 or 1)

    # Ensure progression covers all bars
    while len(chord_prog) < cfg.bars:
        chord_prog += chord_prog
    chord_prog = chord_prog[:cfg.bars]

    # Harmonic color passes:
    # - Secondary dominants add "adventure lift" before key targets
    # - Picardy ending gives major resolution in minor-ish loops (towns only)
    _DARK_STYLES = {"cave_generic", "cave_gba", "cave_snes",
                    "dungeon_dark", "combat_intense", "encounter_alarm",
                    "boss_epic"}
    is_dark = cfg.lead_style in _DARK_STYLES or cfg.lead_style.startswith("cave_")
    sec_dom_strength = {
        "nes": 0.16,
        "gba": 0.34,
        "snes": 0.24,
    }.get(cfg.progression_family, 0.20)
    if is_dark:
        sec_dom_strength *= 0.3   # dark styles stay minor — minimal dominant lifts
    chord_prog = _inject_secondary_dominants(chord_prog, strength=sec_dom_strength)
    # Picardy third only for bright themes (dark styles should stay minor)
    if not is_dark:
        chord_prog = _apply_picardy_third(chord_prog, scale_name=scale_name, chance=0.65)

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
                                bar_ticks, tpb, bpb, lead_notes=lead_notes)

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
            pan=cfg.lead_pan,
        ),
        GeneratedTrack(
            "Bass",
            cfg.bass_instrument[0],
            cfg.bass_instrument[1],
            _rand_mix(cfg.bass_mix_range),
            bass_notes,
            pan=cfg.bass_pan,
        ),
        GeneratedTrack(
            "Harmony",
            cfg.harmony_instrument[0],
            cfg.harmony_instrument[1],
            _rand_mix(cfg.harmony_mix_range),
            harmony_notes,
            pan=cfg.harmony_pan,
        ),
        GeneratedTrack(
            "Drums",
            cfg.drum_instrument[0],
            cfg.drum_instrument[1],
            _rand_mix(cfg.drum_mix_range),
            drum_notes,
            pan=cfg.drum_pan,
        ),
    ]

    # ── Lead-doubling track (GBA "sparkle" technique) ───────────
    if cfg.lead_doubling is not None:
        dbl_instr, dbl_wave, dbl_vol = cfg.lead_doubling
        # Clone the lead notes at lower volume
        dbl_notes = [NoteEvent(n.start_tick, n.length_tick, n.midi_note, n.velocity)
                     for n in lead_notes]
        tracks.append(GeneratedTrack(
            "Lead Sparkle",
            dbl_instr,
            dbl_wave,
            round(dbl_vol, 2),
            dbl_notes,
            pan=cfg.lead_pan,
        ))

    # ── Generate EXTRA tracks (flexible track count) ────────────
    for extra_cfg in cfg.extra_tracks:
        extra_notes = _generate_extra_track(
            extra_cfg, cfg, scale, chord_prog, intervals, root,
            bar_ticks, tpb, bpb, lead_notes,
        )
        tracks.append(GeneratedTrack(
            extra_cfg.role,
            extra_cfg.instrument[0],
            extra_cfg.instrument[1],
            _rand_mix(extra_cfg.mix_range),
            extra_notes,
            pan=extra_cfg.pan,
        ))

    # ── Track-density filtering ──────────────────────────────────────
    _CORE_ROLES = {"Lead", "Bass", "Harmony", "Drums"}
    if track_density == "minimal":
        # 3 tracks: Lead + Bass + Drums (harmony is implicit in bass root notes)
        tracks = [t for t in tracks if t.role in {"Lead", "Bass", "Drums"}]
    elif track_density == "standard":
        # 4 tracks: the classic town quartet
        tracks = [t for t in tracks if t.role in _CORE_ROLES]
    elif track_density == "rich":
        # 5-6 tracks: core 4 + up to 2 best extras (prefer Pad first)
        core = [t for t in tracks if t.role in _CORE_ROLES]
        extras = [t for t in tracks if t.role not in _CORE_ROLES
                  and t.role != "Lead Sparkle"]
        # Prioritise Pad > Counter Melody > Arpeggio
        _EXTRA_PRIORITY = {"Pad": 0, "Strings Pad": 0,
                           "Counter Melody": 1, "Marimba Counter": 1,
                           "Harp Arpeggio": 2, "Guitar Arpeggio": 2}
        extras.sort(key=lambda t: _EXTRA_PRIORITY.get(t.role, 9))
        tracks = core + extras[:2]
    # "full" keeps everything as-is

    # ── Swing / shuffle post-processing ────────────────────────────────
    if cfg.swing > 0:
        # Shift every off-beat 8th note forward by (swing * half-8th-note)
        eighth = tpb // 2 or 1           # 8th-note duration in ticks
        shift = max(1, int(eighth * cfg.swing))
        for t in tracks:
            for n in t.notes:
                # Detect off-beat 8th positions (those at odd multiples of eighth)
                pos_in_bar = n.start_tick % bar_ticks
                if eighth > 0 and (pos_in_bar // eighth) % 2 == 1:
                    n.start_tick += shift
                    # Shorten to avoid overlap (never shorter than 1 tick)
                    n.length_tick = max(1, n.length_tick - shift)

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
    prev_pitch: int | None = None          # for velocity ramps (gba_town)
    last_note: int | None = None
    repeat_run = 0

    rhythm_patterns = cfg.lead_rhythm or [
        _simple_rhythm(tpb, bpb, [0, 1, 2, 3], [1, 1, 1, 1])
    ]

    # Per-song motif identity (keeps one song coherent, makes songs distinct)
    motif_seed = random.randint(0, 999_999)
    nes_motif_bank = [
        [0, 2, 4, 2], [0, 2, 3, 1], [0, 1, 3, 1], [0, 3, 4, 2],
        [0, -1, 2, 1], [0, 2, 1, 3], [0, 3, 2, 4], [0, 2, 0, 3],
        # Extended: more contour variety
        [0, 4, 2, 0], [0, -2, 0, 2], [0, 1, -1, 2], [0, 3, 1, -1],
        [0, 2, 4, 5], [0, -1, -2, 0], [0, 1, 0, -1], [0, 3, 5, 3],
        [0, 2, -1, 1], [0, 4, 3, 1], [0, -1, 3, 2], [0, 1, 4, 2],
        [0, 0, 2, 4], [0, 3, 0, 2], [0, -2, 1, 3], [0, 2, 3, 0],
    ]
    gba_motif_bank = [
        [0, 2, 4, 5, 4, 2], [0, 1, 3, 4, 3, 1], [0, 3, 5, 6, 5, 3],
        [0, 2, 4, 6, 4, 2], [0, -1, 2, 4, 2, 0], [0, 3, 4, 6, 4, 3],
        # Extended: more contour variety
        [0, 1, 2, 4, 3, 0], [0, 4, 3, 1, 2, 0], [0, 2, 5, 3, 1, 2],
        [0, -1, 1, 3, 5, 3], [0, 3, 2, 0, 1, 3], [0, 5, 4, 2, 3, 1],
        [0, 2, 3, 5, 4, 1], [0, 1, 4, 2, 5, 3], [0, -2, 1, 3, 2, 0],
        [0, 4, 5, 3, 2, 4], [0, 3, 1, 4, 2, 0], [0, 2, 0, 3, 5, 2],
    ]
    snes_motif_bank = [
        [0, 2, 4, 3], [0, 2, 5, 4], [0, 3, 5, 4], [0, 3, 6, 5],
        [0, 4, 5, 4], [0, 2, 4, 6], [0, 3, 4, 5],
        # Extended: more contour variety
        [0, 5, 3, 1], [0, 1, 5, 3], [0, 4, 2, 5], [0, -1, 3, 5],
        [0, 2, 6, 4], [0, 3, 1, 4], [0, 5, 4, 2], [0, 1, 2, 6],
        [0, 4, 6, 3], [0, -1, 2, 4], [0, 3, 5, 1], [0, 2, 3, 6],
    ]

    # Cave motif banks — darker, more constrained, minor-biased
    cave_motif_bank = [
        [0, -1, 0, -2], [0, -2, -1, 0], [0, 1, -1, 0],
        [0, -1, -2, -1], [0, 0, -1, -2], [0, -2, 0, -1],
        [0, 1, 0, -1], [0, -1, 1, -2], [0, -2, -1, -3],
        [0, 0, -2, 0], [0, 1, -2, -1], [0, -3, -1, 0],
        [0, -1, -3, -2], [0, 2, 0, -1], [0, -2, 1, 0],
    ]

    # ── New motif banks for expanded genres ──
    # HM pastoral — very gentle, narrow stepwise, sentimental
    hm_motif_bank = [
        [0, 1, 2, 1], [0, -1, 0, 1], [0, 1, 0, -1],
        [0, 2, 1, 0], [0, 1, 2, 3], [0, -1, 1, 0],
        [0, 0, 1, 2], [0, 2, 0, 1], [0, 1, -1, 0],
        [0, -1, -2, -1], [0, 1, 3, 2], [0, 2, 1, 3],
    ]
    # Zelda folk — bouncy, wider intervals, modal charm
    zelda_motif_bank = [
        [0, 2, 4, 2], [0, 3, 1, 4], [0, -1, 2, 4],
        [0, 4, 2, 0], [0, 2, -1, 3], [0, 1, 3, 5],
        [0, 3, 5, 3], [0, -2, 1, 3], [0, 4, 3, 1],
        [0, 2, 5, 2], [0, -1, 3, 1], [0, 3, 2, 5],
    ]
    # Combat — aggressive, fast, angular
    combat_motif_bank = [
        [0, -2, 3, -1], [0, 3, -2, 1], [0, -3, 0, 3],
        [0, 2, -1, -3], [0, -1, 3, 0], [0, 4, -2, 2],
        [0, -2, -1, 3], [0, 3, 1, -2], [0, -3, 2, -1],
        [0, 0, -3, 3], [0, 2, 4, -1], [0, -1, -3, 0],
    ]
    # Boss — epic, dramatic wide leaps
    boss_motif_bank = [
        [0, -3, 4, -2], [0, 4, -3, 5], [0, -2, 5, -1],
        [0, 3, -4, 2], [0, -4, 3, -2], [0, 5, -2, 4],
        [0, -1, 4, -3], [0, 2, -3, 5], [0, -3, -1, 4],
        [0, 4, 2, -4], [0, -2, 3, 5], [0, 5, -3, 0],
    ]
    # Overworld — heroic, confident, march-like
    overworld_motif_bank = [
        [0, 2, 4, 2], [0, 1, 3, 4], [0, 3, 2, 4],
        [0, 2, 0, 3], [0, 4, 3, 2], [0, 1, 4, 3],
        [0, 3, 5, 4], [0, 2, 3, 1], [0, 4, 2, 3],
        [0, 1, 2, 4], [0, 3, 4, 5], [0, 2, 5, 3],
    ]
    # Victory — ascending, triumphant, short punchy
    victory_motif_bank = [
        [0, 2, 4, 7], [0, 4, 7, 9], [0, 2, 5, 7],
        [0, 3, 5, 7], [0, 4, 5, 7], [0, 2, 4, 5],
        [0, 5, 7, 9], [0, 3, 7, 5], [0, 4, 2, 7],
    ]
    # Dungeon — like cave but sparser, more tritone
    dungeon_motif_bank = [
        [0, -1, -3, -1], [0, -2, 0, -3], [0, 1, -2, 0],
        [0, -3, 0, -1], [0, 0, -2, -3], [0, -1, 0, -3],
        [0, -2, -3, 0], [0, 1, -3, -1], [0, -3, -2, 0],
        [0, -1, -2, 1], [0, 0, -1, -3], [0, -2, 1, -1],
    ]
    # Encounter — alarming ascending runs
    encounter_motif_bank = [
        [0, 1, 2, 3], [0, 2, 3, 5], [0, 1, 3, 4],
        [0, 3, 4, 6], [0, 2, 4, 5], [0, 1, 4, 3],
        [0, 3, 2, 4], [0, 2, 5, 3], [0, 4, 3, 5],
    ]

    # Signature chorus window (the "selling point" / hook section)
    # 8 bars  -> bars 2..3
    # 16 bars -> bars 6..9
    # 32 bars -> bars 14..17
    if cfg.bars <= 8:
        chorus_start, chorus_end = 2, 4
    elif cfg.bars <= 16:
        chorus_start, chorus_end = 6, 10
    else:
        chorus_start, chorus_end = 14, 18

    # Pick one motif family per generation, then repeat/answer it in chorus.
    # 5 variants per style for much more variety between songs.
    signature_variant = random.randint(0, 4)
    call_phrase_offsets: list[int] | None = None

    # Per-song rhythm personality: select a SUBSET of rhythm patterns
    # so each song has a consistent rhythmic feel, but different songs differ.
    _lead_rhythm_pool = cfg.lead_rhythm or [
        _simple_rhythm(tpb, bpb, [0, 1, 2, 3], [1, 1, 1, 1])
    ]
    if len(_lead_rhythm_pool) > 2:
        rhythm_subset = random.sample(_lead_rhythm_pool, k=min(2, len(_lead_rhythm_pool)))
    else:
        rhythm_subset = list(_lead_rhythm_pool)

    for bar_idx, (degree, _quality) in enumerate(chord_prog):
        bar_start = bar_idx * bar_ticks
        pattern = random.choice(rhythm_subset)
        response_mode = False

        # ── Genre signature chorus hooks ───────────────────────────
        # Each genre gets a distinct middle hook so songs don't blur together.
        _HOOK_STYLES = {"nes_town", "gba_town", "snes_town",
                        "hm_pastoral", "zelda_folk", "combat_intense",
                        "encounter_alarm", "boss_epic", "overworld_heroic",
                        "victory_fanfare", "dungeon_dark"}
        if chorus_start <= bar_idx < chorus_end and cfg.lead_style in _HOOK_STYLES:
            idx_in_scale = degree % len(intervals)
            chord_root_pc = (root + intervals[idx_in_scale]) % 12
            root_candidates = [i for i, n in enumerate(lead_scale) if n % 12 == chord_root_pc]
            if root_candidates:
                anchor_idx = min(root_candidates, key=lambda i: abs(i - current_idx))
            else:
                anchor_idx = current_idx

            phrase_pos = bar_idx - chorus_start

            # NES hook: simple, catchy, singable (chip-town style)
            if cfg.lead_style == "nes_town":
                _nes_hook_variants = [
                    [[0, 2, 4, 2], [0, 2, 5, 4]],
                    [[0, 1, 3, 1], [0, 3, 4, 2]],
                    [[0, 4, 2, 0], [0, 2, 3, 4]],
                    [[0, -1, 2, 3], [0, 3, 1, 0]],
                    [[0, 2, 0, 4], [0, 1, -1, 2]],
                ]
                motifs = _nes_hook_variants[signature_variant % len(_nes_hook_variants)]
                offsets = motifs[phrase_pos % len(motifs)]
                hook_pattern = _simple_rhythm(tpb, bpb, [0, 1, 2, 3], [1, 1, 1, 1])
                for note_i, (tick_off, length) in enumerate(hook_pattern):
                    if random.random() < max(0.02, cfg.lead_rest_prob * 0.25):
                        continue
                    idx = max(0, min(len(lead_scale) - 1, anchor_idx + offsets[note_i]))
                    midi = lead_scale[idx]
                    vel = random.randint(*cfg.lead_vel)
                    notes.append(NoteEvent(
                        start_tick=bar_start + tick_off,
                        length_tick=length,
                        midi_note=midi,
                        velocity=max(30, min(127, vel)),
                    ))
                    current_idx = idx
                    prev_pitch = midi
                continue

            # GBA hook: syncopated bounce + jump peak (gen III town identity)
            if cfg.lead_style == "gba_town":
                _gba_hook_variants = [
                    [[0, 2, 4, 5, 4], [0, 1, 3, 4, 3]],
                    [[0, 3, 5, 6, 5], [0, 2, 4, 5, 4]],
                    [[0, 1, 4, 3, 1], [0, 3, 2, 5, 3]],
                    [[0, 4, 2, 5, 3], [0, -1, 2, 4, 2]],
                    [[0, 2, 5, 4, 1], [0, 3, 1, 4, 2]],
                ]
                motifs = _gba_hook_variants[signature_variant % len(_gba_hook_variants)]
                offsets = motifs[phrase_pos % len(motifs)]
                hook_pattern = _simple_rhythm(
                    tpb, bpb,
                    [0, 0.5, 1.5, 2.5, 3.0],
                    [0.5, 0.5, 0.75, 0.5, 1.0],
                )
                for note_i, (tick_off, length) in enumerate(hook_pattern):
                    if random.random() < max(0.01, cfg.lead_rest_prob * 0.2):
                        continue
                    idx = max(0, min(len(lead_scale) - 1, anchor_idx + offsets[note_i]))
                    midi = lead_scale[idx]
                    vel = random.randint(*cfg.lead_vel)
                    if prev_pitch is not None:
                        if midi > prev_pitch:
                            vel += random.randint(3, 6)
                        elif midi < prev_pitch:
                            vel -= random.randint(2, 5)
                    notes.append(NoteEvent(
                        start_tick=bar_start + tick_off,
                        length_tick=length,
                        midi_note=midi,
                        velocity=max(30, min(127, vel)),
                    ))
                    current_idx = idx
                    prev_pitch = midi
                continue

            # SNES hook: long, lyrical arch with dreamy colour (maj7/sus world)
            if cfg.lead_style == "snes_town":
                _snes_hook_variants = [
                    [[0, 2, 4, 3], [0, 2, 5, 4]],
                    [[0, 3, 5, 4], [0, 3, 4, 2]],
                    [[0, 4, 3, 1], [0, 1, 3, 5]],
                    [[0, 5, 4, 2], [0, 2, 6, 4]],
                    [[0, 1, 4, 6], [0, 3, 2, 5]],
                ]
                motifs = _snes_hook_variants[signature_variant % len(_snes_hook_variants)]
                offsets = motifs[phrase_pos % len(motifs)]
                hook_pattern = _simple_rhythm(
                    tpb, bpb,
                    [0, 1.0, 2.5, 3.5],
                    [1.0, 1.5, 1.0, 0.5],
                )
                for note_i, (tick_off, length) in enumerate(hook_pattern):
                    if random.random() < max(0.01, cfg.lead_rest_prob * 0.15):
                        continue
                    idx = max(0, min(len(lead_scale) - 1, anchor_idx + offsets[note_i]))
                    midi = lead_scale[idx]
                    vel = random.randint(*cfg.lead_vel) + 3
                    notes.append(NoteEvent(
                        start_tick=bar_start + tick_off,
                        length_tick=length,
                        midi_note=midi,
                        velocity=max(30, min(127, vel)),
                    ))
                    current_idx = idx
                    prev_pitch = midi
                continue

            # ── HM Pastoral hook: gentle, narrow, sentimental ──
            if cfg.lead_style == "hm_pastoral":
                _hm_hook_variants = [
                    [[0, 1, 2, 1], [0, -1, 0, 1]],
                    [[0, 2, 1, 0], [0, 1, 2, 3]],
                    [[0, -1, 1, 2], [0, 2, 0, 1]],
                    [[0, 1, 3, 2], [0, 0, 1, -1]],
                    [[0, 2, 3, 1], [0, 1, -1, 0]],
                ]
                motifs = _hm_hook_variants[signature_variant % len(_hm_hook_variants)]
                offsets = motifs[phrase_pos % len(motifs)]
                hook_pattern = _simple_rhythm(tpb, bpb, [0, 1.5, 2.5, 3.5], [1.5, 1, 1, 0.5])
                for note_i, (tick_off, length) in enumerate(hook_pattern):
                    if random.random() < max(0.02, cfg.lead_rest_prob * 0.25):
                        continue
                    idx = max(0, min(len(lead_scale) - 1, anchor_idx + offsets[note_i]))
                    midi = lead_scale[idx]
                    vel = random.randint(*cfg.lead_vel) - 2
                    notes.append(NoteEvent(bar_start + tick_off, length, midi,
                                           max(30, min(127, vel))))
                    current_idx, prev_pitch = idx, midi
                continue

            # ── Zelda folk hook: bouncy pickup + wide leaps ──
            if cfg.lead_style == "zelda_folk":
                _zel_hook_variants = [
                    [[0, 2, 4, 2, 0], [0, 3, 1, 4, 2]],
                    [[0, -1, 3, 2, 4], [0, 4, 2, -1, 3]],
                    [[0, 3, 5, 3, 1], [0, 1, 4, 3, 5]],
                    [[0, 2, -1, 3, 1], [0, 4, 3, 1, 2]],
                    [[0, 1, 3, 5, 3], [0, -1, 2, 4, 2]],
                ]
                motifs = _zel_hook_variants[signature_variant % len(_zel_hook_variants)]
                offsets = motifs[phrase_pos % len(motifs)]
                hook_pattern = _simple_rhythm(tpb, bpb, [0, 0.5, 1, 2, 3], [0.5, 0.5, 1, 1, 1])
                for note_i, (tick_off, length) in enumerate(hook_pattern):
                    if random.random() < max(0.01, cfg.lead_rest_prob * 0.2):
                        continue
                    idx = max(0, min(len(lead_scale) - 1, anchor_idx + offsets[note_i]))
                    midi = lead_scale[idx]
                    vel = random.randint(*cfg.lead_vel)
                    notes.append(NoteEvent(bar_start + tick_off, length, midi,
                                           max(30, min(127, vel))))
                    current_idx, prev_pitch = idx, midi
                continue

            # ── Combat hook: aggressive, angular, fast ──
            if cfg.lead_style == "combat_intense":
                _cmb_hook_variants = [
                    [[0, -2, 3, -1, 2, -3], [0, 3, -1, 2, -2, 1]],
                    [[0, 2, -3, 1, 3, -2], [0, -1, 3, -2, 0, 2]],
                    [[0, -3, 0, 3, -1, 2], [0, 2, -2, 3, 0, -1]],
                    [[0, 3, 1, -2, 2, -1], [0, -2, 2, -1, 3, 0]],
                    [[0, -1, -3, 2, 0, 3], [0, 3, -3, 1, -1, 2]],
                ]
                motifs = _cmb_hook_variants[signature_variant % len(_cmb_hook_variants)]
                offsets = motifs[phrase_pos % len(motifs)]
                hook_pattern = _simple_rhythm(tpb, bpb,
                    [0, 0.5, 1, 1.5, 2, 2.5], [0.5] * 6)
                for note_i, (tick_off, length) in enumerate(hook_pattern):
                    if random.random() < 0.02:
                        continue
                    idx = max(0, min(len(lead_scale) - 1, anchor_idx + offsets[note_i]))
                    midi = lead_scale[idx]
                    vel = random.randint(*cfg.lead_vel) + 5
                    notes.append(NoteEvent(bar_start + tick_off, length, midi,
                                           max(30, min(127, vel))))
                    current_idx, prev_pitch = idx, midi
                continue

            # ── Encounter hook: alarming ascending runs ──
            if cfg.lead_style == "encounter_alarm":
                _enc_hook_variants = [
                    [[0, 1, 2, 3, 4, 5], [0, 2, 3, 5, 4, 3]],
                    [[0, 2, 4, 5, 3, 4], [0, 1, 3, 4, 6, 5]],
                    [[0, 3, 2, 4, 5, 3], [0, 1, 4, 3, 5, 4]],
                    [[0, 2, 1, 3, 5, 4], [0, 4, 3, 5, 2, 4]],
                    [[0, 1, 3, 5, 3, 6], [0, 3, 5, 4, 6, 5]],
                ]
                motifs = _enc_hook_variants[signature_variant % len(_enc_hook_variants)]
                offsets = motifs[phrase_pos % len(motifs)]
                hook_pattern = _simple_rhythm(tpb, bpb,
                    [0, 0.5, 1, 1.5, 2, 3], [0.5] * 6)
                for note_i, (tick_off, length) in enumerate(hook_pattern):
                    idx = max(0, min(len(lead_scale) - 1, anchor_idx + offsets[note_i]))
                    midi = lead_scale[idx]
                    vel = random.randint(*cfg.lead_vel) + 6
                    notes.append(NoteEvent(bar_start + tick_off, length, midi,
                                           max(30, min(127, vel))))
                    current_idx, prev_pitch = idx, midi
                continue

            # ── Boss hook: dramatic wide leaps, epic phrasing ──
            if cfg.lead_style == "boss_epic":
                _bss_hook_variants = [
                    [[0, -3, 4, -2, 5], [0, 4, -3, 5, -1]],
                    [[0, 3, -4, 2, -3], [0, -2, 5, -1, 4]],
                    [[0, 5, -2, 3, -4], [0, -3, 4, -1, 5]],
                    [[0, -4, 3, -2, 4], [0, 5, -3, 2, -1]],
                    [[0, 2, -3, 5, -2], [0, -1, 4, -3, 5]],
                ]
                motifs = _bss_hook_variants[signature_variant % len(_bss_hook_variants)]
                offsets = motifs[phrase_pos % len(motifs)]
                hook_pattern = _simple_rhythm(tpb, bpb,
                    [0, 0.5, 1, 2.5, 3], [0.5, 0.5, 1.5, 0.5, 1])
                for note_i, (tick_off, length) in enumerate(hook_pattern):
                    if random.random() < 0.02:
                        continue
                    idx = max(0, min(len(lead_scale) - 1, anchor_idx + offsets[note_i]))
                    midi = lead_scale[idx]
                    vel = random.randint(*cfg.lead_vel) + 8
                    notes.append(NoteEvent(bar_start + tick_off, length, midi,
                                           max(30, min(127, vel))))
                    current_idx, prev_pitch = idx, midi
                continue

            # ── Overworld hook: heroic march, confident ──
            if cfg.lead_style == "overworld_heroic":
                _ovw_hook_variants = [
                    [[0, 2, 4, 2, 0], [0, 1, 3, 4, 3]],
                    [[0, 4, 3, 5, 4], [0, 2, 3, 5, 2]],
                    [[0, 3, 5, 4, 2], [0, 1, 4, 3, 5]],
                    [[0, 2, 5, 3, 4], [0, 4, 2, 5, 3]],
                    [[0, 1, 3, 5, 3], [0, 3, 4, 2, 4]],
                ]
                motifs = _ovw_hook_variants[signature_variant % len(_ovw_hook_variants)]
                offsets = motifs[phrase_pos % len(motifs)]
                hook_pattern = _simple_rhythm(tpb, bpb,
                    [0, 0.5, 1, 2, 3], [0.5, 0.5, 1, 1, 1])
                for note_i, (tick_off, length) in enumerate(hook_pattern):
                    if random.random() < max(0.01, cfg.lead_rest_prob * 0.2):
                        continue
                    idx = max(0, min(len(lead_scale) - 1, anchor_idx + offsets[note_i]))
                    midi = lead_scale[idx]
                    vel = random.randint(*cfg.lead_vel) + 3
                    notes.append(NoteEvent(bar_start + tick_off, length, midi,
                                           max(30, min(127, vel))))
                    current_idx, prev_pitch = idx, midi
                continue

            # ── Victory hook: ascending triumphant fanfare ──
            if cfg.lead_style == "victory_fanfare":
                _vic_hook_variants = [
                    [[0, 2, 4, 7], [0, 4, 7, 9]],
                    [[0, 3, 5, 7], [0, 2, 5, 7]],
                    [[0, 4, 5, 7], [0, 2, 4, 5]],
                    [[0, 5, 7, 9], [0, 3, 7, 5]],
                    [[0, 2, 7, 5], [0, 4, 2, 7]],
                ]
                motifs = _vic_hook_variants[signature_variant % len(_vic_hook_variants)]
                offsets = motifs[phrase_pos % len(motifs)]
                hook_pattern = _simple_rhythm(tpb, bpb,
                    [0, 0.5, 1, 2], [0.5, 0.5, 1, 2])
                for note_i, (tick_off, length) in enumerate(hook_pattern):
                    idx = max(0, min(len(lead_scale) - 1, anchor_idx + offsets[note_i]))
                    midi = lead_scale[idx]
                    vel = random.randint(*cfg.lead_vel) + 6
                    notes.append(NoteEvent(bar_start + tick_off, length, midi,
                                           max(30, min(127, vel))))
                    current_idx, prev_pitch = idx, midi
                continue

            # ── Dungeon hook: sparse, dark, angular ──
            if cfg.lead_style == "dungeon_dark":
                _dng_hook_variants = [
                    [[0, -1, -3], [0, -2, 0]],
                    [[0, 0, -2], [0, -3, -1]],
                    [[0, -2, 1], [0, -1, -3]],
                    [[0, 1, -2], [0, -3, 0]],
                    [[0, -1, 0], [0, -2, -1]],
                ]
                motifs = _dng_hook_variants[signature_variant % len(_dng_hook_variants)]
                offsets = motifs[phrase_pos % len(motifs)]
                hook_pattern = _simple_rhythm(tpb, bpb,
                    [0, 2, 3.5], [2, 1.5, 0.5])
                for note_i, (tick_off, length) in enumerate(hook_pattern):
                    if random.random() < 0.15:
                        continue
                    idx = max(0, min(len(lead_scale) - 1, anchor_idx + offsets[note_i]))
                    midi = lead_scale[idx]
                    vel = random.randint(*cfg.lead_vel) - 5
                    notes.append(NoteEvent(bar_start + tick_off, length, midi,
                                           max(30, min(127, vel))))
                    current_idx, prev_pitch = idx, midi
                continue

        # ── Lead-style direction bias ──
        step_bias = 0
        if cfg.lead_style == "arpeggio_peaks":
            # Every other bar: jump to upper register, then cascade down
            if bar_idx % 2 == 0:
                current_idx = min(len(lead_scale) - 1,
                                  int(len(lead_scale) * 0.80)
                                  + random.randint(-2, 2))
            step_bias = -1  # cascade downward
        elif cfg.lead_style == "question_answer":
            # 4-bar phrases: bars 0-1 ascend (question), bars 2-3 descend (answer)
            phrase_bar = bar_idx % 4
            if phrase_bar == 0:
                current_idx = max(0, int(len(lead_scale) * 0.3)
                                  + random.randint(-2, 2))
            step_bias = 1 if phrase_bar < 2 else -1
        elif cfg.lead_style == "gba_town":
            # ── GBA Town: section-aware melody (adapts to bar count) ─
            sec = _town_sections(cfg.bars)

            # Silent bars: intro or tail
            if bar_idx < sec["intro_end"] or bar_idx >= sec["silent_tail"]:
                continue

            # Determine section & register target
            if bar_idx < sec["head_end"]:                 # Head
                register_target = 0.40
                step_bias = 0
            elif bar_idx < sec["dev_end"]:                # Development
                register_target = 0.65
                step_bias = 1                             # ascending bias
            elif bar_idx < sec["climax_end"]:             # Climax
                register_target = 0.80
                half = (bar_idx - sec["dev_end"]) < (sec["climax_end"] - sec["dev_end"]) // 2
                step_bias = 1 if half else -1
            else:                                         # Reset
                register_target = 0.30
                step_bias = -1

            # Jump register at section boundaries
            boundaries = {sec["intro_end"], sec["head_end"],
                          sec["dev_end"], sec["climax_end"]}
            if bar_idx in boundaries:
                target_idx = int(len(lead_scale) * register_target)
                current_idx = max(0, min(len(lead_scale) - 1,
                                         target_idx + random.randint(-2, 2)))

            # ── "Gap" rule: breathing bars (every other bar in section)
            if bar_idx < sec["head_end"]:
                section_start = sec["intro_end"]
            elif bar_idx < sec["dev_end"]:
                section_start = sec["head_end"]
            elif bar_idx < sec["climax_end"]:
                section_start = sec["dev_end"]
            else:
                section_start = sec["climax_end"]
            if (bar_idx - section_start) % 2 == 1:
                keep = random.randint(1, 2)
                pattern = pattern[:keep]

        elif cfg.lead_style == "nes_town":
            # ── NES Town: pentatonic-biased, short repeating phrases ──
            #
            # NES Town character: pentatonic, short repeating phrases
            # - 70% pentatonic snap (singable, simple)
            # - 2-bar phrase repetition with ±1 step variation
            # - Small steps only (max 2), occasional 3rd/4th jump
            # - No section-awareness — just loops naturally
            # - Flat dynamics (NES had limited volume control)

            # Build pentatonic subset of lead_scale
            penta_intervals = {0, 2, 4, 7, 9}  # major pentatonic degrees
            penta_scale = [i for i, n in enumerate(lead_scale)
                           if n % 12 in {(root + iv) % 12
                                          for iv in penta_intervals}]
            if not penta_scale:
                penta_scale = list(range(len(lead_scale)))

            # 2-bar phrase repetition: odd bars echo even bars with ±1 step
            if bar_idx % 2 == 1 and bar_idx > 0:
                # small deviation from previous bar's register
                current_idx = max(0, min(len(lead_scale) - 1,
                                         current_idx + random.choice([-1, 0, 1])))
            step_bias = 0

            # Pentatonic snap: 70% of the time, constrain to pentatonic
            if random.random() < 0.70 and penta_scale:
                # Override pattern to use only pentatonic-safe notes
                closest_penta = min(penta_scale,
                                    key=lambda i: abs(i - current_idx))
                current_idx = closest_penta

        elif cfg.lead_style == "snes_town":
            # ── SNES Town: legato, Lydian touches, wide phrases ──────
            #
            # SNES Town character: legato, Lydian touches, wide phrases
            # - Section-aware (like GBA) but with longer phrases
            # - Wider intervals allowed (up to 5 scale steps)
            # - Lydian #4 colour tone occasionally inserted
            # - Fewer rests, more flowing legato
            # - Expressive dynamics: crescendo toward section peaks
            sec = _town_sections(cfg.bars)

            if bar_idx < sec["intro_end"] or bar_idx >= sec["silent_tail"]:
                continue

            # Register arc: lower start, gradual rise to climax, gentle fall
            total = max(1, cfg.bars)
            progress = bar_idx / total  # 0.0 → 1.0
            if progress < 0.3:
                register_target = 0.35 + progress * 0.5       # gentle rise
                step_bias = 1
            elif progress < 0.7:
                register_target = 0.55 + (progress - 0.3) * 0.625  # up to 0.80
                step_bias = random.choice([0, 1])
            else:
                register_target = 0.80 - (progress - 0.7) * 1.5    # fall back
                step_bias = -1

            # Smooth register evolution (no abrupt jumps, just gentle guidance)
            target_idx = int(len(lead_scale) * register_target)
            if abs(current_idx - target_idx) > 4:
                current_idx += 1 if current_idx < target_idx else -1

            # SNES uses full pattern (no gap rule — legato!)
            # But elongate some notes for sustained feel
            if random.random() < 0.3 and len(pattern) > 2:
                pattern = pattern[:len(pattern) - 1]  # drop last note, let previous ring

        elif cfg.lead_style in {"cave_generic", "cave_gba", "cave_snes"}:
            # ── Cave lead: dark, sparse, atmospheric ─────────────────
            #
            # Caves are the opposite of towns: few notes, lots of space,
            # lower register, minor-biased, echoey feel.
            # Section-aware with longer intros/drones.
            sec = _cave_sections(cfg.bars)

            # Silent bars: intro drone or tail
            if bar_idx < sec["intro_end"] or bar_idx >= sec["silent_tail"]:
                continue

            # Low register, slowly rising in development, falling back
            total = max(1, cfg.bars)
            progress = bar_idx / total
            if progress < 0.25:
                register_target = 0.25 + progress * 0.4
                step_bias = 0
            elif progress < 0.6:
                register_target = 0.35 + (progress - 0.25) * 0.7
                step_bias = random.choice([0, 0, 1])
            else:
                register_target = 0.60 - (progress - 0.6) * 1.2
                step_bias = random.choice([-1, -1, 0])

            target_idx = int(len(lead_scale) * register_target)
            if abs(current_idx - target_idx) > 3:
                current_idx += 1 if current_idx < target_idx else -1

            # Caves have lots of silence — thin out the pattern
            if random.random() < 0.35 and len(pattern) > 1:
                pattern = random.sample(pattern, k=max(1, len(pattern) // 2))
                pattern.sort(key=lambda p: p[0])

            # GBA cave: slightly more rhythmic, nervous energy
            if cfg.lead_style == "cave_gba":
                if random.random() < 0.2 and len(pattern) > 1:
                    # Occasional stutter: double a note
                    pattern = pattern + [pattern[0]]
                    pattern.sort(key=lambda p: p[0])

            # SNES cave: elongate notes for reverb-soaked feel
            if cfg.lead_style == "cave_snes":
                if random.random() < 0.4 and len(pattern) > 1:
                    pattern = pattern[:max(1, len(pattern) - 1)]

        # ── New genre direction biases ──
        elif cfg.lead_style == "hm_pastoral":
            # Gentle, narrow, section-aware with pastoral sections
            sec = (cfg.section_builder(cfg.bars) if cfg.section_builder
                   else _town_sections(cfg.bars))
            if bar_idx < sec.get("intro_end", 0) or bar_idx >= sec.get("silent_tail", cfg.bars):
                continue
            total = max(1, cfg.bars)
            progress = bar_idx / total
            if progress < 0.3:
                step_bias = 0
            elif progress < 0.65:
                step_bias = random.choice([0, 1])
            else:
                step_bias = random.choice([-1, 0])
            # Very gentle register arc
            target = 0.35 + 0.3 * (1 - abs(progress - 0.5) * 2)
            target_idx = int(len(lead_scale) * target)
            if abs(current_idx - target_idx) > 3:
                current_idx += 1 if current_idx < target_idx else -1
            # Occasional breathing
            if random.random() < 0.20 and len(pattern) > 2:
                pattern = pattern[:len(pattern) - 1]

        elif cfg.lead_style == "zelda_folk":
            # Bouncy, section-aware, wider intervals welcome
            sec = (cfg.section_builder(cfg.bars) if cfg.section_builder
                   else _town_sections(cfg.bars))
            if bar_idx < sec.get("intro_end", 0) or bar_idx >= sec.get("silent_tail", cfg.bars):
                continue
            if bar_idx < sec.get("head_end", 8):
                step_bias = 0
            elif bar_idx < sec.get("dev_end", 16):
                step_bias = 1
            elif bar_idx < sec.get("climax_end", 28):
                half = (bar_idx - sec.get("dev_end", 16)) < (sec.get("climax_end", 28) - sec.get("dev_end", 16)) // 2
                step_bias = 1 if half else -1
            else:
                step_bias = -1
            boundaries = {sec.get("intro_end", 0), sec.get("head_end", 8),
                          sec.get("dev_end", 16), sec.get("climax_end", 28)}
            if bar_idx in boundaries:
                target_idx = int(len(lead_scale) * 0.50)
                current_idx = max(0, min(len(lead_scale) - 1,
                                         target_idx + random.randint(-3, 3)))
            if (bar_idx - sec.get("intro_end", 0)) % 2 == 1:
                keep = random.randint(2, 3)
                pattern = pattern[:keep]

        elif cfg.lead_style == "dungeon_dark":
            # Sparse, dark, atmospheric — like cave but more structured
            sec = (cfg.section_builder(cfg.bars) if cfg.section_builder
                   else _town_sections(cfg.bars))
            if bar_idx < sec.get("intro_end", 0) or bar_idx >= sec.get("silent_tail", cfg.bars):
                continue
            total = max(1, cfg.bars)
            progress = bar_idx / total
            if progress < 0.3:
                step_bias = 0
            elif progress < 0.6:
                step_bias = random.choice([0, 0, 1])
            else:
                step_bias = random.choice([-1, -1, 0])
            register_target = 0.30 + 0.2 * (1 - abs(progress - 0.5) * 2)
            target_idx = int(len(lead_scale) * register_target)
            if abs(current_idx - target_idx) > 3:
                current_idx += 1 if current_idx < target_idx else -1
            # Very sparse
            if random.random() < 0.30 and len(pattern) > 1:
                pattern = random.sample(pattern, k=max(1, len(pattern) // 2))
                pattern.sort(key=lambda p: p[0])

        elif cfg.lead_style in {"combat_intense", "encounter_alarm"}:
            # Intense, fast, no breathing — always driving forward
            sec = (cfg.section_builder(cfg.bars) if cfg.section_builder
                   else _town_sections(cfg.bars))
            total = max(1, cfg.bars)
            progress = bar_idx / total
            # Constantly rising register toward climax
            register_target = 0.40 + progress * 0.45
            target_idx = int(len(lead_scale) * min(0.85, register_target))
            if abs(current_idx - target_idx) > 4:
                current_idx += 1 if current_idx < target_idx else -1
            step_bias = random.choice([0, 1, 1])
            # Encounters: even faster, no thinning
            if cfg.lead_style == "encounter_alarm":
                step_bias = 1  # always ascending urgency

        elif cfg.lead_style == "boss_epic":
            # Epic, wide leaps, dramatic dynamics
            sec = (cfg.section_builder(cfg.bars) if cfg.section_builder
                   else _town_sections(cfg.bars))
            if bar_idx < sec.get("intro_end", 0):
                continue
            total = max(1, cfg.bars)
            progress = bar_idx / total
            if progress < 0.25:
                register_target = 0.35
                step_bias = 0
            elif progress < 0.6:
                register_target = 0.55 + (progress - 0.25) * 0.7
                step_bias = random.choice([0, 1, 1])
            else:
                register_target = 0.80 - (progress - 0.6) * 0.8
                step_bias = random.choice([-1, 0])
            target_idx = int(len(lead_scale) * register_target)
            if abs(current_idx - target_idx) > 5:
                current_idx += 2 if current_idx < target_idx else -2
            elif abs(current_idx - target_idx) > 2:
                current_idx += 1 if current_idx < target_idx else -1

        elif cfg.lead_style == "overworld_heroic":
            # Confident, march-like, section-aware
            sec = (cfg.section_builder(cfg.bars) if cfg.section_builder
                   else _town_sections(cfg.bars))
            if bar_idx < sec.get("intro_end", 0) or bar_idx >= sec.get("silent_tail", cfg.bars):
                continue
            if bar_idx < sec.get("head_end", 8):
                step_bias = 0
            elif bar_idx < sec.get("dev_end", 16):
                step_bias = 1
            elif bar_idx < sec.get("climax_end", 28):
                half = (bar_idx - sec.get("dev_end", 16)) < (sec.get("climax_end", 28) - sec.get("dev_end", 16)) // 2
                step_bias = 1 if half else -1
            else:
                step_bias = -1
            boundaries = {sec.get("intro_end", 0), sec.get("head_end", 8),
                          sec.get("dev_end", 16), sec.get("climax_end", 28)}
            if bar_idx in boundaries:
                target_idx = int(len(lead_scale) * 0.50)
                current_idx = max(0, min(len(lead_scale) - 1,
                                         target_idx + random.randint(-2, 2)))

        elif cfg.lead_style == "victory_fanfare":
            # Short, ascending, triumphant — always rising
            step_bias = 1
            progress = bar_idx / max(1, cfg.bars)
            register_target = 0.40 + progress * 0.45
            target_idx = int(len(lead_scale) * min(0.85, register_target))
            if bar_idx == 0:
                current_idx = int(len(lead_scale) * 0.30)
            elif abs(current_idx - target_idx) > 3:
                current_idx += 1 if current_idx < target_idx else -1

        # Get chord tones for guidance
        idx_in_scale = degree % len(intervals)
        chord_root_semitone = root + intervals[idx_in_scale]
        chord_tones = set()
        for ct_iv in _CHORD_TYPES.get(_quality, [0, 4, 7]):
            chord_tones.add((chord_root_semitone + ct_iv) % 12)

        # Non-chorus motif identity per style/bar (chorus handled above)
        bar_offsets: list[int] | None = None
        bar_anchor_idx: int | None = None
        if not (chorus_start <= bar_idx < chorus_end):
            phrase_pos4 = bar_idx % 4
            is_call_bar = phrase_pos4 in {0, 1}
            response_mode = not is_call_bar

            if is_call_bar:
                if cfg.lead_style == "nes_town":
                    motif_idx = (motif_seed + (bar_idx // 2)) % len(nes_motif_bank)
                    bar_offsets = nes_motif_bank[motif_idx]
                elif cfg.lead_style == "gba_town":
                    motif_idx = (motif_seed + (bar_idx // 2) * 3) % len(gba_motif_bank)
                    bar_offsets = gba_motif_bank[motif_idx]
                elif cfg.lead_style == "snes_town":
                    motif_idx = (motif_seed + (bar_idx // 2) * 5) % len(snes_motif_bank)
                    bar_offsets = snes_motif_bank[motif_idx]
                elif cfg.lead_style in {"cave_generic", "cave_gba", "cave_snes"}:
                    motif_idx = (motif_seed + (bar_idx // 2) * 7) % len(cave_motif_bank)
                    bar_offsets = cave_motif_bank[motif_idx]
                elif cfg.lead_style == "hm_pastoral":
                    motif_idx = (motif_seed + (bar_idx // 2)) % len(hm_motif_bank)
                    bar_offsets = hm_motif_bank[motif_idx]
                elif cfg.lead_style == "zelda_folk":
                    motif_idx = (motif_seed + (bar_idx // 2) * 3) % len(zelda_motif_bank)
                    bar_offsets = zelda_motif_bank[motif_idx]
                elif cfg.lead_style == "dungeon_dark":
                    motif_idx = (motif_seed + (bar_idx // 2) * 7) % len(dungeon_motif_bank)
                    bar_offsets = dungeon_motif_bank[motif_idx]
                elif cfg.lead_style == "combat_intense":
                    motif_idx = (motif_seed + (bar_idx // 2) * 11) % len(combat_motif_bank)
                    bar_offsets = combat_motif_bank[motif_idx]
                elif cfg.lead_style == "encounter_alarm":
                    motif_idx = (motif_seed + (bar_idx // 2) * 13) % len(encounter_motif_bank)
                    bar_offsets = encounter_motif_bank[motif_idx]
                elif cfg.lead_style == "boss_epic":
                    motif_idx = (motif_seed + (bar_idx // 2) * 7) % len(boss_motif_bank)
                    bar_offsets = boss_motif_bank[motif_idx]
                elif cfg.lead_style == "overworld_heroic":
                    motif_idx = (motif_seed + (bar_idx // 2) * 5) % len(overworld_motif_bank)
                    bar_offsets = overworld_motif_bank[motif_idx]
                elif cfg.lead_style == "victory_fanfare":
                    motif_idx = (motif_seed + (bar_idx // 2) * 3) % len(victory_motif_bank)
                    bar_offsets = victory_motif_bank[motif_idx]
                if bar_offsets:
                    call_phrase_offsets = list(bar_offsets)
            else:
                # Answer phrase: close variation of the previous call.
                if call_phrase_offsets:
                    bar_offsets = [off + random.choice([-1, 0, 1]) for off in call_phrase_offsets]
                elif cfg.lead_style == "nes_town":
                    motif_idx = (motif_seed + (bar_idx // 2)) % len(nes_motif_bank)
                    bar_offsets = nes_motif_bank[motif_idx]
                elif cfg.lead_style == "gba_town":
                    motif_idx = (motif_seed + (bar_idx // 2) * 3) % len(gba_motif_bank)
                    bar_offsets = gba_motif_bank[motif_idx]
                elif cfg.lead_style == "snes_town":
                    motif_idx = (motif_seed + (bar_idx // 2) * 5) % len(snes_motif_bank)
                    bar_offsets = snes_motif_bank[motif_idx]
                elif cfg.lead_style in {"cave_generic", "cave_gba", "cave_snes"}:
                    motif_idx = (motif_seed + (bar_idx // 2) * 7) % len(cave_motif_bank)
                    bar_offsets = cave_motif_bank[motif_idx]
                elif cfg.lead_style == "hm_pastoral":
                    motif_idx = (motif_seed + (bar_idx // 2)) % len(hm_motif_bank)
                    bar_offsets = hm_motif_bank[motif_idx]
                elif cfg.lead_style == "zelda_folk":
                    motif_idx = (motif_seed + (bar_idx // 2) * 3) % len(zelda_motif_bank)
                    bar_offsets = zelda_motif_bank[motif_idx]
                elif cfg.lead_style == "dungeon_dark":
                    motif_idx = (motif_seed + (bar_idx // 2) * 7) % len(dungeon_motif_bank)
                    bar_offsets = dungeon_motif_bank[motif_idx]
                elif cfg.lead_style == "combat_intense":
                    motif_idx = (motif_seed + (bar_idx // 2) * 11) % len(combat_motif_bank)
                    bar_offsets = combat_motif_bank[motif_idx]
                elif cfg.lead_style == "encounter_alarm":
                    motif_idx = (motif_seed + (bar_idx // 2) * 13) % len(encounter_motif_bank)
                    bar_offsets = encounter_motif_bank[motif_idx]
                elif cfg.lead_style == "boss_epic":
                    motif_idx = (motif_seed + (bar_idx // 2) * 7) % len(boss_motif_bank)
                    bar_offsets = boss_motif_bank[motif_idx]
                elif cfg.lead_style == "overworld_heroic":
                    motif_idx = (motif_seed + (bar_idx // 2) * 5) % len(overworld_motif_bank)
                    bar_offsets = overworld_motif_bank[motif_idx]
                elif cfg.lead_style == "victory_fanfare":
                    motif_idx = (motif_seed + (bar_idx // 2) * 3) % len(victory_motif_bank)
                    bar_offsets = victory_motif_bank[motif_idx]

            if bar_offsets:
                bar_anchor_idx = current_idx
                if cfg.lead_style == "gba_town":
                    bar_offsets = [off + random.choice([0, 0, 0, 1]) for off in bar_offsets]
                elif cfg.lead_style == "snes_town":
                    bar_offsets = [off + random.choice([0, 0, 1]) for off in bar_offsets]
                elif cfg.lead_style in {"cave_generic", "cave_gba", "cave_snes"}:
                    bar_offsets = [off + random.choice([0, 0, -1]) for off in bar_offsets]
                elif cfg.lead_style == "hm_pastoral":
                    bar_offsets = [off + random.choice([0, 0, 0, 1]) for off in bar_offsets]
                elif cfg.lead_style == "zelda_folk":
                    bar_offsets = [off + random.choice([0, 0, 1, -1]) for off in bar_offsets]
                elif cfg.lead_style == "dungeon_dark":
                    bar_offsets = [off + random.choice([0, 0, 0, -1]) for off in bar_offsets]
                elif cfg.lead_style in {"combat_intense", "encounter_alarm"}:
                    bar_offsets = [off + random.choice([-1, 0, 1, 1]) for off in bar_offsets]
                elif cfg.lead_style == "boss_epic":
                    bar_offsets = [off + random.choice([-1, 0, 0, 1]) for off in bar_offsets]
                elif cfg.lead_style == "overworld_heroic":
                    bar_offsets = [off + random.choice([0, 0, 1]) for off in bar_offsets]
                elif cfg.lead_style == "victory_fanfare":
                    bar_offsets = [off + random.choice([0, 1, 1]) for off in bar_offsets]

        for note_slot, (tick_off, length) in enumerate(pattern):
            # Decide rest
            rest_prob = cfg.lead_rest_prob + (0.10 if response_mode else 0.0)
            if random.random() < min(0.85, rest_prob):
                continue

            # Walk with bias toward chord tones
            max_step = cfg.lead_step_max
            if step_bias > 0:
                step = random.randint(0, max_step)
            elif step_bias < 0:
                step = random.randint(-max_step, 0)
            else:
                step = random.randint(-max_step, max_step)

            # Bias: if current note's pitch class is not a chord tone, try to step toward one
            candidate_idx = max(0, min(len(lead_scale) - 1, current_idx + step))
            candidate_note = lead_scale[candidate_idx]

            # Style motif override (dominant source of identity outside chorus)
            if bar_offsets and bar_anchor_idx is not None:
                phrase_pos = note_slot % len(bar_offsets)
                idx = bar_anchor_idx + bar_offsets[phrase_pos]
                # Humanized variation while preserving motif contour
                if cfg.lead_style == "nes_town" and random.random() < 0.25:
                    idx += random.choice([-1, 0, 1])
                if cfg.lead_style == "gba_town" and random.random() < 0.35:
                    idx += random.choice([-1, 0, 1])
                if cfg.lead_style == "snes_town" and random.random() < 0.30:
                    idx += random.choice([0, 1])
                if cfg.lead_style in {"cave_generic", "cave_gba", "cave_snes"} and random.random() < 0.20:
                    idx += random.choice([-1, 0, 0])
                if cfg.lead_style == "hm_pastoral" and random.random() < 0.25:
                    idx += random.choice([0, 0, 1])
                if cfg.lead_style == "zelda_folk" and random.random() < 0.30:
                    idx += random.choice([-1, 0, 1])
                if cfg.lead_style == "dungeon_dark" and random.random() < 0.15:
                    idx += random.choice([-1, 0, 0])
                if cfg.lead_style in {"combat_intense", "encounter_alarm"} and random.random() < 0.40:
                    idx += random.choice([-1, 0, 1, 2])
                if cfg.lead_style == "boss_epic" and random.random() < 0.35:
                    idx += random.choice([-2, -1, 0, 1, 2])
                if cfg.lead_style == "overworld_heroic" and random.random() < 0.28:
                    idx += random.choice([0, 1, 1])
                if cfg.lead_style == "victory_fanfare" and random.random() < 0.20:
                    idx += random.choice([0, 1])
                candidate_idx = max(0, min(len(lead_scale) - 1, idx))
                candidate_note = lead_scale[candidate_idx]

            # Chord-tone snap probability differs by style (avoids same contour)
            snap_prob = 0.4
            if cfg.lead_style == "nes_town":
                snap_prob = 0.48
            elif cfg.lead_style == "gba_town":
                snap_prob = 0.72
            elif cfg.lead_style == "snes_town":
                snap_prob = 0.52
            elif cfg.lead_style in {"cave_generic", "cave_gba", "cave_snes"}:
                snap_prob = 0.55  # dark chord tones (minor)
            elif cfg.lead_style == "hm_pastoral":
                snap_prob = 0.45
            elif cfg.lead_style == "zelda_folk":
                snap_prob = 0.55
            elif cfg.lead_style == "dungeon_dark":
                snap_prob = 0.55
            elif cfg.lead_style in {"combat_intense", "encounter_alarm"}:
                snap_prob = 0.62
            elif cfg.lead_style == "boss_epic":
                snap_prob = 0.58
            elif cfg.lead_style == "overworld_heroic":
                snap_prob = 0.55
            elif cfg.lead_style == "victory_fanfare":
                snap_prob = 0.70

            # Extra bias: snap to nearest chord tone based on style
            if random.random() < snap_prob:
                # Find nearest lead_scale note whose pc is in chord_tones
                chord_scale = [i for i, n in enumerate(lead_scale)
                               if n % 12 in chord_tones]
                if chord_scale:
                    candidate_idx = min(chord_scale,
                                        key=lambda i: abs(i - current_idx))
                    candidate_note = lead_scale[candidate_idx]

            # ── NES Town: extra pentatonic bias on note selection ────
            if cfg.lead_style == "nes_town" and random.random() < 0.55:
                penta_iv = {0, 2, 4, 7, 9}
                penta_pcs = {(root + iv) % 12 for iv in penta_iv}
                penta_idx = [i for i, n in enumerate(lead_scale)
                             if n % 12 in penta_pcs]
                if penta_idx:
                    candidate_idx = min(penta_idx,
                                        key=lambda i: abs(i - candidate_idx))
                    candidate_note = lead_scale[candidate_idx]

            # ── SNES Town: occasional Lydian #4 colour tone ─────────
            if cfg.lead_style == "snes_town" and random.random() < 0.12:
                # Inject #4 (tritone above root) as a passing colour
                lydian_pc = (root + 6) % 12  # #4
                ly_idx = [i for i, n in enumerate(lead_scale)
                          if n % 12 == lydian_pc]
                if ly_idx:
                    candidate_idx = min(ly_idx,
                                        key=lambda i: abs(i - candidate_idx))
                    candidate_note = lead_scale[candidate_idx]

            # Keep GBA melodic color diatonic to avoid harsh accidental clashes.

            # Strong beats should land on chord tones most of the time.
            strong_beat = (tick_off % tpb == 0)
            if strong_beat and random.random() < 0.78:
                chord_scale = [i for i, n in enumerate(lead_scale)
                               if n % 12 in chord_tones]
                if chord_scale:
                    candidate_idx = min(chord_scale,
                                        key=lambda i: abs(i - candidate_idx))
                    candidate_note = lead_scale[candidate_idx]

            # Phrase cadence: prefer next bar's root/chord tone on final slot.
            if note_slot == len(pattern) - 1 and random.random() < 0.70:
                next_degree, _ = chord_prog[(bar_idx + 1) % len(chord_prog)]
                next_pc = (root + intervals[next_degree % len(intervals)]) % 12
                cadence_pcs = {next_pc, (next_pc + 4) % 12, (next_pc + 7) % 12}
                # Leading tone pull into tonic for stronger cadential motion.
                if next_pc == root and random.random() < 0.35:
                    cadence_pcs.add((root - 1) % 12)
                next_targets = [i for i, n in enumerate(lead_scale)
                                if n % 12 in cadence_pcs]
                if next_targets:
                    candidate_idx = min(next_targets,
                                        key=lambda i: abs(i - candidate_idx))
                    candidate_note = lead_scale[candidate_idx]

            # Avoid harsh leaps (> perfect fifth): smooth into stepwise motion.
            if last_note is not None and abs(candidate_note - last_note) > 7:
                step_dir = 1 if candidate_note > last_note else -1
                smoothed = _snap_to_scale(last_note + step_dir * random.choice([2, 3, 4]), lead_scale)
                candidate_idx = min(range(len(lead_scale)), key=lambda i: abs(lead_scale[i] - smoothed))
                candidate_note = lead_scale[candidate_idx]

            # Break long repeated runs so melody breathes and stays singable.
            if last_note is not None and candidate_note == last_note:
                repeat_run += 1
            else:
                repeat_run = 1
            if repeat_run >= 3:
                nidx = max(0, min(len(lead_scale) - 1,
                                  candidate_idx + random.choice([-1, 1])))
                candidate_idx = nidx
                candidate_note = lead_scale[candidate_idx]
                repeat_run = 1

            current_idx = candidate_idx
            vel = random.randint(*cfg.lead_vel)
            if response_mode:
                vel -= random.randint(4, 9)

            # ── GBA Town velocity ramps: pitch direction → velocity ──
            if cfg.lead_style == "gba_town" and prev_pitch is not None:
                if candidate_note > prev_pitch:
                    vel += random.randint(3, 5)      # ascending = crescendo
                elif candidate_note < prev_pitch:
                    vel -= random.randint(3, 5)      # descending = decrescendo
                vel += random.randint(-5, 5)         # humanize ±5

            # ── SNES Town dynamics: crescendo toward climax ──────────
            if cfg.lead_style == "snes_town":
                progress = bar_idx / max(1, cfg.bars)
                # Bell curve: loudest at 60-70% through
                dyn_boost = int(12 * (1 - abs(progress - 0.65) * 2.5))
                vel += max(-5, min(12, dyn_boost))
                vel += random.randint(-3, 3)         # humanize

            # ── Cave dynamics: generally quiet with occasional swells ──
            if cfg.lead_style in {"cave_generic", "cave_gba", "cave_snes"}:
                vel -= random.randint(5, 12)          # caves are quiet
                progress = bar_idx / max(1, cfg.bars)
                # Slight swell at 50-60% then fade
                if 0.45 < progress < 0.65:
                    vel += random.randint(2, 8)
                vel += random.randint(-4, 2)          # humanize (favor softer)

            # ── HM Pastoral dynamics: gentle & warm ──────────────────
            if cfg.lead_style == "hm_pastoral":
                vel -= random.randint(3, 8)
                vel += random.randint(-3, 3)

            # ── Zelda Folk dynamics: lively with slight swells ───────
            if cfg.lead_style == "zelda_folk":
                progress = bar_idx / max(1, cfg.bars)
                dyn = int(8 * (1 - abs(progress - 0.5) * 2.0))
                vel += max(-4, min(8, dyn))
                vel += random.randint(-3, 3)

            # ── Dungeon dynamics: dark and foreboding ────────────────
            if cfg.lead_style == "dungeon_dark":
                vel -= random.randint(6, 14)
                if bar_idx % 8 in {6, 7}:
                    vel += random.randint(3, 7)
                vel += random.randint(-3, 1)

            # ── Combat dynamics: aggressive with accent hits ─────────
            if cfg.lead_style == "combat_intense":
                vel += random.randint(2, 8)
                if note_slot == 0:
                    vel += random.randint(3, 6)  # accent downbeats
                vel += random.randint(-4, 4)

            # ── Encounter dynamics: urgent crescendo ─────────────────
            if cfg.lead_style == "encounter_alarm":
                progress = bar_idx / max(1, cfg.bars)
                vel += int(15 * progress)
                vel += random.randint(-3, 5)

            # ── Boss dynamics: dramatic swells on phrase boundaries ──
            if cfg.lead_style == "boss_epic":
                progress = bar_idx / max(1, cfg.bars)
                # Wave-like dynamics: two peaks
                wave = abs((progress * 4) % 2 - 1)
                vel += int(10 * wave)
                if note_slot == 0:
                    vel += random.randint(2, 5)
                vel += random.randint(-3, 4)

            # ── Overworld dynamics: confident, slightly loud ─────────
            if cfg.lead_style == "overworld_heroic":
                vel += random.randint(0, 5)
                progress = bar_idx / max(1, cfg.bars)
                dyn = int(8 * (1 - abs(progress - 0.6) * 2.5))
                vel += max(-3, min(8, dyn))
                vel += random.randint(-3, 3)

            # ── Victory dynamics: ascending triumphant crescendo ─────
            if cfg.lead_style == "victory_fanfare":
                progress = bar_idx / max(1, cfg.bars)
                vel += int(12 * progress)
                vel += random.randint(-2, 5)

            prev_pitch = candidate_note
            last_note = candidate_note

            notes.append(NoteEvent(
                start_tick=bar_start + tick_off,
                length_tick=length,
                midi_note=candidate_note,
                velocity=max(30, min(127, vel)),
            ))

    return notes


def _generate_bass(cfg: GenreConfig, scale: list[int],
                   chord_prog: list[tuple[int, str]],
                   intervals: list[int], root: int,
                   bar_ticks: int, tpb: int, bpb: int,
                   lead_notes: list[NoteEvent] | None = None) -> list[NoteEvent]:
    """Generate a bass line: root/fifth patterns following chord progression."""
    notes: list[NoteEvent] = []
    lo, hi = cfg.bass_range
    bass_scale = [s for s in scale if lo <= s <= hi]
    if not bass_scale:
        bass_scale = list(range(lo, hi + 1))

    rhythm_patterns = cfg.bass_rhythm or [
        _simple_rhythm(tpb, bpb, [0, 2], [2, 2])
    ]
    prev_bass_anchor: int | None = None

    def _lead_bar_trend(bar_index: int) -> int:
        """Return +1 if lead tends upward in this bar, -1 if downward, else 0."""
        if not lead_notes:
            return 0
        start = bar_index * bar_ticks
        end = start + bar_ticks
        bar_ns = [n for n in lead_notes if start <= n.start_tick < end]
        if len(bar_ns) < 2:
            return 0
        bar_ns.sort(key=lambda n: n.start_tick)
        delta = bar_ns[-1].midi_note - bar_ns[0].midi_note
        return 1 if delta > 0 else (-1 if delta < 0 else 0)

    for bar_idx, (degree, quality) in enumerate(chord_prog):
        bar_start = bar_idx * bar_ticks
        pattern = random.choice(rhythm_patterns)
        lead_trend = _lead_bar_trend(bar_idx)

        # Find the chord root in bass range
        idx_in_scale = degree % len(intervals)
        chord_root_pc = (root + intervals[idx_in_scale]) % 12

        # Find closest bass scale note matching chord_root_pc
        root_candidates = [n for n in bass_scale if n % 12 == chord_root_pc]
        if not root_candidates:
            root_candidates = [_snap_to_scale((lo + hi) // 2, bass_scale)]
        if prev_bass_anchor is None:
            bass_root = min(root_candidates, key=lambda n: abs(n - ((lo + hi) // 2)))
        else:
            if lead_trend > 0:
                downward = [n for n in root_candidates if n <= prev_bass_anchor]
                bass_root = (max(downward) if downward
                             else min(root_candidates, key=lambda n: abs(n - prev_bass_anchor)))
            elif lead_trend < 0:
                upward = [n for n in root_candidates if n >= prev_bass_anchor]
                bass_root = (min(upward) if upward
                             else min(root_candidates, key=lambda n: abs(n - prev_bass_anchor)))
            else:
                bass_root = min(root_candidates, key=lambda n: abs(n - prev_bass_anchor))

        # Also get the fifth
        fifth_pc = (chord_root_pc + 7) % 12
        fifth_candidates = [n for n in bass_scale if n % 12 == fifth_pc]
        bass_fifth = (min(fifth_candidates, key=lambda n: abs(n - bass_root))
                      if fifth_candidates else bass_root)

        # ── Bass style overrides ──
        if cfg.bass_style == "waltz":
            # Beat 1: low root (2 beats), Beat 3: higher 3rd, Beat 4: 5th
            vel = random.randint(*cfg.bass_vel)
            notes.append(NoteEvent(bar_start, tpb * 2, bass_root,
                                   min(127, vel)))
            third_pc = (chord_root_pc + 4) % 12
            third_cands = [n for n in bass_scale
                           if n % 12 == third_pc and n >= bass_root]
            higher = (random.choice(third_cands)
                      if third_cands else bass_fifth)
            v2 = max(30, random.randint(*cfg.bass_vel) - 10)
            notes.append(NoteEvent(bar_start + tpb * 2, tpb, higher,
                                   min(127, v2)))
            v3 = max(30, random.randint(*cfg.bass_vel) - 10)
            last_midi = bass_fifth if random.random() < 0.5 else higher
            notes.append(NoteEvent(
                bar_start + tpb * 3, tpb,
                last_midi,
                min(127, v3)))
            prev_bass_anchor = last_midi
            continue

        if cfg.bass_style == "walking":
            # Step through scale notes toward next chord root
            next_deg = chord_prog[(bar_idx + 1) % len(chord_prog)][0]
            next_pc = (root + intervals[next_deg % len(intervals)]) % 12
            next_cands = [n for n in bass_scale if n % 12 == next_pc]
            next_root = (random.choice(next_cands)
                         if next_cands else bass_root)
            cur_bs = min(range(len(bass_scale)),
                         key=lambda i: abs(bass_scale[i] - bass_root))
            tgt_bs = min(range(len(bass_scale)),
                         key=lambda i: abs(bass_scale[i] - next_root))
            for beat in range(bpb):
                t = beat / bpb
                interp = int(cur_bs + (tgt_bs - cur_bs) * t)
                interp = max(0, min(len(bass_scale) - 1, interp))
                if beat > 0 and random.random() < 0.3:
                    interp = max(0, min(len(bass_scale) - 1,
                                        interp + random.choice([-1, 1])))
                vel = random.randint(*cfg.bass_vel)
                notes.append(NoteEvent(bar_start + beat * tpb, tpb,
                                       bass_scale[interp], min(127, vel)))
                prev_bass_anchor = bass_scale[interp]
            continue

        if cfg.bass_style == "simple_root":
            # ── NES Town bass: root note on beats 1 & 3 only ────────
            # Simple pump bass — just the chord root, nothing fancy.
            # This is how Game Boy bass worked: limited to root notes
            # because there was only one bass-capable channel.
            vel = random.randint(*cfg.bass_vel)
            notes.append(NoteEvent(bar_start, tpb * 2, bass_root,
                                   min(127, vel)))
            v2 = random.randint(*cfg.bass_vel)
            notes.append(NoteEvent(bar_start + tpb * 2, tpb * 2, bass_root,
                                   min(127, v2)))
            prev_bass_anchor = bass_root
            continue

        if cfg.bass_style == "root_fifth":
            # GBA Town bass: ONLY root & 5th, never chords.
            # Follows the rhythm pattern (typically 3+3+2) and strictly
            # alternates root ↔ fifth.  Section-aware for 32-bar Town:
            #   Intro (0-3): play but softer, Reset (28-31): drop at bar 30.
            is_intro = bar_idx < 4
            if bar_idx >= 30:
                continue                     # drop bass for last 2 bars

            toggle = lead_trend < 0          # contrary entry on phrase start
            for _note_idx, (tick_off, length) in enumerate(pattern):
                midi = bass_root if not toggle else bass_fifth
                toggle = not toggle
                vel = random.randint(*cfg.bass_vel)
                if is_intro:
                    vel = max(30, vel - 12)  # softer in intro
                notes.append(NoteEvent(
                    start_tick=bar_start + tick_off,
                    length_tick=length,
                    midi_note=midi,
                    velocity=max(30, min(127, vel)),
                ))
                prev_bass_anchor = midi
            continue

        if cfg.bass_style == "drone_pedal":
            # ── Cave bass: sustained drone on root, minimal movement ──
            vel = random.randint(*cfg.bass_vel)
            if random.random() < 0.70:
                notes.append(NoteEvent(bar_start, bar_ticks, bass_root,
                                       min(127, vel)))
            else:
                notes.append(NoteEvent(bar_start, bar_ticks // 2, bass_root,
                                       min(127, vel)))
                v2 = max(30, random.randint(*cfg.bass_vel) - 6)
                notes.append(NoteEvent(bar_start + bar_ticks // 2,
                                       bar_ticks // 2, bass_fifth,
                                       min(127, v2)))
            prev_bass_anchor = bass_root
            continue

        if cfg.bass_style == "gentle_arp":
            # ── HM Pastoral bass: gentle arpeggiated root-third-fifth ──
            third_pc = (chord_root_pc + (3 if quality in {"min", "min7", "dim"} else 4)) % 12
            third_cands = [n for n in bass_scale if n % 12 == third_pc]
            bass_third = (min(third_cands, key=lambda n: abs(n - bass_root))
                          if third_cands else bass_root)
            arp_notes = [bass_root, bass_third, bass_fifth]
            beat_len = bar_ticks // max(len(arp_notes), 1)
            for i, mn in enumerate(arp_notes):
                vel = max(30, random.randint(*cfg.bass_vel) - random.randint(0, 6))
                notes.append(NoteEvent(bar_start + i * beat_len, beat_len, mn,
                                       min(127, vel)))
            prev_bass_anchor = bass_fifth
            continue

        if cfg.bass_style == "folk_bounce":
            # ── Zelda Folk bass: root-fifth bounce on beats ──
            toggle = False
            for beat in range(bpb):
                midi = bass_root if not toggle else bass_fifth
                toggle = not toggle
                vel = random.randint(*cfg.bass_vel)
                if beat == 0:
                    vel += random.randint(2, 5)
                notes.append(NoteEvent(bar_start + beat * tpb, tpb, midi,
                                       min(127, vel)))
                prev_bass_anchor = midi
            continue

        if cfg.bass_style == "driving_eighth":
            # ── Combat/Encounter bass: driving eighth notes on root ──
            eighth = tpb // 2 if tpb >= 2 else tpb
            num_eighths = bar_ticks // eighth
            for i in range(num_eighths):
                midi = bass_root if i % 2 == 0 else (bass_fifth if random.random() < 0.3 else bass_root)
                vel = random.randint(*cfg.bass_vel)
                if i % 2 == 0:
                    vel += random.randint(1, 4)
                notes.append(NoteEvent(bar_start + i * eighth, eighth, midi,
                                       min(127, vel)))
                prev_bass_anchor = midi
            continue

        if cfg.bass_style == "boss_heavy":
            # ── Boss bass: heavy syncopated root with power octave ──
            vel = random.randint(*cfg.bass_vel) + random.randint(3, 8)
            # Beat 1: root (half bar)
            notes.append(NoteEvent(bar_start, tpb * 2, bass_root,
                                   min(127, vel)))
            # Syncopated hit before beat 3
            syn_tick = bar_start + tpb * 2 + tpb // 2
            v2 = random.randint(*cfg.bass_vel) + random.randint(2, 6)
            oct_midi = bass_root + 12 if bass_root + 12 <= hi else bass_root
            notes.append(NoteEvent(syn_tick, tpb, oct_midi, min(127, v2)))
            # Beat 4: fifth
            v3 = random.randint(*cfg.bass_vel)
            notes.append(NoteEvent(bar_start + tpb * 3, tpb, bass_fifth,
                                   min(127, v3)))
            prev_bass_anchor = bass_fifth
            continue

        if cfg.bass_style == "march_bass":
            # ── Overworld bass: steady quarter-note march ──
            march_pattern = [bass_root, bass_fifth, bass_root, bass_fifth]
            for beat in range(min(bpb, len(march_pattern))):
                midi = march_pattern[beat]
                vel = random.randint(*cfg.bass_vel)
                if beat == 0:
                    vel += random.randint(3, 6)
                notes.append(NoteEvent(bar_start + beat * tpb, tpb, midi,
                                       min(127, vel)))
                prev_bass_anchor = midi
            continue

        if cfg.bass_style == "fanfare_bass":
            # ── Victory bass: ascending root-fifth-octave pattern ──
            oct_root = bass_root + 12 if bass_root + 12 <= hi else bass_root
            fanfare_seq = [bass_root, bass_fifth, oct_root]
            beat_len = bar_ticks // max(len(fanfare_seq), 1)
            progress = bar_idx / max(1, len(chord_prog))
            for i, mn in enumerate(fanfare_seq):
                vel = random.randint(*cfg.bass_vel) + int(8 * progress)
                notes.append(NoteEvent(bar_start + i * beat_len, beat_len, mn,
                                       min(127, vel)))
            prev_bass_anchor = oct_root
            continue

        for note_idx, (tick_off, length) in enumerate(pattern):
            # Alternate root and fifth with some randomness
            if note_idx == 0 or random.random() < 0.6:
                midi = bass_root
            else:
                midi = bass_fifth if random.random() < 0.7 else bass_root

            # Occasional octave variation
            if random.random() < 0.15 and lo <= midi + 12 <= hi:
                midi += 12

            # Contrary-motion nudge against current lead contour.
            if prev_bass_anchor is not None:
                if lead_trend > 0 and midi > prev_bass_anchor and lo <= midi - 12 <= hi:
                    midi -= 12
                elif lead_trend < 0 and midi < prev_bass_anchor and lo <= midi + 12 <= hi:
                    midi += 12

            vel = random.randint(*cfg.bass_vel)
            notes.append(NoteEvent(
                start_tick=bar_start + tick_off,
                length_tick=length,
                midi_note=midi,
                velocity=min(127, vel),
            ))
            prev_bass_anchor = midi

    return notes


def _generate_harmony(cfg: GenreConfig, scale: list[int],
                      chord_prog: list[tuple[int, str]],
                      intervals: list[int], root: int,
                      bar_ticks: int, tpb: int, bpb: int) -> list[NoteEvent]:
    """Generate harmony: chord voicings, one per bar (or broken chords)."""
    notes: list[NoteEvent] = []
    lo, hi = cfg.harmony_range

    # Choose style: block chords or arpeggiated (unless overridden)
    use_arpeggio = random.random() < 0.4
    prev_voicing: list[int] | None = None

    harmony_rhythm = cfg.harmony_rhythm or []

    for bar_idx, (degree, quality) in enumerate(chord_prog):
        bar_start = bar_idx * bar_ticks

        # Chord root in harmony range
        idx_in_scale = degree % len(intervals)
        chord_root_pc = (root + intervals[idx_in_scale]) % 12

        # Find octave for root in range, biased toward warm midrange (F#3..C5)
        candidates = [
            chord_root_pc + octave * 12
            for octave in range(11)
            if lo <= chord_root_pc + octave * 12 <= hi
        ]
        if candidates:
            target_center = max(lo, min(hi, 63))  # ~D#4, center of F#3..C5 band
            chord_root = min(candidates, key=lambda n: abs(n - target_center))
        else:
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

        # Voice-leading: keep harmony notes near previous bar's voicing.
        if prev_voicing:
            sorted_prev = sorted(prev_voicing)
            led: list[int] = []
            for i, note in enumerate(sorted(voicing)):
                ref = sorted_prev[min(i, len(sorted_prev) - 1)]
                candidates = [n for n in (note - 12, note, note + 12)
                              if lo <= n <= hi]
                if not candidates:
                    candidates = [max(lo, min(hi, note))]
                led.append(min(candidates, key=lambda n: abs(n - ref)))
            voicing = sorted(led)

        vel = random.randint(*cfg.harmony_vel)

        # ── Staccato harmony (3+3+2 off-beat chords for GBA Town) ──
        if cfg.harmony_style == "staccato" and harmony_rhythm:
            # Section-aware for 32-bar Town:
            #   Reset (bars 28-31): drop harmony at bar 30
            if bar_idx >= 30:
                continue
            is_intro = bar_idx < 4

            # Occasional rapid arpeggio shimmer (hardware chord-cheat feel).
            if random.random() < 0.24:
                arp_voicing = voicing[:3] if len(voicing) >= 3 else list(voicing)
                if arp_voicing:
                    arp_dur = max(1, tpb // 4)
                    tick = 0
                    idx = 0
                    while tick < bar_ticks:
                        if random.random() < 0.12:
                            idx += 1
                            tick += arp_dur
                            continue
                        midi = arp_voicing[idx % len(arp_voicing)]
                        v = vel + random.randint(-6, 4)
                        if is_intro:
                            v = max(30, v - 10)
                        notes.append(NoteEvent(
                            start_tick=bar_start + tick,
                            length_tick=min(arp_dur, bar_ticks - tick),
                            midi_note=midi,
                            velocity=max(30, min(127, v)),
                        ))
                        idx += 1
                        tick += arp_dur
                    prev_voicing = arp_voicing
                    continue

            pattern = random.choice(harmony_rhythm)
            for tick_off, length in pattern:
                v = vel + random.randint(-5, 5)
                if is_intro:
                    v = max(30, v - 10)  # softer during intro
                # Play short staccato chord stabs
                for midi in voicing:
                    notes.append(NoteEvent(
                        start_tick=bar_start + tick_off,
                        length_tick=length,
                        midi_note=midi,
                        velocity=max(30, min(127, v)),
                    ))
            prev_voicing = voicing
            continue

        # ── NES Chip arpeggio (rapid cycling through chord tones) ──
        if cfg.harmony_style == "arpeggio_chip":
            # Simulates how NES/chiptune plays "chords": rapidly cycling
            # through root, 3rd, 5th as 16th-note arpeggios.
            # Only use 3 notes at most (hardware constraint simulation).
            chip_voicing = voicing[:3]
            arp_dur = max(1, tpb // 2)  # 8th note (was 16th — less dense)
            tick = 0
            idx = 0
            while tick < bar_ticks:
                # 20% chance to skip a note (breathing room)
                if random.random() < 0.20:
                    idx += 1
                    tick += arp_dur
                    continue
                midi = chip_voicing[idx % len(chip_voicing)]
                v = vel + random.randint(-3, 3)
                notes.append(NoteEvent(
                    start_tick=bar_start + tick,
                    length_tick=min(arp_dur, bar_ticks - tick),
                    midi_note=midi,
                    velocity=max(30, min(127, v)),
                ))
                idx += 1
                tick += arp_dur
            prev_voicing = chip_voicing
            continue

        # ── SNES Sustained pad (held chords, sus/maj7 voicings) ────
        if cfg.harmony_style == "sustained_pad":
            # Long held voicings — root + one colour tone, sustained
            # for the entire bar. Creates warm orchestral backdrop.
            # Use only 2 notes for clarity (root + 3rd or root + 7th).
            pad_voicing = voicing[:2] if len(voicing) >= 2 else voicing
            v = vel + random.randint(-4, 4)
            for midi in pad_voicing:
                notes.append(NoteEvent(
                    start_tick=bar_start,
                    length_tick=bar_ticks,
                    midi_note=midi,
                    velocity=max(30, min(127, v)),
                ))
            prev_voicing = pad_voicing
            continue

        # ── Cave dark drone (sparse held minor chords, very quiet) ──
        if cfg.harmony_style == "dark_drone":
            # Single note or root+fifth only, sustained across the bar.
            # Occasionally skip bars entirely for tension.
            if random.random() < 0.25:
                # Silent bar — let the echo ring
                continue
            drone_voicing = [voicing[0]]
            if len(voicing) >= 2 and random.random() < 0.5:
                drone_voicing.append(voicing[1])
            v = max(30, vel + random.randint(-8, 2))
            for midi in drone_voicing:
                notes.append(NoteEvent(
                    start_tick=bar_start,
                    length_tick=bar_ticks,
                    midi_note=midi,
                    velocity=max(25, min(127, v)),
                ))
            prev_voicing = drone_voicing
            continue

        # ── HM/Zelda Folk strum: rhythmic chord on beats ─────────
        if cfg.harmony_style == "folk_strum" and harmony_rhythm:
            pattern = random.choice(harmony_rhythm)
            for tick_off, length in pattern:
                v = max(30, vel + random.randint(-6, 4))
                for midi in voicing:
                    notes.append(NoteEvent(
                        start_tick=bar_start + tick_off,
                        length_tick=max(1, length - random.randint(0, tpb // 4)),
                        midi_note=midi,
                        velocity=max(30, min(127, v)),
                    ))
            prev_voicing = list(voicing)
            continue

        # ── Combat power stab: short accented chords ─────────────
        if cfg.harmony_style == "power_stab" and harmony_rhythm:
            pattern = random.choice(harmony_rhythm)
            for tick_off, length in pattern:
                v = max(30, vel + random.randint(2, 10))
                stab_len = max(1, min(length, tpb // 2))
                for midi in voicing:
                    notes.append(NoteEvent(
                        start_tick=bar_start + tick_off,
                        length_tick=stab_len,
                        midi_note=midi,
                        velocity=max(30, min(127, v)),
                    ))
            prev_voicing = list(voicing)
            continue

        # ── Boss dramatic: wide voiced sustained with swells ─────
        if cfg.harmony_style == "boss_dramatic":
            progress = bar_idx / max(1, len(chord_prog))
            wave = abs((progress * 4) % 2 - 1)
            v = max(30, vel + int(8 * wave) + random.randint(-3, 3))
            for midi in voicing:
                notes.append(NoteEvent(
                    start_tick=bar_start,
                    length_tick=bar_ticks,
                    midi_note=midi,
                    velocity=max(30, min(127, v)),
                ))
            prev_voicing = list(voicing)
            continue

        # ── Overworld heroic hold: sustained confident chords ────
        if cfg.harmony_style == "heroic_hold":
            v = max(30, vel + random.randint(0, 6))
            for midi in voicing:
                notes.append(NoteEvent(
                    start_tick=bar_start,
                    length_tick=bar_ticks,
                    midi_note=midi,
                    velocity=max(30, min(127, v)),
                ))
            prev_voicing = list(voicing)
            continue

        # ── Victory fanfare chord: triumphant block chords ───────
        if cfg.harmony_style == "fanfare_chord":
            progress = bar_idx / max(1, len(chord_prog))
            v = max(30, vel + int(10 * progress) + random.randint(-2, 5))
            # Two hits per bar for fanfare feel
            half = bar_ticks // 2
            for midi in voicing:
                notes.append(NoteEvent(
                    start_tick=bar_start,
                    length_tick=half,
                    midi_note=midi,
                    velocity=max(30, min(127, v)),
                ))
            v2 = max(30, v + random.randint(1, 4))
            for midi in voicing:
                notes.append(NoteEvent(
                    start_tick=bar_start + half,
                    length_tick=half,
                    midi_note=midi,
                    velocity=max(30, min(127, v2)),
                ))
            prev_voicing = list(voicing)
            continue

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
            prev_voicing = voicing
        else:
            # Block chord: whole bar sustained
            for midi in voicing:
                notes.append(NoteEvent(
                    start_tick=bar_start,
                    length_tick=bar_ticks,
                    midi_note=midi,
                    velocity=min(127, vel),
                ))
            prev_voicing = voicing

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

        # ── NES simple tick: minimal hi-hat on quarter notes ─────────
        if cfg.drum_style == "simple_tick":
            # NES Town drums: almost nothing.
            # Just a quiet hi-hat tick on each quarter note.
            # Occasional light accent on beat 1.
            p_hihat = pitches[0]
            for beat in range(bpb):
                vel = random.randint(*cfg.drum_vel)
                if beat == 0:
                    vel = min(127, vel + 4)    # slight accent on beat 1
                # Skip some beats randomly (20%) for breathing
                if beat > 0 and random.random() < 0.20:
                    continue
                notes.append(NoteEvent(
                    start_tick=bar_start + beat * tpb,
                    length_tick=max(1, tpb // 4),
                    midi_note=p_hihat,
                    velocity=max(20, min(127, vel)),
                ))
            continue

        # ── SNES brush: gentle brush/ride pattern ────────────────────
        if cfg.drum_style == "snes_brush":
            # SNES Town drums: very soft, brush-like.
            # Quarter-note pulse with soft accent on 2 & 4.
            # No kick, no snare fills — just texture.
            p_hihat = pitches[0]
            for beat in range(bpb):
                vel = random.randint(*cfg.drum_vel)
                # Soft accent on 2 & 4 (like a brush swish)
                if beat == 1 or beat == 3:
                    vel = min(127, vel + 5)
                else:
                    vel = max(15, vel - 3)
                # Ghost note randomization
                if random.random() < 0.15:
                    vel = max(15, vel - 10)
                notes.append(NoteEvent(
                    start_tick=bar_start + beat * tpb,
                    length_tick=max(1, tpb // 4),
                    midi_note=p_hihat,
                    velocity=max(15, min(127, vel)),
                ))
            # Occasional very quiet uptick between beats (brush texture)
            if random.random() < 0.30:
                upbeat = random.choice([0, 1, 2, 3])
                uptick = bar_start + upbeat * tpb + tpb // 2
                notes.append(NoteEvent(
                    start_tick=uptick,
                    length_tick=max(1, tpb // 4),
                    midi_note=p_hihat,
                    velocity=max(15, min(127, random.randint(*cfg.drum_vel) - 8)),
                ))
            continue

        # ── Cave drip: very sparse, atmospheric percussion ───────────
        if cfg.drum_style == "cave_drip":
            # Simulates water drips and distant echoes.
            # Extremely sparse — many bars have no percussion at all.
            if random.random() < 0.40:
                # Silent bar (40% of bars have no drums)
                continue
            p_drip = pitches[0]
            # 1-3 drip sounds per bar at random positions
            num_drips = random.randint(1, 3)
            used_positions: set[int] = set()
            for _ in range(num_drips):
                beat_pos = random.uniform(0, bpb - 0.5)
                tick_pos = int(beat_pos * tpb)
                if tick_pos in used_positions:
                    continue
                used_positions.add(tick_pos)
                vel = random.randint(*cfg.drum_vel)
                vel = max(12, vel - random.randint(0, 8))
                notes.append(NoteEvent(
                    start_tick=bar_start + tick_pos,
                    length_tick=max(1, tpb // 4),
                    midi_note=p_drip,
                    velocity=max(12, min(127, vel)),
                ))
            continue

        # ── GBA Town drums: section-aware, snare + hi-hat only ──────
        if cfg.drum_style == "gba_town":
            sec = _town_sections(cfg.bars)

            # Drop drums in silent tail
            if bar_idx >= sec["silent_tail"]:
                continue
            is_intro = bar_idx < sec["intro_end"]
            is_reset = bar_idx >= sec["reset_start"]

            # In Town themes only 2 pitches: snare (idx 0) + hi-hat (idx 1)
            p_snare = pitches[0]
            p_hihat = pitches[1] if len(pitches) > 1 else pitches[0]

            for _note_idx, (tick_off, length) in enumerate(pattern):
                beat_pos = tick_off / tpb

                # Soft snare on beats 2 & 4 (backbeats)
                if abs(beat_pos - 1.0) < 0.01 or abs(beat_pos - 3.0) < 0.01:
                    midi = p_snare
                else:
                    # Everything else is hi-hat (closed, on 8th notes)
                    midi = p_hihat

                vel = random.randint(*cfg.drum_vel)
                if is_intro:
                    vel = max(25, vel - 15)       # very soft in intro
                if is_reset:
                    vel = max(25, vel - 8)        # tapering off

                # Ghost notes on upbeats
                if random.random() < 0.12:
                    vel = max(25, vel - 18)

                notes.append(NoteEvent(
                    start_tick=bar_start + tick_off,
                    length_tick=length,
                    midi_note=midi,
                    velocity=max(25, min(127, vel)),
                ))

            # Light snare fill 2 bars before reset
            fill_bars = {max(0, sec["reset_start"] - 1), max(0, sec["reset_start"])}
            if bar_idx in fill_bars and random.random() < 0.5:
                fill_tick = bar_start + bar_ticks - tpb
                for fi in range(random.randint(2, 3)):
                    offset = fi * max(1, tpb // 4)
                    if fill_tick + offset < bar_start + bar_ticks:
                        notes.append(NoteEvent(
                            start_tick=fill_tick + offset,
                            length_tick=max(1, tpb // 4),
                            midi_note=p_snare,
                            velocity=max(25, min(127,
                                         random.randint(*cfg.drum_vel))),
                        ))
            continue

        # ── Gentle brush: very soft quarter-note (HM Pastoral) ───────
        if cfg.drum_style == "gentle_brush":
            p_hihat = pitches[0]
            for beat in range(bpb):
                vel = max(15, random.randint(*cfg.drum_vel) - random.randint(5, 12))
                if beat > 0 and random.random() < 0.25:
                    continue  # sparse
                notes.append(NoteEvent(
                    start_tick=bar_start + beat * tpb,
                    length_tick=max(1, tpb // 4),
                    midi_note=p_hihat,
                    velocity=max(12, min(127, vel)),
                ))
            continue

        # ── Folk beat: steady kick + backbeat snare (Zelda) ──────────
        if cfg.drum_style == "folk_beat":
            p_kick = pitches[0]
            p_snare = pitches[1] if len(pitches) > 1 else pitches[0]
            p_hihat = pitches[2] if len(pitches) > 2 else pitches[0]
            for beat in range(bpb):
                vel = random.randint(*cfg.drum_vel)
                if beat == 0 or beat == 2:
                    # Kick on 1 and 3
                    notes.append(NoteEvent(bar_start + beat * tpb, max(1, tpb // 2),
                                           p_kick, min(127, vel + 2)))
                if beat == 1 or beat == 3:
                    # Snare on 2 and 4
                    notes.append(NoteEvent(bar_start + beat * tpb, max(1, tpb // 4),
                                           p_snare, min(127, vel)))
                # Hi-hat on every beat
                hv = max(20, vel - 8)
                notes.append(NoteEvent(bar_start + beat * tpb, max(1, tpb // 4),
                                       p_hihat, min(127, hv)))
            continue

        # ── Dungeon march: slow foreboding quarter pulse ─────────────
        if cfg.drum_style == "dungeon_march":
            p_low = pitches[0]
            for beat in range(bpb):
                if beat > 0 and random.random() < 0.35:
                    continue
                vel = max(15, random.randint(*cfg.drum_vel) - random.randint(3, 10))
                if beat == 0:
                    vel += random.randint(2, 5)
                notes.append(NoteEvent(
                    start_tick=bar_start + beat * tpb,
                    length_tick=max(1, tpb // 2),
                    midi_note=p_low,
                    velocity=max(15, min(127, vel)),
                ))
            continue

        # ── Combat drive: driving eighth-note percussion ─────────────
        if cfg.drum_style == "combat_drive":
            p_kick = pitches[0]
            p_snare = pitches[1] if len(pitches) > 1 else pitches[0]
            p_hihat = pitches[2] if len(pitches) > 2 else pitches[0]
            eighth = max(1, tpb // 2)
            num_eighths = bar_ticks // eighth
            for i in range(num_eighths):
                beat_f = i * 0.5
                vel = random.randint(*cfg.drum_vel) + random.randint(0, 5)
                if i % 4 == 0:
                    midi = p_kick
                    vel += 3
                elif i % 4 == 2:
                    midi = p_snare
                else:
                    midi = p_hihat
                    vel -= 5
                notes.append(NoteEvent(bar_start + i * eighth, eighth,
                                       midi, max(25, min(127, vel))))
            continue

        # ── Boss pound: heavy polyrhythmic hits ──────────────────────
        if cfg.drum_style == "boss_pound":
            p_kick = pitches[0]
            p_snare = pitches[1] if len(pitches) > 1 else pitches[0]
            p_accent = pitches[2] if len(pitches) > 2 else pitches[0]
            # Heavy kick on 1 and 3
            for beat in [0, 2]:
                if beat < bpb:
                    vel = random.randint(*cfg.drum_vel) + random.randint(5, 10)
                    notes.append(NoteEvent(bar_start + beat * tpb, tpb,
                                           p_kick, min(127, vel)))
            # Snare on 2 and 4 with accent
            for beat in [1, 3]:
                if beat < bpb:
                    vel = random.randint(*cfg.drum_vel) + random.randint(3, 8)
                    notes.append(NoteEvent(bar_start + beat * tpb, max(1, tpb // 2),
                                           p_snare, min(127, vel)))
            # Syncopated accent hits
            if random.random() < 0.6:
                syn = random.choice([1, 3, 5, 7])
                syn_tick = bar_start + syn * (tpb // 2)
                if syn_tick < bar_start + bar_ticks:
                    notes.append(NoteEvent(syn_tick, max(1, tpb // 4),
                                           p_accent,
                                           min(127, random.randint(*cfg.drum_vel) + 4)))
            continue

        # ── March steady: military march pattern (Overworld) ─────────
        if cfg.drum_style == "march_steady":
            p_kick = pitches[0]
            p_snare = pitches[1] if len(pitches) > 1 else pitches[0]
            for beat in range(bpb):
                vel = random.randint(*cfg.drum_vel)
                if beat % 2 == 0:
                    notes.append(NoteEvent(bar_start + beat * tpb, max(1, tpb // 2),
                                           p_kick, min(127, vel + 3)))
                else:
                    notes.append(NoteEvent(bar_start + beat * tpb, max(1, tpb // 4),
                                           p_snare, min(127, vel)))
            continue

        # ── Fanfare roll: snare roll + crash accents (Victory) ───────
        if cfg.drum_style == "fanfare_roll":
            p_snare = pitches[0]
            p_crash = pitches[1] if len(pitches) > 1 else pitches[0]
            progress = bar_idx / max(1, cfg.bars)
            # Crash on beat 1
            vel = random.randint(*cfg.drum_vel) + int(10 * progress)
            notes.append(NoteEvent(bar_start, max(1, tpb), p_crash,
                                   min(127, vel + 5)))
            # Snare roll (sixteenth notes)
            sixteenth = max(1, tpb // 4)
            num_16th = bar_ticks // sixteenth
            for i in range(1, num_16th):
                t = bar_start + i * sixteenth
                if t >= bar_start + bar_ticks:
                    break
                sv = max(20, random.randint(*cfg.drum_vel) - 5 + int(6 * progress))
                if random.random() < 0.15:
                    continue  # small gap for realism
                notes.append(NoteEvent(t, sixteenth, p_snare,
                                       min(127, sv)))
            continue

        # ── Default drum style ──────────────────────────────────────
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


# ── Extra-track generators (for flexible track count) ─────────────

def _generate_extra_track(extra: ExtraTrackConfig, cfg: GenreConfig,
                          scale: list[int],
                          chord_prog: list[tuple[int, str]],
                          intervals: list[int], root: int,
                          bar_ticks: int, tpb: int, bpb: int,
                          lead_notes: list[NoteEvent]) -> list[NoteEvent]:
    """Dispatch to the appropriate extra-track generator."""
    if extra.gen_type == "counter_melody":
        return _gen_counter_melody(extra, cfg, scale, chord_prog,
                                   intervals, root, bar_ticks, tpb, bpb,
                                   lead_notes)
    if extra.gen_type == "pad":
        return _gen_pad(extra, cfg, scale, chord_prog,
                        intervals, root, bar_ticks, tpb, bpb)
    if extra.gen_type == "arpeggio":
        return _gen_arpeggio(extra, cfg, scale, chord_prog,
                             intervals, root, bar_ticks, tpb, bpb)
    return []


def _gen_counter_melody(
    extra: ExtraTrackConfig, cfg: GenreConfig,
    scale: list[int], chord_prog: list[tuple[int, str]],
    intervals: list[int], root: int,
    bar_ticks: int, tpb: int, bpb: int,
    lead_notes: list[NoteEvent],
) -> list[NoteEvent]:
    """Counter-melody that weaves around the lead, filling gaps."""
    notes: list[NoteEvent] = []
    lo, hi = extra.pitch_range
    counter_scale = [s for s in scale if lo <= s <= hi]
    if not counter_scale:
        counter_scale = list(range(lo, hi + 1))
    current_idx = len(counter_scale) // 2

    # Build tick-level occupancy from lead
    lead_occupied: set[int] = set()
    for n in lead_notes:
        for t in range(n.start_tick, n.start_tick + n.length_tick):
            lead_occupied.add(t)

    rhythm_patterns = extra.rhythm or [
        _simple_rhythm(tpb, bpb, [0, 2], [1.5, 1.5]),
        _simple_rhythm(tpb, bpb, [1, 3], [1, 1]),
        _simple_rhythm(tpb, bpb, [0.5, 1.5, 2.5, 3.5],
                       [0.5, 0.5, 0.5, 0.5]),
    ]

    for bar_idx, (degree, _quality) in enumerate(chord_prog):
        bar_start = bar_idx * bar_ticks
        pattern = random.choice(rhythm_patterns)

        idx_in_scale = degree % len(intervals)
        chord_root_semi = root + intervals[idx_in_scale]
        chord_tones = {(chord_root_semi + iv) % 12
                       for iv in _CHORD_TYPES.get(_quality, [0, 4, 7])}

        for tick_off, length in pattern:
            abs_tick = bar_start + tick_off
            if abs_tick in lead_occupied and random.random() < 0.6:
                continue
            if random.random() < extra.rest_prob:
                continue

            step = random.randint(-extra.step_max, extra.step_max)
            candidate_idx = max(0, min(len(counter_scale) - 1,
                                       current_idx + step))
            candidate_note = counter_scale[candidate_idx]

            # Bias toward chord tones ~50 %
            if random.random() < 0.5:
                chord_idx = [i for i, n in enumerate(counter_scale)
                             if n % 12 in chord_tones]
                if chord_idx:
                    candidate_idx = min(chord_idx,
                                        key=lambda i: abs(i - current_idx))
                    candidate_note = counter_scale[candidate_idx]

            current_idx = candidate_idx
            vel = random.randint(*extra.vel)
            notes.append(NoteEvent(
                start_tick=abs_tick,
                length_tick=length,
                midi_note=candidate_note,
                velocity=min(127, vel),
            ))

    return notes


def _gen_pad(
    extra: ExtraTrackConfig, cfg: GenreConfig,
    scale: list[int], chord_prog: list[tuple[int, str]],
    intervals: list[int], root: int,
    bar_ticks: int, tpb: int, bpb: int,
) -> list[NoteEvent]:
    """Sustained pad voicings following chord progression."""
    notes: list[NoteEvent] = []
    lo, hi = extra.pitch_range

    for bar_idx, (degree, quality) in enumerate(chord_prog):
        bar_start = bar_idx * bar_ticks
        if random.random() < extra.rest_prob:
            continue

        idx_in_scale = degree % len(intervals)
        chord_root_pc = (root + intervals[idx_in_scale]) % 12

        chord_root = None
        for octave in range(11):
            candidate = chord_root_pc + octave * 12
            if lo <= candidate <= hi:
                chord_root = candidate
                break
        if chord_root is None:
            chord_root = (lo + hi) // 2

        # Two-note voicing (root + one colour tone)
        chord_ivs = _CHORD_TYPES.get(quality, [0, 4, 7])
        voicing: list[int] = []
        for iv in chord_ivs[:2]:
            note = chord_root + iv
            while note > hi and note > 12:
                note -= 12
            while note < lo:
                note += 12
            if lo <= note <= hi:
                voicing.append(note)
        if not voicing:
            voicing = [chord_root]

        vel = random.randint(*extra.vel)
        for midi in voicing:
            notes.append(NoteEvent(
                start_tick=bar_start,
                length_tick=bar_ticks,
                midi_note=midi,
                velocity=min(127, vel),
            ))

    return notes


def _gen_arpeggio(
    extra: ExtraTrackConfig, cfg: GenreConfig,
    scale: list[int], chord_prog: list[tuple[int, str]],
    intervals: list[int], root: int,
    bar_ticks: int, tpb: int, bpb: int,
) -> list[NoteEvent]:
    """Arpeggiated extra track cycling through chord voicings."""
    notes: list[NoteEvent] = []
    lo, hi = extra.pitch_range

    for bar_idx, (degree, quality) in enumerate(chord_prog):
        bar_start = bar_idx * bar_ticks
        idx_in_scale = degree % len(intervals)
        chord_root_pc = (root + intervals[idx_in_scale]) % 12

        chord_root = None
        for octave in range(11):
            candidate = chord_root_pc + octave * 12
            if lo <= candidate <= hi:
                chord_root = candidate
                break
        if chord_root is None:
            chord_root = (lo + hi) // 2

        chord_ivs = _CHORD_TYPES.get(quality, [0, 4, 7])
        voicing: list[int] = []
        for iv in chord_ivs:
            note = chord_root + iv
            while note > hi and note > 12:
                note -= 12
            while note < lo:
                note += 12
            if lo <= note <= hi:
                voicing.append(note)
        if not voicing:
            voicing = [chord_root]

        arp_style = random.choice(["up", "down", "updown"])
        if arp_style == "down":
            pattern_notes = list(reversed(voicing))
        elif arp_style == "updown":
            mid = voicing[1:-1] if len(voicing) > 2 else voicing
            pattern_notes = voicing + list(reversed(mid))
        else:
            pattern_notes = list(voicing)

        note_dur = max(1, tpb // 2)
        tick = 0
        idx = 0
        while tick < bar_ticks:
            if random.random() >= extra.rest_prob:
                midi = pattern_notes[idx % len(pattern_notes)]
                vel = random.randint(*extra.vel)
                notes.append(NoteEvent(
                    start_tick=bar_start + tick,
                    length_tick=min(note_dur, bar_ticks - tick),
                    midi_note=midi,
                    velocity=min(127, vel),
                ))
            idx += 1
            tick += note_dur

    return notes
