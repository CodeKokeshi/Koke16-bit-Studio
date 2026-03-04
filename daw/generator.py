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


# ── Shared 32-bar Town progression builder ──────────────────────────

def _town_32bar_progression(vibe: str) -> list[tuple[int, str]]:
    """Build a 32-bar chord progression for Town themes.

    *vibe* is one of ``"inland"`` or ``"island"``.
    """
    cadence_v = [(4, "maj")]
    cadence_sus = [(4, "sus4")]

    if vibe == "island":
        sea = [(0, "maj"), (6, "maj"), (3, "maj"), (0, "maj")]
        island = [(0, "maj"), (3, "maj"), (6, "maj"), (0, "maj")]
        bounce = [(0, "maj"), (4, "maj"), (3, "maj"), (0, "maj")]

        intro = [(0, "maj")] * 4
        head_a = random.choice([sea, island])
        head_b = random.choice([bounce, sea])
        dev_a = [(3, "maj"), (6, "maj"), (0, "maj"), (4, "maj")]
        dev_b = [(0, "maj"), (6, "maj"), (3, "maj"), (4, "7")]
        clim_a = random.choice([sea, bounce])
        clim_b = [(6, "maj"), (3, "maj"), (4, "7"), (0, "maj")]
    else:  # inland
        cozy = [(0, "maj"), (1, "7"), (4, "7"), (0, "maj")]
        soft = [(0, "maj"), (5, "min"), (1, "min"), (4, "maj")]
        walk = [(0, "maj"), (3, "maj"), (4, "maj"), (0, "maj")]

        intro = [(0, "maj")] * 4
        head_a = random.choice([cozy, walk])
        head_b = random.choice([soft, cozy])
        dev_a = [(5, "min"), (3, "maj"), (1, "min"), (4, "7")]
        dev_b = [(0, "maj"), (5, "min"), (3, "maj"), (4, "maj")]
        clim_a = [(0, "maj"), (1, "7"), (4, "7"), (0, "maj")]
        clim_b = [(5, "min"), (3, "maj"), (1, "7"), (4, "7")]

    reset = [(0, "maj"), (0, "maj")] + random.choice([cadence_v, cadence_sus]) * 2
    return intro + head_a + head_b + dev_a + dev_b + clim_a + clim_b + reset


# ── Standard rhythm presets for Town themes ─────────────────────────

_TOWN_LEAD_RHYTHMS = [
    _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 2, 2.5, 3],
                   [0.5, 0.5, 1, 0.5, 0.5, 1]),
    _simple_rhythm(_TPB, _BPB, [0, 1, 1.5, 2, 3],
                   [1, 0.5, 0.5, 1, 1]),
]

_TOWN_BASS_332 = [
    _simple_rhythm(_TPB, _BPB, [0, 1.5, 3], [1.5, 1.5, 1]),
]

_TOWN_DRUM_8THS = [
    _simple_rhythm(_TPB, _BPB, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
                   [0.25] * 8),
]

_TOWN_HARMONY_STACCATO = [
    _simple_rhythm(_TPB, _BPB, [0, 1.5, 3], [0.5, 0.5, 0.5]),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generic Town  —  plain waveforms, 4–5 tracks, works-anywhere default
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_generic_town() -> GenreConfig:
    """Generic Town — clean waveforms, no specific console family.

    Uses Generic + NES instruments for a simple but pleasant sound.
    Lighter arrangement: 4 core tracks + optional pad.
    """
    vibe = random.choice(["inland", "inland", "inland", "island"])
    bpm_range = (90, 108) if vibe == "inland" else (108, 118)
    scale_choices = ["major"] if vibe == "inland" else ["mixolydian"]
    root_choices = [0, 2, 5, 7] if vibe == "inland" else [0, 5, 7]

    prog_32 = _town_32bar_progression(vibe)

    return GenreConfig(
        name="Generic Town",
        bpm_range=bpm_range,
        bars=32,
        root_choices=root_choices,
        scale_choices=scale_choices,
        progressions=[prog_32],

        lead_rhythm=_TOWN_LEAD_RHYTHMS,
        bass_rhythm=_TOWN_BASS_332,
        drum_rhythm=_TOWN_DRUM_8THS,

        lead_instrument=random.choice([
            ("Generic Sine", "sine"),
            ("Generic Triangle", "triangle"),
        ]),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Generic Pulse 25%", "pulse25"),
        drum_instrument=("Generic Noise Drum", "noise"),

        lead_vel=(82, 102),
        bass_vel=(88, 96),
        harmony_vel=(70, 92),
        drum_vel=(35, 52),
        lead_mix_range=(0.72, 0.88),
        bass_mix_range=(0.60, 0.76),
        harmony_mix_range=(0.42, 0.58),
        drum_mix_range=(0.26, 0.38),
        lead_range=(60, 84),
        bass_range=(36, 55),
        harmony_range=(48, 72),
        drum_pitches=[38, 42],           # snare + hi-hat only
        lead_step_max=3,
        lead_rest_prob=0.10,
        lead_style="gba_town",           # reuse section-aware melody
        bass_style="root_fifth",
        harmony_style="staccato",
        drum_style="gba_town",
        harmony_rhythm=_TOWN_HARMONY_STACCATO,
        swing=0.08,

        # Lead doubling with NES square sparkle
        lead_doubling=("NES Square", "square", 0.15),

        extra_tracks=[
            ExtraTrackConfig(
                role="Pad",
                gen_type="pad",
                instrument=("Generic Saw", "sawtooth"),
                vel=(38, 55),
                pitch_range=(48, 72),
                mix_range=(0.25, 0.40),
                rest_prob=0.10,
            ),
        ],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GBA Town  —  authentic Sappy-engine Pokémon town theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_gba_town() -> GenreConfig:
    """GBA Town — authentic Pokémon-style town theme (Sappy / m4a engine).

    Full 32-bar structure with 6–8 tracks:
    Lead, Bass, Harmony, Drums, Lead Sparkle, Strings Pad,
    Counter Melody, and (inland only) Guitar Arpeggio.

    Inland (Pallet/Littleroot/Azalea) or Island (Dewford/Slateport).
    """
    vibe = random.choice(["inland", "inland", "inland", "island"])
    bpm_range = (95, 105) if vibe == "inland" else (110, 120)
    scale_choices = ["major"] if vibe == "inland" else ["mixolydian"]
    root_choices = [0, 2, 5, 7] if vibe == "inland" else [0, 5, 7]

    prog_32 = _town_32bar_progression(vibe)

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
            mix_range=(0.30, 0.48),
            rest_prob=0.08,
            pan=0.30,
        ),
        ExtraTrackConfig(
            role="Counter Melody",
            gen_type="counter_melody",
            instrument=counter_inst,
            vel=(55, 75),
            pitch_range=(60, 84),
            mix_range=(0.32, 0.50),
            rest_prob=0.38,
            step_max=2,
            pan=0.15 if vibe == "inland" else 0.20,
        ),
    ]

    if vibe == "inland":
        extras.append(ExtraTrackConfig(
            role="Guitar Arpeggio",
            gen_type="arpeggio",
            instrument=("GBA Acoustic Guitar", "triangle"),
            vel=(50, 70),
            pitch_range=(48, 72),
            mix_range=(0.30, 0.48),
            rest_prob=0.15,
            pan=-0.15,
        ))

    return GenreConfig(
        name="GBA Town",
        bpm_range=bpm_range,
        bars=32,
        root_choices=root_choices,
        scale_choices=scale_choices,
        progressions=[prog_32],

        lead_rhythm=_TOWN_LEAD_RHYTHMS,
        bass_rhythm=_TOWN_BASS_332,
        drum_rhythm=_TOWN_DRUM_8THS,

        lead_instrument=lead_inst,
        bass_instrument=random.choice([("GBA Fretless Bass", "triangle"),
                                       ("GBA Slap Bass", "square")]),
        harmony_instrument=("GBA Piano", "pulse25"),
        drum_instrument=("GBA Light Kit", "noise"),

        lead_vel=(85, 105),
        bass_vel=(92, 98),
        harmony_vel=(80, 105) if vibe == "inland" else (78, 98),
        drum_vel=(35, 55) if vibe == "inland" else (40, 58),
        lead_mix_range=(0.75, 0.90),
        bass_mix_range=(0.62, 0.78),
        harmony_mix_range=(0.45, 0.62),
        drum_mix_range=(0.28, 0.42),
        lead_range=(60, 84),
        bass_range=(36, 55),
        harmony_range=(48, 72),
        drum_pitches=[38, 42],
        lead_step_max=3,
        lead_rest_prob=0.08 if vibe == "inland" else 0.06,
        lead_style="gba_town",
        bass_style="root_fifth",
        harmony_style="staccato",
        drum_style="gba_town",
        harmony_rhythm=_TOWN_HARMONY_STACCATO,

        lead_pan=0.0,
        bass_pan=0.0,
        harmony_pan=-0.30,
        drum_pan=0.0,

        lead_doubling=("NES Square", "square", 0.18),
        swing=0.10,

        extra_tracks=extras,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SNES Town  —  warm 16-bit RPG town (FF / Chrono Trigger / EarthBound)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _genre_snes_town() -> GenreConfig:
    """SNES Town — warm 16-bit RPG feel (Final Fantasy / Chrono Trigger).

    Rich orchestration using SNES instruments: Flute, Piano, Strings,
    Harp arpeggios, Marimba accents, and optional Trumpet counter-melody.
    32-bar structure with section awareness.
    """
    vibe = random.choice(["inland", "inland", "inland", "island"])
    bpm_range = (88, 100) if vibe == "inland" else (104, 115)
    scale_choices = ["major", "dorian"] if vibe == "inland" else ["mixolydian"]
    root_choices = [0, 2, 5, 7] if vibe == "inland" else [0, 5, 7]

    prog_32 = _town_32bar_progression(vibe)

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
            vel=(42, 62),
            pitch_range=(48, 72),
            mix_range=(0.32, 0.50),
            rest_prob=0.06,
            pan=0.25,
        ),
        # Harp arpeggios
        ExtraTrackConfig(
            role="Harp Arpeggio",
            gen_type="arpeggio",
            instrument=("SNES Harp", "sine"),
            vel=(48, 68),
            pitch_range=(48, 72),
            mix_range=(0.28, 0.44),
            rest_prob=0.12,
            pan=-0.20,
        ),
        # Marimba counter / twinkle
        ExtraTrackConfig(
            role="Marimba Counter",
            gen_type="counter_melody",
            instrument=("SNES Marimba", "triangle"),
            vel=(50, 70),
            pitch_range=(60, 84),
            mix_range=(0.28, 0.42),
            rest_prob=0.40,
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
        progressions=[prog_32],

        lead_rhythm=_TOWN_LEAD_RHYTHMS,
        bass_rhythm=_TOWN_BASS_332,
        drum_rhythm=_TOWN_DRUM_8THS,

        lead_instrument=lead_inst,
        bass_instrument=("SNES Slap Bass", "square"),
        harmony_instrument=("SNES Piano", "pulse25"),
        drum_instrument=("SNES Kit", "noise"),

        lead_vel=(82, 104),
        bass_vel=(90, 98),
        harmony_vel=(75, 100),
        drum_vel=(32, 52),
        lead_mix_range=(0.72, 0.88),
        bass_mix_range=(0.60, 0.76),
        harmony_mix_range=(0.44, 0.60),
        drum_mix_range=(0.26, 0.40),
        lead_range=(60, 84),
        bass_range=(36, 55),
        harmony_range=(48, 72),
        drum_pitches=[38, 42],
        lead_step_max=3,
        lead_rest_prob=0.08,
        lead_style="gba_town",
        bass_style="root_fifth",
        harmony_style="staccato",
        drum_style="gba_town",
        harmony_rhythm=_TOWN_HARMONY_STACCATO,

        lead_pan=0.0,
        bass_pan=0.0,
        harmony_pan=-0.25,
        drum_pan=0.0,

        # SNES doubling: flute + triangle sparkle
        lead_doubling=("SNES Acoustic", "triangle", 0.16),
        swing=0.08,

        extra_tracks=extras,
    )


# ── Registry of all genres ──────────────────────────────────────────

GENRE_BUILDERS: dict[str, callable] = {
    "Generic Town":     _genre_generic_town,
    "GBA Town":         _genre_gba_town,
    "SNES Town":        _genre_snes_town,
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

    # Choose a chord progression — if vibe is set, build dynamically to match bar count
    if cfg.vibe:
        chord_prog = _town_progression(cfg.vibe, cfg.bars)
    elif cfg.progressions:
        chord_prog = list(random.choice(cfg.progressions))
    else:
        # Fallback: simple I-IV-V-I * bars/4
        chord_prog = [(0, "maj"), (3, "maj"), (4, "maj"), (0, "maj")] * (cfg.bars // 4 or 1)

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

    rhythm_patterns = cfg.lead_rhythm or [
        _simple_rhythm(tpb, bpb, [0, 1, 2, 3], [1, 1, 1, 1])
    ]

    for bar_idx, (degree, _quality) in enumerate(chord_prog):
        bar_start = bar_idx * bar_ticks
        pattern = random.choice(rhythm_patterns)

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
            # ── GBA Town: 32-bar section-aware melody ──────────────
            # Intro (0-3): silence — no melody
            # Head (4-11): call/response, lower-mid register
            # Development (12-19): upper register, ascending bias
            # Climax (20-27): highest energy, widest range
            # Reset (28-31): sparse descending, fading out

            if bar_idx < 4:
                continue                     # Intro: no melody at all
            if bar_idx >= 30:
                continue                     # Last 2 bars of Reset: silence

            # Determine section & register target
            if bar_idx < 12:                 # Head
                register_target = 0.35
                step_bias = 0
            elif bar_idx < 20:               # Development
                register_target = 0.65
                step_bias = 1                # ascending bias
            elif bar_idx < 28:               # Climax
                register_target = 0.80
                half = (bar_idx - 20) < 4
                step_bias = 1 if half else -1
            else:                            # Reset (bars 28-29)
                register_target = 0.30
                step_bias = -1

            # Jump register at section boundaries
            if bar_idx in (4, 12, 20, 28):
                target_idx = int(len(lead_scale) * register_target)
                current_idx = max(0, min(len(lead_scale) - 1,
                                         target_idx + random.randint(-2, 2)))

            # ── "Gap" rule: every other bar within each section is a
            #    "response" / breathing bar — keep only 1-2 notes max.
            section_start = (4 if bar_idx < 12 else
                             12 if bar_idx < 20 else
                             20 if bar_idx < 28 else 28)
            if (bar_idx - section_start) % 2 == 1:
                keep = random.randint(1, 2)
                pattern = pattern[:keep]

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
            if step_bias > 0:
                step = random.randint(0, max_step)
            elif step_bias < 0:
                step = random.randint(-max_step, 0)
            else:
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

            # ── GBA Town velocity ramps: pitch direction → velocity ──
            if cfg.lead_style == "gba_town" and prev_pitch is not None:
                if candidate_note > prev_pitch:
                    vel += random.randint(3, 5)      # ascending = crescendo
                elif candidate_note < prev_pitch:
                    vel -= random.randint(3, 5)      # descending = decrescendo
                vel += random.randint(-5, 5)         # humanize ±5

            prev_pitch = candidate_note

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
            notes.append(NoteEvent(
                bar_start + tpb * 3, tpb,
                bass_fifth if random.random() < 0.5 else higher,
                min(127, v3)))
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
            continue

        if cfg.bass_style == "root_fifth":
            # GBA Town bass: ONLY root & 5th, never chords.
            # Follows the rhythm pattern (typically 3+3+2) and strictly
            # alternates root ↔ fifth.  Section-aware for 32-bar Town:
            #   Intro (0-3): play but softer, Reset (28-31): drop at bar 30.
            is_intro = bar_idx < 4
            if bar_idx >= 30:
                continue                     # drop bass for last 2 bars

            toggle = False                   # alternates root / fifth
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

    # Choose style: block chords or arpeggiated (unless overridden)
    use_arpeggio = random.random() < 0.4

    harmony_rhythm = cfg.harmony_rhythm or []

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

        # ── Staccato harmony (3+3+2 off-beat chords for GBA Town) ──
        if cfg.harmony_style == "staccato" and harmony_rhythm:
            # Section-aware for 32-bar Town:
            #   Reset (bars 28-31): drop harmony at bar 30
            if bar_idx >= 30:
                continue
            is_intro = bar_idx < 4
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

        # ── GBA Town drums: section-aware, snare + hi-hat only ──────
        if cfg.drum_style == "gba_town":
            # Drop drums entirely in last 2 bars (reset tail)
            if bar_idx >= 30:
                continue
            is_intro = bar_idx < 4
            is_reset = bar_idx >= 28

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

            # Light snare fill right before reset (bars 27-28)
            if bar_idx in (27, 28) and random.random() < 0.5:
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
