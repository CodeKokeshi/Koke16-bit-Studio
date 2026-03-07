"""SmartTheoryFixer – context-aware music-theory post-processor.

Cleans up raw CQT transcription output by applying scale-aware
corrections **without** destroying intentional dissonance or character.

Design principles
-----------------
* Detect the key / scale from the note histogram (Krumhansl-Schmuckler).
* Preserve "blue notes" common in retro soundtracks (b3 in major, b5).
* Only snap *short, quiet* out-of-scale notes; leave long or loud ones
  (they are intentional).
* Fill melodic gaps with scale-aware linear interpolation that respects
  the direction of the preceding melody contour.
* All behaviour is tuneable via a ``strictness`` parameter (0 = raw,
  1 = strict theory enforcement).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import groupby
from typing import Sequence

from daw.models import NoteEvent


# ─── Scale / key data ──────────────────────────────────────────────────

# Pitch-class profiles (Krumhansl-Kessler) – used for key detection.
_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Semitone intervals for common scales (relative to root)
_MAJOR_INTERVALS = {0, 2, 4, 5, 7, 9, 11}
_MINOR_INTERVALS = {0, 2, 3, 5, 7, 8, 10}

# Retro "blue note" extensions – tolerated extra pitch classes
_BLUE_NOTES_MAJOR = {3, 6}      # minor 3rd + tritone (b5) in a major key
_BLUE_NOTES_MINOR = {6, 1}      # tritone + b9 (passing tones in minor)

# Note names for debug / display
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ─── Key detection ─────────────────────────────────────────────────────

def _detect_key(notes: Sequence[NoteEvent]) -> tuple[int, str, set[int]]:
    """Return ``(root_pc, quality, scale_pcs)`` detected from *notes*.

    Uses the Krumhansl-Schmuckler algorithm: correlate the
    pitch-class histogram of the input with the major/minor profile
    rotated to every possible root.
    """
    if not notes:
        return 0, "major", _MAJOR_INTERVALS

    # Build weighted pitch-class histogram (weight by duration * velocity)
    hist = [0.0] * 12
    for n in notes:
        pc = n.midi_note % 12
        hist[pc] += n.length_tick * (n.velocity / 127.0)

    best_root = 0
    best_quality = "major"
    best_corr = -999.0

    for root in range(12):
        # Rotate histogram so 'root' aligns with index 0
        rotated = [hist[(root + i) % 12] for i in range(12)]
        for quality, profile in [("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)]:
            # Pearson correlation
            mean_r = sum(rotated) / 12.0
            mean_p = sum(profile) / 12.0
            num = sum((r - mean_r) * (p - mean_p) for r, p in zip(rotated, profile))
            den_r = math.sqrt(sum((r - mean_r) ** 2 for r in rotated)) or 1e-9
            den_p = math.sqrt(sum((p - mean_p) ** 2 for p in profile)) or 1e-9
            corr = num / (den_r * den_p)
            if corr > best_corr:
                best_corr = corr
                best_root = root
                best_quality = quality

    if best_quality == "major":
        scale_pcs = {(best_root + i) % 12 for i in _MAJOR_INTERVALS}
    else:
        scale_pcs = {(best_root + i) % 12 for i in _MINOR_INTERVALS}

    return best_root, best_quality, scale_pcs


def _get_blue_pcs(root: int, quality: str) -> set[int]:
    """Return pitch classes that qualify as 'blue notes'."""
    raw = _BLUE_NOTES_MAJOR if quality == "major" else _BLUE_NOTES_MINOR
    return {(root + offset) % 12 for offset in raw}


# ─── Chord progression detection ──────────────────────────────────────

_CHORD_TEMPLATES: dict[str, list[int]] = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "7":    [0, 4, 7, 10],
    "m7":   [0, 3, 7, 10],
    "maj7": [0, 4, 7, 11],
    "dim7": [0, 3, 6, 9],
    "sus4": [0, 5, 7],
    "sus2": [0, 2, 7],
}

# Diatonic chord qualities built on each scale degree (0-based semitones)
_DIATONIC_MAJOR: dict[int, str] = {
    0: "maj", 2: "min", 4: "min", 5: "maj", 7: "maj", 9: "min", 11: "dim",
}
_DIATONIC_MINOR: dict[int, str] = {
    0: "min", 2: "dim", 3: "maj", 5: "min", 7: "min", 8: "maj", 10: "maj",
}


@dataclass
class ChordRegion:
    """A detected chord spanning a time region."""
    start_tick: int
    end_tick: int
    root_pc: int           # 0–11 pitch class
    quality: str           # key into _CHORD_TEMPLATES
    chord_pcs: frozenset[int] = field(default_factory=frozenset)
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if not self.chord_pcs:
            intervals = _CHORD_TEMPLATES.get(self.quality, [0, 4, 7])
            self.chord_pcs = frozenset((self.root_pc + iv) % 12 for iv in intervals)


@dataclass
class SectionRegion:
    """A structural section of the song (verse, chorus, bridge, etc.)."""
    start_tick: int
    end_tick: int
    label: str          # "A", "B", "C", …
    energy: float = 0.5  # 0.0–1.0 average normalised energy


@dataclass
class SectionAnalysis:
    """Per-section musical analysis."""
    section: SectionRegion
    key_root: int
    key_quality: str
    scale_pcs: set[int]
    chord_regions: list[ChordRegion]
    phrase_boundaries: list[tuple[int, int]]


@dataclass
class ProjectAnalysis:
    """Complete analysis of a multi-track project."""
    key_root: int
    key_quality: str    # "major" | "minor"
    scale_pcs: set[int]
    blue_pcs: set[int]
    chord_regions: list[ChordRegion]
    groove_grid: int        # detected quantisation grid in ticks
    swing_amount: float     # 0.0 = straight, 0.3+ = swing feel
    phrase_boundaries: list[tuple[int, int]]  # (start_tick, end_tick)
    sections: list[SectionAnalysis] = field(default_factory=list)
    loop_aware: bool = False
    total_ticks: int = 0


def _detect_chord_progression(
    all_notes: Sequence[NoteEvent],
    ticks_per_beat: int,
    key_root: int,
    key_quality: str,
    scale_pcs: set[int],
    harmony_notes: Sequence[NoteEvent] | None = None,
    bass_notes: Sequence[NoteEvent] | None = None,
) -> list[ChordRegion]:
    """Read chords from the actual music rather than brute-force template matching.

    Prioritises the harmony track (if provided) as the chord source,
    uses the bass track to confirm/set the root, and falls back to all
    pitched notes otherwise.  The chord's ``chord_pcs`` field stores
    the *actual* sounding pitch classes — not an idealised template — so
    downstream corrections respect what the music is really doing.

    Only AFTER reading the sounding notes do we label the quality
    (major / minor / etc.) by finding the closest known template, purely
    for display purposes.  The raw ``chord_pcs`` is what matters for
    note classification and correction.
    """
    if not all_notes:
        return []

    last_tick = max(n.start_tick + n.length_tick for n in all_notes)
    beat = max(1, ticks_per_beat)
    harm = list(harmony_notes) if harmony_notes else []
    bass = list(bass_notes) if bass_notes else []

    regions: list[ChordRegion] = []
    prev: ChordRegion | None = None

    for b_start in range(0, last_tick, beat):
        b_end = b_start + beat

        # ── Collect sounding PCs per role ────────────────────────
        def _collect(src: Sequence[NoteEvent]) -> dict[int, float]:
            pcs: dict[int, float] = {}
            for n in src:
                n_end = n.start_tick + n.length_tick
                if n.start_tick >= b_end or n_end <= b_start:
                    continue
                overlap = min(n_end, b_end) - max(n.start_tick, b_start)
                w = overlap * (n.velocity / 127.0)
                pc = n.midi_note % 12
                pcs[pc] = pcs.get(pc, 0.0) + w
            return pcs

        harm_pcs = _collect(harm)
        bass_pcs = _collect(bass)
        all_pcs = _collect(all_notes)

        # Primary chord source: harmony if it has ≥2 PCs, else merged
        chord_src = harm_pcs if len(harm_pcs) >= 2 else all_pcs

        if not chord_src:
            # Silence — carry previous chord
            if prev:
                regions.append(ChordRegion(
                    b_start, b_end, prev.root_pc, prev.quality,
                    prev.chord_pcs, 0.2,
                ))
            else:
                qual = "maj" if key_quality == "major" else "min"
                regions.append(ChordRegion(b_start, b_end, key_root, qual))
            prev = regions[-1]
            continue

        if len(chord_src) < 2:
            if prev:
                regions.append(ChordRegion(
                    b_start, b_end, prev.root_pc, prev.quality,
                    prev.chord_pcs, 0.3,
                ))
            else:
                qual = "maj" if key_quality == "major" else "min"
                regions.append(ChordRegion(b_start, b_end, key_root, qual))
            prev = regions[-1]
            continue

        # ── Determine root ────────────────────────────────────────
        if bass_pcs:
            root_pc = max(bass_pcs, key=lambda k: bass_pcs[k])
        else:
            # Lowest-sounding pitch in this beat
            lowest_note = min(
                (n for n in all_notes
                 if n.start_tick < b_end and n.start_tick + n.length_tick > b_start),
                key=lambda n: n.midi_note,
                default=None,
            )
            root_pc = lowest_note.midi_note % 12 if lowest_note else max(chord_src, key=lambda k: chord_src[k])

        # ── Build the chord from actual sounding PCs ──────────────
        sounding_pcs = frozenset(chord_src.keys())

        # ── Label quality (for display only; chord_pcs is truth) ──
        intervals = frozenset((pc - root_pc) % 12 for pc in sounding_pcs)
        best_quality = "?"
        best_match = 0
        for qual_name, tpl_intervals in _CHORD_TEMPLATES.items():
            tpl_set = frozenset(tpl_intervals)
            match_count = len(intervals & tpl_set)
            if match_count > best_match or (match_count == best_match and len(tpl_intervals) > len(_CHORD_TEMPLATES.get(best_quality, []))):
                best_match = match_count
                best_quality = qual_name

        confidence = min(1.0, best_match / max(1, len(sounding_pcs)))
        region = ChordRegion(
            b_start, b_end, root_pc, best_quality,
            sounding_pcs, confidence,
        )
        regions.append(region)
        prev = region

    # Merge consecutive chords that share the same root + sounding PCs
    if not regions:
        return regions
    merged: list[ChordRegion] = [regions[0]]
    for r in regions[1:]:
        m = merged[-1]
        if m.root_pc == r.root_pc and m.chord_pcs == r.chord_pcs:
            merged[-1] = ChordRegion(
                m.start_tick, r.end_tick, m.root_pc, m.quality,
                m.chord_pcs, (m.confidence + r.confidence) / 2,
            )
        else:
            merged.append(r)
    return merged


def _chord_at_tick(regions: list[ChordRegion], tick: int) -> ChordRegion | None:
    """Return the chord region active at *tick* (binary-ish search)."""
    for r in regions:
        if r.start_tick <= tick < r.end_tick:
            return r
    return regions[-1] if regions else None


# ─── Groove / rhythm analysis ─────────────────────────────────────────

def _detect_groove(
    notes: Sequence[NoteEvent],
    ticks_per_beat: int,
) -> tuple[int, float]:
    """Detect the quantisation grid and swing feel from note placements.

    Returns ``(grid_size, swing_amount)`` where *grid_size* is the best-fit
    grid in ticks and *swing_amount* is 0.0 (straight) to ~0.5 (heavy swing).
    """
    if not notes or ticks_per_beat < 1:
        return ticks_per_beat, 0.0

    candidates = sorted({
        max(1, ticks_per_beat // 4),
        max(1, ticks_per_beat // 2),
        ticks_per_beat,
        ticks_per_beat * 2,
    })

    best_grid = ticks_per_beat
    best_avg = float(ticks_per_beat)

    for grid in candidates:
        offsets = [min(n.start_tick % grid, grid - n.start_tick % grid) for n in notes]
        avg = sum(offsets) / len(offsets) if offsets else float(grid)
        # Prefer larger grids when offsets are tied (more musically meaningful)
        if avg < best_avg or (avg == best_avg and grid > best_grid):
            best_avg = avg
            best_grid = grid

    # Swing detection: are off-beat notes consistently late?
    half = best_grid
    swing = 0.0
    if half >= 2 and len(notes) >= 6:
        on_offsets: list[float] = []
        off_offsets: list[float] = []
        for n in notes:
            nearest = round(n.start_tick / half) * half
            offset = n.start_tick - nearest
            beat_idx = round(n.start_tick / half)
            if beat_idx % 2 == 0:
                on_offsets.append(offset)
            else:
                off_offsets.append(offset)
        if off_offsets and on_offsets:
            avg_off = sum(off_offsets) / len(off_offsets)
            avg_on = sum(on_offsets) / len(on_offsets)
            delay = avg_off - avg_on
            swing = max(0.0, min(0.5, delay / max(1, half)))

    return best_grid, swing


# ─── Phrase detection ──────────────────────────────────────────────────

def _detect_phrases(
    notes: Sequence[NoteEvent],
    ticks_per_beat: int,
) -> list[tuple[int, int]]:
    """Detect musical phrase boundaries from gap / leap / velocity cues.

    Returns a list of ``(start_tick, end_tick)`` regions.
    """
    if not notes:
        return []
    if len(notes) < 3:
        end = notes[-1].start_tick + notes[-1].length_tick
        return [(notes[0].start_tick, end)]

    med_len = _median_length(notes)
    phrase_gap = max(ticks_per_beat * 2, int(med_len * 3))

    boundaries: list[int] = [notes[0].start_tick]
    for i in range(1, len(notes)):
        prev_end = notes[i - 1].start_tick + notes[i - 1].length_tick
        gap = notes[i].start_tick - prev_end
        if gap >= phrase_gap:
            boundaries.append(notes[i].start_tick)
            continue
        # Large pitch leap + dynamic shift → likely new phrase
        if (abs(notes[i].midi_note - notes[i - 1].midi_note) >= 12
                and abs(notes[i].velocity - notes[i - 1].velocity) > 20):
            boundaries.append(notes[i].start_tick)

    phrases: list[tuple[int, int]] = []
    for i, start in enumerate(boundaries):
        if i + 1 < len(boundaries):
            end = boundaries[i + 1]
        else:
            end = notes[-1].start_tick + notes[-1].length_tick
        phrases.append((start, end))
    return phrases


# ─── Section detection ─────────────────────────────────────────────────

def _detect_sections(
    all_notes: Sequence[NoteEvent],
    ticks_per_beat: int,
    phrases: list[tuple[int, int]],
) -> list[SectionRegion]:
    """Detect structural sections (A/B/C = verse/chorus/bridge, etc.).

    Divides the music into fixed-size windows (4 bars), computes a
    multi-feature fingerprint for each (pitch-class histogram **plus**
    normalised energy and average pitch), and clusters similar windows.
    Consecutive same-label windows are merged into a single
    :class:`SectionRegion`.
    """
    if not all_notes:
        return []

    last_tick = max(n.start_tick + n.length_tick for n in all_notes)
    bar_ticks = ticks_per_beat * 4
    section_window = bar_ticks * 4  # 4 bars

    if last_tick <= section_window:
        energy = (sum(n.velocity for n in all_notes) / max(1, len(all_notes))) / 127.0
        return [SectionRegion(0, last_tick, "A", energy)]

    # ── build per-window fingerprints ─────────────────────────────
    # Fingerprint = 14-element vector: 12 × PC histogram + normalised
    # avg-velocity + normalised avg-pitch.  This lets the clustering
    # differentiate sections that share pitch classes but differ in
    # energy or register.
    windows: list[tuple[int, int, list[float], float]] = []
    for w_start in range(0, last_tick, section_window):
        w_end = min(w_start + section_window, last_tick)
        w_notes = [n for n in all_notes
                   if n.start_tick >= w_start and n.start_tick < w_end]

        pc_hist = [0.0] * 12
        total_vel = 0.0
        total_pitch = 0.0
        for n in w_notes:
            pc_hist[n.midi_note % 12] += n.length_tick
            total_vel += n.velocity
            total_pitch += n.midi_note

        total_h = sum(pc_hist) or 1.0
        pc_hist = [x / total_h for x in pc_hist]

        n_notes = max(1, len(w_notes))
        avg_vel = total_vel / n_notes if w_notes else 0.0
        avg_pitch = total_pitch / n_notes if w_notes else 60.0

        # Normalise velocity (0–1) and pitch (map 24–108 → 0–1)
        norm_vel = avg_vel / 127.0
        norm_pitch = max(0.0, min(1.0, (avg_pitch - 24) / 84.0))

        # 14-dim fingerprint • velocity and pitch weighted ×2 so they
        # are not drowned out by the 12-dim PC histogram
        fp = pc_hist + [norm_vel * 2.0, norm_pitch * 2.0]

        density = len(w_notes) / max(1, w_end - w_start)
        energy = min(1.0, (density * 500 + avg_vel) / 200.0)
        windows.append((w_start, w_end, fp, energy))

    if len(windows) <= 1:
        return [SectionRegion(0, last_tick, "A", windows[0][3] if windows else 0.5)]

    # ── distance metric ─────────────────────────────────────────
    # Euclidean distance is more sensitive to the energy/pitch
    # dimensions than cosine, which is dominated by the PC histogram.
    def _fp_dist(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    # ── cluster windows by fingerprint distance ───────────────────
    label_chars = "ABCDEFGH"
    labels: list[str] = [""] * len(windows)
    label_fps: dict[str, list[float]] = {}
    next_label = 0

    # Adaptive threshold: compute the average pairwise distance
    dists: list[float] = []
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            dists.append(_fp_dist(windows[i][2], windows[j][2]))
    if dists:
        mean_dist = sum(dists) / len(dists)
        threshold = max(0.15, mean_dist * 0.6)
    else:
        threshold = 0.15

    for i, (_, _, fp, _energy) in enumerate(windows):
        best_lbl: str | None = None
        best_dist = float("inf")
        for lbl, lbl_fp in label_fps.items():
            d = _fp_dist(fp, lbl_fp)
            if d < best_dist:
                best_dist = d
                best_lbl = lbl

        if best_lbl is not None and best_dist < threshold:
            labels[i] = best_lbl
        else:
            lbl = label_chars[min(next_label, len(label_chars) - 1)]
            next_label += 1
            labels[i] = lbl
            label_fps[lbl] = fp

    # ── merge consecutive same-label windows ──────────────────────
    sections: list[SectionRegion] = []
    i = 0
    while i < len(windows):
        start = windows[i][0]
        lbl = labels[i]
        j = i + 1
        while j < len(windows) and labels[j] == lbl:
            j += 1
        end = windows[j - 1][1]
        avg_energy = sum(windows[k][3] for k in range(i, j)) / max(1, j - i)
        sections.append(SectionRegion(start, end, lbl, avg_energy))
        i = j

    return sections


# ─── Note-function classification ─────────────────────────────────────

def _classify_note(
    note: NoteEvent,
    prev_note: NoteEvent | None,
    next_note: NoteEvent | None,
    chord: ChordRegion | None,
    scale_pcs: set[int],
    blue_pcs: set[int],
) -> str:
    """Classify a note's musical function relative to its harmonic context.

    Returns one of:
        ``'chord_tone'``   – note belongs to the current chord
        ``'scale_tone'``   – in scale, consonant extension
        ``'passing'``      – stepwise connection between chord / scale tones
        ``'neighbor'``     – decoration of a chord tone (step away, step back)
        ``'approach'``     – chromatic/diatonic lead-in to a chord tone
        ``'blue'``         – characteristic blue note (retro flavour)
        ``'suspension'``   – held from previous chord context
        ``'wrong'``        – no identifiable musical function
    """
    pc = note.midi_note % 12
    cpcs = chord.chord_pcs if chord else frozenset()

    if pc in cpcs:
        return "chord_tone"
    if pc in blue_pcs:
        return "blue"
    if pc in scale_pcs:
        # Could still be a passing / neighbor tone
        if prev_note and next_note:
            pp = prev_note.midi_note % 12
            np = next_note.midi_note % 12
            if pp in cpcs and np in cpcs:
                if (abs(note.midi_note - prev_note.midi_note) <= 2
                        and abs(note.midi_note - next_note.midi_note) <= 2):
                    return "passing"
        if next_note:
            np = next_note.midi_note % 12
            if np in cpcs and abs(note.midi_note - next_note.midi_note) <= 2:
                return "neighbor"
        return "scale_tone"

    # Chromatic note — check approach-tone pattern
    if next_note:
        np = next_note.midi_note % 12
        if np in cpcs and abs(note.midi_note - next_note.midi_note) == 1:
            return "approach"

    # Check suspension (note was a chord tone in the previous chord region)
    # Heuristic: if the note resolves down by step to a chord tone
    if next_note:
        np = next_note.midi_note % 12
        if np in cpcs and (note.midi_note - next_note.midi_note) in (1, 2):
            return "suspension"

    return "wrong"


def _classify_track_notes(
    notes: list[NoteEvent],
    chord_regions: list[ChordRegion],
    scale_pcs: set[int],
    blue_pcs: set[int],
    loop_aware: bool = False,
) -> list[str]:
    """Return a classification string for every note in *notes* (same order).

    When *loop_aware* the last note's "next" wraps to the first note
    and vice-versa, so boundary notes are not unfairly penalised.
    """
    n = len(notes)
    result: list[str] = []
    for i, note in enumerate(notes):
        if loop_aware and n > 1:
            prev_n = notes[(i - 1) % n]
            next_n = notes[(i + 1) % n]
        else:
            prev_n = notes[i - 1] if i > 0 else None
            next_n = notes[i + 1] if i < n - 1 else None
        chord = _chord_at_tick(chord_regions, note.start_tick)
        result.append(_classify_note(note, prev_n, next_n, chord, scale_pcs, blue_pcs))
    return result


# ─── Helpers ───────────────────────────────────────────────────────────

def _nearest_scale_tone(midi_note: int, scale_pcs: set[int]) -> int:
    """Snap *midi_note* to the closest pitch class in *scale_pcs*."""
    pc = midi_note % 12
    if pc in scale_pcs:
        return midi_note
    # Search ±1, ±2, ... semitones
    for delta in range(1, 7):
        if (pc + delta) % 12 in scale_pcs:
            return midi_note + delta
        if (pc - delta) % 12 in scale_pcs:
            return midi_note - delta
    return midi_note  # fallback (shouldn't happen with 7-note scales)


def _median_length(notes: Sequence[NoteEvent]) -> float:
    """Median note length – used as the 'typical' duration baseline."""
    if not notes:
        return 4.0
    lengths = sorted(n.length_tick for n in notes)
    mid = len(lengths) // 2
    if len(lengths) % 2 == 0:
        return (lengths[mid - 1] + lengths[mid]) / 2.0
    return float(lengths[mid])


def _median_velocity(notes: Sequence[NoteEvent]) -> float:
    if not notes:
        return 90.0
    vels = sorted(n.velocity for n in notes)
    mid = len(vels) // 2
    return float(vels[mid])


# ─── Track-role detection ──────────────────────────────────────────────

# Waveforms / names that strongly suggest a drum track
_DRUM_WAVEFORMS = {"noise"}
_DRUM_KEYWORDS = {"drum", "noise", "perc", "kick", "snare", "hat", "cymbal"}
_BASS_KEYWORDS = {"bass"}
_LEAD_KEYWORDS = {"lead", "melody", "vocal"}
_HARMONY_KEYWORDS = {"harmony", "chord", "pad"}

# MIDI drum notes used by the transcriber
_DRUM_MIDI_SET = {35, 36, 38, 42, 46, 49}


def detect_track_role(
    notes: list[NoteEvent],
    waveform: str = "",
    instrument_name: str = "",
    track_name: str = "",
) -> str:
    """Infer whether a track is ``lead``, ``bass``, ``harmony``, or ``drums``.

    Uses a combination of heuristics:
    1. **Instrument / waveform keywords** — if the name or waveform screams
       "drum" or "bass", trust it.
    2. **Pitch distribution** — drums cluster around specific MIDI notes,
       bass lives below MIDI 52, lead above 60.
    3. **Rhythmic profile** — drums tend to have very short notes (length 1)
       with limited pitch variety.
    4. **Polyphony** — harmony tracks have many simultaneous notes;
       lead/bass are mostly monophonic.
    """
    name_lower = (instrument_name + " " + track_name).lower()
    wf_lower = waveform.lower()

    # ── 1) keyword / waveform check ───────────────────────────────
    if wf_lower in _DRUM_WAVEFORMS or any(k in name_lower for k in _DRUM_KEYWORDS):
        return "drums"
    if any(k in name_lower for k in _BASS_KEYWORDS):
        return "bass"
    if any(k in name_lower for k in _LEAD_KEYWORDS):
        return "lead"
    if any(k in name_lower for k in _HARMONY_KEYWORDS):
        return "harmony"

    if not notes:
        return "lead"  # empty track → default

    # ── 2) pitch statistics ───────────────────────────────────────
    pitches = [n.midi_note for n in notes]
    unique_pitches = set(pitches)
    avg_pitch = sum(pitches) / len(pitches)
    lengths = [n.length_tick for n in notes]
    avg_len = sum(lengths) / len(lengths)

    # Drums: few unique pitches, short notes, many notes clustered
    #        around percussion MIDI values
    drum_hits = sum(1 for p in pitches if p in _DRUM_MIDI_SET)
    if drum_hits > len(pitches) * 0.5 and avg_len <= 2:
        return "drums"
    if len(unique_pitches) <= 4 and avg_len <= 1.5 and len(notes) > 8:
        return "drums"

    # ── 3) pitch range → bass vs lead vs harmony ─────────────────
    if avg_pitch <= 52:
        return "bass"

    # Polyphony check: count how many notes overlap at the same tick
    from collections import Counter as _C
    tick_counts = _C(n.start_tick for n in notes)
    avg_polyphony = sum(tick_counts.values()) / max(1, len(tick_counts))

    if avg_polyphony >= 2.0 and 50 <= avg_pitch <= 72:
        return "harmony"

    return "lead"


# ─── SmartTheoryFixer ──────────────────────────────────────────────────

class SmartTheoryFixer:
    """Context-aware music-theory post-processor.

    Parameters
    ----------
    strictness : float
        0.0 = leave everything untouched (raw import).
        1.0 = strict theory enforcement (robot-clean).
        Default 0.65 balances clean output with character preservation.

    Typical usage::

        fixer = SmartTheoryFixer(strictness=0.65)
        clean_lead = fixer.fix(lead_events, role="lead")
        clean_bass = fixer.fix(bass_events, role="bass")
    """

    def __init__(self, strictness: float = 0.65) -> None:
        self.strictness = max(0.0, min(1.0, strictness))

    # ─── public API ────────────────────────────────────────────────

    def fix(
        self,
        notes: list[NoteEvent],
        role: str = "lead",
    ) -> list[NoteEvent]:
        """Apply theory cleaning to *notes* for a given track *role*.

        Steps performed (in order):
        1. Detect key / scale from note content.
        2. Snap out-of-scale "glitch" notes (short + quiet).
        3. Remove duplicate-pitch overlaps.
        4. Fill melodic gaps with interpolated passing tones (lead only).
        """
        if not notes or self.strictness <= 0.0:
            return notes

        # Work on copies so caller's data isn't mutated unexpectedly
        notes = [NoteEvent(n.start_tick, n.length_tick, n.midi_note, n.velocity)
                 for n in notes]

        # Sort by start time
        notes.sort(key=lambda n: n.start_tick)

        # 1) Detect key from ALL notes (best accuracy from full set)
        root, quality, scale_pcs = _detect_key(notes)
        blue_pcs = _get_blue_pcs(root, quality)

        # 2) Chromatic snap (glitch notes)
        notes = self._snap_glitches(notes, scale_pcs, blue_pcs)

        # 3) Remove duplicate-pitch overlaps
        notes = self._remove_overlaps(notes)

        # 4) Melodic gap fill (lead & harmony only – bass should stay sparse)
        if role in ("lead", "harmony") and self.strictness > 0.3:
            notes = self._fill_gaps(notes, scale_pcs)

        return notes

    def fix_multitrack(
        self,
        split: dict[str, list[NoteEvent]],
    ) -> dict[str, list[NoteEvent]]:
        """Convenience: fix every pitched track in a split dict.

        Drums are never theory-fixed (they don't have pitched semantics).
        The detected key uses ALL pitched notes combined for accuracy.
        """
        # Combine pitched notes for global key detection
        pitched = []
        for role in ("lead", "bass", "harmony"):
            pitched.extend(split.get(role, []))

        if not pitched or self.strictness <= 0.0:
            return split

        root, quality, scale_pcs = _detect_key(pitched)
        blue_pcs = _get_blue_pcs(root, quality)

        result: dict[str, list[NoteEvent]] = {}
        for role in ("lead", "bass", "harmony"):
            events = split.get(role, [])
            if not events:
                result[role] = events
                continue

            # Copy
            events = [NoteEvent(n.start_tick, n.length_tick, n.midi_note, n.velocity)
                      for n in events]
            events.sort(key=lambda n: n.start_tick)

            events = self._snap_glitches(events, scale_pcs, blue_pcs)
            events = self._remove_overlaps(events)
            if role in ("lead", "harmony") and self.strictness > 0.3:
                events = self._fill_gaps(events, scale_pcs)

            result[role] = events

        # Pass drums through untouched
        result["drums"] = split.get("drums", [])
        return result

    # ─── beautify (context-aware, multi-track music-theory engine) ───

    def analyze_project(
        self,
        tracks_notes: list[list[NoteEvent]],
        tracks_roles: list[str],
        ticks_per_beat: int = 4,
        loop_aware: bool = False,
    ) -> ProjectAnalysis:
        """Perform deep, role-aware, section-aware project analysis.

        1. Separate notes by role (harmony defines chords, bass confirms
           roots, lead/other are context).
        2. Detect global key from all pitched notes.
        3. Read chords from the music — harmony track has priority.
        4. Detect groove, phrases, and structural sections.
        5. Build per-section sub-analyses (local key, local chords,
           local phrases) so beautification can adapt to each part.
        """
        # ── 1) separate by role ───────────────────────────────────
        all_pitched: list[NoteEvent] = []
        harmony_notes: list[NoteEvent] = []
        bass_notes: list[NoteEvent] = []
        for notes, role in zip(tracks_notes, tracks_roles):
            if role == "drums":
                continue
            all_pitched.extend(notes)
            if role == "harmony":
                harmony_notes.extend(notes)
            elif role == "bass":
                bass_notes.extend(notes)

        # ── 2) global key ────────────────────────────────────────
        root, quality, scale_pcs = _detect_key(all_pitched)
        blue_pcs = _get_blue_pcs(root, quality)

        # ── 3) bottom-up chord reading from harmony + bass ───────
        chord_regions = _detect_chord_progression(
            all_pitched, ticks_per_beat, root, quality, scale_pcs,
            harmony_notes=harmony_notes or None,
            bass_notes=bass_notes or None,
        )

        # ── 4) groove / phrases / sections ────────────────────────
        groove_grid, swing = _detect_groove(all_pitched, ticks_per_beat)
        phrases = _detect_phrases(all_pitched, ticks_per_beat)
        sections = _detect_sections(all_pitched, ticks_per_beat, phrases)

        # ── 5) per-section sub-analyses ───────────────────────────
        section_analyses: list[SectionAnalysis] = []
        for sec in sections:
            s_all = [n for n in all_pitched
                     if n.start_tick >= sec.start_tick and n.start_tick < sec.end_tick]
            s_harm = [n for n in harmony_notes
                      if n.start_tick >= sec.start_tick and n.start_tick < sec.end_tick]
            s_bass = [n for n in bass_notes
                      if n.start_tick >= sec.start_tick and n.start_tick < sec.end_tick]

            # Local key — fall back to global if section is too short
            if len(s_all) >= 8:
                s_root, s_qual, s_scale = _detect_key(s_all)
            else:
                s_root, s_qual, s_scale = root, quality, scale_pcs

            s_chords = _detect_chord_progression(
                s_all, ticks_per_beat, s_root, s_qual, s_scale,
                harmony_notes=s_harm or None,
                bass_notes=s_bass or None,
            )
            s_phrases = [(s, e) for s, e in phrases
                         if s >= sec.start_tick and e <= sec.end_tick]

            section_analyses.append(SectionAnalysis(
                section=sec,
                key_root=s_root,
                key_quality=s_qual,
                scale_pcs=s_scale,
                chord_regions=s_chords,
                phrase_boundaries=s_phrases,
            ))

        total_ticks = max((n.start_tick + n.length_tick for n in all_pitched), default=0)

        return ProjectAnalysis(
            key_root=root,
            key_quality=quality,
            scale_pcs=scale_pcs,
            blue_pcs=blue_pcs,
            chord_regions=chord_regions,
            groove_grid=groove_grid,
            swing_amount=swing,
            phrase_boundaries=phrases,
            sections=section_analyses,
            loop_aware=loop_aware,
            total_ticks=total_ticks,
        )

    def beautify(
        self,
        notes: list[NoteEvent],
        role: str = "lead",
        ticks_per_beat: int = 4,
        analysis: ProjectAnalysis | None = None,
    ) -> list[NoteEvent]:
        """Context-aware, section-aware, loop-aware beautification.

        When *analysis* contains per-section sub-analyses, each note is
        corrected against the *local* chord/key context of its own
        section rather than a single global pass.  When
        ``analysis.loop_aware`` is True, the pipeline treats the music
        as circular — the end wraps to the start for classification,
        gap filling, and dynamics, preventing the boundary from
        becoming chaotic.

        **Pipeline (all pitched roles):**

        1. *Classify* notes per section (section-local chords & key).
        2. *Fix wrong notes* per section.
        3. *Role-specific* passes (lead / bass / harmony / drums).
        4. *Groove quantise* (detected grid + swing).
        5. *Phrase dynamics* — but **skip end-of-song diminuendo**
           when loop-aware.
        6. *Final cleanup* — overlaps, re-sort.
        """
        if not notes:
            return notes

        # Deep-copy
        notes = [NoteEvent(n.start_tick, n.length_tick, n.midi_note, n.velocity)
                 for n in notes]
        notes.sort(key=lambda n: n.start_tick)

        if role == "drums":
            return self._beautify_drums(
                notes, ticks_per_beat,
                analysis.groove_grid if analysis else ticks_per_beat,
            )

        # ── build / use analysis ──────────────────────────────────
        if analysis is None:
            root, quality, scale_pcs = _detect_key(notes)
            blue_pcs = _get_blue_pcs(root, quality)
            chord_regions = _detect_chord_progression(
                notes, ticks_per_beat, root, quality, scale_pcs,
            )
            groove_grid, swing = _detect_groove(notes, ticks_per_beat)
            phrases = _detect_phrases(notes, ticks_per_beat)
            analysis = ProjectAnalysis(
                root, quality, scale_pcs, blue_pcs,
                chord_regions, groove_grid, swing, phrases,
            )

        a = analysis  # shorthand
        loop = a.loop_aware

        # ── 1) classify — use section-local analysis when available ──
        if a.sections:
            classifications: list[str] = ["scale_tone"] * len(notes)
            for sa in a.sections:
                sec = sa.section
                blue = _get_blue_pcs(sa.key_root, sa.key_quality)
                for i, note in enumerate(notes):
                    if sec.start_tick <= note.start_tick < sec.end_tick:
                        prev_n = notes[i - 1] if i > 0 else (notes[-1] if loop and len(notes) > 1 else None)
                        next_n = notes[i + 1] if i < len(notes) - 1 else (notes[0] if loop and len(notes) > 1 else None)
                        chord = _chord_at_tick(sa.chord_regions, note.start_tick)
                        classifications[i] = _classify_note(
                            note, prev_n, next_n, chord, sa.scale_pcs, blue,
                        )
        else:
            classifications = _classify_track_notes(
                notes, a.chord_regions, a.scale_pcs, a.blue_pcs, loop,
            )

        # ── 2) fix wrong notes (section-context-aware) ───────────
        notes = self._fix_wrong_notes(notes, classifications, a)

        # ── 3) role-specific passes ───────────────────────────────
        if role == "bass":
            notes = self._beautify_bass(notes, a, ticks_per_beat)
        elif role == "harmony":
            notes = self._beautify_harmony(notes, a, ticks_per_beat)
        else:
            notes = self._beautify_lead(notes, a, ticks_per_beat)

        # ── 4) groove-aware quantisation ──────────────────────────
        notes = self._quantize_to_groove(notes, a.groove_grid, a.swing_amount)

        # ── 5) phrase-aware dynamics ──────────────────────────────
        notes = self._shape_phrase_dynamics(notes, a.phrase_boundaries, role, loop)

        # ── 6) final cleanup ──────────────────────────────────────
        notes = self._remove_overlaps(notes)
        notes.sort(key=lambda n: n.start_tick)
        return notes

    # ─── wrong-note correction (chord-context-aware) ───────────────

    def _fix_wrong_notes(
        self,
        notes: list[NoteEvent],
        classifications: list[str],
        analysis: ProjectAnalysis,
    ) -> list[NoteEvent]:
        """Correct notes classified as ``'wrong'``.

        Uses the **section-local** chord and scale context when sections
        are available, so a note that is "wrong" in one key is corrected
        using the *local* key, not the global one.

        Long or loud wrong notes are treated more gently — they may
        be intentional dissonance, so they are only nudged by 1
        semitone toward a chord tone rather than fully snapped.
        """
        med_len = _median_length(notes)
        med_vel = _median_velocity(notes)

        # Build a quick section lookup
        sec_map: list[SectionAnalysis | None] = []
        if analysis.sections:
            for note in notes:
                found = None
                for sa in analysis.sections:
                    if sa.section.start_tick <= note.start_tick < sa.section.end_tick:
                        found = sa
                        break
                sec_map.append(found)
        else:
            sec_map = [None] * len(notes)

        for i, (note, cls) in enumerate(zip(notes, classifications)):
            if cls != "wrong":
                continue

            sa = sec_map[i]
            local_chords = sa.chord_regions if sa else analysis.chord_regions
            local_scale = sa.scale_pcs if sa else analysis.scale_pcs

            chord = _chord_at_tick(local_chords, note.start_tick)
            target_pcs = chord.chord_pcs if chord else local_scale

            is_long = note.length_tick >= med_len * (1.5 - self.strictness * 0.8)
            is_loud = note.velocity >= med_vel * (1.2 - self.strictness * 0.3)

            if is_long or is_loud:
                if self.strictness >= 0.7:
                    nearest = _nearest_scale_tone(note.midi_note, target_pcs)
                    diff = nearest - note.midi_note
                    if abs(diff) <= 2:
                        note.midi_note = nearest
                    else:
                        note.midi_note += (1 if diff > 0 else -1)
            else:
                nearest_chord = _nearest_scale_tone(note.midi_note, target_pcs)
                if abs(nearest_chord - note.midi_note) > 3:
                    nearest_chord = _nearest_scale_tone(note.midi_note, local_scale)
                note.midi_note = nearest_chord

        return notes

    # ─── role-specific beautification ──────────────────────────────

    def _beautify_lead(
        self,
        notes: list[NoteEvent],
        analysis: ProjectAnalysis,
        ticks_per_beat: int,
    ) -> list[NoteEvent]:
        """Lead: smooth jitter, trim legato, fill gaps with chord-aware passing tones."""
        # 1) Smooth rapid pitch oscillations
        notes = self._smooth_jitter(notes, ticks_per_beat)

        # 2) Legato trim — prevent overlap into the next note
        for i in range(len(notes) - 1):
            note_end = notes[i].start_tick + notes[i].length_tick
            next_start = notes[i + 1].start_tick
            if note_end > next_start:
                notes[i].length_tick = max(1, next_start - notes[i].start_tick)

        # 3) Chord-aware gap fill — use chord tones at each position
        if self.strictness > 0.3:
            notes = self._fill_gaps_chordaware(notes, analysis)

        # 4) Contour smoothing — large leaps that land on a wrong note
        #    get an intermediate passing tone inserted
        notes = self._smooth_large_leaps(notes, analysis)

        return notes

    def _beautify_bass(
        self,
        notes: list[NoteEvent],
        analysis: ProjectAnalysis,
        ticks_per_beat: int,
    ) -> list[NoteEvent]:
        """Bass: chord-root lock, range enforcement, sparse cleanup."""
        a = analysis

        # 1) Clamp to bass range (MIDI 28–55)
        for note in notes:
            while note.midi_note > 55:
                note.midi_note -= 12
            while note.midi_note < 28:
                note.midi_note += 12

        # 2) At chord changes, snap the first bass note to the chord root
        #    (the hallmark of well-written bass lines)
        med_len = _median_length(notes)
        chord_starts = {r.start_tick: r for r in a.chord_regions}

        for note in notes:
            chord = _chord_at_tick(a.chord_regions, note.start_tick)
            if not chord:
                continue
            pc = note.midi_note % 12

            # Is this note near a chord boundary?  Bass should anchor the root.
            dist_to_chord_start = note.start_tick - chord.start_tick
            near_boundary = dist_to_chord_start <= ticks_per_beat

            if near_boundary and pc != chord.root_pc:
                # Snap to chord root in the bass range
                best = self._nearest_in_range(note.midi_note, {chord.root_pc}, 28, 55)
                if best is not None:
                    note.midi_note = best
            elif pc not in chord.chord_pcs and pc not in a.scale_pcs:
                # Off-chord, off-scale bass note → snap to chord 5th or root
                fifth_pc = (chord.root_pc + 7) % 12
                best = self._nearest_in_range(
                    note.midi_note, {chord.root_pc, fifth_pc}, 28, 55,
                )
                if best is not None:
                    note.midi_note = best

        # 3) Merge truly overlapping same-pitch notes (keep re-articulations)
        cleaned: list[NoteEvent] = []
        for note in notes:
            if cleaned:
                prev = cleaned[-1]
                prev_end = prev.start_tick + prev.length_tick
                if (note.midi_note == prev.midi_note
                        and note.start_tick < prev_end):
                    vel_diff = abs(prev.velocity - note.velocity)
                    if vel_diff <= 10:
                        prev.length_tick = max(
                            prev.length_tick,
                            note.start_tick + note.length_tick - prev.start_tick,
                        )
                        prev.velocity = max(prev.velocity, note.velocity)
                        continue
                    else:
                        prev.length_tick = max(1, note.start_tick - prev.start_tick)
            cleaned.append(note)

        return cleaned

    def _beautify_harmony(
        self,
        notes: list[NoteEvent],
        analysis: ProjectAnalysis,
        ticks_per_beat: int,
    ) -> list[NoteEvent]:
        """Harmony: align onsets, voice-lead through the chord progression."""
        a = analysis
        grid = max(1, ticks_per_beat)

        # 1) Quantise onsets to the beat grid
        for note in notes:
            quantized = round(note.start_tick / grid) * grid
            shift = quantized - note.start_tick
            note.start_tick = max(0, quantized)
            note.length_tick = max(1, note.length_tick - shift)

        notes.sort(key=lambda n: n.start_tick)

        # 2) For each onset group, snap notes to the chord voicing
        #    detected at that tick (not a hardcoded I/IV/V/vi list)
        result: list[NoteEvent] = []
        for _tick, grp in groupby(notes, key=lambda n: n.start_tick):
            chord_notes_list = list(grp)
            tick = chord_notes_list[0].start_tick
            chord = _chord_at_tick(a.chord_regions, tick)

            if chord and chord.confidence > 0.2:
                target_pcs = chord.chord_pcs
            else:
                target_pcs = a.scale_pcs

            for note in chord_notes_list:
                pc = note.midi_note % 12
                if pc not in target_pcs and pc not in a.scale_pcs:
                    note.midi_note = _nearest_scale_tone(note.midi_note, target_pcs)
            result.extend(chord_notes_list)

        # 3) Voice-leading pass — minimise movement between consecutive
        #    onset groups.  For each group, if a note could be moved an
        #    octave closer to the previous group's centroid, do so.
        result.sort(key=lambda n: n.start_tick)
        prev_centroid: float | None = None
        for _tick, grp in groupby(result, key=lambda n: n.start_tick):
            group_list = list(grp)
            if prev_centroid is not None:
                for note in group_list:
                    # Try ±12 to see if a different octave is closer
                    for delta in (0, -12, 12):
                        candidate = note.midi_note + delta
                        if 36 <= candidate <= 96:
                            if abs(candidate - prev_centroid) < abs(note.midi_note - prev_centroid):
                                note.midi_note = candidate
            # Update centroid
            if group_list:
                prev_centroid = sum(n.midi_note for n in group_list) / len(group_list)

        return result

    def _beautify_drums(
        self,
        notes: list[NoteEvent],
        ticks_per_beat: int,
        groove_grid: int | None = None,
    ) -> list[NoteEvent]:
        """Drums: groove-aware quantisation, ghost removal, accent shaping."""
        if not notes:
            return notes

        grid = groove_grid or max(1, ticks_per_beat // 2)

        # 1) Remove ghost hits (velocity < 35) unless strictness is low
        ghost_thresh = int(25 + self.strictness * 20)  # 25..45
        notes = [n for n in notes if n.velocity >= ghost_thresh]

        # 2) Groove-aware quantisation (not rigid snap)
        for note in notes:
            quantized = round(note.start_tick / grid) * grid
            note.start_tick = max(0, quantized)
            note.length_tick = 1

        # 3) De-duplicate same tick + pitch
        seen: set[tuple[int, int]] = set()
        deduped: list[NoteEvent] = []
        for note in notes:
            key = (note.start_tick, note.midi_note)
            if key not in seen:
                seen.add(key)
                deduped.append(note)
        notes = deduped

        # 4) Musical accent pattern — emphasise beat 1, slightly accent 3,
        #    create a natural backbeat feel (beat 2 & 4 snare accent)
        _KICK_MIDI = {35, 36}
        _SNARE_MIDI = {38, 40}
        _HAT_MIDI = {42, 44, 46}

        for note in notes:
            beat = (note.start_tick // ticks_per_beat) % 4
            sub = note.start_tick % ticks_per_beat

            if note.midi_note in _KICK_MIDI:
                # Kick: strong on 1, medium on 3
                if beat == 0:
                    note.velocity = max(note.velocity, 100)
                elif beat == 2:
                    note.velocity = max(note.velocity, 85)
            elif note.midi_note in _SNARE_MIDI:
                # Snare: backbeat emphasis on 2 and 4
                if beat in (1, 3):
                    note.velocity = max(note.velocity, 95)
            elif note.midi_note in _HAT_MIDI:
                # Hi-hat: subtle dynamics — louder on downbeats
                if sub == 0:
                    note.velocity = min(127, max(70, note.velocity))
                else:
                    note.velocity = min(100, max(45, note.velocity))
            else:
                # Other percussion — gentle normalisation
                if beat in (0, 2):
                    note.velocity = min(127, max(75, note.velocity))
                else:
                    note.velocity = min(110, max(55, note.velocity))

            note.velocity = max(30, min(127, note.velocity))

        notes.sort(key=lambda n: n.start_tick)
        return notes

    # ─── inter-track coherence (called from beautify_project) ──────

    def _fix_intertrack_clashes(
        self,
        tracks_notes: list[list[NoteEvent]],
        tracks_roles: list[str],
        analysis: ProjectAnalysis,
        ticks_per_beat: int,
    ) -> list[list[NoteEvent]]:
        """Resolve clashing notes between tracks on the same beat.

        Rules applied:
        * Lead + Harmony: if they collide on the same pitch class and
          the result is a minor 2nd / tritone against the chord, nudge
          the harmony note.
        * Bass + Harmony: if bass note conflicts with harmony chord,
          prefer the bass note (it defines the root).
        """
        a = analysis

        # Find lead and harmony track indices
        lead_idx = next((i for i, r in enumerate(tracks_roles) if r == "lead"), None)
        harm_idx = next((i for i, r in enumerate(tracks_roles) if r == "harmony"), None)
        bass_idx = next((i for i, r in enumerate(tracks_roles) if r == "bass"), None)

        if lead_idx is not None and harm_idx is not None:
            lead_notes = tracks_notes[lead_idx]
            harm_notes = tracks_notes[harm_idx]
            self._resolve_lead_harmony_clash(lead_notes, harm_notes, a, ticks_per_beat)

        if bass_idx is not None and harm_idx is not None:
            bass_notes = tracks_notes[bass_idx]
            harm_notes = tracks_notes[harm_idx]
            self._resolve_bass_harmony_clash(bass_notes, harm_notes, a, ticks_per_beat)

        return tracks_notes

    def _resolve_lead_harmony_clash(
        self,
        lead: list[NoteEvent],
        harmony: list[NoteEvent],
        analysis: ProjectAnalysis,
        ticks_per_beat: int,
    ) -> None:
        """Nudge harmony notes that form harsh intervals with the lead."""
        _HARSH_INTERVALS = {1, 6, 11}  # minor 2nd, tritone, major 7th

        for h_note in harmony:
            chord = _chord_at_tick(analysis.chord_regions, h_note.start_tick)

            for l_note in lead:
                l_end = l_note.start_tick + l_note.length_tick
                h_end = h_note.start_tick + h_note.length_tick
                # Check temporal overlap
                if l_note.start_tick < h_end and h_note.start_tick < l_end:
                    interval = abs(h_note.midi_note - l_note.midi_note) % 12
                    if interval in _HARSH_INTERVALS:
                        # Nudge harmony note to nearest chord tone
                        target_pcs = chord.chord_pcs if chord else analysis.scale_pcs
                        h_note.midi_note = _nearest_scale_tone(h_note.midi_note, target_pcs)
                    break  # only check the first overlapping lead note

    def _resolve_bass_harmony_clash(
        self,
        bass: list[NoteEvent],
        harmony: list[NoteEvent],
        analysis: ProjectAnalysis,
        ticks_per_beat: int,
    ) -> None:
        """If harmony notes form a minor-2nd with the bass, move harmony."""
        for h_note in harmony:
            for b_note in bass:
                b_end = b_note.start_tick + b_note.length_tick
                h_end = h_note.start_tick + h_note.length_tick
                if b_note.start_tick < h_end and h_note.start_tick < b_end:
                    interval = abs(h_note.midi_note - b_note.midi_note) % 12
                    if interval in (1, 11):  # minor 2nd
                        chord = _chord_at_tick(analysis.chord_regions, h_note.start_tick)
                        target = chord.chord_pcs if chord else analysis.scale_pcs
                        h_note.midi_note = _nearest_scale_tone(h_note.midi_note, target)
                    break

    # ─── groove-aware quantisation ─────────────────────────────────

    def _quantize_to_groove(
        self,
        notes: list[NoteEvent],
        groove_grid: int,
        swing_amount: float,
    ) -> list[NoteEvent]:
        """Quantise note starts to the detected grid, preserving swing.

        Straight feel → snap uniformly.
        Swing feel   → off-beat notes are intentionally late; preserve
                        the swing offset rather than snapping rigid.
        """
        if not notes:
            return notes

        grid = max(1, groove_grid)
        strength = 0.4 + self.strictness * 0.5  # 0.4..0.9

        for note in notes:
            nearest = round(note.start_tick / grid) * grid
            beat_idx = round(note.start_tick / grid)

            # Apply swing to off-beats
            target = nearest
            if swing_amount > 0.05 and beat_idx % 2 == 1:
                target = nearest + int(grid * swing_amount)

            # Partial quantise: blend between original and target
            diff = target - note.start_tick
            shift = int(diff * strength)
            note.start_tick = max(0, note.start_tick + shift)
            note.length_tick = max(1, note.length_tick - shift)

        return notes

    # ─── phrase-aware dynamics ─────────────────────────────────────

    def _shape_phrase_dynamics(
        self,
        notes: list[NoteEvent],
        phrases: list[tuple[int, int]],
        role: str,
        loop_aware: bool = False,
    ) -> list[NoteEvent]:
        """Shape velocity curves to give phrases a natural breathing arc.

        Each phrase gets a slight crescendo toward ~60 % of its length
        and a diminuendo toward its end.

        When *loop_aware* is True the **last phrase** keeps a flat arc
        (no diminuendo) so the transition back to the start is smooth
        rather than dying out.
        """
        if not notes or not phrases:
            return notes

        # Role-specific target centres
        targets = {"lead": 95, "bass": 100, "harmony": 80, "drums": 90}
        target = targets.get(role, 90)

        shape_intensity = 0.1 + self.strictness * 0.25  # 0.1..0.35
        last_phrase = phrases[-1] if phrases else None

        for note in notes:
            # Find which phrase this note belongs to
            phrase = None
            for p_start, p_end in phrases:
                if p_start <= note.start_tick < p_end:
                    phrase = (p_start, p_end)
                    break
            if phrase is None:
                continue

            p_start, p_end = phrase
            p_len = max(1, p_end - p_start)
            position = (note.start_tick - p_start) / p_len  # 0..1

            # When loop-aware and this is the last phrase, keep dynamics
            # flat so the music doesn't die before wrapping back.
            if loop_aware and phrase == last_phrase:
                arc = 1.0
            elif position <= 0.6:
                # Rising: 0.85 → 1.1
                arc = 0.85 + (position / 0.6) * 0.25
            else:
                # Falling: 1.1 → 0.8
                arc = 1.1 - ((position - 0.6) / 0.4) * 0.3

            # Blend arc with original velocity
            shaped = note.velocity * (1.0 - shape_intensity) + (note.velocity * arc) * shape_intensity

            # Also gently compress toward role target
            compression = self.strictness * 0.2
            shaped = shaped * (1 - compression) + target * compression

            note.velocity = max(30, min(127, int(shaped)))

        return notes

    # ─── chord-aware gap filling ───────────────────────────────────

    def _fill_gaps_chordaware(
        self,
        notes: list[NoteEvent],
        analysis: ProjectAnalysis,
    ) -> list[NoteEvent]:
        """Fill melodic gaps using chord tones at each position.

        Like the original ``_fill_gaps`` but the passing tones are
        chosen from the *chord* that's active at the fill position,
        giving a much more harmonically grounded result.
        """
        if len(notes) < 3:
            return notes

        med_len = _median_length(notes)
        gap_threshold = max(2, int(med_len * (2.5 - self.strictness * 1.5)))
        filler_len = max(1, int(med_len * 0.5))

        inserts: list[NoteEvent] = []
        for i in range(1, len(notes)):
            prev_end = notes[i - 1].start_tick + notes[i - 1].length_tick
            gap = notes[i].start_tick - prev_end
            if gap < gap_threshold:
                continue

            # Direction from previous contour
            if i >= 3:
                pitches = [notes[i - 3].midi_note, notes[i - 2].midi_note, notes[i - 1].midi_note]
                avg_dir = ((pitches[1] - pitches[0]) + (pitches[2] - pitches[1])) / 2.0
            else:
                avg_dir = notes[i].midi_note - notes[i - 1].midi_note

            if abs(avg_dir) < 0.5:
                continue

            going_up = avg_dir > 0
            cur_pitch = notes[i - 1].midi_note
            target_pitch = notes[i].midi_note
            fill_start = prev_end
            max_fillers = min(4, gap // max(1, filler_len + 1))
            filled = 0

            while filled < max_fillers and fill_start + filler_len < notes[i].start_tick:
                # Use chord tones at this tick position for the filler
                chord = _chord_at_tick(analysis.chord_regions, fill_start)
                if chord:
                    sorted_cpcs = sorted(chord.chord_pcs)
                else:
                    sorted_cpcs = sorted(analysis.scale_pcs)

                next_pitch = self._next_scale_pitch(cur_pitch, sorted_cpcs, going_up)

                if going_up and next_pitch > target_pitch + 4:
                    break
                if not going_up and next_pitch < target_pitch - 4:
                    break

                vel = int(notes[i - 1].velocity * (0.5 + 0.3 * self.strictness))
                vel = max(40, min(120, vel))

                inserts.append(NoteEvent(
                    start_tick=fill_start,
                    length_tick=filler_len,
                    midi_note=next_pitch,
                    velocity=vel,
                ))
                cur_pitch = next_pitch
                fill_start += filler_len + 1
                filled += 1

        notes = list(notes) + inserts
        notes.sort(key=lambda n: n.start_tick)
        return notes

    # ─── leap smoothing ───────────────────────────────────────────

    def _smooth_large_leaps(
        self,
        notes: list[NoteEvent],
        analysis: ProjectAnalysis,
    ) -> list[NoteEvent]:
        """Insert a passing tone when consecutive notes leap > an octave
        and at least one of them is not a chord tone.
        """
        if len(notes) < 2:
            return notes

        inserts: list[NoteEvent] = []
        for i in range(len(notes) - 1):
            leap = abs(notes[i + 1].midi_note - notes[i].midi_note)
            if leap <= 12:
                continue

            # Only intervene if the gap has room for a filler
            gap_start = notes[i].start_tick + notes[i].length_tick
            gap_end = notes[i + 1].start_tick
            if gap_end - gap_start < 2:
                continue

            mid_tick = (gap_start + gap_end) // 2
            mid_pitch = (notes[i].midi_note + notes[i + 1].midi_note) // 2

            # Snap the midpoint to a chord tone
            chord = _chord_at_tick(analysis.chord_regions, mid_tick)
            target_pcs = chord.chord_pcs if chord else analysis.scale_pcs
            mid_pitch = _nearest_scale_tone(mid_pitch, target_pcs)

            vel = int((notes[i].velocity + notes[i + 1].velocity) * 0.4)
            vel = max(40, min(120, vel))

            inserts.append(NoteEvent(
                start_tick=mid_tick,
                length_tick=max(1, gap_end - mid_tick - 1),
                midi_note=mid_pitch,
                velocity=vel,
            ))

        if inserts:
            notes = list(notes) + inserts
            notes.sort(key=lambda n: n.start_tick)

        return notes

    # ─── shared beautification helpers ─────────────────────────────

    def _smooth_jitter(
        self,
        notes: list[NoteEvent],
        ticks_per_beat: int,
    ) -> list[NoteEvent]:
        """Remove rapid pitch oscillations (A→B→A jitter patterns)."""
        if len(notes) < 3:
            return notes

        med_len = _median_length(notes)
        short_thresh = max(1, int(med_len * 0.4))

        result = list(notes)
        changed = True
        passes = 0
        while changed and passes < 3:
            changed = False
            passes += 1
            new_result: list[NoteEvent] = []
            i = 0
            while i < len(result):
                if 0 < i < len(result) - 1:
                    prev = new_result[-1] if new_result else result[i - 1]
                    curr = result[i]
                    nxt = result[i + 1]
                    if (curr.length_tick <= short_thresh
                            and prev.midi_note == nxt.midi_note
                            and abs(curr.midi_note - prev.midi_note) <= 2):
                        if new_result:
                            new_result[-1].length_tick += curr.length_tick
                        changed = True
                        i += 1
                        continue
                new_result.append(NoteEvent(
                    result[i].start_tick, result[i].length_tick,
                    result[i].midi_note, result[i].velocity,
                ))
                i += 1
            result = new_result

        return result

    @staticmethod
    def _nearest_in_range(
        midi_note: int,
        target_pcs: set[int] | frozenset[int],
        lo: int,
        hi: int,
    ) -> int | None:
        """Find the nearest MIDI note with a pitch class in *target_pcs*
        within the range [lo, hi]."""
        best: int | None = None
        best_dist = 999
        for pc in target_pcs:
            for offset in range(-12, 13):
                candidate = midi_note + offset
                if lo <= candidate <= hi and candidate % 12 == pc:
                    if abs(offset) < best_dist:
                        best_dist = abs(offset)
                        best = candidate
        return best

    # ─── internal steps ────────────────────────────────────────────

    def _snap_glitches(
        self,
        notes: list[NoteEvent],
        scale_pcs: set[int],
        blue_pcs: set[int],
    ) -> list[NoteEvent]:
        """Snap short/quiet out-of-scale notes to the nearest scale tone.

        Long or loud chromatic notes are presumed intentional and kept.
        Blue notes (b3 in major, b5) are always preserved.
        """
        if not notes:
            return notes

        med_len = _median_length(notes)
        med_vel = _median_velocity(notes)

        # Thresholds scale with strictness:
        # At strictness=1.0: snap anything shorter than 1.2× median
        # At strictness=0.5: snap only things shorter than 0.4× median
        len_threshold = med_len * (0.2 + self.strictness * 1.0)
        vel_threshold = med_vel * (0.4 + self.strictness * 0.4)

        cleaned: list[NoteEvent] = []
        for note in notes:
            pc = note.midi_note % 12

            if pc in scale_pcs:
                # Already in scale → keep as-is
                cleaned.append(note)
                continue

            if pc in blue_pcs:
                # Blue note → always preserve (retro character)
                cleaned.append(note)
                continue

            # Out of scale – decide: intentional or glitch?
            is_long = note.length_tick >= len_threshold
            is_loud = note.velocity >= vel_threshold

            if is_long or is_loud:
                # Intentional dissonance → keep
                cleaned.append(note)
            else:
                # Glitch → snap to nearest scale tone
                note.midi_note = _nearest_scale_tone(note.midi_note, scale_pcs)
                cleaned.append(note)

        return cleaned

    def _remove_overlaps(self, notes: list[NoteEvent]) -> list[NoteEvent]:
        """Resolve same-pitch overlaps while preserving intentional re-articulations.

        Only truly overlapping notes (second starts *inside* the first) are
        touched.  Even then, if the velocities differ significantly the first
        note is *trimmed* rather than merged so the re-attack is kept.
        Adjacent notes (second starts at or after the first ends) are never
        merged — they represent intentional separate hits.
        """
        if len(notes) < 2:
            return notes

        notes.sort(key=lambda n: (n.midi_note, n.start_tick))
        cleaned: list[NoteEvent] = []

        for note in notes:
            if cleaned and cleaned[-1].midi_note == note.midi_note:
                prev = cleaned[-1]
                prev_end = prev.start_tick + prev.length_tick
                # Only act on genuine overlaps (second note starts inside the first)
                if note.start_tick < prev_end:
                    vel_diff = abs(prev.velocity - note.velocity)
                    if vel_diff <= 10:
                        # Nearly identical velocity → likely a duplicate; merge
                        new_end = max(prev_end, note.start_tick + note.length_tick)
                        prev.length_tick = new_end - prev.start_tick
                        prev.velocity = max(prev.velocity, note.velocity)
                        continue
                    else:
                        # Different velocity → intentional re-articulation; trim first note
                        prev.length_tick = max(1, note.start_tick - prev.start_tick)
            cleaned.append(note)

        # Re-sort by time for downstream consumers
        cleaned.sort(key=lambda n: n.start_tick)
        return cleaned

    def _fill_gaps(
        self,
        notes: list[NoteEvent],
        scale_pcs: set[int],
    ) -> list[NoteEvent]:
        """Fill large melodic gaps with scale-aware passing tones.

        Analyses the direction of the preceding 3 notes and continues
        the contour through the gap using scale tones.

        Only activates when strictness > 0.3 and the gap is
        significantly larger than the median note length.
        """
        if len(notes) < 3:
            return notes

        med_len = _median_length(notes)
        # Only fill gaps bigger than this threshold
        gap_threshold = max(2, int(med_len * (2.5 - self.strictness * 1.5)))
        # How long should filler notes be?  Roughly half the median
        filler_len = max(1, int(med_len * 0.5))

        sorted_pcs = sorted(scale_pcs)

        result: list[NoteEvent] = list(notes)
        inserts: list[NoteEvent] = []

        for i in range(3, len(result)):
            prev_end = result[i - 1].start_tick + result[i - 1].length_tick
            gap = result[i].start_tick - prev_end

            if gap < gap_threshold:
                continue

            # Determine contour direction from previous 3 notes
            pitches = [result[i - 3].midi_note, result[i - 2].midi_note, result[i - 1].midi_note]
            diffs = [pitches[1] - pitches[0], pitches[2] - pitches[1]]
            avg_dir = sum(diffs) / len(diffs)

            if abs(avg_dir) < 0.5:
                # Melody is roughly flat → don't fill
                continue

            going_up = avg_dir > 0
            cur_pitch = pitches[-1]
            target_pitch = result[i].midi_note
            fill_start = prev_end

            # Generate at most a few passing tones (don't overdo it)
            max_fillers = min(4, gap // max(1, filler_len + 1))
            filled = 0

            while filled < max_fillers and fill_start + filler_len < result[i].start_tick:
                # Step to next scale tone in the contour direction
                next_pitch = self._next_scale_pitch(cur_pitch, sorted_pcs, going_up)

                # Safety: don't overshoot the target note
                if going_up and next_pitch > target_pitch + 4:
                    break
                if not going_up and next_pitch < target_pitch - 4:
                    break

                vel = int(result[i - 1].velocity * (0.5 + 0.3 * self.strictness))
                vel = max(40, min(120, vel))

                inserts.append(NoteEvent(
                    start_tick=fill_start,
                    length_tick=filler_len,
                    midi_note=next_pitch,
                    velocity=vel,
                ))

                cur_pitch = next_pitch
                fill_start += filler_len + 1
                filled += 1

        result.extend(inserts)
        result.sort(key=lambda n: n.start_tick)
        return result

    @staticmethod
    def _next_scale_pitch(current: int, sorted_pcs: list[int], going_up: bool) -> int:
        """Return the next scale pitch above/below *current*."""
        pc = current % 12
        octave = current // 12

        if going_up:
            # Find the next higher pitch class in the scale
            for spc in sorted_pcs:
                if spc > pc:
                    return octave * 12 + spc
            # Wrap to next octave
            return (octave + 1) * 12 + sorted_pcs[0]
        else:
            # Find the next lower pitch class
            for spc in reversed(sorted_pcs):
                if spc < pc:
                    return octave * 12 + spc
            # Wrap to previous octave
            return (octave - 1) * 12 + sorted_pcs[-1]


# ─── Remove Gaps ───────────────────────────────────────────────────────

def remove_gaps(notes: list[NoteEvent], min_gap: int = 2) -> list[NoteEvent]:
    """Remove dead-space gaps from a list of notes.

    A "gap" is a contiguous range of ticks where **no** note is sounding
    (no note starts in it and no note sustains through it).  Gaps smaller
    than *min_gap* ticks are ignored — only significant empty stretches
    are collapsed.

    How it works
    ------------
    1. Build a "coverage" set — every tick where at least one note is
       active (from ``start_tick`` to ``start_tick + length_tick - 1``).
    2. Walk from the first active tick to the last, identifying contiguous
       uncovered runs of length ≥ *min_gap*.
    3. Compute a cumulative shift: for each gap, all notes whose
       ``start_tick`` is **after** the gap get shifted left by the gap's
       width.
    4. Apply the shift to each note.

    The result is that notes keep their exact relative timing within
    active regions, but the dead space between them is removed.

    Returns a **new** list (the input is not mutated).
    """
    if not notes:
        return []

    # Copy so we don't mutate
    notes = [NoteEvent(n.start_tick, n.length_tick, n.midi_note, n.velocity)
             for n in notes]
    notes.sort(key=lambda n: n.start_tick)

    # 1) Build coverage — set of ticks where at least one note is active
    first_tick = notes[0].start_tick
    last_tick = max(n.start_tick + n.length_tick for n in notes)

    # Use an interval-merge approach to find gaps efficiently
    # (building a full set for very large tick ranges would be wasteful)
    intervals: list[tuple[int, int]] = []
    for n in notes:
        intervals.append((n.start_tick, n.start_tick + n.length_tick))

    # Merge overlapping intervals
    intervals.sort()
    merged: list[tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    # 2) Identify gaps between merged intervals
    gaps: list[tuple[int, int]] = []  # (gap_start, gap_width)
    for i in range(1, len(merged)):
        gap_start = merged[i - 1][1]
        gap_end = merged[i][0]
        gap_width = gap_end - gap_start
        if gap_width >= min_gap:
            gaps.append((gap_start, gap_width))

    # Also check gap before first note (from tick 0 to first note)
    if first_tick >= min_gap:
        gaps.insert(0, (0, first_tick))

    if not gaps:
        return notes

    # 3) For each note, compute the total shift (sum of all gap widths
    #    that precede the note's start_tick)
    # Pre-compute cumulative shifts at each gap boundary
    cum_shift = 0
    gap_boundaries: list[tuple[int, int]] = []  # (tick_threshold, cum_shift_after)
    for gap_start, gap_width in gaps:
        cum_shift += gap_width
        gap_boundaries.append((gap_start, cum_shift))

    # 4) Apply shift to each note
    for note in notes:
        shift = 0
        for threshold, cs in gap_boundaries:
            if note.start_tick > threshold:
                shift = cs
            else:
                break
        note.start_tick = max(0, note.start_tick - shift)

    notes.sort(key=lambda n: n.start_tick)
    return notes


# ─── Balance Tracks ────────────────────────────────────────────────────

def _track_end_tick(notes: list[NoteEvent]) -> int:
    """Return the last tick at which any note is still sounding."""
    if not notes:
        return 0
    return max(n.start_tick + n.length_tick for n in notes)


def balance_tracks(
    tracks_notes: list[list[NoteEvent]],
    ticks_per_beat: int = 4,
) -> list[list[NoteEvent]]:
    """Extend shorter tracks so every track matches the longest one.

    For each track that ends before the global maximum, the musical
    pattern is intelligently repeated (looped) to fill the remaining
    space.  The looped copies are slightly softer each cycle
    (velocity × 0.92) to avoid a "wall of sound" buildup, and the
    final copy is trimmed so no notes exceed the target length.

    **Algorithm**

    1. Identify the longest track end tick → ``target_end``.
    2. For each shorter track, detect its *pattern length* (the span
       from its first note to its last note's end).
    3. Tile copies of that pattern after the current end until the
       track reaches ``target_end``.
    4. Lightly attenuate each repeated cycle so it doesn't become
       monotonous.

    Returns a **new** list of note-lists (inputs are never mutated).
    """
    if not tracks_notes:
        return tracks_notes

    # Compute the global target end tick (the longest track)
    target_end = max(_track_end_tick(tn) for tn in tracks_notes)
    if target_end <= 0:
        return tracks_notes

    result: list[list[NoteEvent]] = []

    for notes in tracks_notes:
        # Deep-copy so we never mutate caller data
        notes = [NoteEvent(n.start_tick, n.length_tick, n.midi_note, n.velocity)
                 for n in notes]
        notes.sort(key=lambda n: n.start_tick)

        track_end = _track_end_tick(notes)

        if not notes or track_end >= target_end:
            # Already long enough or empty — keep as-is
            result.append(notes)
            continue

        # Detect the pattern span
        first_tick = notes[0].start_tick
        pattern_len = track_end - first_tick
        if pattern_len <= 0:
            result.append(notes)
            continue

        # Build the pattern (notes shifted to start at tick 0)
        pattern = [
            NoteEvent(
                start_tick=n.start_tick - first_tick,
                length_tick=n.length_tick,
                midi_note=n.midi_note,
                velocity=n.velocity,
            )
            for n in notes
        ]

        # Tile the pattern forward from track_end
        cursor = track_end
        cycle = 1
        vel_decay = 0.92  # each repeat is slightly softer

        while cursor < target_end:
            decay_factor = vel_decay ** cycle
            for pn in pattern:
                new_start = cursor + pn.start_tick
                new_end = new_start + pn.length_tick

                # Skip notes that would start past the target
                if new_start >= target_end:
                    continue

                # Trim notes that would overshoot the target
                trimmed_len = pn.length_tick
                if new_end > target_end:
                    trimmed_len = target_end - new_start
                if trimmed_len <= 0:
                    continue

                notes.append(NoteEvent(
                    start_tick=new_start,
                    length_tick=trimmed_len,
                    midi_note=pn.midi_note,
                    velocity=max(30, int(pn.velocity * decay_factor)),
                ))

            cursor += pattern_len
            cycle += 1

        notes.sort(key=lambda n: n.start_tick)
        result.append(notes)

    return result


# ─── Fix Loops (seamless loop transition) ──────────────────────────────

def fix_loops(
    tracks_notes: list[list[NoteEvent]],
    ticks_per_beat: int = 4,
) -> list[list[NoteEvent]]:
    """Make the end→start transition seamless for looping playback.

    This is intentionally *gentle* — the heavy lifting (no end-of-song
    diminuendo, wrapped note classification) is already handled by the
    loop-aware beautify pipeline.  This pass only:

    1. **Rhythmic alignment** — snap track length to a clean bar
       boundary so the loop doesn't drift off-grid.
    2. **Soft velocity crossfade** — slight fade-out in the last beat
       and fade-in on the first beat to soften the seam.

    Previous behaviours (pitch rewriting, pickup notes, bridge notes)
    were removed because they introduced notes that didn't belong in the
    original composition and caused audible chaos at the boundary.

    Returns a **new** list of note-lists (never mutates the inputs).
    """
    if not tracks_notes:
        return tracks_notes

    result: list[list[NoteEvent]] = []

    for notes in tracks_notes:
        notes = [NoteEvent(n.start_tick, n.length_tick, n.midi_note, n.velocity)
                 for n in notes]
        notes.sort(key=lambda n: n.start_tick)

        if len(notes) < 2:
            result.append(notes)
            continue

        track_end = _track_end_tick(notes)

        # ── 1) Rhythmic alignment — snap to bar boundary ─────────
        beat_ticks = max(1, ticks_per_beat)
        bar_ticks = beat_ticks * 4  # assume 4/4

        aligned_end = track_end
        remainder = track_end % bar_ticks
        if remainder != 0:
            aligned_end = track_end + (bar_ticks - remainder)

        # Extend the last note to fill the gap if the boundary is close
        last_note = notes[-1]
        last_note_end = last_note.start_tick + last_note.length_tick
        gap_to_boundary = aligned_end - last_note_end
        if 0 < gap_to_boundary <= bar_ticks:
            last_note.length_tick += gap_to_boundary

        track_end = aligned_end

        # ── 2) Soft velocity crossfade ────────────────────────────
        fade_window = beat_ticks  # 1 beat (gentle)

        fade_out_start = max(0, track_end - fade_window)
        for note in notes:
            if note.start_tick >= fade_out_start:
                progress = (note.start_tick - fade_out_start) / max(1, fade_window)
                fade = 1.0 - (progress * 0.15)  # 1.0 → 0.85
                note.velocity = max(30, int(note.velocity * fade))

        first_tick = notes[0].start_tick
        fade_in_end = first_tick + fade_window
        for note in notes:
            if note.start_tick < fade_in_end:
                progress = (note.start_tick - first_tick) / max(1, fade_window)
                fade = 0.85 + (progress * 0.15)  # 0.85 → 1.0
                note.velocity = max(30, int(note.velocity * fade))

        notes.sort(key=lambda n: n.start_tick)
        result.append(notes)

    return result


def _step_in_scale(midi_note: int, sorted_pcs: list[int], going_up: bool) -> int:
    """Move one scale step up or down from *midi_note*."""
    if not sorted_pcs:
        return midi_note + (1 if going_up else -1)

    pc = midi_note % 12
    octave = midi_note // 12

    if going_up:
        for spc in sorted_pcs:
            if spc > pc:
                return octave * 12 + spc
        return (octave + 1) * 12 + sorted_pcs[0]
    else:
        for spc in reversed(sorted_pcs):
            if spc < pc:
                return octave * 12 + spc
        return (octave - 1) * 12 + sorted_pcs[-1]
