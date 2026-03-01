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

    # ─── beautify (deep theory pass for user-initiated cleanup) ────

    def beautify(
        self,
        notes: list[NoteEvent],
        role: str = "lead",
        ticks_per_beat: int = 4,
    ) -> list[NoteEvent]:
        """Deep music-theory beautification for user-initiated cleanup.

        Goes further than ``fix()`` — applies role-specific rules:

        **All roles:**
        - Detect key/scale from the full note set.
        - Snap out-of-scale glitches (same as ``fix``).
        - Remove overlaps.
        - Quantize note start times to a rhythmic grid.
        - Normalize velocities to a musically coherent range.

        **Lead:**
        - Fill melodic gaps with passing tones.
        - Remove rapid pitch oscillations (jitter smoothing).
        - Ensure melody notes don't clip into each other (legato trim).

        **Harmony:**
        - Snap notes to proper chord voicings within the detected key.
        - Align simultaneous notes to start at the same tick.

        **Bass:**
        - Enforce root-note preference (snap to scale root / 5th).
        - Ensure notes are in a bass-appropriate range (MIDI 28–55).
        - Keep sparse: remove overlapping notes aggressively.

        **Drums:**
        - Quantize to strict grid (rhythmic tightness).
        - Normalize velocities for a consistent groove.
        - Remove ghost hits (very quiet notes).

        Returns a new list of ``NoteEvent`` (never mutates the input).
        """
        if not notes:
            return notes

        # Work on copies
        notes = [NoteEvent(n.start_tick, n.length_tick, n.midi_note, n.velocity)
                 for n in notes]
        notes.sort(key=lambda n: n.start_tick)

        if role == "drums":
            return self._beautify_drums(notes, ticks_per_beat)

        # ── detect key ────────────────────────────────────────────
        root, quality, scale_pcs = _detect_key(notes)
        blue_pcs = _get_blue_pcs(root, quality)

        # ── snap glitches ─────────────────────────────────────────
        notes = self._snap_glitches(notes, scale_pcs, blue_pcs)

        # ── role-specific processing ──────────────────────────────
        if role == "bass":
            notes = self._beautify_bass(notes, root, scale_pcs, ticks_per_beat)
        elif role == "harmony":
            notes = self._beautify_harmony(notes, root, quality, scale_pcs, ticks_per_beat)
        else:
            notes = self._beautify_lead(notes, scale_pcs, ticks_per_beat)

        # ── common final passes ───────────────────────────────────
        notes = self._remove_overlaps(notes)
        notes = self._quantize_starts(notes, ticks_per_beat)
        notes = self._normalize_velocities(notes, role)
        notes = self._remove_overlaps(notes)

        notes.sort(key=lambda n: n.start_tick)
        return notes

    # ─── role-specific beautification internals ────────────────────

    def _beautify_lead(
        self,
        notes: list[NoteEvent],
        scale_pcs: set[int],
        ticks_per_beat: int,
    ) -> list[NoteEvent]:
        """Lead-specific: smooth jitter, trim legato, fill gaps."""
        # 1) Smooth rapid pitch oscillations
        notes = self._smooth_jitter(notes, ticks_per_beat)

        # 2) Legato trim — prevent notes from bleeding into the next note
        for i in range(len(notes) - 1):
            note_end = notes[i].start_tick + notes[i].length_tick
            next_start = notes[i + 1].start_tick
            if note_end > next_start:
                # Trim current note to end just before the next starts
                notes[i].length_tick = max(1, next_start - notes[i].start_tick)

        # 3) Fill melodic gaps with passing tones
        if self.strictness > 0.3:
            notes = self._fill_gaps(notes, scale_pcs)

        return notes

    def _beautify_bass(
        self,
        notes: list[NoteEvent],
        root: int,
        scale_pcs: set[int],
        ticks_per_beat: int,
    ) -> list[NoteEvent]:
        """Bass-specific: enforce range, prefer root/5th, keep sparse."""
        # 1) Clamp to bass range (MIDI 28–55)
        for note in notes:
            while note.midi_note > 55:
                note.midi_note -= 12
            while note.midi_note < 28:
                note.midi_note += 12

        # 2) Snap short out-of-scale notes to root or 5th
        fifth_pc = (root + 7) % 12
        strong_pcs = {root, fifth_pc}
        med_len = _median_length(notes)

        for note in notes:
            pc = note.midi_note % 12
            if pc not in scale_pcs and note.length_tick < med_len * 0.6:
                # Snap to root or 5th (whichever is closer)
                best = note.midi_note
                best_dist = 999
                for target_pc in strong_pcs:
                    for offset in range(-6, 7):
                        candidate = note.midi_note + offset
                        if candidate % 12 == target_pc and 28 <= candidate <= 55:
                            if abs(offset) < best_dist:
                                best_dist = abs(offset)
                                best = candidate
                note.midi_note = best

        # 3) Remove rapid repeated notes (bass should be sparse)
        cleaned: list[NoteEvent] = []
        for note in notes:
            if cleaned:
                prev = cleaned[-1]
                prev_end = prev.start_tick + prev.length_tick
                if (note.midi_note == prev.midi_note
                        and note.start_tick - prev_end <= 1):
                    # Merge into previous
                    prev.length_tick = max(
                        prev.length_tick,
                        note.start_tick + note.length_tick - prev.start_tick,
                    )
                    prev.velocity = max(prev.velocity, note.velocity)
                    continue
            cleaned.append(note)

        return cleaned

    def _beautify_harmony(
        self,
        notes: list[NoteEvent],
        root: int,
        quality: str,
        scale_pcs: set[int],
        ticks_per_beat: int,
    ) -> list[NoteEvent]:
        """Harmony-specific: align chord onsets, clean voicings."""
        # Build chord intervals from the detected scale
        if quality == "major":
            # Common triads: I, IV, V, vi
            chord_intervals = [
                {0, 4, 7},        # I
                {5, 9, 0},        # IV
                {7, 11, 2},       # V
                {9, 0, 4},        # vi
            ]
        else:
            # Minor: i, iv, v, III
            chord_intervals = [
                {0, 3, 7},        # i
                {5, 8, 0},        # iv
                {7, 10, 2},       # v
                {3, 7, 10},       # III
            ]
        # Resolve to absolute pitch classes
        chords_pcs = [{(root + i) % 12 for i in ch} for ch in chord_intervals]

        # 1) Align simultaneous-ish notes to the same start tick
        grid = max(1, ticks_per_beat)
        for note in notes:
            quantized = round(note.start_tick / grid) * grid
            shift = quantized - note.start_tick
            note.start_tick = max(0, quantized)
            note.length_tick = max(1, note.length_tick - shift)

        # 2) For each onset group, snap notes to the closest matching chord
        from itertools import groupby
        notes.sort(key=lambda n: n.start_tick)
        result: list[NoteEvent] = []

        for tick, group in groupby(notes, key=lambda n: n.start_tick):
            chord_notes = list(group)
            pcs_present = {n.midi_note % 12 for n in chord_notes}

            # Find best matching chord
            best_chord = None
            best_overlap = -1
            for ch_pcs in chords_pcs:
                overlap = len(pcs_present & ch_pcs)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_chord = ch_pcs

            if best_chord and best_overlap > 0:
                for note in chord_notes:
                    pc = note.midi_note % 12
                    if pc not in best_chord and pc not in scale_pcs:
                        note.midi_note = _nearest_scale_tone(note.midi_note, best_chord)

            result.extend(chord_notes)

        return result

    def _beautify_drums(
        self,
        notes: list[NoteEvent],
        ticks_per_beat: int,
    ) -> list[NoteEvent]:
        """Drums: tight quantization, remove ghosts, normalize velocity."""
        if not notes:
            return notes

        # 1) Remove ghost hits (velocity < 40)
        notes = [n for n in notes if n.velocity >= 40]

        # 2) Strict grid quantization
        grid = max(1, ticks_per_beat // 2)
        for note in notes:
            note.start_tick = round(note.start_tick / grid) * grid
            note.length_tick = 1  # drums are always single-tick hits

        # 3) Remove duplicate hits on the same tick+pitch
        seen: set[tuple[int, int]] = set()
        deduped: list[NoteEvent] = []
        for note in notes:
            key = (note.start_tick, note.midi_note)
            if key not in seen:
                seen.add(key)
                deduped.append(note)
        notes = deduped

        # 4) Normalize velocities — accent beats 1 and 3 a bit
        for note in notes:
            beat_in_bar = (note.start_tick // ticks_per_beat) % 4
            if beat_in_bar in (0, 2):
                note.velocity = min(127, max(80, note.velocity))
            else:
                note.velocity = min(110, max(60, note.velocity))

        notes.sort(key=lambda n: n.start_tick)
        return notes

    # ─── shared beautification helpers ─────────────────────────────

    def _smooth_jitter(
        self,
        notes: list[NoteEvent],
        ticks_per_beat: int,
    ) -> list[NoteEvent]:
        """Remove rapid pitch oscillations (e.g. C→D→C→D jitter).

        A note is considered "jitter" if it is short and its pitch
        is immediately repeated by a neighbouring note's pitch (A→B→A pattern).
        """
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
                    # A→B→A pattern where B is short
                    if (curr.length_tick <= short_thresh
                            and prev.midi_note == nxt.midi_note
                            and abs(curr.midi_note - prev.midi_note) <= 2):
                        # Absorb B into A (extend previous)
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

    def _quantize_starts(
        self,
        notes: list[NoteEvent],
        ticks_per_beat: int,
    ) -> list[NoteEvent]:
        """Quantize note starts to the nearest half-beat grid."""
        grid = max(1, ticks_per_beat // 2)
        for note in notes:
            old_start = note.start_tick
            quantized = round(old_start / grid) * grid
            shift = quantized - old_start
            note.start_tick = max(0, quantized)
            note.length_tick = max(1, note.length_tick - shift)
        return notes

    def _normalize_velocities(
        self,
        notes: list[NoteEvent],
        role: str,
    ) -> list[NoteEvent]:
        """Gently compress velocities toward the role's sweet spot.

        Lead: centre around 95  ·  Bass: centre around 100
        Harmony: centre around 80  ·  Drums: handled separately
        """
        if not notes:
            return notes

        targets = {"lead": 95, "bass": 100, "harmony": 80, "drums": 90}
        target = targets.get(role, 90)

        # Compute current median velocity
        vels = sorted(n.velocity for n in notes)
        med = vels[len(vels) // 2]

        if med == 0:
            return notes

        # Ratio to shift median toward target (gentle, capped)
        ratio = target / med
        ratio = max(0.7, min(1.4, ratio))  # don't overdo it

        compression = 0.3 + self.strictness * 0.4  # 0.3..0.7

        for note in notes:
            # Blend original velocity toward target
            adjusted = note.velocity * ratio
            note.velocity = int(
                note.velocity * (1 - compression) + adjusted * compression
            )
            note.velocity = max(30, min(127, note.velocity))

        return notes

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
        """Merge or trim notes with the same pitch that overlap in time."""
        if len(notes) < 2:
            return notes

        notes.sort(key=lambda n: (n.midi_note, n.start_tick))
        cleaned: list[NoteEvent] = []

        for note in notes:
            if cleaned and cleaned[-1].midi_note == note.midi_note:
                prev = cleaned[-1]
                prev_end = prev.start_tick + prev.length_tick
                # Overlapping or adjacent → merge
                if note.start_tick <= prev_end + 1:
                    new_end = max(prev_end, note.start_tick + note.length_tick)
                    prev.length_tick = new_end - prev.start_tick
                    prev.velocity = max(prev.velocity, note.velocity)
                    continue
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

    When a piece loops, the jump from the last tick back to the first
    tick often sounds jarring: sudden silence, abrupt pitch jumps,
    velocity spikes, or rhythmic discontinuity.  This function smooths
    the transition so the loop feels like a continuous performance.

    **Techniques applied (per track):**

    1. **Velocity crossfade** — notes in the last ~2 beats are gently
       faded out while notes in the first ~2 beats are gently faded in.
       This prevents the hard cut.

    2. **Tail extension** — if the last note ends well before the loop
       boundary, a "sustain tail" note is added to carry the sound
       into the restart.

    3. **Pitch contour bridging** — the last few notes' pitch direction
       is analysed.  If the ending contour clashes with the opening
       (e.g., melody rises at the end but starts on a note far below),
       the final notes are adjusted to create a stepwise approach
       toward the first note, using scale-aware movement.

    4. **Pickup / anacrusis note** — if there's a gap between the loop
       end and the first note of the track, a quiet "pickup" note is
       placed just before tick 0 (using the track's opening pitch)
       to bridge the silence.

    5. **Rhythmic alignment** — if the track's total length doesn't
       land on a clean beat boundary, the last note is trimmed or
       extended to align with the nearest beat, preventing the loop
       from drifting off-grid.

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
        first_tick = notes[0].start_tick

        # Detect key for pitch-aware operations
        root, quality, scale_pcs = _detect_key(notes)
        sorted_pcs = sorted(scale_pcs)

        # ── 1) Rhythmic alignment — snap track end to beat boundary ──
        beat_ticks = max(1, ticks_per_beat)
        bar_ticks = beat_ticks * 4  # assume 4/4

        # Round up to next bar boundary for clean loops
        aligned_end = track_end
        remainder = track_end % bar_ticks
        if remainder != 0:
            aligned_end = track_end + (bar_ticks - remainder)

        # Extend the last note to fill the gap if it's close
        last_note = notes[-1]
        last_note_end = last_note.start_tick + last_note.length_tick
        gap_to_boundary = aligned_end - last_note_end

        if 0 < gap_to_boundary <= bar_ticks:
            # Extend last note to reach the boundary (sustain)
            last_note.length_tick += gap_to_boundary

        track_end = aligned_end

        # ── 2) Velocity crossfade — fade out tail, fade in head ──────
        fade_window = beat_ticks * 2  # 2 beats

        # Fade-out: notes starting in the last fade_window ticks
        fade_out_start = max(0, track_end - fade_window)
        for note in notes:
            if note.start_tick >= fade_out_start:
                # Linear fade: full volume at fade_out_start → 60% at track_end
                progress = (note.start_tick - fade_out_start) / max(1, fade_window)
                fade = 1.0 - (progress * 0.4)  # 1.0 → 0.6
                note.velocity = max(30, int(note.velocity * fade))

        # Fade-in: notes starting in the first fade_window ticks
        fade_in_end = first_tick + fade_window
        for note in notes:
            if note.start_tick < fade_in_end:
                progress = (note.start_tick - first_tick) / max(1, fade_window)
                fade = 0.6 + (progress * 0.4)  # 0.6 → 1.0
                note.velocity = max(30, int(note.velocity * fade))

        # ── 3) Pitch contour bridging ────────────────────────────────
        #    Analyse last 3-4 notes and first note; adjust if the
        #    interval is too large (> 7 semitones)

        first_note = notes[0]
        # Get the last few pitched notes (skip very short ones)
        tail_notes = [n for n in notes[-6:] if n.length_tick >= 1]
        if len(tail_notes) >= 2:
            final_pitch = tail_notes[-1].midi_note
            opening_pitch = first_note.midi_note

            interval = abs(final_pitch - opening_pitch)

            if interval > 7:
                # The jump is too large — create a stepwise approach
                # Adjust the last 2-3 notes to walk toward the opening pitch
                going_down = final_pitch > opening_pitch
                n_adjust = min(3, len(tail_notes) - 1)

                for i in range(n_adjust):
                    idx = len(tail_notes) - 1 - i
                    if idx < 0:
                        break
                    note = tail_notes[idx]
                    # How far should this note be from the target?
                    # Last note → closest, earlier notes → farther
                    steps_away = i + 1
                    target = opening_pitch
                    # Step away from target by scale steps
                    cur = target
                    for _ in range(steps_away):
                        cur = _step_in_scale(
                            cur, sorted_pcs,
                            going_up=not going_down,
                        )

                    # Only adjust if it brings the note closer to the target
                    old_dist = abs(note.midi_note - opening_pitch)
                    new_dist = abs(cur - opening_pitch)
                    if new_dist < old_dist:
                        note.midi_note = cur

        # ── 4) Tail extension — bridge silence at end ────────────────
        last_note_end = _track_end_tick(notes)
        silence_at_end = track_end - last_note_end

        if silence_at_end > beat_ticks:
            # Add a sustain/bridge note using the opening pitch
            bridge_pitch = first_note.midi_note
            # Use a quiet velocity for a gentle lead-in
            bridge_vel = max(30, int(first_note.velocity * 0.5))
            notes.append(NoteEvent(
                start_tick=last_note_end,
                length_tick=silence_at_end,
                midi_note=bridge_pitch,
                velocity=bridge_vel,
            ))

        # ── 5) Pickup / anacrusis — fill silence before first note ───
        if first_tick > 0 and first_tick >= beat_ticks:
            # There's a gap before the first note — add a quiet pickup
            pickup_pitch = first_note.midi_note
            # Step one scale tone below for an approach effect
            pickup_pitch = _step_in_scale(pickup_pitch, sorted_pcs, going_up=False)
            pickup_start = max(0, first_tick - beat_ticks)
            pickup_len = first_tick - pickup_start
            pickup_vel = max(30, int(first_note.velocity * 0.45))

            notes.append(NoteEvent(
                start_tick=pickup_start,
                length_tick=pickup_len,
                midi_note=pickup_pitch,
                velocity=pickup_vel,
            ))

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
