"""Audio-to-MIDI transcription for Koke16-Bit Studio.

Uses **CQT** (Constant-Q Transform) for polyphonic pitch detection –
every CQT bin maps to one MIDI semitone, so the transform acts like a
spectrogram whose rows *are* piano-roll notes.  Combined with HPSS for
harmonic/percussive separation this produces faithful multi-track
transcriptions of real music.
"""

from __future__ import annotations

import math
import wave
from typing import Callable

import librosa
import numpy as np

from daw.models import NoteEvent
from daw.theory import SmartTheoryFixer


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TranscriptionCancelled(Exception):
    """Raised when the user requests cancellation."""


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(file_path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load audio via librosa (WAV, MP3, FLAC, OGG …).

    Falls back to the manual WAV reader when librosa/soundfile is unavailable.
    """
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0:
            y = y / peak
        return y.astype(np.float32), int(sr)
    except Exception:
        return load_wav_mono(file_path, target_sr)


def load_wav_mono(
    file_path: str, target_sample_rate: int = 22050
) -> tuple[np.ndarray, int]:
    """Manual WAV reader – 8/16/24/32-bit PCM."""
    with wave.open(file_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        src_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sample_width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        vals = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        vals = vals - ((vals & 0x800000) << 1)
        data = vals.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("Unsupported WAV sample width")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    if src_rate != target_sample_rate and data.size > 8:
        tgt_len = max(1, int(data.size * (target_sample_rate / src_rate)))
        data = np.interp(
            np.linspace(0, 1, tgt_len, endpoint=False),
            np.linspace(0, 1, data.size, endpoint=False),
            data,
        ).astype(np.float32)
        src_rate = target_sample_rate

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0:
        data = data / peak
    return data.astype(np.float32), src_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(audio: np.ndarray) -> np.ndarray:
    y = audio.astype(np.float32)
    y -= float(np.mean(y))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y /= peak
    return y


def _time_to_tick(t: float, duration_s: float, total_ticks: int) -> int:
    return max(0, min(total_ticks - 1,
                      int(round(t / max(1e-6, duration_s) * total_ticks))))


def _compute_total_ticks(duration_s: float, bpm: int, ticks_per_beat: int) -> int:
    """BPM-aware tick count so temporal detail scales with clip length."""
    beats = duration_s * bpm / 60.0
    return max(32, int(round(beats * ticks_per_beat)))


# ---------------------------------------------------------------------------
# CQT → NoteEvents  (polyphonic, the core of the new approach)
# ---------------------------------------------------------------------------

_MIDI_LO = 24   # C1
_MIDI_HI = 96   # C7
_HOP = 512


def _find_active_regions(
    active_1d: np.ndarray,
    max_gap: int = 3,
    min_frames: int = 2,
) -> list[tuple[int, int]]:
    """Find contiguous *True* runs, bridging gaps ≤ *max_gap* frames."""
    regions: list[tuple[int, int]] = []
    start: int | None = None
    last_active: int | None = None

    for i in range(len(active_1d)):
        if active_1d[i]:
            if start is None:
                start = i
            last_active = i
        else:
            if start is not None and last_active is not None and (i - last_active) > max_gap:
                end = last_active + 1
                if end - start >= min_frames:
                    regions.append((start, end))
                start = None
                last_active = None

    if start is not None and last_active is not None:
        end = last_active + 1
        if end - start >= min_frames:
            regions.append((start, end))

    return regions


def _cqt_to_notes(
    C_mag: np.ndarray,
    max_mag: float,
    midi_lo: int,
    frame_times: np.ndarray,
    duration_s: float,
    total_ticks: int,
    global_threshold: float,
    frame_threshold: np.ndarray,
    max_gap: int = 3,
    min_frames: int = 2,
) -> list[NoteEvent]:
    """Walk the thresholded CQT magnitude and emit NoteEvents."""
    n_bins, n_frames = C_mag.shape

    # Combined threshold: must beat BOTH the noise floor and 25 % of frame peak
    combined = np.maximum(global_threshold, frame_threshold[np.newaxis, :])
    active = C_mag > combined

    notes: list[NoteEvent] = []
    for bin_idx in range(n_bins):
        midi_note = midi_lo + bin_idx
        regions = _find_active_regions(active[bin_idx], max_gap=max_gap, min_frames=min_frames)

        for start_f, end_f in regions:
            t0 = float(frame_times[min(start_f, n_frames - 1)])
            t1 = float(frame_times[min(end_f - 1, n_frames - 1)])

            tick_start = _time_to_tick(t0, duration_s, total_ticks)
            tick_end = _time_to_tick(t1, duration_s, total_ticks)
            length = max(1, tick_end - tick_start + 1)

            seg_mag = C_mag[bin_idx, start_f:end_f]
            avg_mag = float(np.mean(seg_mag)) / max(max_mag, 1e-6)
            vel = int(max(50, min(127, 55 + avg_mag * 90)))

            notes.append(NoteEvent(
                start_tick=tick_start,
                length_tick=length,
                midi_note=midi_note,
                velocity=vel,
            ))

    return notes


# ---------------------------------------------------------------------------
# CQT cleaning  (peak picking + harmonic suppression + polyphony limit)
# ---------------------------------------------------------------------------

def _clean_cqt(C_mag: np.ndarray) -> np.ndarray:
    """Remove spectral leakage and harmonic overtone contamination.

    1) **Peak picking** – a bin survives only if it is a local maximum
       in the frequency dimension (± 1 bin).  This kills spectral leakage
       that "smears" energy across neighbouring semitones.
    2) **Harmonic suppression** – process bins low → high.  When a bin
       has significant energy, attenuate the bins at its harmonic
       overtone positions (+12, +19, +24, +28, +31, +34, +36 semitones)
       *if* the overtone is weaker than the fundamental (suggesting it
       really is just an overtone, not an independently played note).
    """
    n_bins, n_frames = C_mag.shape
    out = C_mag.copy()

    # ── 1) Spectral peak picking ──────────────────────────────────
    if n_bins > 2:
        peak_mask = np.ones_like(out, dtype=bool)
        # Interior bins: must be >= both neighbours
        peak_mask[1:-1, :] = (
            (out[1:-1, :] >= out[:-2, :]) &
            (out[1:-1, :] >= out[2:, :])
        )
        # Edge bins: compare with single neighbour
        peak_mask[0, :] = out[0, :] >= out[1, :]
        peak_mask[-1, :] = out[-1, :] >= out[-2, :]
        out[~peak_mask] = 0.0

    # ── 2) Harmonic suppression (low → high) ─────────────────────
    # Semitone offsets of the first 7 harmonics above the fundamental
    harm_offsets = [12, 19, 24, 28, 31, 34, 36]

    for b in range(n_bins):
        fund = out[b, :]
        active_frames = fund > 0
        if not np.any(active_frames):
            continue
        for ho in harm_offsets:
            hb = b + ho
            if hb >= n_bins:
                break
            # Suppress overtone only when it is clearly weaker than
            # the fundamental (ratio < 0.8).  If the overtone is
            # comparable in strength it is likely an independently
            # played note and we leave it alone.
            suppress = active_frames & (out[hb, :] < fund * 0.8)
            out[hb, suppress] *= 0.05

    return out


def _limit_polyphony(C_mag: np.ndarray, max_voices: int = 6) -> np.ndarray:
    """Keep only the *max_voices* loudest bins per frame.

    Prevents the piano roll from becoming an unplayable wall of notes.
    """
    n_bins, n_frames = C_mag.shape
    if n_bins <= max_voices:
        return C_mag
    out = C_mag.copy()
    # Sort magnitudes descending per frame; the Nth value is our cutoff
    sorted_desc = np.sort(out, axis=0)[::-1, :]
    cutoff = sorted_desc[max_voices - 1, :]          # shape (n_frames,)
    # Zero bins that fall below the per-frame cutoff
    out[out < cutoff[np.newaxis, :]] = 0.0
    return out


# ---------------------------------------------------------------------------
# Note consolidation  (remove fragments, merge neighbours, quantize)
# ---------------------------------------------------------------------------

def _consolidate_notes(
    notes: list[NoteEvent],
    ticks_per_beat: int = 4,
) -> list[NoteEvent]:
    """Clean up raw CQT note output to produce musically coherent results.

    Steps:
    0. Stabilise rapid pitch oscillations (absorb jitter into neighbours).
    1. Remove very short notes (< 2 ticks) — these are click/noise artefacts.
    2. Merge same-pitch notes that overlap or are separated by ≤ 1 tick gap.
    3. Absorb ±1 semitone fragments into stronger neighbours (CQT bin leakage).
    4. Quantize note starts to the nearest half-beat grid.
    5. Re-merge after quantization (quantizing can create new overlaps).
    6. Final cleanup — remove any remaining micro-notes after merging.
    """
    if not notes:
        return notes

    # ── 0) Pitch stabilisation (absorb jitter / oscillations) ─────────
    notes = _stabilize_pitch_sequence(notes, ticks_per_beat)

    # ── 1) Remove tiny fragments ──────────────────────────────────────
    notes = [n for n in notes if n.length_tick >= 2]
    if not notes:
        return notes

    # ── 2) Merge same-pitch overlaps ──────────────────────────────────
    notes = _merge_same_pitch(notes, max_gap=1)

    # ── 3) Absorb ±1 semitone leakage into stronger neighbours ───────
    notes = _absorb_semitone_leakage(notes)

    # ── 4) Quantize note starts to half-beat grid ─────────────────────
    grid = max(1, ticks_per_beat // 2)  # e.g. 2 for ticks_per_beat=4
    for note in notes:
        old_start = note.start_tick
        quantized = round(old_start / grid) * grid
        shift = quantized - old_start
        note.start_tick = max(0, quantized)
        # Preserve note end position as closely as possible
        note.length_tick = max(1, note.length_tick - shift)

    # ── 5) Merge again after quantization ─────────────────────────────
    notes = _merge_same_pitch(notes, max_gap=0)

    # ── 6) Final cleanup — remove micro-notes that survived merging ───
    if notes:
        lengths = sorted(n.length_tick for n in notes)
        med = float(lengths[len(lengths) // 2])
        # Drop notes shorter than 25 % of the median (clearly artefacts)
        cutoff = max(1, int(med * 0.25))
        notes = [n for n in notes if n.length_tick >= cutoff]

    # Final sort
    notes.sort(key=lambda n: n.start_tick)
    return notes


def _merge_same_pitch(notes: list[NoteEvent], max_gap: int = 1) -> list[NoteEvent]:
    """Merge notes with the same MIDI pitch that overlap or abut within *max_gap* ticks."""
    if not notes:
        return notes

    notes = sorted(notes, key=lambda n: (n.midi_note, n.start_tick))
    merged: list[NoteEvent] = []

    for note in notes:
        if merged and merged[-1].midi_note == note.midi_note:
            prev = merged[-1]
            prev_end = prev.start_tick + prev.length_tick
            if note.start_tick <= prev_end + max_gap:
                new_end = max(prev_end, note.start_tick + note.length_tick)
                prev.length_tick = new_end - prev.start_tick
                prev.velocity = max(prev.velocity, note.velocity)
                continue
        merged.append(NoteEvent(
            start_tick=note.start_tick,
            length_tick=note.length_tick,
            midi_note=note.midi_note,
            velocity=note.velocity,
        ))

    return merged


def _absorb_semitone_leakage(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Remove notes that are ±1 semitone from a stronger overlapping note.

    CQT often leaks energy to adjacent bins, creating phantom notes
    one semitone above or below the real pitch.  If two notes overlap
    in time and are ±1 semitone apart, the weaker/shorter one is removed.
    """
    if len(notes) < 2:
        return notes

    notes = sorted(notes, key=lambda n: (-n.length_tick * n.velocity, n.start_tick))

    survivors: list[NoteEvent] = []
    removed: set[int] = set()

    # Index by id for removal tracking
    indexed = list(enumerate(notes))

    for i, note_a in indexed:
        if i in removed:
            continue
        survivors.append(note_a)
        a_start = note_a.start_tick
        a_end = a_start + note_a.length_tick
        a_strength = note_a.length_tick * note_a.velocity

        for j, note_b in indexed:
            if j <= i or j in removed:
                continue
            # Check if ±1 semitone
            if abs(note_b.midi_note - note_a.midi_note) != 1:
                continue
            # Check temporal overlap
            b_start = note_b.start_tick
            b_end = b_start + note_b.length_tick
            overlap = min(a_end, b_end) - max(a_start, b_start)
            if overlap <= 0:
                continue
            # The weaker note is absorbed if overlap is significant
            b_strength = note_b.length_tick * note_b.velocity
            if b_strength < a_strength * 0.8:
                removed.add(j)

    return survivors


# ---------------------------------------------------------------------------
# Pitch stabilisation  (eliminate rapid jitter / oscillations)
# ---------------------------------------------------------------------------

def _stabilize_pitch_sequence(
    notes: list[NoteEvent],
    ticks_per_beat: int = 4,
) -> list[NoteEvent]:
    """Smooth out rapid pitch oscillations caused by transcription artefacts.

    When a short note is sandwiched between longer notes of similar pitch,
    it is almost certainly a pitch-detection wobble (vibrato, noise,
    spectral leakage).  This function absorbs those short outliers into
    their stronger neighbours, then re-merges same-pitch runs.

    The algorithm applies music-theory reasoning:
    * Notes shorter than a 16th-note equivalent are candidates for absorption.
    * A candidate is absorbed if its pitch is within ±2 semitones of the
      dominant pitch in its local time window.
    * The dominant pitch is determined by duration-weighted voting among
      the candidate's immediate neighbours.
    * Multiple passes ensure cascading jitter is fully resolved.
    """
    if len(notes) < 3:
        return notes

    notes = sorted(notes, key=lambda n: n.start_tick)

    # Minimum length to be considered "stable" (a 16th note)
    min_stable = max(1, ticks_per_beat // 4) if ticks_per_beat >= 4 else 1

    # Use median note length as the baseline — notes significantly shorter
    # than this are suspicious.
    lengths = sorted(n.length_tick for n in notes)
    med_len = float(lengths[len(lengths) // 2])
    short_threshold = max(min_stable, int(med_len * 0.4))

    for _pass in range(3):
        changed = False
        stable_flags = [n.length_tick >= short_threshold for n in notes]

        result: list[NoteEvent] = []
        for i, note in enumerate(notes):
            if stable_flags[i]:
                result.append(NoteEvent(
                    note.start_tick, note.length_tick,
                    note.midi_note, note.velocity,
                ))
                continue

            # Find nearest stable neighbours (left and right)
            left: NoteEvent | None = None
            right: NoteEvent | None = None
            for j in range(i - 1, -1, -1):
                if stable_flags[j]:
                    left = notes[j]
                    break
            for j in range(i + 1, len(notes)):
                if stable_flags[j]:
                    right = notes[j]
                    break

            # Determine dominant pitch from neighbours
            target_pitch: int | None = None
            if left and right:
                if left.midi_note == right.midi_note:
                    # Sandwiched between same pitch — clearly an artefact
                    target_pitch = left.midi_note
                elif abs(note.midi_note - left.midi_note) <= 2:
                    # Closer to left neighbour
                    target_pitch = left.midi_note
                elif abs(note.midi_note - right.midi_note) <= 2:
                    target_pitch = right.midi_note
                else:
                    # Pick the one with more "weight" (duration × velocity)
                    l_w = left.length_tick * left.velocity
                    r_w = right.length_tick * right.velocity
                    cand = left if l_w >= r_w else right
                    if abs(note.midi_note - cand.midi_note) <= 3:
                        target_pitch = cand.midi_note
            elif left and abs(note.midi_note - left.midi_note) <= 2:
                target_pitch = left.midi_note
            elif right and abs(note.midi_note - right.midi_note) <= 2:
                target_pitch = right.midi_note

            if target_pitch is not None and target_pitch != note.midi_note:
                changed = True

            result.append(NoteEvent(
                note.start_tick,
                note.length_tick,
                target_pitch if target_pitch is not None else note.midi_note,
                note.velocity,
            ))

        notes = result
        if not changed:
            break

    # After snapping pitches, merge same-pitch neighbours that now abut
    notes = _merge_same_pitch(notes, max_gap=1)
    return notes


# ---------------------------------------------------------------------------
# Drum extraction  (onset-based, from percussive HPSS residual)
# ---------------------------------------------------------------------------

def _extract_drums(
    y_perc: np.ndarray,
    sr: int,
    total_ticks: int,
    duration_s: float,
    min_tick_gap: int = 2,
) -> list[NoteEvent]:
    if y_perc.size < sr // 4:
        return []

    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr)

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, units="frames",
        delta=0.15,    # moderate – higher than default 0.07 to skip weak noise
        wait=4,        # minimum 4 frames (~90 ms) between onsets
        backtrack=True,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    events: list[NoteEvent] = []
    last_tick = -999
    for t in onset_times:
        tick = _time_to_tick(float(t), duration_s, total_ticks)

        # Enforce minimum spacing between drum hits
        if tick - last_tick < min_tick_gap:
            continue

        centre = int(float(t) * sr)
        seg = y_perc[max(0, centre - sr // 20): min(len(y_perc), centre + sr // 10)]

        # Only keep hits with sufficient energy
        if seg.size > 256:
            rms = float(np.sqrt(np.mean(seg ** 2)))
            if rms < 0.015:  # discard very quiet ghost hits
                continue
        else:
            continue

        midi_note, vel = 38, 100
        centroid = float(np.mean(
            librosa.feature.spectral_centroid(y=seg, sr=sr)
        ))
        if centroid < 300:
            midi_note = 36      # bass drum
        elif centroid < 2000:
            midi_note = 38      # snare
        else:
            midi_note = 42      # hi-hat
        vel = int(max(60, min(127, 80 + rms * 250)))

        events.append(NoteEvent(
            start_tick=tick, length_tick=1,
            midi_note=midi_note, velocity=vel,
        ))
        last_tick = tick

    return events


# ---------------------------------------------------------------------------
# Single-track (hum)  – pyin is fine here because humming is monophonic
# ---------------------------------------------------------------------------

def audio_to_note_events(
    audio: np.ndarray,
    sr: int,
    waveform: str,
    bpm: int = 120,
    ticks_per_beat: int = 4,
) -> list[NoteEvent]:
    y = _normalize(audio)
    if y.size < sr // 4:
        return []

    duration_s = max(0.1, y.size / sr)
    total_ticks = _compute_total_ticks(duration_s, bpm, ticks_per_beat)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=80.0, fmax=2000.0, sr=sr,
        frame_length=2048, hop_length=512,
    )
    frame_times = librosa.times_like(f0, sr=sr, hop_length=512)
    midi_raw = librosa.hz_to_midi(f0)
    n_frames = len(f0)

    events: list[NoteEvent] = []
    cur: int | None = None
    start_f = 0
    cum_prob = 0.0
    cnt = 0
    _pitch_votes: list[int] = []

    def _flush(end_f: int) -> None:
        nonlocal cur, cum_prob, cnt, _pitch_votes
        if cur is None or cnt == 0:
            return
        ef = min(end_f, n_frames - 1)
        t0 = float(frame_times[start_f])
        t1 = float(frame_times[ef])
        ts = _time_to_tick(t0, duration_s, total_ticks)
        te = _time_to_tick(t1, duration_s, total_ticks)
        avg_p = cum_prob / max(1, cnt)
        vel = int(max(50, min(127, 60 + avg_p * 80)))
        events.append(NoteEvent(
            start_tick=ts,
            length_tick=max(1, te - ts + 1),
            midi_note=cur,
            velocity=vel,
        ))
        cur = None
        cum_prob = 0.0
        cnt = 0
        _pitch_votes = []

    for i in range(n_frames):
        voiced = (
            bool(voiced_flag[i])
            and float(voiced_prob[i]) >= 0.20
            and np.isfinite(midi_raw[i])
        )
        if voiced:
            mv = max(21, min(108, int(round(float(midi_raw[i])))))
            if cur is None:
                cur, start_f, cum_prob, cnt = mv, i, float(voiced_prob[i]), 1
                _pitch_votes = [mv]
            elif abs(mv - cur) <= 2:
                cum_prob += float(voiced_prob[i])
                cnt += 1
                _pitch_votes.append(mv)
            else:
                # Use the most-voted pitch for this region (mode) instead
                # of the first detected pitch — reduces jitter.
                if _pitch_votes:
                    from collections import Counter as _C
                    cur = _C(_pitch_votes).most_common(1)[0][0]
                _flush(i)
                cur, start_f, cum_prob, cnt = mv, i, float(voiced_prob[i]), 1
                _pitch_votes = [mv]
        else:
            if cur is not None and _pitch_votes:
                from collections import Counter as _C
                cur = _C(_pitch_votes).most_common(1)[0][0]
            _flush(i)

    # Final flush – resolve pitch from votes
    if cur is not None and _pitch_votes:
        from collections import Counter as _C
        cur = _C(_pitch_votes).most_common(1)[0][0]
    _flush(n_frames - 1)

    # merge same-pitch neighbours
    events.sort(key=lambda n: n.start_tick)
    merged: list[NoteEvent] = []
    for note in events:
        if merged:
            prev = merged[-1]
            if (prev.midi_note == note.midi_note
                    and note.start_tick <= prev.start_tick + prev.length_tick + 1):
                new_end = max(prev.start_tick + prev.length_tick,
                              note.start_tick + note.length_tick)
                prev.length_tick = max(1, new_end - prev.start_tick)
                prev.velocity = (prev.velocity + note.velocity) // 2
                continue
        merged.append(note)

    # Stabilise jitter and apply theory cleaning (same as multitrack path)
    merged = _stabilize_pitch_sequence(merged, ticks_per_beat)
    merged = _merge_same_pitch(merged, max_gap=1)
    fixer = SmartTheoryFixer(strictness=0.75)
    merged = fixer.fix(merged, role="lead")

    return merged


# ---------------------------------------------------------------------------
# Multi-track  (CQT-based polyphonic — the main import path)
# ---------------------------------------------------------------------------

def audio_to_multitrack_events(
    audio: np.ndarray,
    sr: int,
    bpm: int = 120,
    ticks_per_beat: int = 4,
    progress_callback: Callable[[int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, list[NoteEvent]]:
    """CQT polyphonic transcription → Lead / Bass / Harmony / Drums."""

    def _report(pct: int, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    def _check() -> None:
        if should_cancel and should_cancel():
            raise TranscriptionCancelled()

    empty = {"lead": [], "bass": [], "harmony": [], "drums": []}

    # ── prepare ────────────────────────────────────────────────────────
    _report(3, "Preparing audio\u2026")
    _check()

    y = _normalize(audio)
    duration_s = max(0.1, y.size / sr)
    total_ticks = _compute_total_ticks(duration_s, bpm, ticks_per_beat)

    if y.size < sr // 3:
        _report(100, "Audio too short.")
        return empty

    # ── HPSS ───────────────────────────────────────────────────────────
    _report(12, "Separating harmonic & percussive layers\u2026")
    _check()
    y_harm, y_perc = librosa.effects.hpss(y)

    # ── CQT on harmonic component ──────────────────────────────────────
    _report(28, "Computing pitch spectrogram (CQT)\u2026")
    _check()

    n_bins = _MIDI_HI - _MIDI_LO + 1          # 73 bins, C1–C7
    fmin = float(librosa.midi_to_hz(_MIDI_LO))

    C = np.abs(librosa.cqt(
        y_harm, sr=sr, fmin=fmin,
        n_bins=n_bins, bins_per_octave=12, hop_length=_HOP,
    ))

    max_mag = float(np.max(C))
    if max_mag < 1e-6:
        _report(100, "No pitched content found.")
        return empty

    n_frames = C.shape[1]
    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=_HOP,
    )

    # ── clean CQT (remove overtones & leakage) ─────────────────────────
    _report(40, "Removing harmonic overtones\u2026")
    _check()
    C = _clean_cqt(C)

    # ── polyphony limit ────────────────────────────────────────────────
    _report(48, "Limiting polyphony\u2026")
    C = _limit_polyphony(C, max_voices=6)

    # Recompute max after cleaning
    max_mag = float(np.max(C))
    if max_mag < 1e-6:
        _report(100, "No pitched content after cleaning.")
        return empty

    # ── threshold ──────────────────────────────────────────────────────
    _report(55, "Detecting active notes\u2026")
    _check()

    global_thresh = 0.15 * max_mag                    # noise floor (raised)
    frame_max = np.max(C, axis=0)                     # loudest note each frame
    frame_thresh = 0.40 * frame_max                   # relative per-frame gate (raised)

    all_notes = _cqt_to_notes(
        C, max_mag, _MIDI_LO, frame_times,
        duration_s, total_ticks,
        global_thresh, frame_thresh,
        max_gap=2, min_frames=2,
    )

    # ── consolidate notes (remove noise, merge fragments, quantize) ───
    _report(65, "Consolidating notes\u2026")
    _check()
    all_notes = _consolidate_notes(all_notes, ticks_per_beat)

    # ── split by pitch range ───────────────────────────────────────────
    _report(70, "Splitting into tracks\u2026")
    _check()

    bass_events    = sorted([n for n in all_notes if n.midi_note <= 52],
                            key=lambda n: n.start_tick)
    harmony_events = sorted([n for n in all_notes if 53 <= n.midi_note <= 59],
                            key=lambda n: n.start_tick)
    lead_events    = sorted([n for n in all_notes if n.midi_note >= 60],
                            key=lambda n: n.start_tick)

    # ── drums ──────────────────────────────────────────────────────────
    _report(80, "Detecting drum hits\u2026")
    _check()
    drums_events = _extract_drums(y_perc, sr, total_ticks, duration_s)

    # ── music-theory cleaning ──────────────────────────────────────────
    _report(90, "Applying music theory\u2026")
    _check()

    raw_split = {
        "lead":    lead_events,
        "bass":    bass_events,
        "harmony": harmony_events,
        "drums":   drums_events,
    }
    fixer = SmartTheoryFixer(strictness=0.75)
    clean_split = fixer.fix_multitrack(raw_split)

    _report(100, "Analysis complete.")
    return clean_split
