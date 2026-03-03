"""Offline WAV renderer for Koke16-Bit Studio.

Renders the project to a numpy buffer (or directly to a WAV file)
by synthesising every note with the same waveforms used by the
real-time playback engine.  Supports exporting multiple loops.
"""

from __future__ import annotations

import struct
import wave
from typing import Callable

import numpy as np

from daw.models import Project


# ─── Waveform generation (mirrors audio.py _build_sound) ──────────────

def _generate_wave(
    waveform: str,
    midi_note: int,
    duration_s: float,
    amp: float,
    apply_attack: bool = True,
    apply_release: bool = True,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Synthesise a single note as float32 samples in [-1, 1]."""
    n_samples = max(1, int(sample_rate * duration_s))
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    if waveform == "square":
        w = np.where((t * freq) % 1.0 < 0.5, 1.0, -1.0).astype(np.float32)
    elif waveform == "pulse25":
        w = np.where((t * freq) % 1.0 < 0.25, 1.0, -1.0).astype(np.float32)
    elif waveform == "pulse12":
        w = np.where((t * freq) % 1.0 < 0.125, 1.0, -1.0).astype(np.float32)
    elif waveform == "sawtooth":
        w = (2.0 * ((t * freq) % 1.0) - 1.0).astype(np.float32)
    elif waveform == "triangle":
        ph = (t * freq) % 1.0
        w = (2.0 * np.abs(2.0 * ph - 1.0) - 1.0).astype(np.float32)
    elif waveform == "noise":
        rng = np.random.default_rng(seed=(midi_note * 31 + int(duration_s * 1000)))
        w = rng.uniform(-1.0, 1.0, n_samples).astype(np.float32)
        decay = np.linspace(1.0, 0.0, n_samples, dtype=np.float32)
        w *= decay
    else:  # sine
        w = np.sin(2.0 * np.pi * freq * t).astype(np.float32)

    # Envelope: tiny attack + release to avoid clicks
    attack = min(int(0.005 * sample_rate), n_samples // 4)
    release = min(int(0.08 * sample_rate), n_samples // 2)
    env = np.ones(n_samples, dtype=np.float32)
    if apply_attack and attack > 0:
        env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    if apply_release and release > 0:
        env[-release:] = np.linspace(1.0, 0.0, release, dtype=np.float32)

    return np.clip(w * env * amp, -1.0, 1.0)


# ─── Loop-window helper (mirrors audio.py) ────────────────────────────

def _dynamic_loop_window(project: Project) -> tuple[int, int]:
    """Compute the loop range from all notes across all tracks."""
    min_start: int | None = None
    max_end = 0
    for track in project.tracks:
        for note in track.notes:
            if min_start is None:
                min_start = note.start_tick
            else:
                min_start = min(min_start, note.start_tick)
            max_end = max(max_end, note.start_tick + note.length_tick)
    if max_end <= 0 or min_start is None:
        return 0, max(16, project.ticks_per_beat * 4)
    if max_end <= min_start:
        return min_start, min_start + 1
    return min_start, max_end


def _loop_window(project: Project) -> tuple[int, int]:
    if project.loop_mode == "timeline":
        return 0, 256
    if project.loop_mode == "custom":
        return 0, max(1, project.custom_loop_ticks)
    return _dynamic_loop_window(project)


# ─── Render ────────────────────────────────────────────────────────────

def render_project(
    project: Project,
    loops: int = 1,
    sample_rate: int = 44100,
    progress_callback: Callable[[int, str], None] | None = None,
) -> np.ndarray:
    """Render the whole project to a float32 mono buffer.

    Parameters
    ----------
    project : Project
        The project to render.
    loops : int
        Number of times to repeat the loop region (≥ 1).
    sample_rate : int
        Output sample rate (default 44 100).
    progress_callback : callable, optional
        ``(percent: int, message: str) -> None``

    Returns
    -------
    np.ndarray
        Mono float32 samples in ``[-1, 1]``.
    """
    loops = max(1, loops)

    loop_start, loop_end = _loop_window(project)
    loop_ticks = max(1, loop_end - loop_start)

    # Seconds per tick
    spt = 60.0 / (project.bpm * project.ticks_per_beat)

    # Total duration
    total_ticks = loop_ticks * loops
    total_seconds = total_ticks * spt
    total_samples = int(total_seconds * sample_rate) + sample_rate  # +1s safety

    buf = np.zeros(total_samples, dtype=np.float32)

    if progress_callback:
        progress_callback(5, "Preparing render\u2026")

    # Count total notes to render for progress
    active_tracks = [t for t in project.tracks if t.notes]
    total_notes = 0
    for track in active_tracks:
        notes_in_range = [n for n in track.notes
                          if loop_start <= n.start_tick < loop_end]
        total_notes += len(notes_in_range) * loops
    rendered_notes = 0

    for track in active_tracks:
        notes_in_range = [n for n in track.notes
                          if loop_start <= n.start_tick < loop_end]
        if not notes_in_range:
            continue

        for loop_i in range(loops):
            tick_offset = loop_i * loop_ticks

            for note in notes_in_range:
                # Note start relative to loop start + loop offset
                rel_tick = (note.start_tick - loop_start) + tick_offset
                start_s = rel_tick * spt
                dur_s = max(0.04, note.length_tick * spt)
                amp = max(0.05, min(1.0, note.velocity / 127.0)) * track.volume
                note_end_tick = note.start_tick + note.length_tick
                apply_attack = not (loop_i > 0 and note.start_tick == loop_start)
                apply_release = not (loop_i < loops - 1 and note_end_tick >= loop_end)

                wave = _generate_wave(
                    track.waveform,
                    note.midi_note,
                    dur_s,
                    amp,
                    apply_attack=apply_attack,
                    apply_release=apply_release,
                    sample_rate=sample_rate,
                )

                start_idx = int(start_s * sample_rate)
                end_idx = start_idx + len(wave)

                if start_idx >= total_samples:
                    continue
                if end_idx > total_samples:
                    wave = wave[: total_samples - start_idx]
                    end_idx = total_samples

                buf[start_idx:end_idx] += wave

                rendered_notes += 1
                if progress_callback and total_notes > 0:
                    pct = 5 + int(90 * rendered_notes / total_notes)
                    progress_callback(
                        min(95, pct),
                        f"Rendering note {rendered_notes}/{total_notes}\u2026",
                    )

    # Trim trailing silence (no extra tail pad for seamless loops)
    last_nonzero = np.flatnonzero(np.abs(buf) > 1e-6)
    if last_nonzero.size > 0:
        buf = buf[: last_nonzero[-1] + 1]
    else:
        buf = buf[:sample_rate]  # 1 second of silence if nothing rendered

    # Normalize to avoid clipping
    peak = float(np.max(np.abs(buf)))
    if peak > 1.0:
        buf /= peak
    elif peak > 0:
        # Gentle boost if quiet
        buf *= min(1.0 / peak, 2.0)
        buf = np.clip(buf, -1.0, 1.0)

    if progress_callback:
        progress_callback(100, "Render complete.")

    return buf


# ─── WAV writer ────────────────────────────────────────────────────────

def export_wav(
    path: str,
    project: Project,
    loops: int = 1,
    sample_rate: int = 44100,
    progress_callback: Callable[[int, str], None] | None = None,
) -> None:
    """Render and write a 16-bit mono WAV file."""
    buf = render_project(project, loops, sample_rate, progress_callback)

    samples_16 = np.clip(buf * 32767, -32768, 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_16.tobytes())
