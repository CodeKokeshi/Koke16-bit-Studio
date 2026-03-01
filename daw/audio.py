from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
import pygame
import pygame.mixer

from daw.models import Project


class AudioEngine(QObject):
    position_changed = pyqtSignal(int)

    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.sample_rate = 44100
        self.current_tick = 0
        self.playing = False
        self._solo_track_index: int | None = None
        self._start_tick: int | None = None
        self._cache: dict[tuple[str, int, int], object] = {}

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self.available = False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.pre_init(self.sample_rate, -16, 1, 512)
                pygame.mixer.init()
            pygame.mixer.set_num_channels(64)
            self.available = True
        except Exception:
            self.available = False

    def set_bpm(self, bpm: int) -> None:
        self.project.bpm = max(40, min(260, bpm))
        if self.playing:
            self._timer.setInterval(self._tick_ms())

    def start(self, solo_track_index: int | None = None, start_tick: int | None = None) -> None:
        if not self.available:
            return
        self._solo_track_index = solo_track_index
        self._start_tick = start_tick
        loop_start, loop_end = self._loop_window()
        self.playing = True
        self.current_tick = self._resolve_start_tick(loop_start, loop_end)
        self.position_changed.emit(self.current_tick)
        self._timer.start(self._tick_ms())

    def stop(self) -> None:
        self.playing = False
        loop_start, loop_end = self._loop_window()
        self.current_tick = self._resolve_start_tick(loop_start, loop_end)
        self.position_changed.emit(self.current_tick)
        self._timer.stop()
        if self.available:
            pygame.mixer.stop()
        self._solo_track_index = None

    def preview_note(self, waveform: str, midi_note: int, velocity: int = 100) -> None:
        if not self.available:
            return
        duration = 0.25
        amp = max(0.05, min(1.0, velocity / 127.0)) * 0.6
        sound = self._get_sound(waveform, midi_note, duration)
        ch = pygame.mixer.find_channel(True)
        ch.set_volume(amp)
        ch.play(sound)  # type: ignore[arg-type]

    def _tick_ms(self) -> int:
        return max(15, int(60000 / (self.project.bpm * self.project.ticks_per_beat)))

    def _tick(self) -> None:
        loop_start, loop_end = self._loop_window()
        for track in self._playback_tracks():
            for note in track.notes:
                if note.start_tick == self.current_tick:
                    dur = max(0.04, note.length_tick / self.project.ticks_per_beat * 60.0 / self.project.bpm)
                    vel_amp = max(0.05, min(1.0, note.velocity / 127.0))
                    sound = self._get_sound(track.waveform, note.midi_note, dur)
                    ch = pygame.mixer.find_channel(True)
                    if ch is not None:
                        ch.set_volume(vel_amp * track.volume)
                        ch.play(sound)  # type: ignore[arg-type]

        next_tick = self.current_tick + 1
        self.current_tick = loop_start if next_tick >= loop_end else next_tick
        self.position_changed.emit(self.current_tick)

    def _dynamic_loop_window(self) -> tuple[int, int]:
        min_start = None
        max_end = 0
        for track in self._playback_tracks():
            for note in track.notes:
                if min_start is None:
                    min_start = note.start_tick
                else:
                    min_start = min(min_start, note.start_tick)
                max_end = max(max_end, note.start_tick + note.length_tick)

        if max_end <= 0 or min_start is None:
            default_len = max(16, self.project.ticks_per_beat * 4)
            return 0, default_len

        if max_end <= min_start:
            return min_start, min_start + 1

        return min_start, max_end

    def _loop_window(self) -> tuple[int, int]:
        if self.project.loop_mode == "timeline":
            return 0, 256
        if self.project.loop_mode == "custom":
            return 0, max(1, self.project.custom_loop_ticks)
        return self._dynamic_loop_window()

    def _resolve_start_tick(self, loop_start: int, loop_end: int) -> int:
        if loop_end <= loop_start:
            return loop_start
        if self._start_tick is None:
            return loop_start
        return max(loop_start, min(loop_end - 1, self._start_tick))

    def _playback_tracks(self) -> list:
        if self._solo_track_index is None:
            return self.project.tracks
        if 0 <= self._solo_track_index < len(self.project.tracks):
            return [self.project.tracks[self._solo_track_index]]
        return []

    def _get_sound(self, waveform: str, midi_note: int, duration: float):
        """Return a cached Sound at full volume (volume applied via channel)."""
        key = (waveform, midi_note, int(duration * 1000))
        if key not in self._cache:
            self._cache[key] = self._build_sound(waveform, midi_note, duration, amp=1.0)
        return self._cache[key]

    def _build_sound(self, waveform: str, midi_note: int, duration: float, amp: float):
        n_samples = max(1, int(self.sample_rate * duration))
        t = np.arange(n_samples, dtype=np.float32) / self.sample_rate
        freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

        if waveform == "square":
            wave = np.where((t * freq) % 1.0 < 0.5, 1.0, -1.0).astype(np.float32)
        elif waveform == "pulse25":
            wave = np.where((t * freq) % 1.0 < 0.25, 1.0, -1.0).astype(np.float32)
        elif waveform == "pulse12":
            wave = np.where((t * freq) % 1.0 < 0.125, 1.0, -1.0).astype(np.float32)
        elif waveform == "sawtooth":
            wave = (2.0 * ((t * freq) % 1.0) - 1.0).astype(np.float32)
        elif waveform == "triangle":
            ph = (t * freq) % 1.0
            wave = (2.0 * np.abs(2.0 * ph - 1.0) - 1.0).astype(np.float32)
        elif waveform == "noise":
            rng = np.random.default_rng(seed=(midi_note * 31 + int(duration * 1000)))
            wave = rng.uniform(-1.0, 1.0, n_samples).astype(np.float32)
            decay = np.linspace(1.0, 0.0, n_samples, dtype=np.float32)
            wave *= decay
        else:
            wave = np.sin(2.0 * np.pi * freq * t).astype(np.float32)

        # Envelope: match exporter exactly (attack 5ms, release 80ms)
        attack = min(int(0.005 * self.sample_rate), n_samples // 4)
        release = min(int(0.08 * self.sample_rate), n_samples // 2)
        env = np.ones(n_samples, dtype=np.float32)
        if attack > 0:
            env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
        if release > 0:
            env[-release:] = np.linspace(1.0, 0.0, release, dtype=np.float32)

        shaped = np.clip(wave * env * amp, -1.0, 1.0)

        # Append a tiny silent tail to prevent click when channel is reused
        tail = np.zeros(int(0.005 * self.sample_rate), dtype=np.float32)
        shaped = np.concatenate([shaped, tail])

        samples = np.ascontiguousarray(shaped * 32767, dtype=np.int16)
        return pygame.mixer.Sound(samples)
