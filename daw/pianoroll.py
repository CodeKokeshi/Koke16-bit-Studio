from __future__ import annotations

import copy

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from daw.models import NoteEvent, Track
from daw.shortcuts import ShortcutConfig, binding_matches, is_pure_modifier_key, modifiers_equal


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_name(midi_note: int) -> str:
    octave = midi_note // 12 - 1
    return f"{NOTE_NAMES[midi_note % 12]}{octave}"


class PianoRollCanvas(QWidget):
    note_selected = pyqtSignal(object)
    note_audition = pyqtSignal(int)
    zoom_requested = pyqtSignal(int)

    KEY_WIDTH = 64
    TICK_WIDTH = 22
    MIN_TICK_WIDTH = 6
    ROW_HEIGHT = 18
    TOTAL_TICKS = 256
    NOTE_TOP = 108
    NOTE_BOTTOM = 21
    DEFAULT_VIEW_TOP = 84      # C6
    DEFAULT_VIEW_BOTTOM = 41   # F2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.track: Track | None = None
        self.playhead_tick = 0
        self._music_end_tick: int = 0        # project-wide end-of-music marker
        self.selected_note: NoteEvent | None = None
        self.selected_note_ids: set[int] = set()
        self.hover_note: NoteEvent | None = None
        self.hover_edge: str | None = None
        self.tick_width = self.TICK_WIDTH
        self._drag_note: NoteEvent | None = None
        self._drag_notes: list[NoteEvent] = []
        self._drag_mode: str | None = None
        self._drag_origin_tick = 0
        self._drag_origin_note = 0
        self._drag_origin_length = 0
        self._drag_mouse_tick = 0
        self._drag_mouse_pitch = 0
        self._drag_end_tick = 0
        self._drag_group_origins: dict[int, tuple[int, int]] = {}
        self._last_audition_pitch: int | None = None
        self._selecting = False
        self._selection_start = QPoint()
        self._selection_rect = QRect()
        self._held_keys: set[int] = set()
        self._undo_stack: list[list[tuple[int, int, int, int]]] = []
        self._redo_stack: list[list[tuple[int, int, int, int]]] = []
        self._max_undo = 120
        self.shortcut_config = ShortcutConfig()
        self._clipboard: list[tuple[int, int, int, int]] = []  # (rel_tick, length, midi_note, velocity)
        self.snap_grid: int = 1  # 1 = off, 4 = quarter note, etc.
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._update_canvas_size()

    def set_shortcut_config(self, config: ShortcutConfig):
        self.shortcut_config = copy.deepcopy(config)

    def get_shortcut_config(self) -> ShortcutConfig:
        return copy.deepcopy(self.shortcut_config)

    def _gesture_matches(self, event_mods: Qt.KeyboardModifier, binding) -> bool:
        if not modifiers_equal(event_mods, binding.modifiers):
            return False
        if binding.key == int(Qt.Key.Key_unknown):
            return True
        return int(binding.key) in self._held_keys

    def _snapshot_notes(self) -> list[tuple[int, int, int, int]]:
        if not self.track:
            return []
        return [
            (note.start_tick, note.length_tick, note.midi_note, note.velocity)
            for note in self.track.notes
        ]

    def _restore_notes(self, snapshot: list[tuple[int, int, int, int]]):
        if self.track is None:
            return
        self.track.notes = [
            NoteEvent(start_tick=s, length_tick=l, midi_note=m, velocity=v)
            for s, l, m, v in snapshot
        ]
        self._set_single_selection(None)
        self.update()

    def _push_undo_state(self):
        if self.track is None:
            return
        snap = self._snapshot_notes()
        if self._undo_stack and self._undo_stack[-1] == snap:
            return
        self._undo_stack.append(snap)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if self.track is None or not self._undo_stack:
            return
        current = self._snapshot_notes()
        previous = self._undo_stack.pop()
        self._redo_stack.append(current)
        self._restore_notes(previous)

    def redo(self):
        if self.track is None or not self._redo_stack:
            return
        current = self._snapshot_notes()
        nxt = self._redo_stack.pop()
        self._undo_stack.append(current)
        self._restore_notes(nxt)

    def select_all_notes(self):
        if not self.track:
            return
        self._set_multi_selection(list(self.track.notes))
        self.update()

    def delete_selected_notes(self):
        if not self.track or not self.selected_note_ids:
            return
        self._push_undo_state()
        self.track.notes = [note for note in self.track.notes if id(note) not in self.selected_note_ids]
        self._set_single_selection(None)
        self.update()

    # ── Snap helper ─────────────────────────────────────────────────
    def _snap(self, tick: int) -> int:
        """Snap a tick value to the current grid size (1 = no snap)."""
        if self.snap_grid <= 1:
            return tick
        return round(tick / self.snap_grid) * self.snap_grid

    # ── Copy / Paste ────────────────────────────────────────────────
    def copy_selected_notes(self):
        if not self.track or not self.selected_note_ids:
            return
        selected = [n for n in self.track.notes if id(n) in self.selected_note_ids]
        if not selected:
            return
        min_tick = min(n.start_tick for n in selected)
        self._clipboard = [
            (n.start_tick - min_tick, n.length_tick, n.midi_note, n.velocity)
            for n in selected
        ]

    def paste_notes(self):
        if not self.track or not self._clipboard:
            return
        self._push_undo_state()
        # Paste at playhead position (or at the start)
        base_tick = self.playhead_tick
        new_notes = []
        for rel_tick, length, pitch, vel in self._clipboard:
            new_note = NoteEvent(
                start_tick=self._snap(base_tick + rel_tick),
                length_tick=length, midi_note=pitch, velocity=vel,
            )
            self.track.notes.append(new_note)
            new_notes.append(new_note)
        self._set_multi_selection(new_notes)
        self.update()

    # ── Transpose ───────────────────────────────────────────────────
    def transpose_selected(self, semitones: int):
        if not self.track or not self.selected_note_ids:
            return
        self._push_undo_state()
        for note in self.track.notes:
            if id(note) in self.selected_note_ids:
                new_pitch = note.midi_note + semitones
                note.midi_note = max(self.NOTE_BOTTOM, min(self.NOTE_TOP, new_pitch))
        if self.selected_note:
            self.note_selected.emit(self.selected_note)
        self.update()

    # ------------------------------------------------------------------
    @property
    def effective_total_ticks(self) -> int:
        """Total displayable ticks — at least TOTAL_TICKS but grows to fit content."""
        content_end = self._music_end_tick
        if self.track and self.track.notes:
            content_end = max(content_end,
                              max(n.start_tick + n.length_tick for n in self.track.notes))
        return max(self.TOTAL_TICKS, content_end + 16)   # +16 ticks of padding

    def set_music_end_tick(self, tick: int):
        """Set the project-wide end-of-music position (drawn as a marker)."""
        self._music_end_tick = max(0, tick)
        self._update_canvas_size()
        self.update()

    def _update_canvas_size(self):
        self.setMinimumSize(
            self.KEY_WIDTH + self.effective_total_ticks * self.tick_width,
            (self.NOTE_TOP - self.NOTE_BOTTOM + 1) * self.ROW_HEIGHT,
        )

    def set_tick_width(self, tick_width: int):
        bounded = max(self.MIN_TICK_WIDTH, min(self.TICK_WIDTH, tick_width))
        if bounded == self.tick_width:
            return
        self.tick_width = bounded
        self._update_canvas_size()
        self.update()

    def _is_selected(self, note: NoteEvent) -> bool:
        return id(note) in self.selected_note_ids

    def _set_single_selection(self, note: NoteEvent | None):
        if note is None:
            self.selected_note = None
            self.selected_note_ids.clear()
            self.note_selected.emit(None)
            return
        self.selected_note = note
        self.selected_note_ids = {id(note)}
        self.note_selected.emit(note)

    def _set_multi_selection(self, notes: list[NoteEvent]):
        self.selected_note_ids = {id(note) for note in notes}
        self.selected_note = notes[0] if notes else None
        self.note_selected.emit(self.selected_note)

    def set_track(self, track: Track | None):
        self.track = track
        self.selected_note = None
        self.selected_note_ids.clear()
        self.hover_note = None
        self.hover_edge = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_canvas_size()      # resize to fit content
        self.update()

    def set_playhead(self, tick: int):
        self.playhead_tick = tick
        self.update()

    def pitch_to_row(self, midi_note: int) -> int:
        return self.NOTE_TOP - midi_note

    def row_to_pitch(self, row: int) -> int:
        return self.NOTE_TOP - row

    def x_to_tick(self, x: int) -> int:
        return max(0, min(self.effective_total_ticks - 1, int((x - self.KEY_WIDTH) / self.tick_width)))

    def y_to_pitch(self, y: int) -> int:
        row = max(0, min(self.NOTE_TOP - self.NOTE_BOTTOM, int(y / self.ROW_HEIGHT)))
        return self.row_to_pitch(row)

    def note_rect(self, note: NoteEvent) -> QRect:
        row = self.pitch_to_row(note.midi_note)
        x = self.KEY_WIDTH + note.start_tick * self.tick_width
        y = row * self.ROW_HEIGHT + 1
        w = max(self.tick_width, note.length_tick * self.tick_width)
        h = self.ROW_HEIGHT - 2
        return QRect(x, y, w, h)

    def hit_note(self, pos: QPoint) -> tuple[NoteEvent | None, str | None]:
        if not self.track:
            return None, None
        for note in reversed(self.track.notes):
            rect = self.note_rect(note)
            if rect.contains(pos):
                near_left = abs(pos.x() - rect.left()) <= 6
                near_right = abs(pos.x() - rect.right()) <= 6
                if near_left:
                    return note, "left"
                if near_right:
                    return note, "right"
                return note, "body"
        return None, None

    def _update_hover_state(self, pos: QPoint):
        if pos.x() < self.KEY_WIDTH:
            self.hover_note = None
            self.hover_edge = None
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.update()
            return

        note, edge = self.hit_note(pos)
        self.hover_note = note
        self.hover_edge = edge
        if note is None:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif edge in ("left", "right"):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.update()

    def mousePressEvent(self, event):  # noqa: N802
        if not self.track:
            return

        self.setFocus()

        pos = event.position().toPoint()

        if pos.x() < self.KEY_WIDTH:
            pitch = self.y_to_pitch(pos.y())
            self.note_audition.emit(pitch)
            return

        if event.button() == Qt.MouseButton.RightButton:
            note, _ = self.hit_note(pos)
            if note:
                self._push_undo_state()
                self.track.notes.remove(note)
                if self.selected_note is note:
                    self.selected_note = None
                    self.note_selected.emit(None)
                self.update()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self._gesture_matches(event.modifiers(), self.shortcut_config.box_select) and pos.x() >= self.KEY_WIDTH:
            self._selecting = True
            self._selection_start = pos
            self._selection_rect = QRect(pos, pos)
            self.update()
            return

        note, edge = self.hit_note(pos)
        if note:
            if not self._is_selected(note):
                self._set_single_selection(note)
            else:
                self.selected_note = note
                self.note_selected.emit(note)
            self.note_audition.emit(note.midi_note)
            self._drag_note = note
            if edge == "left":
                self._drag_mode = "resize-left"
            elif edge == "right":
                self._drag_mode = "resize-right"
            elif len(self.selected_note_ids) > 1 and self._is_selected(note):
                self._drag_mode = "move-group"
            else:
                self._drag_mode = "move"
            self._drag_origin_tick = note.start_tick
            self._drag_origin_note = note.midi_note
            self._drag_origin_length = note.length_tick
            self._drag_end_tick = note.start_tick + note.length_tick
            self._drag_mouse_tick = self.x_to_tick(pos.x())
            self._drag_mouse_pitch = self.y_to_pitch(pos.y())
            if self._drag_mode in ("resize-left", "resize-right", "move", "move-group"):
                self._push_undo_state()
            if self._drag_mode == "move-group":
                self._drag_notes = [n for n in self.track.notes if self._is_selected(n)]
                self._drag_group_origins = {id(n): (n.start_tick, n.midi_note) for n in self._drag_notes}
            else:
                self._drag_notes = []
                self._drag_group_origins = {}
            self._last_audition_pitch = note.midi_note
            self.update()
            return

        # create note (snap to grid)
        tick = self._snap(self.x_to_tick(pos.x()))
        pitch = self.y_to_pitch(pos.y())
        self._push_undo_state()
        default_len = max(1, self.snap_grid) if self.snap_grid > 1 else 4
        new_note = NoteEvent(start_tick=tick, length_tick=default_len, midi_note=pitch, velocity=100)
        self.track.notes.append(new_note)
        self._set_single_selection(new_note)
        self._drag_note = new_note
        self._drag_mode = "resize"
        self._drag_origin_tick = new_note.start_tick
        self._drag_origin_note = new_note.midi_note
        self._drag_origin_length = new_note.length_tick
        self._drag_mouse_tick = tick
        self.note_audition.emit(new_note.midi_note)
        self._last_audition_pitch = new_note.midi_note
        self.update()

    def mouseMoveEvent(self, event):  # noqa: N802
        if not self.track:
            return

        pos = event.position().toPoint()
        if self._selecting:
            self._selection_rect = QRect(self._selection_start, pos).normalized()
            self.update()
            return

        if self._drag_note is None or self._drag_mode is None:
            self._update_hover_state(pos)
            return

        tick_now = self.x_to_tick(pos.x())
        delta_tick = tick_now - self._drag_mouse_tick

        if self._drag_mode == "resize-right":
            self._drag_note.length_tick = max(1, self._drag_origin_length + delta_tick)
        elif self._drag_mode == "resize-left":
            new_start = max(0, self._drag_origin_tick + delta_tick)
            new_end = max(new_start + 1, self._drag_end_tick)
            self._drag_note.start_tick = new_start
            self._drag_note.length_tick = max(1, new_end - new_start)
        elif self._drag_mode == "move-group":
            pitch_now = self.y_to_pitch(pos.y())
            delta_pitch = pitch_now - self._drag_mouse_pitch
            starts = [origin_start + delta_tick for origin_start, _ in self._drag_group_origins.values()]
            min_start = min(starts) if starts else 0
            if min_start < 0:
                delta_tick -= min_start
            for note in self._drag_notes:
                origin_start, origin_pitch = self._drag_group_origins[id(note)]
                note.start_tick = max(0, origin_start + delta_tick)
                note.midi_note = max(self.NOTE_BOTTOM, min(self.NOTE_TOP, origin_pitch + delta_pitch))
            if self.selected_note and self.selected_note in self._drag_notes:
                self.note_selected.emit(self.selected_note)
        else:
            self._drag_note.start_tick = max(0, self._drag_origin_tick + delta_tick)
            self._drag_note.midi_note = self.y_to_pitch(pos.y())
            if self._drag_note.midi_note != self._last_audition_pitch:
                self.note_audition.emit(self._drag_note.midi_note)
                self._last_audition_pitch = self._drag_note.midi_note

        self.note_selected.emit(self._drag_note)
        self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._selecting:
            self._selecting = False
            if self.track:
                selected = [note for note in self.track.notes if self._selection_rect.intersects(self.note_rect(note))]
                self._set_multi_selection(selected)
            self._selection_rect = QRect()
            self._update_hover_state(event.position().toPoint())
            self.update()
            return

        self._drag_note = None
        self._drag_notes = []
        self._drag_mode = None
        self._drag_group_origins = {}
        self._last_audition_pitch = None
        self._update_hover_state(event.position().toPoint())

    def wheelEvent(self, event):  # noqa: N802
        if self._gesture_matches(event.modifiers(), self.shortcut_config.zoom):
            self.zoom_requested.emit(event.angleDelta().y())
            event.accept()
            return
        event.ignore()

    def keyPressEvent(self, event):  # noqa: N802
        cfg = self.shortcut_config
        key = int(event.key())
        mods = event.modifiers()

        if not is_pure_modifier_key(key):
            self._held_keys.add(key)

        if binding_matches(key, mods, cfg.redo_secondary):
            self.redo()
            event.accept()
            return
        if binding_matches(key, mods, cfg.undo):
            self.undo()
            event.accept()
            return
        if binding_matches(key, mods, cfg.redo_primary):
            self.redo()
            event.accept()
            return
        if binding_matches(key, mods, cfg.select_all_notes):
            self.select_all_notes()
            event.accept()
            return
        if binding_matches(key, mods, cfg.delete_primary) or binding_matches(key, mods, cfg.delete_secondary):
            self.delete_selected_notes()
            self.note_selected.emit(None)
            event.accept()
            return

        # Copy / Paste / Cut
        if key == int(Qt.Key.Key_C) and mods == Qt.KeyboardModifier.ControlModifier:
            self.copy_selected_notes()
            event.accept()
            return
        if key == int(Qt.Key.Key_V) and mods == Qt.KeyboardModifier.ControlModifier:
            self.paste_notes()
            event.accept()
            return
        if key == int(Qt.Key.Key_X) and mods == Qt.KeyboardModifier.ControlModifier:
            self.copy_selected_notes()
            self.delete_selected_notes()
            event.accept()
            return

        # Transpose: Shift+Up/Down = ±1 semitone, Ctrl+Shift+Up/Down = ±12
        ctrl_shift = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        if key == int(Qt.Key.Key_Up) and mods == Qt.KeyboardModifier.ShiftModifier:
            self.transpose_selected(1)
            event.accept()
            return
        if key == int(Qt.Key.Key_Down) and mods == Qt.KeyboardModifier.ShiftModifier:
            self.transpose_selected(-1)
            event.accept()
            return
        if key == int(Qt.Key.Key_Up) and mods == ctrl_shift:
            self.transpose_selected(12)
            event.accept()
            return
        if key == int(Qt.Key.Key_Down) and mods == ctrl_shift:
            self.transpose_selected(-12)
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):  # noqa: N802
        key = int(event.key())
        self._held_keys.discard(key)
        super().keyReleaseEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        self.hover_note = None
        self.hover_edge = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        width = self.width()
        height = self.height()

        p.fillRect(0, 0, width, height, QColor("#101010"))

        # key background
        p.fillRect(0, 0, self.KEY_WIDTH, height, QColor("#0a0a0a"))

        rows = self.NOTE_TOP - self.NOTE_BOTTOM + 1
        for row in range(rows):
            midi_note = self.row_to_pitch(row)
            y = row * self.ROW_HEIGHT
            is_black = NOTE_NAMES[midi_note % 12].endswith("#")
            row_bg = QColor("#161616" if is_black else "#131313")
            p.fillRect(self.KEY_WIDTH, y, width - self.KEY_WIDTH, self.ROW_HEIGHT, row_bg)
            p.setPen(QPen(QColor("#222222"), 1))
            p.drawLine(0, y, width, y)

            if row % 12 == 0:
                p.setPen(QPen(QColor("#2f2f2f"), 1))
                p.drawLine(self.KEY_WIDTH, y, width, y)

            p.setPen(QColor("#666666"))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(6, y + self.ROW_HEIGHT - 5, midi_name(midi_note))

        # vertical timeline grid
        total = self.effective_total_ticks
        for t in range(total + 1):
            x = self.KEY_WIDTH + t * self.tick_width
            if t % 16 == 0:
                c = QColor("#3f3f3f")
            elif t % 4 == 0:
                c = QColor("#2b2b2b")
            else:
                c = QColor("#1c1c1c")
            p.setPen(QPen(c, 1))
            p.drawLine(x, 0, x, height)

        # ── end-of-music marker ──────────────────────────────────
        if self._music_end_tick > 0:
            ex = self.KEY_WIDTH + self._music_end_tick * self.tick_width
            # translucent red stripe
            p.fillRect(ex, 0, max(2, self.tick_width), height, QColor("#ff355520"))
            p.setPen(QPen(QColor("#ff3555"), 2, Qt.PenStyle.DashLine))
            p.drawLine(ex, 0, ex, height)
            # tiny label
            p.setPen(QColor("#ff3555"))
            p.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            p.drawText(ex + 4, 12, "END")

        # playhead
        px = self.KEY_WIDTH + self.playhead_tick * self.tick_width
        p.fillRect(px, 0, self.tick_width, height, QColor("#00ffc81c"))
        p.setPen(QPen(QColor("#00ffc8"), 1))
        p.drawLine(px, 0, px, height)

        if self.track:
            for note in self.track.notes:
                rect = self.note_rect(note)
                selected = self._is_selected(note)
                hovered = note is self.hover_note
                # Velocity-based coloring: quiet=darker teal, loud=brighter cyan
                vel_t = max(0.0, min(1.0, note.velocity / 127.0))
                base_r = int(0 + vel_t * 0)
                base_g = int(128 + vel_t * 88)   # 128..216
                base_b = int(100 + vel_t * 68)    # 100..168
                fill = QColor(base_r, base_g, base_b) if not selected else QColor("#25ffcf")
                if hovered and not selected:
                    fill = fill.lighter(120)
                border = QColor("#b6fff0" if selected else "#00ffc8")
                if hovered and not selected:
                    border = QColor("#78ffe2")
                p.setPen(QPen(border, 1))
                p.setBrush(fill)
                p.drawRect(rect)

                if hovered and self.hover_edge in ("left", "right"):
                    edge_x = rect.left() if self.hover_edge == "left" else rect.right() - 1
                    p.fillRect(edge_x, rect.top(), 2, rect.height(), QColor("#d5fff5"))

                label = f"{midi_name(note.midi_note)} v{note.velocity}"
                p.setPen(QColor("#001e17"))
                p.setFont(QFont("Segoe UI", 7))
                p.drawText(rect.adjusted(2, 1, -2, -1), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)

        if self._selecting and not self._selection_rect.isNull():
            p.setPen(QPen(QColor("#72ffe4"), 1, Qt.PenStyle.DashLine))
            p.setBrush(QColor("#00ffc820"))
            p.drawRect(self._selection_rect)

        p.end()


class StartMarkerBar(QWidget):
    tick_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tick = 0
        self._end_tick = 0              # end-of-music marker (non-draggable)
        self._max_tick = PianoRollCanvas.TOTAL_TICKS  # dynamic upper bound
        self._scroll_x = 0
        self._tick_width = PianoRollCanvas.TICK_WIDTH
        self.setFixedHeight(26)
        self.setMouseTracking(True)

    def tick(self) -> int:
        return self._tick

    def set_tick(self, tick: int):
        bounded = max(0, min(self._max_tick - 1, tick))
        if bounded == self._tick:
            return
        self._tick = bounded
        self.update()

    def set_end_tick(self, tick: int):
        """Set end-of-music visual marker (non-draggable)."""
        self._end_tick = max(0, tick)
        self._max_tick = max(PianoRollCanvas.TOTAL_TICKS, self._end_tick + 16)
        self.update()

    def set_max_tick(self, value: int):
        self._max_tick = max(PianoRollCanvas.TOTAL_TICKS, value)
        self.update()

    def set_scroll_x(self, value: int):
        if value == self._scroll_x:
            return
        self._scroll_x = value
        self.update()

    def set_tick_width(self, value: int):
        if value == self._tick_width:
            return
        self._tick_width = value
        self.update()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._set_tick_from_x(event.position().toPoint().x())

    def mouseMoveEvent(self, event):  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._set_tick_from_x(event.position().toPoint().x())

    def _set_tick_from_x(self, x: int):
        canvas_x = x + self._scroll_x
        tick = int((canvas_x - PianoRollCanvas.KEY_WIDTH) / self._tick_width)
        bounded = max(0, min(self._max_tick - 1, tick))
        if bounded != self._tick:
            self._tick = bounded
            self.tick_changed.emit(self._tick)
        self.update()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        w = self.width()
        h = self.height()

        p.fillRect(0, 0, w, h, QColor("#0d0d0d"))

        key_split_x = PianoRollCanvas.KEY_WIDTH - self._scroll_x
        p.fillRect(0, 0, max(0, min(w, key_split_x)), h, QColor("#090909"))

        p.setPen(QPen(QColor("#232323"), 1))
        p.drawLine(0, h - 1, w, h - 1)

        marker_x = PianoRollCanvas.KEY_WIDTH + self._tick * self._tick_width - self._scroll_x
        if 0 <= marker_x <= w:
            p.setPen(QPen(QColor("#00ffc8"), 1))
            p.drawLine(marker_x, 0, marker_x, h)

            tri_half = 6
            center_y = 9
            p.setBrush(QColor("#ff5cd1"))
            p.setPen(QPen(QColor("#ff5cd1"), 1))
            p.drawPolygon([
                QPoint(marker_x - tri_half, center_y - tri_half),
                QPoint(marker_x - tri_half, center_y + tri_half),
                QPoint(marker_x + tri_half, center_y),
            ])

        # ── end-of-music marker (non-draggable) ──────────────────
        if self._end_tick > 0:
            end_x = PianoRollCanvas.KEY_WIDTH + self._end_tick * self._tick_width - self._scroll_x
            if -10 <= end_x <= w + 10:
                p.setPen(QPen(QColor("#ff3555"), 2))
                p.drawLine(end_x, 0, end_x, h)

                sq = 5
                center_y = 9
                p.setBrush(QColor("#ff3555"))
                p.setPen(QPen(QColor("#ff3555"), 1))
                p.drawRect(end_x - sq, center_y - sq, sq * 2, sq * 2)

                p.setPen(QColor("#ffffff"))
                p.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
                p.drawText(end_x + sq + 3, center_y + 4, "END")

        p.end()


class PianoRollEditor(QWidget):
    note_selected = pyqtSignal(object)
    note_audition = pyqtSignal(int)
    start_tick_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        top_bar = QWidget()
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(0, 0, 0, 0)
        top_bar_layout.setSpacing(6)
        self.start_marker = StartMarkerBar()
        top_bar_layout.addWidget(self.start_marker, 1)

        self.lbl_start_tick = QLabel("Tick 0")
        self.lbl_start_tick.setMinimumWidth(56)
        top_bar_layout.addWidget(self.lbl_start_tick)

        # Snap-to-grid selector
        snap_label = QLabel("Snap:")
        snap_label.setStyleSheet("color:#aaa; font-size:11px;")
        top_bar_layout.addWidget(snap_label)
        self.snap_combo = QComboBox()
        self.snap_combo.addItem("Off", 1)
        self.snap_combo.addItem("1/4", 1)     # 1 tick (= quarter note when tpb=4)
        self.snap_combo.addItem("1/2", 2)
        self.snap_combo.addItem("Beat", 4)
        self.snap_combo.addItem("Bar", 16)
        self.snap_combo.setFixedWidth(64)
        self.snap_combo.currentIndexChanged.connect(self._on_snap_changed)
        top_bar_layout.addWidget(self.snap_combo)

        layout.addWidget(top_bar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.canvas = PianoRollCanvas()
        self.canvas.note_selected.connect(self.note_selected)
        self.canvas.note_audition.connect(self.note_audition)
        self.canvas.zoom_requested.connect(self._on_zoom_requested)
        self.start_marker.tick_changed.connect(self._on_start_marker_changed)
        self.scroll.horizontalScrollBar().valueChanged.connect(self.start_marker.set_scroll_x)

        self.scroll.setWidget(self.canvas)
        layout.addWidget(self.scroll)
        self.start_marker.set_tick_width(self.canvas.tick_width)
        self._apply_default_viewport()

    def _apply_default_viewport(self):
        top_row = self.canvas.pitch_to_row(self.canvas.DEFAULT_VIEW_TOP)
        top_y = top_row * self.canvas.ROW_HEIGHT
        self.scroll.verticalScrollBar().setValue(top_y)

    def set_track(self, track: Track | None):
        self.canvas.set_track(track)
        # keep start-marker bar in sync with canvas extent
        self.start_marker.set_max_tick(self.canvas.effective_total_ticks)

    def set_music_end_tick(self, tick: int):
        """Set end-of-music marker on both the canvas and the start-marker bar."""
        self.canvas.set_music_end_tick(tick)
        self.start_marker.set_end_tick(tick)
        self.start_marker.set_max_tick(self.canvas.effective_total_ticks)

    def set_playhead(self, tick: int):
        self.canvas.set_playhead(tick)

    def view_state(self) -> dict[str, int]:
        return {
            "tick_width": int(self.canvas.tick_width),
            "h_scroll": int(self.scroll.horizontalScrollBar().value()),
            "v_scroll": int(self.scroll.verticalScrollBar().value()),
        }

    def apply_view_state(self, state: dict[str, int]):
        tick_width = int(state.get("tick_width", self.canvas.tick_width))
        self.canvas.set_tick_width(tick_width)
        self.start_marker.set_tick_width(self.canvas.tick_width)

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        h_val = int(state.get("h_scroll", hbar.value()))
        v_val = int(state.get("v_scroll", vbar.value()))

        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), h_val)))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), v_val)))

    def shortcut_config(self) -> ShortcutConfig:
        return self.canvas.get_shortcut_config()

    def set_shortcut_config(self, config: ShortcutConfig):
        self.canvas.set_shortcut_config(config)

    def start_tick(self) -> int:
        return self.start_marker.tick()

    def set_start_tick(self, tick: int):
        self.start_marker.set_tick(tick)
        self._update_start_tick_label(self.start_marker.tick())

    def _on_start_marker_changed(self, tick: int):
        self._update_start_tick_label(tick)
        self.start_tick_changed.emit(tick)

    def _update_start_tick_label(self, tick: int):
        self.lbl_start_tick.setText(f"Tick {tick}")

    def _on_zoom_requested(self, delta: int):
        old_width = self.canvas.tick_width
        step = 2
        if delta < 0:
            new_width = old_width - step
        elif delta > 0:
            new_width = old_width + step
        else:
            return

        new_width = max(self.canvas.MIN_TICK_WIDTH, min(self.canvas.TICK_WIDTH, new_width))
        if new_width == old_width:
            return

        hbar = self.scroll.horizontalScrollBar()
        left_tick = (hbar.value() + self.canvas.KEY_WIDTH) / old_width

        self.canvas.set_tick_width(new_width)
        self.start_marker.set_tick_width(new_width)

        new_left_x = int(left_tick * new_width - self.canvas.KEY_WIDTH)
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), new_left_x)))

    def _on_snap_changed(self, index: int):
        grid = self.snap_combo.currentData()
        if grid is not None:
            self.canvas.snap_grid = int(grid)
