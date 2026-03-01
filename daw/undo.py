"""Project-level undo / redo manager.

Stores deep-copy snapshots of the *entire* project state (tracks, BPM,
selected index, loop settings) so that **any** destructive operation —
Beautify, Balance, Fix Loops, Remove Gaps, Generate, track add/delete,
instrument change, etc. — can be reversed with Ctrl+Z / Ctrl+Y.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from daw.models import Project


@dataclass(slots=True)
class _Snapshot:
    """Immutable deep-copy of a Project's mutable state."""
    label: str
    tracks_data: list  # deep-copied list[Track]
    bpm: int
    ticks_per_beat: int
    loop_mode: str
    custom_loop_ticks: int
    selected_track_index: int


class ProjectUndoManager:
    """Stack-based undo/redo for the whole project.

    Usage:
        undo_mgr = ProjectUndoManager(max_depth=40)
        undo_mgr.snapshot(project, "Beautify")   # capture BEFORE mutation
        # ... perform the mutation ...
        undo_mgr.undo(project)                    # restores previous state
        undo_mgr.redo(project)                    # re-applies the undone change
    """

    def __init__(self, max_depth: int = 40) -> None:
        self._undo: list[_Snapshot] = []
        self._redo: list[_Snapshot] = []
        self._max = max_depth

    # ── public API ──────────────────────────────────────────────────

    def snapshot(self, project: "Project", label: str = "") -> None:
        """Capture current project state **before** a destructive change."""
        snap = self._capture(project, label)
        self._undo.append(snap)
        if len(self._undo) > self._max:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, project: "Project") -> str | None:
        """Restore the most recent snapshot.  Returns the label or *None*."""
        if not self._undo:
            return None
        # Save current state so it can be redo'd
        current = self._capture(project, "")
        prev = self._undo.pop()
        current.label = prev.label          # transfer label for redo display
        self._redo.append(current)
        self._apply(prev, project)
        return prev.label

    def redo(self, project: "Project") -> str | None:
        """Re-apply the last undone change.  Returns the label or *None*."""
        if not self._redo:
            return None
        current = self._capture(project, "")
        nxt = self._redo.pop()
        current.label = nxt.label
        self._undo.append(current)
        if len(self._undo) > self._max:
            self._undo.pop(0)
        self._apply(nxt, project)
        return nxt.label

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return len(self._undo) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo) > 0

    @property
    def undo_label(self) -> str:
        return self._undo[-1].label if self._undo else ""

    @property
    def redo_label(self) -> str:
        return self._redo[-1].label if self._redo else ""

    @property
    def depth(self) -> int:
        return len(self._undo)

    # ── internals ───────────────────────────────────────────────────

    @staticmethod
    def _capture(project: "Project", label: str) -> _Snapshot:
        return _Snapshot(
            label=label,
            tracks_data=copy.deepcopy(project.tracks),
            bpm=project.bpm,
            ticks_per_beat=project.ticks_per_beat,
            loop_mode=project.loop_mode,
            custom_loop_ticks=project.custom_loop_ticks,
            selected_track_index=project.selected_track_index,
        )

    @staticmethod
    def _apply(snap: _Snapshot, project: "Project") -> None:
        project.tracks = copy.deepcopy(snap.tracks_data)
        project.bpm = snap.bpm
        project.ticks_per_beat = snap.ticks_per_beat
        project.loop_mode = snap.loop_mode
        project.custom_loop_ticks = snap.custom_loop_ticks
        project.selected_track_index = snap.selected_track_index
