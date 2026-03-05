"""Export a Kokesynth-Station project as a clean music-sheet image (PNG/PDF).

Uses **matplotlib** to draw standard Western staff notation:
  • Treble + bass clef per track (auto-selected based on note range)
  • Noteheads (whole / half / filled) with stems
  • Ledger lines above/below the staff
  • Barlines, time signature, tempo marking
  • Track / instrument labels
  • Multi-system layout (wraps after N bars per line)

No external engraving tools (LilyPond, MuseScore) are required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no GUI backend needed

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm

from daw.models import NoteEvent, Project, Track

# ── Constants ────────────────────────────────────────────────────────

_STAFF_LINES = 5
_LINE_SPACING = 0.25          # vertical distance between staff lines
_STAFF_HEIGHT = _LINE_SPACING * (_STAFF_LINES - 1)  # 1.0

_TREBLE_BOTTOM_MIDI = 60      # Middle C (C4) = bottom ledger line for treble
_BASS_BOTTOM_MIDI = 40        # E2 = bottom line of bass staff

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Diatonic step offsets within an octave (C=0): C D E F G A B
_DIATONIC_PC = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}
# Map every chromatic pitch-class to the nearest diatonic position (for sharps/flats)
_PC_TO_DIATONIC = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5, 11: 6}
_IS_ACCIDENTAL = {1, 3, 6, 8, 10}  # pitch-classes that need a sharp symbol


def _midi_to_staff_position(midi: int) -> tuple[float, bool]:
    """Return (diatonic_position_from_C0, needs_sharp).

    Diatonic position 0 = C0, 1 = D0, 2 = E0, ... 7 = C1, etc.
    The staff renderer places notes relative to a clef reference line.
    """
    octave = midi // 12
    pc = midi % 12
    diatonic = _PC_TO_DIATONIC[pc]
    needs_sharp = pc in _IS_ACCIDENTAL
    return octave * 7 + diatonic, needs_sharp


def _treble_y(midi: int, staff_bottom_y: float) -> tuple[float, bool]:
    """Y-position on a treble-clef staff (bottom line = E4 = MIDI 64).

    Returns (y_coordinate, needs_sharp).
    """
    pos, sharp = _midi_to_staff_position(midi)
    ref_pos, _ = _midi_to_staff_position(64)  # E4 = bottom line of treble staff
    offset = (pos - ref_pos) * (_LINE_SPACING / 2)
    return staff_bottom_y + offset, sharp


def _bass_y(midi: int, staff_bottom_y: float) -> tuple[float, bool]:
    """Y-position on a bass-clef staff (bottom line = G2 = MIDI 43).

    Returns (y_coordinate, needs_sharp).
    """
    pos, sharp = _midi_to_staff_position(midi)
    ref_pos, _ = _midi_to_staff_position(43)  # G2 = bottom line of bass staff
    offset = (pos - ref_pos) * (_LINE_SPACING / 2)
    return staff_bottom_y + offset, sharp


# ── Quantise notes to beat grid ──────────────────────────────────────

@dataclass
class QuantisedNote:
    beat: float          # beat position within the bar (0-based)
    duration_beats: float
    midi: int
    velocity: int


def _quantise_track(track: Track, tpb: int, bpb: int) -> list[list[QuantisedNote]]:
    """Group notes into bars and quantise to the nearest beat.

    Returns a list-of-lists: bars[bar_idx] = [QuantisedNote, ...].
    """
    bar_ticks = tpb * bpb
    bars: dict[int, list[QuantisedNote]] = {}

    for n in sorted(track.notes, key=lambda e: e.start_tick):
        bar_idx = n.start_tick // bar_ticks
        beat_in_bar = (n.start_tick % bar_ticks) / tpb
        dur_beats = max(0.25, n.length_tick / tpb)
        bars.setdefault(bar_idx, []).append(
            QuantisedNote(beat=beat_in_bar, duration_beats=dur_beats,
                          midi=n.midi_note, velocity=n.velocity)
        )

    if not bars:
        return []
    max_bar = max(bars.keys())
    return [bars.get(i, []) for i in range(max_bar + 1)]


# ── Rendering ────────────────────────────────────────────────────────

def _choose_clef(track) -> str:
    """Pick 'treble' or 'bass' based on average note pitch."""
    notes = getattr(track, "notes", [])
    if not notes:
        return "treble"
    avg = sum(n.midi_note for n in notes) / len(notes)
    return "bass" if avg < 55 else "treble"


def _notehead_type(dur_beats: float) -> str:
    """Decide notehead appearance from duration in beats."""
    if dur_beats >= 3.5:
        return "whole"
    if dur_beats >= 1.5:
        return "half"
    return "filled"


def _draw_staff_lines(ax, x_start: float, x_end: float, y_bottom: float,
                      color: str = "#aaaaaa", lw: float = 0.7):
    """Draw the 5 horizontal staff lines."""
    for i in range(_STAFF_LINES):
        y = y_bottom + i * _LINE_SPACING
        ax.plot([x_start, x_end], [y, y], color=color, linewidth=lw,
                zorder=1, solid_capstyle="butt")


def _draw_barline(ax, x: float, y_bottom: float, color: str = "#777777",
                  lw: float = 0.8):
    """Draw a vertical barline spanning the full staff."""
    y_top = y_bottom + _STAFF_HEIGHT
    ax.plot([x, x], [y_bottom, y_top], color=color, linewidth=lw, zorder=2)


def _draw_clef(ax, x: float, y_bottom: float, clef: str):
    """Draw a clef indicator at the beginning of the staff."""
    if clef == "treble":
        # Treble clef: draw a stylised G-clef using line art
        y_center = y_bottom + _LINE_SPACING * 2.0  # sits near B line
        # Vertical spine
        ax.plot([x, x], [y_bottom - 0.1, y_bottom + _STAFF_HEIGHT + 0.25],
                color="#222222", linewidth=1.3, zorder=5, solid_capstyle="round")
        # Curved hook at bottom
        from matplotlib.patches import Arc
        arc1 = Arc((x, y_bottom + 0.05), 0.25, 0.3, angle=0,
                   theta1=180, theta2=360, color="#222222", linewidth=1.3, zorder=5)
        ax.add_patch(arc1)
        # Main loop
        arc2 = Arc((x + 0.04, y_center), 0.35, 0.6, angle=0,
                   theta1=30, theta2=330, color="#222222", linewidth=1.3, zorder=5)
        ax.add_patch(arc2)
        # Small circle at top
        from matplotlib.patches import Circle
        circ = Circle((x, y_bottom + _STAFF_HEIGHT + 0.2), 0.06,
                       facecolor="#222222", edgecolor="#222222", zorder=5)
        ax.add_patch(circ)
    else:
        # Bass clef: simplified F-clef
        y_f = y_bottom + _LINE_SPACING * 3.0  # F line
        # Main curve
        from matplotlib.patches import Arc, Circle
        arc = Arc((x + 0.05, y_f - 0.05), 0.35, 0.6, angle=0,
                  theta1=60, theta2=300, color="#222222", linewidth=1.5, zorder=5)
        ax.add_patch(arc)
        # Dot
        circ = Circle((x - 0.02, y_f), 0.06,
                       facecolor="#222222", edgecolor="#222222", zorder=5)
        ax.add_patch(circ)
        # Two dots
        dot1 = Circle((x + 0.25, y_f + _LINE_SPACING * 0.5), 0.035,
                       facecolor="#222222", edgecolor="#222222", zorder=5)
        dot2 = Circle((x + 0.25, y_f - _LINE_SPACING * 0.5), 0.035,
                       facecolor="#222222", edgecolor="#222222", zorder=5)
        ax.add_patch(dot1)
        ax.add_patch(dot2)


def _draw_time_sig(ax, x: float, y_bottom: float, numerator: int,
                   denominator: int):
    """Draw time signature numbers on the staff."""
    y_top = y_bottom + _LINE_SPACING * 2
    y_bot = y_bottom + _LINE_SPACING * 0
    ax.text(x, y_top + _LINE_SPACING * 0.5, str(numerator),
            fontsize=13, ha="center", va="center",
            color="#222222", fontweight="bold", zorder=5)
    ax.text(x, y_bot + _LINE_SPACING * 0.5, str(denominator),
            fontsize=13, ha="center", va="center",
            color="#222222", fontweight="bold", zorder=5)


def _draw_note(ax, x: float, y: float, ntype: str, sharp: bool,
               stem_up: bool = True, color: str = "#00ffc8"):
    """Draw a single notehead with stem and optional sharp."""
    note_w = 0.18
    note_h = _LINE_SPACING * 0.72

    if ntype == "whole":
        ell = Ellipse((x, y), note_w, note_h, facecolor="none",
                      edgecolor=color, linewidth=1.4, zorder=4)
        ax.add_patch(ell)
    elif ntype == "half":
        ell = Ellipse((x, y), note_w, note_h, facecolor="none",
                      edgecolor=color, linewidth=1.4, zorder=4)
        ax.add_patch(ell)
        # Stem
        sx = x + note_w / 2 if stem_up else x - note_w / 2
        sy_end = y + 0.8 if stem_up else y - 0.8
        ax.plot([sx, sx], [y, sy_end], color=color, linewidth=1.0, zorder=3)
    else:
        # Filled
        ell = Ellipse((x, y), note_w, note_h, facecolor=color,
                      edgecolor=color, linewidth=0.6, zorder=4)
        ax.add_patch(ell)
        sx = x + note_w / 2 if stem_up else x - note_w / 2
        sy_end = y + 0.8 if stem_up else y - 0.8
        ax.plot([sx, sx], [y, sy_end], color=color, linewidth=1.0, zorder=3)

    # Sharp symbol
    if sharp:
        ax.text(x - note_w * 0.9, y, "♯", fontsize=9, ha="center",
                va="center", color=color, zorder=5)


def _draw_ledger_lines(ax, x: float, y_note: float, y_bottom: float,
                       color: str = "#aaaaaa"):
    """Draw ledger lines for notes above/below the staff."""
    y_top = y_bottom + _STAFF_HEIGHT
    ledger_w = 0.14
    half_sp = _LINE_SPACING

    if y_note < y_bottom:
        y = y_bottom - half_sp
        while y >= y_note - half_sp * 0.3:
            ax.plot([x - ledger_w, x + ledger_w], [y, y],
                    color=color, linewidth=0.7, zorder=2)
            y -= half_sp
    elif y_note > y_top:
        y = y_top + half_sp
        while y <= y_note + half_sp * 0.3:
            ax.plot([x - ledger_w, x + ledger_w], [y, y],
                    color=color, linewidth=0.7, zorder=2)
            y += half_sp


# ── Public API ───────────────────────────────────────────────────────

def export_sheet_image(
    project: Project,
    output_path: str,
    *,
    bars_per_line: int = 3,
    dpi: int = 200,
    progress_callback=None,
) -> str:
    """Render a music-sheet image of the entire project.

    Parameters
    ----------
    project : Project
        The project to render.
    output_path : str
        Destination file (supports .png and .pdf via matplotlib).
    bars_per_line : int
        How many bars fit on one line/system before wrapping.
    dpi : int
        Output image resolution.
    progress_callback : callable or None
        ``(percent: int, message: str) -> None``

    Returns
    -------
    str
        The path written.
    """
    if not project.tracks:
        raise ValueError("No tracks to render.")

    tpb = project.ticks_per_beat
    bpb = 4  # Assume 4/4 time

    # Quantise all non-muted tracks
    render_tracks: list[tuple[object, str, list[list[QuantisedNote]]]] = []
    for track in project.tracks:
        if getattr(track, "muted", False):
            continue
        notes = track.notes if hasattr(track, "notes") else []
        if not notes:
            continue
        bars = _quantise_track(track, tpb, bpb)
        if not bars:
            continue
        clef = _choose_clef(track)
        render_tracks.append((track, clef, bars))

    if not render_tracks:
        raise ValueError("All tracks are empty or muted.")

    # Determine total bars (max across all tracks)
    total_bars = max(len(bars) for _, _, bars in render_tracks)
    num_systems = math.ceil(total_bars / bars_per_line)

    # Progress
    def _prog(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _prog(5, "Laying out notation…")

    # ── Layout parameters ────────────────────────────────────────
    n_tracks = len(render_tracks)
    bar_width = 5.5               # horizontal space per bar
    clef_margin = 0.7             # space for clef + time sig
    label_margin = 1.8            # space for track name
    system_gap = 0.6              # vertical gap between systems
    track_gap = 0.35              # vertical gap between tracks within a system
    staff_v = _STAFF_HEIGHT       # 1.0

    system_height = n_tracks * (staff_v + track_gap) - track_gap + system_gap

    page_w = label_margin + clef_margin + bars_per_line * bar_width + 0.6
    page_h = 1.8 + num_systems * system_height + 0.5

    fig, ax = plt.subplots(1, 1, figsize=(page_w, max(page_h, 3.0)), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(-0.3, page_w - 0.3)
    ax.set_ylim(-0.3, page_h)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Title + tempo ────────────────────────────────────────────
    ax.text(page_w / 2, page_h - 0.3, "Kokesynth-Station",
            fontsize=16, ha="center", va="top", color="#111111",
            fontweight="bold")
    ax.text(page_w / 2, page_h - 0.7, f"♩ = {project.bpm}   |   4/4 Time",
            fontsize=10, ha="center", va="top", color="#555555")

    # Track colour palette
    _COLORS = [
        "#1a5c8a", "#8a1a2e", "#1a6b3a", "#5c3a8a",
        "#7a4a10", "#2a5a5a", "#6b2a5a", "#3a4a8a",
    ]

    _prog(10, "Drawing staves…")

    for sys_idx in range(num_systems):
        bar_start = sys_idx * bars_per_line
        bar_end = min(bar_start + bars_per_line, total_bars)
        n_bars_in_line = bar_end - bar_start

        sys_top_y = page_h - 1.2 - sys_idx * system_height

        for t_idx, (track, clef, bars_data) in enumerate(render_tracks):
            color = _COLORS[t_idx % len(_COLORS)]
            staff_bottom_y = sys_top_y - t_idx * (staff_v + track_gap) - staff_v

            x_staff_start = label_margin
            x_staff_end = label_margin + clef_margin + n_bars_in_line * bar_width

            # Staff lines
            _draw_staff_lines(ax, x_staff_start, x_staff_end, staff_bottom_y)

            # Track label (only on first system)
            if sys_idx == 0:
                t_name = getattr(track, "name", getattr(track, "role", "Track"))
                t_inst = getattr(track, "instrument_name", "")
                label = t_name
                if len(label) > 18:
                    label = label[:16] + "…"
                ax.text(label_margin - 0.15, staff_bottom_y + staff_v / 2,
                        label, fontsize=7, ha="right", va="center",
                        color=color, fontweight="bold")
                ax.text(label_margin - 0.15,
                        staff_bottom_y + staff_v / 2 - 0.18,
                        t_inst, fontsize=5, ha="right",
                        va="center", color="#666666")

            # Clef
            _draw_clef(ax, x_staff_start + 0.2, staff_bottom_y, clef)

            # Time sig (only first system)
            if sys_idx == 0:
                _draw_time_sig(ax, x_staff_start + 0.50, staff_bottom_y, 4, 4)

            # Pick the y-mapping function
            y_fn = _treble_y if clef == "treble" else _bass_y

            # Draw bars
            for bi in range(n_bars_in_line):
                abs_bar = bar_start + bi
                bar_x_start = x_staff_start + clef_margin + bi * bar_width

                # Barline
                if bi > 0 or sys_idx > 0:
                    _draw_barline(ax, bar_x_start, staff_bottom_y)

                # End barline (double at final bar)
                if abs_bar == total_bars - 1 and bi == n_bars_in_line - 1:
                    end_x = bar_x_start + bar_width
                    _draw_barline(ax, end_x, staff_bottom_y, lw=0.8)
                    _draw_barline(ax, end_x + 0.05, staff_bottom_y, lw=2.0)

                # Bar number
                if t_idx == 0:
                    ax.text(bar_x_start + 0.08,
                            staff_bottom_y + staff_v + 0.15,
                            str(abs_bar + 1), fontsize=5,
                            color="#aaaaaa", va="bottom")

                # Notes
                if abs_bar < len(bars_data):
                    for note in bars_data[abs_bar]:
                        note_x = bar_x_start + 0.2 + (note.beat / bpb) * (bar_width - 0.4)
                        note_y, sharp = y_fn(note.midi, staff_bottom_y)

                        ntype = _notehead_type(note.duration_beats)
                        # Stem direction: up if below middle of staff, down if above
                        staff_mid = staff_bottom_y + staff_v / 2
                        stem_up = note_y < staff_mid

                        _draw_ledger_lines(ax, note_x, note_y,
                                           staff_bottom_y)
                        _draw_note(ax, note_x, note_y, ntype, sharp,
                                   stem_up=stem_up, color=color)

        # Draw final barline for this system (rightmost edge)
        pct = 10 + int(80 * (sys_idx + 1) / num_systems)
        _prog(min(90, pct), f"System {sys_idx + 1}/{num_systems}…")

    _prog(92, "Saving image…")

    plt.tight_layout(pad=0.3)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    _prog(100, "Done!")
    return output_path
