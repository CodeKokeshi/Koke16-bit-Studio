"""
Music Sheet Reader — OMR (Optical Music Recognition) pipeline.

Accepts a PDF (multi-page) or single image file, runs *oemer* for end-to-end
recognition, parses the resulting MusicXML with *music21*, and writes a
human-readable / LLM-promptable .txt report with every note's details.

Heavy processing is offloaded to a QThread so the GUI stays responsive.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
import subprocess
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import QThread, pyqtSignal


# ---------------------------------------------------------------------------
#  Utility: MIDI note number → human‑readable pitch name
# ---------------------------------------------------------------------------
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _midi_to_name(midi: int) -> str:
    octave = (midi // 12) - 1
    return f"{_NOTE_NAMES[midi % 12]}{octave}"


# ---------------------------------------------------------------------------
#  PDF → Images  (via PyMuPDF / fitz)
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[str]:
    """Convert each page of *pdf_path* to a temporary PNG and return paths."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    image_paths: List[str] = []
    tmp_dir = tempfile.mkdtemp(prefix="kokesheet_")
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(tmp_dir, f"page_{page_num + 1:03d}.png")
        pix.save(out_path)
        image_paths.append(out_path)

    doc.close()
    return image_paths


# ---------------------------------------------------------------------------
#  oemer wrapper — runs OMR on a single image, returns MusicXML path
# ---------------------------------------------------------------------------

def _run_oemer_on_image(img_path: str, output_dir: str) -> str:
    """Run oemer on one image and return the resulting .musicxml path."""
    try:
        from oemer.ete import extract, clear_data, CHECKPOINTS_URL, download_file
        from oemer import MODULE_PATH

        # Ensure checkpoints are present
        chk_path = os.path.join(MODULE_PATH, "checkpoints/unet_big/model.onnx")
        if not os.path.exists(chk_path):
            for idx, (title, url) in enumerate(CHECKPOINTS_URL.items()):
                save_dir = "unet_big" if title.startswith("1st") else "seg_net"
                save_dir = os.path.join(MODULE_PATH, "checkpoints", save_dir)
                save_path = os.path.join(save_dir, title.split("_")[1])
                download_file(title, url, save_path)

        args = Namespace(
            img_path=img_path,
            output_path=output_dir,
            use_tf=False,
            save_cache=False,
            without_deskew=False,
        )

        clear_data()
        return extract(args)

    except Exception as exc:
        # Some Windows setups fail loading ORT DLLs in-process (e.g. from GUI thread)
        # but work fine in a fresh Python process. Fallback to subprocess execution.
        msg = str(exc).lower()
        if "onnxruntime" not in msg and "dll load failed" not in msg:
            raise

        cmd = [
            sys.executable,
            "-m",
            "oemer.ete",
            img_path,
            "-o",
            output_dir,
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "oemer subprocess failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
            )

        basename = os.path.splitext(os.path.basename(img_path))[0]
        mxl_path = os.path.join(output_dir, f"{basename}.musicxml")
        if not os.path.exists(mxl_path):
            raise RuntimeError(
                "oemer subprocess completed but MusicXML output was not found at:\n"
                f"{mxl_path}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
            )
        return mxl_path


# ---------------------------------------------------------------------------
#  MusicXML → structured text
# ---------------------------------------------------------------------------

def _parse_musicxml_to_text(mxl_path: str) -> str:
    """Parse a MusicXML file with music21 and return a detailed text report."""
    import music21

    score = music21.converter.parse(mxl_path)

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("  MUSIC SHEET EXTRACTION REPORT")
    lines.append("=" * 72)
    lines.append("")

    # ── Global metadata ───────────────────────────────────────────────
    # Title
    if score.metadata and score.metadata.title:
        lines.append(f"Title          : {score.metadata.title}")

    # Key signature (from first measure)
    keys = score.flatten().getElementsByClass(music21.key.KeySignature)
    if keys:
        ks = keys[0]
        lines.append(f"Key Signature  : {ks.asKey('major')} major / {ks.asKey('minor')} minor")

    # Time signature
    time_sigs = score.flatten().getElementsByClass(music21.meter.TimeSignature)
    if time_sigs:
        lines.append(f"Time Signature : {time_sigs[0].ratioString}")

    # Tempo / BPM
    tempos = score.flatten().getElementsByClass(music21.tempo.MetronomeMark)
    if tempos:
        bpm = tempos[0].number
        lines.append(f"BPM            : {bpm}")
    else:
        lines.append("BPM            : Not specified in score")

    lines.append("")

    # ── Per-part breakdown ────────────────────────────────────────────
    parts = score.parts
    for part_idx, part in enumerate(parts):
        part_name = part.partName or f"Part {part_idx + 1}"
        lines.append("-" * 72)
        lines.append(f"  PART {part_idx + 1}: {part_name}")
        lines.append("-" * 72)

        # Collect part-level key/clef if different
        clefs = part.flatten().getElementsByClass(music21.clef.Clef)
        if clefs:
            lines.append(f"  Clef: {clefs[0].name}")

        lines.append("")
        lines.append(
            f"  {'#':<5} {'Beat':>6} | {'Pitch':<8} {'MIDI':>4} | "
            f"{'Duration':<12} {'Beats':>5} | {'Velocity':>8} | {'Tie':>4} | Notes"
        )
        lines.append(f"  {'—' * 70}")

        note_counter = 0
        for measure in part.getElementsByClass(music21.stream.Measure):
            measure_num = measure.number
            lines.append(f"  ── Measure {measure_num} {'─' * 50}")

            for elem in measure.flatten().notesAndRests:
                offset_in_measure = elem.offset

                if isinstance(elem, music21.note.Rest):
                    dur_type = elem.duration.type
                    dur_ql = elem.duration.quarterLength
                    lines.append(
                        f"  {'R':<5} {offset_in_measure:6.2f} | {'rest':<8} {'—':>4} | "
                        f"{dur_type:<12} {dur_ql:5.2f} | {'—':>8} | {'—':>4} | "
                    )
                    continue

                if isinstance(elem, music21.chord.Chord):
                    pitches = elem.pitches
                    chord_name = " + ".join(p.nameWithOctave for p in pitches)
                    midi_nums = [p.midi for p in pitches]
                    midi_str = ",".join(str(m) for m in midi_nums)
                    dur_type = elem.duration.type
                    dur_ql = elem.duration.quarterLength
                    vel = elem.volume.velocity if elem.volume and elem.volume.velocity else "—"
                    tie_info = ""
                    if elem.tie:
                        tie_info = elem.tie.type
                    note_counter += 1
                    lines.append(
                        f"  {note_counter:<5} {offset_in_measure:6.2f} | "
                        f"{chord_name:<8} {midi_str:>4} | "
                        f"{dur_type:<12} {dur_ql:5.2f} | {str(vel):>8} | "
                        f"{tie_info:>4} | chord"
                    )
                    continue

                if isinstance(elem, music21.note.Note):
                    pitch_name = elem.pitch.nameWithOctave
                    midi_num = elem.pitch.midi
                    dur_type = elem.duration.type
                    dur_ql = elem.duration.quarterLength
                    vel = elem.volume.velocity if elem.volume and elem.volume.velocity else "—"
                    tie_info = ""
                    if elem.tie:
                        tie_info = elem.tie.type
                    note_counter += 1
                    lines.append(
                        f"  {note_counter:<5} {offset_in_measure:6.2f} | "
                        f"{pitch_name:<8} {midi_num:>4} | "
                        f"{dur_type:<12} {dur_ql:5.2f} | {str(vel):>8} | "
                        f"{tie_info:>4} | "
                    )

        lines.append("")

        # ── Quick note sequence (the F#3 → G4 → E5 style) ───────────
        flat_notes = part.flatten().getElementsByClass(music21.note.Note)
        if flat_notes:
            seq = " → ".join(n.pitch.nameWithOctave for n in flat_notes)
            lines.append(f"  Sequence: {seq}")
            lines.append("")

    # ── Summary ───────────────────────────────────────────────────────
    all_notes = score.flatten().getElementsByClass(music21.note.Note)
    all_rests = score.flatten().getElementsByClass(music21.note.Rest)
    all_chords = score.flatten().getElementsByClass(music21.chord.Chord)

    lines.append("=" * 72)
    lines.append("  SUMMARY")
    lines.append("=" * 72)
    lines.append(f"  Total parts   : {len(parts)}")
    lines.append(f"  Total notes   : {len(all_notes)}")
    lines.append(f"  Total chords  : {len(all_chords)}")
    lines.append(f"  Total rests   : {len(all_rests)}")

    if all_notes:
        pitches = [n.pitch.midi for n in all_notes]
        lines.append(f"  Pitch range   : {_midi_to_name(min(pitches))} (MIDI {min(pitches)}) "
                      f"— {_midi_to_name(max(pitches))} (MIDI {max(pitches)})")

    lines.append("")
    lines.append("— End of extraction —")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  High-level pipeline
# ---------------------------------------------------------------------------

def process_sheet_music(
    input_path: str,
    output_dir: str,
    progress_callback=None,
) -> str:
    """
    Full pipeline: input (PDF or image) → oemer → MusicXML → text report.

    Returns the path to the written .txt file.
    """
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stem = Path(input_path).stem
    ext = Path(input_path).suffix.lower()

    # Determine image list
    if ext == ".pdf":
        if progress_callback:
            progress_callback("Converting PDF pages to images…")
        image_paths = pdf_to_images(input_path)
    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif"):
        image_paths = [input_path]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    all_texts: List[str] = []
    total = len(image_paths)

    for idx, img_path in enumerate(image_paths):
        page_label = f"Page {idx + 1}/{total}" if total > 1 else "Image"
        if progress_callback:
            progress_callback(f"[{page_label}] Running OMR (this may take a few minutes)…")

        # oemer → MusicXML
        mxl_path = _run_oemer_on_image(img_path, output_dir)

        if progress_callback:
            progress_callback(f"[{page_label}] Parsing MusicXML…")

        # MusicXML → text
        text = _parse_musicxml_to_text(mxl_path)

        if total > 1:
            text = f"\n{'#' * 72}\n  PAGE {idx + 1}\n{'#' * 72}\n\n" + text

        all_texts.append(text)

    # Write final text file
    out_txt = os.path.join(output_dir, f"{stem}_sheet_extraction.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(all_texts))

    if progress_callback:
        progress_callback("Done!")

    return out_txt


# ---------------------------------------------------------------------------
#  QThread wrapper for GUI integration
# ---------------------------------------------------------------------------

class SheetReaderThread(QThread):
    """Runs the full OMR pipeline off the main thread."""

    progress = pyqtSignal(str)       # status message
    finished_ok = pyqtSignal(str)    # path to .txt file
    finished_err = pyqtSignal(str)   # error message

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self.input_path = input_path
        self.output_dir = output_dir

    def run(self):
        try:
            txt_path = process_sheet_music(
                self.input_path,
                self.output_dir,
                progress_callback=lambda msg: self.progress.emit(msg),
            )
            self.finished_ok.emit(txt_path)
        except Exception as exc:
            tb = traceback.format_exc()
            self.finished_err.emit(f"{exc}\n\n{tb}")
