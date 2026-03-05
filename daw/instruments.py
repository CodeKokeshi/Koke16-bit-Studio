from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SynthPreset:
    """Per-instrument synthesis parameters that make each instrument unique."""
    attack: float = 0.005      # seconds
    decay: float = 0.0         # seconds  (0 = skip decay, jump to sustain)
    sustain: float = 1.0       # 0.0–1.0  amplitude level during sustain
    release: float = 0.08      # seconds
    vibrato_rate: float = 0.0  # Hz  (0 = off)
    vibrato_depth: float = 0.0 # semitones peak deviation
    filter_cutoff: float = 1.0 # 0.0–1.0  (1.0 = no filter, lower = darker)
    detune: float = 0.0        # cents of pitch offset (chorus-like)


# Default preset (the old hardcoded behaviour — backward compatible)
_DEFAULT = SynthPreset()

# ── Per-instrument presets ──────────────────────────────────────────
SYNTH_PRESETS: dict[str, SynthPreset] = {
    # --- Generic family (raw waveforms, minimal shaping) ---
    "Generic Saw":         SynthPreset(attack=0.008, decay=0.0,  sustain=1.0,  release=0.10, filter_cutoff=0.70),
    "Generic Sine":        SynthPreset(attack=0.005, decay=0.0,  sustain=1.0,  release=0.12),
    "Generic Square":      SynthPreset(attack=0.005, decay=0.0,  sustain=1.0,  release=0.08, filter_cutoff=0.75),
    "Generic Triangle":    SynthPreset(attack=0.005, decay=0.0,  sustain=1.0,  release=0.08),
    "Generic Pulse 25%":   SynthPreset(attack=0.005, decay=0.0,  sustain=1.0,  release=0.08, filter_cutoff=0.78),
    "Generic Pulse 12.5%": SynthPreset(attack=0.005, decay=0.0,  sustain=1.0,  release=0.08, filter_cutoff=0.80),
    "Generic Noise Drum":  SynthPreset(attack=0.001, decay=0.12, sustain=0.0,  release=0.04),

    # --- NES family (crisp, chip-tune punchy) ---
    "NES Square":    SynthPreset(attack=0.002, decay=0.03, sustain=0.85, release=0.06, filter_cutoff=0.82),
    "NES Triangle":  SynthPreset(attack=0.002, decay=0.0,  sustain=1.0,  release=0.05),
    "NES Pulse 25%": SynthPreset(attack=0.002, decay=0.04, sustain=0.80, release=0.06, filter_cutoff=0.80),
    "NES Noise":     SynthPreset(attack=0.001, decay=0.10, sustain=0.0,  release=0.03),

    # --- Gameboy family (DMG-01: 2× pulse, 1× wave, 1× noise) ---
    "Gameboy Square":      SynthPreset(attack=0.001, decay=0.05, sustain=0.75, release=0.05, filter_cutoff=0.78),
    "Gameboy Pulse 12.5%": SynthPreset(attack=0.001, decay=0.05, sustain=0.70, release=0.05, filter_cutoff=0.75),
    "Gameboy Wave":        SynthPreset(attack=0.002, decay=0.04, sustain=0.80, release=0.06, filter_cutoff=0.72),
    "Gameboy Noise":       SynthPreset(attack=0.001, decay=0.08, sustain=0.0,  release=0.03, filter_cutoff=0.70),

    # --- SNES family (richer, more "produced" sound) ---
    "SNES Flute":     SynthPreset(attack=0.06,  decay=0.05, sustain=0.90, release=0.18,
                                   vibrato_rate=5.0, vibrato_depth=0.15, filter_cutoff=0.55),
    "SNES Strings":   SynthPreset(attack=0.10,  decay=0.08, sustain=0.85, release=0.25,
                                   vibrato_rate=4.5, vibrato_depth=0.12, filter_cutoff=0.50),
    "SNES Acoustic":  SynthPreset(attack=0.003, decay=0.15, sustain=0.55, release=0.20,
                                   filter_cutoff=0.60),
    "SNES Trumpet":   SynthPreset(attack=0.03,  decay=0.06, sustain=0.88, release=0.15,
                                   vibrato_rate=5.5, vibrato_depth=0.18, filter_cutoff=0.62),
    "SNES Piano":     SynthPreset(attack=0.002, decay=0.25, sustain=0.35, release=0.30,
                                   filter_cutoff=0.65),
    "SNES Slap Bass": SynthPreset(attack=0.001, decay=0.08, sustain=0.50, release=0.10,
                                   filter_cutoff=0.58),
    "SNES Harp":      SynthPreset(attack=0.002, decay=0.20, sustain=0.40, release=0.35,
                                   vibrato_rate=3.0, vibrato_depth=0.08, filter_cutoff=0.55),
    "SNES Marimba":   SynthPreset(attack=0.001, decay=0.18, sustain=0.15, release=0.12,
                                   filter_cutoff=0.60),
    "SNES Kit":       SynthPreset(attack=0.001, decay=0.12, sustain=0.0,  release=0.04,
                                   filter_cutoff=0.65),

    # --- GBA family (authentic Sappy / m4a engine character) ---
    "GBA Flute":       SynthPreset(attack=0.04,  decay=0.06, sustain=0.88, release=0.15,
                                   vibrato_rate=5.0, vibrato_depth=0.12, filter_cutoff=0.52),
    "GBA Ocarina":     SynthPreset(attack=0.05,  decay=0.04, sustain=0.92, release=0.18,
                                   vibrato_rate=4.0, vibrato_depth=0.10, filter_cutoff=0.48),
    "GBA Vibraphone":  SynthPreset(attack=0.001, decay=0.30, sustain=0.20, release=0.40,
                                   vibrato_rate=6.0, vibrato_depth=0.06, filter_cutoff=0.58),
    "GBA Glockenspiel": SynthPreset(attack=0.001, decay=0.25, sustain=0.10, release=0.30,
                                    filter_cutoff=0.65),
    "GBA Piano":       SynthPreset(attack=0.002, decay=0.22, sustain=0.38, release=0.25,
                                   filter_cutoff=0.60),
    "GBA Strings":     SynthPreset(attack=0.08,  decay=0.06, sustain=0.82, release=0.30,
                                   vibrato_rate=4.5, vibrato_depth=0.10, filter_cutoff=0.45),
    "GBA Fretless Bass": SynthPreset(attack=0.003, decay=0.12, sustain=0.60, release=0.10,
                                     filter_cutoff=0.42),
    "GBA Slap Bass":   SynthPreset(attack=0.001, decay=0.09, sustain=0.45, release=0.08,
                                   filter_cutoff=0.55),
    "GBA Acoustic Guitar": SynthPreset(attack=0.002, decay=0.20, sustain=0.30, release=0.22,
                                       filter_cutoff=0.58),
    "GBA Muted Trumpet": SynthPreset(attack=0.02,  decay=0.05, sustain=0.80, release=0.10,
                                     vibrato_rate=5.5, vibrato_depth=0.15, filter_cutoff=0.50),
    "GBA Steel Drums":  SynthPreset(attack=0.001, decay=0.18, sustain=0.25, release=0.20,
                                    filter_cutoff=0.55),
    "GBA Light Kit":    SynthPreset(attack=0.001, decay=0.10, sustain=0.0,  release=0.03,
                                    filter_cutoff=0.70),
}


def get_preset(instrument_name: str) -> SynthPreset:
    """Look up the synthesis preset for an instrument (falls back to default)."""
    return SYNTH_PRESETS.get(instrument_name, _DEFAULT)


INSTRUMENT_LIBRARY: list[dict[str, str]] = [
    {"name": "Generic Saw", "waveform": "sawtooth", "family": "Generic"},
    {"name": "Generic Sine", "waveform": "sine", "family": "Generic"},
    {"name": "Generic Square", "waveform": "square", "family": "Generic"},
    {"name": "Generic Triangle", "waveform": "triangle", "family": "Generic"},
    {"name": "Generic Pulse 25%", "waveform": "pulse25", "family": "Generic"},
    {"name": "Generic Pulse 12.5%", "waveform": "pulse12", "family": "Generic"},
    {"name": "Generic Noise Drum", "waveform": "noise", "family": "Generic"},
    {"name": "NES Square", "waveform": "square", "family": "NES"},
    {"name": "NES Triangle", "waveform": "triangle", "family": "NES"},
    {"name": "NES Pulse 25%", "waveform": "pulse25", "family": "NES"},
    {"name": "NES Noise", "waveform": "noise", "family": "NES"},
    {"name": "Gameboy Square", "waveform": "square", "family": "Gameboy"},
    {"name": "Gameboy Pulse 12.5%", "waveform": "pulse12", "family": "Gameboy"},
    {"name": "Gameboy Wave", "waveform": "triangle", "family": "Gameboy"},
    {"name": "Gameboy Noise", "waveform": "noise", "family": "Gameboy"},
    {"name": "SNES Flute", "waveform": "sine", "family": "SNES"},
    {"name": "SNES Strings", "waveform": "sawtooth", "family": "SNES"},
    {"name": "SNES Acoustic", "waveform": "triangle", "family": "SNES"},
    {"name": "SNES Trumpet", "waveform": "sawtooth", "family": "SNES"},
    {"name": "SNES Piano", "waveform": "pulse25", "family": "SNES"},
    {"name": "SNES Slap Bass", "waveform": "square", "family": "SNES"},
    {"name": "SNES Harp", "waveform": "sine", "family": "SNES"},
    {"name": "SNES Marimba", "waveform": "triangle", "family": "SNES"},
    {"name": "SNES Kit", "waveform": "noise", "family": "SNES"},
    # GBA family (Sappy / m4a engine)
    {"name": "GBA Flute", "waveform": "sine", "family": "GBA"},
    {"name": "GBA Ocarina", "waveform": "sine", "family": "GBA"},
    {"name": "GBA Vibraphone", "waveform": "triangle", "family": "GBA"},
    {"name": "GBA Glockenspiel", "waveform": "triangle", "family": "GBA"},
    {"name": "GBA Piano", "waveform": "pulse25", "family": "GBA"},
    {"name": "GBA Strings", "waveform": "sawtooth", "family": "GBA"},
    {"name": "GBA Fretless Bass", "waveform": "triangle", "family": "GBA"},
    {"name": "GBA Slap Bass", "waveform": "square", "family": "GBA"},
    {"name": "GBA Acoustic Guitar", "waveform": "triangle", "family": "GBA"},
    {"name": "GBA Muted Trumpet", "waveform": "sawtooth", "family": "GBA"},
    {"name": "GBA Steel Drums", "waveform": "triangle", "family": "GBA"},
    {"name": "GBA Light Kit", "waveform": "noise", "family": "GBA"},
]

# ── Role → Platform → Best instrument mapping ───────────────────────
# Used by "Change Instrument (Auto)" to pick the best-fit instrument
# for a given musical role and hardware platform.

ROLE_INSTRUMENT_MAP: dict[str, dict[str, dict[str, str]]] = {
    "Lead": {
        "Generic":  {"name": "Generic Sine",        "waveform": "sine"},
        "NES":      {"name": "NES Pulse 25%",       "waveform": "pulse25"},
        "Gameboy":  {"name": "Gameboy Square",      "waveform": "square"},
        "SNES":     {"name": "SNES Flute",          "waveform": "sine"},
        "GBA":      {"name": "GBA Flute",           "waveform": "sine"},
    },
    "Bass": {
        "Generic":  {"name": "Generic Triangle",    "waveform": "triangle"},
        "NES":      {"name": "NES Triangle",        "waveform": "triangle"},
        "Gameboy":  {"name": "Gameboy Wave",        "waveform": "triangle"},
        "SNES":     {"name": "SNES Slap Bass",      "waveform": "square"},
        "GBA":      {"name": "GBA Fretless Bass",   "waveform": "triangle"},
    },
    "Harmony": {
        "Generic":  {"name": "Generic Pulse 25%",   "waveform": "pulse25"},
        "NES":      {"name": "NES Square",          "waveform": "square"},
        "Gameboy":  {"name": "Gameboy Square",      "waveform": "square"},
        "SNES":     {"name": "SNES Piano",          "waveform": "pulse25"},
        "GBA":      {"name": "GBA Piano",           "waveform": "pulse25"},
    },
    "Pad / Strings": {
        "Generic":  {"name": "Generic Saw",         "waveform": "sawtooth"},
        "NES":      {"name": "NES Square",          "waveform": "square"},
        "Gameboy":  {"name": "Gameboy Wave",        "waveform": "triangle"},
        "SNES":     {"name": "SNES Strings",        "waveform": "sawtooth"},
        "GBA":      {"name": "GBA Strings",         "waveform": "sawtooth"},
    },
    "Drums": {
        "Generic":  {"name": "Generic Noise Drum",  "waveform": "noise"},
        "NES":      {"name": "NES Noise",           "waveform": "noise"},
        "Gameboy":  {"name": "Gameboy Noise",       "waveform": "noise"},
        "SNES":     {"name": "SNES Kit",            "waveform": "noise"},
        "GBA":      {"name": "GBA Light Kit",       "waveform": "noise"},
    },
    "Melody 2 / Counter": {
        "Generic":  {"name": "Generic Square",      "waveform": "square"},
        "NES":      {"name": "NES Square",          "waveform": "square"},
        "Gameboy":  {"name": "Gameboy Pulse 12.5%", "waveform": "pulse12"},
        "SNES":     {"name": "SNES Trumpet",        "waveform": "sawtooth"},
        "GBA":      {"name": "GBA Muted Trumpet",   "waveform": "sawtooth"},
    },
    "Arpeggio / Bells": {
        "Generic":  {"name": "Generic Triangle",    "waveform": "triangle"},
        "NES":      {"name": "NES Triangle",        "waveform": "triangle"},
        "Gameboy":  {"name": "Gameboy Square",      "waveform": "square"},
        "SNES":     {"name": "SNES Harp",           "waveform": "sine"},
        "GBA":      {"name": "GBA Glockenspiel",    "waveform": "triangle"},
    },
}

AUTO_ROLES = list(ROLE_INSTRUMENT_MAP.keys())
AUTO_PLATFORMS = ["Generic", "NES", "Gameboy", "SNES", "GBA", "Random"]
