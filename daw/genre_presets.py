"""Genre presets for procedural game-music generation.

Defines progression builders, rhythm patterns, section boundaries,
and GenreConfig builder functions for all game-area music styles:

  ❖ Harvest Moon (HM) Town     ❖ Overworld / Field
  ❖ Zelda-like Town            ❖ Boss Battle
  ❖ Dungeon                    ❖ Victory / Fanfare
  ❖ Combat / Battle            ❖ Encounter

Each style has 4 hardware variants: Generic (NES-era), GBA, SNES, Mix.
Every generation is randomised — no two outputs sound the same.
"""

from __future__ import annotations

import random
from daw.generator import (
    GenreConfig,
    ExtraTrackConfig,
)


# ════════════════════════════════════════════════════════════════════
# LOCAL HELPERS
# ════════════════════════════════════════════════════════════════════

_TPB = 4   # ticks per beat (must match generator._TPB)
_BPB = 4   # beats per bar


def _sr(divs: list[float], lens: list[float]) -> list[tuple[int, int]]:
    """Build bar rhythm pattern from beat-relative positions/lengths."""
    return [(int(p * _TPB), max(1, int(d * _TPB))) for p, d in zip(divs, lens)]


def _fill(phrases, dev_pool, bars, ic=None, cc=None):
    """Generic progression assembler with song-form awareness."""
    if ic is None:
        ic = phrases[0][0]
    if cc is None:
        cc = phrases[0][-1]
    if bars <= 8:
        return (random.choice(phrases) + random.choice(phrases))[:bars]
    if bars <= 16:
        p1, p2, p3 = (random.choice(phrases) for _ in range(3))
        form = random.choice(["AABA", "ABAC", "ABBA", "AABC"])
        if form == "AABA":
            prog = p1 + p1 + p2 + p1
        elif form == "ABAC":
            prog = p1 + p2 + p1 + p3
        elif form == "ABBA":
            prog = p1 + p2 + p2 + p1
        else:
            prog = p1 + p1 + p2 + p3
        while len(prog) < bars:
            prog.append(ic)
        return prog[:bars]
    # 32+
    intro = [ic] * 2
    head = random.choice(phrases) + random.choice(phrases)
    dev = random.choice(dev_pool) + random.choice(dev_pool)
    clim = random.choice(phrases) + random.choice(dev_pool)
    reset = random.choice(phrases)
    tail = [cc] * 2
    prog = intro + head + dev + clim + reset + tail
    while len(prog) < bars:
        prog.append(ic)
    return prog[:bars]


# ════════════════════════════════════════════════════════════════════
# INSTRUMENT POOLS (for Mix variants)
# ════════════════════════════════════════════════════════════════════

_ALL_BRIGHT_LEADS = [
    ("Generic Sine", "sine"), ("Generic Triangle", "triangle"),
    ("NES Square", "square"), ("NES Pulse 25%", "pulse25"),
    ("Gameboy Square", "square"), ("Gameboy Wave", "triangle"),
    ("SNES Flute", "sine"), ("SNES Trumpet", "sawtooth"),
    ("GBA Flute", "sine"), ("GBA Ocarina", "sine"),
    ("GBA Muted Trumpet", "sawtooth"),
]
_ALL_DARK_LEADS = [
    ("Generic Sine", "sine"), ("Generic Triangle", "triangle"),
    ("NES Pulse 25%", "pulse25"), ("Gameboy Pulse 12.5%", "pulse12"),
    ("SNES Flute", "sine"), ("SNES Harp", "sine"),
    ("GBA Flute", "sine"), ("GBA Ocarina", "sine"),
]
_ALL_BASS = [
    ("Generic Triangle", "triangle"),
    ("NES Triangle", "triangle"), ("Gameboy Wave", "triangle"),
    ("SNES Slap Bass", "square"),
    ("GBA Fretless Bass", "triangle"), ("GBA Slap Bass", "square"),
]
_ALL_HARMONY = [
    ("Generic Pulse 25%", "pulse25"), ("Generic Square", "square"),
    ("NES Square", "square"), ("Gameboy Square", "square"),
    ("SNES Piano", "pulse25"), ("GBA Piano", "pulse25"),
]
_ALL_DRUMS = [
    ("Generic Noise Drum", "noise"), ("NES Noise", "noise"),
    ("Gameboy Noise", "noise"),
    ("SNES Kit", "noise"), ("GBA Light Kit", "noise"),
]
_ALL_PADS = [
    ("Generic Saw", "sawtooth"), ("SNES Strings", "sawtooth"),
    ("GBA Strings", "sawtooth"), ("Gameboy Wave", "triangle"),
]


# ════════════════════════════════════════════════════════════════════
# SECTION BOUNDARY FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def _pastoral_sections(total_bars: int) -> dict[str, int]:
    """HM Town — gentle flow, no harsh transitions."""
    if total_bars <= 8:
        return {"intro_end": 0, "head_end": 4, "dev_end": 7,
                "climax_end": 7, "reset_start": 7, "silent_tail": total_bars}
    if total_bars <= 16:
        return {"intro_end": 1, "head_end": 6, "dev_end": 11,
                "climax_end": 14, "reset_start": 14, "silent_tail": total_bars}
    return {"intro_end": random.choice([1, 2]),
            "head_end": random.choice([9, 10]),
            "dev_end": random.choice([18, 20]),
            "climax_end": total_bars - random.choice([3, 4]),
            "reset_start": total_bars - random.choice([3, 4]),
            "silent_tail": total_bars - 1}


def _zelda_sections(total_bars: int) -> dict[str, int]:
    """Zelda Town — adventure structure with clear arc."""
    if total_bars <= 8:
        return {"intro_end": 0, "head_end": 4, "dev_end": 7,
                "climax_end": 7, "reset_start": 7, "silent_tail": total_bars}
    if total_bars <= 16:
        return {"intro_end": 1, "head_end": random.choice([5, 6]),
                "dev_end": 11, "climax_end": 14,
                "reset_start": 14, "silent_tail": total_bars}
    _i = random.choice([1, 2])
    _h = _i + random.choice([7, 8])
    _d = _h + random.choice([7, 8])
    return {"intro_end": _i, "head_end": _h, "dev_end": _d,
            "climax_end": total_bars - 4,
            "reset_start": total_bars - 4,
            "silent_tail": total_bars - 1}


def _dungeon_sections(total_bars: int) -> dict[str, int]:
    """Dungeon — long atmospheric intro, tension build."""
    if total_bars <= 8:
        return {"intro_end": 1, "head_end": 4, "dev_end": 7,
                "climax_end": 7, "reset_start": 7, "silent_tail": total_bars}
    if total_bars <= 16:
        return {"intro_end": 2, "head_end": 7, "dev_end": 12,
                "climax_end": 14, "reset_start": 14, "silent_tail": total_bars}
    _i = random.choice([2, 3, 4])
    _h = _i + random.choice([6, 7, 8])
    _d = _h + random.choice([7, 8, 9])
    return {"intro_end": _i, "head_end": _h, "dev_end": _d,
            "climax_end": total_bars - 5,
            "reset_start": total_bars - 5,
            "silent_tail": total_bars - 2}


def _combat_sections(total_bars: int) -> dict[str, int]:
    """Combat — immediate intensity, no rest."""
    if total_bars <= 8:
        return {"intro_end": 0, "head_end": 3, "dev_end": 6,
                "climax_end": 7, "reset_start": 7, "silent_tail": total_bars}
    if total_bars <= 16:
        return {"intro_end": 0, "head_end": 5, "dev_end": 10,
                "climax_end": 14, "reset_start": 14, "silent_tail": total_bars}
    return {"intro_end": 0, "head_end": 8, "dev_end": 18,
            "climax_end": total_bars - 3,
            "reset_start": total_bars - 3,
            "silent_tail": total_bars}


def _boss_sections(total_bars: int) -> dict[str, int]:
    """Boss — dramatic intro, epic build, relentless climax."""
    if total_bars <= 8:
        return {"intro_end": 0, "head_end": 3, "dev_end": 6,
                "climax_end": 7, "reset_start": 7, "silent_tail": total_bars}
    if total_bars <= 16:
        return {"intro_end": 1, "head_end": 5, "dev_end": 10,
                "climax_end": 14, "reset_start": 14, "silent_tail": total_bars}
    _i = random.choice([1, 2])
    _h = _i + random.choice([7, 8])
    _d = _h + random.choice([8, 9, 10])
    return {"intro_end": _i, "head_end": _h, "dev_end": _d,
            "climax_end": total_bars - 3,
            "reset_start": total_bars - 3,
            "silent_tail": total_bars}


def _overworld_sections(total_bars: int) -> dict[str, int]:
    """Overworld — epic journey arc."""
    if total_bars <= 8:
        return {"intro_end": 0, "head_end": 4, "dev_end": 7,
                "climax_end": 7, "reset_start": 7, "silent_tail": total_bars}
    if total_bars <= 16:
        return {"intro_end": 1, "head_end": 6, "dev_end": 11,
                "climax_end": 14, "reset_start": 14, "silent_tail": total_bars}
    _i = random.choice([1, 2])
    _h = _i + random.choice([8, 9])
    _d = _h + random.choice([7, 8])
    return {"intro_end": _i, "head_end": _h, "dev_end": _d,
            "climax_end": total_bars - 4,
            "reset_start": total_bars - 4,
            "silent_tail": total_bars - 1}


def _victory_sections(total_bars: int) -> dict[str, int]:
    """Victory — immediate fanfare, short celebration."""
    if total_bars <= 8:
        return {"intro_end": 0, "head_end": 3, "dev_end": 6,
                "climax_end": 7, "reset_start": 7, "silent_tail": total_bars}
    if total_bars <= 16:
        return {"intro_end": 0, "head_end": 5, "dev_end": 10,
                "climax_end": 14, "reset_start": 14, "silent_tail": total_bars}
    return {"intro_end": 0, "head_end": 8, "dev_end": 18,
            "climax_end": total_bars - 3,
            "reset_start": total_bars - 3,
            "silent_tail": total_bars}


# ════════════════════════════════════════════════════════════════════
# PROGRESSION BUILDERS
# ════════════════════════════════════════════════════════════════════

# ── HM TOWN ────────────────────────────────────────────────────────

def _hm_town_prog(vibe: str, bars: int, family: str = "gba"):
    """Harvest Moon Town — warm, pastoral, gently rolling."""
    M, m, d7, m7, M7, s2, s4, a9 = "maj", "min", "7", "m7", "maj7", "sus2", "sus4", "add9"
    if family == "nes":
        phrases = ([
            [(0,M),(3,M),(4,M),(0,M)], [(0,M),(5,m),(3,M),(4,M)],
            [(0,M),(3,M),(5,m),(0,M)], [(0,M),(4,M),(3,M),(0,M)],
            [(3,M),(0,M),(5,m),(4,M)],
        ] if vibe == "spring" else [
            [(0,M),(5,m),(3,M),(0,M)], [(5,m),(3,M),(0,M),(4,M)],
            [(0,M),(2,m),(5,m),(4,M)], [(0,M),(3,M),(2,m),(0,M)],
        ])
        dev = [[(5,m),(3,M),(4,M),(0,M)], [(3,M),(4,M),(5,m),(0,M)]]
    elif family == "snes":
        phrases = ([
            [(0,a9),(3,M7),(4,s4),(0,M)], [(0,s2),(5,m),(3,a9),(4,s4)],
            [(0,M7),(3,s2),(5,m),(0,a9)], [(3,a9),(0,s2),(5,m),(4,s4)],
        ] if vibe == "spring" else [
            [(0,s2),(5,m7),(3,a9),(0,M7)], [(5,m),(3,s2),(0,a9),(4,s4)],
            [(0,a9),(2,m),(5,m7),(0,s2)], [(0,M7),(3,a9),(2,m),(0,s2)],
        ])
        dev = [[(5,m7),(3,a9),(4,s4),(0,s2)], [(3,s2),(0,a9),(5,m),(4,s4)]]
    else:  # gba
        phrases = ([
            [(0,M7),(3,M7),(1,m7),(4,d7)], [(0,M),(5,m7),(3,M7),(4,d7)],
            [(0,M7),(3,M),(5,m7),(4,d7)], [(3,M7),(0,M),(1,m7),(4,d7)],
        ] if vibe == "spring" else [
            [(0,M7),(5,m7),(3,M7),(4,d7)], [(5,m7),(3,M7),(1,m7),(4,d7)],
            [(0,M),(2,m),(5,m7),(4,d7)], [(0,a9),(5,m7),(4,d7),(0,M7)],
        ])
        dev = [[(5,m7),(3,M7),(1,m7),(4,d7)], [(3,M7),(0,M),(5,m7),(4,d7)]]
    return _fill(phrases, dev, bars, ic=(0,M), cc=(0,M))


# ── ZELDA TOWN ──────────────────────────────────────────────────────

def _zelda_town_prog(vibe: str, bars: int, family: str = "gba"):
    """Zelda-style Town — folk-adventure with modal colour (bVII)."""
    M, m, d7, m7, M7, s2, s4 = "maj", "min", "7", "m7", "maj7", "sus2", "sus4"
    if family == "nes":
        if vibe == "village":
            phrases = [
                [(0,M),(6,M),(3,M),(4,M)], [(0,M),(3,M),(6,M),(0,M)],
                [(0,M),(5,m),(6,M),(3,M)], [(6,M),(3,M),(0,M),(4,M)],
                [(0,M),(4,M),(6,M),(0,M)],
            ]
        else:  # mystery
            phrases = [
                [(0,m),(6,M),(5,M),(0,m)], [(0,m),(3,m),(6,M),(0,m)],
                [(0,m),(5,M),(3,m),(4,M)], [(5,M),(6,M),(0,m),(0,m)],
            ]
        dev = [[(6,M),(3,M),(0,M if vibe == "village" else m),(4,M)],
               [(3,M),(6,M),(4,M),(0,M if vibe == "village" else m)]]
    elif family == "snes":
        if vibe == "village":
            phrases = [
                [(0,s2),(6,M7),(3,s2),(4,s4)], [(0,M7),(3,s2),(6,M),(0,s2)],
                [(0,s2),(5,m7),(6,s2),(3,M7)], [(6,M7),(3,s2),(0,s2),(4,s4)],
            ]
        else:
            phrases = [
                [(0,s4),(6,M7),(5,M),(0,s2)], [(0,m7),(3,s2),(6,s4),(0,m)],
                [(0,s2),(5,M),(3,m7),(4,s4)], [(5,s2),(6,M7),(0,s4),(0,m)],
            ]
        dev = [[(6,M7),(5,s2),(3,s2),(4,s4)], [(3,s2),(6,M7),(0,s2),(4,s4)]]
    else:  # gba
        if vibe == "village":
            phrases = [
                [(0,M7),(6,d7),(3,M7),(4,d7)], [(0,M),(5,m7),(6,d7),(3,M7)],
                [(0,M7),(3,M),(6,d7),(0,M)], [(6,d7),(3,M7),(1,m7),(4,d7)],
            ]
        else:
            phrases = [
                [(0,m7),(6,d7),(5,M7),(0,m)], [(0,m),(3,m7),(6,d7),(4,d7)],
                [(0,m7),(5,M7),(3,m7),(4,d7)], [(6,d7),(5,M7),(0,m7),(4,d7)],
            ]
        dev = [[(6,d7),(5,m7 if vibe == "village" else M7),(3,M7),(4,d7)],
               [(3,M7),(6,d7),(0,M7 if vibe == "village" else m7),(4,d7)]]
    ic = (0, M if vibe == "village" else m)
    return _fill(phrases, dev, bars, ic=ic, cc=ic)


# ── DUNGEON ─────────────────────────────────────────────────────────

def _dungeon_prog(vibe: str, bars: int, family: str = "nes"):
    """Dungeon — dark, foreboding, structured menace."""
    M, m, dm, d7, m7, M7, s2, s4 = "maj", "min", "dim", "7", "m7", "maj7", "sus2", "sus4"
    dm7 = "dim7"
    if family == "nes":
        if vibe == "deep":
            phrases = [
                [(0,m),(5,dm),(4,m),(0,m)], [(0,m),(3,m),(5,dm),(0,m)],
                [(0,m),(0,m),(5,dm),(3,m)], [(3,m),(5,dm),(0,m),(0,m)],
            ]
        else:  # labyrinth
            phrases = [
                [(0,m),(6,M),(5,M),(0,m)], [(0,m),(3,m),(6,M),(5,M)],
                [(0,m),(5,M),(4,m),(0,m)], [(5,M),(6,M),(0,m),(0,m)],
            ]
        dev = [[(5,dm),(3,m),(0,m),(4,m)], [(0,m),(5,dm),(4,m),(3,m)]]
    elif family == "snes":
        if vibe == "deep":
            phrases = [
                [(0,s4),(5,dm),(3,m7),(0,s2)], [(0,m7),(3,s4),(5,dm),(0,m)],
                [(0,s2),(5,dm),(4,s4),(0,m7)], [(3,m7),(0,s2),(5,dm),(0,m)],
            ]
        else:
            phrases = [
                [(0,s4),(6,M7),(5,s2),(0,m7)], [(0,m7),(3,s4),(6,s2),(5,M)],
                [(0,s2),(5,s4),(4,m7),(0,s4)], [(5,s2),(6,M),(0,s4),(0,m)],
            ]
        dev = [[(5,dm),(3,s4),(0,s2),(4,m7)], [(0,s4),(5,dm),(3,m7),(0,s2)]]
    else:  # gba
        if vibe == "deep":
            phrases = [
                [(0,m7),(5,dm7),(3,m7),(0,m)], [(0,m7),(3,m7),(5,dm7),(4,d7)],
                [(0,m),(5,dm7),(4,m7),(0,m7)], [(3,m7),(5,dm7),(4,d7),(0,m)],
            ]
        else:
            phrases = [
                [(0,m7),(6,d7),(5,M7),(0,m)], [(0,m),(3,m7),(6,d7),(5,M)],
                [(0,m7),(5,M7),(4,m7),(0,m7)], [(6,d7),(5,M7),(0,m7),(4,d7)],
            ]
        dev = [[(5,dm7),(3,m7),(0,m7),(4,d7)], [(0,m7),(5,dm7),(4,d7),(0,m)]]
    return _fill(phrases, dev, bars, ic=(0,m), cc=(0,m))


# ── COMBAT ──────────────────────────────────────────────────────────

def _combat_prog(vibe: str, bars: int, family: str = "nes"):
    """Combat / Battle — intense, driving, urgent."""
    M, m, d7, m7, s2, s4 = "maj", "min", "7", "m7", "sus2", "sus4"
    if family == "nes":
        phrases = ([
            [(0,m),(5,M),(6,M),(0,m)], [(0,m),(3,m),(6,M),(0,m)],
            [(0,m),(4,M),(3,m),(0,m)], [(0,m),(6,M),(5,M),(4,M)],
            [(0,m),(0,m),(5,M),(6,M)],
        ] if vibe == "fierce" else [
            [(0,m),(3,m),(4,M),(0,m)], [(0,m),(6,M),(3,m),(4,M)],
            [(0,m),(5,M),(3,m),(0,m)], [(6,M),(5,M),(0,m),(0,m)],
        ])
        dev = [[(5,M),(6,M),(0,m),(4,M)], [(0,m),(3,m),(6,M),(5,M)]]
    elif family == "snes":
        phrases = ([
            [(0,s4),(5,M),(6,s2),(0,m)], [(0,m),(3,s4),(6,M),(0,s2)],
            [(0,s2),(4,s4),(3,m7),(0,m)], [(0,s4),(0,m),(5,s2),(6,M)],
        ] if vibe == "fierce" else [
            [(0,s2),(3,s4),(4,s4),(0,m)], [(0,m),(6,s2),(3,s4),(4,s4)],
            [(0,s4),(5,s2),(3,m7),(0,m)], [(6,s2),(5,s4),(0,s4),(0,m)],
        ])
        dev = [[(5,s2),(6,s4),(0,m),(4,s4)], [(0,s4),(3,m7),(6,s2),(5,s4)]]
    else:  # gba
        phrases = ([
            [(0,m7),(5,d7),(6,d7),(0,m)], [(0,m),(3,m7),(6,d7),(0,m7)],
            [(0,m7),(4,d7),(3,m7),(0,m)], [(0,m),(6,d7),(5,d7),(4,d7)],
        ] if vibe == "fierce" else [
            [(0,m7),(3,m7),(4,d7),(0,m)], [(0,m),(6,d7),(3,m7),(4,d7)],
            [(0,m7),(5,d7),(3,m7),(0,m)], [(6,d7),(5,d7),(0,m7),(0,m)],
        ])
        dev = [[(5,d7),(6,d7),(0,m7),(4,d7)], [(0,m7),(3,m7),(6,d7),(5,d7)]]
    return _fill(phrases, dev, bars, ic=(0,m), cc=(0,m))


# ── ENCOUNTER ───────────────────────────────────────────────────────

def _encounter_prog(vibe: str, bars: int, family: str = "nes"):
    """Encounter — alarming, urgent, short transitional fight opener."""
    M, m, dm, d7, m7, dm7, s2, s4 = "maj", "min", "dim", "7", "m7", "dim7", "sus2", "sus4"
    if family == "nes":
        phrases = [
            [(0,m),(5,dm),(6,M),(4,M)], [(0,m),(6,M),(5,M),(0,m)],
            [(0,m),(4,M),(5,M),(6,M)], [(6,M),(0,m),(5,M),(0,m)],
        ]
    elif family == "snes":
        phrases = [
            [(0,s4),(5,dm),(6,s2),(4,s4)], [(0,m),(6,s4),(5,s2),(0,s4)],
            [(0,s2),(4,s4),(5,s2),(6,s4)], [(6,s4),(0,m),(5,s2),(0,s4)],
        ]
    else:  # gba
        phrases = [
            [(0,m7),(5,dm7),(6,d7),(4,d7)], [(0,m),(6,d7),(5,d7),(0,m7)],
            [(0,m7),(4,d7),(5,d7),(6,d7)], [(6,d7),(0,m7),(5,d7),(0,m)],
        ]
    dev = phrases  # short piece, dev = phrase pool
    return _fill(phrases, dev, bars, ic=(0,m), cc=(0,m))


# ── BOSS BATTLE ─────────────────────────────────────────────────────

def _boss_prog(vibe: str, bars: int, family: str = "nes"):
    """Boss Battle — epic, dramatic, heavy."""
    M, m, dm, ag, d7, m7, dm7, s2, s4 = "maj", "min", "dim", "aug", "7", "m7", "dim7", "sus2", "sus4"
    if family == "nes":
        phrases = ([
            [(0,m),(5,dm),(6,M),(4,M)], [(0,m),(3,m),(5,dm),(6,M)],
            [(0,m),(6,M),(0,m),(5,dm)], [(5,dm),(6,M),(4,M),(0,m)],
            [(0,m),(4,M),(5,dm),(0,m)],
        ] if vibe == "final" else [
            [(0,m),(5,M),(6,M),(0,m)], [(0,m),(3,m),(6,M),(4,M)],
            [(6,M),(5,M),(0,m),(3,m)], [(0,m),(4,M),(6,M),(0,m)],
        ])
        dev = [[(5,dm),(4,M),(6,M),(0,m)], [(0,m),(5,dm),(3,m),(6,M)]]
    elif family == "snes":
        phrases = ([
            [(0,s4),(5,dm),(6,s2),(4,ag)], [(0,m),(3,s4),(5,dm),(6,s2)],
            [(0,s2),(6,s4),(0,m),(5,dm)], [(5,dm),(6,s4),(4,s4),(0,s2)],
        ] if vibe == "final" else [
            [(0,s2),(5,s4),(6,s2),(0,s4)], [(0,m),(3,s4),(6,s2),(4,s4)],
            [(6,s4),(5,s2),(0,s4),(3,m7)], [(0,s4),(4,s4),(6,s2),(0,m)],
        ])
        dev = [[(5,dm),(4,s4),(6,s2),(0,s4)], [(0,s2),(5,dm),(3,s4),(6,s4)]]
    else:  # gba
        phrases = ([
            [(0,m7),(5,dm7),(6,d7),(4,d7)], [(0,m),(3,m7),(5,dm7),(6,d7)],
            [(0,m7),(6,d7),(0,m),(5,dm7)], [(5,dm7),(6,d7),(4,d7),(0,m7)],
        ] if vibe == "final" else [
            [(0,m7),(5,d7),(6,d7),(0,m7)], [(0,m),(3,m7),(6,d7),(4,d7)],
            [(6,d7),(5,d7),(0,m7),(3,m7)], [(0,m7),(4,d7),(6,d7),(0,m)],
        ])
        dev = [[(5,dm7),(4,d7),(6,d7),(0,m7)], [(0,m7),(5,dm7),(3,m7),(6,d7)]]
    return _fill(phrases, dev, bars, ic=(0,m), cc=(0,m))


# ── OVERWORLD ───────────────────────────────────────────────────────

def _overworld_prog(vibe: str, bars: int, family: str = "gba"):
    """Overworld / Field — adventurous, hopeful, wide-open."""
    M, m, d7, m7, M7, s2, s4, a9 = "maj", "min", "7", "m7", "maj7", "sus2", "sus4", "add9"
    if family == "nes":
        phrases = ([
            [(0,M),(4,M),(5,m),(3,M)], [(0,M),(3,M),(6,M),(0,M)],
            [(0,M),(6,M),(3,M),(4,M)], [(3,M),(4,M),(0,M),(0,M)],
            [(0,M),(4,M),(3,M),(0,M)],
        ] if vibe == "heroic" else [
            [(0,M),(5,m),(4,M),(3,M)], [(0,M),(3,M),(5,m),(4,M)],
            [(0,M),(6,M),(5,m),(4,M)], [(6,M),(0,M),(3,M),(4,M)],
        ])
        dev = [[(4,M),(5,m),(3,M),(0,M)], [(6,M),(3,M),(4,M),(0,M)]]
    elif family == "snes":
        phrases = ([
            [(0,a9),(4,s4),(5,m7),(3,M7)], [(0,s2),(3,a9),(6,M7),(0,M)],
            [(0,M7),(6,s2),(3,a9),(4,s4)], [(3,a9),(4,s4),(0,s2),(0,M)],
        ] if vibe == "heroic" else [
            [(0,s2),(5,m7),(4,s4),(3,a9)], [(0,a9),(3,s2),(5,m),(4,s4)],
            [(0,M7),(6,s2),(5,m7),(4,s4)], [(6,s2),(0,a9),(3,s2),(4,s4)],
        ])
        dev = [[(4,s4),(5,m7),(3,a9),(0,s2)], [(6,s2),(3,a9),(4,s4),(0,M7)]]
    else:  # gba
        phrases = ([
            [(0,M7),(4,d7),(5,m7),(3,M7)], [(0,M),(3,M7),(6,d7),(0,M7)],
            [(0,M7),(6,d7),(3,M7),(4,d7)], [(3,M7),(4,d7),(0,M7),(0,M)],
        ] if vibe == "heroic" else [
            [(0,M7),(5,m7),(4,d7),(3,M7)], [(0,M),(3,M7),(5,m7),(4,d7)],
            [(0,M7),(6,d7),(5,m7),(4,d7)], [(6,d7),(0,M7),(3,M7),(4,d7)],
        ])
        dev = [[(4,d7),(5,m7),(3,M7),(0,M7)], [(6,d7),(3,M7),(4,d7),(0,M)]]
    return _fill(phrases, dev, bars, ic=(0,M), cc=(0,M))


# ── VICTORY / FANFARE ──────────────────────────────────────────────

def _victory_prog(vibe: str, bars: int, family: str = "gba"):
    """Victory Fanfare — triumphant, celebratory."""
    M, m, d7, m7, M7, s2, s4, a9 = "maj", "min", "7", "m7", "maj7", "sus2", "sus4", "add9"
    if family == "nes":
        phrases = [
            [(0,M),(3,M),(4,M),(0,M)], [(0,M),(4,M),(0,M),(0,M)],
            [(3,M),(4,M),(0,M),(0,M)], [(0,M),(0,M),(3,M),(4,M)],
        ]
    elif family == "snes":
        phrases = [
            [(0,a9),(3,s2),(4,s4),(0,M)], [(0,s2),(4,s4),(0,a9),(0,M)],
            [(3,a9),(4,s4),(0,s2),(0,M7)], [(0,s2),(0,a9),(3,s2),(4,s4)],
        ]
    else:  # gba
        phrases = [
            [(0,M7),(3,M7),(4,d7),(0,M)], [(0,M),(4,d7),(0,M7),(0,M)],
            [(3,M7),(4,d7),(0,M7),(0,M)], [(0,M),(0,M7),(3,M7),(4,d7)],
        ]
    return _fill(phrases, phrases, bars, ic=(0,M), cc=(0,M))


# ════════════════════════════════════════════════════════════════════
# RHYTHM PRESETS
# ════════════════════════════════════════════════════════════════════

# ── HM Town ────────────────────────────────────────────────────────
_HM_LEAD = [
    _sr([0, 1.5, 3], [1.5, 1.5, 1]),
    _sr([0, 2, 3], [2, 1, 1]),
    _sr([0, 1, 2.5], [1, 1.5, 1.5]),
    _sr([0, 2], [2, 2]),
]
_HM_BASS = [_sr([0, 2], [2, 2])]
_HM_BASS_W = [_sr([0, 1.5, 3], [1.5, 1, 0.5])]
_HM_DRUM = [_sr([0, 2], [0.25, 0.25])]
_HM_HARM = [_sr([0], [4])]

# ── Zelda Town ─────────────────────────────────────────────────────
_ZEL_LEAD = [
    _sr([0, 0.5, 1, 2, 3], [0.5, 0.5, 1, 1, 1]),
    _sr([0, 1, 1.5, 2.5, 3.5], [1, 0.5, 0.75, 0.75, 0.5]),
    _sr([0, 0.75, 1.5, 2.5, 3], [0.5, 0.5, 0.75, 0.5, 1]),
    _sr([0, 1, 2, 2.5, 3.5], [1, 1, 0.5, 0.75, 0.5]),
]
_ZEL_BASS = [_sr([0, 1.5, 3], [1.5, 1.5, 1]), _sr([0, 2], [2, 2])]
_ZEL_DRUM = [
    _sr([0, 1, 2, 3], [0.25] * 4),
    _sr([0, 1, 2, 2.5, 3], [0.25, 0.25, 0.5, 0.25, 0.25]),
]
_ZEL_HARM = [_sr([0, 1.5, 3], [0.5, 0.5, 0.5]), _sr([0, 2], [1, 1])]

# ── Dungeon ────────────────────────────────────────────────────────
_DNG_LEAD = [
    _sr([0, 2.5], [2, 1.5]),
    _sr([0, 1.5, 3], [1.5, 1, 0.5]),
    _sr([0, 1, 3], [1, 1.5, 0.5]),
]
_DNG_BASS = [_sr([0], [4]), _sr([0, 2], [2, 2])]
_DNG_DRUM = [_sr([0, 2], [0.25, 0.25]), _sr([0, 1, 2, 3], [0.25] * 4)]
_DNG_HARM = [_sr([0], [4])]

# ── Combat ─────────────────────────────────────────────────────────
_CMB_LEAD = [
    _sr([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], [0.5] * 8),
    _sr([0, 0.5, 1, 2, 2.5, 3], [0.5, 0.5, 1, 0.5, 0.5, 1]),
    _sr([0, 0.75, 1.5, 2, 2.75, 3.5], [0.5] * 6),
]
_CMB_BASS = [
    _sr([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], [0.5] * 8),
    _sr([0, 1.5, 2, 3.5], [1, 0.5, 1, 0.5]),
]
_CMB_DRUM = [
    _sr([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], [0.25] * 8),
]
_CMB_HARM = [_sr([0, 2], [1, 1]), _sr([0, 1, 2, 3], [0.5] * 4)]

# ── Encounter ──────────────────────────────────────────────────────
_ENC_LEAD = [
    _sr([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5], [0.25] * 10),
    _sr([0, 0.5, 1, 1.5, 2, 3], [0.5] * 6),
]

# ── Boss ───────────────────────────────────────────────────────────
_BSS_LEAD = [
    _sr([0, 0.5, 1, 2, 2.5, 3], [0.5, 0.5, 1, 0.5, 0.5, 1]),
    _sr([0, 1, 1.5, 2.5, 3], [1, 0.5, 0.75, 0.5, 1]),
    _sr([0, 0.5, 1, 1.5, 2, 3], [0.5, 0.5, 0.5, 0.5, 1, 1]),
]
_BSS_BASS = [
    _sr([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], [0.5] * 8),
    _sr([0, 0.75, 1.5, 2, 2.75, 3.5], [0.5] * 6),
]
_BSS_DRUM = [
    _sr([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], [0.25] * 8),
]
_BSS_HARM = [_sr([0, 2], [1.5, 1.5]), _sr([0, 1.5, 3], [1, 1, 1])]

# ── Overworld ──────────────────────────────────────────────────────
_OVW_LEAD = [
    _sr([0, 1, 2, 3], [1, 1, 1, 1]),
    _sr([0, 0.5, 1, 2, 3], [0.5, 0.5, 1, 1, 1]),
    _sr([0, 1.5, 2, 3.5], [1, 0.5, 1, 0.5]),
    _sr([0, 0.5, 1.5, 2.5, 3], [0.5, 1, 1, 0.5, 1]),
]
_OVW_BASS = [_sr([0, 1, 2, 3], [1, 1, 1, 1]), _sr([0, 2], [2, 2])]
_OVW_DRUM = [
    _sr([0, 1, 2, 3], [0.25] * 4),
    _sr([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], [0.25] * 8),
]
_OVW_HARM = [_sr([0], [4]), _sr([0, 2], [2, 2])]

# ── Victory ────────────────────────────────────────────────────────
_VIC_LEAD = [
    _sr([0, 0.5, 1, 2, 3], [0.5, 0.5, 1, 1, 1]),
    _sr([0, 1, 2, 2.5, 3], [1, 1, 0.5, 0.5, 1]),
    _sr([0, 0.5, 1, 1.5, 2, 3], [0.5, 0.5, 0.5, 0.5, 1, 1]),
]
_VIC_BASS = [_sr([0, 2], [2, 2]), _sr([0, 1, 2, 3], [1, 1, 1, 1])]
_VIC_DRUM = [
    _sr([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], [0.25] * 8),
    _sr([0, 1, 2, 3], [0.25] * 4),
]
_VIC_HARM = [_sr([0, 2], [1.5, 1.5]), _sr([0], [4])]


# ════════════════════════════════════════════════════════════════════
# GENRE CONFIG BUILDERS  (32 new genres)
# ════════════════════════════════════════════════════════════════════

# ────────────────────  HM TOWN  ────────────────────────────────────

def _genre_generic_hm_town() -> GenreConfig:
    vibe = random.choice(["spring"] * 3 + ["autumn"])
    return GenreConfig(
        name="Generic - HM Town", bpm_range=(88, 108), bars=32,
        root_choices=[0, 2, 5, 7, 9],
        scale_choices=["major", "pentatonic_major", "lydian"],
        vibe=vibe, progression_family="nes",
        progression_builder=_hm_town_prog, section_builder=_pastoral_sections,
        lead_rhythm=_HM_LEAD, bass_rhythm=_HM_BASS, drum_rhythm=_HM_DRUM,
        harmony_rhythm=_HM_HARM,
        lead_instrument=random.choice([("Generic Sine", "sine"), ("Generic Triangle", "triangle")]),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Generic Square", "square"),
        drum_instrument=("Generic Noise Drum", "noise"),
        lead_vel=(78, 98), bass_vel=(72, 88), harmony_vel=(35, 52), drum_vel=(10, 20),
        lead_mix_range=(0.75, 0.90), bass_mix_range=(0.50, 0.65),
        harmony_mix_range=(0.15, 0.28), drum_mix_range=(0.04, 0.10),
        lead_range=(58, 76), bass_range=(36, 55), harmony_range=(50, 70),
        drum_pitches=[42],
        lead_step_max=2, lead_rest_prob=0.15,
        lead_style="hm_pastoral", bass_style="gentle_arp",
        harmony_style="sustained_pad", drum_style="gentle_brush",
        swing=0.0, lead_doubling=("NES Triangle", "triangle", 0.08),
    )


def _genre_gba_hm_town() -> GenreConfig:
    vibe = random.choice(["spring"] * 3 + ["autumn"])
    return GenreConfig(
        name="GBA - HM Town", bpm_range=(92, 112), bars=32,
        root_choices=[0, 2, 4, 5, 7, 9],
        scale_choices=["major", "lydian", "mixolydian"],
        vibe=vibe, progression_family="gba",
        progression_builder=_hm_town_prog, section_builder=_pastoral_sections,
        lead_rhythm=_HM_LEAD, bass_rhythm=_HM_BASS, drum_rhythm=_HM_DRUM,
        harmony_rhythm=_HM_HARM,
        lead_instrument=random.choice([("GBA Flute", "sine"), ("GBA Ocarina", "sine")]),
        bass_instrument=("GBA Fretless Bass", "triangle"),
        harmony_instrument=("GBA Piano", "pulse25"),
        drum_instrument=("GBA Light Kit", "noise"),
        lead_vel=(82, 100), bass_vel=(75, 90), harmony_vel=(38, 55), drum_vel=(12, 22),
        lead_mix_range=(0.78, 0.92), bass_mix_range=(0.52, 0.68),
        harmony_mix_range=(0.20, 0.35), drum_mix_range=(0.05, 0.12),
        lead_range=(58, 76), bass_range=(36, 55), harmony_range=(50, 70),
        drum_pitches=[42],
        lead_step_max=3, lead_rest_prob=0.12,
        lead_style="hm_pastoral", bass_style="gentle_arp",
        harmony_style="sustained_pad", drum_style="gentle_brush",
        swing=0.05, lead_doubling=("GBA Vibraphone", "triangle", 0.08),
        extra_tracks=[
            ExtraTrackConfig(role="Strings Pad", gen_type="pad",
                             instrument=("GBA Strings", "sawtooth"),
                             vel=(30, 45), pitch_range=(48, 72),
                             mix_range=(0.15, 0.28), rest_prob=0.20, pan=0.20),
            ExtraTrackConfig(role="Guitar Arpeggio", gen_type="arpeggio",
                             instrument=("GBA Acoustic Guitar", "triangle"),
                             vel=(35, 50), pitch_range=(48, 72),
                             mix_range=(0.18, 0.30), rest_prob=0.25, pan=-0.15),
        ],
    )


def _genre_snes_hm_town() -> GenreConfig:
    vibe = random.choice(["spring"] * 3 + ["autumn"])
    return GenreConfig(
        name="SNES - HM Town", bpm_range=(88, 105), bars=32,
        root_choices=[0, 2, 5, 7, 9],
        scale_choices=["major", "lydian", "pentatonic_major"],
        vibe=vibe, progression_family="snes",
        progression_builder=_hm_town_prog, section_builder=_pastoral_sections,
        lead_rhythm=_HM_LEAD, bass_rhythm=_HM_BASS_W, drum_rhythm=_HM_DRUM,
        harmony_rhythm=[],
        lead_instrument=random.choice([("SNES Flute", "sine"), ("SNES Harp", "sine")]),
        bass_instrument=("SNES Slap Bass", "square"),
        harmony_instrument=("SNES Piano", "pulse25"),
        drum_instrument=("SNES Kit", "noise"),
        lead_vel=(80, 100), bass_vel=(70, 88), harmony_vel=(32, 50), drum_vel=(8, 18),
        lead_mix_range=(0.76, 0.90), bass_mix_range=(0.48, 0.62),
        harmony_mix_range=(0.18, 0.32), drum_mix_range=(0.03, 0.08),
        lead_range=(56, 74), bass_range=(36, 55), harmony_range=(48, 68),
        drum_pitches=[42],
        lead_step_max=4, lead_rest_prob=0.10,
        lead_style="hm_pastoral", bass_style="gentle_arp",
        harmony_style="sustained_pad", drum_style="gentle_brush",
        swing=0.0, lead_doubling=("SNES Acoustic", "triangle", 0.06),
        extra_tracks=[
            ExtraTrackConfig(role="Strings Pad", gen_type="pad",
                             instrument=("SNES Strings", "sawtooth"),
                             vel=(28, 42), pitch_range=(48, 72),
                             mix_range=(0.18, 0.32), rest_prob=0.15, pan=0.25),
            ExtraTrackConfig(role="Harp Arpeggio", gen_type="arpeggio",
                             instrument=("SNES Harp", "sine"),
                             vel=(30, 45), pitch_range=(48, 72),
                             mix_range=(0.15, 0.28), rest_prob=0.20, pan=-0.20),
        ],
    )


def _genre_mix_hm_town() -> GenreConfig:
    vibe = random.choice(["spring"] * 3 + ["autumn"])
    pf = random.choice(["nes", "gba", "snes"])
    return GenreConfig(
        name="Mix - HM Town", bpm_range=(88, 112), bars=32,
        root_choices=list(range(12)),
        scale_choices=["major", "lydian", "pentatonic_major", "mixolydian"],
        vibe=vibe, progression_family=pf,
        progression_builder=_hm_town_prog, section_builder=_pastoral_sections,
        lead_rhythm=_HM_LEAD, bass_rhythm=_HM_BASS, drum_rhythm=_HM_DRUM,
        harmony_rhythm=_HM_HARM,
        lead_instrument=random.choice(_ALL_BRIGHT_LEADS),
        bass_instrument=random.choice(_ALL_BASS),
        harmony_instrument=random.choice(_ALL_HARMONY),
        drum_instrument=random.choice(_ALL_DRUMS),
        lead_vel=(78, 100), bass_vel=(72, 90), harmony_vel=(35, 55), drum_vel=(10, 22),
        lead_mix_range=(0.76, 0.92), bass_mix_range=(0.50, 0.66),
        harmony_mix_range=(0.16, 0.30), drum_mix_range=(0.04, 0.10),
        lead_range=(56, 76), bass_range=(36, 55), harmony_range=(48, 70),
        drum_pitches=[42],
        lead_step_max=3, lead_rest_prob=0.12,
        lead_style="hm_pastoral", bass_style="gentle_arp",
        harmony_style="sustained_pad", drum_style="gentle_brush",
        swing=random.choice([0.0, 0.0, 0.05]),
        lead_doubling=random.choice([
            ("NES Triangle", "triangle", 0.08),
            ("SNES Acoustic", "triangle", 0.06),
            ("GBA Vibraphone", "triangle", 0.08),
        ]),
        extra_tracks=[ExtraTrackConfig(role="Pad", gen_type="pad",
                                       instrument=random.choice(_ALL_PADS),
                                       vel=(28, 45), pitch_range=(48, 72),
                                       mix_range=(0.15, 0.28), rest_prob=0.20)],
    )


# ────────────────────  ZELDA TOWN  ─────────────────────────────────

def _genre_generic_zelda_town() -> GenreConfig:
    vibe = random.choice(["village"] * 3 + ["mystery"])
    sc = ["mixolydian", "major", "dorian"] if vibe == "village" else ["natural_minor", "dorian", "phrygian"]
    return GenreConfig(
        name="Generic - Zelda Town", bpm_range=(100, 128), bars=32,
        root_choices=[0, 2, 5, 7, 9, 10],
        scale_choices=sc, vibe=vibe, progression_family="nes",
        progression_builder=_zelda_town_prog, section_builder=_zelda_sections,
        lead_rhythm=_ZEL_LEAD, bass_rhythm=_ZEL_BASS, drum_rhythm=_ZEL_DRUM,
        harmony_rhythm=_ZEL_HARM,
        lead_instrument=random.choice([("Generic Sine", "sine"), ("Generic Triangle", "triangle")]),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Generic Pulse 25%", "pulse25"),
        drum_instrument=("Generic Noise Drum", "noise"),
        lead_vel=(84, 104), bass_vel=(78, 92), harmony_vel=(42, 60), drum_vel=(14, 24),
        lead_mix_range=(0.78, 0.92), bass_mix_range=(0.55, 0.70),
        harmony_mix_range=(0.20, 0.35), drum_mix_range=(0.06, 0.14),
        lead_range=(56, 78), bass_range=(36, 55), harmony_range=(50, 72),
        drum_pitches=[42],
        lead_step_max=3, lead_rest_prob=0.10,
        lead_style="zelda_folk", bass_style="folk_bounce",
        harmony_style="staccato", drum_style="folk_beat",
        swing=0.0, lead_doubling=("NES Square", "square", 0.10),
    )


def _genre_gba_zelda_town() -> GenreConfig:
    vibe = random.choice(["village"] * 3 + ["mystery"])
    sc = ["mixolydian", "major", "dorian"] if vibe == "village" else ["natural_minor", "dorian", "harmonic_minor"]
    return GenreConfig(
        name="GBA - Zelda Town", bpm_range=(104, 130), bars=32,
        root_choices=[0, 2, 3, 5, 7, 9, 10],
        scale_choices=sc, vibe=vibe, progression_family="gba",
        progression_builder=_zelda_town_prog, section_builder=_zelda_sections,
        lead_rhythm=_ZEL_LEAD, bass_rhythm=_ZEL_BASS, drum_rhythm=_ZEL_DRUM,
        harmony_rhythm=_ZEL_HARM,
        lead_instrument=random.choice([("GBA Flute", "sine"), ("GBA Muted Trumpet", "sawtooth")]),
        bass_instrument=random.choice([("GBA Fretless Bass", "triangle"), ("GBA Slap Bass", "square")]),
        harmony_instrument=("GBA Piano", "pulse25"),
        drum_instrument=("GBA Light Kit", "noise"),
        lead_vel=(86, 106), bass_vel=(80, 94), harmony_vel=(44, 64), drum_vel=(16, 28),
        lead_mix_range=(0.80, 0.94), bass_mix_range=(0.56, 0.72),
        harmony_mix_range=(0.24, 0.38), drum_mix_range=(0.08, 0.16),
        lead_range=(56, 78), bass_range=(36, 55), harmony_range=(50, 72),
        drum_pitches=[38, 42],
        lead_step_max=3, lead_rest_prob=0.08,
        lead_style="zelda_folk", bass_style="folk_bounce",
        harmony_style="staccato", drum_style="folk_beat",
        swing=0.08, lead_doubling=("GBA Glockenspiel", "triangle", 0.10),
        extra_tracks=[
            ExtraTrackConfig(role="Strings Pad", gen_type="pad",
                             instrument=("GBA Strings", "sawtooth"),
                             vel=(35, 50), pitch_range=(48, 72),
                             mix_range=(0.18, 0.30), rest_prob=0.20, pan=0.20),
        ],
    )


def _genre_snes_zelda_town() -> GenreConfig:
    vibe = random.choice(["village"] * 3 + ["mystery"])
    sc = ["mixolydian", "lydian", "major"] if vibe == "village" else ["natural_minor", "dorian", "melodic_minor"]
    return GenreConfig(
        name="SNES - Zelda Town", bpm_range=(100, 125), bars=32,
        root_choices=[0, 2, 3, 5, 7, 8, 10],
        scale_choices=sc, vibe=vibe, progression_family="snes",
        progression_builder=_zelda_town_prog, section_builder=_zelda_sections,
        lead_rhythm=_ZEL_LEAD, bass_rhythm=_ZEL_BASS, drum_rhythm=_ZEL_DRUM,
        harmony_rhythm=[],
        lead_instrument=random.choice([("SNES Flute", "sine"), ("SNES Trumpet", "sawtooth")]),
        bass_instrument=("SNES Slap Bass", "square"),
        harmony_instrument=("SNES Piano", "pulse25"),
        drum_instrument=("SNES Kit", "noise"),
        lead_vel=(84, 106), bass_vel=(78, 92), harmony_vel=(38, 58), drum_vel=(12, 22),
        lead_mix_range=(0.78, 0.92), bass_mix_range=(0.52, 0.68),
        harmony_mix_range=(0.20, 0.35), drum_mix_range=(0.05, 0.12),
        lead_range=(56, 78), bass_range=(36, 55), harmony_range=(48, 70),
        drum_pitches=[42],
        lead_step_max=4, lead_rest_prob=0.08,
        lead_style="zelda_folk", bass_style="folk_bounce",
        harmony_style="sustained_pad", drum_style="folk_beat",
        swing=0.05, lead_doubling=("SNES Harp", "sine", 0.08),
        extra_tracks=[
            ExtraTrackConfig(role="Strings Pad", gen_type="pad",
                             instrument=("SNES Strings", "sawtooth"),
                             vel=(30, 48), pitch_range=(48, 72),
                             mix_range=(0.18, 0.32), rest_prob=0.18, pan=0.25),
            ExtraTrackConfig(role="Harp Arpeggio", gen_type="arpeggio",
                             instrument=("SNES Harp", "sine"),
                             vel=(32, 48), pitch_range=(48, 72),
                             mix_range=(0.15, 0.28), rest_prob=0.22, pan=-0.20),
        ],
    )


def _genre_mix_zelda_town() -> GenreConfig:
    vibe = random.choice(["village"] * 3 + ["mystery"])
    pf = random.choice(["nes", "gba", "snes"])
    sc = ["mixolydian", "major", "dorian"] if vibe == "village" else ["natural_minor", "dorian", "phrygian"]
    return GenreConfig(
        name="Mix - Zelda Town", bpm_range=(100, 130), bars=32,
        root_choices=list(range(12)), scale_choices=sc,
        vibe=vibe, progression_family=pf,
        progression_builder=_zelda_town_prog, section_builder=_zelda_sections,
        lead_rhythm=_ZEL_LEAD, bass_rhythm=_ZEL_BASS, drum_rhythm=_ZEL_DRUM,
        harmony_rhythm=_ZEL_HARM,
        lead_instrument=random.choice(_ALL_BRIGHT_LEADS),
        bass_instrument=random.choice(_ALL_BASS),
        harmony_instrument=random.choice(_ALL_HARMONY),
        drum_instrument=random.choice(_ALL_DRUMS),
        lead_vel=(84, 106), bass_vel=(78, 92), harmony_vel=(40, 60), drum_vel=(14, 26),
        lead_mix_range=(0.78, 0.93), bass_mix_range=(0.54, 0.70),
        harmony_mix_range=(0.20, 0.35), drum_mix_range=(0.06, 0.14),
        lead_range=(56, 78), bass_range=(36, 55), harmony_range=(48, 72),
        drum_pitches=[38, 42] if pf == "gba" else [42],
        lead_step_max=3, lead_rest_prob=0.10, lead_style="zelda_folk",
        bass_style="folk_bounce", harmony_style="staccato", drum_style="folk_beat",
        swing=random.choice([0.0, 0.05, 0.08]),
        extra_tracks=[ExtraTrackConfig(role="Pad", gen_type="pad",
                                       instrument=random.choice(_ALL_PADS),
                                       vel=(32, 48), pitch_range=(48, 72),
                                       mix_range=(0.15, 0.28), rest_prob=0.20)],
    )


# ────────────────────  DUNGEON  ────────────────────────────────────

def _genre_generic_dungeon() -> GenreConfig:
    vibe = random.choice(["deep", "deep", "labyrinth"])
    return GenreConfig(
        name="Generic - Dungeon", bpm_range=(68, 96), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["natural_minor", "phrygian", "harmonic_minor"] if vibe == "deep"
                 else ["natural_minor", "dorian", "harmonic_minor"],
        vibe=vibe, progression_family="nes",
        progression_builder=_dungeon_prog, section_builder=_dungeon_sections,
        lead_rhythm=_DNG_LEAD, bass_rhythm=_DNG_BASS, drum_rhythm=_DNG_DRUM,
        harmony_rhythm=_DNG_HARM,
        lead_instrument=random.choice([("Generic Sine", "sine"), ("Generic Triangle", "triangle")]),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Generic Pulse 25%", "pulse25"),
        drum_instrument=("Generic Noise Drum", "noise"),
        lead_vel=(58, 80), bass_vel=(62, 78), harmony_vel=(28, 45), drum_vel=(10, 20),
        lead_mix_range=(0.62, 0.80), bass_mix_range=(0.48, 0.62),
        harmony_mix_range=(0.14, 0.26), drum_mix_range=(0.04, 0.10),
        lead_range=(48, 68), bass_range=(32, 50), harmony_range=(44, 64),
        drum_pitches=[42],
        lead_step_max=3, lead_rest_prob=0.28, lead_style="dungeon_dark",
        bass_style="drone_pedal", harmony_style="dark_drone", drum_style="dungeon_march",
        swing=0.0,
    )


def _genre_gba_dungeon() -> GenreConfig:
    vibe = random.choice(["deep", "deep", "labyrinth"])
    return GenreConfig(
        name="GBA - Dungeon", bpm_range=(72, 100), bars=32,
        root_choices=[0, 1, 3, 5, 6, 8, 10],
        scale_choices=["harmonic_minor", "phrygian"] if vibe == "deep"
                 else ["natural_minor", "dorian", "harmonic_minor"],
        vibe=vibe, progression_family="gba",
        progression_builder=_dungeon_prog, section_builder=_dungeon_sections,
        lead_rhythm=_DNG_LEAD, bass_rhythm=_DNG_BASS, drum_rhythm=_DNG_DRUM,
        harmony_rhythm=_DNG_HARM,
        lead_instrument=random.choice([("GBA Flute", "sine"), ("GBA Ocarina", "sine")]),
        bass_instrument=random.choice([("GBA Fretless Bass", "triangle"), ("GBA Slap Bass", "square")]),
        harmony_instrument=("GBA Piano", "pulse25"),
        drum_instrument=("GBA Light Kit", "noise"),
        lead_vel=(62, 85), bass_vel=(65, 82), harmony_vel=(30, 48), drum_vel=(12, 22),
        lead_mix_range=(0.65, 0.82), bass_mix_range=(0.50, 0.65),
        harmony_mix_range=(0.16, 0.28), drum_mix_range=(0.05, 0.12),
        lead_range=(50, 70), bass_range=(34, 52), harmony_range=(46, 66),
        drum_pitches=[38, 42],
        lead_step_max=3, lead_rest_prob=0.24, lead_style="dungeon_dark",
        bass_style="drone_pedal", harmony_style="dark_drone", drum_style="dungeon_march",
        swing=0.05,
        extra_tracks=[
            ExtraTrackConfig(role="Strings Pad", gen_type="pad",
                             instrument=("GBA Strings", "sawtooth"),
                             vel=(25, 40), pitch_range=(42, 64),
                             mix_range=(0.12, 0.24), rest_prob=0.35, pan=0.20),
        ],
    )


def _genre_snes_dungeon() -> GenreConfig:
    vibe = random.choice(["deep", "deep", "labyrinth"])
    return GenreConfig(
        name="SNES - Dungeon", bpm_range=(65, 92), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["phrygian", "harmonic_minor"] if vibe == "deep"
                 else ["natural_minor", "dorian", "melodic_minor"],
        vibe=vibe, progression_family="snes",
        progression_builder=_dungeon_prog, section_builder=_dungeon_sections,
        lead_rhythm=_DNG_LEAD, bass_rhythm=_DNG_BASS, drum_rhythm=_DNG_DRUM,
        harmony_rhythm=[],
        lead_instrument=random.choice([("SNES Flute", "sine"), ("SNES Harp", "sine")]),
        bass_instrument=("SNES Slap Bass", "square"),
        harmony_instrument=("SNES Piano", "pulse25"),
        drum_instrument=("SNES Kit", "noise"),
        lead_vel=(55, 78), bass_vel=(58, 75), harmony_vel=(25, 42), drum_vel=(8, 16),
        lead_mix_range=(0.60, 0.78), bass_mix_range=(0.46, 0.60),
        harmony_mix_range=(0.15, 0.28), drum_mix_range=(0.03, 0.08),
        lead_range=(48, 70), bass_range=(32, 50), harmony_range=(44, 64),
        drum_pitches=[42],
        lead_step_max=4, lead_rest_prob=0.26, lead_style="dungeon_dark",
        bass_style="drone_pedal", harmony_style="sustained_pad", drum_style="dungeon_march",
        swing=0.0, lead_doubling=("SNES Acoustic", "triangle", 0.06),
        extra_tracks=[
            ExtraTrackConfig(role="Strings Pad", gen_type="pad",
                             instrument=("SNES Strings", "sawtooth"),
                             vel=(22, 38), pitch_range=(42, 64),
                             mix_range=(0.14, 0.26), rest_prob=0.30, pan=0.25),
        ],
    )


def _genre_mix_dungeon() -> GenreConfig:
    vibe = random.choice(["deep", "deep", "labyrinth"])
    pf = random.choice(["nes", "gba", "snes"])
    return GenreConfig(
        name="Mix - Dungeon", bpm_range=(65, 100), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["natural_minor", "harmonic_minor", "phrygian", "dorian"],
        vibe=vibe, progression_family=pf,
        progression_builder=_dungeon_prog, section_builder=_dungeon_sections,
        lead_rhythm=_DNG_LEAD, bass_rhythm=_DNG_BASS, drum_rhythm=_DNG_DRUM,
        harmony_rhythm=_DNG_HARM,
        lead_instrument=random.choice(_ALL_DARK_LEADS),
        bass_instrument=random.choice(_ALL_BASS),
        harmony_instrument=random.choice(_ALL_HARMONY),
        drum_instrument=random.choice(_ALL_DRUMS),
        lead_vel=(58, 82), bass_vel=(60, 78), harmony_vel=(26, 45), drum_vel=(10, 20),
        lead_mix_range=(0.62, 0.80), bass_mix_range=(0.48, 0.64),
        harmony_mix_range=(0.14, 0.28), drum_mix_range=(0.04, 0.10),
        lead_range=(48, 70), bass_range=(32, 52), harmony_range=(44, 66),
        drum_pitches=[42],
        lead_step_max=3, lead_rest_prob=0.26, lead_style="dungeon_dark",
        bass_style="drone_pedal", harmony_style="dark_drone", drum_style="dungeon_march",
        swing=0.0,
    )


# ────────────────────  COMBAT  ─────────────────────────────────────

def _genre_generic_combat() -> GenreConfig:
    vibe = random.choice(["fierce", "fierce", "skirmish"])
    return GenreConfig(
        name="Generic - Combat", bpm_range=(142, 168), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["natural_minor", "harmonic_minor", "dorian"],
        vibe=vibe, progression_family="nes",
        progression_builder=_combat_prog, section_builder=_combat_sections,
        lead_rhythm=_CMB_LEAD, bass_rhythm=_CMB_BASS, drum_rhythm=_CMB_DRUM,
        harmony_rhythm=_CMB_HARM,
        lead_instrument=random.choice([("Generic Square", "square"), ("Generic Saw", "sawtooth")]),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Generic Pulse 25%", "pulse25"),
        drum_instrument=("Generic Noise Drum", "noise"),
        lead_vel=(92, 118), bass_vel=(88, 105), harmony_vel=(55, 75), drum_vel=(28, 40),
        lead_mix_range=(0.82, 0.96), bass_mix_range=(0.60, 0.78),
        harmony_mix_range=(0.30, 0.45), drum_mix_range=(0.15, 0.28),
        lead_range=(54, 80), bass_range=(30, 50), harmony_range=(48, 72),
        drum_pitches=[36, 38, 42],
        lead_step_max=4, lead_rest_prob=0.04, lead_style="combat_intense",
        bass_style="driving_eighth", harmony_style="power_stab", drum_style="combat_drive",
        swing=0.0,
    )


def _genre_gba_combat() -> GenreConfig:
    vibe = random.choice(["fierce", "fierce", "skirmish"])
    return GenreConfig(
        name="GBA - Combat", bpm_range=(145, 170), bars=32,
        root_choices=[0, 1, 3, 5, 6, 8, 10],
        scale_choices=["harmonic_minor", "natural_minor", "dorian"],
        vibe=vibe, progression_family="gba",
        progression_builder=_combat_prog, section_builder=_combat_sections,
        lead_rhythm=_CMB_LEAD, bass_rhythm=_CMB_BASS, drum_rhythm=_CMB_DRUM,
        harmony_rhythm=_CMB_HARM,
        lead_instrument=random.choice([("GBA Muted Trumpet", "sawtooth"), ("GBA Flute", "sine")]),
        bass_instrument=random.choice([("GBA Slap Bass", "square"), ("GBA Fretless Bass", "triangle")]),
        harmony_instrument=("GBA Piano", "pulse25"),
        drum_instrument=("GBA Light Kit", "noise"),
        lead_vel=(94, 120), bass_vel=(90, 108), harmony_vel=(58, 78), drum_vel=(30, 42),
        lead_mix_range=(0.84, 0.96), bass_mix_range=(0.62, 0.80),
        harmony_mix_range=(0.32, 0.48), drum_mix_range=(0.16, 0.30),
        lead_range=(54, 80), bass_range=(30, 50), harmony_range=(48, 72),
        drum_pitches=[36, 38, 42],
        lead_step_max=4, lead_rest_prob=0.03, lead_style="combat_intense",
        bass_style="driving_eighth", harmony_style="power_stab", drum_style="combat_drive",
        swing=0.0,
        extra_tracks=[
            ExtraTrackConfig(role="Strings Pad", gen_type="pad",
                             instrument=("GBA Strings", "sawtooth"),
                             vel=(45, 62), pitch_range=(42, 66),
                             mix_range=(0.20, 0.32), rest_prob=0.15, pan=0.15),
        ],
    )


def _genre_snes_combat() -> GenreConfig:
    vibe = random.choice(["fierce", "fierce", "skirmish"])
    return GenreConfig(
        name="SNES - Combat", bpm_range=(140, 165), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["harmonic_minor", "natural_minor", "melodic_minor"],
        vibe=vibe, progression_family="snes",
        progression_builder=_combat_prog, section_builder=_combat_sections,
        lead_rhythm=_CMB_LEAD, bass_rhythm=_CMB_BASS, drum_rhythm=_CMB_DRUM,
        harmony_rhythm=_CMB_HARM,
        lead_instrument=random.choice([("SNES Trumpet", "sawtooth"), ("SNES Flute", "sine")]),
        bass_instrument=("SNES Slap Bass", "square"),
        harmony_instrument=("SNES Piano", "pulse25"),
        drum_instrument=("SNES Kit", "noise"),
        lead_vel=(92, 118), bass_vel=(88, 105), harmony_vel=(52, 72), drum_vel=(26, 38),
        lead_mix_range=(0.82, 0.96), bass_mix_range=(0.60, 0.78),
        harmony_mix_range=(0.28, 0.42), drum_mix_range=(0.14, 0.26),
        lead_range=(54, 80), bass_range=(30, 50), harmony_range=(48, 72),
        drum_pitches=[36, 38, 42],
        lead_step_max=5, lead_rest_prob=0.03, lead_style="combat_intense",
        bass_style="driving_eighth", harmony_style="power_stab", drum_style="combat_drive",
        swing=0.0,
        extra_tracks=[
            ExtraTrackConfig(role="Strings Pad", gen_type="pad",
                             instrument=("SNES Strings", "sawtooth"),
                             vel=(42, 60), pitch_range=(42, 66),
                             mix_range=(0.20, 0.34), rest_prob=0.12, pan=0.20),
        ],
    )


def _genre_mix_combat() -> GenreConfig:
    vibe = random.choice(["fierce", "fierce", "skirmish"])
    pf = random.choice(["nes", "gba", "snes"])
    return GenreConfig(
        name="Mix - Combat", bpm_range=(140, 170), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["natural_minor", "harmonic_minor", "dorian"],
        vibe=vibe, progression_family=pf,
        progression_builder=_combat_prog, section_builder=_combat_sections,
        lead_rhythm=_CMB_LEAD, bass_rhythm=_CMB_BASS, drum_rhythm=_CMB_DRUM,
        harmony_rhythm=_CMB_HARM,
        lead_instrument=random.choice(_ALL_BRIGHT_LEADS),
        bass_instrument=random.choice(_ALL_BASS),
        harmony_instrument=random.choice(_ALL_HARMONY),
        drum_instrument=random.choice(_ALL_DRUMS),
        lead_vel=(92, 118), bass_vel=(88, 105), harmony_vel=(55, 75), drum_vel=(28, 40),
        lead_mix_range=(0.82, 0.96), bass_mix_range=(0.60, 0.78),
        harmony_mix_range=(0.30, 0.45), drum_mix_range=(0.15, 0.28),
        lead_range=(54, 80), bass_range=(30, 50), harmony_range=(48, 72),
        drum_pitches=[36, 38, 42],
        lead_step_max=4, lead_rest_prob=0.04, lead_style="combat_intense",
        bass_style="driving_eighth", harmony_style="power_stab", drum_style="combat_drive",
        swing=0.0,
    )


# ────────────────────  ENCOUNTER  ──────────────────────────────────

def _genre_generic_encounter() -> GenreConfig:
    return GenreConfig(
        name="Generic - Encounter", bpm_range=(155, 180), bars=16,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["harmonic_minor", "natural_minor"],
        vibe="alarm", progression_family="nes",
        progression_builder=_encounter_prog, section_builder=_combat_sections,
        lead_rhythm=_ENC_LEAD, bass_rhythm=_CMB_BASS, drum_rhythm=_CMB_DRUM,
        harmony_rhythm=_CMB_HARM,
        lead_instrument=random.choice([("Generic Square", "square"), ("Generic Saw", "sawtooth")]),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Generic Pulse 25%", "pulse25"),
        drum_instrument=("Generic Noise Drum", "noise"),
        lead_vel=(96, 120), bass_vel=(90, 108), harmony_vel=(58, 78), drum_vel=(30, 42),
        lead_mix_range=(0.85, 0.98), bass_mix_range=(0.62, 0.80),
        harmony_mix_range=(0.32, 0.48), drum_mix_range=(0.18, 0.30),
        lead_range=(56, 84), bass_range=(30, 50), harmony_range=(48, 72),
        drum_pitches=[36, 38, 42],
        lead_step_max=5, lead_rest_prob=0.02, lead_style="encounter_alarm",
        bass_style="driving_eighth", harmony_style="power_stab", drum_style="combat_drive",
        swing=0.0,
    )


def _genre_gba_encounter() -> GenreConfig:
    return GenreConfig(
        name="GBA - Encounter", bpm_range=(158, 182), bars=16,
        root_choices=[0, 1, 3, 5, 6, 8, 10],
        scale_choices=["harmonic_minor", "natural_minor", "phrygian"],
        vibe="alarm", progression_family="gba",
        progression_builder=_encounter_prog, section_builder=_combat_sections,
        lead_rhythm=_ENC_LEAD, bass_rhythm=_CMB_BASS, drum_rhythm=_CMB_DRUM,
        harmony_rhythm=_CMB_HARM,
        lead_instrument=random.choice([("GBA Muted Trumpet", "sawtooth"), ("GBA Flute", "sine")]),
        bass_instrument=("GBA Slap Bass", "square"),
        harmony_instrument=("GBA Piano", "pulse25"),
        drum_instrument=("GBA Light Kit", "noise"),
        lead_vel=(98, 122), bass_vel=(92, 110), harmony_vel=(60, 80), drum_vel=(32, 44),
        lead_mix_range=(0.86, 0.98), bass_mix_range=(0.64, 0.82),
        harmony_mix_range=(0.34, 0.50), drum_mix_range=(0.20, 0.32),
        lead_range=(56, 84), bass_range=(30, 50), harmony_range=(48, 72),
        drum_pitches=[36, 38, 42],
        lead_step_max=5, lead_rest_prob=0.02, lead_style="encounter_alarm",
        bass_style="driving_eighth", harmony_style="power_stab", drum_style="combat_drive",
        swing=0.0,
    )


def _genre_snes_encounter() -> GenreConfig:
    return GenreConfig(
        name="SNES - Encounter", bpm_range=(152, 178), bars=16,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["harmonic_minor", "phrygian", "natural_minor"],
        vibe="alarm", progression_family="snes",
        progression_builder=_encounter_prog, section_builder=_combat_sections,
        lead_rhythm=_ENC_LEAD, bass_rhythm=_CMB_BASS, drum_rhythm=_CMB_DRUM,
        harmony_rhythm=_CMB_HARM,
        lead_instrument=random.choice([("SNES Trumpet", "sawtooth"), ("SNES Flute", "sine")]),
        bass_instrument=("SNES Slap Bass", "square"),
        harmony_instrument=("SNES Piano", "pulse25"),
        drum_instrument=("SNES Kit", "noise"),
        lead_vel=(96, 120), bass_vel=(90, 108), harmony_vel=(56, 76), drum_vel=(30, 42),
        lead_mix_range=(0.85, 0.98), bass_mix_range=(0.62, 0.80),
        harmony_mix_range=(0.30, 0.46), drum_mix_range=(0.18, 0.30),
        lead_range=(56, 84), bass_range=(30, 50), harmony_range=(48, 72),
        drum_pitches=[36, 38, 42],
        lead_step_max=5, lead_rest_prob=0.02, lead_style="encounter_alarm",
        bass_style="driving_eighth", harmony_style="power_stab", drum_style="combat_drive",
        swing=0.0,
    )


def _genre_mix_encounter() -> GenreConfig:
    pf = random.choice(["nes", "gba", "snes"])
    return GenreConfig(
        name="Mix - Encounter", bpm_range=(152, 182), bars=16,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["harmonic_minor", "natural_minor", "phrygian"],
        vibe="alarm", progression_family=pf,
        progression_builder=_encounter_prog, section_builder=_combat_sections,
        lead_rhythm=_ENC_LEAD, bass_rhythm=_CMB_BASS, drum_rhythm=_CMB_DRUM,
        harmony_rhythm=_CMB_HARM,
        lead_instrument=random.choice(_ALL_BRIGHT_LEADS),
        bass_instrument=random.choice(_ALL_BASS),
        harmony_instrument=random.choice(_ALL_HARMONY),
        drum_instrument=random.choice(_ALL_DRUMS),
        lead_vel=(96, 122), bass_vel=(90, 108), harmony_vel=(58, 78), drum_vel=(30, 42),
        lead_mix_range=(0.85, 0.98), bass_mix_range=(0.62, 0.80),
        harmony_mix_range=(0.32, 0.48), drum_mix_range=(0.18, 0.30),
        lead_range=(56, 84), bass_range=(30, 50), harmony_range=(48, 72),
        drum_pitches=[36, 38, 42],
        lead_step_max=5, lead_rest_prob=0.02, lead_style="encounter_alarm",
        bass_style="driving_eighth", harmony_style="power_stab", drum_style="combat_drive",
        swing=0.0,
    )


# ────────────────────  BOSS BATTLE  ────────────────────────────────

def _genre_generic_boss() -> GenreConfig:
    vibe = random.choice(["final", "final", "mid"])
    return GenreConfig(
        name="Generic - Boss Battle", bpm_range=(132, 158), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["harmonic_minor", "phrygian", "diminished"] if vibe == "final"
                 else ["natural_minor", "harmonic_minor", "dorian"],
        vibe=vibe, progression_family="nes",
        progression_builder=_boss_prog, section_builder=_boss_sections,
        lead_rhythm=_BSS_LEAD, bass_rhythm=_BSS_BASS, drum_rhythm=_BSS_DRUM,
        harmony_rhythm=_BSS_HARM,
        lead_instrument=random.choice([("Generic Square", "square"), ("Generic Saw", "sawtooth")]),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Generic Pulse 25%", "pulse25"),
        drum_instrument=("Generic Noise Drum", "noise"),
        lead_vel=(96, 122), bass_vel=(92, 110), harmony_vel=(60, 80), drum_vel=(32, 45),
        lead_mix_range=(0.84, 0.98), bass_mix_range=(0.64, 0.82),
        harmony_mix_range=(0.34, 0.50), drum_mix_range=(0.18, 0.32),
        lead_range=(52, 82), bass_range=(28, 48), harmony_range=(46, 72),
        drum_pitches=[36, 38, 42, 46],
        lead_step_max=5, lead_rest_prob=0.03, lead_style="boss_epic",
        bass_style="boss_heavy", harmony_style="boss_dramatic", drum_style="boss_pound",
        swing=0.0,
    )


def _genre_gba_boss() -> GenreConfig:
    vibe = random.choice(["final", "final", "mid"])
    return GenreConfig(
        name="GBA - Boss Battle", bpm_range=(135, 162), bars=32,
        root_choices=[0, 1, 3, 5, 6, 8, 10],
        scale_choices=["harmonic_minor", "phrygian"] if vibe == "final"
                 else ["natural_minor", "harmonic_minor"],
        vibe=vibe, progression_family="gba",
        progression_builder=_boss_prog, section_builder=_boss_sections,
        lead_rhythm=_BSS_LEAD, bass_rhythm=_BSS_BASS, drum_rhythm=_BSS_DRUM,
        harmony_rhythm=_BSS_HARM,
        lead_instrument=random.choice([("GBA Muted Trumpet", "sawtooth"), ("GBA Flute", "sine")]),
        bass_instrument=random.choice([("GBA Slap Bass", "square"), ("GBA Fretless Bass", "triangle")]),
        harmony_instrument=("GBA Piano", "pulse25"),
        drum_instrument=("GBA Light Kit", "noise"),
        lead_vel=(98, 124), bass_vel=(94, 112), harmony_vel=(62, 82), drum_vel=(34, 46),
        lead_mix_range=(0.86, 0.98), bass_mix_range=(0.66, 0.84),
        harmony_mix_range=(0.36, 0.52), drum_mix_range=(0.20, 0.34),
        lead_range=(52, 82), bass_range=(28, 48), harmony_range=(46, 72),
        drum_pitches=[36, 38, 42, 46],
        lead_step_max=5, lead_rest_prob=0.02, lead_style="boss_epic",
        bass_style="boss_heavy", harmony_style="boss_dramatic", drum_style="boss_pound",
        swing=0.0,
        extra_tracks=[
            ExtraTrackConfig(role="Strings Pad", gen_type="pad",
                             instrument=("GBA Strings", "sawtooth"),
                             vel=(48, 65), pitch_range=(42, 66),
                             mix_range=(0.22, 0.36), rest_prob=0.10, pan=0.15),
        ],
    )


def _genre_snes_boss() -> GenreConfig:
    vibe = random.choice(["final", "final", "mid"])
    return GenreConfig(
        name="SNES - Boss Battle", bpm_range=(130, 158), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["harmonic_minor", "phrygian", "diminished"] if vibe == "final"
                 else ["natural_minor", "harmonic_minor", "melodic_minor"],
        vibe=vibe, progression_family="snes",
        progression_builder=_boss_prog, section_builder=_boss_sections,
        lead_rhythm=_BSS_LEAD, bass_rhythm=_BSS_BASS, drum_rhythm=_BSS_DRUM,
        harmony_rhythm=_BSS_HARM,
        lead_instrument=random.choice([("SNES Trumpet", "sawtooth"), ("SNES Flute", "sine")]),
        bass_instrument=("SNES Slap Bass", "square"),
        harmony_instrument=("SNES Piano", "pulse25"),
        drum_instrument=("SNES Kit", "noise"),
        lead_vel=(96, 122), bass_vel=(92, 110), harmony_vel=(58, 78), drum_vel=(30, 42),
        lead_mix_range=(0.84, 0.98), bass_mix_range=(0.64, 0.82),
        harmony_mix_range=(0.32, 0.48), drum_mix_range=(0.18, 0.30),
        lead_range=(52, 82), bass_range=(28, 48), harmony_range=(46, 72),
        drum_pitches=[36, 38, 42, 46],
        lead_step_max=5, lead_rest_prob=0.02, lead_style="boss_epic",
        bass_style="boss_heavy", harmony_style="boss_dramatic", drum_style="boss_pound",
        swing=0.0,
        extra_tracks=[
            ExtraTrackConfig(role="Strings Pad", gen_type="pad",
                             instrument=("SNES Strings", "sawtooth"),
                             vel=(45, 62), pitch_range=(42, 66),
                             mix_range=(0.22, 0.36), rest_prob=0.08, pan=0.20),
        ],
    )


def _genre_mix_boss() -> GenreConfig:
    vibe = random.choice(["final", "final", "mid"])
    pf = random.choice(["nes", "gba", "snes"])
    return GenreConfig(
        name="Mix - Boss Battle", bpm_range=(130, 162), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["harmonic_minor", "phrygian", "natural_minor", "diminished"],
        vibe=vibe, progression_family=pf,
        progression_builder=_boss_prog, section_builder=_boss_sections,
        lead_rhythm=_BSS_LEAD, bass_rhythm=_BSS_BASS, drum_rhythm=_BSS_DRUM,
        harmony_rhythm=_BSS_HARM,
        lead_instrument=random.choice(_ALL_BRIGHT_LEADS),
        bass_instrument=random.choice(_ALL_BASS),
        harmony_instrument=random.choice(_ALL_HARMONY),
        drum_instrument=random.choice(_ALL_DRUMS),
        lead_vel=(96, 122), bass_vel=(92, 110), harmony_vel=(60, 80), drum_vel=(32, 45),
        lead_mix_range=(0.84, 0.98), bass_mix_range=(0.64, 0.82),
        harmony_mix_range=(0.34, 0.50), drum_mix_range=(0.18, 0.32),
        lead_range=(52, 82), bass_range=(28, 48), harmony_range=(46, 72),
        drum_pitches=[36, 38, 42, 46],
        lead_step_max=5, lead_rest_prob=0.03, lead_style="boss_epic",
        bass_style="boss_heavy", harmony_style="boss_dramatic", drum_style="boss_pound",
        swing=0.0,
    )


# ────────────────────  OVERWORLD  ──────────────────────────────────

def _genre_generic_overworld() -> GenreConfig:
    vibe = random.choice(["heroic", "heroic", "journey"])
    return GenreConfig(
        name="Generic - Overworld", bpm_range=(112, 132), bars=32,
        root_choices=[0, 2, 4, 5, 7, 9],
        scale_choices=["major", "mixolydian", "lydian"],
        vibe=vibe, progression_family="nes",
        progression_builder=_overworld_prog, section_builder=_overworld_sections,
        lead_rhythm=_OVW_LEAD, bass_rhythm=_OVW_BASS, drum_rhythm=_OVW_DRUM,
        harmony_rhythm=_OVW_HARM,
        lead_instrument=random.choice([("Generic Square", "square"), ("Generic Saw", "sawtooth")]),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Generic Pulse 25%", "pulse25"),
        drum_instrument=("Generic Noise Drum", "noise"),
        lead_vel=(86, 108), bass_vel=(82, 96), harmony_vel=(48, 66), drum_vel=(20, 32),
        lead_mix_range=(0.80, 0.94), bass_mix_range=(0.58, 0.74),
        harmony_mix_range=(0.25, 0.40), drum_mix_range=(0.10, 0.20),
        lead_range=(56, 80), bass_range=(34, 54), harmony_range=(50, 72),
        drum_pitches=[36, 38, 42],
        lead_step_max=3, lead_rest_prob=0.08, lead_style="overworld_heroic",
        bass_style="march_bass", harmony_style="heroic_hold", drum_style="march_steady",
        swing=0.0,
    )


def _genre_gba_overworld() -> GenreConfig:
    vibe = random.choice(["heroic", "heroic", "journey"])
    return GenreConfig(
        name="GBA - Overworld", bpm_range=(115, 136), bars=32,
        root_choices=[0, 2, 3, 4, 5, 7, 9, 10],
        scale_choices=["major", "mixolydian", "dorian"],
        vibe=vibe, progression_family="gba",
        progression_builder=_overworld_prog, section_builder=_overworld_sections,
        lead_rhythm=_OVW_LEAD, bass_rhythm=_OVW_BASS, drum_rhythm=_OVW_DRUM,
        harmony_rhythm=_OVW_HARM,
        lead_instrument=random.choice([("GBA Muted Trumpet", "sawtooth"), ("GBA Flute", "sine")]),
        bass_instrument=random.choice([("GBA Slap Bass", "square"), ("GBA Fretless Bass", "triangle")]),
        harmony_instrument=("GBA Piano", "pulse25"),
        drum_instrument=("GBA Light Kit", "noise"),
        lead_vel=(88, 110), bass_vel=(84, 98), harmony_vel=(50, 68), drum_vel=(22, 34),
        lead_mix_range=(0.82, 0.96), bass_mix_range=(0.60, 0.76),
        harmony_mix_range=(0.28, 0.42), drum_mix_range=(0.12, 0.22),
        lead_range=(56, 80), bass_range=(34, 54), harmony_range=(50, 72),
        drum_pitches=[36, 38, 42],
        lead_step_max=4, lead_rest_prob=0.06, lead_style="overworld_heroic",
        bass_style="march_bass", harmony_style="heroic_hold", drum_style="march_steady",
        swing=0.08,
        extra_tracks=[
            ExtraTrackConfig(role="Strings Pad", gen_type="pad",
                             instrument=("GBA Strings", "sawtooth"),
                             vel=(38, 55), pitch_range=(48, 72),
                             mix_range=(0.20, 0.34), rest_prob=0.15, pan=0.20),
        ],
    )


def _genre_snes_overworld() -> GenreConfig:
    vibe = random.choice(["heroic", "heroic", "journey"])
    return GenreConfig(
        name="SNES - Overworld", bpm_range=(110, 130), bars=32,
        root_choices=[0, 2, 3, 5, 7, 9, 10],
        scale_choices=["major", "lydian", "mixolydian"],
        vibe=vibe, progression_family="snes",
        progression_builder=_overworld_prog, section_builder=_overworld_sections,
        lead_rhythm=_OVW_LEAD, bass_rhythm=_OVW_BASS, drum_rhythm=_OVW_DRUM,
        harmony_rhythm=[],
        lead_instrument=random.choice([("SNES Trumpet", "sawtooth"), ("SNES Flute", "sine")]),
        bass_instrument=("SNES Slap Bass", "square"),
        harmony_instrument=("SNES Piano", "pulse25"),
        drum_instrument=("SNES Kit", "noise"),
        lead_vel=(86, 110), bass_vel=(82, 96), harmony_vel=(45, 65), drum_vel=(18, 30),
        lead_mix_range=(0.80, 0.94), bass_mix_range=(0.56, 0.72),
        harmony_mix_range=(0.24, 0.38), drum_mix_range=(0.08, 0.18),
        lead_range=(56, 80), bass_range=(34, 54), harmony_range=(48, 72),
        drum_pitches=[36, 38, 42],
        lead_step_max=5, lead_rest_prob=0.06, lead_style="overworld_heroic",
        bass_style="march_bass", harmony_style="heroic_hold", drum_style="march_steady",
        swing=0.05, lead_doubling=("SNES Acoustic", "triangle", 0.08),
        extra_tracks=[
            ExtraTrackConfig(role="Strings Pad", gen_type="pad",
                             instrument=("SNES Strings", "sawtooth"),
                             vel=(35, 52), pitch_range=(48, 72),
                             mix_range=(0.22, 0.36), rest_prob=0.12, pan=0.25),
            ExtraTrackConfig(role="Harp Arpeggio", gen_type="arpeggio",
                             instrument=("SNES Harp", "sine"),
                             vel=(30, 48), pitch_range=(48, 72),
                             mix_range=(0.15, 0.28), rest_prob=0.20, pan=-0.20),
        ],
    )


def _genre_mix_overworld() -> GenreConfig:
    vibe = random.choice(["heroic", "heroic", "journey"])
    pf = random.choice(["nes", "gba", "snes"])
    return GenreConfig(
        name="Mix - Overworld", bpm_range=(110, 136), bars=32,
        root_choices=list(range(12)),
        scale_choices=["major", "mixolydian", "lydian", "dorian"],
        vibe=vibe, progression_family=pf,
        progression_builder=_overworld_prog, section_builder=_overworld_sections,
        lead_rhythm=_OVW_LEAD, bass_rhythm=_OVW_BASS, drum_rhythm=_OVW_DRUM,
        harmony_rhythm=_OVW_HARM,
        lead_instrument=random.choice(_ALL_BRIGHT_LEADS),
        bass_instrument=random.choice(_ALL_BASS),
        harmony_instrument=random.choice(_ALL_HARMONY),
        drum_instrument=random.choice(_ALL_DRUMS),
        lead_vel=(86, 110), bass_vel=(82, 96), harmony_vel=(48, 66), drum_vel=(20, 32),
        lead_mix_range=(0.80, 0.94), bass_mix_range=(0.58, 0.74),
        harmony_mix_range=(0.25, 0.40), drum_mix_range=(0.10, 0.20),
        lead_range=(56, 80), bass_range=(34, 54), harmony_range=(50, 72),
        drum_pitches=[36, 38, 42],
        lead_step_max=4, lead_rest_prob=0.07, lead_style="overworld_heroic",
        bass_style="march_bass", harmony_style="heroic_hold", drum_style="march_steady",
        swing=random.choice([0.0, 0.05, 0.08]),
        extra_tracks=[ExtraTrackConfig(role="Pad", gen_type="pad",
                                       instrument=random.choice(_ALL_PADS),
                                       vel=(35, 52), pitch_range=(48, 72),
                                       mix_range=(0.18, 0.32), rest_prob=0.15)],
    )


# ────────────────────  VICTORY  ────────────────────────────────────

def _genre_generic_victory() -> GenreConfig:
    return GenreConfig(
        name="Generic - Victory", bpm_range=(122, 140), bars=8,
        root_choices=[0, 2, 4, 5, 7, 9],
        scale_choices=["major", "lydian"],
        vibe="triumph", progression_family="nes",
        progression_builder=_victory_prog, section_builder=_victory_sections,
        lead_rhythm=_VIC_LEAD, bass_rhythm=_VIC_BASS, drum_rhythm=_VIC_DRUM,
        harmony_rhythm=_VIC_HARM,
        lead_instrument=random.choice([("Generic Square", "square"), ("Generic Saw", "sawtooth")]),
        bass_instrument=("Generic Triangle", "triangle"),
        harmony_instrument=("Generic Pulse 25%", "pulse25"),
        drum_instrument=("Generic Noise Drum", "noise"),
        lead_vel=(92, 115), bass_vel=(86, 100), harmony_vel=(55, 72), drum_vel=(24, 36),
        lead_mix_range=(0.82, 0.96), bass_mix_range=(0.58, 0.74),
        harmony_mix_range=(0.30, 0.45), drum_mix_range=(0.12, 0.24),
        lead_range=(58, 84), bass_range=(36, 56), harmony_range=(52, 74),
        drum_pitches=[36, 38, 42],
        lead_step_max=4, lead_rest_prob=0.04, lead_style="victory_fanfare",
        bass_style="fanfare_bass", harmony_style="fanfare_chord", drum_style="fanfare_roll",
        swing=0.0,
    )


def _genre_gba_victory() -> GenreConfig:
    return GenreConfig(
        name="GBA - Victory", bpm_range=(124, 142), bars=8,
        root_choices=[0, 2, 4, 5, 7, 9],
        scale_choices=["major", "lydian", "mixolydian"],
        vibe="triumph", progression_family="gba",
        progression_builder=_victory_prog, section_builder=_victory_sections,
        lead_rhythm=_VIC_LEAD, bass_rhythm=_VIC_BASS, drum_rhythm=_VIC_DRUM,
        harmony_rhythm=_VIC_HARM,
        lead_instrument=random.choice([("GBA Muted Trumpet", "sawtooth"), ("GBA Flute", "sine")]),
        bass_instrument=("GBA Slap Bass", "square"),
        harmony_instrument=("GBA Piano", "pulse25"),
        drum_instrument=("GBA Light Kit", "noise"),
        lead_vel=(94, 118), bass_vel=(88, 102), harmony_vel=(58, 75), drum_vel=(26, 38),
        lead_mix_range=(0.84, 0.96), bass_mix_range=(0.60, 0.76),
        harmony_mix_range=(0.32, 0.48), drum_mix_range=(0.14, 0.26),
        lead_range=(58, 84), bass_range=(36, 56), harmony_range=(52, 74),
        drum_pitches=[36, 38, 42],
        lead_step_max=4, lead_rest_prob=0.03, lead_style="victory_fanfare",
        bass_style="fanfare_bass", harmony_style="fanfare_chord", drum_style="fanfare_roll",
        swing=0.0,
    )


def _genre_snes_victory() -> GenreConfig:
    return GenreConfig(
        name="SNES - Victory", bpm_range=(120, 138), bars=8,
        root_choices=[0, 2, 4, 5, 7, 9],
        scale_choices=["major", "lydian"],
        vibe="triumph", progression_family="snes",
        progression_builder=_victory_prog, section_builder=_victory_sections,
        lead_rhythm=_VIC_LEAD, bass_rhythm=_VIC_BASS, drum_rhythm=_VIC_DRUM,
        harmony_rhythm=_VIC_HARM,
        lead_instrument=random.choice([("SNES Trumpet", "sawtooth"), ("SNES Flute", "sine")]),
        bass_instrument=("SNES Slap Bass", "square"),
        harmony_instrument=("SNES Piano", "pulse25"),
        drum_instrument=("SNES Kit", "noise"),
        lead_vel=(90, 114), bass_vel=(85, 100), harmony_vel=(52, 70), drum_vel=(22, 34),
        lead_mix_range=(0.82, 0.96), bass_mix_range=(0.56, 0.72),
        harmony_mix_range=(0.28, 0.42), drum_mix_range=(0.10, 0.22),
        lead_range=(58, 84), bass_range=(36, 56), harmony_range=(50, 74),
        drum_pitches=[36, 38, 42],
        lead_step_max=5, lead_rest_prob=0.03, lead_style="victory_fanfare",
        bass_style="fanfare_bass", harmony_style="fanfare_chord", drum_style="fanfare_roll",
        swing=0.0, lead_doubling=("SNES Acoustic", "triangle", 0.08),
    )


def _genre_mix_victory() -> GenreConfig:
    pf = random.choice(["nes", "gba", "snes"])
    return GenreConfig(
        name="Mix - Victory", bpm_range=(120, 142), bars=8,
        root_choices=list(range(12)),
        scale_choices=["major", "lydian", "mixolydian"],
        vibe="triumph", progression_family=pf,
        progression_builder=_victory_prog, section_builder=_victory_sections,
        lead_rhythm=_VIC_LEAD, bass_rhythm=_VIC_BASS, drum_rhythm=_VIC_DRUM,
        harmony_rhythm=_VIC_HARM,
        lead_instrument=random.choice(_ALL_BRIGHT_LEADS),
        bass_instrument=random.choice(_ALL_BASS),
        harmony_instrument=random.choice(_ALL_HARMONY),
        drum_instrument=random.choice(_ALL_DRUMS),
        lead_vel=(92, 116), bass_vel=(86, 100), harmony_vel=(55, 72), drum_vel=(24, 36),
        lead_mix_range=(0.82, 0.96), bass_mix_range=(0.58, 0.74),
        harmony_mix_range=(0.30, 0.45), drum_mix_range=(0.12, 0.24),
        lead_range=(58, 84), bass_range=(36, 56), harmony_range=(52, 74),
        drum_pitches=[36, 38, 42],
        lead_step_max=4, lead_rest_prob=0.04, lead_style="victory_fanfare",
        bass_style="fanfare_bass", harmony_style="fanfare_chord", drum_style="fanfare_roll",
        swing=0.0,
    )


# ════════════════════════════════════════════════════════════════════
# NES + GAMEBOY VARIANTS
# ════════════════════════════════════════════════════════════════════

# ── NES HM Town ────────────────────────────────────────────────────
def _genre_nes_hm_town() -> GenreConfig:
    vibe = random.choice(["spring"] * 3 + ["autumn"])
    return GenreConfig(
        name="NES - HM Town", bpm_range=(92, 112), bars=32,
        root_choices=[0, 2, 5, 7, 9],
        scale_choices=["major", "pentatonic_major", "lydian"],
        vibe=vibe, progression_family="nes",
        progression_builder=_hm_town_prog, section_builder=_pastoral_sections,
        lead_rhythm=_HM_LEAD, bass_rhythm=_HM_BASS, drum_rhythm=_HM_DRUM,
        harmony_rhythm=_HM_HARM,
        lead_instrument=random.choice([("NES Square", "square"), ("NES Pulse 25%", "pulse25")]),
        bass_instrument=("NES Triangle", "triangle"),
        harmony_instrument=("NES Pulse 25%", "pulse25"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(80, 100), bass_vel=(74, 90), harmony_vel=(36, 54), drum_vel=(12, 22),
        lead_mix_range=(0.78, 0.92), bass_mix_range=(0.52, 0.66),
        harmony_mix_range=(0.16, 0.30), drum_mix_range=(0.05, 0.12),
        lead_range=(58, 76), bass_range=(36, 55), harmony_range=(50, 70),
        drum_pitches=[42],
        lead_step_max=2, lead_rest_prob=0.14,
        lead_style="hm_pastoral", bass_style="gentle_arp",
        harmony_style="sustained_pad", drum_style="gentle_brush",
        swing=0.0, lead_doubling=("NES Pulse 25%", "pulse25", 0.08),
    )

# ── Gameboy HM Town ───────────────────────────────────────────────
def _genre_gameboy_hm_town() -> GenreConfig:
    vibe = random.choice(["spring"] * 3 + ["autumn"])
    return GenreConfig(
        name="Gameboy - HM Town", bpm_range=(88, 106), bars=32,
        root_choices=[0, 2, 5, 7, 9],
        scale_choices=["major", "pentatonic_major", "lydian"],
        vibe=vibe, progression_family="nes",
        progression_builder=_hm_town_prog, section_builder=_pastoral_sections,
        lead_rhythm=_HM_LEAD, bass_rhythm=_HM_BASS, drum_rhythm=_HM_DRUM,
        harmony_rhythm=_HM_HARM,
        lead_instrument=("Gameboy Square", "square"),
        bass_instrument=("Gameboy Wave", "triangle"),
        harmony_instrument=("Gameboy Pulse 12.5%", "pulse12"),
        drum_instrument=("Gameboy Noise", "noise"),
        lead_vel=(76, 96), bass_vel=(70, 86), harmony_vel=(34, 50), drum_vel=(10, 18),
        lead_mix_range=(0.74, 0.88), bass_mix_range=(0.50, 0.64),
        harmony_mix_range=(0.14, 0.26), drum_mix_range=(0.04, 0.10),
        lead_range=(58, 76), bass_range=(36, 55), harmony_range=(50, 70),
        drum_pitches=[42],
        lead_step_max=2, lead_rest_prob=0.16,
        lead_style="hm_pastoral", bass_style="gentle_arp",
        harmony_style="sustained_pad", drum_style="gentle_brush",
        swing=0.0, lead_doubling=("Gameboy Pulse 12.5%", "pulse12", 0.06),
    )

# ── NES Zelda Town ────────────────────────────────────────────────
def _genre_nes_zelda_town() -> GenreConfig:
    vibe = random.choice(["village"] * 3 + ["mystery"])
    sc = ["mixolydian", "major", "dorian"] if vibe == "village" else ["natural_minor", "dorian", "phrygian"]
    return GenreConfig(
        name="NES - Zelda Town", bpm_range=(104, 130), bars=32,
        root_choices=[0, 2, 5, 7, 9, 10],
        scale_choices=sc, vibe=vibe, progression_family="nes",
        progression_builder=_zelda_town_prog, section_builder=_zelda_sections,
        lead_rhythm=_ZEL_LEAD, bass_rhythm=_ZEL_BASS, drum_rhythm=_ZEL_DRUM,
        harmony_rhythm=_ZEL_HARM,
        lead_instrument=random.choice([("NES Square", "square"), ("NES Pulse 25%", "pulse25")]),
        bass_instrument=("NES Triangle", "triangle"),
        harmony_instrument=("NES Pulse 25%", "pulse25"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(86, 106), bass_vel=(80, 94), harmony_vel=(44, 62), drum_vel=(16, 26),
        lead_mix_range=(0.80, 0.94), bass_mix_range=(0.56, 0.72),
        harmony_mix_range=(0.22, 0.36), drum_mix_range=(0.07, 0.15),
        lead_range=(56, 78), bass_range=(36, 55), harmony_range=(50, 72),
        drum_pitches=[42],
        lead_step_max=3, lead_rest_prob=0.10,
        lead_style="zelda_folk", bass_style="folk_bounce",
        harmony_style="staccato", drum_style="folk_beat",
        swing=0.0, lead_doubling=("NES Square", "square", 0.10),
    )

# ── Gameboy Zelda Town ────────────────────────────────────────────
def _genre_gameboy_zelda_town() -> GenreConfig:
    vibe = random.choice(["village"] * 3 + ["mystery"])
    sc = ["mixolydian", "major", "dorian"] if vibe == "village" else ["natural_minor", "dorian", "phrygian"]
    return GenreConfig(
        name="Gameboy - Zelda Town", bpm_range=(100, 126), bars=32,
        root_choices=[0, 2, 5, 7, 9, 10],
        scale_choices=sc, vibe=vibe, progression_family="nes",
        progression_builder=_zelda_town_prog, section_builder=_zelda_sections,
        lead_rhythm=_ZEL_LEAD, bass_rhythm=_ZEL_BASS, drum_rhythm=_ZEL_DRUM,
        harmony_rhythm=_ZEL_HARM,
        lead_instrument=("Gameboy Square", "square"),
        bass_instrument=("Gameboy Wave", "triangle"),
        harmony_instrument=("Gameboy Pulse 12.5%", "pulse12"),
        drum_instrument=("Gameboy Noise", "noise"),
        lead_vel=(82, 102), bass_vel=(76, 90), harmony_vel=(40, 58), drum_vel=(12, 22),
        lead_mix_range=(0.76, 0.90), bass_mix_range=(0.54, 0.68),
        harmony_mix_range=(0.18, 0.32), drum_mix_range=(0.05, 0.12),
        lead_range=(56, 78), bass_range=(36, 55), harmony_range=(50, 72),
        drum_pitches=[42],
        lead_step_max=3, lead_rest_prob=0.12,
        lead_style="zelda_folk", bass_style="folk_bounce",
        harmony_style="staccato", drum_style="folk_beat",
        swing=0.0, lead_doubling=("Gameboy Pulse 12.5%", "pulse12", 0.08),
    )

# ── NES Dungeon ───────────────────────────────────────────────────
def _genre_nes_dungeon() -> GenreConfig:
    vibe = random.choice(["deep", "deep", "labyrinth"])
    return GenreConfig(
        name="NES - Dungeon", bpm_range=(70, 98), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["natural_minor", "phrygian", "harmonic_minor"] if vibe == "deep"
                 else ["natural_minor", "dorian", "harmonic_minor"],
        vibe=vibe, progression_family="nes",
        progression_builder=_dungeon_prog, section_builder=_dungeon_sections,
        lead_rhythm=_DNG_LEAD, bass_rhythm=_DNG_BASS, drum_rhythm=_DNG_DRUM,
        harmony_rhythm=_DNG_HARM,
        lead_instrument=random.choice([("NES Pulse 25%", "pulse25"), ("NES Square", "square")]),
        bass_instrument=("NES Triangle", "triangle"),
        harmony_instrument=("NES Square", "square"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(60, 82), bass_vel=(64, 80), harmony_vel=(30, 46), drum_vel=(12, 22),
        lead_mix_range=(0.64, 0.82), bass_mix_range=(0.50, 0.64),
        harmony_mix_range=(0.15, 0.28), drum_mix_range=(0.05, 0.12),
        lead_range=(48, 68), bass_range=(32, 50), harmony_range=(44, 64),
        drum_pitches=[42],
        lead_step_max=3, lead_rest_prob=0.28,
        lead_style="dungeon_dark", bass_style="drone_pedal",
        harmony_style="dark_drone", drum_style="dungeon_march",
        swing=0.0,
    )

# ── Gameboy Dungeon ───────────────────────────────────────────────
def _genre_gameboy_dungeon() -> GenreConfig:
    vibe = random.choice(["deep", "deep", "labyrinth"])
    return GenreConfig(
        name="Gameboy - Dungeon", bpm_range=(66, 94), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["natural_minor", "phrygian", "harmonic_minor"] if vibe == "deep"
                 else ["natural_minor", "dorian", "harmonic_minor"],
        vibe=vibe, progression_family="nes",
        progression_builder=_dungeon_prog, section_builder=_dungeon_sections,
        lead_rhythm=_DNG_LEAD, bass_rhythm=_DNG_BASS, drum_rhythm=_DNG_DRUM,
        harmony_rhythm=_DNG_HARM,
        lead_instrument=("Gameboy Square", "square"),
        bass_instrument=("Gameboy Wave", "triangle"),
        harmony_instrument=("Gameboy Pulse 12.5%", "pulse12"),
        drum_instrument=("Gameboy Noise", "noise"),
        lead_vel=(56, 78), bass_vel=(60, 76), harmony_vel=(26, 42), drum_vel=(8, 18),
        lead_mix_range=(0.60, 0.78), bass_mix_range=(0.48, 0.62),
        harmony_mix_range=(0.12, 0.24), drum_mix_range=(0.04, 0.10),
        lead_range=(48, 68), bass_range=(32, 50), harmony_range=(44, 64),
        drum_pitches=[42],
        lead_step_max=3, lead_rest_prob=0.30,
        lead_style="dungeon_dark", bass_style="drone_pedal",
        harmony_style="dark_drone", drum_style="dungeon_march",
        swing=0.0,
    )

# ── NES Combat ────────────────────────────────────────────────────
def _genre_nes_combat() -> GenreConfig:
    vibe = random.choice(["fierce", "fierce", "skirmish"])
    return GenreConfig(
        name="NES - Combat", bpm_range=(146, 172), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["natural_minor", "harmonic_minor", "dorian"],
        vibe=vibe, progression_family="nes",
        progression_builder=_combat_prog, section_builder=_combat_sections,
        lead_rhythm=_CMB_LEAD, bass_rhythm=_CMB_BASS, drum_rhythm=_CMB_DRUM,
        harmony_rhythm=_CMB_HARM,
        lead_instrument=random.choice([("NES Square", "square"), ("NES Pulse 25%", "pulse25")]),
        bass_instrument=("NES Triangle", "triangle"),
        harmony_instrument=("NES Square", "square"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(94, 120), bass_vel=(90, 106), harmony_vel=(56, 76), drum_vel=(30, 42),
        lead_mix_range=(0.84, 0.98), bass_mix_range=(0.62, 0.80),
        harmony_mix_range=(0.32, 0.46), drum_mix_range=(0.16, 0.28),
        lead_range=(54, 80), bass_range=(30, 50), harmony_range=(48, 72),
        drum_pitches=[36, 42],
        lead_step_max=4, lead_rest_prob=0.04,
        lead_style="combat_intense", bass_style="driving_eighth",
        harmony_style="power_stab", drum_style="combat_drive",
        swing=0.0,
    )

# ── Gameboy Combat ────────────────────────────────────────────────
def _genre_gameboy_combat() -> GenreConfig:
    vibe = random.choice(["fierce", "fierce", "skirmish"])
    return GenreConfig(
        name="Gameboy - Combat", bpm_range=(140, 166), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["natural_minor", "harmonic_minor", "dorian"],
        vibe=vibe, progression_family="nes",
        progression_builder=_combat_prog, section_builder=_combat_sections,
        lead_rhythm=_CMB_LEAD, bass_rhythm=_CMB_BASS, drum_rhythm=_CMB_DRUM,
        harmony_rhythm=_CMB_HARM,
        lead_instrument=("Gameboy Square", "square"),
        bass_instrument=("Gameboy Wave", "triangle"),
        harmony_instrument=("Gameboy Pulse 12.5%", "pulse12"),
        drum_instrument=("Gameboy Noise", "noise"),
        lead_vel=(90, 116), bass_vel=(86, 102), harmony_vel=(52, 72), drum_vel=(26, 38),
        lead_mix_range=(0.82, 0.96), bass_mix_range=(0.58, 0.76),
        harmony_mix_range=(0.28, 0.42), drum_mix_range=(0.14, 0.26),
        lead_range=(54, 80), bass_range=(30, 50), harmony_range=(48, 72),
        drum_pitches=[36, 42],
        lead_step_max=4, lead_rest_prob=0.05,
        lead_style="combat_intense", bass_style="driving_eighth",
        harmony_style="power_stab", drum_style="combat_drive",
        swing=0.0,
    )

# ── NES Encounter ─────────────────────────────────────────────────
def _genre_nes_encounter() -> GenreConfig:
    return GenreConfig(
        name="NES - Encounter", bpm_range=(158, 184), bars=16,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["harmonic_minor", "natural_minor"],
        vibe="alarm", progression_family="nes",
        progression_builder=_encounter_prog, section_builder=_combat_sections,
        lead_rhythm=_ENC_LEAD, bass_rhythm=_CMB_BASS, drum_rhythm=_CMB_DRUM,
        harmony_rhythm=_CMB_HARM,
        lead_instrument=random.choice([("NES Square", "square"), ("NES Pulse 25%", "pulse25")]),
        bass_instrument=("NES Triangle", "triangle"),
        harmony_instrument=("NES Square", "square"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(98, 122), bass_vel=(92, 110), harmony_vel=(60, 80), drum_vel=(32, 44),
        lead_mix_range=(0.86, 0.98), bass_mix_range=(0.64, 0.82),
        harmony_mix_range=(0.34, 0.50), drum_mix_range=(0.18, 0.30),
        lead_range=(56, 84), bass_range=(30, 50), harmony_range=(48, 72),
        drum_pitches=[36, 42],
        lead_step_max=5, lead_rest_prob=0.02,
        lead_style="encounter_alarm", bass_style="driving_eighth",
        harmony_style="power_stab", drum_style="combat_drive",
        swing=0.0,
    )

# ── Gameboy Encounter ─────────────────────────────────────────────
def _genre_gameboy_encounter() -> GenreConfig:
    return GenreConfig(
        name="Gameboy - Encounter", bpm_range=(152, 178), bars=16,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["harmonic_minor", "natural_minor"],
        vibe="alarm", progression_family="nes",
        progression_builder=_encounter_prog, section_builder=_combat_sections,
        lead_rhythm=_ENC_LEAD, bass_rhythm=_CMB_BASS, drum_rhythm=_CMB_DRUM,
        harmony_rhythm=_CMB_HARM,
        lead_instrument=("Gameboy Square", "square"),
        bass_instrument=("Gameboy Wave", "triangle"),
        harmony_instrument=("Gameboy Pulse 12.5%", "pulse12"),
        drum_instrument=("Gameboy Noise", "noise"),
        lead_vel=(94, 118), bass_vel=(88, 106), harmony_vel=(56, 76), drum_vel=(28, 40),
        lead_mix_range=(0.84, 0.96), bass_mix_range=(0.60, 0.78),
        harmony_mix_range=(0.30, 0.46), drum_mix_range=(0.16, 0.28),
        lead_range=(56, 84), bass_range=(30, 50), harmony_range=(48, 72),
        drum_pitches=[36, 42],
        lead_step_max=5, lead_rest_prob=0.02,
        lead_style="encounter_alarm", bass_style="driving_eighth",
        harmony_style="power_stab", drum_style="combat_drive",
        swing=0.0,
    )

# ── NES Boss Battle ───────────────────────────────────────────────
def _genre_nes_boss() -> GenreConfig:
    vibe = random.choice(["final", "final", "mid"])
    return GenreConfig(
        name="NES - Boss Battle", bpm_range=(136, 162), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["harmonic_minor", "phrygian", "diminished"] if vibe == "final"
                 else ["natural_minor", "harmonic_minor", "dorian"],
        vibe=vibe, progression_family="nes",
        progression_builder=_boss_prog, section_builder=_boss_sections,
        lead_rhythm=_BSS_LEAD, bass_rhythm=_BSS_BASS, drum_rhythm=_BSS_DRUM,
        harmony_rhythm=_BSS_HARM,
        lead_instrument=random.choice([("NES Square", "square"), ("NES Pulse 25%", "pulse25")]),
        bass_instrument=("NES Triangle", "triangle"),
        harmony_instrument=("NES Pulse 25%", "pulse25"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(98, 124), bass_vel=(94, 112), harmony_vel=(62, 82), drum_vel=(34, 46),
        lead_mix_range=(0.86, 0.98), bass_mix_range=(0.66, 0.84),
        harmony_mix_range=(0.36, 0.52), drum_mix_range=(0.20, 0.34),
        lead_range=(52, 82), bass_range=(28, 48), harmony_range=(46, 72),
        drum_pitches=[36, 42],
        lead_step_max=5, lead_rest_prob=0.03,
        lead_style="boss_epic", bass_style="boss_heavy",
        harmony_style="boss_dramatic", drum_style="boss_pound",
        swing=0.0,
    )

# ── Gameboy Boss Battle ──────────────────────────────────────────
def _genre_gameboy_boss() -> GenreConfig:
    vibe = random.choice(["final", "final", "mid"])
    return GenreConfig(
        name="Gameboy - Boss Battle", bpm_range=(130, 156), bars=32,
        root_choices=[0, 1, 3, 5, 7, 8, 10],
        scale_choices=["harmonic_minor", "phrygian", "diminished"] if vibe == "final"
                 else ["natural_minor", "harmonic_minor", "dorian"],
        vibe=vibe, progression_family="nes",
        progression_builder=_boss_prog, section_builder=_boss_sections,
        lead_rhythm=_BSS_LEAD, bass_rhythm=_BSS_BASS, drum_rhythm=_BSS_DRUM,
        harmony_rhythm=_BSS_HARM,
        lead_instrument=("Gameboy Square", "square"),
        bass_instrument=("Gameboy Wave", "triangle"),
        harmony_instrument=("Gameboy Pulse 12.5%", "pulse12"),
        drum_instrument=("Gameboy Noise", "noise"),
        lead_vel=(94, 120), bass_vel=(90, 108), harmony_vel=(58, 78), drum_vel=(30, 42),
        lead_mix_range=(0.84, 0.96), bass_mix_range=(0.62, 0.80),
        harmony_mix_range=(0.32, 0.48), drum_mix_range=(0.18, 0.30),
        lead_range=(52, 82), bass_range=(28, 48), harmony_range=(46, 72),
        drum_pitches=[36, 42],
        lead_step_max=5, lead_rest_prob=0.03,
        lead_style="boss_epic", bass_style="boss_heavy",
        harmony_style="boss_dramatic", drum_style="boss_pound",
        swing=0.0,
    )

# ── NES Overworld ─────────────────────────────────────────────────
def _genre_nes_overworld() -> GenreConfig:
    vibe = random.choice(["heroic", "heroic", "journey"])
    return GenreConfig(
        name="NES - Overworld", bpm_range=(116, 136), bars=32,
        root_choices=[0, 2, 4, 5, 7, 9],
        scale_choices=["major", "mixolydian", "lydian"],
        vibe=vibe, progression_family="nes",
        progression_builder=_overworld_prog, section_builder=_overworld_sections,
        lead_rhythm=_OVW_LEAD, bass_rhythm=_OVW_BASS, drum_rhythm=_OVW_DRUM,
        harmony_rhythm=_OVW_HARM,
        lead_instrument=random.choice([("NES Square", "square"), ("NES Pulse 25%", "pulse25")]),
        bass_instrument=("NES Triangle", "triangle"),
        harmony_instrument=("NES Pulse 25%", "pulse25"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(88, 110), bass_vel=(84, 98), harmony_vel=(50, 68), drum_vel=(22, 34),
        lead_mix_range=(0.82, 0.96), bass_mix_range=(0.60, 0.76),
        harmony_mix_range=(0.26, 0.42), drum_mix_range=(0.12, 0.22),
        lead_range=(56, 80), bass_range=(34, 54), harmony_range=(50, 72),
        drum_pitches=[36, 42],
        lead_step_max=3, lead_rest_prob=0.08,
        lead_style="overworld_heroic", bass_style="march_bass",
        harmony_style="heroic_hold", drum_style="march_steady",
        swing=0.0,
    )

# ── Gameboy Overworld ─────────────────────────────────────────────
def _genre_gameboy_overworld() -> GenreConfig:
    vibe = random.choice(["heroic", "heroic", "journey"])
    return GenreConfig(
        name="Gameboy - Overworld", bpm_range=(110, 130), bars=32,
        root_choices=[0, 2, 4, 5, 7, 9],
        scale_choices=["major", "mixolydian", "lydian"],
        vibe=vibe, progression_family="nes",
        progression_builder=_overworld_prog, section_builder=_overworld_sections,
        lead_rhythm=_OVW_LEAD, bass_rhythm=_OVW_BASS, drum_rhythm=_OVW_DRUM,
        harmony_rhythm=_OVW_HARM,
        lead_instrument=("Gameboy Square", "square"),
        bass_instrument=("Gameboy Wave", "triangle"),
        harmony_instrument=("Gameboy Pulse 12.5%", "pulse12"),
        drum_instrument=("Gameboy Noise", "noise"),
        lead_vel=(84, 106), bass_vel=(80, 94), harmony_vel=(46, 64), drum_vel=(18, 30),
        lead_mix_range=(0.78, 0.92), bass_mix_range=(0.56, 0.72),
        harmony_mix_range=(0.24, 0.38), drum_mix_range=(0.10, 0.20),
        lead_range=(56, 80), bass_range=(34, 54), harmony_range=(50, 72),
        drum_pitches=[36, 42],
        lead_step_max=3, lead_rest_prob=0.09,
        lead_style="overworld_heroic", bass_style="march_bass",
        harmony_style="heroic_hold", drum_style="march_steady",
        swing=0.0,
    )

# ── NES Victory ───────────────────────────────────────────────────
def _genre_nes_victory() -> GenreConfig:
    return GenreConfig(
        name="NES - Victory", bpm_range=(126, 144), bars=8,
        root_choices=[0, 2, 4, 5, 7, 9],
        scale_choices=["major", "lydian"],
        vibe="triumph", progression_family="nes",
        progression_builder=_victory_prog, section_builder=_victory_sections,
        lead_rhythm=_VIC_LEAD, bass_rhythm=_VIC_BASS, drum_rhythm=_VIC_DRUM,
        harmony_rhythm=_VIC_HARM,
        lead_instrument=random.choice([("NES Square", "square"), ("NES Pulse 25%", "pulse25")]),
        bass_instrument=("NES Triangle", "triangle"),
        harmony_instrument=("NES Pulse 25%", "pulse25"),
        drum_instrument=("NES Noise", "noise"),
        lead_vel=(94, 118), bass_vel=(88, 102), harmony_vel=(58, 74), drum_vel=(26, 38),
        lead_mix_range=(0.84, 0.98), bass_mix_range=(0.60, 0.76),
        harmony_mix_range=(0.32, 0.46), drum_mix_range=(0.14, 0.26),
        lead_range=(58, 84), bass_range=(36, 56), harmony_range=(52, 74),
        drum_pitches=[36, 42],
        lead_step_max=4, lead_rest_prob=0.04,
        lead_style="victory_fanfare", bass_style="fanfare_bass",
        harmony_style="fanfare_chord", drum_style="fanfare_roll",
        swing=0.0,
    )

# ── Gameboy Victory ───────────────────────────────────────────────
def _genre_gameboy_victory() -> GenreConfig:
    return GenreConfig(
        name="Gameboy - Victory", bpm_range=(120, 138), bars=8,
        root_choices=[0, 2, 4, 5, 7, 9],
        scale_choices=["major", "lydian"],
        vibe="triumph", progression_family="nes",
        progression_builder=_victory_prog, section_builder=_victory_sections,
        lead_rhythm=_VIC_LEAD, bass_rhythm=_VIC_BASS, drum_rhythm=_VIC_DRUM,
        harmony_rhythm=_VIC_HARM,
        lead_instrument=("Gameboy Square", "square"),
        bass_instrument=("Gameboy Wave", "triangle"),
        harmony_instrument=("Gameboy Pulse 12.5%", "pulse12"),
        drum_instrument=("Gameboy Noise", "noise"),
        lead_vel=(90, 114), bass_vel=(84, 98), harmony_vel=(54, 70), drum_vel=(22, 34),
        lead_mix_range=(0.80, 0.94), bass_mix_range=(0.56, 0.72),
        harmony_mix_range=(0.28, 0.42), drum_mix_range=(0.12, 0.22),
        lead_range=(58, 84), bass_range=(36, 56), harmony_range=(52, 74),
        drum_pitches=[36, 42],
        lead_step_max=4, lead_rest_prob=0.05,
        lead_style="victory_fanfare", bass_style="fanfare_bass",
        harmony_style="fanfare_chord", drum_style="fanfare_roll",
        swing=0.0,
    )


# ════════════════════════════════════════════════════════════════════
# EXPORT: dict of all new genre builders
# ════════════════════════════════════════════════════════════════════

NEW_GENRE_BUILDERS: dict[str, object] = {
    # HM Town
    "Generic - HM Town":     _genre_generic_hm_town,
    "NES - HM Town":         _genre_nes_hm_town,
    "Gameboy - HM Town":     _genre_gameboy_hm_town,
    "GBA - HM Town":         _genre_gba_hm_town,
    "SNES - HM Town":        _genre_snes_hm_town,
    "Mix - HM Town":         _genre_mix_hm_town,
    # Zelda Town
    "Generic - Zelda Town":  _genre_generic_zelda_town,
    "NES - Zelda Town":      _genre_nes_zelda_town,
    "Gameboy - Zelda Town":  _genre_gameboy_zelda_town,
    "GBA - Zelda Town":      _genre_gba_zelda_town,
    "SNES - Zelda Town":     _genre_snes_zelda_town,
    "Mix - Zelda Town":      _genre_mix_zelda_town,
    # Dungeon
    "Generic - Dungeon":     _genre_generic_dungeon,
    "NES - Dungeon":         _genre_nes_dungeon,
    "Gameboy - Dungeon":     _genre_gameboy_dungeon,
    "GBA - Dungeon":         _genre_gba_dungeon,
    "SNES - Dungeon":        _genre_snes_dungeon,
    "Mix - Dungeon":         _genre_mix_dungeon,
    # Combat
    "Generic - Combat":      _genre_generic_combat,
    "NES - Combat":          _genre_nes_combat,
    "Gameboy - Combat":      _genre_gameboy_combat,
    "GBA - Combat":          _genre_gba_combat,
    "SNES - Combat":         _genre_snes_combat,
    "Mix - Combat":          _genre_mix_combat,
    # Encounter
    "Generic - Encounter":   _genre_generic_encounter,
    "NES - Encounter":       _genre_nes_encounter,
    "Gameboy - Encounter":   _genre_gameboy_encounter,
    "GBA - Encounter":       _genre_gba_encounter,
    "SNES - Encounter":      _genre_snes_encounter,
    "Mix - Encounter":       _genre_mix_encounter,
    # Boss Battle
    "Generic - Boss Battle": _genre_generic_boss,
    "NES - Boss Battle":     _genre_nes_boss,
    "Gameboy - Boss Battle": _genre_gameboy_boss,
    "GBA - Boss Battle":     _genre_gba_boss,
    "SNES - Boss Battle":    _genre_snes_boss,
    "Mix - Boss Battle":     _genre_mix_boss,
    # Overworld
    "Generic - Overworld":   _genre_generic_overworld,
    "NES - Overworld":       _genre_nes_overworld,
    "Gameboy - Overworld":   _genre_gameboy_overworld,
    "GBA - Overworld":       _genre_gba_overworld,
    "SNES - Overworld":      _genre_snes_overworld,
    "Mix - Overworld":       _genre_mix_overworld,
    # Victory
    "Generic - Victory":     _genre_generic_victory,
    "NES - Victory":         _genre_nes_victory,
    "Gameboy - Victory":     _genre_gameboy_victory,
    "GBA - Victory":         _genre_gba_victory,
    "SNES - Victory":        _genre_snes_victory,
    "Mix - Victory":         _genre_mix_victory,
}
