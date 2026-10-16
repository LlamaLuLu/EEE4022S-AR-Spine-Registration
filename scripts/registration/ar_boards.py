"""
Board registry and auto selection. Pure Python, no Qt, so the phone port can
mirror it line for line.

A Board is one marker map plus the things that depend on it (error anchor,
bracket or dome rules). The selector picks which board the tracker uses:
fixed, or auto from the marker IDs seen in the frame.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ar_board import load_corner_lut
from ar_error import ANCHORS

AUTO = "auto"
SWITCH_FRAMES = 5          # frames in a row that must favour the other board


@dataclass(frozen=True, eq=False)
class Board:
    name: str
    lut: object
    is_v3: bool
    anchor: object

    @property
    def title(self):
        return self.name.capitalize()

    @property
    def n_total(self):
        return len(self.lut)

    @property
    def gate(self):
        return self.lut.min_markers

    @property
    def ids(self):
        return frozenset(self.lut)


def make_board(lut):
    is_v3 = set(lut) <= set(range(8))
    name = "bracket" if is_v3 else "dome"
    return Board(name, lut, is_v3, ANCHORS[name])


def load_boards(paths):
    """paths: marker map JSON paths (None means the built in V3 bracket)."""
    boards = [make_board(load_corner_lut(p)) for p in paths]
    names = [b.name for b in boards]
    if len(set(names)) != len(names):
        raise ValueError(f"two maps for the same board: {names}")
    return boards


def default_map_paths(repo_root):
    """Dome and bracket maps from data/, whichever exist. [None] = built in bracket."""
    found = [p for p in (Path(repo_root) / "data" / "dome_markers" / "marker_map.json",
                         Path(repo_root) / "data" / "bracket_markers" / "marker_map.json")
             if p.exists()]
    return found or [None]


class BoardSelector:
    def __init__(self, boards, mode=AUTO, switch_frames=SWITCH_FRAMES):
        if not boards:
            raise ValueError("no boards")
        self.boards = {b.name: b for b in boards}
        self.switch_frames = switch_frames
        self.modes = [AUTO] + [n for n in ("dome", "bracket") if n in self.boards]
        if len(self.boards) == 1:
            self.modes = list(self.boards)               # nothing to choose between
            if mode == AUTO:
                mode = next(iter(self.boards))
        if mode not in self.modes:
            raise ValueError(f"board mode {mode!r} not available, have {self.modes}")
        self.mode = mode
        self.board = self.boards[mode] if mode != AUTO else (
            self.boards.get("dome") or boards[0])         # dome is the default
        self._streak = 0

    @property
    def label(self):
        if len(self.boards) == 1:
            return self.board.title
        return f"{self.board.title} ({'auto' if self.mode == AUTO else 'fixed'})"

    def update(self, ids):
        """Call once per frame with the detected IDs. True if the board changed."""
        if self.mode != AUTO or ids is None or len(ids) == 0:
            self._streak = 0
            return False
        seen = {int(i) for i in np.asarray(ids).flatten()}
        counts = {n: len(seen & b.ids) for n, b in self.boards.items()}
        best = max(counts, key=counts.get)
        if counts[best] > counts[self.board.name]:       # ties keep the current board
            self._streak += 1
            if self._streak >= self.switch_frames:
                self.board = self.boards[best]
                self._streak = 0
                return True
        else:
            self._streak = 0
        return False

    def set_mode(self, mode):
        """True if the active board changed."""
        if mode not in self.modes:
            raise ValueError(f"board mode {mode!r} not available, have {self.modes}")
        old = self.board
        self.mode, self._streak = mode, 0
        if mode != AUTO:
            self.board = self.boards[mode]
        return self.board is not old

    def cycle_mode(self):
        return self.set_mode(self.modes[(self.modes.index(self.mode) + 1) % len(self.modes)])