"""
Pedicle selection: the single source of truth for "which pedicle is the user
working on". Inputs (keys, rail buttons, clicks) all call into this class and
the renderer only reads it, so every front end behaves the same.

State is either overview (nothing selected) or one pedicle = (level, side).
Labels follow the annotation convention: isthmus 'L3_R', entry 'L3_E_R'.
"""
import re

SIDE_NAMES = {"L": "left", "R": "right"}
_LABEL = re.compile(r"L(\d)_(?:E_)?([LR])$")


def parse_label(label):
    """'L3_E_R' -> (3, 'R', True); 'L3_L' -> (3, 'L', False); None if it does not parse."""
    m = _LABEL.match(str(label))
    if not m:
        return None
    return int(m.group(1)), m.group(2), "_E_" in str(label)


class Selection:
    def __init__(self, labels):
        self.pedicles = {}                         # (level, side) -> {"entry": i, "isthmus": j}
        for i, lbl in enumerate(labels):
            p = parse_label(lbl)
            if p is None:
                continue
            lv, side, is_entry = p
            self.pedicles.setdefault((lv, side), {})["entry" if is_entry else "isthmus"] = i
        # keep only pedicles that have both ends, so a trajectory can always be drawn
        self.pedicles = {k: v for k, v in self.pedicles.items()
                         if "entry" in v and "isthmus" in v}
        self.order = sorted(self.pedicles)         # L1 L, L1 R, L2 L, ... the ten pedicles
        self.levels = sorted({lv for lv, _ in self.order})
        self.current = None                        # None = overview

    # ---- reading ----
    @property
    def is_overview(self):
        return self.current is None

    @property
    def level(self):
        return None if self.current is None else self.current[0]

    @property
    def side(self):
        return None if self.current is None else self.current[1]

    @property
    def entry_idx(self):
        return None if self.current is None else self.pedicles[self.current]["entry"]

    @property
    def isthmus_idx(self):
        return None if self.current is None else self.pedicles[self.current]["isthmus"]

    @property
    def all_entry_idx(self):
        return [v["entry"] for v in self.pedicles.values()]

    @property
    def name(self):
        """'L3 left' for the HUD, 'all levels' in overview."""
        if self.current is None:
            return "all levels"
        return f"L{self.current[0]} {SIDE_NAMES[self.current[1]]}"

    def neighbour_levels(self):
        """(previous, next) level numbers for hints, None at the ends."""
        if self.current is None:
            return None, None
        k = self.levels.index(self.current[0])
        return (self.levels[k - 1] if k > 0 else None,
                self.levels[k + 1] if k < len(self.levels) - 1 else None)

    # ---- changing (each returns True if the selection changed) ----
    def _set(self, new):
        if new is not None and new not in self.pedicles:
            return False
        changed = new != self.current
        self.current = new
        return changed

    def overview(self):
        return self._set(None)

    def select(self, level=None, side=None):
        """Select a pedicle. Missing parts default to the current ones, then L1 / left."""
        level = level if level is not None else (self.level if self.level is not None
                                                 else (self.levels[0] if self.levels else None))
        side = side if side is not None else (self.side or "L")
        if (level, side) not in self.pedicles:           # e.g. only one side annotated
            other = "R" if side == "L" else "L"
            if (level, other) in self.pedicles:
                side = other
        return self._set((level, side))

    def step_level(self, delta):
        """Previous (-1) / next (+1) vertebra, keeping the side. Stops at the ends."""
        if self.current is None:
            return self.select(self.levels[0] if delta > 0 else self.levels[-1])
        k = self.levels.index(self.current[0]) + delta
        if not 0 <= k < len(self.levels):
            return False
        return self.select(self.levels[k], self.current[1])

    def set_side(self, side):
        return self.select(side=side)

    def step_pedicle(self, delta):
        """Walk the ten pedicles in order (L1 left, L1 right, L2 left, ...). Stops at the ends."""
        if not self.order:
            return False
        if self.current is None:
            return self._set(self.order[0] if delta > 0 else self.order[-1])
        k = self.order.index(self.current) + delta
        return self._set(self.order[k]) if 0 <= k < len(self.order) else False