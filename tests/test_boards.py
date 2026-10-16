import unittest

import numpy as np

import context  # noqa: F401
from ar_boards import BoardSelector, make_board, SWITCH_FRAMES
from ar_tracker import Tracker


class FakeLut(dict):
    def __init__(self, ids, gate):
        super().__init__({i: np.zeros((4, 3), np.float32) for i in ids})
        self.min_markers, self.marker_size_mm, self.source = gate, 20.0, "fake"


def dome():
    return make_board(FakeLut(range(10, 17), 5))


def bracket():
    return make_board(FakeLut(range(8), 4))


def feed(sel, ids, n):
    return [sel.update(np.array(ids, np.int32).reshape(-1, 1)) for _ in range(n)]


class MakeBoard(unittest.TestCase):
    def test_dome_and_bracket_are_told_apart(self):
        d, b = dome(), bracket()
        self.assertEqual((d.name, d.is_v3, d.gate, d.n_total), ("dome", False, 5, 7))
        self.assertEqual((b.name, b.is_v3, b.gate, b.n_total), ("bracket", True, 4, 8))


class Auto(unittest.TestCase):
    def setUp(self):
        self.sel = BoardSelector([bracket(), dome()])

    def test_starts_on_dome(self):
        self.assertEqual(self.sel.board.name, "dome")
        self.assertEqual(self.sel.label, "Dome (auto)")

    def test_no_markers_keeps_the_board(self):
        self.assertFalse(self.sel.update(None))
        self.assertFalse(self.sel.update(np.zeros((0, 1), np.int32)))
        self.assertEqual(self.sel.board.name, "dome")

    def test_switches_after_enough_consecutive_frames(self):
        flags = feed(self.sel, [0, 1, 2, 3], SWITCH_FRAMES)
        self.assertEqual(flags, [False] * (SWITCH_FRAMES - 1) + [True])
        self.assertEqual(self.sel.board.name, "bracket")

    def test_a_glitch_frame_resets_the_streak(self):
        feed(self.sel, [0, 1, 2], SWITCH_FRAMES - 1)
        feed(self.sel, [10, 11, 12], 1)
        feed(self.sel, [0, 1, 2], SWITCH_FRAMES - 1)
        self.assertEqual(self.sel.board.name, "dome")

    def test_tie_keeps_the_current_board(self):
        feed(self.sel, [10, 0], SWITCH_FRAMES * 2)
        self.assertEqual(self.sel.board.name, "dome")

    def test_switches_back(self):
        feed(self.sel, [0, 1, 2, 3], SWITCH_FRAMES)
        feed(self.sel, [10, 11, 12, 13, 14], SWITCH_FRAMES)
        self.assertEqual(self.sel.board.name, "dome")


class Modes(unittest.TestCase):
    def test_fixed_never_switches(self):
        sel = BoardSelector([dome(), bracket()], "dome")
        feed(sel, [0, 1, 2, 3], 20)
        self.assertEqual(sel.board.name, "dome")

    def test_cycle_order_and_change_flags(self):
        sel = BoardSelector([dome(), bracket()])
        self.assertEqual(sel.mode, "auto")
        self.assertFalse(sel.cycle_mode())      # auto -> dome, already on dome
        self.assertTrue(sel.cycle_mode())       # dome -> bracket
        self.assertFalse(sel.cycle_mode())      # bracket -> auto, stays until markers say otherwise
        self.assertEqual((sel.mode, sel.board.name), ("auto", "bracket"))

    def test_one_board_has_nothing_to_choose(self):
        sel = BoardSelector([dome()])
        self.assertEqual(sel.modes, ["dome"])
        self.assertEqual(sel.label, "Dome")

    def test_unknown_mode_is_an_error(self):
        with self.assertRaises(ValueError):
            BoardSelector([dome()], "bracket")


class FakeDetector:
    def __init__(self, ids):
        self.ids = np.array(ids, np.int32).reshape(-1, 1)

    def detectMarkers(self, gray):
        return [], self.ids, None


class TrackerSwitch(unittest.TestCase):
    def test_tracker_reports_and_resets_on_a_switch(self):
        sel = BoardSelector([dome(), bracket()])
        tr = Tracker(FakeDetector([0, 1, 2, 3]), sel, np.eye(3), np.zeros(5), np.zeros((2, 3)), 0.2)
        tr.state = "NAVIGATING"
        tr.prev_rvec = tr.prev_tvec = np.zeros(3)
        frame = np.zeros((8, 8, 3), np.uint8)
        results = [tr.process(frame, i * 0.03) for i in range(SWITCH_FRAMES)]
        self.assertEqual([r.board for r in results[:-1]], ["dome"] * (SWITCH_FRAMES - 1))
        self.assertTrue(results[-1].board_switched)
        self.assertEqual(results[-1].board, "bracket")
        self.assertEqual(tr.state, "SEARCHING")


if __name__ == "__main__":
    unittest.main()