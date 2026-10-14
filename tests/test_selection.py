import unittest

import context  # noqa: F401
from ar_selection import Selection, parse_label


def labels(levels=(1, 2, 3, 4, 5), sides="LR"):
    out = []
    for lv in levels:
        for sd in sides:
            out += [f"L{lv}_E_{sd}", f"L{lv}_{sd}"]
    return out


class Labels(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_label("L3_E_R"), (3, "R", True))
        self.assertEqual(parse_label("L3_L"), (3, "L", False))
        self.assertIsNone(parse_label("sacrum"))


class Stepping(unittest.TestCase):
    def setUp(self):
        self.s = Selection(labels())

    def test_ten_pedicles_in_order(self):
        self.assertEqual(len(self.s.order), 10)
        self.assertEqual(self.s.order[0], (1, "L"))
        self.assertEqual(self.s.order[-1], (5, "R"))
        self.assertTrue(self.s.is_overview)
        self.assertEqual(self.s.name, "all levels")

    def test_indices_point_at_the_right_labels(self):
        lbl = labels()
        self.s.select(3, "R")
        self.assertEqual(lbl[self.s.entry_idx], "L3_E_R")
        self.assertEqual(lbl[self.s.isthmus_idx], "L3_R")
        self.assertEqual(len(self.s.all_entry_idx), 10)

    def test_level_steps_stop_at_the_ends(self):
        self.s.select(1, "L")
        self.assertFalse(self.s.step_level(-1))
        self.assertEqual(self.s.current, (1, "L"))
        self.s.select(5, "L")
        self.assertFalse(self.s.step_level(+1))
        self.assertTrue(self.s.step_level(-1))
        self.assertEqual(self.s.current, (4, "L"))

    def test_side_is_kept_when_changing_level(self):
        self.s.select(2, "R")
        self.s.step_level(+1)
        self.assertEqual(self.s.current, (3, "R"))

    def test_first_step_from_overview(self):
        self.s.step_level(+1)
        self.assertEqual(self.s.current, (1, "L"))
        self.s.overview()
        self.s.step_level(-1)
        self.assertEqual(self.s.current[0], 5)

    def test_pedicle_walk(self):
        self.s.step_pedicle(+1)
        self.assertEqual(self.s.current, (1, "L"))
        self.s.step_pedicle(+1)
        self.assertEqual(self.s.current, (1, "R"))
        self.s.step_pedicle(+1)
        self.assertEqual(self.s.current, (2, "L"))
        self.s.select(5, "R")
        self.assertFalse(self.s.step_pedicle(+1))

    def test_change_reports_whether_anything_changed(self):
        self.assertTrue(self.s.select(2, "L"))
        self.assertFalse(self.s.select(2, "L"))
        self.assertTrue(self.s.overview())
        self.assertFalse(self.s.overview())

    def test_unknown_pedicle_is_refused(self):
        self.assertFalse(self.s.select(9, "L"))
        self.assertTrue(self.s.is_overview)


class Incomplete(unittest.TestCase):
    def test_pedicle_without_isthmus_is_dropped(self):
        s = Selection(labels(levels=(1, 2)) + ["L3_E_L"])
        self.assertEqual(s.levels, [1, 2])

    def test_missing_side_falls_back_to_the_other(self):
        s = Selection(labels(levels=(1, 2), sides="R"))
        s.select(1, "L")
        self.assertEqual(s.current, (1, "R"))


if __name__ == "__main__":
    unittest.main()