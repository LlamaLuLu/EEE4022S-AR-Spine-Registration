import unittest

import numpy as np

import context  # noqa: F401
from ar_error import ANCHORS, RAYLEIGH_95, BAD_MM, GOOD_MM, error_radius95, expected_rms, grade


class ErrorModel(unittest.TestCase):
    def test_at_the_anchor_the_error_is_the_measured_rms(self):
        for a in ANCHORS.values():
            self.assertAlmostEqual(float(expected_rms(a.tre_mm, a)), a.rms_mm)
            self.assertAlmostEqual(float(error_radius95(a.tre_mm, a)), RAYLEIGH_95 * a.rms_mm)

    def test_better_than_the_anchor_never_goes_below_the_measured_rms(self):
        a = ANCHORS["dome"]
        self.assertAlmostEqual(float(expected_rms(0.2, a)), a.rms_mm)

    def test_grows_with_predicted_tre(self):
        a = ANCHORS["bracket"]
        r = error_radius95(np.array([1.0, 2.0, 3.0, 4.0]), a)
        self.assertTrue(np.all(np.diff(r) > 0))

    def test_adds_in_quadrature_above_the_anchor(self):
        a = ANCHORS["dome"]
        tre = 2.5
        want = np.sqrt(a.rms_mm ** 2 + tre ** 2 - a.tre_mm ** 2)
        self.assertAlmostEqual(float(expected_rms(tre, a)), want)


class Grades(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(grade(GOOD_MM - 0.01, "full"), "good")
        self.assertEqual(grade(GOOD_MM + 0.01, "full"), "warn")
        self.assertEqual(grade(BAD_MM - 0.01, "full"), "warn")
        self.assertEqual(grade(BAD_MM + 0.01, "full"), "bad")

    def test_envelope_overrides(self):
        self.assertEqual(grade(1.0, "outside"), "bad")
        self.assertEqual(grade(1.0, "reduced"), "warn")      # reduced tier is capped at amber

    def test_not_a_number_is_bad(self):
        self.assertEqual(grade(float("nan"), "full"), "bad")
        self.assertEqual(grade(float("inf"), "full"), "bad")


if __name__ == "__main__":
    unittest.main()