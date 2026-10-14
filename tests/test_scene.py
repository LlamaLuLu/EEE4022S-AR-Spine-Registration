import json
import unittest
from types import SimpleNamespace as NS

import numpy as np

import context  # noqa: F401
from ar_error import ANCHORS, RAYLEIGH_95
from ar_scene import SCENE_VERSION, build_scene, error_summary, scene_from_json, scene_to_json
from ar_selection import Selection

LABELS = []
for lv in range(1, 6):
    for sd in "LR":
        LABELS += [f"L{lv}_E_{sd}", f"L{lv}_{sd}"]
PTS = np.random.default_rng(1).normal(size=(20, 3)) * 30
ANCHOR = ANCHORS["dome"]


def result(**kw):
    base = dict(now=12.3456, state="NAVIGATING", verify_remaining=0.0,
                rvec=np.array([0.1, 0.2, 0.3]), tvec=np.array([[1.0, 2.0, 500.0]]),
                pnp_ok=True, n_markers=8, visible_ids=list(range(8)), reproj_rms=0.9,
                distance_mm=520.0, elev_deg=24.0, az_deg=45.0, env_tier="full",
                tre_values=np.full(20, 1.03))
    base.update(kw)
    return NS(**base)


def searching():
    return result(state="SEARCHING", rvec=None, tvec=None, tre_values=None, pnp_ok=False,
                  n_markers=0, visible_ids=[], reproj_rms=float("nan"), env_tier="outside")


class Scene(unittest.TestCase):
    def setUp(self):
        self.sel = Selection(LABELS)

    def build(self, r, **kw):
        return build_scene(r, self.sel, PTS, LABELS, ANCHOR, False, 8, **kw)

    def test_round_trip(self):
        for r in (result(), searching()):
            for cur in (None, (2, "L")):
                self.sel.current = cur
                sc = self.build(r, jitter=(0.12, 0.1, 15))
                self.assertEqual(scene_from_json(scene_to_json(sc)), sc)

    def test_nan_becomes_null_and_json_is_strict(self):
        sc = self.build(result(reproj_rms=float("nan")))
        self.assertIsNone(sc["quality"]["rms_px"])
        json.loads(scene_to_json(sc), parse_constant=lambda c: self.fail(c))   # no NaN or Infinity tokens

    def test_error_at_the_anchor(self):
        self.sel.select(2, "L")
        sc = self.build(result())
        self.assertAlmostEqual(sc["error"]["r95_mm"], RAYLEIGH_95 * ANCHOR.rms_mm, places=2)
        self.assertEqual(sc["error"]["label"], "L2_E_L")
        self.assertEqual(sc["error"]["grade"], "good")

    def test_overview_reports_the_worst_entry(self):
        tre = np.full(20, 1.0)
        worst = LABELS.index("L4_E_R")
        tre[worst] = 3.0
        sc = self.build(result(tre_values=tre))
        self.assertTrue(sc["selection"]["overview"])
        self.assertEqual(sc["error"]["target"], worst)

    def test_isthmus_points_are_never_the_headline(self):
        tre = np.full(20, 1.0)
        tre[LABELS.index("L1_L")] = 9.0                      # a bad isthmus point
        self.assertNotEqual(error_summary(result(tre_values=tre), self.sel, ANCHOR)[0], LABELS.index("L1_L"))

    def test_searching_has_no_error_and_no_pose(self):
        sc = self.build(searching())
        self.assertIsNone(sc["error"])
        self.assertIsNone(sc["pose"])
        self.assertEqual(sc["trust"]["status"], "stale")

    def test_outside_the_envelope_grades_bad(self):
        self.sel.select(1, "L")
        sc = self.build(result(env_tier="outside"))
        self.assertEqual(sc["error"]["grade"], "bad")

    def test_targets_and_pedicles(self):
        sc = self.build(result())
        self.assertEqual(len(sc["targets"]), 20)
        self.assertEqual(len(sc["pedicles"]), 10)
        t = sc["targets"][LABELS.index("L3_E_R")]
        self.assertEqual((t["level"], t["side"], t["kind"]), (3, "R", "entry"))
        np.testing.assert_allclose(t["xyz"], PTS[LABELS.index("L3_E_R")], atol=1e-3)

    def test_jitter_is_optional(self):
        self.assertIsNone(self.build(result())["jitter"])
        self.assertEqual(self.build(result(), jitter=(0.1, 0.08, 15))["jitter"]["n"], 15)

    def test_version_is_checked(self):
        sc = self.build(result())
        sc["version"] = SCENE_VERSION + 1
        with self.assertRaises(ValueError):
            scene_from_json(json.dumps(sc))


if __name__ == "__main__":
    unittest.main()