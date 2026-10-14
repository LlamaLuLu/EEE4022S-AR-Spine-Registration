import unittest
from types import SimpleNamespace as NS

import numpy as np

import context  # noqa: F401
from ar_jitter import MIN_SAMPLES, JitterMonitor

X = np.array([[10.0, 20.0, 30.0], [-40.0, 5.0, 60.0]])
RVEC = np.array([0.1, 0.2, 0.0])


def feed(mon, tvecs, dt=1 / 30, start=0.0):
    for i, t in enumerate(tvecs):
        mon.push(RVEC, t, start + i * dt)


class Jitter(unittest.TestCase):
    def test_needs_enough_poses(self):
        m = JitterMonitor()
        feed(m, [np.array([0, 0, 500.0])] * (MIN_SAMPLES - 1))
        self.assertIsNone(m.measure(X))
        feed(m, [np.array([0, 0, 500.0])] * 2, start=1.0 / 30 * MIN_SAMPLES)
        self.assertIsNotNone(m.measure(X))

    def test_a_still_pose_has_no_jitter(self):
        m = JitterMonitor()
        feed(m, [np.array([1.0, 2.0, 500.0])] * 15)
        total, depth, lateral = m.measure(X)
        self.assertTrue(np.all(total < 1e-9))

    def test_steady_motion_is_not_jitter(self):
        m = JitterMonitor()
        feed(m, [np.array([0.5 * i, 0.0, 500.0 + 0.3 * i]) for i in range(15)])
        self.assertTrue(np.all(m.measure(X)[0] < 1e-9))

    def test_noise_is_measured_in_mm(self):
        rng = np.random.default_rng(0)
        m = JitterMonitor(n=200)
        feed(m, [np.array([0, 0, 500.0]) + rng.normal(0, 0.1, 3) for _ in range(200)])
        total, depth, lateral = m.measure(X)
        self.assertTrue(np.all(np.abs(total - 0.1 * np.sqrt(3)) < 0.03))

    def test_depth_noise_shows_as_depth(self):
        rng = np.random.default_rng(1)
        m = JitterMonitor(n=200)
        feed(m, [np.array([0, 0, 500.0 + rng.normal(0, 0.5)]) for _ in range(200)])
        total, depth, lateral = m.measure(X)
        self.assertTrue(np.all(depth > 5 * lateral))

    def test_window_keeps_only_the_last_n(self):
        m = JitterMonitor(n=10)
        feed(m, [np.array([0, 0, 500.0])] * 30)
        self.assertEqual(m.n_samples, 10)

    def test_a_gap_starts_a_new_window(self):
        m = JitterMonitor(max_gap=0.5)
        feed(m, [np.array([0, 0, 500.0])] * 10)
        m.push(RVEC, np.array([0, 0, 500.0]), 5.0)
        self.assertEqual(m.n_samples, 1)

    def test_lost_tracking_resets(self):
        m = JitterMonitor()
        feed(m, [np.array([0, 0, 500.0])] * 10)
        m.update(NS(rvec=None, tvec=None, pnp_ok=False, state="SEARCHING", now=1.0))
        self.assertEqual(m.n_samples, 0)
        m.update(NS(rvec=RVEC, tvec=np.array([0, 0, 500.0]), pnp_ok=True, state="STALE", now=1.1))
        self.assertEqual(m.n_samples, 0)
        m.update(NS(rvec=RVEC, tvec=np.array([0, 0, 500.0]), pnp_ok=True, state="NAVIGATING", now=1.2))
        self.assertEqual(m.n_samples, 1)


if __name__ == "__main__":
    unittest.main()