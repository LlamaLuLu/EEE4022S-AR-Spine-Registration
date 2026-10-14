import unittest

import cv2
import numpy as np

import context  # noqa: F401
from ar_check import entry_check

K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
D = np.zeros(5)
TARGET = np.array([12.0, -8.0, 25.0])


def project(p, rvec, tvec):
    c, _ = cv2.projectPoints(np.asarray(p, float).reshape(1, 3), rvec, tvec, K, D)
    return c.reshape(2)


class Check(unittest.TestCase):
    rvec = np.array([0.3, -0.5, 0.2])
    tvec = np.array([10.0, -20.0, 550.0])

    def test_clicking_the_target_gives_zero(self):
        c = entry_check(project(TARGET, self.rvec, self.tvec), self.rvec, self.tvec, K, D, TARGET)
        self.assertLess(c["err_mm"], 1e-6)
        self.assertLess(c["err_px"], 1e-6)

    def test_ten_pixels_is_ten_over_f_times_depth(self):
        u, v = project(TARGET, self.rvec, self.tvec)
        c = entry_check((u + 10.0, v), self.rvec, self.tvec, K, D, TARGET)
        self.assertAlmostEqual(c["err_mm"], 10.0 / 1000.0 * c["depth_mm"], places=6)
        self.assertAlmostEqual(c["err_px"], 10.0, places=6)

    def test_error_has_no_depth_component(self):
        u, v = project(TARGET, self.rvec, self.tvec)
        c = entry_check((u + 7.0, v - 4.0), self.rvec, self.tvec, K, D, TARGET)
        R, _ = cv2.Rodrigues(self.rvec.reshape(3, 1))
        along_view = float((R @ c["dv_mm"])[2])       # camera z of the error vector
        self.assertAlmostEqual(along_view, 0.0, places=6)

    def test_depth_is_the_targets_camera_z(self):
        c = entry_check(project(TARGET, self.rvec, self.tvec), self.rvec, self.tvec, K, D, TARGET)
        R, _ = cv2.Rodrigues(self.rvec.reshape(3, 1))
        self.assertAlmostEqual(c["depth_mm"], float((R @ TARGET + self.tvec)[2]), places=6)

    def test_click_point_is_in_the_phantom_frame(self):
        # the click point must land on the target when the click is on the target
        c = entry_check(project(TARGET, self.rvec, self.tvec), self.rvec, self.tvec, K, D, TARGET)
        np.testing.assert_allclose(c["Q_mm"], TARGET, atol=1e-6)


if __name__ == "__main__":
    unittest.main()