import unittest

import numpy as np

import context  # noqa: F401
from ui.axial import slice_mesh
from ui.zoom import project_point, zoom_frame

K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
D = np.zeros(5)


def cube(side=10.0):
    """Axis aligned cube centred on the origin: vertices and 12 triangles."""
    h = side / 2
    v = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    t = []
    for a, b, c, d in quads:
        t += [(a, b, c), (a, c, d)]
    return v, np.array(t)


class SliceMesh(unittest.TestCase):
    def test_cube_cut_through_the_middle_is_a_square(self):
        v, t = cube(10.0)
        seg = slice_mesh(v, t, [0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0])
        pts = seg.reshape(-1, 2)
        self.assertTrue(np.allclose(np.abs(pts).max(axis=0), 5.0))
        length = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1).sum()
        self.assertAlmostEqual(length, 40.0, places=6)           # perimeter of a 10 mm square

    def test_plane_outside_the_mesh_gives_nothing(self):
        v, t = cube(10.0)
        seg = slice_mesh(v, t, [0, 0, 20], [0, 0, 1], [1, 0, 0], [0, 1, 0])
        self.assertEqual(seg.shape, (0, 2, 2))

    def test_plane_through_a_vertex_plane_still_closes(self):
        v, t = cube(10.0)
        seg = slice_mesh(v, t, [0, 0, 5.0], [0, 0, 1], [1, 0, 0], [0, 1, 0])   # exactly on a face
        self.assertIsNotNone(seg)                                 # does not crash on d == 0

    def test_plane_coordinates_are_relative_to_the_cut_point(self):
        v, t = cube(10.0)
        seg = slice_mesh(v, t, [3, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0])
        self.assertAlmostEqual(seg[..., 0].min(), -8.0, places=6)    # world x of -5 and 5, minus the point's 3
        self.assertAlmostEqual(seg[..., 0].max(), 2.0, places=6)
        self.assertAlmostEqual(seg[..., 1].min(), -5.0, places=6)
        self.assertAlmostEqual(seg[..., 1].max(), 5.0, places=6)

    def test_e1_e2_choose_the_axes(self):
        v, t = cube(10.0)
        stretched = v * np.array([2.0, 1.0, 1.0])                    # 20 mm along x, 10 mm along y
        seg = slice_mesh(stretched, t, [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0])   # swap the axes
        self.assertAlmostEqual(np.ptp(seg[..., 0]), 10.0, places=6)
        self.assertAlmostEqual(np.ptp(seg[..., 1]), 20.0, places=6)


class Projection(unittest.TestCase):
    def test_point_on_the_optical_axis_lands_on_the_principal_point(self):
        u, v, z = project_point([0, 0, 0], np.zeros(3), np.array([0.0, 0.0, 500.0]), K, D)
        self.assertAlmostEqual(u, 640.0)
        self.assertAlmostEqual(v, 360.0)
        self.assertAlmostEqual(z, 500.0)

    def test_behind_the_camera_is_none(self):
        self.assertIsNone(project_point([0, 0, 0], np.zeros(3), np.array([0.0, 0.0, -10.0]), K, D))


class Zoom(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((720, 1280, 3), np.uint8)

    def test_magnification_matches_the_requested_field_of_view(self):
        _, _, s = zoom_frame(self.img, K, (640, 360), 2.3, 800, 600, 60.0)
        self.assertAlmostEqual(s, 800 / (60.0 * 2.3))

    def test_centre_lands_in_the_middle_of_the_output(self):
        _, Kz, s = zoom_frame(self.img, K, (500.0, 300.0), 2.3, 800, 600, 60.0)
        # a camera pixel at the crop centre, pushed through the zoom matrix, is the output centre
        p = Kz @ np.linalg.inv(K) @ np.array([500.0, 300.0, 1.0])
        self.assertAlmostEqual(p[0] / p[2], 400.0, places=6)
        self.assertAlmostEqual(p[1] / p[2], 300.0, places=6)

    def test_zoom_matrix_projects_like_the_camera_then_scales(self):
        rvec, tvec = np.array([0.1, 0.2, 0.0]), np.array([5.0, -3.0, 480.0])
        pt = [4.0, 6.0, 12.0]
        u, v, _ = project_point(pt, rvec, tvec, K, D)
        _, Kz, s = zoom_frame(self.img, K, (u, v), 2.3, 800, 600, 60.0)
        uz, vz, _ = project_point(pt, rvec, tvec, Kz, D)
        self.assertAlmostEqual(uz, 400.0, places=6)               # the point we centred on
        self.assertAlmostEqual(vz, 300.0, places=6)

    def test_outside_the_frame_is_black(self):
        img = np.full((720, 1280, 3), 255, np.uint8)
        out, _, _ = zoom_frame(img, K, (0.0, 0.0), 2.3, 200, 200, 30.0)
        self.assertEqual(int(out[0, 0].max()), 0)                  # top left corner is off the camera frame
        self.assertEqual(int(out[190, 190].max()), 255)


if __name__ == "__main__":
    unittest.main()