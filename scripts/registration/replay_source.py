"""Replay source: feeds saved frames to the live UI as if they came from a camera.

Usage (in live_ar_overlay_qt.py):  --source data/screenshots/atp2_crisp
  - a folder   -> every *_clean.png inside, sorted by name
  - a .png/.jpg -> that single frame
Pose estimation runs on the image exactly as it does live, so this tests the
full pipeline, not just the drawing. Saved *_clean.png frames are already
rotated, so do not pass --rotate when replaying.
"""
import time
from pathlib import Path
import cv2

IMG_EXT = {".png", ".jpg", ".jpeg"}


class ReplayCapture:
    """Mimics the parts of cv2.VideoCapture that FrameGrabber uses."""

    def __init__(self, path, fps=30.0):
        p = Path(path)
        if p.is_dir():
            files = sorted(f for f in p.glob("*_clean.png"))
            if not files:
                files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        else:
            files = [p]
        self.files = files
        self.idx = 0
        self._dt = 1.0 / fps
        self._cache = {}
        print(f"Replay: {len(files)} frame(s) from {path}")

    def isOpened(self):
        return len(self.files) > 0

    def _load(self, i):
        if i not in self._cache:
            self._cache = {i: cv2.imread(str(self.files[i]))}  # keep one frame
        return self._cache[i]

    def read(self):
        time.sleep(self._dt)               # pace like a live camera
        img = self._load(self.idx)
        return (img is not None), (None if img is None else img.copy())

    def step(self, delta):
        self.idx = (self.idx + delta) % len(self.files)
        return self.files[self.idx].name

    def release(self):
        pass

    def set(self, *_):
        return False

    def get(self, *_):
        return 0.0


def is_replay_source(source):
    p = Path(str(source))
    return p.is_dir() or (p.suffix.lower() in IMG_EXT and p.exists())