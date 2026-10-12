import csv, json, sys, glob
from pathlib import Path
import numpy as np
sys.path[:0] = ["scripts/registration", "scripts/tre_analysis"]
from ar_targets import load_targets
from ar_board import load_corner_lut
from ar_tre import compute_tre_scalar

lut = load_corner_lut("data/dome_markers/marker_map.json")
pts, labels, _ = load_targets("annotations/pedicle_isthmus_targets.json",
                              "annotations/pedicle_entry_targets.json")
cen = np.mean([c.mean(axis=0) for c in lut.values()], axis=0)
poses = {Path(f).stem.replace("_pose", ""): json.load(open(f))
         for f in glob.glob("data/screenshots/atp2_v2/shot_*_pose.json")}
for path in ("data/atp_v2/atp2_dome.csv", "data/atp_v2/atp2_dome_3d.csv"):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        i = labels.index(r["label"])
        r["dist_from_markers_mm"] = round(float(np.linalg.norm(pts[i] - cen)), 1)
        if "shot" in r:
            p = poses[r["shot"]]
            r["pred_tre_mm"] = round(float(compute_tre_scalar(p["visible_ids"], lut, pts, 0.8)[i]), 3)
    w = csv.DictWriter(open(path, "w", newline=""), fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
    print("fixed", path)