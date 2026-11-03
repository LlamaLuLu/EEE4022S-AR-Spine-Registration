# Data

Calibration, marker maps and acceptance test logs used by the scripts and the report. The raw captures and the CT volumes are too large for the repository and are left out (listed below).

| Folder | Contents |
|---|---|
| `calibration/` | `camera_calibration.npz` for each calibration run (`s22_ultra` is the one the Makefile uses) and the printable boards |
| `bracket_markers/`, `dome_markers/` | Marker maps and printable pages for P1 (L-bracket) and P2 (dome) |
| `fle_measurement/` | Static-board frames for the FLE and pixel noise measurement |
| `atp/`, `atp_v2/` | Per-frame logs and summaries for the acceptance tests: `atp/` for P1, `atp_v2/` for P2 |
| `screenshots/` | The `*_pose.json` file saved with each screenshot, and `click_checks.csv` |
| `verse20/derivatives/` | Segmentation centroids and previews for the VerSe cases screened for the phantom (sub-verse835 was used) |

## Not included

| Left out | Why | Where to get it |
|---|---|---|
| `calibration/*/calibration_imgs/` | Calibration photos, about 115 MB | Not needed: the resulting `.npz` files are included |
| `screenshots/**/*.png` | Raw capture images, about 1.3 GB | Not needed for the analysis scripts, which read the pose files. Needed to replay a capture in the interface. Download `screenshots.zip` from the [data release](https://github.com/LlamaLuLu/EEE4022S-AR-Spine-Registration/releases/tag/data-v1) and unzip it into `data/screenshots/` |
| `verse20/rawdata/`, `*.nii`, `*.mrb` | CT volumes and the Slicer scene, about 900 MB | [VerSe on GitHub](https://github.com/anjany/verse) links to the open dataset. The target annotations made on sub-verse835 are in `annotations/` |

Scripts that read the missing files, such as `scripts/tools/filter_verse_shortlist.py` (segmentation volumes) and the replay mode of the live interface (capture images), will not run until those files are put back in place.
