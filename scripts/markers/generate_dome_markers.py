"""
Print-ready ArUco stickers for the D1_final dome, at exact physical size.

Each sticker is a 25 mm black ArUco square (DICT_6X6_50, including its one-cell
black border) centred in a 32 mm white square, so the white margin is 3.5 mm per
side (more than the one-cell quiet zone of 25/8 = 3.125 mm). Crop marks sit
OUTSIDE the 32 mm square, so nothing dark is printed on the sticker margin.
The ID label and a TOP arrow are also outside the cut line: the arrow marks the
marker's top edge (the TL-TR edge in the marker map).

Placement on the dome:
  ID 10      top marker, TOP edge points cranial (+Z)
  ID 11..16  ring markers at 30, 90, 150, 210, 270, 330 deg, TOP edge points outward
  ID 17      spare for the single-face fit test (not part of the dome)

Usage (from any directory; output goes to <repo>/data/dome_markers/ by default):
  python scripts/markers/generate_dome_markers.py
  python scripts/markers/generate_dome_markers.py --ids 10 11 12 --name test_markers.pdf

Print at 100% / "Actual size" (turn off "fit to page"), on matte paper.
Then measure the 100 mm scale bar and one black square with calipers: use the
measured black-square size in the live marker map if it is not exactly 25.0 mm.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

REPO_ROOT = Path(__file__).resolve().parents[2]      # scripts/markers/<this file>
OUT_DIR = REPO_ROOT / "data" / "dome_markers"

DICT = cv2.aruco.DICT_6X6_50
CELLS = 8                      # 6x6 pattern plus a one-cell black border


def marker_bits(marker_id):
    """8x8 array, True where the cell is black. Row 0 is the marker's top edge."""
    d = cv2.aruco.getPredefinedDictionary(DICT)
    img = cv2.aruco.generateImageMarker(d, marker_id, CELLS, borderBits=1)
    return img < 128


def draw_sticker(c, x0, y0, marker_id, marker_mm, sticker_mm, label):
    """Draw one sticker with its lower-left corner (of the 32 mm square) at (x0, y0) in mm."""
    margin = (sticker_mm - marker_mm) / 2.0
    cell = marker_mm / CELLS
    bits = marker_bits(marker_id)

    # black square first, then white cells on top (avoids hairline seams inside the border)
    mx, my = x0 + margin, y0 + margin
    c.setFillColorRGB(0, 0, 0)
    c.rect(mx * mm, my * mm, marker_mm * mm, marker_mm * mm, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1)
    for r in range(CELLS):
        row_y = my + (CELLS - 1 - r) * cell          # PDF y grows upward; row 0 is the top
        col = 0
        while col < CELLS:                            # merge horizontal runs of white cells
            if bits[r, col]:
                col += 1
                continue
            start = col
            while col < CELLS and not bits[r, col]:
                col += 1
            c.rect((mx + start * cell) * mm, row_y * mm, (col - start) * cell * mm, cell * mm,
                   stroke=0, fill=1)

    # dotted cut outline exactly on the sticker edge: light grey and thin, 3.5 mm away
    # from the black border, so any trace left after cutting does not affect detection
    c.setStrokeColorRGB(0.7, 0.7, 0.7)
    c.setLineWidth(0.25)
    c.setDash(1.5, 1.5)
    c.rect(x0 * mm, y0 * mm, sticker_mm * mm, sticker_mm * mm, stroke=1, fill=0)
    c.setDash()

    # crop marks, outside the sticker, with a 1 mm gap from each corner
    c.setStrokeColorRGB(0.55, 0.55, 0.55)
    c.setLineWidth(0.2)
    gap, length = 1.0, 4.0
    for cx, sx in ((x0, -1), (x0 + sticker_mm, 1)):
        for cy, sy in ((y0, -1), (y0 + sticker_mm, 1)):
            c.line(cx * mm, (cy + sy * gap) * mm, cx * mm, (cy + sy * (gap + length)) * mm)
            c.line((cx + sx * gap) * mm, cy * mm, (cx + sx * (gap + length)) * mm, cy * mm)

    # label and TOP arrow above the sticker, outside the cut line
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 8)
    cx = x0 + sticker_mm / 2
    c.drawCentredString(cx * mm, (y0 + sticker_mm + 7.5) * mm, f"ID {marker_id}  TOP")
    c.setLineWidth(0.5)
    c.setStrokeColorRGB(0, 0, 0)
    c.line(cx * mm, (y0 + sticker_mm + 2.0) * mm, cx * mm, (y0 + sticker_mm + 6.0) * mm)
    c.line((cx - 1.2) * mm, (y0 + sticker_mm + 4.8) * mm, cx * mm, (y0 + sticker_mm + 6.0) * mm)
    c.line((cx + 1.2) * mm, (y0 + sticker_mm + 4.8) * mm, cx * mm, (y0 + sticker_mm + 6.0) * mm)
    c.setFont("Helvetica", 6.5)
    c.drawCentredString(cx * mm, (y0 - 7.0) * mm, label)


def scale_bar(c, x0, y0):
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.4)
    c.line(x0 * mm, y0 * mm, (x0 + 100) * mm, y0 * mm)
    for t in range(0, 101, 10):
        h = 3 if t % 50 == 0 else 1.5
        c.line((x0 + t) * mm, y0 * mm, (x0 + t) * mm, (y0 + h) * mm)
    c.setFont("Helvetica", 7)
    c.drawString(x0 * mm, (y0 - 4) * mm, "Scale check: this bar must measure 100.0 mm. If not, reprint at 100% / Actual size.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", type=int, nargs="+", default=[10, 11, 12, 13, 14, 15, 16, 17])
    ap.add_argument("--marker-mm", type=float, default=25.0, help="black square edge, including its black border")
    ap.add_argument("--sticker-mm", type=float, default=32.0, help="cut size, including the white margin")
    ap.add_argument("--outdir", default=str(OUT_DIR), help="output folder (default: <repo>/data/dome_markers)")
    ap.add_argument("--name", default="dome_markers_D1_final.pdf", help="PDF file name")
    a = ap.parse_args()

    assert (a.sticker_mm - a.marker_mm) / 2 >= a.marker_mm / CELLS, \
        "white margin is thinner than one marker cell (quiet zone too small)"

    labels = {10: "top marker, TOP = cranial", 17: "spare, single-face test"}
    for i, ang in zip(range(11, 17), (30, 90, 150, 210, 270, 330)):
        labels[i] = f"ring {ang} deg, TOP = outward"

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_pdf = outdir / a.name

    page_w, page_h = A4[0] / mm, A4[1] / mm
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    c.setTitle("D1_final dome ArUco markers")
    c.setFont("Helvetica-Bold", 11)
    c.drawString(15 * mm, (page_h - 15) * mm,
                 f"D1_final dome markers: DICT_6X6_50, {a.marker_mm:g} mm black, {a.sticker_mm:g} mm cut")
    c.setFont("Helvetica", 7.5)
    c.drawString(15 * mm, (page_h - 20) * mm,
                 "Print on matte paper at 100% (no fit to page). Cut along the dotted outlines with a craft knife and steel ruler.")

    cols, pitch_x, pitch_y = 3, 62.0, 60.0
    x_start = (page_w - (cols - 1) * pitch_x - a.sticker_mm) / 2
    y_top = page_h - 70.0
    per_page = cols * 3
    for k, mid in enumerate(a.ids):
        page, idx = divmod(k, per_page)
        if idx == 0 and page > 0:
            scale_bar(c, 15, 18)
            c.showPage()
        r, col = divmod(idx, cols)
        x0 = x_start + col * pitch_x
        y0 = y_top - r * pitch_y
        draw_sticker(c, x0, y0, mid, a.marker_mm, a.sticker_mm, labels.get(mid, ""))

    scale_bar(c, 15, 18)
    c.save()
    print(f"wrote {out_pdf} with IDs {a.ids}")


if __name__ == "__main__":
    main()