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
  ID 11..16  ring markers at 30, 90, 150, 210, 270, 330 deg; TOP edge points INWARD as built
             (--ring-top outward for the original plan)
  ID 17      spare for the single-face fit test (not part of the dome)

Outputs (in <repo>/data/dome_markers/ by default):
  dome_markers_D1_final.pdf   stickers, cut outlines and a placement key
  marker_map.json             3D corners of every marker in the phantom frame (mm),
                              taken from DOME_CONFIGS["D1_final"] in
                              scripts/tre_analysis/design_variants_oriented.py, so the
                              live map and the simulator always use the same geometry

Usage (from any directory):
  python scripts/markers/generate_dome_markers.py
  python scripts/markers/generate_dome_markers.py --marker-mm 24.9       # measured black square
  python scripts/markers/generate_dome_markers.py --surface-offset-mm -0.15
  python scripts/markers/generate_dome_markers.py --top-offset-mm -0.3   # printed top face sits lower than CAD

Print at 100% / "Actual size" (turn off "fit to page"), on matte paper.
Then measure the 100 mm scale bar and one black square with calipers: use the
measured black-square size in the live marker map if it is not exactly 25.0 mm.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

REPO_ROOT = Path(__file__).resolve().parents[2]      # scripts/markers/<this file>
OUT_DIR = REPO_ROOT / "data" / "dome_markers"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "tre_analysis"))
from design_variants_oriented import build_dome      # noqa: E402

DOME_NAME = "D1_final"
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


# dome layout in phantom mm (LPS): centre, ring radius, ring angles (deg, from +X toward +Z)
DOME_CENTRE_XZ = (120.0, 0.0)
RING_R = 50.0
RING_IDS = {11: 30, 12: 90, 13: 150, 14: 210, 15: 270, 16: 330}
TOP_ID = 10
PLATE_X, PLATE_Z = (-50.2, 50.2), (-83.6, 79.4)


def draw_key(c, bx, by, bw, bh, marker_mm, ring_top="inward"):
    """Top-view key (looking down on the phantom, cranial up, patient left on the left):
    base plate, dome ring and each marker's ID, with its TOP edge drawn thick."""
    s = 0.25                                         # page mm per phantom mm
    x_max, z_min = 190.0, -85.0
    ox, oy = bx + 8, by + 13

    def P(X, Z):                                     # phantom (X, Z) -> page mm; +X drawn to the left
        return ox + (x_max - X) * s, oy + (Z - z_min) * s

    c.setStrokeColorRGB(0.6, 0.6, 0.6)
    c.setLineWidth(0.3)
    c.rect(bx * mm, by * mm, bw * mm, bh * mm, stroke=1, fill=0)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString((bx + 2) * mm, (by + bh - 5) * mm, "Key: top view, looking down")
    c.setFont("Helvetica", 6)
    c.drawString((bx + 2) * mm, (by + bh - 9) * mm, "cranial up, patient left on the left")

    # base plate with level labels
    x0, y0 = P(PLATE_X[1], PLATE_Z[0])
    x1, y1 = P(PLATE_X[0], PLATE_Z[1])
    c.setFillColorRGB(0.88, 0.88, 0.88)
    c.setStrokeColorRGB(0.5, 0.5, 0.5)
    c.rect(x0 * mm, y0 * mm, (x1 - x0) * mm, (y1 - y0) * mm, stroke=1, fill=1)
    c.setFillColorRGB(0.3, 0.3, 0.3)
    c.setFont("Helvetica", 5.5)
    xm = (x0 + x1) / 2
    c.drawCentredString(xm * mm, (y1 - 3.5) * mm, "L1")
    c.drawCentredString(xm * mm, ((y0 + y1) / 2 - 1) * mm, "base +")
    c.drawCentredString(xm * mm, ((y0 + y1) / 2 - 3.5) * mm, "spine")
    c.drawCentredString(xm * mm, (y0 + 1.5) * mm, "L5")

    # ring outline and arms
    cxp, cyp = P(*DOME_CENTRE_XZ)
    c.setStrokeColorRGB(0.6, 0.6, 0.6)
    c.setDash(1, 1)
    c.circle(cxp * mm, cyp * mm, RING_R * s * mm, stroke=1, fill=0)
    c.setDash()

    h = marker_mm / 2 * s

    def square(cx, cy, r_hat, t_hat, mid, top_edge):
        corners = [(cx + a * r_hat[0] * h + b * t_hat[0] * h, cy + a * r_hat[1] * h + b * t_hat[1] * h)
                   for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        path = c.beginPath()
        path.moveTo(corners[0][0] * mm, corners[0][1] * mm)
        for px, py in corners[1:]:
            path.lineTo(px * mm, py * mm)
        path.close()
        c.setFillColorRGB(1, 1, 1)
        c.setStrokeColorRGB(0.2, 0.2, 0.2)
        c.setLineWidth(0.4)
        c.drawPath(path, stroke=1, fill=1)
        (ax, ay), (bx2, by2) = top_edge(corners)
        c.setLineWidth(1.6)
        c.setStrokeColorRGB(0, 0, 0)
        c.line(ax * mm, ay * mm, bx2 * mm, by2 * mm)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 5.5)
        c.drawCentredString(cx * mm, (cy - 0.9) * mm, str(mid))

    for mid, ang in RING_IDS.items():
        phi = np.radians(ang)
        X = DOME_CENTRE_XZ[0] + RING_R * np.cos(phi)
        Z = DOME_CENTRE_XZ[1] + RING_R * np.sin(phi)
        px, py = P(X, Z)
        r_hat = (-np.cos(phi), np.sin(phi))          # outward on the page (X is mirrored)
        t_hat = (-r_hat[1], r_hat[0])
        c.setStrokeColorRGB(0.6, 0.6, 0.6)
        c.setLineWidth(0.6)
        c.line(cxp * mm, cyp * mm, (px - r_hat[0] * h) * mm, (py - r_hat[1] * h) * mm)
        edge = (lambda k: (k[0], k[1])) if ring_top == "outward" else (lambda k: (k[2], k[3]))
        square(px, py, r_hat, t_hat, mid, edge)                        # outer edge (k0,k1) or inner edge (k2,k3)

    # top marker, axis aligned, TOP edge cranial
    square(cxp, cyp, (0.0, 1.0), (1.0, 0.0), TOP_ID, lambda k: (k[0], k[1]))

    c.setFont("Helvetica", 5.5)
    c.setFillColorRGB(0, 0, 0)
    c.drawString((bx + 2) * mm, (by + 7) * mm, f"thick edge = marker TOP edge ({ring_top} on the ring, cranial on ID 10)")
    c.drawString((bx + 2) * mm, (by + 3.5) * mm, "ID 10 sits raised in the centre; 17 is the spare")


def marker_corners(centre, R, size_mm):
    """TL, TR, BR, BL corners (ArUco detector order) of one marker, phantom mm.
    Same convention as monte_carlo_tre.corners_from_oriented_marker."""
    h = size_mm / 2.0
    local = np.array([[-h, +h, 0], [+h, +h, 0], [+h, -h, 0], [-h, -h, 0]])
    return np.asarray(centre, float) + local @ np.asarray(R, float).T


def write_marker_map(path, marker_mm, surface_offset_mm, ring_top="inward", top_offset_mm=0.0):
    """marker_map.json with explicit 3D corners for every dome marker."""
    dome = build_dome(DOME_NAME)
    ids = [TOP_ID] + list(RING_IDS)
    assert len(dome) == len(ids), f"{DOME_NAME} has {len(dome)} markers, expected {len(ids)}"

    # the PDF key is drawn from DOME_CENTRE_XZ / RING_R / RING_IDS; check it matches the config
    for (c, _), mid in zip(dome[1:], RING_IDS):
        phi = np.radians(RING_IDS[mid])
        exp = (DOME_CENTRE_XZ[0] + RING_R * np.cos(phi), DOME_CENTRE_XZ[1] + RING_R * np.sin(phi))
        assert np.allclose((c[0], c[2]), exp, atol=0.1), \
            f"ID {mid}: config centre {c[[0, 2]]} does not match key {exp}; update the key constants"

    slots = []
    for k, (mid, (c, R)) in enumerate(zip(ids, dome)):
        R = np.asarray(R, float)
        assert np.isclose(np.linalg.det(R), 1.0), f"ID {mid}: frame is not a proper rotation"
        off = surface_offset_mm + (top_offset_mm if mid == TOP_ID else 0.0)
        centre = np.asarray(c, float) + off * R[:, 2]
        corners = marker_corners(centre, R, marker_mm)
        if mid != TOP_ID and ring_top == "inward":
            corners = corners[[2, 3, 0, 1]]     # sticker turned 180 deg: TL,TR,BR,BL -> BR,BL,TL,TR
        slots.append({
            "slot_index": k,
            "aruco_id": mid,
            "face": "dome_top" if mid == TOP_ID else "dome_ring",
            "position": "top, TOP edge cranial" if mid == TOP_ID else f"ring {RING_IDS[mid]} deg, TOP edge {ring_top}",
            "centre_mm": [round(float(v), 3) for v in centre],
            "normal": [round(float(v), 5) for v in R[:, 2]],
            "corners_mm": [[round(float(v), 3) for v in p] for p in corners],
        })
    data = {
        "marker_size_mm": marker_mm,
        "dictionary": "DICT_6X6_50",
        "design": DOME_NAME,
        "ring_top": ring_top,
        "min_markers_pnp": 5,
        "frame": "phantom LPS, origin at plate top centre, mm (+X patient left, +Y up, +Z cranial)",
        "corner_order": "TL, TR, BR, BL (OpenCV ArUco detector order)",
        "surface_offset_mm": surface_offset_mm,
        "top_offset_mm": top_offset_mm,
        "slots": slots,
    }
    path.write_text(json.dumps(data, indent=2))
    return slots


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
    ap.add_argument("--ring-top", choices=["outward", "inward"], default="inward",
                    help="which way the ring stickers' TOP edge points as built (default: inward, as stuck)")
    ap.add_argument("--marker-mm", type=float, default=25.0, help="black square edge, including its black border")
    ap.add_argument("--sticker-mm", type=float, default=32.0, help="cut size, including the white margin")
    ap.add_argument("--outdir", default=str(OUT_DIR), help="output folder (default: <repo>/data/dome_markers)")
    ap.add_argument("--name", default="dome_markers_D1_final.pdf", help="PDF file name")
    ap.add_argument("--map-name", default="marker_map.json", help="marker map file name")
    ap.add_argument("--surface-offset-mm", type=float, default=3.3,
                    help="shift every marker along its normal, e.g. -0.15 if the printed sticker "
                         "surface sits 0.15 mm below the design surface")
    ap.add_argument("--top-offset-mm", type=float, default=0.0,
                    help="extra shift of the top marker (ID 10) along its normal; negative = lower, "
                         "e.g. -0.3 if the printed top face sits 0.3 mm below the CAD")
    a = ap.parse_args()

    assert (a.sticker_mm - a.marker_mm) / 2 >= a.marker_mm / CELLS, \
        "white margin is thinner than one marker cell (quiet zone too small)"

    labels = {10: "top marker, TOP = cranial", 17: "spare, single-face test"}
    for i, ang in zip(range(11, 17), (30, 90, 150, 210, 270, 330)):
        labels[i] = f"ring {ang} deg, TOP = {a.ring_top}"

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
    draw_key(c, page_w - 15 - 80, 22, 80, 74, a.marker_mm, a.ring_top)
    c.save()
    print(f"wrote {out_pdf} with IDs {a.ids}")

    out_map = outdir / a.map_name
    slots = write_marker_map(out_map, a.marker_mm, a.surface_offset_mm, a.ring_top, a.top_offset_mm)
    print(f"wrote {out_map} ({DOME_NAME}, {len(slots)} markers, {a.marker_mm:g} mm)")
    for sl in slots:
        print(f"  ID {sl['aruco_id']:>2}  centre {sl['centre_mm']}  {sl['position']}")


if __name__ == "__main__":
    main()