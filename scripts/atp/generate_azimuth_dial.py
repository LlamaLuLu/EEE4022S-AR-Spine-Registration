"""
Printable azimuth dial for ATP-3.2 (azimuth sweep).

A4 page with a 190 mm circle, major ticks and labels every 15 deg, minor
ticks every 5 deg, a base plate outline (110 x 170 mm) and a 50 mm scale
bar for checking print scale.

Orientation matches the HUD azimuth convention (phantom seen from above):
  0 deg   = caudal end (bracket side)
  90 deg  = patient left (+X, Face B side)
  180 deg = cranial end (L1)
  270 deg = patient right
Angles increase clockwise when viewed from above.

Place the base plate on the outline, keep the camera fixed, and rotate
the card. The tick pointing at the camera is the current azimuth.

Usage (from repo root):
    python scripts/atp/generate_azimuth_dial.py --out data/atp/azimuth_dial.pdf
Print at 100% (NOT "fit to page") and check the 50 mm bar.
"""

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DPI = 300
MM_PER_INCH = 25.4
PAGE_W_MM, PAGE_H_MM = 210, 297
DIAMETER_MM = 190
# BASE_v2.stl: plate top X -50.2..50.2, Z -83.6..79.4; bracket seat X 35.2..60.1, Z -93.5..-23.7
PLATE_X = (-50.2, 50.2)
PLATE_Z = (-83.6, 79.4)
SEAT_X = (35.2, 60.1)
SEAT_Z = (-93.5, -23.7)


def mm(v):
    return int(round(v * DPI / MM_PER_INCH))


def font(size_mm, bold=False):
    px = mm(size_mm)
    names = (["/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
              else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, px)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def centred_text(draw, xy, text, fnt, fill=0, backing=False):
    l, t, r, b = draw.textbbox((0, 0), text, font=fnt)
    x, y = xy[0] - (r - l) / 2 - l, xy[1] - (b - t) / 2 - t
    if backing:
        pad = mm(0.8)
        draw.rectangle([x + l - pad, y + t - pad, x + r + pad, y + b + pad], fill=255)
    draw.text((x, y), text, font=fnt, fill=fill)


def dashed_line(draw, p0, p1, dash, gap, width, fill=0):
    x0, y0 = p0
    x1, y1 = p1
    L = math.hypot(x1 - x0, y1 - y0)
    if L == 0:
        return
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    s = 0.0
    while s < L:
        e = min(s + dash, L)
        draw.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)], fill=fill, width=width)
        s = e + gap


def page_point(cx, cy, r_mm, az_deg):
    """Azimuth 0 at bottom (caudal), increasing clockwise seen from above."""
    a = math.radians(az_deg)
    # bottom = +y on page; clockwise from bottom goes towards page-left
    x = cx - mm(r_mm) * math.sin(a)
    y = cy + mm(r_mm) * math.cos(a)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/atp/azimuth_dial.pdf")
    args = ap.parse_args()

    W, H = mm(PAGE_W_MM), mm(PAGE_H_MM)
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)

    cx, cy = W / 2, mm(30 + DIAMETER_MM / 2)
    R = DIAMETER_MM / 2

    # Title
    centred_text(d, (W / 2, mm(12)), "ATP-3.2 azimuth dial", font(6, True))
    centred_text(d, (W / 2, mm(20)),
                 "Print at 100%. Place base plate on outline. Rotate the card, not the phantom.",
                 font(3))

    # Circles
    for r_mm, w in [(R, 0.5), (R - 12, 0.25)]:
        r = mm(r_mm)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=mm(w))

    # Radial guide lines every 45 deg (light, dashed)
    for az in range(0, 360, 45):
        dashed_line(d, page_point(cx, cy, 6, az), page_point(cx, cy, R - 22, az),
                    mm(3), mm(3), mm(0.15), fill=150)

    # Base plate and bracket-seat outlines (BASE_v2). Page-left = patient left (+X), page-up = cranial (+Z).
    def pg(X, Z):
        return (cx - mm(X), cy - mm(Z))
    def dashed_rect(xr, zr):
        c = [pg(xr[1], zr[1]), pg(xr[0], zr[1]), pg(xr[0], zr[0]), pg(xr[1], zr[0])]
        for i in range(4):
            dashed_line(d, c[i], c[(i + 1) % 4], mm(4), mm(2), mm(0.35))
    dashed_rect(PLATE_X, PLATE_Z)
    dashed_rect(SEAT_X, SEAT_Z)
    hw = mm((PLATE_X[1] - PLATE_X[0]) / 2)
    hl_top, hl_bot = mm(PLATE_Z[1]), mm(-PLATE_Z[0])

    # Ticks and labels
    lab_font = font(4.2, True)
    for az in range(0, 360, 5):
        if az % 15 == 0:
            r_in, w = R - 10, 0.45
        else:
            r_in, w = R - 4, 0.2
        d.line([page_point(cx, cy, R, az), page_point(cx, cy, r_in, az)], fill=0, width=mm(w))
        if az % 15 == 0:
            centred_text(d, page_point(cx, cy, R - 16, az), f"{az}\u00b0", lab_font, backing=True)

    # Centre crosshair (phantom origin)
    c = mm(5)
    d.line([(cx - c, cy), (cx + c, cy)], fill=0, width=mm(0.3))
    d.line([(cx, cy - c), (cx, cy + c)], fill=0, width=mm(0.3))
    d.ellipse([cx - mm(1.5), cy - mm(1.5), cx + mm(1.5), cy + mm(1.5)], outline=0, width=mm(0.25))

    # Anatomical labels inside the plate outline
    small = font(3.2, True)
    centred_text(d, (cx, cy - hl_top + mm(24)), "CRANIAL (L1)", small, backing=True)
    centred_text(d, (cx, cy + hl_bot - mm(24)), "CAUDAL (bracket)", small, backing=True)
    centred_text(d, (cx - hw + mm(10), cy), "LEFT", small, backing=True)
    centred_text(d, (cx + hw - mm(11), cy), "RIGHT", small, backing=True)

    # 50 mm scale bar
    y = mm(30 + DIAMETER_MM + 22)
    x0 = W / 2 - mm(25)
    d.line([(x0, y), (x0 + mm(50), y)], fill=0, width=mm(0.3))
    for i in range(0, 51, 5):
        th = mm(4) if i % 10 == 0 else mm(2)
        d.line([(x0 + mm(i), y), (x0 + mm(i), y - th)], fill=0, width=mm(0.2))
    centred_text(d, (W / 2, y + mm(6)), "50 mm (measure to confirm print scale)", font(3))

    centred_text(d, (W / 2, y + mm(16)),
                 "Azimuth = tick pointing at the camera. 0\u00b0 caudal, 90\u00b0 patient left (viewed from above).",
                 font(3))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, resolution=DPI)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()