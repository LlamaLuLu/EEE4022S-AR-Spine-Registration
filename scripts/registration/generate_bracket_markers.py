"""
Generate ArUco marker stickers for the V3 L-bracket.

Outputs:
    bracket_markers/
        marker_slot0_id0.png  ... individual marker images
        bracket_markers.pdf   ... A4 print sheet at exact 20mm marker size
        marker_map.json       ... ID-to-slot mapping for live_ar_overlay.py

Print bracket_markers.pdf at 100% scale (NOT "fit to page"). Measure the
included 50mm ruler after printing to confirm scale. Cut along the
dashed lines and stick each marker on the bracket at the labelled slot.

Usage:
    python generate_bracket_markers.py --outdir data/bracket_markers
    python generate_bracket_markers.py --outdir data/bracket_markers --start-id 10
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
DPI = 600
MM_PER_INCH = 25.4

SLOT_INFO = [
    {"slot": 0, "face": "A", "position": "top-left",     "face_local": "(15, 15)"},
    {"slot": 1, "face": "A", "position": "top-right",    "face_local": "(65, 15)"},
    {"slot": 2, "face": "A", "position": "bottom-left",  "face_local": "(15, 65)"},
    {"slot": 3, "face": "A", "position": "bottom-right", "face_local": "(65, 65)"},
    {"slot": 4, "face": "B", "position": "top-left",     "face_local": "(15, 15)"},
    {"slot": 5, "face": "B", "position": "top-right",    "face_local": "(65, 15)"},
    {"slot": 6, "face": "B", "position": "bottom-left",  "face_local": "(15, 65)"},
    {"slot": 7, "face": "B", "position": "bottom-right", "face_local": "(65, 65)"},
]


def mm(val):
    """Convert mm to pixels at print DPI."""
    return int(round(val * DPI / MM_PER_INCH))


def get_font(size_mm):
    """Load a font at the given mm height. Tries system fonts, then PIL default."""
    size_px = mm(size_mm)
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size_px)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:
        return ImageFont.load_default()


def make_marker(aruco_id, size_px):
    """Generate one ArUco marker image (no border, exact size)."""
    return cv2.aruco.generateImageMarker(ARUCO_DICT, aruco_id, size_px)


def compose_print_sheet(marker_size_mm, start_id):
    """
    A4 PDF with all 8 markers at true print size, arranged in a table
    with face headers, slot labels, dashed cut lines, and a ruler.
    """
    page_w, page_h = mm(210), mm(297)
    page = Image.new("L", (page_w, page_h), 255)
    draw = ImageDraw.Draw(page)

    font_title = get_font(5.0)
    font_heading = get_font(4.0)
    font_label = get_font(3.0)
    font_small = get_font(2.5)

    marker_px = mm(marker_size_mm)
    # Each cell: marker + white border around it + label below
    border_mm = 5.0          # white space around the marker inside the cell
    label_h_mm = 8.0         # space below marker for text
    cell_w_mm = marker_size_mm + 2 * border_mm
    cell_h_mm = marker_size_mm + 2 * border_mm + label_h_mm

    # Layout: 4 columns, 2 rows per face group, with face header above each row
    cols = 4
    left_margin_mm = (210 - cols * cell_w_mm) / 2
    top_mm = 25.0

    # Title
    draw.text((mm(left_margin_mm), mm(10)),
              "V3 L-bracket ArUco markers (DICT_6X6_50)",
              fill=0, font=font_title)
    draw.text((mm(left_margin_mm), mm(17)),
              f"Marker size: {marker_size_mm}mm  |  Print at 100% scale  |  Cut along dashed lines",
              fill=0, font=font_small)

    y_cursor_mm = top_mm

    for face_label, face_slots in [("Face A (caudal wall, Z = -95)", [0, 1, 2, 3]),
                                    ("Face B (lateral wall, X = +30)", [4, 5, 6, 7])]:
        # Face header
        draw.text((mm(left_margin_mm), mm(y_cursor_mm)),
                  face_label, fill=0, font=font_heading)
        y_cursor_mm += 7.0

        for i, slot_idx in enumerate(face_slots):
            col = i % cols
            row = i // cols

            info = SLOT_INFO[slot_idx]
            aruco_id = start_id + slot_idx

            x_mm = left_margin_mm + col * cell_w_mm
            y_mm = y_cursor_mm + row * cell_h_mm

            # Dashed cut rectangle around the cell
            x0, y0 = mm(x_mm), mm(y_mm)
            x1 = mm(x_mm + cell_w_mm)
            y1 = mm(y_mm + cell_h_mm)
            dash_len = mm(1.5)
            gap_len = mm(1.0)
            for edge_pts in [
                [(x0, y0, x1, y0)],     # top
                [(x0, y1, x1, y1)],     # bottom
                [(x0, y0, x0, y1)],     # left
                [(x1, y0, x1, y1)],     # right
            ]:
                ex0, ey0, ex1, ey1 = edge_pts[0]
                _draw_dashed_line(draw, ex0, ey0, ex1, ey1, dash_len, gap_len)

            # Marker image (exact print size, centered in cell)
            marker_img = make_marker(aruco_id, marker_px)
            mx = mm(x_mm + border_mm)
            my = mm(y_mm + border_mm)
            page.paste(Image.fromarray(marker_img), (mx, my))

            # Label below marker
            label_y = mm(y_mm + border_mm + marker_size_mm + 2)
            draw.text((mm(x_mm + border_mm), label_y),
                      f"ID {aruco_id}", fill=0, font=font_heading)
            draw.text((mm(x_mm + border_mm), label_y + mm(4)),
                      f"Slot {slot_idx}: {info['position']}",
                      fill=0, font=font_label)

        # Advance past this face's row
        rows_in_face = (len(face_slots) + cols - 1) // cols
        y_cursor_mm += rows_in_face * cell_h_mm + 6.0

    # 50mm ruler near the bottom
    _draw_ruler(draw, page_h, font_small, font_label)

    return page


def _draw_dashed_line(draw, x0, y0, x1, y1, dash, gap):
    """Draw a dashed line between two points."""
    dx, dy = x1 - x0, y1 - y0
    length = (dx**2 + dy**2) ** 0.5
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    pos = 0
    stroke_w = max(1, mm(0.15))
    while pos < length:
        end = min(pos + dash, length)
        draw.line([(int(x0 + ux * pos), int(y0 + uy * pos)),
                   (int(x0 + ux * end), int(y0 + uy * end))],
                  fill=100, width=stroke_w)
        pos = end + gap


def _draw_ruler(draw, page_h_px, font_small, font_label):
    """50mm verification ruler near the bottom of the page."""
    ruler_y = page_h_px - mm(40)
    ruler_x = mm(80)  # centered-ish on A4

    # Baseline
    draw.line([(ruler_x, ruler_y), (ruler_x + mm(50), ruler_y)],
              fill=0, width=mm(0.2))

    for tick_mm in range(0, 51):
        x = ruler_x + mm(tick_mm)
        if tick_mm % 10 == 0:
            tick_h = mm(4)
        elif tick_mm % 5 == 0:
            tick_h = mm(2.5)
        else:
            tick_h = mm(1)
        draw.line([(x, ruler_y), (x, ruler_y + tick_h)],
                  fill=0, width=mm(0.12))
        if tick_mm % 10 == 0:
            draw.text((x + mm(0.3), ruler_y + mm(5)),
                      str(tick_mm), fill=0, font=font_small)

    draw.text((ruler_x, ruler_y - mm(6)),
              "50mm ruler (measure after printing to verify scale)",
              fill=0, font=font_label)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ArUco stickers + mapping for V3 L-bracket")
    parser.add_argument("--outdir", default="data/bracket_markers")
    parser.add_argument("--start-id", type=int, default=0,
                        help="First ArUco dictionary ID to assign (default 0)")
    parser.add_argument("--marker-size-mm", type=float, default=20.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    marker_px = mm(args.marker_size_mm)
    print(f"Marker size: {args.marker_size_mm}mm = {marker_px}px at {DPI} DPI")

    mapping = {"marker_size_mm": args.marker_size_mm,
               "dictionary": "DICT_6X6_50", "slots": []}

    for slot in SLOT_INFO:
        aruco_id = args.start_id + slot["slot"]

        # Individual PNG (with white border for standalone use)
        raw = make_marker(aruco_id, marker_px)
        border = marker_px // 10
        bordered = cv2.copyMakeBorder(raw, border, border, border, border,
                                      cv2.BORDER_CONSTANT, value=255)
        filename = f"marker_slot{slot['slot']}_id{aruco_id}.png"
        cv2.imwrite(str(outdir / filename), bordered)

        label = f"Slot {slot['slot']}: Face {slot['face']} {slot['position']} (ID {aruco_id})"
        print(f"  {label}")

        mapping["slots"].append({
            "slot_index": slot["slot"],
            "aruco_id": aruco_id,
            "face": slot["face"],
            "position": slot["position"],
            "face_local_hv": slot["face_local"],
        })

    # A4 print-ready PDF
    pdf_sheet = compose_print_sheet(args.marker_size_mm, args.start_id)
    pdf_path = outdir / "bracket_markers.pdf"
    pdf_sheet.save(str(pdf_path), resolution=DPI)
    print(f"\nPrint-ready PDF: {pdf_path}")

    # Mapping JSON
    map_path = outdir / "marker_map.json"
    with open(map_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Marker map: {map_path}")
    print(f"\nNext steps:")
    print(f"  1. Print {pdf_path.name} at 100% scale (NOT 'fit to page')")
    print(f"  2. Measure the 50mm ruler to confirm scale")
    print(f"  3. Cut along dashed lines, stick on bracket at labelled positions")
    print(f"  4. Pass --marker-map {map_path} to live_ar_overlay.py")


if __name__ == "__main__":
    main()