"""
Generate a printable ArUco board for FLE measurement.

Layout: 2x2 grid of 20mm ArUco markers (dictionary 6X6_50), 40mm centre-to-centre.
Total board: 100mm x 100mm printed. Includes a 100mm ruler for print-scale
verification.

Output: fle_board.pdf. Print at 100% scale (NOT "fit to page"). Measure the
included ruler after printing to confirm true 100mm.

Usage:
  python generate_fle_board.py
  python generate_fle_board.py --marker-size-mm 25 --outfile my_board.pdf
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


DPI = 600            # print at 600 DPI for clean edges
MM_PER_INCH = 25.4


def get_font(size_px: int) -> ImageFont.ImageFont:
    """Try common system fonts, fall back to PIL's built-in with size scaling."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",              # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",     # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "C:/Windows/Fonts/arial.ttf",                       # Windows
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size_px)
        except (OSError, IOError):
            continue
    # PIL 10+ supports sized default; older PIL returns tiny bitmap.
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:
        return ImageFont.load_default()


def mm_to_px(mm: float, dpi: int = DPI) -> int:
    return int(round(mm * dpi / MM_PER_INCH))


def make_aruco_marker(marker_id: int, size_px: int, dictionary) -> np.ndarray:
    """Generate a single ArUco marker at the requested pixel size."""
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    return img


def compose_board(marker_size_mm: float, spacing_mm: float,
                  ids: list, page_size_mm: tuple = (210, 297)) -> Image.Image:
    """Lay out markers on an A4 page with a mm ruler for scale verification."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

    page_w_px = mm_to_px(page_size_mm[0])
    page_h_px = mm_to_px(page_size_mm[1])
    page = Image.new("L", (page_w_px, page_h_px), color=255)   # white A4

    # 2x2 grid, centred horizontally, near the top of the page.
    board_w_mm = 2 * marker_size_mm + spacing_mm
    board_h_mm = board_w_mm
    board_left_mm = (page_size_mm[0] - board_w_mm) / 2
    board_top_mm = 30   # 30mm margin from top

    marker_size_px = mm_to_px(marker_size_mm)
    spacing_px = mm_to_px(spacing_mm)

    for i, mid in enumerate(ids):
        row, col = i // 2, i % 2
        marker = make_aruco_marker(mid, marker_size_px, dictionary)
        x_mm = board_left_mm + col * (marker_size_mm + spacing_mm)
        y_mm = board_top_mm + row * (marker_size_mm + spacing_mm)
        x_px = mm_to_px(x_mm)
        y_px = mm_to_px(y_mm)
        page.paste(Image.fromarray(marker), (x_px, y_px))

    # 100mm ruler for print scale verification, near the bottom.
    draw = ImageDraw.Draw(page)
    ruler_font = get_font(mm_to_px(3.5))    # ~3.5mm tall text for ruler labels
    note_font = get_font(mm_to_px(4.0))     # ~4mm text for the descriptive lines

    ruler_y_mm = page_size_mm[1] - 60
    ruler_x_start_mm = (page_size_mm[0] - 100) / 2
    for mm in range(0, 101, 1):
        x_px = mm_to_px(ruler_x_start_mm + mm)
        y_px = mm_to_px(ruler_y_mm)
        tick_mm = 5 if mm % 10 == 0 else (3 if mm % 5 == 0 else 1)
        draw.line([(x_px, y_px), (x_px, y_px + mm_to_px(tick_mm))],
                  fill=0, width=mm_to_px(0.15))
        # Label every 10mm below the tick
        if mm % 10 == 0:
            draw.text((x_px + mm_to_px(0.5), y_px + mm_to_px(6)),
                      str(mm), fill=0, font=ruler_font)
    # baseline
    draw.line([(mm_to_px(ruler_x_start_mm), mm_to_px(ruler_y_mm)),
               (mm_to_px(ruler_x_start_mm + 100), mm_to_px(ruler_y_mm))],
              fill=0, width=mm_to_px(0.2))
    # label above ruler
    label_y_px = mm_to_px(ruler_y_mm - 8)
    draw.text((mm_to_px(ruler_x_start_mm), label_y_px),
              "100 mm ruler (measure this after printing to verify scale)",
              fill=0, font=note_font)

    # Print-scale note near markers
    note_y_px = mm_to_px(board_top_mm + board_h_mm + 8)
    draw.text((mm_to_px(board_left_mm), note_y_px),
              f"{marker_size_mm}mm markers, {spacing_mm}mm gap (dict 6X6_50, ids {ids})",
              fill=0, font=note_font)
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker-size-mm", type=float, default=20.0)
    parser.add_argument("--spacing-mm", type=float, default=40.0,
                        help="gap between adjacent markers")
    parser.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="four ArUco IDs to use (must be < 50 for 6X6_50)")
    parser.add_argument("--outfile", type=str, default="fle_board.pdf")
    args = parser.parse_args()

    if len(args.ids) != 4:
        raise SystemExit("Provide exactly 4 --ids for the 2x2 layout")

    page = compose_board(args.marker_size_mm, args.spacing_mm, args.ids)
    out = Path(args.outfile)
    page.save(out, resolution=DPI)   # PIL saves as PDF or PNG based on extension
    print(f"Saved {out}")
    print(f"Print at 100% scale (NOT 'fit to page').")
    print(f"After printing, measure the 100mm ruler to confirm scale.")


if __name__ == "__main__":
    main()