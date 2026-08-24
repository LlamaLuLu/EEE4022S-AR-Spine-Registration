"""
Generates printable calibration materials as both PNG and PDF:
  1. charuco_board.png/.pdf   - ChArUco board for camera intrinsics calibration
  2. test_markers.png/.pdf    - a small sheet of standalone ArUco markers for
                                 sanity-testing detection + pose overlay

The PDFs are sized so printing at 100% / actual scale (NOT "Fit to page")
reproduces the true physical dimensions. Each sheet also has a printed
50mm reference bar, measure it with a ruler after printing to confirm the
scale actually came out right, printer drivers don't always honor 100%
exactly. If it's off, update SQUARE_LENGTH_M / MARKER_LENGTH_M below to
the measured value rather than the nominal one.
"""
import argparse
import os
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdfcanvas

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

# Board sized to fit A4 for a first-pass test. Vally (2020) found 6x8
# squares at 50mm gave the best reprojection error, but that board is
# 300x400mm and won't fit A4. Use this smaller version to validate the
# pipeline now, then reprint at full size later (poster print, or tiled
# across sheets) once you're doing the calibration run you'll report.
SQUARES_X, SQUARES_Y = 5, 7
SQUARE_LENGTH_M = 0.030   # 30mm nominal, re-measure after printing
MARKER_LENGTH_M = 0.015   # 15mm nominal

DPI = 300
PX_PER_M = DPI / 0.0254

SCALE_BAR_LENGTH_M = 0.050
SCALE_BAR_STRIP_M = 0.030  # height of the margin strip holding the bar + label


def add_scale_bar(img):
    """Append a labeled 50mm reference bar below the image, printed as part
    of the same sheet, so print scale can be checked with a ruler."""
    strip_h = int(SCALE_BAR_STRIP_M * PX_PER_M)
    bar_len_px = int(SCALE_BAR_LENGTH_M * PX_PER_M)
    h, w = img.shape[:2]
    strip = np.full((strip_h, w), 255, dtype=np.uint8)

    y_line = strip_h // 3
    x0, x1 = 20, 20 + bar_len_px
    tick_h = 10
    cv2.line(strip, (x0, y_line), (x1, y_line), 0, 3)
    cv2.line(strip, (x0, y_line - tick_h), (x0, y_line + tick_h), 0, 3)
    cv2.line(strip, (x1, y_line - tick_h), (x1, y_line + tick_h), 0, 3)
    cv2.putText(strip, "50mm - measure after printing", (x0, y_line + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    return np.vstack([img, strip])


def save_pdf(png_path, pdf_path, width_m, height_m):
    """Place the PNG on an A4 page at its true physical size (points, not
    pixels), rather than letting the page equal the image, so it prints
    correctly on standard paper at 100% scale."""
    c = pdfcanvas.Canvas(pdf_path, pagesize=A4)
    page_w, page_h = A4
    img_w_pt = width_m * 1000 * mm
    img_h_pt = height_m * 1000 * mm
    x = (page_w - img_w_pt) / 2
    y = page_h - img_h_pt - 20 * mm
    c.drawImage(png_path, x, y, width=img_w_pt, height=img_h_pt)
    c.showPage()
    c.save()
    print(f"Saved {pdf_path}, print at 100% / actual size on A4, not fit-to-page")


def generate_charuco_board(outdir):
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, DICTIONARY
    )
    width_px = int(SQUARES_X * SQUARE_LENGTH_M * PX_PER_M)
    height_px = int(SQUARES_Y * SQUARE_LENGTH_M * PX_PER_M)
    img = board.generateImage((width_px, height_px), marginSize=20, borderBits=1)
    img = add_scale_bar(img)

    png_path = os.path.join(outdir, "charuco_board.png")
    cv2.imwrite(png_path, img)
    print(f"Saved {png_path} at {img.shape[1]}x{img.shape[0]}px "
          f"({SQUARES_X}x{SQUARES_Y} squares, {SQUARE_LENGTH_M*1000:.0f}mm nominal, {DPI} DPI)")

    width_m = img.shape[1] / PX_PER_M
    height_m = img.shape[0] / PX_PER_M
    pdf_path = os.path.join(outdir, "charuco_board.pdf")
    save_pdf(png_path, pdf_path, width_m, height_m)


def generate_test_markers(outdir):
    marker_side_m = 0.05  # 5cm, larger and easier to detect than the board's embedded markers
    marker_side_px = int(marker_side_m * PX_PER_M)
    gap_px = int(0.02 * PX_PER_M)
    # A 5x7 ChArUco board uses roughly the first 18 IDs (0-17) internally,
    # one marker per alternating square. Start test marker IDs well above
    # that range so they never collide with the board's own markers if
    # both end up in frame together.
    ids = [20, 21, 22, 23]
    cols, rows = 2, 2

    sheet_w = cols * marker_side_px + (cols + 1) * gap_px
    sheet_h = rows * marker_side_px + (rows + 1) * gap_px
    sheet = np.full((sheet_h, sheet_w), 255, dtype=np.uint8)

    for i, marker_id in enumerate(ids):
        marker_img = cv2.aruco.generateImageMarker(DICTIONARY, marker_id, marker_side_px)
        r, c = divmod(i, cols)
        y = gap_px + r * (marker_side_px + gap_px)
        x = gap_px + c * (marker_side_px + gap_px)
        sheet[y:y + marker_side_px, x:x + marker_side_px] = marker_img

    sheet = add_scale_bar(sheet)

    png_path = os.path.join(outdir, "test_markers.png")
    cv2.imwrite(png_path, sheet)
    print(f"Saved {png_path}, {len(ids)} markers at {marker_side_m*1000:.0f}mm nominal, ids {ids}")

    width_m = sheet.shape[1] / PX_PER_M
    height_m = sheet.shape[0] / PX_PER_M
    pdf_path = os.path.join(outdir, "test_markers.pdf")
    save_pdf(png_path, pdf_path, width_m, height_m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    generate_charuco_board(args.outdir)
    generate_test_markers(args.outdir)