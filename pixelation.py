"""Pure image-processing functions for the Digit Pixelation Visualizer.

No Streamlit imports live here, so every function can be unit-tested by
passing plain numpy arrays.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TARGET_SIZE: int = 32          # final pixel grid is 32 x 32
MARGIN_RATIO: float = 0.20     # total padding added around the digit (20%)

_PLACEHOLDER_FILL = (45, 45, 45)   # gray shown for a not-yet-drawn digit
_GRID_COLOR = (110, 110, 110)      # thin grid line colour


# --------------------------------------------------------------------------
# Processing pipeline
# --------------------------------------------------------------------------
def rgba_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGBA / RGB / grayscale array to a single-channel uint8 array.

    Canvas strokes are white on black. Transparent pixels are composited over
    black, so ink -> high intensity and background -> 0.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        rgb = arr[..., :3].astype(np.float32)
        # ITU-R BT.601 luminance weights
        gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        if arr.shape[2] == 4:
            # Composite over a black background using the alpha channel.
            gray *= arr[..., 3].astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def find_bounding_box(gray: np.ndarray, threshold: int = 0) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) of all pixels brighter than `threshold`.

    x1/y1 are exclusive so the box can be used directly for slicing.
    Returns None when nothing has been drawn.
    """
    ys, xs = np.nonzero(gray > threshold)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def center_in_square(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin_ratio: float = MARGIN_RATIO,
) -> np.ndarray:
    """Crop to the ink bounding box and centre it on a square black canvas.

    Centring is based on the bounding box of the actual ink (MNIST-style),
    not on the raw canvas, and the square keeps the digit's aspect ratio.
    """
    x0, y0, x1, y1 = bbox
    crop = gray[y0:y1, x0:x1]
    h, w = crop.shape
    side = max(h, w)
    # margin_ratio is the TOTAL extra space (e.g. 0.20 -> 10% on each side)
    padded_side = int(math.ceil(side * (1.0 + margin_ratio)))
    square = np.zeros((padded_side, padded_side), dtype=np.uint8)
    top = (padded_side - h) // 2
    left = (padded_side - w) // 2
    square[top:top + h, left:left + w] = crop
    return square


def process_digit(raw_rgba_array: np.ndarray) -> np.ndarray:
    """Turn a raw canvas image into a centred 32x32 uint8 grayscale array.

    Pipeline: grayscale -> bounding box -> centre in padded square -> resize.
    An empty drawing returns an all-zero 32x32 array.
    """
    # 1. Grayscale: white stroke -> high intensity, background -> 0
    gray = rgba_to_grayscale(raw_rgba_array)

    # 2. Bounding box of every drawn (non-zero) pixel; blank canvas -> zeros
    bbox = find_bounding_box(gray)
    if bbox is None:
        return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    # 3. Centre the ink in a square canvas with ~20% padding
    square = center_in_square(gray, bbox)

    # 4. Area-based downsampling gives proper anti-aliased intensities
    small = cv2.resize(square, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

    # 5. Guarantee dtype/shape contract
    return small.astype(np.uint8)


# --------------------------------------------------------------------------
# Rendering the numeric pixel grid
# --------------------------------------------------------------------------
def _make_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load a small scalable font, falling back to Pillow's built-in one."""
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _fit_font(max_width: float) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Pick the largest font (<= 9px) whose widest label ('255') fits a cell."""
    font = _make_font(4)
    for size in range(9, 3, -1):
        font = _make_font(size)
        left, _, right, _ = font.getbbox("255")
        if right - left <= max_width:
            break
    return font


def render_pixel_grid_with_values(pixel_array: np.ndarray, upscale_to: int = 320) -> Image.Image:
    """Upscale a pixel array (nearest-neighbour), draw a grid and cell values.

    - Nearest-neighbour keeps every cell a crisp block.
    - Text is white on dark cells and black on light cells for legibility.
    - An all-zero array is drawn on a gray placeholder background.
    """
    arr = np.asarray(pixel_array)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected a square 2-D array, got shape {arr.shape}")
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    n = arr.shape[0]

    # Upscale: each of the n x n cells becomes a crisp block
    if arr.any():
        base = Image.fromarray(arr, mode="L").resize((upscale_to, upscale_to), Image.NEAREST).convert("RGB")
    else:
        base = Image.new("RGB", (upscale_to, upscale_to), _PLACEHOLDER_FILL)
    draw = ImageDraw.Draw(base)
    cell = upscale_to / n

    # Thin grid lines between cells
    for i in range(1, n):
        pos = int(round(i * cell))
        draw.line([(pos, 0), (pos, upscale_to - 1)], fill=_GRID_COLOR, width=1)
        draw.line([(0, pos), (upscale_to - 1, pos)], fill=_GRID_COLOR, width=1)

    # Centered intensity value in each cell, contrasting with the cell colour
    font = _fit_font(cell - 1)
    for row in range(n):
        for col in range(n):
            value = int(arr[row, col])
            text_color = (0, 0, 0) if value >= 128 else (255, 255, 255)
            cx = (col + 0.5) * cell
            cy = (row + 0.5) * cell
            draw.text((cx, cy), str(value), fill=text_color, font=font, anchor="mm")
    return base
