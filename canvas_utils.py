"""Helpers that turn streamlit-drawable-canvas output into numpy arrays.

Kept free of Streamlit imports: it only inspects the result object that the
canvas component returns (it exposes `.image_data`, an RGBA numpy array).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from pixelation import rgba_to_grayscale


def canvas_to_rgba(canvas_result: Any) -> np.ndarray | None:
    """Extract a uint8 RGBA (H, W, 4) array from a canvas result.

    Returns None if the canvas has not produced any image data yet
    (e.g. on the very first render) or the data has an unexpected shape.
    """
    image_data = getattr(canvas_result, "image_data", None)
    if image_data is None:
        return None
    arr = np.asarray(image_data)
    if arr.ndim != 3 or arr.shape[2] != 4:
        return None
    return np.clip(arr, 0, 255).astype(np.uint8)


def canvas_to_grayscale(canvas_result: Any) -> np.ndarray | None:
    """Convert a canvas result to a grayscale uint8 array (white ink on 0).

    Transparent canvas pixels are composited over black, matching the
    black-background / white-stroke canvas configuration.
    """
    rgba = canvas_to_rgba(canvas_result)
    return None if rgba is None else rgba_to_grayscale(rgba)
