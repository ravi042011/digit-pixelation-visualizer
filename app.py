"""Digit Pixelation Visualizer - Streamlit UI (layout, state, wiring only).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from canvas_utils import canvas_to_rgba
from pixelation import TARGET_SIZE, process_digit, render_pixel_grid_with_values

CANVAS_SIZE = 280      # drawing canvas is 280 x 280 px
STROKE_WIDTH = 20      # white pen thickness (within the ~18-22px spec)
DISPLAY_SIZE = 320     # pixelated result is shown at 320 x 320 px
DIGITS = list(range(10))

# Ten fixed-width columns would overflow on most screens, so let the row scroll
# horizontally instead of squeezing the 280px canvases.
_CSS = """
<style>
div[data-testid="stHorizontalBlock"] { overflow-x: auto; flex-wrap: nowrap; }
div[data-testid="stColumn"], div[data-testid="column"] { min-width: 340px; }
.digit-label { text-align: center; font-size: 1.6rem; font-weight: 700; }
.arrow { text-align: center; font-size: 2rem; line-height: 1.1; margin-top: 0.4rem; }
.arrow-caption { text-align: center; font-size: 0.8rem; color: #888; margin-bottom: 0.4rem; }
</style>
"""


def init_state() -> None:
    """Create the counter used to build canvas keys (bumping it clears canvases)."""
    if "canvas_version" not in st.session_state:
        st.session_state["canvas_version"] = 0


def clear_all() -> None:
    """Button callback: new canvas keys force Streamlit to build blank canvases."""
    st.session_state["canvas_version"] += 1


def render_digit_column(digit: int) -> None:
    """Draw one digit's column: label, canvas, arrow, pixelated grid."""
    st.markdown(f'<div class="digit-label">{digit}</div>', unsafe_allow_html=True)

    # 1. Drawing canvas (white stroke on black). The key includes the version
    #    so "Clear All" resets it; update_streamlit=True gives live updates.
    result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=STROKE_WIDTH,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        update_streamlit=True,
        key=f"canvas_{digit}_{st.session_state['canvas_version']}",
    )

    # 2. Down arrow + caption
    st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)
    st.markdown('<div class="arrow-caption">raw → pixelated</div>', unsafe_allow_html=True)

    # 3. Process (empty / not-yet-ready canvas -> blank 32x32 placeholder)
    rgba = canvas_to_rgba(result)
    if rgba is None:
        pixels = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    else:
        try:
            pixels = process_digit(rgba)
        except Exception:  # keep the other nine digits usable
            st.warning("Could not process this drawing. Try clearing and redrawing.")
            pixels = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    # 4. Upscaled grid with the 0-255 value printed in every cell.
    #    PNG output avoids JPEG blur so each block stays crisp.
    grid_image = render_pixel_grid_with_values(pixels, upscale_to=DISPLAY_SIZE)
    st.image(grid_image, width=DISPLAY_SIZE, output_format="PNG")


def main() -> None:
    st.set_page_config(page_title="Digit Pixelation Visualizer", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    init_state()

    st.title("Digit Pixelation Visualizer")
    st.caption(
        "Draw each digit (mouse or pen). Every drawing is centred, resized to "
        "32×32 and shown with its 0–255 grayscale value per pixel."
    )
    st.button("Clear All", on_click=clear_all)

    columns = st.columns(10)
    for digit, column in zip(DIGITS, columns):
        with column:
            render_digit_column(digit)


main()
