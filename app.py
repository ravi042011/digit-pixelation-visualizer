"""Digit Pixelation Visualizer.

Run with: streamlit run app.py
Draw any digit in the shared pad; the preview animates the processing pipeline.
"""
from __future__ import annotations

import base64
from io import BytesIO
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
from streamlit_drawable_canvas import st_canvas

from canvas_utils import canvas_to_rgba
from pixelation import (
    TARGET_SIZE,
    center_in_square,
    find_bounding_box,
    process_digit,
    rgba_to_grayscale,
)

CANVAS_SIZE = 280
STROKE_WIDTH = 20
LOGGER = logging.getLogger(__name__)
PREVIEW_TEMPLATE = Path(__file__).with_name("pixel_preview.html")

_CSS = """
<style>
.stApp { background: #f7f8fc; color: #20233a; }
.block-container { max-width: 1000px; padding-top: 2.5rem; padding-bottom: 2rem; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stVerticalBlockBorderWrapper"]:has(> div >
    [data-testid="stVerticalBlock"] > [data-testid="element-container"] .panel-heading) {
    background: white; border: 1px solid #e5e7f0; border-radius: 20px;
    box-shadow: 0 8px 32px #26325b06; padding: 1.25rem;
}
[data-testid="stHorizontalBlock"] { gap: 1.5rem; }
[data-testid="stIFrame"] { display: block; margin: auto; max-width: 100%; }
iframe[title="streamlit_drawable_canvas.st_canvas"] { width: 280px; border-radius: 12px; }
.eyebrow { color: #7060d7; font-size: 11px; font-weight: 750;
    letter-spacing: 2px; margin-bottom: 14px; }
h1 { color: #20233a !important; font-size: clamp(2rem, 5vw, 3rem) !important;
    letter-spacing: -1.8px; padding-top: 0 !important; padding-bottom: .5rem !important; }
.intro { color: #74788f; font-size: 16px; max-width: 640px; margin-bottom: 28px; }
.panel-heading { display: flex; align-items: center; gap: 10px;
    font-weight: 700; font-size: 18px; margin-bottom: 4px; }
.step { display: inline-grid; place-items: center; width: 27px; height: 27px;
    border-radius: 9px; background: #efedff; color: #7562dc; font-size: 12px; }
.panel-caption { color: #84889d; font-size: 13px; margin-bottom: 16px; }
.pad-hint { color: #85899d; font-size: 12px; text-align: center; margin: 8px 0 16px; }
[data-testid="stButton"] button { border: 1px solid #e1dff2; border-radius: 10px;
    color: #6959c4; background: #f7f6ff; font-weight: 600; }
[data-testid="stButton"] button:hover { border-color: #9785e8; color: #5844b4; }
.how-it-works { margin-top: 28px; }
.how-it-works h2 { color: #30334b; font-size: 23px; letter-spacing: -.5px; }
.how-intro { color: #74788f; font-size: 14px; margin-bottom: 18px; }
.pipeline { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px; padding: 0; margin: 0; list-style: none; }
.pipeline li { background: white; border: 1px solid #e5e7f0; border-radius: 14px;
    padding: 18px; }
.pipeline h3 { font-size: 15px; color: #30334b; padding: 12px 0 7px; margin: 0; }
.pipeline p { color: #74788f; font-size: 13px; line-height: 1.65; margin: 0; }
.note { margin-top: 18px; padding: 18px 22px; background: #efedf9;
    border-radius: 14px; color: #77718d; font-size: 13px; line-height: 1.7; }
.note strong { color: #554976; }
.footer { text-align: center; color: #999caf; font-size: 11px; margin-top: 24px;
    letter-spacing: .5px; }
@media (max-width: 640px) {
    .block-container { padding: 1.5rem 1rem; }
    .pipeline { grid-template-columns: 1fr; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div >
        [data-testid="stVerticalBlock"] > [data-testid="element-container"] .panel-heading) {
        padding: 1rem !important;
    }
}
@media (max-width: 380px) {
    .block-container { padding-left: .75rem; padding-right: .75rem; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div >
        [data-testid="stVerticalBlock"] > [data-testid="element-container"] .panel-heading) {
        padding: .375rem !important;
    }
}
</style>
"""


def init_state() -> None:
    """Create the version used to reset the shared drawing pad."""
    if "canvas_version" not in st.session_state:
        st.session_state["canvas_version"] = 0


def clear_drawing() -> None:
    st.session_state["canvas_version"] += 1


def image_uri(gray: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(gray).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def render_preview(rgba: np.ndarray | None) -> None:
    """Pass actual pipeline images to the single, replayable animated preview."""
    gray = (
        np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        if rgba is None else rgba_to_grayscale(rgba)
    )
    bbox = find_bounding_box(gray)
    square = gray if bbox is None else center_in_square(gray, bbox)
    pixels = process_digit(gray)
    frames = [gray, square]
    frames.extend(
        cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
        for size in (128, 64)
    )
    frames.append(pixels)
    payload = {
        "frames": [image_uri(frame) for frame in frames],
        "pixels": pixels.tolist(),
        "hasInk": bbox is not None,
        "size": TARGET_SIZE,
    }
    template = PREVIEW_TEMPLATE.read_text(encoding="utf-8")
    components.html(template.replace("__PREVIEW_DATA__", json.dumps(payload)), height=518)


def main() -> None:
    st.set_page_config(page_title="Digit Pixelation Visualizer", page_icon=":pencil2:", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    init_state()
    st.markdown('<div class="eyebrow">THE PIXEL LAB / INTERACTIVE EXPLORER</div>', unsafe_allow_html=True)
    st.title("A little drawing. A thousand pixels.")
    st.markdown(
        '<div class="intro">Draw any digit from 0 to 9 and watch it become a tiny grid of numbers. '
        'One canvas, endless experiments.</div>',
        unsafe_allow_html=True,
    )

    draw_column, preview_column = st.columns(2)
    with draw_column, st.container(border=True):
        st.markdown(
            '<div class="panel-heading"><span class="step">01</span> Make your mark</div>'
            '<div class="panel-caption">Any digit, 0 to 9. Just draw it below.</div>',
            unsafe_allow_html=True,
        )
        result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=STROKE_WIDTH,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            update_streamlit=True,
            display_toolbar=True,
            key=f"digit_canvas_{st.session_state['canvas_version']}",
        )
        st.markdown(
            '<div class="pad-hint">White ink. Black canvas. Your handwriting.</div>',
            unsafe_allow_html=True,
        )
        st.button("Clear drawing", on_click=clear_drawing, use_container_width=True)
        st.caption("Draw with your mouse, pen or finger. Release to see the transformation.")

    with preview_column, st.container(border=True):
        st.markdown(
            '<div class="panel-heading"><span class="step">02</span> Meet your pixels</div>'
            '<div class="panel-caption">Your drawing, simplified into a single 32 x 32 grid.</div>',
            unsafe_allow_html=True,
        )
        try:
            render_preview(canvas_to_rgba(result))
        except (ValueError, cv2.error, OSError):
            LOGGER.exception("Could not render the digit preview")
            st.error("Could not process this drawing. Clear the drawing and try again.")

    st.markdown(
        '<section class="how-it-works" aria-label="How does your drawing become pixels?">'
        '<h2>How does your drawing become pixels?</h2>'
        '<p class="how-intro">Here is what happens behind the scenes each time you finish a stroke.</p>'
        '<ol class="pipeline">'
        '<li><span class="step" aria-hidden="true">01</span><h3>Capture your drawing</h3>'
        '<p>The pad records your white strokes on a black, 280 x 280 canvas. '
        'Use the same pad for any digit from 0 to 9.</p></li>'
        '<li><span class="step" aria-hidden="true">02</span><h3>Read the brightness</h3>'
        '<p>We convert the image to grayscale: one brightness value per pixel. '
        'Transparent areas are treated as black.</p></li>'
        '<li><span class="step" aria-hidden="true">03</span><h3>Find and crop the ink</h3>'
        '<p>We find the smallest rectangle around all non-black pixels, then remove '
        'the empty space outside it. A blank drawing stays blank.</p></li>'
        '<li><span class="step" aria-hidden="true">04</span><h3>Center without stretching</h3>'
        '<p>The cropped drawing is placed in the middle of a black square with a little '
        'padding. This keeps tall or wide digits in proportion.</p></li>'
        '<li><span class="step" aria-hidden="true">05</span><h3>Build the 32 x 32 grid</h3>'
        '<p>We resize the centered image using area-based averaging. Each output pixel '
        'summarizes brightness in an image area, keeping soft gray edges.</p></li>'
        '<li><span class="step" aria-hidden="true">06</span><h3>Look inside each pixel</h3>'
        '<p>The result has 1,024 values: 0 is black, 255 is white, and values in between '
        'are gray. The enlarged view keeps each pixel crisp and square.</p></li>'
        '</ol></section>'
        '<div class="note"><strong>Try it yourself:</strong> Draw a digit off-center and watch '
        'it move to the middle. Select <strong>Values</strong> to inspect brightness, '
        'or <strong>Replay</strong> to watch the transformation again. '
        'This app visualizes pixels; it does not predict which digit you drew.</div>'
        '<div class="footer">DRAW / CENTER / DOWNSAMPLE / EXPLORE</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
