"""Focused tests for the shared digit preview payload."""
import base64
from io import BytesIO
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from app import render_preview
from pixelation import process_digit


class PreviewTests(unittest.TestCase):
    def payload(self, image, color="#ffffff"):
        with patch("app.components.html") as html:
            render_preview(image, color)
        document = html.call_args.args[0]
        encoded = document.split("const data = ", 1)[1].split(";\n", 1)[0]
        return json.loads(encoded)

    def test_initial_and_cleared_canvas_have_no_ink(self):
        for image in (None, np.zeros((280, 280, 4), dtype=np.uint8)):
            with self.subTest(initial=image is None):
                payload = self.payload(image)
                self.assertFalse(payload["hasInk"])
                self.assertEqual(np.count_nonzero(payload["pixels"]), 0)

    def test_all_digits_animate_to_the_existing_pipeline_result(self):
        for digit in range(10):
            with self.subTest(digit=digit):
                rgba = np.zeros((280, 280, 4), dtype=np.uint8)
                cv2.putText(rgba, str(digit), (25, 210), cv2.FONT_HERSHEY_SIMPLEX,
                            6, (255, 255, 255, 255), 20, cv2.LINE_AA)
                payload = self.payload(rgba)
                self.assertTrue(payload["hasInk"])
                self.assertEqual(len(payload["frames"]), 5)
                expected = process_digit(rgba)
                self.assertEqual(expected.shape, (32, 32))
                np.testing.assert_array_equal(payload["pixels"], expected)
                frames = [
                    Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1])))
                    for uri in payload["frames"]
                ]
                self.assertEqual([frame.size for frame in frames[2:]],
                                 [(128, 128), (64, 64), (32, 32)])
                np.testing.assert_array_equal(np.asarray(frames[-1]), expected)

    def test_preview_color_does_not_change_pixel_values(self):
        image = np.zeros((280, 280, 4), dtype=np.uint8)
        image[50:220, 130:150] = [255, 255, 255, 255]
        white = self.payload(image)
        colored = self.payload(image, "#eb1000")
        self.assertEqual(white["color"], "#ffffff")
        self.assertEqual(colored["color"], "#eb1000")
        self.assertEqual(colored["pixels"], white["pixels"])
        self.assertEqual(colored["frames"], white["frames"])


if __name__ == "__main__":
    unittest.main()
