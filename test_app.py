"""Focused tests for the shared digit preview payload."""
import base64
from io import BytesIO
import json
import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import psycopg
from streamlit.testing.v1 import AppTest

from app import clear_drawing, render_preview, render_visitor_count, visitor_database_url
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


class VisitorCountTests(unittest.TestCase):
    def setUp(self):
        self.state = {"canvas_version": 0}
        self.total = 1233
        self.state_patch = patch("app.st.session_state", self.state)
        self.secrets_patch = patch(
            "app.st.secrets", {"visitor_counter": {"database_url": "test-database"}}
        )
        self.state_patch.start()
        self.secrets_patch.start()
        self.addCleanup(self.state_patch.stop)
        self.addCleanup(self.secrets_patch.stop)
        env_patch = patch.dict(os.environ, {}, clear=True)
        file_patch = patch("app.dotenv_values", return_value={})
        env_patch.start()
        file_patch.start()
        self.addCleanup(env_patch.stop)
        self.addCleanup(file_patch.stop)

    def count(self, database_url, *, new_visit):
        if new_visit:
            self.total += 1
        return self.total

    def test_reruns_and_clear_count_once_but_new_session_counts_again(self):
        with patch("app.get_visitor_count", side_effect=self.count), \
                patch("app.st.caption") as caption:
            render_visitor_count()
            render_visitor_count()
            clear_drawing()
            render_visitor_count()
            self.assertEqual(self.total, 1234)
            caption.assert_called_with("Total visitors: 1,234 (browser sessions)")
            self.state.clear()
            render_visitor_count()
            self.assertEqual(self.total, 1235)

    def test_failed_first_visit_is_retried_without_displaying_a_false_total(self):
        with patch("app.get_visitor_count", side_effect=psycopg.OperationalError("offline")), \
                patch("app.LOGGER.exception") as log, \
                patch("app.st.warning") as warning, patch("app.st.caption") as caption:
            render_visitor_count()
            self.assertNotIn("visitor_counted", self.state)
            warning.assert_called_once()
            log.assert_called_once()
            caption.assert_not_called()
        with patch("app.get_visitor_count", side_effect=self.count), patch("app.st.caption"):
            render_visitor_count()
            self.assertTrue(self.state["visitor_counted"])
            self.assertEqual(self.total, 1234)

    def test_missing_configuration_keeps_the_drawing_app_available(self):
        with patch("app.st.secrets", {}), patch("app.get_visitor_count") as counter, \
                patch("app.LOGGER.exception"), patch("app.st.warning") as warning:
            render_visitor_count()
            warning.assert_called_once()
            counter.assert_not_called()
            self.assertNotIn("visitor_counted", self.state)


class VisitorPageTests(unittest.TestCase):
    def test_page_displays_total_and_clear_does_not_count_a_second_visit(self):
        with patch("app.get_visitor_count", return_value=42) as counter, \
                patch.dict(os.environ, {}, clear=True), \
                patch("app.dotenv_values", return_value={}):
            page = AppTest.from_string("import app\napp.main()")
            page.secrets["visitor_counter"] = {"database_url": "test-database"}
            page.run()
            self.assertFalse(page.exception)
            self.assertIn("Total visitors: 42 (browser sessions)",
                          [caption.value for caption in page.caption])
            self.assertTrue(counter.call_args.kwargs["new_visit"])
            page.button[0].click().run()
            self.assertFalse(page.exception)
            self.assertFalse(counter.call_args.kwargs["new_visit"])
            self.assertEqual(page.session_state["canvas_version"], 1)


class DatabaseConfigurationTests(unittest.TestCase):
    def test_environment_takes_priority_without_reading_secrets_or_file(self):
        with patch.dict(os.environ, {"DATABASE_URL": "environment-url"}, clear=True), \
                patch("app.dotenv_values") as dotenv, patch("app.st.secrets", {}):
            self.assertEqual(visitor_database_url(), "environment-url")
            dotenv.assert_not_called()

    def test_neon_file_is_used_without_streamlit_secrets(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("app.dotenv_values", return_value={"DATABASE_URL": "file-url"}), \
                patch("app.st.secrets", {}):
            self.assertEqual(visitor_database_url(), "file-url")

    def test_top_level_streamlit_secret_is_supported(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("app.dotenv_values", return_value={}), \
                patch("app.st.secrets", {"DATABASE_URL": "cloud-url"}):
            self.assertEqual(visitor_database_url(), "cloud-url")

    def test_empty_environment_url_does_not_silently_use_another_database(self):
        with patch.dict(os.environ, {"DATABASE_URL": ""}, clear=True), \
                patch("app.dotenv_values") as dotenv:
            with self.assertRaises(ValueError):
                visitor_database_url()
            dotenv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
