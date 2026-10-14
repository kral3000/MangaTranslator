import base64
import io
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import core  # noqa: F401 -- Match the application's package initialization order.
from utils.endpoints.codex_image import GenerationCancelled, generate_image


class ImagePeer:
    def __init__(self, cancel=None):
        self.cancel = cancel
        self._directory = SimpleNamespace(name="unused")
        self._lock = threading.Lock()
        self._events = {}
        self.thread_config = {}
        self.calls = []

    def rpc(self, method, params, *_args):
        self.calls.append(method)
        if method == "thread/start":
            return {"thread": {"id": "t"}}
        if method == "turn/start":
            if self.cancel:
                self.cancel.set()
            data = io.BytesIO()
            Image.new("RGB", (8,8), "red").save(data, format="PNG")
            for method, item in (("item/started", {"type":"imageGeneration"}),
                                  ("item/completed", {"type":"imageGeneration", "status":"completed",
                                   "result":base64.b64encode(data.getvalue()).decode()})):
                self._events["t"].put({"method":method,"params":{"item":item}})
            self._events["t"].put({"method":"turn/completed","params":{"turn":{"status":"completed"}}})
            return {"turn":{"id":"turn"}}
        return {}


class CodexImageTest(unittest.TestCase):
    def test_image_bytes_and_thread_cleanup_without_closing_shared_connection(self):
        peer = ImagePeer()
        with tempfile.TemporaryDirectory() as directory, patch(
                "utils.endpoints.codex_image._get_image_server", return_value=peer):
            output = Path(directory) / "image.png"
            result = generate_image("test", [], output)
            self.assertEqual(result["path"], str(output))
            with Image.open(output) as image:
                self.assertEqual(image.getpixel((0,0)), (255,0,0))
            self.assertNotIn("turn/interrupt", peer.calls)
            self.assertIn("thread/unsubscribe", peer.calls)
            self.assertFalse(peer._events)

    def test_cancellation_interrupts_once_and_never_writes_result(self):
        cancel = threading.Event()
        peer = ImagePeer(cancel)
        with tempfile.TemporaryDirectory() as directory, patch(
                "utils.endpoints.codex_image._get_image_server", return_value=peer):
            output = Path(directory) / "image.png"
            with self.assertRaises(GenerationCancelled):
                generate_image("test", [], output, cancel)
            self.assertFalse(output.exists())
            self.assertEqual(peer.calls.count("turn/interrupt"), 1)
            self.assertFalse(peer._events)

    def test_cancelled_job_does_not_start_native_connection(self):
        cancel = threading.Event()
        cancel.set()
        with patch("utils.endpoints.codex_image._get_image_server") as start:
            with self.assertRaises(GenerationCancelled):
                generate_image("test", [], "unused.png", cancel)
            start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
