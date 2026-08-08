import os
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from runtime_logging import RUNTIME_LOG_PATH_ENV, read_log_tail, redact_sensitive


class RuntimeLoggingTest(TestCase):
    def test_redacts_credentials(self) -> None:
        value = 'api_key="secret-value" Authorization: Bearer abc.def.ghi'
        redacted = redact_sensitive(value)
        self.assertNotIn("secret-value", redacted)
        self.assertNotIn("abc.def.ghi", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_reads_only_the_latest_lines(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "runtime.log")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("first\nsecond\nthird\n")
            with patch.dict(os.environ, {RUNTIME_LOG_PATH_ENV: path}):
                self.assertEqual(read_log_tail(max_lines=2), "second\nthird")
