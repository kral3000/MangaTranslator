"""Settings location shared by the web UI and the standalone editor."""

import os
from pathlib import Path

CONFIG_FILE = (
    Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
    / "MangaTranslator"
    / "config.json"
)
