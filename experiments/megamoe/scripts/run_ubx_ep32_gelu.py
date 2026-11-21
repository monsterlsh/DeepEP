# Copyright (c) 2026, Lu Lu

"""Compatibility entry point for run_ubx_gelu.py."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ubx_gelu import cli  # noqa: E402

sys.path.pop(0)

if __name__ == "__main__":
    sys.exit(cli(["--ep-size", "32", *sys.argv[1:]]))
