# Copyright (c) 2026, Lu Lu

"""Create a readable, unique benchmark result directory on the launch host."""

from datetime import datetime
from itertools import count
from pathlib import Path
import re
from uuid import uuid4


def create_run_directory(base, label, *, mode=0o700, unique_id=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_+-]*", label):
        raise ValueError(f"Invalid benchmark run label: {label!r}")
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{label}-{datetime.now():%Y%m%d-%H%M%S}"
    if unique_id:
        # Batch workers use this name to verify that the controller's run is mounted.
        stem += f"-{uuid4().hex[:8]}"
    for attempt in count(1):
        path = base / (stem if attempt == 1 else f"{stem}-{attempt:02d}")
        try:
            path.mkdir(mode=mode)
        except FileExistsError:
            continue
        return path
