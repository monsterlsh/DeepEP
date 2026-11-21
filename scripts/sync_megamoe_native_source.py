# Copyright (c) 2026, Lu Lu

"""Keep CMake's copy of an archived UBX trial at a stable path between builds."""

import filecmp
from pathlib import Path
import shutil
import stat
import sys
import time


def _sync_directory(source, destination):
    entries = {item.name: item for item in source.iterdir()}
    for old in destination.iterdir():
        new = entries.get(old.name)
        if new is not None and stat.S_IFMT(old.lstat().st_mode) == stat.S_IFMT(new.lstat().st_mode):
            continue
        if old.is_symlink() or not old.is_dir():
            old.unlink()
        else:
            shutil.rmtree(old)
    for name, new in entries.items():
        target = destination / name
        if new.is_symlink():
            if target.is_symlink() and target.readlink() == new.readlink():
                continue
            target.unlink(missing_ok=True)
            target.symlink_to(new.readlink())
        elif new.is_dir():
            target.mkdir(exist_ok=True)
            _sync_directory(new, target)
        elif not target.exists() or not filecmp.cmp(new, target, shallow=False):
            # Stamp changed content with the current time, including on a Git rollback.
            shutil.copy(new, target)
        elif stat.S_IMODE(new.stat().st_mode) != stat.S_IMODE(target.stat().st_mode):
            shutil.copymode(new, target)


def sync_trial(trial):
    trial = Path(trial).resolve(strict=True)
    if not trial.name.startswith("shmem-latest-") or not (trial / "CMakeLists.txt").is_file():
        raise ValueError(f"Expected an archived UBX trial: {trial}")
    # Only this dedicated, out-of-tree CMake mirror is reconciled with the archive.
    destination = trial.parent / "ubx-dependencies/latest-native/source"
    if destination.resolve() != destination:
        raise ValueError(f"Unexpected symlink in UBX native cache: {destination}")
    if destination.is_dir():
        # The wrapper creates each archive under the build lock, after the previous
        # build. Put changed files in a later second even for coarse Make timestamps.
        remaining = int(trial.stat().st_ctime) + 1 - time.time()
        if remaining > 0:
            time.sleep(remaining)
    destination.mkdir(parents=True, exist_ok=True)
    _sync_directory(trial, destination)
    return destination


if __name__ == "__main__":
    sync_trial(sys.argv[1])
