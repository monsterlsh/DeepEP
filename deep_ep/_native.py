# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

"""Lazy native loading. Importing deep_ep never imports a framework or driver."""

import importlib
import importlib.util
from importlib.machinery import EXTENSION_SUFFIXES
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import NoReturn

_SHMEM_LIBRARIES = (
    "libshmem_utils.so",
    "aclshmem_bootstrap_config_store.so",
    "aclshmem_bootstrap_uid.so",
    "libshmem.so",
)
_shmem_lock = RLock()
_shmem_handles = {}  # Keep successfully loaded objects alive, including after a partial failure.


def _loaded_shmem_paths():
    # Native builds target Linux. Inspect mappings without loading another SDK.
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5].removesuffix(" (deleted)"))
            if path.name in _SHMEM_LIBRARIES:
                paths.add(path.resolve())
    return paths


def _load_bundled_shmem(directory, info):
    import ctypes

    bundle = (directory / ".libs").resolve()
    hashes = info.get("shmem_runtime", {})
    with _shmem_lock:
        if set(hashes) != set(_SHMEM_LIBRARIES):
            raise ImportError("Incomplete bundled SHMEM manifest; rebuild and reinstall DeepEP")
        # Check the entire closure before loading anything; never fall back to
        # SHMEM_ROOT or LD_LIBRARY_PATH when a wheel is damaged.
        for name in _SHMEM_LIBRARIES:
            path = bundle / name
            if not path.is_file() or path.resolve().parent != bundle:
                raise ImportError(f"Missing bundled SHMEM library: {path}; reinstall DeepEP")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != hashes[name]:
                raise ImportError(f"Bundled SHMEM checksum mismatch: {name}; reinstall DeepEP")
        for path in _loaded_shmem_paths():
            if path != bundle / path.name:
                raise ImportError(
                    f"Conflicting SHMEM library already loaded: {path}; "
                    "start a new process without preloading another SHMEM SDK"
                )
        for name in _SHMEM_LIBRARIES:
            path = str(bundle / name)
            if path not in _shmem_handles:
                try:
                    # Absolute paths select this wheel. Preloading utils also
                    # resolves the SDK's transitive dependency without an SDK RPATH.
                    mode = ctypes.RTLD_LOCAL if name.startswith("aclshmem_bootstrap_") else ctypes.RTLD_GLOBAL
                    _shmem_handles[path] = ctypes.CDLL(path, mode=mode)
                except OSError as exc:
                    raise ImportError(
                        f"Cannot load bundled SHMEM library {name}: {exc}. "
                        "Load the matching CANN environment and check its driver dependencies."
                    ) from exc
    return bundle


def _unavailable(operation: str) -> NoReturn:
    raise NotImplementedError(
        f"{operation} is not implemented in DeepEP. "
        "No communication operation or simulated result has been performed. "
        "See docs/api.md for implementation status."
    )


class _SourceOnlyCheckout(ImportError):
    """The imported source tree has never had a native build published into it."""


def _missing_native(directory):
    # Run filesystem and distribution diagnostics only on a failed native probe.
    from importlib.metadata import distributions

    candidates = sorted(p for p in directory.glob("_C.*") if p.suffix in {".so", ".pyd"})
    found = [
        f"{p.name} -> {p.readlink()} (broken symlink)" if p.is_symlink() and not p.exists() else p.name
        for p in candidates
    ]
    wheels = sorted({
        Path(dist.locate_file("deep_ep")).resolve()
        for dist in distributions(name="ascend-deepep") if dist.read_text("WHEEL") is not None
    })
    message = (
        f"Cannot load deep_ep._C with {sys.executable}. Imported package: {directory}. "
        f"Expected one of {', '.join('_C' + suffix for suffix in EXTENSION_SUFFIXES)}; "
        f"native candidates: {', '.join(found) or 'none'}. "
    )
    if wheels and directory not in wheels:
        message += "The imported package is outside the installed wheel. "
    if wheels:
        message += f"ascend-deepep wheel locations: {', '.join(map(str, wheels))}. "
    anchor = directory / ".native-build"
    if anchor.is_symlink() and not anchor.exists():
        message += f"Source build link {anchor} -> {anchor.readlink()} is broken. "
    message += "Use the matching native wheel in this interpreter and check the package import path."
    artifacts = [anchor, directory / "_build_info.json", directory / ".libs", directory / "libdeepep_kernels.so"]
    if (directory not in wheels and not candidates and not any(p.exists() or p.is_symlink() for p in artifacts)
            and (directory.parent / "dependencies.lock.json").is_file() and (directory.parent / "setup.py").is_file()):
        return _SourceOnlyCheckout(message)
    return ImportError(message)


@lru_cache(maxsize=1)
def _require_native_cached():
    # Cache only a successfully validated module, shared by all operation names.
    # Exceptions are not cached, so missing files or failed imports can be retried.
    directory = Path(__file__).resolve().parent
    if not any((directory / ("_C" + suffix)).is_file() for suffix in EXTENSION_SUFFIXES):
        raise _missing_native(directory)
    return _load_native()


def require_native(operation):
    try:
        return _require_native_cached()
    except _SourceOnlyCheckout:
        if operation == "MegaMoE":
            raise
        # Preserve the existing API-only behavior for the other entry points.
        _unavailable(operation)


@lru_cache(maxsize=1)
def _load_native():
    directory = Path(__file__).resolve().parent
    spec = importlib.util.find_spec("deep_ep._C")
    if spec is None:
        raise ImportError(f"{sys.executable} cannot resolve deep_ep._C in {directory}")
    # A source build exposes its own native files through package-local links.
    # Check the importing directory, not the target of the extension symlink.
    if not spec.origin or Path(spec.origin).parent.resolve() != directory:
        raise ImportError(f"{sys.executable} resolved deep_ep._C to {spec.origin!r}; expected a file in {directory}")
    manifest = directory / "_build_info.json"
    try:
        info = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ImportError(f"Cannot read native build manifest {manifest}: {exc}") from exc
    if not isinstance(info, dict):
        raise ImportError(f"Expected a JSON object in native build manifest {manifest}")
    bundle = _load_bundled_shmem(directory, info)
    try:
        module = importlib.import_module("deep_ep._C")
    except (ImportError, OSError) as exc:
        raise ImportError(
            f"Cannot load deep_ep._C from {spec.origin} with {sys.executable}: {exc}"
        ) from exc
    protocol = getattr(module, "runtime_protocol", None)
    if protocol != 1:
        raise ImportError(f"Unsupported DeepEP runtime protocol in {spec.origin}: expected 1, found {protocol!r}")
    library = Path(module.shmem_library_path()).resolve()
    if library != bundle / "libshmem.so":
        raise ImportError(
            f"Native extension resolved an external SHMEM library: {library}; restart with a clean environment"
        )
    return module


def npu_api():
    import torch
    import torch_npu  # noqa: F401

    return torch.npu
