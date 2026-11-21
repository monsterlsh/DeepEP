# Copyright (c) 2026, Lu Lu

"""Fingerprint the installed UBX accuracy runtime without opening an NPU."""

import hashlib
from importlib import metadata, util
from importlib.machinery import EXTENSION_SUFFIXES
import json
import os
from pathlib import Path
import sys
from uuid import UUID


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def installed_package(dist, package):
    # torchrun inherits PYTHONPATH. A source checkout shadowing the installed
    # wheel must not accidentally claim the wheel's earlier accuracy results.
    spec = util.find_spec(package)
    path = Path(dist.locate_file(package))
    if not spec or not spec.origin or Path(spec.origin).resolve() != (path / "__init__.py").resolve(strict=True):
        raise ValueError(f"{package} does not resolve to the installed {dist.name} package")
    return path


def wheel_files(core, experiment):
    megamoe = experiment / "megamoe"
    required = [
        core / "_build_info.json",
        core / "libdeepep_kernels.so",
        megamoe / "libmegamoe_kernel.so",
        megamoe / "libmegamoe_urma_kernel.so",
        megamoe / "benchmark.py",
        megamoe / "accuracy.py",
        megamoe / "golden.py",
    ]
    for package in (core, megamoe):
        if not any((package / ("_C" + suffix)).is_file() for suffix in EXTENSION_SUFFIXES):
            raise ValueError(f"Missing native extension in {package}")
    for name in ("libshmem.so", "libshmem_utils.so", "aclshmem_bootstrap_uid.so", "aclshmem_bootstrap_config_store.so"):
        required.append(core / ".libs" / name)
    files = set(required)
    files.add(experiment / "__init__.py")
    for package in (core, megamoe):
        files.update(p for p in package.rglob("*") if p.suffix in (".py", ".so", ".pyd") and p.is_file())
    return {path.relative_to(core.parent).as_posix(): file_hash(path) for path in sorted(files)}


def runtime_identity():
    dist = metadata.distribution("ascend-deepep")
    core = installed_package(dist, "deep_ep")
    experiment = installed_package(dist, "deep_ep_experimental")
    dependencies = {}
    for name in ("numpy", "torch", "torch_npu"):
        dependency = metadata.distribution(name)
        directory = installed_package(dependency, name)
        record = dependency.read_text("RECORD")
        if record:
            package_build = hashlib.sha256(record.encode()).hexdigest()
        else:
            # Some conda packages omit RECORD. Fingerprint the installed main
            # extension by stat without reading multi-gigabyte framework libraries.
            paths = sorted(path for suffix in EXTENSION_SUFFIXES for path in (
                directory / ("_C" + suffix),
                directory / "_core" / ("_multiarray_umath" + suffix),
                directory / "core" / ("_multiarray_umath" + suffix),
            ) if path.is_file())
            if not paths:
                raise ValueError(f"Missing installed build identity for {name}")
            package_build = {}
            for path in paths:
                status = path.stat()
                package_build[str(path.resolve())] = [status.st_size, status.st_mtime_ns, status.st_ctime_ns]
        dependencies[name] = [str(directory.resolve()), dependency.version, package_build]

    cann = os.environ.get("ASCEND_HOME_PATH")
    if not cann:
        raise ValueError("Missing ASCEND_HOME_PATH")
    versions = [Path(cann) / "compiler/version.info"]
    versions.extend(path for path in (
        Path(os.environ.get("ASCEND_DRIVER_PATH") or "/usr/local/Ascend/driver") / "version.info",
        Path("/usr/local/Ascend/firmware/version.info"),
    ) if path.is_file())
    prefixes = ("ASCEND_", "ACL_", "ACLSHMEM_", "CANN_", "DEEPEP_", "HCCL_", "NPU_", "SHMEM_", "TORCH_", "ATB_", "PYTORCH_")
    env = {k: v for k, v in os.environ.items() if k.startswith(prefixes) or k in (
        "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "PYTHONPATH", "RANK_TABLE_FILE",
    )}
    data = {
        "schema": 1,
        "host_boot_id": UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip()).hex,
        "python": [str(Path(sys.executable).resolve()), sys.version],
        "wheel": wheel_files(core, experiment),
        "frameworks": dependencies,
        "runtime_versions": {str(path.resolve()): file_hash(path) for path in versions},
        "environment": env,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    try:
        print(runtime_identity())
    except (OSError, ValueError, ImportError) as error:
        sys.exit(f"[UBX] 无法确认 NPU 精度缓存的运行环境：{error}")
