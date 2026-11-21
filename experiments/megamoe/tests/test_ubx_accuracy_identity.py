# Copyright (c) 2026, Lu Lu

from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from experiments.megamoe.scripts import ubx_accuracy_identity as identity


def test_identity_tracks_installed_code_and_runtime(tmp_path, monkeypatch):
    site = tmp_path / "site"
    core = "deep_ep"
    experiment = "deep_ep_experimental"
    module = f"{experiment}/megamoe"
    suffix = EXTENSION_SUFFIXES[0]
    files = [
        f"{core}/__init__.py", f"{core}/_C{suffix}", f"{core}/libdeepep_kernels.so", f"{core}/_build_info.json",
        f"{experiment}/__init__.py", f"{module}/_C{suffix}", f"{module}/benchmark.py", f"{module}/accuracy.py",
        f"{module}/golden.py", f"{module}/libmegamoe_kernel.so", f"{module}/libmegamoe_urma_kernel.so",
        "numpy/__init__.py", "torch/__init__.py", "torch_npu/__init__.py",
    ]
    for lib in ("libshmem.so", "libshmem_utils.so", "aclshmem_bootstrap_uid.so", "aclshmem_bootstrap_config_store.so"):
        files.append(f"{core}/.libs/{lib}")
    for name in files:
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original")
    dist = SimpleNamespace(locate_file=lambda name: site / name, name="ascend-deepep", version="1",
                           read_text=lambda name: "test RECORD")
    monkeypatch.setattr(identity.metadata, "distribution", lambda _: dist)
    monkeypatch.setattr(identity.util, "find_spec", lambda pkg: SimpleNamespace(origin=site / pkg / "__init__.py"))
    boot_id = Path("/proc/sys/kernel/random/boot_id")
    read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **kw:
                        str(UUID(int=17)) if p == boot_id else read_text(p, *a, **kw))
    cann = tmp_path / "cann"
    (cann / "compiler").mkdir(parents=True)
    (cann / "compiler/version.info").write_text("Version=9.2")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann))

    before = identity.runtime_identity()
    assert identity.runtime_identity() == before
    for filename in (f"{module}/libmegamoe_urma_kernel.so", f"{core}/_C{suffix}", f"{module}/accuracy.py"):
        path = site / filename
        path.write_bytes(b"new implementation")
        assert identity.runtime_identity() != before
        path.write_bytes(b"original")
    (cann / "compiler/version.info").write_text("Version=9.3")
    assert identity.runtime_identity() != before
    (cann / "compiler/version.info").write_text("Version=9.2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "9,8,7")
    assert identity.runtime_identity() != before
    # The conda fallback works without framework RECORDs.
    dist.read_text = lambda _: None
    for filename in (f"torch/_C{suffix}", f"torch_npu/_C{suffix}", f"numpy/_core/_multiarray_umath{suffix}"):
        path = site / filename
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"framework")
    conda = identity.runtime_identity()
    (site / "torch_npu" / ("_C" + suffix)).write_bytes(b"modified framework")
    assert identity.runtime_identity() != conda
    (site / module / "libmegamoe_urma_kernel.so").unlink()
    with pytest.raises(FileNotFoundError):
        identity.runtime_identity()


def test_identity_rejects_package_shadowed_by_pythonpath(tmp_path, monkeypatch):
    installed = tmp_path / "site-packages/deep_ep"
    installed.mkdir(parents=True)
    (installed / "__init__.py").touch()
    dist = SimpleNamespace(name="ascend-deepep", locate_file=lambda _: installed)
    monkeypatch.setattr(identity.util, "find_spec", lambda _: SimpleNamespace(origin=tmp_path / "checkout/__init__.py"))
    with pytest.raises(ValueError, match="does not resolve"):
        identity.installed_package(dist, "deep_ep")
