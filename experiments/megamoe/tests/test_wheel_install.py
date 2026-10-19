# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Exercise wheel installation with real, offline pip and synthetic native payloads."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
OLD_VERSION = "0.1.0.dev5"
NEW_VERSION = "0.1.0.dev7+megamoe.a8w8.ep8"


def write_wheel(directory, version):
    metadata = f"ascend_deepep-{version}.dist-info"
    files = {
        "deep_ep/__init__.py": "",
        "deep_ep/_native.py": "def require_native(context):\n    return None\n",
        f"{metadata}/METADATA": f"Metadata-Version: 2.1\nName: ascend-deepep\nVersion: {version}\n",
        f"{metadata}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = f"{metadata}/RECORD"
    files[record] = "".join(f"{name},,\n" for name in [*files, record])
    path = directory / f"ascend_deepep-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path


@pytest.fixture
def install_case(tmp_path, monkeypatch):
    pytest.importorskip("ensurepip", reason="Offline pip regression needs Python's bundled ensurepip")
    environment = tmp_path / "environment"
    subprocess.run([sys.executable, "-I", "-m", "venv", str(environment)], check=True, timeout=90)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    for name in ("PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_USER", "PYTHONHOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PIP_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    monkeypatch.setenv("PIP_REQUIRE_VIRTUALENV", "1")
    monkeypatch.setenv("DEEPEP_BUILD_MEGAMOE", "OFF")
    monkeypatch.setenv("DEEPEP_MEGAMOE_TIMER", "OFF")
    monkeypatch.setenv("DEEPEP_NPU_ARCH", "Ascend950")
    dependency = tmp_path / "sdk"
    dependency.mkdir()
    for name in ("SHMEM_ROOT", "CATLASS_ROOT"):
        monkeypatch.setenv(name, str(dependency))

    old_wheel = write_wheel(tmp_path, OLD_VERSION)
    subprocess.run(
        [str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", str(old_wheel)],
        check=True,
        timeout=60,
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "VERSION").write_text(NEW_VERSION, encoding="utf-8")
    (source / "dependencies.lock.json").write_text(
        json.dumps({"cann_version": "synthetic", "target": "Ascend950"}), encoding="utf-8"
    )
    egg_info = source / "ascend_deepep.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text(
        f"Metadata-Version: 2.1\nName: ascend-deepep\nVersion: {NEW_VERSION}\n", encoding="utf-8"
    )
    package = source / "deep_ep"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('Source package must not be imported')\n")
    monkeypatch.setenv("PYTHONPATH", str(source))

    def installed_versions(*, isolated=True):
        probe = (
            "import json; from importlib.metadata import distributions; "
            "print(json.dumps([item.version for item in distributions(name='ascend-deepep')]))"
        )
        return json.loads(
            subprocess.check_output(
                [str(python), *(["-I"] if isolated else []), "-c", probe],
                cwd=tmp_path,
                text=True,
                timeout=30,
            )
        )

    assert installed_versions() == [OLD_VERSION]
    assert installed_versions(isolated=False)[0] == NEW_VERSION

    spec = importlib.util.spec_from_file_location(
        "test_megamoe_wheel_builder", ROOT / "experiments/megamoe/scripts/build.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.setattr(builder, "sys", SimpleNamespace(executable=str(python), version_info=sys.version_info, path=[]))
    monkeypatch.setattr(builder, "platform", SimpleNamespace(system=lambda: "Linux"))
    monkeypatch.setattr(builder, "shutil", SimpleNamespace(which=lambda tool: tool))
    monkeypatch.setattr(builder, "cann_version", lambda lock: None)
    monkeypatch.setitem(sys.modules, "build_support", SimpleNamespace(build_configuration=lambda env: None))

    def build(wheel_version=NEW_VERSION, *, duplicate_metadata=False):
        def run(command, **kwargs):
            if "pip" in command and "wheel" in command:
                write_wheel(Path(command[command.index("-w") + 1]), wheel_version)
                return subprocess.CompletedProcess(command, 0)
            if "pip" in command and "-r" in command:
                return subprocess.CompletedProcess(command, 0)
            if "pip" in command and "install" in command:
                command = [*command, "--no-index"]
            result = subprocess.run(command, **kwargs, timeout=60)
            if duplicate_metadata and "pip" in command and "install" in command:
                site = Path(
                    subprocess.check_output(
                        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                        text=True,
                        timeout=30,
                    ).strip()
                )
                stale = site / f"ascend_deepep-{OLD_VERSION}.dist-info"
                stale.mkdir()
                (stale / "METADATA").write_text(f"Metadata-Version: 2.1\nName: ascend-deepep\nVersion: {OLD_VERSION}\n")
            return result

        monkeypatch.setattr(builder, "subprocess", SimpleNamespace(run=run))
        return builder.build(source)

    return build, installed_versions


def test_build_replaces_old_install_despite_source_egg_info(install_case):
    build, installed_versions = install_case
    wheel = build()
    assert wheel.name == f"ascend_deepep-{NEW_VERSION}-py3-none-any.whl"
    assert installed_versions() == [NEW_VERSION]


@pytest.mark.parametrize("fault", ["wrong_version", "duplicate_metadata"])
def test_build_rejects_inconsistent_installed_metadata(install_case, fault):
    build, _ = install_case
    with pytest.raises(subprocess.CalledProcessError) as error:
        build(
            "0.1.0.dev7" if fault == "wrong_version" else NEW_VERSION, duplicate_metadata=fault == "duplicate_metadata"
        )
    assert error.value.cmd[1:3] == ["-I", "-c"]
