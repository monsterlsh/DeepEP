# Copyright (c) 2026, Lu Lu

import json
import os
from pathlib import Path
import subprocess
import venv

from experiments.megamoe.scripts import run_megamoe as launcher


def test_worker_imports_wheel_from_its_interpreter_and_keeps_cann(tmp_path, monkeypatch):
    prefix = tmp_path / "py311_tpr"
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(prefix)
    python = prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    wheel_site = Path(subprocess.check_output(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True,
    ).strip())

    checkout = tmp_path / "checkout"
    for name, source in (("deep_ep", checkout), ("deep_ep_experimental", checkout / "experiments" / "python")):
        for parent in (source, wheel_site):
            package = parent / name
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
    cann = tmp_path / "cann-python"
    cann.mkdir()
    (cann / "cann_test_marker.py").write_text("", encoding="utf-8")
    monkeypatch.chdir(checkout)
    search_path = os.pathsep.join(("", ".", "experiments/python", str(cann), ""))
    initial = dict(os.environ, PYTHONPATH=search_path, PYTHONSAFEPATH="", PYTHONNOUSERSITE="")
    env = launcher.ubx_wheel_env(initial)

    assert initial["PYTHONPATH"] == search_path
    assert env["PYTHONSAFEPATH"] == env["PYTHONNOUSERSITE"] == "1"
    imports = (
        "import json, site, deep_ep, deep_ep_experimental, cann_test_marker; "
        "assert site.ENABLE_USER_SITE is False; "
        "print(json.dumps([m.__file__ for m in (deep_ep, deep_ep_experimental, cann_test_marker)]))"
    )
    files = json.loads(subprocess.check_output([str(python), "-c", imports], env=env, text=True))
    assert list(map(Path, files)) == [
        wheel_site / "deep_ep" / "__init__.py",
        wheel_site / "deep_ep_experimental" / "__init__.py",
        cann / "cann_test_marker.py",
    ]


def test_probe_diagnoses_shadowed_and_incomplete_installs_even_in_isolated_python(tmp_path):
    from importlib.machinery import EXTENSION_SUFFIXES

    env = tmp_path / "env"
    venv.EnvBuilder(with_pip=False).create(env)
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    site = Path(subprocess.check_output(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True,
    ).strip())
    core = site / "deep_ep"
    beta = site / "deep_ep_experimental"
    for pkg in (core, beta, beta / "megamoe"):
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
    dist = site / "ascend_deepep-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: ascend-deepep\nVersion: 1.0\n", encoding="utf-8")
    (dist / "RECORD").write_text("deep_ep/__init__.py,,\ndeep_ep_experimental/__init__.py,,\n", encoding="utf-8")
    ext = "_C" + EXTENSION_SUFFIXES[0]
    (beta / "megamoe" / ext).touch()
    (core / ext).touch()

    def check():
        code = "import runpy,sys; runpy.run_path(sys.argv[1])['check_wheel_packages']()"
        return subprocess.run([str(python), "-I", "-c", code, launcher.DEVICE_CHECK],
                              capture_output=True, text=True)

    (dist / "direct_url.json").write_text(json.dumps({"archive_info": {"hashes": {"sha256": "a" * 64}}}))
    clean = check()
    assert clean.returncode == 0 and f"wheel_sha256={'a' * 64}" in clean.stdout
    src = tmp_path / "checkout"
    (src / "deep_ep").mkdir(parents=True)
    (src / "deep_ep" / "__init__.py").write_text("", encoding="utf-8")
    hook = site / "shadow.pth"
    hook.write_text(f"import sys; sys.path.insert(0, {str(src)!r})\n", encoding="utf-8")
    result = check()
    assert result.returncode and ".pth" in result.stderr
    assert str(src / "deep_ep") in result.stderr and str(core) in result.stderr

    hook.unlink()
    (dist / "direct_url.json").write_text("{")
    (core / ext).rename(core / "_C.incompatible.so")
    result = check()
    assert result.returncode and "No compatible native extension" in result.stderr
    assert ext in result.stderr and "_C.incompatible.so" in result.stderr
    assert "wheel_sha256=unrecorded" in result.stdout
