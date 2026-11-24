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
