# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Validate launch plans and build subprocess environments without SSH or devices."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "experiments/megamoe/scripts"
spec = importlib.util.spec_from_file_location("megamoe_batch_launch", SCRIPTS / "batch_launch.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
spec = importlib.util.spec_from_file_location("megamoe_batch_node", SCRIPTS / "batch_node.py")
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    for key in ("SHARED_CODE", "BS_VALUES", "MASTER_PORT", "SHMEM_PORT", "SSH_PORT", "SSH_AUTH", "REMOTE_USER"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "CONDA_BASE": str(tmp_path / "conda"),
        "CONDA_ENV": "test env",
        "ASCEND_ENV": str(tmp_path / "cann/set_env.sh"),
    }.items():
        monkeypatch.setenv(key, value)
    hosts = [f"192.0.2.{i}" for i in range(1, 5)]
    path = tmp_path / "hosts.json"
    path.write_text(json.dumps({"hosts": hosts, "master": hosts[2]}))
    return ["--hosts-file", str(path), "--ep-size", "32"]


def test_explicit_plan_preserves_rank_order_and_has_no_side_effects(configuration, monkeypatch, capsys):
    with monkeypatch.context() as guard:
        guard.setattr(launcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("SSH during plan"))
        guard.setattr(Path, "mkdir", lambda *a, **kw: pytest.fail("write during plan"))
        launcher.main([*configuration, "--plan"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["hosts"] == ["192.0.2.3", "192.0.2.1", "192.0.2.2", "192.0.2.4"]
    assert plan["root"] == str(ROOT)
    script = launcher.remote_script(plan, Path("/tmp/run"), 0, "prepare")
    assert "experiments/megamoe/scripts/batch_node.py" in script
    assert "conda activate 'test env'" in script
    assert subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode == 0


@pytest.mark.parametrize(
    "hosts", [["192.0.2.1"] * 4, ["192.0.2.1"], ["invalid", "192.0.2.2", "192.0.2.3", "192.0.2.4"]]
)
def test_invalid_hosts_fail_before_launch(hosts):
    with pytest.raises(ValueError):
        launcher.select_hosts({"hosts": hosts}, 32)


def test_missing_environment_is_explicit(configuration, monkeypatch):
    monkeypatch.delenv("ASCEND_ENV")
    with pytest.raises(ValueError, match="ASCEND_ENV"):
        launcher.make_plan(launcher.parse_args(configuration))


def test_master_override_and_default(configuration):
    hosts = [f"192.0.2.{i}" for i in range(1, 5)]
    assert launcher.select_hosts({"hosts": hosts}, 32) == hosts
    plan = launcher.make_plan(launcher.parse_args([*configuration, "--master-ip", hosts[-1]]))
    assert plan["hosts"][0] == hosts[-1]
    with pytest.raises(ValueError, match="Master"):
        launcher.select_hosts({"hosts": hosts}, 32, "192.0.2.99")


def test_default_plan_preserves_a8w4_and_timer(configuration):
    plan = launcher.make_plan(launcher.parse_args(configuration))
    assert plan["mma_type"] == "fp8xfp4"
    assert plan["timer"] and plan["parse_timer"]


@pytest.mark.parametrize(
    "mma_type,input_type", [("fp8xfp8", "fp8"), ("fp8xfp8", "bf16"), ("fp4xfp8", "fp4"), ("fp4xfp8", "bf16")]
)
def test_fp8_weights_plan_passes_matching_format_to_benchmark_and_golden(configuration, tmp_path, input_type, mma_type):
    plan = launcher.make_plan(
        launcher.parse_args(
            [*configuration, "--mma-type", mma_type, "--no-timer", "--input", input_type, "--mode", "accuracy"]
        )
    )
    assert plan["mma_type"] == mma_type
    assert not plan["timer"] and not plan["parse_timer"]
    benchmark = node.benchmark_command(plan, tmp_path, 1, 96)
    golden = node.golden_command(plan, tmp_path, 1, 96)
    for command in (benchmark, golden):
        assert command[command.index("--mma-type") + 1] == mma_type
        assert command[command.index("--input") + 1] == input_type
        assert command[command.index("--golden-dir") + 1] == str(ROOT / "benchmark-golden")
    assert "--timer" not in benchmark
    assert benchmark[benchmark.index("--node-rank") + 1] == "1"
    assert golden[golden.index("--ep-size") + 1] == "32"
    assert golden[golden.index("--ranks") + 1] == "8-15"


def test_a8w8_requires_explicitly_disabled_timer(configuration):
    with pytest.raises(ValueError, match="fp8xfp8.*--no-timer"):
        launcher.make_plan(launcher.parse_args([*configuration, "--mma-type", "fp8xfp8"]))


def test_a8w8_rejects_shared_experts_before_launch(configuration):
    with pytest.raises(ValueError, match="fp8xfp8.*shared experts"):
        launcher.make_plan(
            launcher.parse_args(
                [*configuration, "--mma-type", "fp8xfp8", "--no-timer", "--config", "deepseek_v4_pro_shared1"]
            )
        )


@pytest.mark.parametrize(
    "options,match",
    [
        (["--input", "fp8"], "requires bf16 or fp4"),
        (["--input", "fp4"], "--no-timer"),
        (["--input", "fp4", "--no-timer", "--config", "deepseek_v4_pro_shared1"], "shared experts"),
    ],
)
def test_a4w8_plan_rejects_mismatched_input_timer_and_shared(configuration, options, match):
    with pytest.raises(ValueError, match=match):
        launcher.make_plan(launcher.parse_args([*configuration, "--mma-type", "fp4xfp8", *options]))


@pytest.mark.parametrize("mma_type,expected", [("fp8xfp4", "fp8"), ("fp8xfp8", "fp8"), ("fp4xfp8", "fp4")])
def test_batch_default_input_follows_activation_mode(configuration, mma_type, expected):
    args = launcher.parse_args([*configuration, "--mma-type", mma_type, "--no-timer"])
    assert args.input == expected
    assert launcher.make_plan(args)["input"] == expected


def test_legacy_plan_uses_a8w4_for_benchmark_and_golden(configuration, tmp_path):
    plan = launcher.make_plan(launcher.parse_args(configuration))
    del plan["mma_type"]
    for factory in (node.benchmark_command, node.golden_command):
        command = factory(plan, tmp_path, 0, 96)
        assert command[command.index("--mma-type") + 1] == "fp8xfp4"


@pytest.mark.parametrize("entry", ["build.py", "batch_node.py", "batch_launch.py"])
def test_relocated_entrypoints_work_in_isolated_python(entry):
    result = subprocess.run([sys.executable, "-I", str(SCRIPTS / entry), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_build_isolates_git_libraries_and_preserves_framework_environment(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("megamoe_build", SCRIPTS / "build.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    (tmp_path / "dependencies.lock.json").write_text(json.dumps(lock))
    for name in ("SHMEM_ROOT", "CATLASS_ROOT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in {
        "DEEPEP_NPU_ARCH": lock["target"],
        "DEEPEP_BUILD_MEGAMOE": "ON",
        "DEEPEP_MEGAMOE_TIMER": "ON",
        "LD_LIBRARY_PATH": "/opt/conda/lib:/opt/cann/lib64",
        "LD_PRELOAD": "/opt/conda/lib/libstdc++.so.6",
        "HTTPS_PROXY": "http://proxy.example:8080",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(builder.platform, "system", lambda: "Linux")
    monkeypatch.setattr(builder.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(builder, "cann_version", lambda version: None)
    original_environment = dict(os.environ)
    clones, queries, framework_checks = [], [], []

    class DependenciesPrepared(Exception):
        pass

    def check_environment(command, env):
        assert env["HTTPS_PROXY"] == original_environment["HTTPS_PROXY"]
        if command[0] == "git":
            assert "LD_LIBRARY_PATH" not in env
            assert "LD_PRELOAD" not in env
        else:
            for name in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
                assert env[name] == original_environment[name]

    def run(command, *, cwd, env, check):
        check_environment(command, env)
        assert check
        if command[0] == "git":
            assert command[1:6] == ["submodule", "update", "--init", "--recursive", "--"]
            clones.append(command[-1])
        elif command[1:4] == ["-m", "pip", "install"]:
            # Stop before package installation or compilation; only dependency preparation is exercised.
            raise DependenciesPrepared
        else:
            assert command[:2] == [sys.executable, "-c"]
            framework_checks.append(command)

    def check_output(command, *, env, text):
        check_environment(command, env)
        assert command[:2] == ["git", "-C"] and text
        queries.append(command)
        if command[3:] == ["rev-parse", "HEAD"]:
            return (
                lock["shmem"]["revision"] if Path(command[2]).name == "shmem" else lock["megamoe"]["catlass_revision"]
            )
        assert command[3:] == ["status", "--porcelain", "--untracked-files=normal"]
        return ""

    monkeypatch.setattr(builder.subprocess, "run", run)
    monkeypatch.setattr(builder.subprocess, "check_output", check_output)
    with pytest.raises(DependenciesPrepared):
        builder.build(tmp_path)
    assert clones == ["third-party/shmem", "third-party/catlass"]
    assert len(queries) == 4
    assert len(framework_checks) == 1
    assert dict(os.environ) == original_environment


@pytest.mark.parametrize("git_exit", [0, 7])
def test_shmem_nested_git_preserves_compiler_environment_and_cleans_up(tmp_path, monkeypatch, git_exit):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("shmem_build", ROOT / "scripts/build_shmem.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    (tmp_path / "dependencies.lock.json").write_text(json.dumps(lock))
    source = tmp_path / "third-party/shmem"
    (source / "scripts").mkdir(parents=True)
    (source / ".git").touch()
    tools = tmp_path / "tools with spaces"
    tools.mkdir()
    q = shlex.quote
    probe = (
        "import json, os, pathlib, sys; "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(env=dict(os.environ), args=sys.argv[2:])))"
    )
    git = tools / "git"
    git.write_text(
        '#!/bin/sh\nif [ "$1" = "-C" ]; then\n'
        '  if [ "$3" = "rev-parse" ]; then\n'
        f'    if [ "$4" = "HEAD" ]; then echo {q(lock["shmem"]["revision"])}; else echo {q(str(source))}; fi\n'
        "  fi\n  exit 0\nfi\n"
        f'{q(sys.executable)} -c {q(probe)} git-env.json "$@"\nexit {git_exit}\n'
    )
    git.chmod(0o700)
    (source / "scripts/build.sh").write_text(
        'git clone --branch v3.11.3 --depth 1 https://example.invalid/json.git "json target"\n'
        f"{q(sys.executable)} -c {q(probe)} compiler-env.json\n"
    )
    for name, value in {
        "PATH": str(tools) + os.pathsep + os.defpath,
        "LD_LIBRARY_PATH": "/fixture/conda/lib:/fixture/cann/lib64",
        "LD_PRELOAD": "libm.so.6",
        "https_proxy": "http://proxy.example:8080",
        "CMAKE_BUILD_PARALLEL_LEVEL": "2",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(builder, "cann_version", lambda expected: {"Version": expected})
    monkeypatch.setattr(builder.platform, "system", lambda: "Linux")
    recorded = []

    def record(sdk, revision, **kwargs):
        recorded.append((sdk, revision))
        return sdk / "deepep-sdk.json"

    monkeypatch.setattr(builder, "record", record)
    original_environment = dict(os.environ)
    if git_exit:
        with pytest.raises(subprocess.CalledProcessError) as error:
            builder.prepare(tmp_path)
        assert error.value.returncode == git_exit
        assert not recorded
        assert not (source / "compiler-env.json").exists()
    else:
        builder.prepare(tmp_path)
        assert recorded == [(source / "install/shmem", lock["shmem"]["revision"])]
        compiler_env = json.loads((source / "compiler-env.json").read_text())["env"]
        for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "https_proxy", "CMAKE_BUILD_PARALLEL_LEVEL"):
            assert compiler_env[name] == original_environment[name]
    report = json.loads((source / "git-env.json").read_text())
    git_env = report["env"]
    assert "LD_LIBRARY_PATH" not in git_env and "LD_PRELOAD" not in git_env
    assert git_env["https_proxy"] == original_environment["https_proxy"]
    assert report["args"] == [
        "clone",
        "--branch",
        "v3.11.3",
        "--depth",
        "1",
        "https://example.invalid/json.git",
        "json target",
    ]
    assert not Path(git_env["PATH"].split(os.pathsep)[0]).exists()
    assert dict(os.environ) == original_environment
