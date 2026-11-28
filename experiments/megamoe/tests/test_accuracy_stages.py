# Copyright (c) 2026, Lu Lu

import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.megamoe.scripts import run_megamoe as runner
from deep_ep_experimental.megamoe import api, benchmark, cascade, golden, prefill_cascade, validation


@pytest.mark.parametrize("case", ["L01", "L04"])
@pytest.mark.parametrize("stage", runner.ACCURACY_STAGES)
def test_manual_stage_plan(tmp_path, capsys, case, stage):
    preset = runner.SCRIPT.parent / "configs/l0" / f"{case}.json"
    argv = ["--run-config", str(preset), "--accuracy-stage", stage, "--accuracy-arm", "gelu", "--plan"]
    with pytest.raises(SystemExit):
        runner.entrypoint(argv)
    directory = tmp_path / "saved golden"
    assert runner.entrypoint([*argv, "--golden-dir", str(directory)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["golden_dir"] == str(directory) and plan["accuracy_stage"] == stage
    job, = plan["jobs"]
    assert job["name"] == f"gelu-{stage}" and job["profile_memory_peak"] == (stage != "golden")
    assert not directory.exists()


@pytest.mark.parametrize("stage", runner.ACCURACY_STAGES)
def test_ubx_launches_only_requested_stage_on_both_nodes(tmp_path, monkeypatch, stage):
    golden_dir = tmp_path / "golden"
    preset = runner.read_run_config(runner.SCRIPT.parent / "configs/l0/L01.json")
    for key in ("suite", "case"):
        del preset[key]
    preset.update(dual_reference=True, output_dir=str(tmp_path / "out"))
    path = tmp_path / "case.json"
    path.write_text(json.dumps(preset), encoding="utf-8")
    monkeypatch.setenv("PYTHON", sys.executable)
    monkeypatch.setenv("UBX_CANN_LOADED", "1")
    phases, actions = [], []

    def run_stage(label, scripts, logs, **kw):
        phases.append(label)
        for rank, script in scripts.items():
            assert str(golden_dir) in script and "DUAL_REFERENCE=0" in script
            with monkeypatch.context() as child:
                child.setenv("GOLDEN_DIR", str(golden_dir))
                child.setenv("DUAL_REFERENCE", "0")
                args = shlex.split(script.splitlines()[-1])
                assert runner.entrypoint(args[3:]) == 0

    def prepare(py, env, args, precision, config):
        actions.append(("golden", args.node_rank))
        assert Path(env["GOLDEN_DIR"]) == golden_dir

    def logged(command, env, log):
        actions.append(("cascade" if "--cascade-only" in command else "megamoe", len(actions)))
        assert "--dual-reference" not in command
        if "--cascade-only" in command:
            assert command[command.index("--accuracy-cascade") + 1] == "prefill"
        assert Path(command[command.index("--golden-dir") + 1]) == golden_dir

    monkeypatch.setattr(runner, "run_stage", run_stage)
    monkeypatch.setattr(runner, "prepare_golden", prepare)
    monkeypatch.setattr(runner, "run_logged", logged)
    monkeypatch.setattr(runner, "report_lines", lambda *a: [])
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: None)
    assert runner.entrypoint(["--run-config", str(path), "--accuracy-stage", stage,
                              "--golden-dir", str(golden_dir)]) == 0
    assert phases == [("prepare" if stage == "golden" else "run") + "-accuracy-w4a8"]
    assert actions == [(stage, 0), (stage, 1)]


@pytest.mark.parametrize("stage", runner.ACCURACY_STAGES)
def test_pod_stage_does_not_launch_other_implementations(tmp_path, monkeypatch, stage):
    data = runner.read_run_config(runner.SCRIPT.parent / "configs/950pod_target.json")
    data.update(output_dir=str(tmp_path / "out"), check_tokens=1, precision="a8w4")
    path = tmp_path / "case.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("DEEPEP_950POD_BOOTSTRAPPED", "1")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: None)
    golden_dir = tmp_path / "saved"
    calls = []

    def logged(command, env, log):
        assert Path(command[command.index("--golden-dir") + 1]) == golden_dir
        if runner.MODULE + ".golden" in command:
            calls.append("golden")
            return
        calls.append("cascade" if "--cascade-only" in command else "megamoe")
        assert "--dual-reference" not in command
        if "--cascade-only" in command:
            assert command[command.index("--accuracy-cascade") + 1] == "prefill"
        rank = Path(command[command.index("--output-dir") + 1]) / "rank_0"
        (rank / "bs_39280").mkdir(parents=True)
        (rank / "manifest.json").write_text(json.dumps(dict(backend="mte", ep_size=8)))
        (rank / "bs_39280/accuracy.json").write_text("[]")

    monkeypatch.setattr(runner, "run_logged", logged)
    assert runner.entrypoint(["--run-config", str(path), "--accuracy-stage", stage,
                              "--golden-dir", str(golden_dir)]) == 0
    assert calls == [stage]


@pytest.mark.parametrize("status", ["pass", "mismatch", "missing"])
def test_cascade_only_reads_existing_golden_and_never_calls_fused(tmp_path, monkeypatch, status):
    config = benchmark.load_config("gelu_smoke_k25", "3")
    config["model"].update(hidden=64, intermediate_hidden=64, num_experts=4, num_topk=2)
    args = SimpleNamespace(mode="accuracy", perf_scope="all", input="bf16", routing="balanced",
                           profile_memory_peak=False, backend="mte", timer=False, timer_ranks=None,
                           check_tokens=0, relative_rmse=0.02, cascade_only=True, golden_dir=tmp_path / "golden",
                           output_dir=tmp_path / "out")
    spec = golden.contract(config, 2, 3, "bf16", "balanced", 0)
    if status != "missing":
        expected, counts, selected = golden.generate(config, spec, [0], workers=1)[0]
        golden.save(args.golden_dir, spec, 0, expected, counts, selected)

    class Tensor:
        def __init__(self, value):
            self.value = value

        def to(self, **kw):
            return self

        def npu(self):
            return self

        cpu = float = npu

        def numpy(self):
            return self.value

    torch = SimpleNamespace(from_numpy=Tensor, bfloat16="bf16", npu=SimpleNamespace(synchronize=lambda: None))
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 2, all_gather=lambda obj, name: [obj] * 2, close=lambda: None)
    monkeypatch.setattr(validation, "StoreGroup", type(group))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(benchmark, "files", lambda package: tmp_path)
    (tmp_path / "_build_info.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(api, "SymmBuffer", lambda *a, **kw: pytest.fail("fused path reached"))
    monkeypatch.setattr(cascade, "init_process_group", lambda group: False if status != "missing" else pytest.fail("HCCL reached"))
    monkeypatch.setattr(prefill_cascade, "PrefillCascade", lambda *a, **kw: SimpleNamespace(
        forward_with_counts=lambda *a, **kw: (Tensor(expected if status == "pass" else expected + 10), Tensor(counts))))
    if status == "missing":
        with pytest.raises(ValueError, match="Golden"):
            benchmark.run(args, config, group)
        assert not args.golden_dir.exists()
        return
    if status == "mismatch":
        with pytest.raises(AssertionError, match="Cascade vs CPU"):
            benchmark.run(args, config, group)
    else:
        benchmark.run(args, config, group)
    root = args.output_dir / "rank_0"
    assert json.loads((root / "manifest.json").read_text())["device_communication"] == "hccl"
    report = json.loads((root / "bs_3/accuracy.json").read_text())[0]
    assert report["passed"] == (status == "pass") and report["implementation"] == "cascade"
    assert Path(report["golden_path"]).is_file()
