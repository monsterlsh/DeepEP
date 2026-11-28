# Copyright (c) 2026, Lu Lu

"""Exercise real small-EP command construction and report discovery without an NPU."""

import json
from pathlib import Path
import sys

import pytest

from experiments.megamoe.scripts import run_megamoe as launcher
from experiments.megamoe.tests.accuracy_reports import direct_accuracy_row


@pytest.mark.parametrize("machine,ep,local,node", [
    ("ubx", 4, 4, 0), ("ubx", 8, 8, 0), ("ubx", 16, 16, 0),
    ("950pod", 4, 4, 0), ("950pod", 8, 8, 0), ("950pod", 16, 8, 1),
])
@pytest.mark.parametrize("mode", ["accuracy", "perf"])
def test_local_devices_workers_golden_and_trace_use_requested_ep(
    tmp_path, monkeypatch, capsys, machine, ep, local, node, mode,
):
    model = launcher.read_run_config(launcher.CONFIG_DIR / "target_gelu_k25.json")
    model["model"]["num_experts"] = 384
    model["bs_values"] = [96]
    job = dict(schema_version=1, machine=machine, mode=mode, precision="a8w4", ep_size=ep, backend="mte",
               config=model, output_dir=str(tmp_path / "out"), golden_dir=str(tmp_path / "golden"),
               master_addr="192.0.2.1", timer=mode == "perf", profile_memory_peak=False, check_tokens=1)
    if mode == "accuracy":
        job["dual_reference"] = True
        if ep != 4:
            job["accuracy_cascade"] = "decode"
            model["model"].update(activation="swiglu", num_topk=8, topk_weights_type=0)
    if machine == "950pod":
        job["node_rank"] = node
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    argv = ["--run-config", str(path)]
    assert launcher.entrypoint([*argv, "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    samples = list(range(0, ep, local // 4))
    assert (plan["ep_size"], plan["ranks_per_node"], plan["nnodes"]) == (ep, local, ep // local)
    assert plan["timer_ranks"] == (samples if mode == "perf" else [])
    assert not Path(job["output_dir"]).exists()

    monkeypatch.setenv("DEEPEP_950POD_BOOTSTRAPPED", "1")
    monkeypatch.setenv("UBX_CANN_LOADED", "test-cann")
    monkeypatch.delenv("UBX_CANN_SETUP", raising=False)
    monkeypatch.setenv("PYTHON", sys.executable)
    checked, launched, golden = [], [], []
    monkeypatch.setattr(launcher, "runtime_identity", lambda *a: None)
    first = node * local

    def check_device(command, *, env, check):
        script = launcher.DEVICE_CHECK if machine == "ubx" else launcher.POD_DEVICE_CHECK
        checked.append(int(command[command.index(script) + 1]))

    def logged(command, env, log):
        def option(name):
            return command[command.index(name) + 1]

        if "--module" not in command:
            assert "--accuracy-cascade" not in command
            assert option("--ep-size") == str(ep)
            golden.append(option("--ranks"))
            if "--summary" in command:
                summary = dict(config=model, identity="golden", missing=[])
                Path(option("--summary")).write_text(json.dumps(summary), encoding="utf-8")
            return
        assert {f"--nproc-per-node={local}", f"--nnodes={ep // local}"} <= set(command)
        assert option("--mode") == mode
        assert option("--input") == "fp8"
        if machine == "ubx":
            assert option("--backend") == "mte"
        if machine == "950pod":
            assert f"--node-rank={node}" in command
        if mode == "perf":
            assert option("--timer-ranks") == ",".join(map(str, samples))
            assert "--accuracy-cascade" not in command
        else:
            reference = job.get("accuracy_cascade", "prefill")
            assert "--dual-reference" in command and option("--accuracy-cascade") == reference
        spec = json.loads(Path(option("--config")).read_text(encoding="utf-8"))
        for rank in range(first, first + local):
            directory = Path(option("--output-dir")) / f"rank_{rank}"
            case = directory / "bs_96"
            case.mkdir(parents=True)
            manifest = dict(mode=mode, ep_size=ep, ranks_per_node=local, rank=rank, config=spec, backend="mte",
                            input="fp8", routing="balanced")
            if mode == "accuracy":
                manifest.update(dual_reference=True, accuracy_cascade=reference)
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            if mode == "perf":
                timing = dict(rank_max_mean_us=2, global_input_tokens_per_second=1)
                (case / "profiling.json").write_text(json.dumps(dict(timing, host_e2e=timing, windows_per_forward=1)))
            else:
                rows = [direct_accuracy_row(r, reference, 96, 1, spec["model"]["hidden"]) for r in range(ep)]
                (case / "accuracy.json").write_text(json.dumps(rows))
        launched.append(mode)

    monkeypatch.setattr(launcher.subprocess, "run", check_device)
    monkeypatch.setattr(launcher, "run_logged", logged)
    assert launcher.entrypoint(argv) == 0
    assert checked == [local] and launched == [mode]
    assert golden == ([f"{first}-{first + local - 1}"] if mode == "accuracy" else [])


@pytest.mark.parametrize("machine", ["ubx", "950pod"])
def test_ep4_rejects_800_experts_before_running(tmp_path, capsys, machine):
    model = launcher.read_run_config(launcher.CONFIG_DIR / "target_gelu_k25.json")
    path = tmp_path / "job.json"
    path.write_text(json.dumps(dict(schema_version=1, machine=machine, ep_size=4, mode="perf", precision="a8w4",
                                    master_addr="127.0.0.1", config=model)), encoding="utf-8")
    with pytest.raises(SystemExit):
        launcher.entrypoint(["--run-config", str(path), "--plan"])
    assert "128" in capsys.readouterr().err
