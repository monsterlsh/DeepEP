# Copyright (c) 2026, Lu Lu

import json

import pytest

from experiments.megamoe.scripts import l0_pod


NODES = ["192.0.2.1", "192.0.2.2", "192.0.2.7", "192.0.2.8"]


def job(tmp_path, **options):
    data = l0_pod.runner.read_run_config(l0_pod.runner.SCRIPT.parent / "configs/950pod_target.json")
    data.update(ep_size=32, master_addr=NODES[0], output_dir=str(tmp_path / "out"), **options)
    return data


def test_golden_finishes_everywhere_before_npu(tmp_path, monkeypatch):
    stages = []

    def capture(stage, scripts, logs, *, ssh_targets):
        assert ssh_targets == dict(enumerate(NODES[1:], start=1))
        assert set(scripts) == set(logs) == set(range(4))
        for rank, script in scripts.items():
            path = tmp_path / "configs" / f"out-{stage}-node{rank}.json"
            assert str(path) in script
            node = json.loads(path.read_text(encoding="utf-8"))
            assert node["node_rank"] == rank and node["dual_reference"]
            assert node["golden_only"] == (stage == "golden")
            assert node["skip_golden"] == (stage == "accuracy")
        stages.append(stage)

    monkeypatch.setattr(l0_pod, "run_stage", capture)
    config_dir = tmp_path / "configs"
    data = job(tmp_path, mode="accuracy", dual_reference=True)
    l0_pod.run_pod_job(data, config_dir, nodes=NODES, plan=True)
    assert not config_dir.exists() and not stages
    l0_pod.run_pod_job(data, config_dir, nodes=NODES)
    assert stages == ["golden", "accuracy"]


@pytest.mark.parametrize("options, expected, absent", [
    ({}, {"--profile"}, {"--profile-memory-peak", "--memory-only"}),
    ({"profile_memory_peak": True}, {"--profile", "--profile-memory-peak"}, {"--memory-only"}),
    ({"profile_memory_peak": False}, {"--profile", "--no-profile-memory-peak"}, {"--memory-only"}),
    ({"perf_scope": "memory", "profile_memory_peak": True}, {"--memory-only", "--profile-memory-peak"}, {"--profile"}),
])
def test_l04_pairs_profile_both_arms(tmp_path, monkeypatch, options, expected, absent):
    stages = []
    data = job(tmp_path, mode="perf", **options)

    def capture(name, commands, args, nodes):
        for rank, command in enumerate(commands):
            assert f"--node-rank={rank}" in command
            assert {"--nnodes=4", "--nproc-per-node=8"} | expected <= set(command)
            assert not absent.intersection(command)
            assert command[command.index("--bs") + 1] == "39280"
            assert command[command.index("--pairs") + 1] == "8"
            model = command[command.index("--config") + 1]
            assert json.loads(model.read_text(encoding="utf-8"))["model"] == data["config"]["model"]
        stages.append(name)

    monkeypatch.setattr(l0_pod, "_supervise", capture)
    l0_pod.run_pod_paired(data, tmp_path / "configs", nodes=NODES)
    assert stages == ["paired"]
    with pytest.raises(ValueError, match="A8W4"):
        l0_pod.run_pod_paired(dict(data, precision="a8w8"), tmp_path, nodes=NODES, plan=True)


def test_native_timer_rank_list(tmp_path, monkeypatch):
    cfgdir = tmp_path / "configs"
    data = job(tmp_path, mode="perf", timer=True, precision="a4w4")
    args = l0_pod._settings(data, cfgdir, NODES)
    out = tmp_path / "out"
    out.mkdir()
    args.output_dir = out
    commands = []
    monkeypatch.setattr(l0_pod.runner, "run_logged", lambda command, *a: commands.append(command))
    monkeypatch.setattr(l0_pod.runner, "pod_perf_report", lambda *a: "")
    l0_pod.runner.run_pod_case(args, "w4a4", args.python, {}, 0)
    command, = commands
    assert "--timer" in command and command[command.index("--timer-ranks") + 1] == ",".join(map(str, range(0, 32, 2)))
    with pytest.raises(ValueError, match="timer"):
        l0_pod.run_pod_job(dict(data, precision="a8w8"), cfgdir, nodes=NODES, plan=True)
    with pytest.raises(ValueError, match="master_addr"):
        l0_pod.run_pod_job(data, cfgdir, nodes=NODES[::-1], plan=True)


@pytest.mark.parametrize("ep", [8, 16])
def test_small_l04_pair_only_starts_required_nodes(tmp_path, monkeypatch, ep):
    data = dict(job(tmp_path, mode="perf"), ep_size=ep)
    nodes = NODES[:ep // 8]
    launches = []
    monkeypatch.setattr(l0_pod, "_supervise", lambda name, commands, args, hosts: launches.append((commands, hosts)))
    l0_pod.run_pod_paired(data, tmp_path / "configs", nodes=nodes)
    (commands, hosts), = launches
    assert hosts == nodes and len(commands) == len(nodes)
    for rank, cmd in enumerate(commands):
        assert {f"--nnodes={len(nodes)}", "--nproc-per-node=8", f"--node-rank={rank}"} <= set(cmd)
