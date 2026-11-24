# Copyright (c) 2026, Lu Lu

from collections import Counter
import json

import pytest

from experiments.megamoe.scripts import acceptance_cases as matrix, run_megamoe


ROWS = {(row.case, row.machine, row.ep_size): row for row in matrix.cases()}


def test_matrix_covers_each_workload_once_per_topology():
    assert len(ROWS) == len(list(matrix.cases())) == 403
    for machine in matrix.MACHINES:
        for ep in (4, 8, 16):
            rows = [r for r in ROWS.values() if r.machine == machine and r.ep_size == ep]
            assert Counter(r.case[0] for r in rows) == {"R": 7, "C": 3, "K": 18, "L": 3, "D": 24}
            assert len({(r.bs, r.hidden, r.intermediate, r.experts, r.topk, r.shared, r.activation,
                         r.weight_mode, r.precision, r.groupsize) for r in rows}) == 55
    originals = [r for r in ROWS.values() if r.ep_size == 32 or
                 (r.case.startswith("C") and (r.machine, r.ep_size) == ("950pod", 8)) or
                 (r.case.startswith("K") and (r.machine, r.ep_size) == ("ubx", 16))]
    assert {r.case for r in originals} == {f"{family}{i:02}" for family, count in
                                           (("R", 7), ("C", 3), ("K", 54), ("L", 6), ("D", 24))
                                           for i in range(1, count + 1)}


@pytest.mark.parametrize("name,machine,ep,phase,expected", [
    ("D12", "950pod", 4, "perf", "groupsize=256"),
    ("D24", "ubx", 16, "perf", "groupsize=512"),
    ("L04", "950pod", 4, "full", "200"),
    ("L01", "ubx", 4, "full", "200"),
    ("R07", "950pod", 32, "perf", "1 GiB"),
    ("K44", "950pod", 32, "prefill", "1 GiB"),
    ("C03", "ubx", 4, "timer", "可用阶段"),
])
def test_unavailable_phase_reports_reason(name, machine, ep, phase, expected):
    with pytest.raises(ValueError, match=expected):
        matrix.run_config(ROWS[name, machine, ep], phase)


def test_cpu_golden_remains_available_for_current_full_batch_limit():
    row = ROWS["K53", "950pod", 32]
    assert row.phases() == ["golden", "golden-skewed"]
    cfg = matrix.run_config(row, "golden-skewed", results="/shared/round01/golden", node_rank=1)
    assert cfg["golden_only"] and cfg["routing"] == "skewed" and cfg["node_rank"] == 1
    assert cfg["config"]["model"]["num_topk"] == 32
    with pytest.raises(ValueError, match="--results"):
        matrix.run_config(row, "golden")


@pytest.mark.parametrize("name,machine,ep,phase", [
    ("R02", "ubx", 4, "perf"),
    ("R06", "950pod", 16, "memory"),
    ("R07", "950pod", 8, "perf"),
    ("C01", "ubx", 8, "decode"),
    ("C03", "950pod", 4, "decode"),
    ("K18", "ubx", 4, "accuracy-skewed"),
    ("K53", "950pod", 8, "golden-skewed"),
    ("K44", "950pod", 4, "prefill"),
    ("K54", "950pod", 16, "prefill"),
    ("K20", "ubx", 32, "timer"),
    ("L02", "ubx", 8, "prefill"),
    ("L04", "950pod", 16, "perf"),
])
def test_generated_config_through_public_plan(tmp_path, capsys, name, machine, ep, phase):
    path = tmp_path / "case.json"
    args = ["--case", name, "--ep-size", str(ep), "--machine", machine]
    matrix.main(["list", *args])
    listed, = json.loads(capsys.readouterr().out)
    assert phase in listed["phases"]
    matrix.main(["generate", *args, "--phase", phase, "--output", str(path), "--results", str(tmp_path / "job")])
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert cfg["ep_size"] == ep and cfg["precision"] == listed["precision"] and cfg["backend"] == listed["backend"]
    assert cfg["config"]["model"]["mma_type"] == matrix.MMA[listed["precision"]]
    if name == "R02":
        assert (cfg["config"]["seed"], cfg["config"]["warmup"], cfg["config"]["iters"]) == (20260922, 10, 20)
    if name == "R06":
        assert (cfg["config"]["seed"], cfg["config"]["warmup"], cfg["config"]["iters"]) == (1234, 2, 10)
    if name.startswith("L"):
        assert cfg["suite"] == "l0" and cfg["case"] == name
    if phase == "accuracy-skewed":
        assert cfg["dual_reference"] and cfg["check_tokens"] == 0 and cfg["routing"] == "skewed"
    assert run_megamoe.entrypoint(["--run-config", str(path), "--plan"]) == 0
    assert not (tmp_path / "job").exists()


def test_generation_preserves_existing_config(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("existing")
    with pytest.raises(SystemExit):
        matrix.main(["generate", "--case", "C01", "--machine", "ubx", "--ep-size", "4", "--output", str(path)])
    assert path.read_text() == "existing"
