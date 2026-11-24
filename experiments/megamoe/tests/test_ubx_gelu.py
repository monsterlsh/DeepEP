# Copyright (c) 2026, Lu Lu

import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from deep_ep_experimental.megamoe.benchmark import default_timer_ranks
from experiments.megamoe.scripts import run_megamoe as launcher
from experiments.megamoe.scripts import ubx_device_check


@pytest.fixture(autouse=True)
def clean_launcher_environment(monkeypatch):
    for name in (*launcher.FORWARDED_ENV, "UBX_CANN_LOADED", "MASTER_ADDR", "MASTER_PORT", "REMOTE_USER", "REMOTE_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHON", sys.executable)
    monkeypatch.setenv("UBX_CANN_LOADED", "test-cann")


def test_default_urma_timer_ranks_cover_four_per_node():
    assert default_timer_ranks(16, 16, urma=True) == [0, 4, 8, 12]
    assert default_timer_ranks(32, 16, urma=True) == [0, 4, 8, 12, 16, 20, 24, 28]
    assert default_timer_ranks(32, 8, urma=False) == list(range(8))


@pytest.mark.parametrize(("auto_python", "json_python", "peer_default"), [
    (True, False, True), (False, False, False), (True, True, False),
])
def test_ep32_peer_resolves_automatic_python_locally(tmp_path, monkeypatch, auto_python, json_python, peer_default):
    if auto_python:
        monkeypatch.setenv("UBX_DEFAULT_PYTHON", sys.executable)
    else:
        monkeypatch.delenv("UBX_DEFAULT_PYTHON", raising=False)
    cfg = launcher.read_run_config(launcher.SCRIPT.parent / "configs/ubx_ep32_target.json")
    cfg.update(mode="perf", precision="a8w4", output_dir=str(tmp_path))
    if json_python:
        cfg["python"] = sys.executable
    path = tmp_path / "job.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    peers = []

    def capture(stage, scripts, logs, **ssh):
        assert "--peer-default" not in scripts[0]
        peers.append("--peer-default" in scripts[1])

    monkeypatch.setattr(launcher, "run_stage", capture)
    assert launcher.entrypoint(["--run-config", str(path)]) == 0
    assert peers == [peer_default]


def test_timer_rank_override_is_rejected(capsys):
    with pytest.raises(SystemExit) as error:
        launcher.cli(["--mode", "perf", "--timer", "--timer-ranks=0"])
    assert error.value.code == 2
    assert "unrecognized arguments: --timer-ranks=0" in capsys.readouterr().err


@pytest.mark.parametrize(
    "ep_size, precision, configured, expected, require_w8a8, expected_timer_ranks",
    [
        (16, "w8a8", "fp8xfp4", "fp8xfp8", "1", "0,4,8,12"),
        (32, "w8a8", "fp8xfp4", "fp8xfp8", "1", "0,4,8,12,16,20,24,28"),
        (32, "w4a8", "fp8xfp8", "fp8xfp4", "0", None),
        (32, "w4a8", None, "fp8xfp4", "0", None),
    ],
)
def test_perf_uses_selected_precision_and_fixed_timer_ranks(
    tmp_path, monkeypatch, ep_size, precision, configured, expected, require_w8a8, expected_timer_ranks,
):
    spec = json.loads((launcher.CONFIG_DIR / "target_gelu_k25.json").read_text(encoding="utf-8"))
    if configured is None:
        del spec["model"]["mma_type"]
    else:
        spec["model"]["mma_type"] = configured
    output = tmp_path / "runs"
    master_addr = "192.0.2.1" if ep_size == 32 else "127.0.0.2"
    run = dict(schema_version=1, machine="ubx", backend="urma", mode="perf", precision=precision,
               config=spec, ep_size=ep_size, master_addr=master_addr, output_dir=str(output),
               timer=expected_timer_ranks is not None)
    if ep_size == 32:
        run.update(node_rank=1, stage="run")
    config = tmp_path / "run.json"
    config.write_text(json.dumps(run), encoding="utf-8")
    checks, launches = [], []

    def check_device(command, *, env, check):
        assert command[:3] == [sys.executable, "-P", launcher.DEVICE_CHECK]
        assert check is True
        checks.append(command[3:])

    def run_logged(command, env, log):
        assert env["DEEPEP_SHMEM_ENDPOINT"] == f"tcp://{master_addr}:29180"
        if ep_size == 32:
            assert f"--master-addr={master_addr}" in command
        else:
            assert f"--rdzv-endpoint={master_addr}:0" in command and f"--local-addr={master_addr}" in command
        launched_config = Path(command[command.index("--config") + 1])
        launched_spec = json.loads(launched_config.read_text(encoding="utf-8"))
        launches.append(launched_spec["model"]["mma_type"])
        assert ("--timer" in command) == (expected_timer_ranks is not None)
        if expected_timer_ranks is not None:
            assert command[command.index("--timer-ranks") + 1] == expected_timer_ranks
        else:
            assert "--timer-ranks" not in command
        rank_dir = Path(command[command.index("--output-dir") + 1]) / f"rank_{ep_size - 16}"
        batch_dir = rank_dir / "bs_39280"
        batch_dir.mkdir(parents=True)
        manifest = dict(backend="urma", ep_size=ep_size, config=launched_spec)
        (rank_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        host_e2e = dict(rank_max_mean_us=1, global_input_tokens_per_second=1)
        perf = dict(host_e2e=host_e2e, windows_per_forward=1, rank_max_mean_us=1)
        (batch_dir / "performance.json").write_text(json.dumps(perf), encoding="utf-8")

    monkeypatch.setattr(launcher.subprocess, "run", check_device)
    monkeypatch.setattr(launcher, "run_logged", run_logged)
    assert launcher.entrypoint(["--run-config", str(config)]) == 0
    assert checks == [["16", require_w8a8]]
    assert launches == [expected]
    assert json.loads(config.read_text(encoding="utf-8")) == run


def test_ep32_auto_waits_for_both_golden_preparations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOLDEN_DIR", "relative golden")
    config = tmp_path / "relative model.json"
    config.write_bytes((launcher.CONFIG_DIR / "target_gelu_k25.json").read_bytes())
    output = tmp_path / "relative output"
    events = []

    def prepare_golden(python, env, args, precision, stage_config):
        assert args.force_cpu
        assert not args.timer
        assert env["GOLDEN_DIR"] == str(tmp_path / "relative golden")
        assert stage_config.is_absolute()
        events.append(("prepare", precision, args.node_rank))

    def run_case(python, env, args, mode, precision, stage_config):
        if mode == "accuracy":
            assert ("prepare", precision, 0) in events and ("prepare", precision, 1) in events
        assert not args.force_cpu
        assert args.timer
        events.append((mode, precision, args.node_rank))

    def run_stage(label, scripts, logs, *, ssh_target, ssh_port):
        assert ssh_target == "tester@192.0.2.2"
        assert ssh_port == 2222
        assert logs == {rank: output / f"launch-{label}-node{rank}.log" for rank in (0, 1)}
        for rank, script in scripts.items():
            argv = shlex.split(script.splitlines()[-1])[2:]
            assert argv[argv.index("--node-rank") + 1] == str(rank)
            assert argv[argv.index("--config") + 1] == str(config)
            assert argv[argv.index("--output-dir") + 1] == str(output)
            assert "--timer-ranks" not in argv
            assert ("--timer" in argv) == (argv[argv.index("--stage") + 1] == "run")
            assert argv[1] == "--worker"
            assert launcher.cli(argv[2:]) == 0

    monkeypatch.setattr(launcher, "prepare_golden", prepare_golden)
    monkeypatch.setattr(launcher, "run_case", run_case)
    monkeypatch.setattr(launcher, "run_stage", run_stage)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: None)
    assert launcher.cli([
        "both", "all", "--ep-size", "32", "--node-ip", "192.0.2.1", "192.0.2.2",
        "--ssh-user", "tester", "--ssh-port", "2222", "--force", "--timer",
        "--config", config.name, "--output-dir", output.name,
    ]) == 0
    assert events == [
        (mode, precision, rank)
        for mode, precision in (
            ("prepare", "w4a8"), ("accuracy", "w4a8"), ("prepare", "w8a8"), ("accuracy", "w8a8"),
            ("perf", "w4a8"), ("perf", "w8a8"),
        )
        for rank in (0, 1)
    ]


@pytest.mark.parametrize(("argv", "message"), [
    (["perf", "--ep-size", "16"], "需要 --ep-size 32"),
    (["perf", "w4a4", "--input", "fp8"], "W4A4 需要 fp4"),
    (["perf", "--bs", "65537"], "--bs 必须在 1..65536"),
    (["compare", "--plan"], "A/B 参数适用于 URMA"),
])
def test_mte_roce_rejects_unsuitable_cases(argv, message, capsys):
    with pytest.raises(SystemExit) as error:
        launcher.parse_args(["--backend", "mte_roce", "--ep-size", "32", "--node-ip", "192.0.2.1", "192.0.2.2", *argv])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("timer", [False, True])
@pytest.mark.parametrize(("precision", "input_kind"), [("w4a8", "fp8"), ("w4a4", "fp4")])
def test_mte_roce_forwards_explicit_backend_and_reads_mte_reports(tmp_path, monkeypatch, capsys, timer, precision, input_kind):
    stages, checks, launches = [], [], []

    def run_stage(label, scripts, logs, **ssh):
        stages.append(label)
        for script in scripts.values():
            argv = shlex.split(script.splitlines()[-1])[3:]
            assert argv[argv.index("--backend") + 1] == "mte_roce"
            assert argv[0] == "--worker"
            assert launcher.cli(argv[1:]) == 0

    def check_device(command, *, env, check):
        assert command[2] == launcher.DEVICE_CHECK and check
        checks.append(command[3:])

    def run_logged(command, env, log):
        def option(name):
            return command[command.index(name) + 1]

        assert option("--bs-values") == "32" and option("--input") == input_kind
        spec = json.loads(Path(option("--config")).read_text(encoding="utf-8"))
        model = spec["model"]
        assert (model["activation"], model["mma_type"], model["topk_weights_type"]) == (
            "swiglu", launcher.MMA_TYPES[precision], 0
        )
        assert model["num_shared_experts"] == 0
        if "--module" not in command:
            launches.append(("prepare", option("--ranks")))
            Path(option("--summary")).write_text("{}", encoding="utf-8")
            return
        assert option("--backend") == "mte_roce"
        assert ("--timer" in command) == timer
        if timer:
            assert option("--timer-ranks") == "0,4,8,12,16,20,24,28"
        rank = next(int(arg.partition("=")[2]) * 16 for arg in command if arg.startswith("--node-rank="))
        mode = option("--mode")
        launches.append((mode, rank))
        rank_dir = Path(option("--output-dir")) / f"rank_{rank}"
        batch_dir = rank_dir / "bs_32"
        batch_dir.mkdir(parents=True)
        manifest = dict(backend="mte_roce", ep_size=32, config=spec)
        (rank_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        if mode == "perf":
            host_e2e = dict(rank_max_mean_us=3000, global_input_tokens_per_second=341333)
            perf = dict(host_e2e=host_e2e, windows_per_forward=1, rank_max_mean_us=2000)
            (batch_dir / "profiling.json").write_text(json.dumps(perf), encoding="utf-8")

    monkeypatch.setattr(launcher, "run_stage", run_stage)
    monkeypatch.setattr(launcher.subprocess, "run", check_device)
    monkeypatch.setattr(launcher, "run_logged", run_logged)
    args = ["both", precision, "--backend", "mte_roce", "--ep-size", "32", "--output-dir", str(tmp_path),
            "--node-ip", "192.0.2.1", "192.0.2.2"]
    if timer:
        args.append("--timer")
    assert launcher.cli(args) == 0
    assert stages == [f"prepare-accuracy-{precision}", f"run-accuracy-{precision}", f"run-perf-{precision}"]
    base = ["16", "0", "mte_roce", precision]
    assert checks == [base] * 2 + [base + (["1"] if timer else [])] * 4
    assert launches == [
        ("prepare", "0-15"), ("prepare", "16-31"), ("accuracy", 0), ("accuracy", 16), ("perf", 0), ("perf", 16),
    ]
    assert "profiling.json" in capsys.readouterr().out


def test_mte_roce_rejects_auto_detected_urma_report(tmp_path):
    rank_dir = tmp_path / "rank_0"
    rank_dir.mkdir()
    (rank_dir / "manifest.json").write_text('{"backend": "urma", "ep_size": 32}', encoding="utf-8")
    args = launcher.parse_args(["perf", "--backend", "mte_roce", "--ep-size", "32", "--node-rank", "0", "--stage", "run",
                                "--master-addr", "192.0.2.1", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="预期 mte_roce EP32"):
        launcher.report_lines(tmp_path, args, "perf", "w4a8")


def test_mte_roce_accepts_bs96_shared_fp8_config(tmp_path):
    from deep_ep_experimental.megamoe.benchmark import load_config

    args = launcher.parse_args([
        "perf", "--backend", "mte_roce", "--ep-size", "32", "--config", "ep32_roce_bs96",
        "--bs", "96", "--output-dir", str(tmp_path), "--node-ip", "192.0.2.1", "192.0.2.2",
    ])
    config = load_config(str(args.config), str(args.bs))
    assert config["bs_values"] == [96] and config["shared_dtype"] == "fp8"
    assert config["model"]["num_shared_experts"] == 1
    assert config["model"]["activation_clamp"] == 10


@pytest.mark.parametrize(("ep_size", "backend"), [(16, "mte"), (32, "mte_roce")])
@pytest.mark.parametrize(("suffix", "activation", "weight_mode"), [("", "gelu", 1), ("_swiglu", "swiglu", 0)])
def test_target_run_json_covers_three_precisions(tmp_path, ep_size, backend, suffix, activation, weight_mode):
    path = launcher.SCRIPT.parent / "configs" / f"ubx_ep{ep_size}_target{suffix}.json"
    args = launcher.parse_args(["--run-config", str(path), "--output-dir", str(tmp_path)])
    assert (args.ep_size, args.backend, args.bs) == (ep_size, backend, 39280)
    assert args.accuracy_precisions == args.perf_precisions == ("w4a8", "w8a8", "w4a4")
    for precision in args.accuracy_precisions:
        config_path = launcher.config_for(args, precision)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = config["model"]
        assert (model["num_experts"], model["num_topk"], model["hidden"], model["intermediate_hidden"]) == (800, 25, 4096, 1536)
        assert model["mma_type"] == launcher.MMA_TYPES[precision]
        assert (model["activation"], model["topk_weights_type"]) == (activation, weight_mode)
        forwarded = launcher.case_args(args, precision, config_path)
        assert forwarded[forwarded.index("--input") + 1] == ("fp4" if precision == "w4a4" else "fp8")


@pytest.mark.parametrize("suffix", ["", "_swiglu"])
def test_ubx_ep16_a4w4_launches_fp4_timer(tmp_path, monkeypatch, suffix):
    preset = launcher.SCRIPT.parent / "configs" / f"ubx_ep16_target{suffix}.json"
    data = launcher.read_run_config(preset)
    data.update(mode="perf", precision="a4w4", timer=True, output_dir="run")
    source = tmp_path / "a4w4.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    checks, launches = [], []

    def run_logged(command, env, log):
        assert command[command.index("--input") + 1] == "fp4"
        assert command[command.index("--backend") + 1] == "mte"
        assert "--timer" in command
        assert command[command.index("--timer-ranks") + 1] == "0,4,8,12"
        config = Path(command[command.index("--config") + 1])
        assert json.loads(config.read_text(encoding="utf-8"))["model"] == dict(
            data["config"]["model"], mma_type="fp4xfp4"
        )
        launches.append(True)

    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: checks.append(command[3:]))
    monkeypatch.setattr(launcher, "run_logged", run_logged)
    monkeypatch.setattr(launcher, "report_lines", lambda *args: ())
    assert launcher.entrypoint(["--run-config", str(source)]) == 0
    assert checks == [["16", "0", "mte", "w4a4", "1"]]
    assert launches == [True]


def test_ubx_ep16_a4w4_timer_requires_instrumented_wheel(monkeypatch):
    monkeypatch.setattr(ubx_device_check, "check_wheel_packages", lambda: None)
    npu = SimpleNamespace(device_count=lambda: 16, set_device=lambda rank: None)
    ext = SimpleNamespace(mte_a4w4_enabled=True, timer_enabled=True, __file__="mock-extension")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=npu))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "deep_ep._machine",
                        SimpleNamespace(physical_device=lambda: 0, board_id=lambda rank: 0x44))
    monkeypatch.setitem(sys.modules, "deep_ep_experimental.megamoe.api", SimpleNamespace(_extension=lambda: ext))
    monkeypatch.setattr(sys, "argv", ["-c", "16", "0", "mte", "w4a4", "1"])
    with pytest.raises(SystemExit, match="单机 MTE W4A4 设备 timer"):
        ubx_device_check.main()
    ext.mte_a4w4_timer_enabled = True
    ubx_device_check.main()


def test_run_json_forwards_resolved_settings_and_accepts_cli_overrides(tmp_path, monkeypatch):
    model_path = tmp_path / "model.json"
    model_path.write_bytes((launcher.CONFIG_DIR / "target_gelu_k25.json").read_bytes())
    source = tmp_path / "launch.json"
    source.write_text(json.dumps(dict(
        schema_version=1, ep_size=32, backend="mte_roce", mode="both", precision="all",
        activation="swiglu", weighting="late", config="model.json", bs=39280, output_dir="out",
        golden_dir="golden storage", golden_workers=5, routing="skewed", check_tokens=3,
        node_ip=["192.0.2.1", "192.0.2.2"], shmem_port=29801, master_port=29800,
    )), encoding="utf-8")
    stages = []

    def run_stage(label, scripts, log_paths, *, ssh_target, ssh_port):
        stages.append(label)
        assert ssh_target.endswith("@192.0.2.2")
        for node, script in scripts.items():
            option = shlex.split(script.splitlines()[-1])[3:]
            assert option[option.index("--config") + 1] == str(model_path)
            assert option[option.index("--output-dir") + 1] == str(tmp_path / "out")
            assert option[option.index("--node-rank") + 1] == str(node)
            assert option[option.index("--activation") + 1] == "swiglu"
            assert option[option.index("--weighting") + 1] == "late"
            assert option[option.index("--bs") + 1] == "17"
            assert "--run-config" not in option
            assert "DEEPEP_SHMEM_ENDPOINT=tcp://192.0.2.1:29801" in script
            assert "ROUTING=skewed" in script and "GOLDEN_WORKERS=5" in script
            assert str(tmp_path / "golden storage") in script

    monkeypatch.setattr(launcher, "run_stage", run_stage)
    assert launcher.cli(["perf", "a4w4", "--run-config", str(source), "--bs", "17"]) == 0
    assert stages == ["run-perf-w4a4"]


def test_inline_model_reaches_both_ubx_nodes_and_950pod(tmp_path, monkeypatch):
    from deep_ep_experimental.megamoe.benchmark import load_config

    preset = launcher.SCRIPT.parent / "configs/ubx_ep32_target.json"
    run = json.loads(preset.read_text(encoding="utf-8"))
    run.update(mode="perf", precision="a8w4", output_dir="ubx",
               node_ip=["192.0.2.11", "192.0.2.12"], master_addr="192.0.2.13", shmem_port=29811)
    spec = run["config"]
    spec["model"].update(hidden=3584, intermediate_hidden=2048, num_experts=768, num_topk=12)
    spec["bs_values"] = [257]
    source = tmp_path / "custom run.json"
    source.write_text(json.dumps(run), encoding="utf-8")
    workers = []

    def run_stage(label, scripts, logs, *, ssh_target, ssh_port):
        assert label == "run-perf-w4a8" and ssh_target.endswith("@192.0.2.12")
        for script in scripts.values():
            argv = shlex.split(script.splitlines()[-1])[3:]
            assert launcher.entrypoint(argv) == 0

    def run_case(python, env, args, mode, precision, config):
        assert args.ep_size == 32 and args.backend == "mte_roce" and args.bs == 257
        assert args.master_addr == "192.0.2.13" and env["DEEPEP_SHMEM_ENDPOINT"] == "tcp://192.0.2.13:29811"
        assert load_config(str(config)) == spec
        workers.append(args.node_rank)

    monkeypatch.setattr(launcher, "run_stage", run_stage)
    monkeypatch.setattr(launcher, "run_case", run_case)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setenv("MASTER_ADDR", "192.0.2.99")
    monkeypatch.setenv("DEEPEP_SHMEM_ENDPOINT", "tcp://192.0.2.99:29999")
    assert launcher.entrypoint(["--run-config", str(source)]) == 0
    assert workers == [0, 1]

    pod_stages = []

    def run_logged(command, env, log):
        assert env["DEEPEP_SHMEM_ENDPOINT"] == "tcp://192.0.2.14:29180"
        config = command[command.index("--config") + 1]
        assert command[command.index("--bs-values") + 1] == "257"
        assert load_config(config) == dict(spec, model=dict(spec["model"], mma_type="fp8xfp8"))
        if "--module" in command:
            assert "--nproc-per-node=8" in command and "--master-addr=192.0.2.14" in command
            assert command[command.index("--mode") + 1] == "accuracy"
            rank = Path(command[command.index("--output-dir") + 1]) / "rank_0"
            reports = rank / "bs_257"
            reports.mkdir(parents=True)
            (rank / "manifest.json").write_text('{"backend": "mte", "ep_size": 8}', encoding="utf-8")
            (reports / "accuracy.json").write_text("{}", encoding="utf-8")
            pod_stages.append("accuracy")
        else:
            assert command[command.index("--ep-size") + 1] == "8"
            assert command[command.index("--ranks") + 1] == "0-7"
            pod_stages.append("golden")

    monkeypatch.setenv("DEEPEP_950POD_BOOTSTRAPPED", "1")
    monkeypatch.delenv("SHMEM_PORT", raising=False)
    monkeypatch.setattr(launcher, "run_logged", run_logged)
    preset = launcher.SCRIPT.parent / "configs/950pod_target.json"
    pod = json.loads(preset.read_text(encoding="utf-8"))
    pod.update(config=spec, precision="a8w8", check_tokens=8, output_dir="pod", master_addr="192.0.2.14")
    source.write_text(json.dumps(pod), encoding="utf-8")
    assert launcher.entrypoint(["--run-config", str(source)]) == 0
    assert pod_stages == ["golden", "accuracy"]


@pytest.mark.parametrize(("machine", "ep_size", "cli_flag"), [("ubx", 16, True), ("ubx", 32, False), ("950pod", 32, True)])
@pytest.mark.parametrize(("stage", "model_config", "activation", "precision"), [
    ("decode", "ep32_roce_decode", "swiglu", "w8a8"),
    ("decode", "ep32_roce_decode", "gelu", "w8a8"),
    ("decode", "ep32_roce_decode", "swiglu", "w4a4"),
    ("decode", "ep32_roce_decode", "gelu", "w4a4"),
    ("prefill", "target_gelu_k25", "gelu", "w8a8"),
    ("prefill", "target_gelu_k25", "gelu", "w4a4"),
    ("prefill", "ep32_roce_decode", "swiglu", "w4a4"),
])
def test_cascade_comparison_reaches_workers_and_reads_reports(
    tmp_path, monkeypatch, capsys, machine, ep_size, cli_flag, stage, model_config, activation, precision,
):
    spec = json.loads((launcher.CONFIG_DIR / f"{model_config}.json").read_text(encoding="utf-8"))
    spec["model"]["activation"] = activation
    bs, = spec["bs_values"]
    backend = "mte_roce" if machine == "ubx" and ep_size == 32 else "mte"
    data = dict(schema_version=1, machine=machine, ep_size=ep_size, backend=backend, mode="perf",
                precision=precision,
                config=spec, output_dir="run", master_addr="192.0.2.1")
    input_kind = "fp4" if precision == "w4a4" else "fp8"
    stage_field = f"compare_{stage}_cascade"
    stage_option = f"--compare-{stage}-cascade"
    if not cli_flag:
        data[stage_field] = True
    if backend == "mte_roce":
        data["node_ip"] = ["192.0.2.1", "192.0.2.2"]
    if machine == "950pod":
        data["node_rank"] = 1
        spec["model"]["topk_weights_type"] = 0
    source = tmp_path / "run.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    launched = []

    def run_logged(command, env, log):
        assert stage_option in command and "--timer" not in command
        assert "--dual-reference" not in command and "--golden-dir" not in command
        assert command[command.index("--input") + 1] == input_kind
        output = Path(command[command.index("--output-dir") + 1])
        config = json.loads(Path(command[command.index("--config") + 1]).read_text(encoding="utf-8"))
        assert config["model"]["activation"] == activation
        assert config["model"]["mma_type"] == launcher.MMA_TYPES[precision]
        assert config["model"]["topk_weights_type"] == spec["model"]["topk_weights_type"]
        assert command[command.index("--bs-values") + 1] == str(bs)
        ranks_per_node = 8 if machine == "950pod" else 16
        node = next((int(opt.partition("=")[2]) for opt in command if opt.startswith("--node-rank=")), 0)
        for rank in range(node * ranks_per_node, (node + 1) * ranks_per_node):
            rank_dir = output / f"rank_{rank}"
            case = rank_dir / f"bs_{bs}"
            case.mkdir(parents=True)
            manifest = dict(backend=backend, mode="perf", ep_size=ep_size, rank=rank, ranks_per_node=ranks_per_node,
                            routing="balanced", input=input_kind, config=config)
            manifest[stage_field] = True
            (rank_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = {
                arm: {key: {"rank_max_mean_us": value} for key in ("kernel_sum", "device_span", "host_e2e")}
                for arm, value in (("fused", 25), (f"{stage}_cascade", 50))
            }
            (case / f"{stage}_comparison.json").write_text(json.dumps(report), encoding="utf-8")
        launched.append(node)

    def run_stage(label, scripts, logs, **ssh):
        assert label == f"run-perf-{precision}"
        for script in scripts.values():
            worker_args = shlex.split(script.splitlines()[-1])[3:]
            assert launcher.entrypoint(worker_args) == 0

    monkeypatch.setenv("DEEPEP_950POD_BOOTSTRAPPED", "1")
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher, "run_logged", run_logged)
    monkeypatch.setattr(launcher, "run_stage", run_stage)
    argv = ["--run-config", str(source)] + ([stage_option] if cli_flag else [])
    assert launcher.entrypoint(argv) == 0
    assert launched == ([1] if machine == "950pod" else [0, 1] if ep_size == 32 else [0])
    printed = capsys.readouterr().out
    assert "融合 25.000 us，级联 50.000 us，融合加速 2.000x" in printed
    assert f"{stage}_comparison.json" in printed


@pytest.mark.parametrize(("args", "message"), [
    (["--mode", "accuracy"], "mode=perf"),
    (["--mode", "perf", "--timer"], "timer"),
    (["--mode", "perf", "--weighting", "early"], "Finalize"),
    (["--mode", "perf", "--bs", "513"], "BS <= 512"),
    (["--mode", "perf", "--topk", "25"], "TopK <= 16"),
])
def test_decode_comparison_rejects_incomparable_cases(args, message, capsys):
    with pytest.raises(SystemExit, match="2"):
        launcher.parse_args(["--backend", "mte", "--config", "ep32_roce_decode", "--compare-decode-cascade", *args])
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(("extra_args", "message"), [
    (["--precision", "a8w4", "--weighting", "late"], "支持 A8W8 和 A4W4"),
    (["--precision", "a8w8"], "Finalize"),
])
def test_decode_gelu_requires_matching_operands_and_finalize_weights(extra_args, message, capsys):
    with pytest.raises(SystemExit, match="2"):
        launcher.parse_args([
            "--mode", "perf", "--backend", "mte", "--config", "ep32_roce_decode",
            "--compare-decode-cascade", "--activation", "gelu", *extra_args,
        ])
    assert message in capsys.readouterr().err


def test_prefill_comparison_accepts_swiglu_and_rejects_decode_flag(capsys):
    args = ["--mode", "perf", "--backend", "mte", "--config", "ep32_roce_decode", "--compare-prefill-cascade"]
    assert launcher.parse_args(args).cascade_stage == "prefill"
    with pytest.raises(SystemExit, match="2"):
        launcher.parse_args([*args, "--compare-decode-cascade"])
    assert "not allowed with argument" in capsys.readouterr().err


def test_prefill_comparison_fp4_respects_native_backend_limits(tmp_path, capsys):
    flags = ["--mode", "perf", "--config", "target_gelu_k25", "--compare-prefill-cascade"]
    with pytest.raises(SystemExit, match="2"):
        launcher.parse_args([*flags, "--backend", "urma", "--precision", "a4w4"])
    assert "URMA 暂不支持 W4A4" in capsys.readouterr().err
    assert launcher.parse_args([*flags, "--backend", "urma", "--precision", "all"]).perf_precisions == ("w4a8", "w8a8")
    assert launcher.parse_args([*flags, "--backend", "mte", "--precision", "all"]).perf_precisions == (
        "w4a8", "w8a8", "w4a4",
    )

    preset = launcher.SCRIPT.parent / "configs/950pod_target.json"
    data = launcher.read_run_config(preset)
    data.update(mode="perf", precision="a4w4", compare_prefill_cascade=True)
    assert launcher.pod_settings(tmp_path / "run.json", data).precisions == ("w4a4",)


@pytest.mark.parametrize("fault", [None, "manifest", "kernel", "host", "throughput", "missing"])
def test_950pod_ep32_perf_only(tmp_path, monkeypatch, capsys, fault):
    preset = launcher.SCRIPT.parent / "configs/950pod_target.json"
    data = json.loads(preset.read_text(encoding="utf-8"))
    data.update(mode="perf", precision="a8w4", ep_size=32, output_dir="pod-perf", dual_reference=True)
    data["config"]["bs_values"] = [257]
    monkeypatch.setenv("DEEPEP_950POD_BOOTSTRAPPED", "1")
    checked, launched = [], []

    def check_device(command, *, env, check):
        assert command[-2:] == [launcher.POD_DEVICE_CHECK, "8"] and check
        checked.append(True)

    def run_logged(command, env, log):
        node = len(launched)
        assert env["DEEPEP_SHMEM_ENDPOINT"] == "tcp://177.22.21.1:29180"
        assert {"--nnodes=4", "--nproc-per-node=8", f"--node-rank={node}", "--max-restarts=0"} <= set(command)
        assert command[command.index("--module") + 1] == f"{launcher.MODULE}.benchmark"
        assert command[command.index("--mode") + 1] == "perf"
        assert not {"--golden-dir", "--check-tokens", "--dual-reference", "--backend"} & set(command)
        output = Path(command[command.index("--output-dir") + 1])
        assert output == tmp_path / "pod-perf/perf"
        config = json.loads(Path(command[command.index("--config") + 1]).read_text(encoding="utf-8"))
        for rank in range(node * 8, node * 8 + 8):
            rank_dir = output / f"rank_{rank}"
            batch = rank_dir / "bs_257"
            batch.mkdir(parents=True)
            manifest = dict(backend="mte", mode="perf", rank=rank, ep_size=32, ranks_per_node=8,
                            config=config, routing="balanced", input="fp8")
            host = dict(rank_max_mean_us=3200.0, global_input_tokens_per_second=2570000.0)
            report = dict(rank_max_mean_us=2800.0, host_e2e=host)
            if rank == 7:
                if fault == "manifest":
                    manifest["mode"] = "accuracy"
                elif fault == "kernel":
                    report["rank_max_mean_us"] = 0
                elif fault == "host":
                    host["rank_max_mean_us"] = float("nan")
                elif fault == "throughput":
                    host["global_input_tokens_per_second"] = -1
            (rank_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            if (rank, fault) != (7, "missing"):
                (batch / "profiling.json").write_text(json.dumps(report), encoding="utf-8")
        launched.append(node)

    monkeypatch.setattr(launcher.subprocess, "run", check_device)
    monkeypatch.setattr(launcher, "run_logged", run_logged)
    for node in range(4 if fault is None else 1):
        source = tmp_path / f"node{node}.json"
        source.write_text(json.dumps(dict(data, node_rank=node)), encoding="utf-8")
        if fault is None:
            assert launcher.entrypoint(["--run-config", str(source)]) == 0
        else:
            with pytest.raises(SystemExit, match="2"):
                launcher.entrypoint(["--run-config", str(source)])
            assert "rank_7" in capsys.readouterr().err
    if fault is None:
        assert launched == [0, 1, 2, 3] and checked == [True] * 4
        assert "3.200 ms" in capsys.readouterr().out
        assert len(list((tmp_path / "pod-perf").glob("perf/rank_*/bs_257/profiling.json"))) == 32


@pytest.mark.parametrize("mode", ["accuracy", "perf"])
@pytest.mark.parametrize(("activation", "weight_mode"), [("gelu", 1), ("swiglu", 0)])
@pytest.mark.parametrize("ep_size", [8, 16, 32])
def test_950pod_a4w4_runs_fp4(tmp_path, monkeypatch, mode, activation, weight_mode, ep_size):
    preset = launcher.SCRIPT.parent / "configs/950pod_target.json"
    data = launcher.read_run_config(preset)
    data.update(mode=mode, precision="a4w4", ep_size=ep_size, output_dir="out", check_tokens=8)
    data["config"]["model"].update(activation=activation, topk_weights_type=weight_mode)
    source = tmp_path / "run.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("DEEPEP_950POD_BOOTSTRAPPED", "1")
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: None)
    launched = []

    def run_logged(command, env, log):
        assert command[command.index("--input") + 1] == "fp4"
        spec = json.loads(Path(command[command.index("--config") + 1]).read_text(encoding="utf-8"))
        assert spec["model"] == dict(data["config"]["model"], mma_type="fp4xfp4")
        if "--module" not in command:
            assert "--backend" not in command
            assert command[command.index("--ranks") + 1] == "0-7"
            launched.append("golden")
            return
        assert {f"--nnodes={ep_size // 8}", "--nproc-per-node=8", "--node-rank=0"} <= set(command)
        assert command[command.index("--backend") + 1] == "mte"
        assert command[command.index("--mode") + 1] == mode
        launched.append(mode)
        if mode == "accuracy":
            rank = Path(command[command.index("--output-dir") + 1]) / "rank_0"
            (rank / "bs_39280").mkdir(parents=True)
            (rank / "manifest.json").write_text(json.dumps(dict(backend="mte", ep_size=ep_size)), encoding="utf-8")
            (rank / "bs_39280/accuracy.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(launcher, "run_logged", run_logged)
    monkeypatch.setattr(launcher, "pod_perf_report", lambda *args: "performance report")
    assert launcher.entrypoint(["--run-config", str(source)]) == 0
    assert launched == (["golden", "accuracy"] if mode == "accuracy" else ["perf"])


@pytest.mark.parametrize("activation", ["gelu", "swiglu"])
@pytest.mark.parametrize("ep_size", [8, 16, 32])
def test_950pod_all_includes_a4w4_for_routed_experts(tmp_path, activation, ep_size):
    data = launcher.read_run_config(launcher.SCRIPT.parent / "configs/950pod_target.json")
    data.update(mode="perf", precision="all", ep_size=ep_size, output_dir="out")
    model = data["config"]["model"]
    model.update(activation=activation, topk_weights_type=int(activation == "gelu"), num_topk=32)
    path = tmp_path / "run.json"
    assert launcher.pod_settings(path, data).precisions == ("w4a8", "w8a8", "w4a4")
    model["num_shared_experts"] = 1
    assert launcher.pod_settings(path, data).precisions == ("w4a8", "w4a4")
    with pytest.raises(ValueError, match="共享专家要求 W4A8 或 W4A4"):
        launcher.pod_settings(path, dict(data, precision="a8w8"))


@pytest.mark.parametrize(("preset", "missing"), [
    ("ubx_ep16_target.json", "master_addr"),
    ("ubx_ep32_target.json", "node_ip"),
    ("950pod_target.json", "master_addr"),
])
def test_public_entry_requires_addresses_in_json(tmp_path, monkeypatch, capsys, preset, missing):
    run = json.loads((launcher.SCRIPT.parent / "configs" / preset).read_text(encoding="utf-8"))
    del run[missing]
    run["plan"] = True
    source = tmp_path / preset
    source.write_text(json.dumps(run), encoding="utf-8")
    monkeypatch.setenv("MASTER_ADDR", "192.0.2.99")
    with pytest.raises(SystemExit) as error:
        launcher.entrypoint(["--run-config", str(source)])
    assert error.value.code == 2
    assert missing in capsys.readouterr().err


def test_public_entry_requires_inline_json(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        launcher.entrypoint(["--config", "target_gelu_k25"])
    assert error.value.code == 2
    assert "--run-config" in capsys.readouterr().err

    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(dict(schema_version=1, machine="ubx", mode="perf", precision="a8w4",
                                      config="target_gelu_k25")), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        launcher.entrypoint(["--run-config", str(source)])
    assert error.value.code == 2
    assert "内嵌模型" in capsys.readouterr().err


def test_urma_a4w4_fails_before_any_launch(capsys):
    with pytest.raises(SystemExit) as error:
        launcher.parse_args(["accuracy", "a4w4"])
    assert error.value.code == 2
    assert "URMA 暂不支持 W4A4" in capsys.readouterr().err


def test_run_json_sources_requested_cann_before_selected_python(tmp_path, monkeypatch):
    source = tmp_path / "launch.json"
    source.write_text(json.dumps(dict(schema_version=1, mode="perf", python=sys.executable,
                                      master_addr="127.0.0.1", cann_env="sdk with spaces/set_env.sh")), encoding="utf-8")
    expected = str(tmp_path / "sdk with spaces" / "set_env.sh")

    class ExecAttempt(Exception):
        pass

    def execute(executable, command, env):
        assert executable == "bash" and 'source "$1"' in command[2]
        assert command[4] == str(launcher.ENV_SCRIPT)
        assert "--preserve-python" in command[2]
        assert env["UBX_CANN_SETUP"] == expected and env["PYTHON"] == sys.executable
        assert command[-2:] == ["--run-config", str(source)]
        raise ExecAttempt

    monkeypatch.setattr(launcher.os, "execvpe", execute)
    with pytest.raises(ExecAttempt):
        launcher.main(["--run-config", str(source)])


@pytest.mark.parametrize("selected_python", [None, sys.executable])
def test_ep32_controller_loads_ubx_environment_before_launch(tmp_path, monkeypatch, selected_python):
    source = tmp_path / "run.json"
    preset = launcher.SCRIPT.parent / "configs/ubx_ep32_target.json"
    data = json.loads(preset.read_text(encoding="utf-8"))
    data.update(mode="perf", precision="a8w4", output_dir="run")
    source.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.delenv("UBX_CANN_LOADED")
    if selected_python is None:
        monkeypatch.delenv("PYTHON")

    class ExecAttempt(Exception):
        pass

    def execute(executable, command, env):
        assert executable == "bash"
        assert command[4:] == [str(launcher.ENV_SCRIPT), str(launcher.SCRIPT), "--run-config", str(source)]
        assert env.get("PYTHON") == selected_python
        raise ExecAttempt

    monkeypatch.setattr(launcher.os, "execvpe", execute)
    monkeypatch.setattr(launcher, "orchestrate", lambda *_: pytest.fail("launched before environment bootstrap"))
    with pytest.raises(ExecAttempt):
        launcher.entrypoint(["--run-config", str(source)])
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("machine,preset", [("950pod", "950pod_target.json"), ("ubx", "ubx_ep16_target.json")])
@pytest.mark.parametrize("kind", ["relative", "absolute", "command"])
def test_bootstrap_preserves_python_entrypoint(tmp_path, monkeypatch, machine, preset, kind):
    if kind == "command":
        configured = expected = "python3-selected"
    else:
        base = tmp_path / "base" / "bin" / "python"
        base.parent.mkdir(parents=True)
        base.touch()
        executable = tmp_path / "selected venv" / "bin" / "python"
        executable.parent.mkdir(parents=True)
        try:
            executable.symlink_to(base)
        except OSError:
            if sys.platform != "win32":
                raise
            # A junction exercises dereferencing a linked directory on Windows hosts
            # where creating a symlink to a file requires additional privileges.
            executable.parent.rmdir()
            subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(executable.parent), str(base.parent)],
                           check=True, capture_output=True)
        assert executable.is_file() and executable.resolve() == base.resolve()
        assert executable != executable.resolve()
        expected = str(executable)
        configured = str(executable.relative_to(tmp_path)) if kind == "relative" else expected

    data = json.loads((launcher.SCRIPT.parent / "configs" / preset).read_text(encoding="utf-8"))
    data.update(mode="perf", precision="a8w4", python=configured, master_addr="127.0.0.1", cann_env="sdk/set_env.sh")
    expected_base = expected
    if machine == "950pod":
        data["base_python"] = data.pop("python")
        data["python"] = ""
    source = tmp_path / "run.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.delenv("DEEPEP_950POD_BOOTSTRAPPED", raising=False)

    class ExecAttempt(Exception):
        pass

    def execute(executable, command, env):
        assert executable == "bash"
        if machine == "950pod":
            assert command[5] == expected
            assert command[6] == str(Path(expected_base).parent.parent / "lib")
        else:
            assert env["PYTHON"] == expected
        raise ExecAttempt

    monkeypatch.setattr(launcher.os, "execvpe", execute)
    with pytest.raises(ExecAttempt):
        launcher.entrypoint(["--run-config", str(source)])
    if machine == "950pod":
        data["python"] = configured
        expected_base = data["base_python"] = "python3-base"
        source.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ExecAttempt):
            launcher.entrypoint(["--run-config", str(source)])
