# Copyright (c) 2026, Lu Lu

"""Expand an L-series acceptance case into its platform's workload launchers."""

import json
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from . import run_megamoe as runner
    from .l0_pod import run_pod_job, run_pod_paired
else:
    import run_megamoe as runner
    from l0_pod import run_pod_job, run_pod_paired


def case_jobs(data, selection):
    pod = data["machine"] == "950pod"
    precision = data["precision"]
    native_b_limited = pod and data["ep_size"] == 32
    jobs = []
    memory_only = selection == "memory"
    configured_peak = runner.configured_memory_peak(data)
    if memory_only and configured_peak is False:
        raise ValueError("L0 memory 要求 profile_memory_peak=true")
    if not memory_only and data.get("perf_scope") == "memory":
        raise ValueError("L0 独立显存诊断请使用 --only memory")

    def add(name, kind, arm=None, reason=None):
        job = dict(name=name, kind=kind, status="unsupported" if reason else "planned")
        if kind == "accuracy":
            job["accuracy_cascade"] = data.get("accuracy_cascade", "prefill")
        if memory_only:
            job.update(name=f"{name}-memory", kind="memory", workload=kind)
        if kind != "overlap":
            job["profile_memory_peak"] = data.get("profile_memory_peak", memory_only or kind == "accuracy")
            if job["profile_memory_peak"]:
                job["memory_scope"] = {"accuracy": "fused_and_accuracy_reference", "perf": "fused",
                                       "paired": "gelu_and_swiglu", "prefill": "fused_and_prefill_cascade"}[kind]
        if arm:
            job["arm"] = arm
        if reason:
            job["reason"] = reason
        jobs.append(job)

    if selection in ("both", "accuracy"):
        for arm in ("gelu", "swiglu"):
            reason = None
            if native_b_limited and arm == "swiglu" and data.get("accuracy_stage") not in ("golden", "cascade"):
                reason = "950POD SwiGLU 的整批对称区超限；此精度尚无同窗的全量双参考入口。"
            add(f"{arm}-accuracy", "accuracy", arm, reason)
    if selection in ("both", "perf", "memory"):
        add("gelu", "perf", "gelu")
        if not (native_b_limited and precision == "a8w4"):
            reason = "950POD A8W8 SwiGLU 的整批对称区超限，配对分窗入口仅支持 A8W4。" if native_b_limited else None
            add("swiglu", "perf", "swiglu", reason)
        if pod and precision == "a8w4":
            add("paired", "paired", "gelu")
    if selection in ("both", "perf"):
        if pod:
            if precision == "a8w4" and not native_b_limited:
                add("native-overlap", "overlap")
            reason = ("950POD MTE A8W8 没有设备 timer，无法计算掩盖率。" if precision == "a8w8" else
                      "950POD 全窗验收需两臂逐窗打点；原生 timer 只保留首窗。")
            add("all-window-overlap", "overlap", reason=reason)
        else:
            reason = "本机 MTE A8W8 没有设备 timer，无法计算掩盖率。" if data["backend"] == "mte" and precision == "a8w8" else None
            add("native-overlap", "overlap", reason=reason)
    if selection in ("both", "perf", "prefill", "memory"):
        add("gelu-prefill", "prefill", "gelu")
    return jobs


def job_config(base, job, root):
    model = dict(base["config"]["model"], activation=job["arm"], topk_weights_type=int(job["arm"] == "gelu"))
    workload = job.get("workload", job["kind"])
    accuracy = workload == "accuracy"
    timing = job["kind"] == "perf" and (base["backend"] == "mte_roce" or base["precision"] != "a8w8")
    config = dict(base, config=dict(base["config"], model=model), output_dir=str(root / job["name"]),
                  mode="accuracy" if accuracy else "perf", dual_reference=accuracy and not base.get("accuracy_stage"), check_tokens=0,
                  timer=timing, compare_prefill_cascade=workload == "prefill",
                  profile_memory_peak=job["profile_memory_peak"])
    if job["kind"] == "memory":
        config["perf_scope"] = "memory"
    if accuracy:
        config["accuracy_cascade"] = job["accuracy_cascade"]
    if accuracy and not base.get("accuracy_stage") and base["machine"] == "ubx" and base["ep_size"] <= runner.LOCAL_RANKS:
        # This acceptance run must own the per-rank accuracy and memory reports it archives.
        config["force_npu"] = True
    return config


def rank_zero_case_dir(cfg):
    run = cfg["mode"]
    if cfg["machine"] == "ubx":
        run += "-" + runner.parse_precision(cfg["precision"])
    if cfg.get("accuracy_stage"):
        run += "-" + cfg["accuracy_stage"]
    bs = cfg["config"]["bs_values"][0]
    return Path(cfg["output_dir"]) / run / "rank_0" / f"bs_{bs}"


def memory_result(cfg, paired):
    path = Path(cfg["output_dir"]) / "paired.json" if paired else rank_zero_case_dir(cfg) / "resources.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = report.get("device_memory_peak_group")
    if (not isinstance(summary, dict) or summary.get("rank_count") != cfg["ep_size"]
            or not {"max_peak_used_bytes", "max_peak_used_rank", "max_peak_used_pct", "max_peak_used_pct_rank",
                    "min_headroom_bytes", "min_headroom_rank"} <= summary.keys()):
        raise ValueError(f"{path} 缺少 EP{cfg['ep_size']} 全组显存峰值统计")
    return dict(device_memory_peak_group=summary, device_memory_peak_report=str(path))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_overlap(base, root):
    pod = base["machine"] == "950pod"
    perf = "perf" if pod else "perf-" + runner.parse_precision(base["precision"])
    if base["ep_size"] < 32:
        windows = {
            arm: json.loads((root / arm / perf / "rank_0/manifest.json").read_text(encoding="utf-8"))["window_tokens"]
            for arm in ("gelu", "swiglu")
        }
        if windows["gelu"] != windows["swiglu"]:
            return dict(status="unsupported", window_tokens=windows, reason="两臂实测首窗大小不同，无法验收同窗掩盖率")
    result = root / "native-overlap.json"
    argv = [runner.SCRIPT.with_name("compare_ubx_native_overlap.py"),
            "--gelu", root / "gelu" / perf, "--swiglu", root / "swiglu" / perf, "--output", result]
    if pod:
        argv.extend(["--platform", "950pod"])
        base_python = base.get("base_python") or os.environ.get("BASE_PYTHON") or runner.POD_PYTHON
        python = base.get("python") or os.environ.get("PYTHON") or base_python
        cann = base.get("cann_env") or os.environ.get("CANN_ENV") or runner.POD_CANN_ENV
        command = ["bash", "-c", runner.POD_BOOTSTRAP, "950pod", cann, python,
                   str(Path(base_python).parent.parent / "lib"), *map(str, argv)]
    else:
        env = {key: base[field] for key, field in (("PYTHON", "python"), ("UBX_CANN_SETUP", "cann_env")) if base.get(field)}
        # This offline reader imports the installed timer protocol and needs no device-side Python paths.
        command = ["bash", "-c", runner.node_script(runner.REPO_ROOT, env, runner.ENV_SCRIPT, ["-I", *argv])]
    subprocess.run(command, check=True)
    metrics = json.loads(result.read_text(encoding="utf-8"))
    a, b = (metrics["arms"][arm]["host_e2e"]["rank_max_mean_us"] for arm in ("gelu", "swiglu"))
    diff = metrics["gelu_minus_swiglu_pp"]["Total"]
    return dict(result=str(result), full_batch_e2e_ratio=a / b, first_window_total_delta_pp=diff,
                e2e_target_met=a <= b, first_window_overlap_target_met=diff >= 0)


def run_l0(path, data, *, only=None, plan=False, accuracy_arm=None):
    case = data.get("case")
    precision = data.get("precision")
    cases = {"L01": ("ubx", "a8w4"), "L02": ("ubx", "a8w8"),
             "L04": ("950pod", "a8w4"), "L05": ("950pod", "a8w8")}
    ep = data.get("ep_size")
    if (data.get("suite") != "l0" or not isinstance(case, str)
            or cases.get(case) != (data.get("machine"), precision) or type(ep) is not int or ep not in (4, 8, 16, 32)):
        raise ValueError("L01/L02 和 L04/L05 分别使用 UBX 或 950POD 的 A8W4 / A8W8 和 EP4/8/16/32")
    spec = data.get("config")
    if not isinstance(spec, dict) or spec.get("schema_version") != 1 or not isinstance(spec.get("model"), dict):
        raise ValueError("L0 需包含 schema_version=1 的内嵌模型 config")
    runner.validate_model_precision(spec["model"])
    if spec.get("bs_values") != [39280] or "bs" in data:
        raise ValueError("L0 按每卡 39280 token 验收，请使用 config.bs_values=[39280]")
    shape = dict(hidden=4096, intermediate_hidden=1536, num_experts=800, num_topk=25, num_shared_experts=0)
    if any(spec["model"].get(key) != value for key, value in shape.items()):
        raise ValueError("L0 验收固定 H4096 / I1536 / E800 / K25 / S0")
    selection = only or data.get("mode", "both")
    if selection not in ("both", "perf", "accuracy", "prefill", "memory"):
        raise ValueError("L0 支持 both、accuracy、perf、prefill、memory")
    if type(data.get("plan", False)) is not bool:
        raise ValueError("plan 必须为布尔值")
    plan = plan or data.get("plan", False)
    jobs = case_jobs(data, selection)
    if accuracy_stage := data.get("accuracy_stage"):
        jobs = [job for job in jobs if accuracy_arm is None or job["arm"] == accuracy_arm]
        for job in jobs:
            job["name"] = f"{job['arm']}-{accuracy_stage}"
            if accuracy_stage == "golden":
                job["profile_memory_peak"] = False
            if job["profile_memory_peak"]:
                job["memory_scope"] = "fused" if accuracy_stage == "megamoe" else "npu_cascade"
            else:
                job.pop("memory_scope", None)
    base = {k: v for k, v in data.items() if k not in ("suite", "case", "nodes", "mode", "plan", "output_dir", "run_base")}
    # The resolved worker configs will live under the run directory, not beside the user's preset.
    for key in ("cann_env", "golden_dir"):
        if base.get(key):
            base[key] = str((path.parent / Path(base[key]).expanduser()).resolve())
    for key in ("python", "base_python"):
        if base.get(key):
            base[key] = runner.python_command(base[key], path.parent)
    base_dir = Path(data.get("run_base") or os.environ.get("RUN_BASE") or runner.REPO_ROOT / "benchmark-runs").expanduser()
    base_dir = (path.parent / base_dir).resolve() if data.get("run_base") else base_dir.resolve()
    explicit = data.get("output_dir")
    root = (path.parent / Path(explicit).expanduser()).resolve() if explicit else base_dir / f"{case}-RUN"
    if explicit and accuracy_stage:
        root = root / (f"{accuracy_stage}-{accuracy_arm}" if accuracy_arm else accuracy_stage)
    nodes = data.get("nodes")
    pod = base["machine"] == "950pod"
    if not pod:
        backend = "mte_roce" if ep == 32 else "mte"
        if base.get("backend") != backend or "node_rank" in base or "stage" in base or nodes is not None:
            raise ValueError(f"UBX EP{ep} L0 需要 {backend} 自动启动配置")
        if ep == 32:
            nodes = base["node_ip"]
        else:
            nodes = [base["master_addr"]]
    # Preserve the E800 workload in the acceptance matrix even where it cannot launch.
    if shape["num_experts"] // ep > 128:
        reason = f"E800 / EP{ep} = {shape['num_experts'] // ep} 个本地专家，超过每 rank 的原生上限 128"
        if not plan:
            raise ValueError(reason)
        for job in jobs:
            job.update(status="unsupported", reason=reason)
    for job in jobs:
        if job["status"] == "unsupported" or job["kind"] == "overlap":
            continue
        cfg = job_config(base, job, root)
        if pod:
            launch = run_pod_paired if job.get("workload", job["kind"]) == "paired" else run_pod_job
            launch(cfg, root / "configs", nodes=nodes, plan=True)
        else:
            runner.parse_args(["--run-config", str(path)], cfg)

    summary = dict(case=case, machine=base["machine"], backend=base.get("backend"), precision=precision,
                   ep_size=ep, nodes=nodes,
                   tokens_per_rank=spec["bs_values"][0], selection=selection, jobs=jobs)
    if not pod and ep == 32:
        summary["node_wait_timeout"] = base.get("node_wait_timeout", runner.DEFAULT_NODE_WAIT_TIMEOUT)
    if accuracy_stage:
        summary.update(accuracy_stage=accuracy_stage, golden_dir=base["golden_dir"])
    if base.get("peer_python"):
        summary["peer_python"] = base["peer_python"]
    if plan:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if explicit:
        root.mkdir(parents=True, exist_ok=False)
    else:
        stage = accuracy_stage or ("full" if selection == "both" else selection)
        if accuracy_arm:
            stage += "-" + accuracy_arm
        if base.get("profile_memory_peak") and selection in ("both", "perf", "prefill"):
            stage += "-hbm"
        label = f"{case}-{base['machine']}-ep{base['ep_size']}-{precision}-{stage}"
        root = runner.create_run_directory(base_dir, label)
    configs = root / "configs"
    configs.mkdir()
    write_json(configs / "case.json", data)
    report = root / "run-summary.json"
    summary.update(output_dir=str(root), status="running")
    write_json(report, summary)
    print(f"[{case}] 输出 {root}", flush=True)
    for job in jobs:
        name = job["name"]
        if job["status"] == "unsupported":
            print(f"[{case}] {name} 待补：{job['reason']}", flush=True)
            continue
        print(f"[{case}] {name}", flush=True)
        job["status"] = "running"
        write_json(report, summary)
        try:
            if job["kind"] == "overlap":
                job.update(collect_overlap(base, root))
            else:
                cfg = job_config(base, job, root)
                src = configs / f"{name}.json"
                write_json(src, cfg)
                job.update(config=str(src), output_dir=cfg["output_dir"])
                if accuracy_stage in ("megamoe", "cascade"):
                    job["accuracy_report"] = str(rank_zero_case_dir(cfg) / "accuracy.json")
                paired_job = job.get("workload", job["kind"]) == "paired"
                if pod:
                    launch = run_pod_paired if paired_job else run_pod_job
                    launch(cfg, configs, nodes=nodes)
                    if paired_job:
                        result = root / name / "paired.json"
                        job["result"] = str(result)
                        if selection != "memory":
                            paired = json.loads(result.read_text(encoding="utf-8"))["host_e2e"]
                            job.update(median_pair_e2e_ratio=paired["median_pair_ratio"],
                                       e2e_target_met=paired["passed_median_pair_ratio_le_one"])
                else:
                    subprocess.run([sys.executable, str(runner.SCRIPT), "--run-config", str(src)], check=True)
                if cfg["profile_memory_peak"]:
                    job.update(memory_result(cfg, paired_job))
            if job["status"] == "running":
                job["status"] = "completed"
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError, KeyboardInterrupt) as exc:
            job.update(status="failed", error=str(exc))
            summary["status"] = "failed"
            raise
        finally:
            write_json(report, summary)
    partial = any(job["status"] == "unsupported" for job in jobs)
    summary["status"] = "partial" if partial else "completed"
    write_json(report, summary)
    print(f"[{case}] {summary['status']}  {report}", flush=True)
    return 3 if partial else 0
