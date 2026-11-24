# Copyright (c) 2026, Lu Lu

"""Expand one EP32 acceptance case into the existing single-workload launchers."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

if __package__:
    from . import run_megamoe as runner
    from .l0_pod import run_pod_job, run_pod_paired
else:
    import run_megamoe as runner
    from l0_pod import run_pod_job, run_pod_paired


def case_jobs(data, selection):
    pod = data["machine"] == "950pod"
    precision = data["precision"]
    native_b_limited = pod and precision in ("a8w4", "a8w8")
    jobs = []

    def add(name, kind, arm=None, reason=None):
        job = dict(name=name, kind=kind, status="unsupported" if reason else "planned")
        if arm:
            job["arm"] = arm
        if reason:
            job["reason"] = reason
        jobs.append(job)

    if selection in ("both", "accuracy"):
        for arm in ("gelu", "swiglu"):
            reason = None
            if native_b_limited and arm == "swiglu":
                reason = "950POD SwiGLU 的整批对称区超限；此精度尚无同窗的全量双参考入口。"
            add(f"{arm}-accuracy", "accuracy", arm, reason)
    if selection in ("both", "perf"):
        add("gelu", "perf", "gelu")
        if native_b_limited and precision == "a8w4":
            add("paired", "paired", "gelu")
        else:
            reason = "950POD A8W8 SwiGLU 的整批对称区超限，配对分窗入口仅支持 A8W4。" if native_b_limited else None
            add("swiglu", "perf", "swiglu", reason)
        if pod:
            if precision == "a4w4":
                add("native-overlap", "overlap")
            add("all-window-overlap", "overlap", reason="950POD 全窗验收需两臂逐窗打点；原生 timer 只保留首窗。")
        else:
            add("native-overlap", "overlap")
    if selection in ("both", "perf", "prefill"):
        add("gelu-prefill", "prefill", "gelu")
    return jobs


def job_config(base, job, root):
    model = dict(base["config"]["model"], activation=job["arm"], topk_weights_type=int(job["arm"] == "gelu"))
    accuracy = job["kind"] == "accuracy"
    timing = job["kind"] == "perf" and (base["machine"] == "ubx" or base["precision"] != "a8w8")
    return dict(base, config=dict(base["config"], model=model), output_dir=str(root / job["name"]),
                mode="accuracy" if accuracy else "perf", dual_reference=accuracy, check_tokens=0,
                timer=timing, compare_prefill_cascade=job["kind"] == "prefill")


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_overlap(base, root):
    pod = base["machine"] == "950pod"
    perf = "perf" if pod else "perf-" + runner.parse_precision(base["precision"])
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


def run_l0(path, data, *, only=None, plan=False):
    case = data.get("case")
    precision = data.get("precision")
    matrix = [(machine, p) for machine in ("ubx", "950pod") for p in ("a8w4", "a8w8", "a4w4")]
    cases = {f"L{i:02}": pair for i, pair in enumerate(matrix, 1)}
    if (data.get("suite") != "l0" or not isinstance(case, str)
            or cases.get(case) != (data.get("machine"), precision) or data.get("ep_size") != 32):
        raise ValueError("L01–L06 依次使用 UBX、950POD 的 EP32 A8W4 / A8W8 / A4W4 预设")
    spec = data.get("config")
    if not isinstance(spec, dict) or spec.get("schema_version") != 1 or not isinstance(spec.get("model"), dict):
        raise ValueError("L0 需包含 schema_version=1 的内嵌模型 config")
    if spec.get("bs_values") != [39280] or "bs" in data:
        raise ValueError("L0 按每卡 39280 token 验收，请使用 config.bs_values=[39280]")
    selection = only or data.get("mode", "both")
    if selection not in ("both", "perf", "accuracy", "prefill"):
        raise ValueError("L0 支持 both、accuracy、perf、prefill")
    if type(data.get("plan", False)) is not bool:
        raise ValueError("plan 必须为布尔值")
    plan = plan or data.get("plan", False)
    jobs = case_jobs(data, selection)
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
    nodes = data.get("nodes")
    pod = base["machine"] == "950pod"
    if not pod and (base.get("backend") != "mte_roce" or "node_rank" in base or "stage" in base or nodes is not None):
        raise ValueError("UBX L0 需要双机自动启动的 mte_roce 配置")
    for job in jobs:
        if job["status"] == "unsupported" or job["kind"] == "overlap":
            continue
        cfg = job_config(base, job, root)
        if pod:
            launch = run_pod_paired if job["kind"] == "paired" else run_pod_job
            launch(cfg, root / "configs", nodes=nodes, plan=True)
        else:
            runner.parse_args(["--run-config", str(path)], cfg)

    summary = dict(case=case, machine=base["machine"], backend=base.get("backend"), precision=precision,
                   ep_size=base["ep_size"], nodes=nodes if pod else base["node_ip"],
                   tokens_per_rank=spec["bs_values"][0], selection=selection, jobs=jobs)
    if base.get("peer_python"):
        summary["peer_python"] = base["peer_python"]
    if plan:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if explicit:
        root.mkdir(parents=True, exist_ok=False)
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=f"{case}-", dir=base_dir))
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
                if pod:
                    launch = run_pod_paired if job["kind"] == "paired" else run_pod_job
                    launch(cfg, configs, nodes=nodes)
                    if job["kind"] == "paired":
                        result = root / name / "paired.json"
                        paired = json.loads(result.read_text(encoding="utf-8"))["host_e2e"]
                        job.update(result=str(result), median_pair_e2e_ratio=paired["median_pair_ratio"],
                                   e2e_target_met=paired["passed_median_pair_ratio_le_one"])
                else:
                    subprocess.run([sys.executable, str(runner.SCRIPT), "--run-config", str(src)], check=True)
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
