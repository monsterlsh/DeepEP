# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Launch MegaMoE on explicitly configured hosts sharing one checkout."""

import argparse
import getpass
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_directory import create_run_directory  # noqa: E402

sys.path.pop(0)


def select_hosts(config, ep, master=None, ranks_per_node=8):
    hosts = config["hosts"]
    if not isinstance(hosts, list) or not all(isinstance(host, str) for host in hosts):
        raise ValueError("hosts must be a list of IPv4 addresses")
    if ranks_per_node not in (8, 16) or ep % ranks_per_node:
        raise ValueError("ranks_per_node must be 8 or 16 and divide EP size")
    if len(hosts) != ep // ranks_per_node or len(set(hosts)) != len(hosts):
        raise ValueError(f"EP{ep} requires exactly {ep // ranks_per_node} distinct hosts")
    for host in hosts:
        ipaddress.IPv4Address(host)
    master = master or config.get("master") or hosts[0]
    if master not in hosts:
        raise ValueError("Master must be in selected hosts")
    return [master] + [host for host in hosts if host != master]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep-size", type=int, choices=(8, 16, 32, 64, 128), default=64)
    parser.add_argument("--ranks-per-node", type=int, choices=(8, 16), default=8)
    parser.add_argument("--hosts-file", type=Path, required=True, help="User-owned JSON with hosts and optional master")
    parser.add_argument("--master-ip")
    parser.add_argument("--shared-code", type=Path, default=Path(os.environ.get("SHARED_CODE", ROOT)))
    parser.add_argument("--config", default="deepseek_v4_pro")
    parser.add_argument("--bs-values", default=os.environ.get("BS_VALUES"))
    parser.add_argument("--input", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--routing", choices=("balanced", "random", "random-balanced", "skewed"), default="balanced")
    parser.add_argument("--mode", choices=("perf", "accuracy"), default="perf")
    parser.add_argument("--check-tokens", type=int, default=0, help="CPU Golden tokens per rank, 0 checks all")
    parser.add_argument("--no-timer", action="store_true", help="Disable timer collection and parsing")
    timer_parse = parser.add_mutually_exclusive_group()
    timer_parse.add_argument(
        "--parse-timer",
        dest="parse_timer",
        action="store_true",
        default=None,
        help="Parse master ranks 0-7 after success (default when timer is enabled)",
    )
    timer_parse.add_argument(
        "--no-parse-timer",
        dest="parse_timer",
        action="store_false",
        help="Save raw timer and breakdown only; defer JSON parsing",
    )
    parser.add_argument("--wheel", type=Path, help="Reuse this exact shared wheel instead of building on master")
    parser.add_argument(
        "--skip-build", action="store_true", help="Reuse the unique local wheel matching master's installed package"
    )
    parser.add_argument("--plan", action="store_true", help="Print resolved plan; no SSH, writes or prompts")
    parser.add_argument("--prepare-only", action="store_true", help="Build/install/verify, but do not run NPU tests")
    parser.add_argument("--timeout", type=int, default=7200, help="Seconds per stage/BS; failure cancels this launch")
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29654")))
    parser.add_argument("--shmem-port", type=int, default=int(os.environ.get("SHMEM_PORT", "26733")))
    return parser.parse_args(argv)


def make_plan(args):
    root = args.shared_code.resolve(strict=True)
    config_path = Path(args.config)
    if not config_path.is_file():
        config_path = root / "experiments/megamoe/python/deep_ep_experimental/megamoe/configs" / (args.config + ".json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.bs_values is not None:
        config["bs_values"] = [int(value) for value in args.bs_values.split(",")]
    batches = config["bs_values"]
    if (
        not batches
        or len(set(batches)) != len(batches)
        or any(type(b) is not int or not 0 < b <= 65536 for b in batches)
    ):
        raise ValueError("BS values must be distinct integers in [1,65536]")
    if config["model"]["num_experts"] % args.ep_size:
        raise ValueError("Global experts must divide EP size")
    if (
        not all(1024 <= port <= 65535 for port in (args.master_port, args.shmem_port))
        or args.master_port == args.shmem_port
    ):
        raise ValueError("Master/SHMEM ports must be distinct unprivileged TCP ports")
    if args.timeout <= 0 or args.check_tokens < 0 or (args.parse_timer and args.no_timer):
        raise ValueError("Invalid timeout, check-tokens or --parse-timer with --no-timer")
    user, auth = os.environ.get("REMOTE_USER", getpass.getuser()), os.environ.get("SSH_AUTH", "key")
    if not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*", user) or auth not in {"key", "password"}:
        raise ValueError("Invalid REMOTE_USER or SSH_AUTH (key/password)")
    ssh_port = int(os.environ.get("SSH_PORT", "22"))
    if not 1 <= ssh_port <= 65535:
        raise ValueError("Invalid SSH_PORT")
    hosts = select_hosts(json.loads(args.hosts_file.read_text(encoding="utf-8")), args.ep_size, args.master_ip, args.ranks_per_node)
    environment = {name: os.environ.get(name, "") for name in ("CONDA_BASE", "CONDA_ENV", "ASCEND_ENV")}
    missing = [name for name, value in environment.items() if not value.strip()]
    if missing:
        raise ValueError("Set remote environment explicitly: " + ", ".join(missing))
    for name in ("CONDA_BASE", "ASCEND_ENV"):
        if not PurePosixPath(environment[name]).is_absolute():
            raise ValueError(f"{name} must be an absolute path on each remote host")
    wheel = str(args.wheel.resolve(strict=True)) if args.wheel else None
    return dict(
        root=str(root),
        hosts=hosts,
        ranks_per_node=args.ranks_per_node,
        ep_size=args.ep_size,
        check_tokens=args.check_tokens,
        config=config,
        input=args.input,
        routing=args.routing,
        mode=args.mode,
        prepare_only=args.prepare_only,
        timer=not args.no_timer,
        parse_timer=not args.no_timer and args.parse_timer is not False,
        wheel=wheel,
        skip_build=args.skip_build,
        timeout=args.timeout,
        master_port=args.master_port,
        shmem_port=args.shmem_port,
        user=user,
        auth=auth,
        ssh_port=ssh_port,
        conda_base=environment["CONDA_BASE"],
        conda_env=environment["CONDA_ENV"],
        ascend_env=environment["ASCEND_ENV"],
    )


def batch_run_label(plan):
    model = plan["config"]["model"]
    precision = {"fp8xfp4": "a8w4", "fp8xfp8": "a8w8", "fp4xfp4": "a4w4"}[model.get("mma_type", "fp8xfp4")]
    mode = "prepare" if plan["prepare_only"] else plan["mode"]
    activation = model.get("activation", "swiglu")
    batches = plan["config"]["bs_values"]
    batch = f"bs{batches[0]}" if len(batches) == 1 else f"bs{min(batches)}-{max(batches)}-n{len(batches)}"
    return f"batch-ep{plan['ep_size']}-{precision}-{mode}-{activation}-k{model['num_topk']:d}-{batch}"


def remote_script(plan, run_dir, rank, stage, bs=None):
    q = shlex.quote
    command = '"$PYTHON" -I ' + shlex.join(
        [
            str(Path(plan["root"]) / "experiments/megamoe/scripts/batch_node.py"),
            stage,
            "--run-dir",
            str(run_dir),
            "--node-rank",
            str(rank),
        ]
        + (["--bs", str(bs)] if bs is not None else [])
    )
    # 独立进程组只包含本阶段的子进程；连接断开/中断不使用全局 pkill。
    return f"""set -eo pipefail
source {q(plan["conda_base"] + "/etc/profile.d/conda.sh")}
conda activate {q(plan["conda_env"])}
TASK_CONDA_PREFIX="${{CONDA_PREFIX:?Conda activation failed}}"
source {q(plan["ascend_env"])}
export PYTHON="$TASK_CONDA_PREFIX/bin/python"
export PATH="$TASK_CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH={q(plan["root"] + "/third-party/shmem/install/shmem/lib:")}"$TASK_CONDA_PREFIX/lib:${{LD_LIBRARY_PATH:-}}"
unset PYTHONHOME PYTHONPATH ASCEND_LAUNCH_BLOCKING
cd /tmp
set -m
child=''
cleanup() {{
  trap - EXIT HUP INT TERM
  if [[ -n $child ]]; then kill -TERM -- "-$child" 2>/dev/null || true; fi
}}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
{command} &
child=$!
rc=0
wait "$child" || rc=$?
exit "$rc"
"""


def run_stage(plan, run_dir, stage, ranks=None, bs=None, password=None):
    ranks = list(range(len(plan["hosts"]))) if ranks is None else ranks
    pending, files = {}, []
    started = time.monotonic()
    try:
        for rank in ranks:
            path = run_dir / "logs" / f"node_{rank}_{stage}{('_bs' + str(bs)) if bs else ''}.log"
            log = path.open("wb")
            files.append(log)
            command = [
                "ssh",
                "-T",
                "-p",
                str(plan["ssh_port"]),
                "-o",
                "ConnectTimeout=10",
                "-o",
                "ServerAliveInterval=10",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                "StrictHostKeyChecking=accept-new",
            ]
            env = dict(os.environ)
            if plan["auth"] == "password":
                command = ["sshpass", "-e"] + command
                env["SSHPASS"] = password
            else:
                command += ["-o", "BatchMode=yes"]
                env.pop("SSHPASS", None)
            command += [f"{plan['user']}@{plan['hosts'][rank]}", "bash -s"]
            print(f"[START] {stage} node={rank} host={plan['hosts'][rank]} log={path}", flush=True)
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, env=env)
            pending[proc] = (rank, path)
            proc.stdin.write(remote_script(plan, run_dir, rank, stage, bs).encode())
            proc.stdin.close()
        while pending:
            if time.monotonic() - started > plan["timeout"]:
                raise TimeoutError(f"Stage {stage} exceeded {plan['timeout']} seconds")
            for proc, (rank, path) in list(pending.items()):
                code = proc.poll()
                if code is None:
                    continue
                del pending[proc]
                if code:
                    print(path.read_text(encoding="utf-8", errors="replace")[-6000:], file=sys.stderr)
                    raise RuntimeError(f"{stage} node={rank} exit={code}; log={path}")
                print(f"[OK] {stage} node={rank}", flush=True)
            if pending:
                time.sleep(0.2)
    finally:
        for proc in pending:
            if proc.poll() is None:
                proc.terminate()
        for proc in pending:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in files:
            log.close()


def summarize(run_dir, batches, world):
    lines = []
    for bs in batches:
        path = run_dir / f"bs_{bs}/rank_0/bs_{bs}/profiling.json"
        urma_path = path.with_name("performance.json")
        urma = not path.exists() and urma_path.exists()
        if urma:
            path = urma_path
        report = json.loads(path.read_text(encoding="utf-8"))
        mean = report["rank_min_mean_us"] if urma else report["mean_us"]
        metric = "mega_moe_urma_kernel_device_duration_us" if urma else "megamoe_kernel_duration_us"
        aggregation = "mean_of_per_round_cross_rank_extrema" if urma else "mean_of_per_round_min_rank"
        if (
            report.get("aggregation") != aggregation
            or report.get("metric") != metric
            or not math.isfinite(mean)
            or mean <= 0
            or len(report["per_rank_rounds_us"]) != world
        ):
            raise ValueError(f"Incomplete/invalid performance report: {path}")
        samples = report["per_rank_rounds_us"]
        rounds = report["rounds"]
        recorded_rounds = report.get("recorded_rounds", 0)
        first_selected = 0 if urma else recorded_rounds // 2
        selection = "all_formal_rounds" if urma else "tail_half_ceil"
        if (
            report.get("iteration_selection") != selection
            or recorded_rounds < 1
            or rounds != recorded_rounds - first_selected
            or report.get("first_selected_iteration") != first_selected
        ):
            raise ValueError(f"Invalid performance round selection: {path}")
        if (
            rounds < 1
            or any(len(row) != rounds or any(not 0 < t < float("inf") for t in row) for row in samples)
            or not math.isclose(mean, sum(min(row) for row in zip(*samples)) / rounds)
            or not math.isclose(report["rank_max_mean_us"], sum(max(row) for row in zip(*samples)) / rounds)
        ):
            raise ValueError(f"Incomplete/invalid performance rounds: {path}")
        lines.append(
            f"[PERF] EP={world} BS={bs} MegaMoe kernel sum={mean:.3f}us "
            f"rank-max-mean={report['rank_max_mean_us']:.3f}us "
            f"windows={report.get('windows_per_forward', 1)} rounds={rounds}/{recorded_rounds} ({selection}) "
            f"kernel-Duration rank-min-mean timer=off"
        )
        if "host_e2e" in report:
            lines.append(f"Host forward through device synchronization: rank-max-mean={report['host_e2e']['rank_max_mean_us']:.3f}us")
        breakdown = run_dir / f"bs_{bs}/bs_{bs}/timer/timer_breakdown.txt"
        if breakdown.exists():
            lines.append(breakdown.read_text(encoding="utf-8").rstrip())
    text = "\n".join(lines) + "\n"
    (run_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def summarize_accuracy(run_dir, batches, world, *, launch_ok=True):
    """Summarize collected CPU checks; missing reports never count as PASS."""
    lines = [
        f"[ACCURACY SUMMARY] EP={world} launch={'OK' if launch_ok else 'FAILED'}",
        "  CPU mixed-error limits: bad elements <=5% per rank (error>0.01), max error <=0.1",
    ]
    if not launch_ok:
        lines.append(f"  launch failure/interruption: see node logs under {run_dir / 'logs'}")
    all_passed = launch_ok
    for bs in batches:
        path = run_dir / f"bs_{bs}/rank_0/bs_{bs}/accuracy.json"
        try:
            reports = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(reports, list) or len(reports) != world:
                raise ValueError(f"expected {world} rank reports")
            if any(type(row["rank"]) is not int for row in reports) or sorted(row["rank"] for row in reports) != list(
                range(world)
            ):
                raise ValueError("missing or duplicate rank")
            for row in reports:
                if (
                    row["total_tokens"] != bs
                    or type(row["checked_token_count"]) is not int
                    or not 0 < row["checked_token_count"] <= bs
                    or type(row["elements"]) is not int
                    or row["elements"] <= 0
                    or type(row["bad_elements"]) is not int
                    or not 0 <= row["bad_elements"] <= row["elements"]
                    or row["policy"] != "cpu_mixed_error"
                    or (row["diff_thd"], row["pct_thd"], row["max_diff"]) != (0.01, 0.05, 0.1)
                    or any(
                        type(row[key]) is not bool
                        for key in ("passed", "finite", "counts_equal", "cumulative_counts_equal")
                    )
                ):
                    raise ValueError(f"invalid rank={row['rank']} check metadata")
                for key in ("max_mixed_error", "max_abs_error", "relative_rmse"):
                    value = row[key]
                    if value is None and not row["finite"]:
                        continue
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"invalid rank={row['rank']} {key}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            lines += [f"[ACCURACY] BS={bs} INCOMPLETE: {error}", f"  report={path}"]
            all_passed = False
            continue

        failed = []
        for row in reports:
            reasons = []
            if not row["finite"]:
                reasons.append("non-finite output/Golden")
            elif row["bad_elements"] > 0.05 * row["elements"] or row["max_mixed_error"] > 0.1:
                reasons.append("output error exceeds tolerance")
            if not row["counts_equal"]:
                reasons.append("expert counts mismatch")
            if not row["cumulative_counts_equal"]:
                reasons.append("cumulative counts mismatch")
            if not row["passed"] and not reasons:
                reasons.append(row.get("reason") or "reported failed")
            if reasons:
                failed.append((row["rank"], "; ".join(reasons)))
        all_passed &= not failed
        checked = sum(row["checked_token_count"] for row in reports)
        status = "FAIL" if failed else "PASS"
        coverage = "FULL" if checked == world * bs else "SAMPLED"
        lines.append(
            f"[ACCURACY] BS={bs} {status} ranks={world - len(failed)}/{world} tokens={checked}/{world * bs} {coverage}"
        )

        def maximum(key):
            values = [row[key] for row in reports if row[key] is not None]
            return f"{max(values):.6g}" if values else "N/A"

        counts_ok = all(row["counts_equal"] for row in reports)
        cumulative_ok = all(row["cumulative_counts_equal"] for row in reports)
        lines.append(
            f"  max_mixed_error={maximum('max_mixed_error')} "
            f"worst_rank_bad_pct={max(row['bad_elements'] / row['elements'] for row in reports):.3%} "
            f"max_abs_error={maximum('max_abs_error')} max_relative_rmse={maximum('relative_rmse')}"
        )
        lines.append(
            f"  counts={'PASS' if counts_ok else 'FAIL'} cumulative_counts={'PASS' if cumulative_ok else 'FAIL'}"
        )
        for rank, reason in failed:
            lines.append(f"  rank={rank} FAIL: {reason}")
        lines.append(f"  report={path}")
    # 数值通过不代表作业正常退出；timer/清理失败也必须在总状态中体现。
    lines.append(f"[ACCURACY OVERALL] {'PASS' if all_passed else 'NOT PASSED'}")
    text = "\n".join(lines) + "\n"
    (run_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
    return all_passed


def verify_nodes(run_dir, plan, stage):
    reports = [
        json.loads((run_dir / f"node_{rank}_{stage}.json").read_text(encoding="utf-8"))
        for rank in range(len(plan["hosts"]))
    ]
    keys = ("commit", "cann") if stage == "preflight" else ("torch", "torch_npu", "python", "build")
    if any(any(row[key] != reports[0][key] for key in keys) for row in reports[1:]):
        raise RuntimeError(f"Nodes differ in {stage}: check per-node JSON under {run_dir}")
    if stage == "preflight":
        if len({row["boot_id"] for row in reports}) != len(reports):
            raise RuntimeError("Multiple host IPs refer to the same running OS; refusing duplicate device allocation")
    else:
        info = reports[0]["build"]["megamoe"]
        if any(reports[0][key] != info[key] for key in ("torch", "torch_npu")):
            raise RuntimeError("Installed Torch/torch_npu differs from wheel build environment")


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan:
        return
    if sys.platform != "linux":
        raise RuntimeError("Multi-node launch requires a Linux controller with the shared checkout mounted")
    import fcntl

    root = Path(plan["root"])
    runs = root / "benchmark-runs"
    runs.mkdir(exist_ok=True)
    # 共享 checkout 只允许本入口的一项部署/测试，避免并行安装和编译互相覆盖。
    with (runs / ".batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_dir = create_run_directory(runs, batch_run_label(plan), mode=0o777, unique_id=True)
        (run_dir / "logs").mkdir()
        plan["run_id"] = run_dir.name
        (run_dir / "launch.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        (run_dir / "case.json").write_text(json.dumps(plan["config"], indent=2), encoding="utf-8")
        password = None
        if plan["auth"] == "password":
            password = os.environ.get("SSHPASS") or getpass.getpass(f"SSH password for {plan['user']}: ")

        def call(stage, **kw):
            return run_stage(plan, run_dir, stage, password=password, **kw)

        print(f"[OUTPUT] {run_dir}", flush=True)
        call("preflight")
        verify_nodes(run_dir, plan, "preflight")
        call("prepare", ranks=[0])
        # Conda 环境可能也在共享盘，安装串行执行，避免多个 pip 同写 site-packages。
        for rank in range(len(plan["hosts"])):
            call("install", ranks=[rank])
        call("verify")
        verify_nodes(run_dir, plan, "verify")
        if not args.prepare_only:
            completed = []
            for bs in plan["config"]["bs_values"]:
                try:
                    if plan["mode"] == "accuracy":
                        # 各节点先生成/校验本机 Golden，全部完成后才启动 NPU，避免 Store 等待超时。
                        call("golden", bs=bs)
                    call("run", bs=bs)
                except (Exception, KeyboardInterrupt):
                    if plan["mode"] == "accuracy":
                        summarize_accuracy(run_dir, completed + [bs], plan["ep_size"], launch_ok=False)
                    raise
                completed.append(bs)
                if plan["mode"] == "perf":
                    summarize(run_dir, completed, plan["ep_size"])
                elif not summarize_accuracy(run_dir, completed, plan["ep_size"]):
                    raise RuntimeError("Accuracy checks incomplete/failed; see summary.txt")
            if plan["parse_timer"]:
                print("[TIMER] Parsing master ranks 0-7 automatically; per-rank JSON only, no global merge", flush=True)
                call("parse", ranks=[0])
        print(f"[DONE] {'prepared' if args.prepare_only else 'all nodes completed'}: {run_dir}", flush=True)


def interrupt(*_):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
