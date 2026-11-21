# Copyright (c) 2026, Lu Lu

"""Run MegaMoE GELU accuracy and performance on one or two 16-card UBX nodes."""

import argparse
from copy import copy
from functools import cache
import getpass
from ipaddress import IPv4Address
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from uuid import uuid4

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
from ubx_accuracy_cache import AccuracyCache, cache_key, file_lock  # noqa: E402
from ubx_multinode import node_script, run_stage  # noqa: E402
from paired_benchmark import order as pair_order  # noqa: E402

sys.path.pop(0)

REPO_ROOT = SCRIPT.parents[3]
CONFIG_DIR = SCRIPT.parents[1] / "python/deep_ep_experimental/megamoe/configs"
ENV_SCRIPT = SCRIPT.with_name("ubx_env.sh")
MODULE = "deep_ep_experimental.megamoe"
LOCAL_RANKS = 16
TIMER_RANKS_PER_NODE = 4
DEFAULT_NODE_IPS = ("90.90.93.84", "90.90.93.184")
MMA_TYPES = {"w4a8": "fp8xfp4", "w8a8": "fp8xfp8"}
MODES = ("accuracy", "perf", "both", "compare")
ROUTINGS = ("balanced", "skewed", "random", "random-balanced")
FORWARDED_ENV = (
    "PYTHON", "ROUTING", "CHECK_TOKENS", "DUAL_REFERENCE", "GOLDEN_WORKERS",
    "GOLDEN_DIR", "DEEPEP_SHMEM_ENDPOINT",
)
COMPARISON_PLAN = "comparison-plan.json"
COMPARISON_REPORT = "comparison.json"
WEIGHT_MODE = {"gelu": 1, "swiglu": 0}
DEVICE_CHECK = """
import sys
import torch
import torch_npu
from deep_ep._machine import board_id, physical_device
from deep_ep_experimental.megamoe.api import _extension

local, require_w8a8 = int(sys.argv[1]), sys.argv[2] == "1"
count = torch.npu.device_count()
if count < local:
    sys.exit(f"UBX 每机需要 {local} 张可见 NPU，当前只有 {count} 张。")
torch.npu.set_device(0)
board = board_id(physical_device())
if board != 0x44:
    sys.exit(f"可见设备 0 的板型为 0x{board:02x}，UBX 要求 0x44。")
extension = _extension()
if require_w8a8 and not getattr(extension, "urma_w8a8_enabled", False):
    sys.exit("当前 MegaMoE wheel 未启用 UBX URMA W8A8。")
print(f"[UBX] MegaMoE wheel: {extension.__file__}", flush=True)
"""


def parse_precision(value):
    name = value.casefold()
    if name == "w8a4":
        raise argparse.ArgumentTypeError("FP4 激活 × FP8 权重暂不支持；FP8 激活 × FP4 权重请选择 a8w4")
    return {"a8w4": "w4a8", "a8w8": "w8a8"}.get(name, name)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=MODES)
    parser.add_argument("--mode", dest="mode_option", choices=MODES,
                        help="accuracy 精度，perf GELU 性能，compare A/B 性能和掩盖率，both 精度加 GELU 性能")
    parser.add_argument("precision", nargs="?", type=parse_precision, choices=("all", *MMA_TYPES),
                        help="显式指定时适用于所选模式。省略时精度测两种，性能测 W4A8")
    parser.add_argument("--precision", dest="precision_option", type=parse_precision, choices=("all", *MMA_TYPES),
                        help="显式选择 w8a8、w4a8（也可写 a8w4）或 all")
    parser.add_argument("--ep-size", type=int, choices=(16, 32), default=16)
    node = parser.add_mutually_exclusive_group()
    node.add_argument("--node-ip", nargs=2, metavar=("NODE0", "NODE1"),
                      help="EP32 自动拉起的两台节点，默认 90.90.93.84 90.90.93.184")
    node.add_argument("--node-rank", type=int, choices=(0, 1), help="EP32 手动分阶段拉起时的节点序号")
    parser.add_argument("--stage", choices=("prepare", "run"), help="EP32 手动模式先 prepare，再 run")
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR"), help="EP32 默认取节点 0 的 IPv4")
    parser.add_argument("--master-port", type=int, default=os.environ.get("MASTER_PORT", "29500"))
    parser.add_argument("--ssh-user", default=os.environ.get("REMOTE_USER") or getpass.getuser())
    parser.add_argument("--ssh-port", type=int, default=os.environ.get("REMOTE_PORT", "22"))
    parser.add_argument("--config", default="target_gelu_k25", help="配置名（如 target_gelu_k32）或 JSON 路径")
    parser.add_argument("--weighting", choices=("auto", "early", "late"), default="auto")
    parser.add_argument("--bs", type=int, default=39280, help="每卡 token 数")
    parser.add_argument("--topk", type=int, help="覆盖模型 TopK")
    parser.add_argument("--window-tokens", type=int, choices=(64, 128, 256),
                        help="compare 的共同窗口上限，默认 256")
    parser.add_argument("--output-dir", type=Path, help="本轮输出目录。EP32 手动模式要求两侧显式给同一路径")
    parser.add_argument("--timer", action="store_true", help="另采一次首窗设备 trace，默认每台 4 个 rank")
    parser.add_argument("--force", action="store_true", help="强制重跑 CPU Golden 和 NPU 精度")
    parser.add_argument("--force-cpu", action="store_true", help="强制重新生成 CPU Golden")
    parser.add_argument("--force-npu", action="store_true", help="EP16 跳过 NPU 精度缓存")
    parser.add_argument("--pairs", type=int, help="compare 的独立交替 A/B 正式作业对数，默认 8")
    comparison_action = parser.add_mutually_exclusive_group()
    comparison_action.add_argument("--plan", action="store_true", help="compare 仅显示支持矩阵和执行顺序")
    comparison_action.add_argument("--report-only", action="store_true", help="compare 仅读取现有产物并写报告")
    parser.add_argument("--peer-output-dir", type=Path, action="append", default=[],
                        help="离线汇总时传入另一节点复制出的同一次运行目录")
    parser.add_argument("--require-no-regression", action="store_true",
                        help="compare 有退化、证据不全或基线不支持时返回非零")
    parser.add_argument("--comparison-arm", choices=WEIGHT_MODE, help=argparse.SUPPRESS)
    parser.add_argument("--perf-scope", choices=("all", "host", "timer"), default="all", help=argparse.SUPPRESS)
    parser.add_argument("--paired-weights", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.mode and args.mode_option:
        parser.error("位置模式参数与 --mode 二选一")
    args.mode = args.mode_option or args.mode or "both"
    if args.precision and args.precision_option:
        parser.error("位置精度参数与 --precision 二选一")
    args.precision = args.precision_option or args.precision

    if args.ep_size == 16:
        if args.node_ip or args.node_rank is not None or args.stage:
            parser.error("--node-ip、--node-rank、--stage 适用于 EP32")
        args.node_rank = 0
    else:
        if (args.node_rank is None) != (args.stage is None):
            parser.error("手动 EP32 需配套使用 --node-rank 和 --stage")
        if args.stage and args.output_dir is None:
            parser.error("手动 EP32 需指定 --output-dir")
        if args.force_npu:
            parser.error("EP32 的 NPU 测试每次实测，--force-npu 适用于 EP16")
        try:
            args.node_ip = tuple(str(IPv4Address(ip)) for ip in (args.node_ip or DEFAULT_NODE_IPS))
            args.master_addr = str(IPv4Address(args.master_addr or args.node_ip[0]))
        except ValueError as error:
            parser.error(str(error))
        if args.node_ip[0] == args.node_ip[1]:
            parser.error("EP32 需要两个不同的节点 IP")
        if not (1 <= args.ssh_port <= 65535 and 1024 <= args.master_port <= 65535):
            parser.error("SSH 或 torchrun 端口超出范围")

    if not 1 <= args.bs <= 65536:
        parser.error("--bs 必须在 1..65536")
    if args.topk is not None and not 1 <= args.topk <= 32:
        parser.error("--topk 必须在 1..32")
    if args.mode == "compare":
        if args.stage or args.comparison_arm:
            parser.error("compare 从主节点启动，节点阶段由入口管理")
        if args.weighting != "auto" or args.force or args.force_cpu or args.force_npu:
            parser.error("compare 固定两侧加权方式，且每轮均实测")
        if args.pairs is not None and args.pairs < 2:
            parser.error("compare 至少需要 2 对独立作业")
        if args.require_no_regression and args.pairs is not None and args.pairs < 7 and not args.report_only:
            parser.error("验收要求至少 7 对独立作业")
        if args.report_only and args.output_dir is None:
            parser.error("--report-only 需指定本轮 --output-dir")
        if args.peer_output_dir and not args.report_only:
            parser.error("--peer-output-dir 仅用于 --report-only")
    elif any((args.pairs is not None, args.plan, args.report_only, args.peer_output_dir, args.require_no_regression)):
        parser.error("配对、计划和离线汇总参数适用于 compare")
    if args.mode == "perf" and (args.force or args.force_cpu or args.force_npu or args.stage == "prepare"):
        parser.error("Golden 准备和强制重跑选项用于精度模式")
    args.routing = os.environ.get("ROUTING") or "balanced"
    if args.routing not in ROUTINGS:
        parser.error(f"不支持的 ROUTING {args.routing}")
    args.check_tokens = os.environ.get("CHECK_TOKENS") or "0"
    args.golden_workers = os.environ.get("GOLDEN_WORKERS") or "2"
    dual_reference = os.environ.get("DUAL_REFERENCE") or "0"
    if args.mode in ("accuracy", "both"):
        if not re.fullmatch(r"0|[1-9][0-9]*", args.check_tokens) or not re.fullmatch(r"[1-9][0-9]*", args.golden_workers):
            parser.error("CHECK_TOKENS 必须是非负整数，GOLDEN_WORKERS 必须是正整数")
        if dual_reference not in ("0", "1"):
            parser.error("DUAL_REFERENCE 必须是 0 或 1")
    args.dual_reference = dual_reference == "1"

    config = Path(args.config)
    if not config.is_file():
        config = CONFIG_DIR / f"{args.config}.json"
    if not config.is_file():
        parser.error(f"找不到配置 {args.config}")
    args.config = config.resolve()
    try:
        args.spec = json.loads(args.config.read_text(encoding="utf-8"))
        model = args.spec["model"]
        if model["activation"] != "gelu" or model["num_shared_experts"] != 0:
            parser.error("UBX 入口使用 GELU 零共享专家配置作为模型模板")
        if model["num_experts"] % args.ep_size:
            parser.error("专家数必须能被 EP size 整除")
        if args.topk is not None and args.topk > model["num_experts"]:
            parser.error("--topk 不能超过全局专家数")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"无法读取 {args.config}：{error}")

    wanted = tuple(MMA_TYPES) if args.precision in (None, "all") else (args.precision,)
    args.accuracy_precisions = wanted if args.mode in ("accuracy", "both") else ()
    args.perf_precisions = wanted if args.mode == "compare" else ()
    if args.mode in ("perf", "both"):
        args.perf_precisions = wanted if args.precision else ("w4a8",)
    if args.comparison_arm == "swiglu" and (
        args.mode != "perf" or len(args.perf_precisions) != 1
        or swiglu_unavailable(args.perf_precisions[0])
    ):
        parser.error("SwiGLU 基线需要 UBX EP16/EP32 W4A8")
    if args.window_tokens is not None and args.mode != "compare" and not args.paired_weights:
        parser.error("--window-tokens 用于 compare")
    return args


def swiglu_unavailable(precision):
    if precision != "w4a8":
        return "当前 UBX W8A8 只支持分窗 GELU，缺少原生 SwiGLU 后置加权实现"
    return None


def ubx_timer_ranks(ep_size):
    count = min(TIMER_RANKS_PER_NODE, LOCAL_RANKS)
    return [
        node_offset + sample * LOCAL_RANKS // count
        for node_offset in range(0, ep_size, LOCAL_RANKS)
        for sample in range(count)
    ]


def comparison_plan(args):
    model = dict(args.spec["model"])
    if args.topk is not None:
        model["num_topk"] = args.topk
    config = dict(args.spec, model=model, bs_values=[args.bs], warmup=max(2, args.spec["warmup"]))
    cases = []
    pairs = args.pairs or 8
    for precision in args.perf_precisions:
        blocked = swiglu_unavailable(precision)
        base = f"comparison/{precision}"
        arms = ("gelu",) if blocked else tuple(WEIGHT_MODE)
        pair_runs = []
        if not blocked:
            for index in range(pairs):
                pair_runs.append(dict(
                    pair=index,
                    **{arm: f"{base}/pairs/{index:02d}/{arm}/perf-{precision}" for arm in pair_order(index)},
                ))
        cases.append(dict(
            precision=precision,
            pairs=pair_runs,
            absolute={"gelu": f"{base}/absolute/gelu/perf-{precision}"} if blocked else {},
            timer={arm: f"{base}/timer/{arm}/perf-{precision}" for arm in arms},
            blocked={"swiglu": blocked} if blocked else {},
        ))
    return dict(
        schema="ubx_ab_plan_v1", ep_size=args.ep_size, bs=args.bs, routing=args.routing, config=config,
        pairs=pairs, cases=cases, timer_ranks=ubx_timer_ranks(args.ep_size),
        window_tokens=min(args.bs, args.window_tokens or 256),
        measurement_order=["formal_host_e2e", "first_window_timer", "report"],
    )


def report_comparison(args, plan):
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        from ubx_ab_report import build_report, print_report
    finally:
        sys.path.pop(0)
    report = build_report(plan, [args.output_dir, *args.peer_output_dir])
    output = args.output_dir / COMPARISON_REPORT
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print_report(report)
    print(f"[COMPARE] 报告 {output}", flush=True)
    return int(args.require_no_regression and (report["status"] != "pass" or plan["pairs"] < 7))


def run_logged(command, env, log):
    with subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def config_for(args, precision):
    model = dict(args.spec["model"], mma_type=MMA_TYPES[precision])
    spec = dict(args.spec, model=model)
    if args.topk is not None:
        model["num_topk"] = args.topk
    if args.comparison_arm is not None:
        model.update(activation=args.comparison_arm, topk_weights_type=WEIGHT_MODE[args.comparison_arm])
        spec["warmup"] = max(2, spec["warmup"])
    elif args.weighting != "auto":
        model["topk_weights_type"] = int(args.weighting == "early")
    if spec == args.spec:
        return args.config
    path = args.output_dir / f"config-{precision}-node{args.node_rank}.json"
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return path


def case_args(args, config):
    return ["--config", str(config), "--bs-values", str(args.bs), "--input", "fp8", "--routing", args.routing]


def prepare_golden(python, env, args, precision, config, *, reuse_only=False):
    first = args.node_rank * LOCAL_RANKS
    summary = args.output_dir / f"golden-{precision}-node{args.node_rank}.json"
    command = [
        python, "-P", str(SCRIPT.with_name("ubx_prepare_golden.py")), *case_args(args, config),
        "--golden-dir", env["GOLDEN_DIR"], "--check-tokens", args.check_tokens, "--ep-size", str(args.ep_size),
        "--ranks", f"{first}-{first + LOCAL_RANKS - 1}", "--workers", args.golden_workers,
        "--chunk-tokens", "128", "--summary", str(summary),
    ]
    if reuse_only:
        command.append("--reuse-only")
    elif args.force_cpu or args.force:
        command.append("--force")
    log_path = args.output_dir / f"golden-{precision}-node{args.node_rank}.log"
    with log_path.open("w" if reuse_only else "a", encoding="utf-8") as log:
        run_logged(command, env, log)
    return json.loads(summary.read_text(encoding="utf-8"))


def runtime_identity(python, env):
    probe = subprocess.run(
        [python, "-P", str(SCRIPT.with_name("ubx_accuracy_identity.py"))],
        env=env, stdout=subprocess.PIPE, text=True,
    )
    identity = probe.stdout.strip()
    if probe.returncode == 0 and re.fullmatch(r"[a-f0-9]{64}", identity):
        return identity
    print("[UBX] 无法确认当前运行环境，本轮重新执行 NPU 精度测试。", flush=True)
    return None


def report_lines(directory, args, mode, precision):
    rank_dir = directory / f"rank_{args.node_rank * LOCAL_RANKS}"
    manifest_path = rank_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["backend"] != "urma" or manifest["ep_size"] != args.ep_size:
        raise ValueError(f"{manifest_path} 预期 URMA EP{args.ep_size}，实际为 {manifest['backend']} EP{manifest['ep_size']}")
    mma_type = MMA_TYPES[precision]
    if manifest["config"]["model"]["mma_type"] != mma_type:
        raise ValueError(f"{manifest_path} 的矩阵乘精度与本轮要求 {mma_type} 不符")
    case_dir = rank_dir / f"bs_{args.bs}"
    if mode == "accuracy":
        return [f"[ACCURACY] 全组报告 {case_dir / 'accuracy.json'}"]
    if args.perf_scope == "timer":
        return [f"[TIMER] 诊断目录 {directory}"]
    report_path = case_dir / "performance.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    host = report["host_e2e"]
    lines = []
    if args.perf_scope == "all":
        lines.append(
            f"[PERF] 每轮 {report['windows_per_forward']} 窗，最慢 rank 的 kernel 耗时均值 "
            f"{report['rank_max_mean_us'] / 1000:.3f} ms"
        )
    return [
        *lines,
        f"[PERF] 最慢 rank 的同步端到端均值 {host['rank_max_mean_us'] / 1000:.3f} ms，"
        f"全组吞吐量 {host['global_input_tokens_per_second']:,.0f} token/s",
        f"[PERF] 全组报告 {report_path}",
    ]


def run_case(python, env, args, mode, precision, config):
    name = f"{mode}-{precision}"
    output = args.output_dir / name
    if (output / f"rank_{args.node_rank * LOCAL_RANKS}").exists():
        raise ValueError(f"{output} 已有本节点的报告，请指定新的 --output-dir")
    extra = []
    if mode == "accuracy":
        extra = ["--golden-dir", env["GOLDEN_DIR"], "--check-tokens", args.check_tokens]
        if args.dual_reference:
            extra.append("--dual-reference")
    if args.perf_scope != "all":
        extra.extend(("--perf-scope", args.perf_scope))
    if args.paired_weights:
        extra.extend(("--paired-weights", "--urma-window-tokens", str(args.window_tokens or 256)))
    if args.timer:
        ranks = ",".join(map(str, ubx_timer_ranks(args.ep_size)))
        extra.extend(("--timer", "--timer-ranks", ranks))
    if args.ep_size == 16:
        rendezvous = [
            "--nnodes=1", "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:0",
            "--local-addr=127.0.0.1", f"--rdzv-id={uuid4()}",
        ]
    else:
        run_id = cache_key([str(args.output_dir), name])[:16]
        rendezvous = [
            "--nnodes=2", f"--node-rank={args.node_rank}", f"--master-addr={args.master_addr}",
            f"--master-port={args.master_port}", f"--rdzv-id={run_id}",
        ]
    command = [
        python, "-c", "from torch.distributed.run import main; main()", *rendezvous,
        f"--nproc-per-node={LOCAL_RANKS}", "--max-restarts=0", "--module", f"{MODULE}.benchmark",
        *case_args(args, config), "--mode", mode, "--output-dir", str(output), *extra,
    ]
    print(f"[UBX] node{args.node_rank} {mode} {MMA_TYPES[precision]}", flush=True)
    with (args.output_dir / f"{name}-node{args.node_rank}.log").open("w", encoding="utf-8") as log:
        run_logged(command, env, log)
        for line in report_lines(output, args, mode, precision):
            print(line, flush=True)
            print(line, file=log)
    return output


def run_single_accuracy(python, env, args, precision, config, check_device):
    spec = json.loads(config.read_text(encoding="utf-8"))
    inputs = dict(ep_size=args.ep_size, bs=args.bs, input="fp8", routing=args.routing, check_tokens=int(args.check_tokens))
    lock_id = cache_key(dict(inputs, model=spec["model"], seed=spec["seed"]))
    lock_path = Path(env["GOLDEN_DIR"]) / ".ubx-launch-locks" / f"{lock_id}.lock"

    def cache_entry(golden, runtime):
        return AccuracyCache(args.output_dir.parent, dict(
            inputs, config=golden["config"], golden=golden["identity"], runtime=runtime,
            dual_reference=args.dual_reference,
        ))

    with file_lock(lock_path):
        force_npu = args.force or args.force_npu or args.timer
        if args.force_cpu and not force_npu:
            golden = prepare_golden(python, env, args, precision, config)
        else:
            golden = prepare_golden(python, env, args, precision, config, reuse_only=True)
            if force_npu:
                previous_cache = cache_entry(golden, None)
                with previous_cache.lock():
                    previous_cache.invalidate()
                check_device(precision)
            if golden["missing"] or args.force or args.force_cpu:
                check_device(precision)
                golden = prepare_golden(python, env, args, precision, config)
        runtime = runtime_identity(python, env)
        result_cache = cache_entry(golden, runtime)
        with result_cache.lock():
            previous = result_cache.lookup() if runtime and not force_npu else None
            if previous is not None:
                lines = [
                    f"[UBX] 复用 {precision.upper()} NPU 精度结果",
                    *report_lines(previous, args, "accuracy", precision),
                ]
                message = "\n".join(lines) + "\n"
                (args.output_dir / f"accuracy-{precision}-node0.log").write_text(message, encoding="utf-8")
                print(message, end="", flush=True)
                return
            result_cache.invalidate()
            check_device(precision)
            output = run_case(python, env, args, "accuracy", precision, config)
            if runtime and runtime_identity(python, env) == runtime:
                result_cache.publish(output)


def orchestrate(args, env):
    extra_env = {name: env[name] for name in FORWARDED_ENV if name in env}
    if "PYTHON" in extra_env and os.sep in extra_env["PYTHON"]:
        extra_env["PYTHON"] = os.path.abspath(extra_env["PYTHON"])
    print(f"[UBX] node0 本机 {args.node_ip[0]}，node1 SSH {args.ssh_user}@{args.node_ip[1]}", flush=True)
    stages = []
    for precision in args.accuracy_precisions:
        stages.extend((("prepare", "accuracy", precision), ("run", "accuracy", precision)))
    stages.extend(("run", "perf", precision) for precision in args.perf_precisions)
    for stage, mode, precision in stages:
        argv = [
            str(SCRIPT), mode, precision, "--ep-size", "32", "--stage", stage,
            "--master-addr", args.master_addr, "--master-port", str(args.master_port),
            "--config", str(args.config), "--bs", str(args.bs), "--weighting", args.weighting,
            "--output-dir", str(args.output_dir),
        ]
        if stage == "prepare" and (args.force or args.force_cpu):
            argv.append("--force-cpu")
        if stage == "run" and args.timer:
            argv.append("--timer")
        if args.topk is not None:
            argv.extend(("--topk", str(args.topk)))
        if args.comparison_arm is not None:
            argv.extend(("--comparison-arm", args.comparison_arm, "--perf-scope", args.perf_scope))
        if args.paired_weights:
            argv.append("--paired-weights")
            if args.window_tokens is not None:
                argv.extend(("--window-tokens", str(args.window_tokens)))
        scripts = {
            rank: node_script(REPO_ROOT, extra_env, ENV_SCRIPT, [*argv, "--node-rank", str(rank)])
            for rank in (0, 1)
        }
        label = f"{stage}-{mode}-{precision}"
        logs = {rank: args.output_dir / f"launch-{label}-node{rank}.log" for rank in scripts}
        run_stage(label, scripts, logs, ssh_target=f"{args.ssh_user}@{args.node_ip[1]}", ssh_port=args.ssh_port)


def run_comparison(args, env, python, check_device):
    plan = comparison_plan(args)
    plan_path = args.output_dir / COMPARISON_PLAN
    with plan_path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    def run_job(case, arm, relative_path, scope):
        job = copy(args)
        job.mode = "perf"
        job.output_dir = args.output_dir / Path(relative_path).parent
        job.output_dir.mkdir(parents=True, exist_ok=True)
        job.comparison_arm = arm
        job.perf_scope = scope
        job.timer = scope == "timer"
        job.paired_weights = not case["blocked"]
        precision = case["precision"]
        job.perf_precisions = (precision,)
        if job.node_rank is None:
            orchestrate(job, env)
        else:
            check_device(precision)
            run_case(python, env, job, "perf", precision, config_for(job, precision))

    for case in plan["cases"]:
        for arm, reason in case["blocked"].items():
            print(f"[COMPARE] {case['precision']} {arm} BLOCKED: {reason}", flush=True)
        for pair in case["pairs"]:
            for arm in pair_order(pair["pair"]):
                run_job(case, arm, pair[arm], "host")
        for arm, relative in case["absolute"].items():
            run_job(case, arm, relative, "host")
    for case in plan["cases"]:
        for arm, relative in case["timer"].items():
            run_job(case, arm, relative, "timer")
    return report_comparison(args, plan)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    if args.plan:
        print(json.dumps(comparison_plan(args), indent=2, ensure_ascii=False))
        return 0
    if args.report_only:
        args.output_dir = args.output_dir.resolve(strict=True)
        plan = json.loads((args.output_dir / COMPARISON_PLAN).read_text(encoding="utf-8"))
        return report_comparison(args, plan)
    python = os.environ.get("PYTHON")
    if args.node_rank is not None and not python:
        command = 'set -e; source "$1"; shift; exec "$PYTHON" "$@"'
        os.execvp("bash", ["bash", "-c", command, "ubx-env", str(ENV_SCRIPT), str(SCRIPT), *argv])

    env = os.environ.copy()
    env.pop("PYTHON_EXEC", None)  # Probe, torchrun and its workers must use the same Python.
    env["PYTHONSAFEPATH"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = "1"
    env["GOLDEN_DIR"] = str(Path(env.get("GOLDEN_DIR") or REPO_ROOT / "benchmark-golden").resolve())
    addr = args.master_addr if args.ep_size == 32 else "127.0.0.1"
    env["DEEPEP_SHMEM_ENDPOINT"] = env.get("DEEPEP_SHMEM_ENDPOINT") or f"tcp://{addr}:29180"
    if args.ep_size == 32 and env["DEEPEP_SHMEM_ENDPOINT"] == f"tcp://{addr}:{args.master_port}":
        raise ValueError("torchrun 与 SHMEM 必须使用不同的端口")
    if args.output_dir is None:
        base = Path(env.get("RUN_BASE") or REPO_ROOT / "benchmark-runs").resolve()
        base.mkdir(parents=True, exist_ok=True)
        args.output_dir = Path(tempfile.mkdtemp(prefix=f"ubx-ep{args.ep_size}-gelu-{args.routing}-", dir=base))
    else:
        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[UBX] EP{args.ep_size} B{args.bs} {args.routing} {args.config.name}\n[UBX] 输出目录 {args.output_dir}", flush=True)
    @cache
    def check_device(precision):
        subprocess.run(
            [python, "-c", DEVICE_CHECK, str(LOCAL_RANKS), str(int(precision == "w8a8"))], env=env, check=True,
        )

    if args.mode == "compare":
        return run_comparison(args, env, python, check_device)
    if args.node_rank is None:
        orchestrate(args, env)
        return

    for precision in args.accuracy_precisions:
        config = config_for(args, precision)
        if args.ep_size == 16:
            run_single_accuracy(python, env, args, precision, config, check_device)
        else:
            check_device(precision)
            if args.stage == "prepare":
                prepare_golden(python, env, args, precision, config)
            else:
                run_case(python, env, args, "accuracy", precision, config)
    if args.stage != "prepare":
        for precision in args.perf_precisions:
            check_device(precision)
            run_case(python, env, args, "perf", precision, config_for(args, precision))


def cli(argv=None):
    try:
        return main(argv) or 0
    except subprocess.CalledProcessError as error:
        print(f"[UBX] 子进程退出，状态 {error.returncode}。", file=sys.stderr)
        return error.returncode if error.returncode >= 0 else 128 - error.returncode
    except (OSError, ValueError) as error:
        print(f"[UBX] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
