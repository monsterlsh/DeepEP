# Copyright (c) 2026, Lu Lu

"""Launch MegaMoE tests on UBX or 950POD using a JSON run configuration."""

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
POD_RANKS = 8
POD_PYTHON = "/opt/miniconda3/envs/lsx/bin/python"
POD_CANN_ENV = "/home/cann/b112/cann-9.2.0/set_env.sh"
TIMER_RANKS_PER_NODE = 4
MMA_TYPES = {"w4a8": "fp8xfp4", "w8a8": "fp8xfp8", "w4a4": "fp4xfp4"}
MODES = ("accuracy", "perf", "both", "compare")
ROUTINGS = ("balanced", "skewed", "random", "random-balanced")
FORWARDED_ENV = (
    "PYTHON", "UBX_CANN_SETUP", "ROUTING", "CHECK_TOKENS", "DUAL_REFERENCE", "GOLDEN_WORKERS",
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
backend = sys.argv[3] if len(sys.argv) > 3 else "urma"
precision = sys.argv[4] if len(sys.argv) > 4 else None
require_roce_timer = len(sys.argv) > 5 and sys.argv[5] == "1"
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
if backend == "mte_roce" and not getattr(extension, "mte_roce_enabled", False):
    sys.exit("当前 MegaMoE wheel 未启用 EP32 MTE+RoCE，请在两台主机安装同版本的新版 wheel。")
if backend == "mte_roce" and not getattr(extension, "mte_roce_mx_enabled", False):
    sys.exit("当前 MegaMoE wheel 未启用新版 MTE+RoCE 矩阵精度与分块支持。")
if require_roce_timer and not getattr(extension, "mte_roce_timer_enabled", False):
    sys.exit("MTE+RoCE 设备 timer 需要两台节点安装支持此功能并启用 DEEPEP_MEGAMOE_TIMER 的 wheel。")
if backend == "mte" and precision == "w4a4" and not getattr(extension, "mte_a4w4_enabled", False):
    sys.exit("当前 MegaMoE wheel 未启用单机 MTE W4A4。")
print(f"[UBX] MegaMoE wheel: {extension.__file__}", flush=True)
"""

POD_BOOTSTRAP = r"""
set -e
source "$1"
if ! python=$(command -v -- "$2"); then
    printf '[950 POD] 找不到 Python %s\n' "$2" >&2
    exit 1
fi
base_lib=$3
shift 3
export PATH="$(dirname -- "$python"):$PATH"
if [[ -d "$base_lib" ]]; then
    export LD_LIBRARY_PATH="$base_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$python" "$@"
"""

POD_DEVICE_CHECK = """
import sys
import torch
import torch_npu

from deep_ep._machine import board_id, physical_device
from deep_ep_experimental.megamoe.api import _extension

required = int(sys.argv[1])
count = torch.npu.device_count()
if count < required:
    sys.exit(f"950 POD 每机需要 {required} 张可见 NPU，当前只有 {count} 张。")
for rank in range(required):
    torch.npu.set_device(rank)
    if board_id(physical_device()) == 0x44:
        sys.exit(f"可见设备 {rank} 属于 UBX，请在 JSON 中使用 machine=ubx。")
print(f"[950 POD] MegaMoE wheel: {_extension().__file__}", flush=True)
"""


def parse_precision(value):
    name = value.casefold()
    if name == "w8a4":
        raise argparse.ArgumentTypeError("FP4 激活 × FP8 权重暂不支持；FP8 激活 × FP4 权重请选择 a8w4")
    return {"a8w4": "w4a8", "a8w8": "w8a8", "a4w4": "w4a4"}.get(name, name)


def comparison_settings(config):
    defaults = {arm: dict(activation=arm, topk_weights_type=weight) for arm, weight in WEIGHT_MODE.items()}
    if not isinstance(config, dict) or config.keys() - {"arms", "precisions"}:
        raise ValueError("comparison 支持 arms 和 precisions")
    arms = config.get("arms", defaults)
    if not isinstance(arms, dict) or arms.keys() != defaults.keys():
        raise ValueError("comparison.arms 需要 gelu 和 swiglu 两组")
    for name, arm in arms.items():
        if (not isinstance(arm, dict) or type(arm.get("topk_weights_type")) is not int
                or arm != defaults[name]):
            raise ValueError("comparison.arms 要求 GELU 提前加权（1）和 SwiGLU Finalize 加权（0）")
    precisions = config.get("precisions", ["w4a8", "w8a8"])
    if (not isinstance(precisions, list) or not precisions
            or any(value not in ("w4a8", "w8a8") for value in precisions)
            or len(set(precisions)) != len(precisions)):
        raise ValueError("comparison.precisions 需要不重复的 w4a8 或 w8a8")
    return arms, tuple(precisions)


def validate_geometry(model, ep_size):
    for dimension in ("hidden", "intermediate_hidden"):
        if type(model[dimension]) is not int or model[dimension] <= 0:
            raise ValueError(f"{dimension} 必须为正整数")
    experts = model["num_experts"]
    if type(experts) is not int or experts <= 0 or experts % ep_size:
        raise ValueError(f"专家数必须为正整数且能被 EP{ep_size} 整除")
    topk = model["num_topk"]
    if type(topk) is not int or not 1 <= topk <= min(32, experts):
        raise ValueError("TopK 必须在 1..32 且不能超过全局专家数")


def read_run_config(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("运行配置要求 schema_version=1")
    data.pop("_comments", None)
    return data


def run_config_arguments(path, parser, argv, cli, data=None):
    """Use the command parser for JSON options and keep explicit CLI overrides."""
    strings = {"mode", "precision", "backend", "activation", "weighting", "input", "stage", "master_addr",
               "ssh_user", "config", "output_dir", "comparison_arm", "perf_scope"}
    integers = {"ep_size", "node_rank", "master_port", "shmem_port", "ssh_port", "bs", "topk", "window_tokens", "pairs"}
    flags = {"timer", "force", "force_cpu", "force_npu", "plan", "report_only", "require_no_regression", "paired_weights"}
    env_types = dict(python=str, cann_env=str, golden_dir=str, run_base=str, routing=str, check_tokens=int,
                     golden_workers=int, dual_reference=bool)
    try:
        if data is None:
            data = read_run_config(path)
        supported = strings | integers | flags | env_types.keys() | {
            "schema_version", "machine", "node_ip", "peer_output_dir", "comparison",
        }
        if unknown := sorted(data.keys() - supported):
            raise ValueError(f"未知运行配置字段 {unknown}")
        comparison = comparison_settings(data.get("comparison", {}))
        present = {token.partition("=")[0] for token in argv}
        explicit_node = "--node-ip" in present or "--node-rank" in present
        def local_path(value):
            return str((path.parent / Path(value).expanduser()).resolve())

        arguments, environment = [], {}
        for key, value in data.items():
            if key in ("schema_version", "machine", "comparison"):
                continue
            option = "--" + key.replace("_", "-")
            if key in env_types:
                if type(value) is not env_types[key]:
                    raise ValueError(f"{key} 必须为 {env_types[key].__name__}")
                if key in {"cann_env", "golden_dir", "run_base"}:
                    value = local_path(value)
                if key == "python" and ("/" in value or "\\" in value):
                    value = os.path.abspath(path.parent / Path(value).expanduser())
                env_name = "UBX_CANN_SETUP" if key == "cann_env" else key.upper()
                environment[env_name] = str(int(value)) if type(value) is bool else str(value)
            elif option in present or key in ("mode", "precision") and getattr(cli, key) is not None:
                continue
            elif key == "config" and "--inline-config" in present:
                continue
            elif key == "config" and isinstance(value, dict):
                arguments.append(f"--inline-config={json.dumps(value, separators=(',', ':'))}")
            elif key in ("node_ip", "node_rank") and explicit_node:
                continue
            elif key in flags:
                if type(value) is not bool:
                    raise ValueError(f"{key} 必须为布尔值")
                if value:
                    arguments.append(option)
            elif key in integers | strings:
                expected = int if key in integers else str
                if type(value) is not expected:
                    raise ValueError(f"{key} 必须为 {expected.__name__}")
                if key == "output_dir" or key == "config" and (Path(value).suffix == ".json" or Path(value).parent != Path(".")):
                    value = local_path(value)
                arguments.append(f"{option}={value}")
            else:
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    raise ValueError(f"{key} 必须为字符串数组")
                if key == "node_ip":
                    if len(value) != 2:
                        raise ValueError("node_ip 必须包含两台节点的 IPv4")
                    arguments.extend((option, *value))
                else:
                    arguments.extend(f"{option}={local_path(item)}" for item in value)
    except (OSError, ValueError, TypeError) as error:
        parser.error(f"无法读取运行配置 {path}：{error}")
    return arguments, environment, comparison


def parse_args(argv, run_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, help="包含运行参数的 JSON，可用显式命令行参数覆盖")
    parser.add_argument("mode", nargs="?", choices=MODES)
    parser.add_argument("--mode", dest="mode_option", choices=MODES,
                        help="accuracy 精度，perf 性能，compare A/B 性能和掩盖率，both 精度加性能")
    parser.add_argument("precision", nargs="?", type=parse_precision, choices=("all", *MMA_TYPES),
                        help="显式指定时适用于所选模式。省略时精度测两种，性能测 W4A8")
    parser.add_argument("--precision", dest="precision_option", type=parse_precision, choices=("all", *MMA_TYPES),
                        help="显式选择 w8a8、w4a8（a8w4）、w4a4（a4w4）或 all")
    parser.add_argument("--ep-size", type=int, choices=(16, 32), default=16)
    parser.add_argument("--backend", choices=("urma", "mte", "mte_roce"), default="urma",
                        help="默认 URMA；mte 为单机 EP16，mte_roce 为双机 EP32")
    parser.add_argument("--activation", choices=("gelu", "swiglu"), help="覆盖模型激活函数")
    parser.add_argument("--input", choices=("fp8", "fp4"), help="默认按激活精度选 FP8 或 packed FP4")
    node = parser.add_mutually_exclusive_group()
    node.add_argument("--node-ip", nargs=2, metavar=("NODE0", "NODE1"),
                      help="EP32 自动拉起的两台节点")
    node.add_argument("--node-rank", type=int, choices=(0, 1), help="EP32 手动分阶段拉起时的节点序号")
    parser.add_argument("--stage", choices=("prepare", "run"), help="EP32 手动模式先 prepare，再 run")
    parser.add_argument("--master-addr", help="本机或手动启动的 IPv4，自动 EP32 默认用 node_ip 的第一项")
    parser.add_argument("--master-port", type=int, default=os.environ.get("MASTER_PORT", "29500"))
    parser.add_argument("--shmem-port", type=int, default=29180, help="指定 SHMEM 引导端口")
    parser.add_argument("--ssh-user", default=os.environ.get("REMOTE_USER") or getpass.getuser())
    parser.add_argument("--ssh-port", type=int, default=os.environ.get("REMOTE_PORT", "22"))
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--config", help="配置名（如 target_gelu_k32）或模型 JSON 路径")
    model_source.add_argument("--inline-config", help=argparse.SUPPRESS)
    parser.add_argument("--weighting", choices=("auto", "early", "late"), default="auto")
    parser.add_argument("--bs", type=int, help="每卡 token 数，默认读取模型 JSON 的单个 bs_values 值")
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
    parser.add_argument("--comparison-arm", choices=WEIGHT_MODE,
                        help="在 accuracy 或 perf 中单测 GELU 提前加权或 SwiGLU Finalize 加权")
    parser.add_argument("--perf-scope", choices=("all", "host", "timer"), default="all", help=argparse.SUPPRESS)
    parser.add_argument("--paired-weights", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    run_environment = {}
    comparison = comparison_settings({})
    if args.run_config is not None:
        config_argv, run_environment, comparison = run_config_arguments(
            args.run_config.resolve(), parser, argv, args, run_config,
        )
        args = parser.parse_args([*config_argv, *argv])
    args.run_environment = run_environment
    args.arm_models, comparison_precisions = comparison
    if args.mode and args.mode_option:
        parser.error("位置模式参数与 --mode 二选一")
    args.mode = args.mode_option or args.mode or "both"
    if args.precision and args.precision_option:
        parser.error("位置精度参数与 --precision 二选一")
    args.precision = args.precision_option or args.precision

    roce = args.backend == "mte_roce"
    if roce and args.ep_size != 32:
        parser.error("MTE+RoCE 需要 --ep-size 32，两台各 16 卡")
    if args.backend == "mte" and args.ep_size != 16:
        parser.error("本机 MTE 需要 --ep-size 16")
    if args.backend != "urma":
        if args.mode == "compare" or args.comparison_arm is not None or args.paired_weights:
            parser.error("MTE 和 MTE+RoCE 使用独立精度/性能轮，A/B 参数适用于 URMA")
        if args.perf_scope != "all":
            parser.error("MTE 和 MTE+RoCE 使用默认性能采样范围")
    elif args.precision == "w4a4":
        parser.error("URMA 暂不支持 W4A4。EP16 使用 --backend mte，EP32 使用 --backend mte_roce")
    if args.ep_size == 16:
        if args.node_ip or args.node_rank is not None or args.stage:
            parser.error("--node-ip、--node-rank、--stage 适用于 EP32")
        if args.master_addr is None:
            if args.run_config is not None:
                parser.error("UBX EP16 需要在 JSON 中设置 master_addr")
            args.master_addr = "127.0.0.1"
        args.node_rank = 0
    else:
        if (args.node_rank is None) != (args.stage is None):
            parser.error("手动 EP32 需配套使用 --node-rank 和 --stage")
        if args.stage and args.output_dir is None:
            parser.error("手动 EP32 需指定 --output-dir")
        if args.force_npu:
            parser.error("EP32 的 NPU 测试每次实测，--force-npu 适用于 EP16")
        if args.node_rank is None:
            if args.node_ip is None:
                parser.error("自动启动 UBX EP32 需要设置 node_ip")
            try:
                args.node_ip = tuple(str(IPv4Address(ip)) for ip in args.node_ip)
            except ValueError as error:
                parser.error(f"node_ip: {error}")
            if args.node_ip[0] == args.node_ip[1]:
                parser.error("EP32 需要两个不同的节点 IP")
            if args.master_addr is None:
                args.master_addr = args.node_ip[0]
        elif args.master_addr is None:
            parser.error("手动启动 UBX EP32 需要设置 master_addr")
        if not (1 <= args.ssh_port <= 65535 and 1024 <= args.master_port <= 65535):
            parser.error("SSH 或 torchrun 端口超出范围")

    try:
        args.master_addr = str(IPv4Address(args.master_addr))
    except ValueError as error:
        parser.error(f"master_addr: {error}")
    if not 1024 <= args.shmem_port <= 65535:
        parser.error("--shmem-port 必须在 1024..65535")
    if args.ep_size == 32 and args.shmem_port == args.master_port:
        parser.error("torchrun 与 SHMEM 必须使用不同的端口")
    if args.topk is not None and not 1 <= args.topk <= 32:
        parser.error("--topk 必须在 1..32")
    if args.mode == "compare":
        if args.stage or args.comparison_arm:
            parser.error("compare 从主节点启动，节点阶段由入口管理")
        if args.activation is not None or args.weighting != "auto" or args.force or args.force_cpu or args.force_npu:
            parser.error("compare 固定两侧加权方式，且每轮均实测")
        if args.pairs is not None and args.pairs < 2:
            parser.error("compare 至少需要 2 对独立作业")
        if args.require_no_regression and args.pairs is not None and args.pairs < 7 and not args.report_only:
            parser.error("验收要求至少 7 对独立作业")
        if args.report_only and args.output_dir is None:
            parser.error("--report-only 需指定本轮 --output-dir")
        if args.peer_output_dir and not args.report_only:
            parser.error("--peer-output-dir 仅用于 --report-only")
    elif any((args.pairs is not None, args.report_only, args.peer_output_dir, args.require_no_regression)):
        parser.error("配对和离线汇总参数适用于 compare")
    if args.mode == "perf" and (args.force or args.force_cpu or args.force_npu or args.stage == "prepare"):
        parser.error("Golden 准备和强制重跑选项用于精度模式")
    settings = os.environ | args.run_environment
    args.routing = settings.get("ROUTING") or "balanced"
    if args.routing not in ROUTINGS:
        parser.error(f"不支持的 ROUTING {args.routing}")
    args.check_tokens = settings.get("CHECK_TOKENS") or "0"
    args.golden_workers = settings.get("GOLDEN_WORKERS") or "2"
    dual_reference = settings.get("DUAL_REFERENCE") or "0"
    if args.mode in ("accuracy", "both"):
        if not re.fullmatch(r"0|[1-9][0-9]*", args.check_tokens) or not re.fullmatch(r"[1-9][0-9]*", args.golden_workers):
            parser.error("CHECK_TOKENS 必须是非负整数，GOLDEN_WORKERS 必须是正整数")
        if dual_reference not in ("0", "1"):
            parser.error("DUAL_REFERENCE 必须是 0 或 1")
    args.dual_reference = dual_reference == "1"

    if args.inline_config is None:
        name = args.config or ("ep32_roce_decode" if roce else "target_gelu_k25")
        config = Path(name)
        if not config.is_file():
            config = CONFIG_DIR / f"{name}.json"
        if not config.is_file():
            parser.error(f"找不到配置 {name}")
        args.config = config.resolve()
    config_source = args.config or args.run_config or "内嵌模型 JSON"
    try:
        args.spec = json.loads(args.config.read_text(encoding="utf-8") if args.config else args.inline_config)
        if not isinstance(args.spec, dict):
            raise ValueError("模型配置必须是 JSON 对象")
        if args.bs is None:
            batches = args.spec["bs_values"]
            if args.inline_config is not None and (not isinstance(batches, list) or len(batches) != 1):
                parser.error("内嵌模型的 bs_values 需包含一个值，也可以用 --bs 指定本轮 BS")
            args.bs = batches[0] if len(batches) == 1 else (32 if roce else 39280)
        if type(args.bs) is not int or not 1 <= args.bs <= 65536:
            parser.error("--bs 必须在 1..65536")
        if args.spec["model"].get("activation", "swiglu") not in WEIGHT_MODE:
            parser.error("模型 activation 必须是 gelu 或 swiglu")
        model = case_model(args)
        activation = model.get("activation", "swiglu")
        weight_mode = model.get("topk_weights_type", WEIGHT_MODE[activation])
        if roce and model["num_shared_experts"] and activation != "swiglu":
            parser.error("MTE+RoCE 共享专家需要 W4A8 SwiGLU")
        if args.backend == "urma" and (model["num_shared_experts"] or activation == "swiglu" and weight_mode != 0):
            parser.error("URMA 入口要求零共享专家，SwiGLU 使用 Finalize 加权")
        validate_geometry(model, args.ep_size)
    except (OSError, ValueError, AttributeError, KeyError, TypeError) as error:
        parser.error(f"无法读取 {config_source}：{error}")

    supported = ("w4a8", "w8a8") if args.backend == "urma" else tuple(MMA_TYPES)
    wanted = supported if args.precision in (None, "all") else (args.precision,)
    if args.mode == "compare" and args.precision is None:
        wanted = comparison_precisions
    elif roce and args.precision is None:
        wanted = ("w4a8",)
    args.accuracy_precisions = wanted if args.mode in ("accuracy", "both") else ()
    args.perf_precisions = wanted if args.mode == "compare" else ()
    if args.mode in ("perf", "both"):
        args.perf_precisions = wanted if args.precision else ("w4a8",)
    if args.comparison_arm is not None and args.mode not in ("accuracy", "perf"):
        parser.error("--comparison-arm 用于单测 accuracy 或 perf")
    if args.comparison_arm is not None and args.activation is not None:
        parser.error("--comparison-arm 与 --activation 二选一")
    if args.window_tokens is not None and args.mode != "compare" and not args.paired_weights:
        parser.error("--window-tokens 用于 compare")
    selected = set(args.accuracy_precisions) | set(args.perf_precisions)
    if args.input is not None and any(input_kind(precision) != args.input for precision in selected):
        parser.error("W4A4 需要 fp4 输入，W4A8/W8A8 需要 fp8 输入，混合精度可省略 --input")
    if args.timer and args.backend == "mte" and selected - {"w4a8"}:
        parser.error("本机 MTE 设备 timer 需要 W4A8")
    if model["num_shared_experts"] and selected - {"w4a8"}:
        parser.error("共享专家要求 W4A8 路由专家")
    return args


def ubx_timer_ranks(ep_size):
    count = min(TIMER_RANKS_PER_NODE, LOCAL_RANKS)
    return [
        node_offset + sample * LOCAL_RANKS // count
        for node_offset in range(0, ep_size, LOCAL_RANKS)
        for sample in range(count)
    ]


def comparison_plan(args):
    config = dict(args.spec, model=case_model(args), bs_values=[args.bs], warmup=max(2, args.spec["warmup"]))
    cases = []
    pairs = args.pairs or 8
    for precision in args.perf_precisions:
        base = f"comparison/{precision}"
        pair_runs = [
            dict(pair=index, **{arm: f"{base}/pairs/{index:02d}/{arm}/perf-{precision}" for arm in pair_order(index)})
            for index in range(pairs)
        ]
        cases.append(dict(
            precision=precision,
            pairs=pair_runs,
            absolute={},
            timer={arm: f"{base}/timer/{arm}/perf-{precision}" for arm in args.arm_models},
            blocked={},
        ))
    return dict(
        schema="ubx_ab_plan_v1", ep_size=args.ep_size, bs=args.bs, routing=args.routing, config=config,
        arms=args.arm_models, pairs=pairs, cases=cases, timer_ranks=ubx_timer_ranks(args.ep_size),
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


def case_model(args):
    model = dict(args.spec["model"])
    if args.topk is not None:
        model["num_topk"] = args.topk
    if args.comparison_arm is not None:
        model.update(args.arm_models[args.comparison_arm])
    else:
        if args.activation is not None and args.activation != model.get("activation", "swiglu"):
            model.update(args.arm_models[args.activation])
        if args.weighting != "auto":
            model["topk_weights_type"] = int(args.weighting == "early")
    return model


def config_for(args, precision):
    model = dict(case_model(args), mma_type=MMA_TYPES[precision])
    spec = dict(args.spec, model=model)
    if args.comparison_arm is not None:
        spec["warmup"] = max(2, spec["warmup"])
    if args.config and spec == args.spec:
        return args.config
    if args.config is None:
        spec["bs_values"] = [args.bs]
    path = args.output_dir / f"config-{precision}-node{args.node_rank}.json"
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return path


def input_kind(precision):
    return "fp4" if precision == "w4a4" else "fp8"


def case_args(args, precision, config):
    return ["--config", str(config), "--bs-values", str(args.bs), "--input", input_kind(precision), "--routing", args.routing]


def prepare_golden(python, env, args, precision, config, *, reuse_only=False):
    first = args.node_rank * LOCAL_RANKS
    summary = args.output_dir / f"golden-{precision}-node{args.node_rank}.json"
    command = [
        python, "-P", str(SCRIPT.with_name("ubx_prepare_golden.py")), *case_args(args, precision, config),
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
    if manifest["backend"] != args.backend or manifest["ep_size"] != args.ep_size:
        raise ValueError(f"{manifest_path} 预期 {args.backend} EP{args.ep_size}，实际为 {manifest['backend']} EP{manifest['ep_size']}")
    mma_type = MMA_TYPES[precision]
    if manifest["config"]["model"]["mma_type"] != mma_type:
        raise ValueError(f"{manifest_path} 的矩阵乘精度与本轮要求 {mma_type} 不符")
    case_dir = rank_dir / f"bs_{args.bs}"
    if mode == "accuracy":
        return [f"[ACCURACY] 全组报告 {case_dir / 'accuracy.json'}"]
    if args.perf_scope == "timer":
        return [f"[TIMER] 诊断目录 {directory}"]
    report_path = case_dir / ("performance.json" if args.backend == "urma" else "profiling.json")
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
    if args.backend != "urma":
        extra.extend(("--backend", args.backend))
    if args.perf_scope != "all":
        extra.extend(("--perf-scope", args.perf_scope))
    if args.paired_weights:
        extra.extend(("--paired-weights", "--urma-window-tokens", str(args.window_tokens or 256)))
    if args.timer:
        ranks = ",".join(map(str, ubx_timer_ranks(args.ep_size)))
        extra.extend(("--timer", "--timer-ranks", ranks))
    if args.ep_size == 16:
        rendezvous = [
            "--nnodes=1", "--rdzv-backend=c10d", f"--rdzv-endpoint={args.master_addr}:0",
            f"--local-addr={args.master_addr}", f"--rdzv-id={uuid4()}",
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
        *case_args(args, precision, config), "--mode", mode, "--output-dir", str(output), *extra,
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
    inputs = dict(ep_size=args.ep_size, bs=args.bs, input=input_kind(precision), routing=args.routing,
                  check_tokens=int(args.check_tokens))
    lock_id = cache_key(dict(inputs, model=spec["model"], seed=spec["seed"]))
    lock_path = Path(env["GOLDEN_DIR"]) / ".ubx-launch-locks" / f"{lock_id}.lock"

    def cache_entry(golden, runtime):
        backend = {"backend": args.backend} if args.backend != "urma" else {}
        return AccuracyCache(args.output_dir.parent, dict(
            inputs, config=golden["config"], golden=golden["identity"], runtime=runtime,
            dual_reference=args.dual_reference, **backend,
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
            if force_npu or golden["missing"]:
                check_device(precision)
            if golden["missing"] or args.force or args.force_cpu:
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
    config_args = (["--config", str(args.config)] if args.config else
                   ["--inline-config", json.dumps(args.spec, separators=(",", ":"))])
    for stage, mode, precision in stages:
        argv = [
            str(SCRIPT), "--worker", mode, precision, "--ep-size", "32", "--stage", stage,
            "--master-addr", args.master_addr, "--master-port", str(args.master_port),
            "--shmem-port", str(args.shmem_port),
            *config_args, "--bs", str(args.bs), "--weighting", args.weighting,
            "--output-dir", str(args.output_dir),
        ]
        if args.backend != "urma":
            argv.extend(("--backend", args.backend))
        if args.activation is not None:
            argv.extend(("--activation", args.activation))
        if stage == "prepare" and (args.force or args.force_cpu):
            argv.append("--force-cpu")
        if stage == "run" and args.timer:
            argv.append("--timer")
        if args.topk is not None:
            argv.extend(("--topk", str(args.topk)))
        if args.comparison_arm is not None:
            argv.extend(("--comparison-arm", args.comparison_arm))
        if args.comparison_arm is not None or args.perf_scope != "all":
            argv.extend(("--perf-scope", args.perf_scope))
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
        job.paired_weights = True
        precision = case["precision"]
        job.perf_precisions = (precision,)
        if job.node_rank is None:
            orchestrate(job, env)
        else:
            check_device(precision)
            run_case(python, env, job, "perf", precision, config_for(job, precision))

    for case in plan["cases"]:
        for pair in case["pairs"]:
            for arm in pair_order(pair["pair"]):
                run_job(case, arm, pair[arm], "host")
    for case in plan["cases"]:
        for arm, relative in case["timer"].items():
            run_job(case, arm, relative, "timer")
    return report_comparison(args, plan)


def main(argv=None, run_config=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv, run_config)
    if args.plan:
        plan = comparison_plan(args) if args.mode == "compare" else dict(
            machine="ubx", backend=args.backend, ep_size=args.ep_size, node_ip=args.node_ip,
            master_addr=args.master_addr, master_port=args.master_port, shmem_port=args.shmem_port,
            mode=args.mode, accuracy_precisions=args.accuracy_precisions, perf_precisions=args.perf_precisions,
            timer_ranks=ubx_timer_ranks(args.ep_size) if args.timer else [],
            routing=args.routing, config=dict(args.spec, model=case_model(args), bs_values=[args.bs]),
        )
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if args.report_only:
        args.output_dir = args.output_dir.resolve(strict=True)
        plan = json.loads((args.output_dir / COMPARISON_PLAN).read_text(encoding="utf-8"))
        return report_comparison(args, plan)
    env = os.environ | args.run_environment
    python = env.get("PYTHON")
    cann_env = env.get("UBX_CANN_SETUP")
    if args.node_rank is not None and (not python or cann_env and cann_env != env.get("UBX_CANN_LOADED")):
        environment_script = cann_env if python else str(ENV_SCRIPT)
        command = 'set -e; source "$1"; export UBX_CANN_LOADED="${UBX_CANN_SETUP:-$1}"; shift; exec "$PYTHON" "$@"'
        worker = ["--worker"] if args.run_config is None else []
        os.execvpe("bash", ["bash", "-c", command, "ubx-env", environment_script, str(SCRIPT), *worker, *argv], env)
    env.pop("PYTHON_EXEC", None)  # Probe, torchrun and its workers must use the same Python.
    env["PYTHONSAFEPATH"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = "1"
    env["GOLDEN_DIR"] = str(Path(env.get("GOLDEN_DIR") or REPO_ROOT / "benchmark-golden").resolve())
    env["DEEPEP_SHMEM_ENDPOINT"] = f"tcp://{args.master_addr}:{args.shmem_port}"
    if args.output_dir is None:
        base = Path(env.get("RUN_BASE") or REPO_ROOT / "benchmark-runs").resolve()
        base.mkdir(parents=True, exist_ok=True)
        workload = args.activation or args.spec["model"].get("activation", "swiglu")
        args.output_dir = Path(tempfile.mkdtemp(prefix=f"ubx-ep{args.ep_size}-{workload}-{args.routing}-", dir=base))
    else:
        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
    config_name = (args.config or args.run_config).name if args.config or args.run_config else "内嵌模型 JSON"
    print(f"[UBX] {args.backend} EP{args.ep_size} B{args.bs} {args.routing} {config_name}\n[UBX] 输出目录 {args.output_dir}", flush=True)
    @cache
    def check_device(precision):
        extra = [args.backend, precision] if args.backend != "urma" else []
        if args.backend == "mte_roce" and args.timer and args.stage != "prepare":
            extra.append("1")
        subprocess.run(
            [python, "-c", DEVICE_CHECK, str(LOCAL_RANKS), str(int(precision == "w8a8" and args.backend == "urma")), *extra],
            env=env, check=True,
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


def cli(argv=None, run_config=None):
    try:
        return main(argv, run_config) or 0
    except subprocess.CalledProcessError as error:
        print(f"[UBX] 子进程退出，状态 {error.returncode}。", file=sys.stderr)
        return error.returncode if error.returncode >= 0 else 128 - error.returncode
    except (OSError, ValueError) as error:
        print(f"[UBX] {error}", file=sys.stderr)
        return 1


def pod_settings(path, data):
    settings = dict(
        ep_size=8, node_rank=0, master_addr="",
        master_port=int(os.environ.get("MASTER_PORT") or 29181),
        shmem_port=int(os.environ.get("SHMEM_PORT") or 29180),
        routing=os.environ.get("ROUTING") or "balanced", check_tokens=int(os.environ.get("CHECK_TOKENS") or 0),
        golden_workers=int(os.environ.get("GOLDEN_WORKERS") or 2), dual_reference=os.environ.get("DUAL_REFERENCE") == "1",
        golden_dir=os.environ.get("GOLDEN_DIR") or str(REPO_ROOT / "benchmark-golden"),
        run_base=os.environ.get("RUN_BASE") or str(REPO_ROOT / "benchmark-runs"), output_dir="",
        python=os.environ.get("PYTHON") or "", base_python=os.environ.get("BASE_PYTHON") or POD_PYTHON,
        cann_env=os.environ.get("CANN_ENV") or POD_CANN_ENV, backend="mte",
        golden_only=False, skip_golden=False, plan=False,
    )
    unknown = data.keys() - settings.keys() - {"schema_version", "machine", "mode", "precision", "config", "bs"}
    if unknown:
        raise ValueError(f"950POD 运行配置不支持字段 {sorted(unknown)}")
    for name, default in settings.items():
        if name in data:
            value = data[name]
            if type(value) is not type(default):
                raise ValueError(f"{name} 必须为 {type(default).__name__}")
            if value and (name in {"golden_dir", "run_base", "output_dir", "cann_env"} or
                          name in {"python", "base_python"} and ("/" in value or "\\" in value)):
                value = str((path.parent / Path(value).expanduser()).resolve())
            settings[name] = value
    args = argparse.Namespace(**settings)
    args.machine, args.mode, args.precision = data["machine"], data["mode"], parse_precision(data["precision"])
    if args.backend != "mte" or args.mode not in {"accuracy", "perf"} or args.precision not in {"w4a8", "w8a8", "all"}:
        raise ValueError("950POD 使用 MTE，mode 支持 accuracy 或 perf，precision 支持 a8w4、a8w8 或 all")
    if args.ep_size not in (8, 16, 32) or not 0 <= args.node_rank < args.ep_size // POD_RANKS:
        raise ValueError("950POD 支持 EP8/16/32，每机 8 卡，node_rank 从 0 开始")
    if not args.master_addr:
        raise ValueError("950POD 需要在 JSON 中设置 master_addr")
    args.master_addr = IPv4Address(args.master_addr)
    if args.ep_size > POD_RANKS and (not args.output_dir or args.master_addr.is_loopback):
        raise ValueError("950POD 多机需要 output_dir 和其他节点可访问的 master_addr")
    if any(not 1024 <= port <= 65535 for port in (args.master_port, args.shmem_port)) or args.master_port == args.shmem_port:
        raise ValueError("torchrun 和 SHMEM 端口必须不同且在 1024..65535")
    if args.golden_only and args.skip_golden:
        raise ValueError("golden_only 和 skip_golden 互斥")
    if args.mode == "perf" and args.golden_only:
        raise ValueError("golden_only 仅用于 950POD accuracy")
    if args.routing not in ROUTINGS or args.check_tokens < 0 or args.golden_workers < 1:
        raise ValueError("检查 routing、check_tokens 和 golden_workers")

    args.spec = data["config"]
    model = args.spec["model"]
    if model.get("activation") != "gelu":
        raise ValueError("950POD 入口需要 GELU 模型")
    validate_geometry(model, args.ep_size)
    if "bs" in data:
        args.bs = data["bs"]
    else:
        batches = args.spec.get("bs_values")
        if not isinstance(batches, list) or len(batches) != 1:
            raise ValueError("内嵌模型的 bs_values 需包含一个值，也可以用 bs 指定本轮 BS")
        args.bs = batches[0]
    if type(args.bs) is not int or not 1 <= args.bs <= 65536:
        raise ValueError("bs 必须在 1..65536")
    args.python = args.python or args.base_python
    return args


def pod_perf_report(args, output, first, spec, precision):
    for rank in range(first, first + POD_RANKS):
        rank_dir = output / f"rank_{rank}"
        manifest_path = rank_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = dict(backend="mte", mode="perf", ep_size=args.ep_size, rank=rank,
                        ranks_per_node=POD_RANKS, routing=args.routing, input=input_kind(precision))
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{manifest_path} 与本轮 950POD 性能配置不符")
        settings = manifest["config"]
        if settings.get("bs_values") != [args.bs] or any(
            settings["model"].get(key) != value for key, value in spec["model"].items()
        ):
            raise ValueError(f"{manifest_path} 的模型或 BS 与本轮配置不符")
        report_path = rank_dir / f"bs_{args.bs}" / "profiling.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        host = report["host_e2e"]
        times = (report["rank_max_mean_us"], host["rank_max_mean_us"], host["global_input_tokens_per_second"])
        if any(type(value) not in (int, float) or not 0 < value < float("inf") for value in times):
            raise ValueError(f"{report_path} 的 kernel、host_e2e 时延和吞吐必须为有限正值")
        if rank == first:
            summary = (report_path, host["rank_max_mean_us"], host["global_input_tokens_per_second"])
    path, latency, throughput = summary
    return f"[PERF] 最慢 rank 的同步端到端均值 {latency / 1000:.3f} ms，全组吞吐量 {throughput:,.0f} token/s\n[PERF] 全组报告 {path}"


def run_pod_case(args, precision, python, env, first):
    label = f"{precision}-node{args.node_rank}"
    spec = dict(args.spec, model=dict(args.spec["model"], mma_type=MMA_TYPES[precision]), bs_values=[args.bs])
    config = args.output_dir / f"config-{label}.json"
    config.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    options = case_args(args, precision, config)
    if args.mode == "accuracy":
        options.extend(("--check-tokens", str(args.check_tokens), "--golden-dir", str(Path(args.golden_dir).resolve())))
    if args.mode == "accuracy" and not args.skip_golden:
        command = [python, "-I", "-m", f"{MODULE}.golden", *options, "--ep-size", str(args.ep_size),
                   "--workers", str(args.golden_workers), "--chunk-tokens", "128"]
        # Bound each CPU invocation to at most one full rank of selected rows.
        batch = args.bs // min(args.check_tokens or args.bs, args.bs)
        with (args.output_dir / f"golden-{label}.log").open("w", encoding="utf-8") as log:
            for start in range(first, first + POD_RANKS, batch):
                last = min(start + batch, first + POD_RANKS) - 1
                run_logged([*command, "--ranks", f"{start}-{last}"], env, log)
    if args.golden_only:
        return

    output = args.output_dir / (f"{args.mode}-{precision}" if args.precision == "all" else args.mode)
    command = [
        python, "-I", "-c", "from torch.distributed.run import main; main()", "--rdzv-backend=static",
        f"--nnodes={args.ep_size // POD_RANKS}", f"--nproc-per-node={POD_RANKS}", f"--node-rank={args.node_rank}",
        f"--master-addr={args.master_addr}", f"--master-port={args.master_port}", "--max-restarts=0",
        "--module", f"{MODULE}.benchmark", *options, "--mode", args.mode, "--output-dir", str(output),
    ]
    if args.mode == "accuracy" and args.dual_reference:
        command.append("--dual-reference")
    with (args.output_dir / f"{args.mode}-{label}.log").open("w", encoding="utf-8") as log:
        run_logged(command, env, log)
        if args.mode == "perf":
            summary = pod_perf_report(args, output, first, spec, precision)
            print(summary, file=log)
            print(summary, flush=True)
            return
    rank_dir = output / f"rank_{first}"
    manifest = json.loads((rank_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["backend"] != "mte" or manifest["ep_size"] != args.ep_size:
        raise ValueError(f"{rank_dir} 预期 MTE EP{args.ep_size}，实际为 {manifest['backend']} EP{manifest['ep_size']}")
    report = rank_dir / f"bs_{args.bs}" / "accuracy.json"
    if not report.is_file():
        raise FileNotFoundError(f"精度报告缺失 {report}")
    print(f"[ACCURACY] 全组报告 {report}", flush=True)


def run_pod(path, data):
    args = pod_settings(path, data)
    if args.plan:
        print(json.dumps(vars(args), default=str, indent=2, ensure_ascii=False))
        return 0
    if not os.environ.get("DEEPEP_950POD_BOOTSTRAPPED"):
        env = dict(os.environ, DEEPEP_950POD_BOOTSTRAPPED="1")
        base_lib = str(Path(args.base_python).parent.parent / "lib")
        os.execvpe("bash", ["bash", "-c", POD_BOOTSTRAP, "950pod-env", args.cann_env, args.python, base_lib,
                           str(SCRIPT), "--run-config", str(path)], env)

    python = sys.executable
    env = dict(os.environ, DEEPEP_SHMEM_ENDPOINT=f"tcp://{args.master_addr}:{args.shmem_port}",
               PYTHONSAFEPATH="1", PYTHONNOUSERSITE="1", PYTHON_EXEC=python)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = "1"
    if not args.golden_only:
        subprocess.run([python, "-I", "-c", POD_DEVICE_CHECK, str(POD_RANKS)], env=env, check=True)
    if args.output_dir:
        args.output_dir = Path(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        base = Path(args.run_base).resolve()
        base.mkdir(parents=True, exist_ok=True)
        args.output_dir = Path(tempfile.mkdtemp(prefix="950pod-gelu-", dir=base))
    first = args.node_rank * POD_RANKS
    print(f"[950 POD] EP{args.ep_size} rank {first}-{first + POD_RANKS - 1} {path.name} "
          f"B{args.bs} {args.routing}\n[950 POD] 输出目录 {args.output_dir}", flush=True)
    precisions = ("w4a8", "w8a8") if args.precision == "all" else (args.precision,)
    for precision in precisions:
        run_pod_case(args, precision, python, env, first)
    return 0


def entrypoint(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--worker"]:
        if not {"--inline-config", "--stage", "--node-rank"} <= set(argv):
            raise SystemExit("内部 UBX worker 缺少节点阶段或内嵌模型")
        return cli(argv[1:])

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-config", required=True, type=Path, help="包含机型和全部测试参数的 JSON")
    args = parser.parse_args(argv)
    path = args.run_config.resolve()
    try:
        data = read_run_config(path)
        if data.get("machine") not in ("ubx", "950pod"):
            raise ValueError("machine 必须为 ubx 或 950pod")
        if data.get("mode") not in MODES or not isinstance(data.get("precision"), str):
            raise ValueError("需要在 JSON 中指定 mode 和 precision")
        spec = data.get("config")
        if (not isinstance(spec, dict) or not isinstance(spec.get("model"), dict)
                or type(spec.get("schema_version")) is not int or spec["schema_version"] != 1):
            raise ValueError("config 必须包含 schema_version=1 的内嵌模型 JSON")
        if data["machine"] == "ubx":
            data.setdefault("backend", "mte_roce" if data.get("ep_size") == 32 else "mte")
            return cli(["--run-config", str(path)], data)
        return run_pod(path, data)
    except subprocess.CalledProcessError as error:
        print(f"[950 POD] 子进程退出，状态 {error.returncode}。", file=sys.stderr)
        return error.returncode if error.returncode >= 0 else 128 - error.returncode
    except (OSError, KeyError, ValueError, argparse.ArgumentTypeError) as error:
        parser.error(f"{path}: {error}")


if __name__ == "__main__":
    sys.exit(entrypoint())
