# Copyright (c) 2026, Lu Lu

"""在单机 UBX EP16 运行 GELU 精度和性能测试。"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[3]
TARGET_CONFIG = SCRIPT.parents[1] / "python/deep_ep_experimental/megamoe/configs/target_gelu_k25.json"
MODULE = "deep_ep_experimental.megamoe"
EP, BS = 16, 39280
MMA_TYPES = {"w4a8": "fp8xfp4", "w8a8": "fp8xfp8"}

DEVICE_CHECK = """
import sys

import torch
import torch_npu

from deep_ep._machine import board_id, physical_device
from deep_ep_experimental.megamoe.api import _extension

ep = int(sys.argv[1])
count = torch.npu.device_count()
if count < ep:
    sys.exit(f"UBX EP{ep} 需要 {ep} 张可见 NPU，当前只有 {count} 张。")
torch.npu.set_device(0)
board = board_id(physical_device())
if board != 0x44:
    sys.exit(f"可见设备 0 的板型为 0x{board:02x}，UBX 要求 0x44。")
extension = _extension()
if sys.argv[2] == "1" and not getattr(extension, "urma_w8a8_enabled", False):
    sys.exit("当前 MegaMoE wheel 缺少 UBX URMA W8A8，请用本次源码重建并安装 wheel。")
print(f"[UBX] MegaMoE wheel: {extension.__file__}", flush=True)
"""


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "单机 UBX EP16 GELU，固定每卡 39280 token、800 个路由专家、TopK 25、"
            "GMM1 4096/1536、GMM2 1536/4096。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""环境变量
  PYTHON                  已加载 CANN 时可指定该环境的 Python，默认加载 ubx_env.sh
  ROUTING                 balanced（默认）、skewed、random 或 random-balanced
  CHECK_TOKENS            每卡精度抽查数，默认 0（全量对比 CPU Golden）
  DUAL_REFERENCE          设为 1 时额外启用 NPU 级联双参考，默认 0
  GOLDEN_WORKERS          CPU Golden 工作线程数，默认 2
  GOLDEN_DIR              CPU Golden 缓存，默认仓库下的 benchmark-golden
  RUN_BASE                输出根目录，默认仓库下的 benchmark-runs
  DEEPEP_SHMEM_ENDPOINT   SHMEM 引导地址，默认 tcp://127.0.0.1:29180""",
    )
    parser.add_argument("mode", nargs="?", choices=("accuracy", "perf", "both"), default="both", help="默认 both")
    parser.add_argument("precision", nargs="?", choices=("all", *MMA_TYPES), help="精度测试的权重类型，默认 all")
    args = parser.parse_args(argv)
    if args.mode == "perf" and args.precision is not None:
        parser.error("perf 使用 W4A8，precision 适用于 accuracy 和 both 的精度阶段")

    args.routing = os.environ.get("ROUTING") or "balanced"
    if args.routing not in ("balanced", "skewed", "random", "random-balanced"):
        parser.error(f"不支持的 ROUTING {args.routing}")
    args.accuracy_precisions = ()
    if args.mode != "perf":
        args.accuracy_precisions = tuple(MMA_TYPES) if args.precision in (None, "all") else (args.precision,)
        args.check_tokens = os.environ.get("CHECK_TOKENS") or "0"
        args.golden_workers = os.environ.get("GOLDEN_WORKERS") or "2"
        dual_reference = os.environ.get("DUAL_REFERENCE") or "0"
        if not re.fullmatch(r"0|[1-9][0-9]*", args.check_tokens) or not re.fullmatch(r"[1-9][0-9]*", args.golden_workers):
            parser.error("CHECK_TOKENS 必须是非负整数，GOLDEN_WORKERS 必须是正整数")
        if dual_reference not in ("0", "1"):
            parser.error("DUAL_REFERENCE 必须是 0 或 1")
        args.dual_reference = dual_reference == "1"
    return args


def run_logged(command, env, log):
    with subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
    ) as process:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def report_lines(output_dir, mode, mma_type):
    rank_dir = output_dir / "rank_0"
    manifest_path = rank_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 启动器已集体确认各 rank 的后端和配置一致，核对 rank 0 的记录即可。
    if manifest["backend"] != "urma" or manifest["ep_size"] != EP:
        raise ValueError(f"{manifest_path} 预期 UBX URMA EP{EP}，实际为 {manifest['backend']} EP{manifest['ep_size']}")
    if manifest["config"]["model"]["mma_type"] != mma_type:
        raise ValueError(f"{manifest_path} 的矩阵乘精度与本轮要求 {mma_type} 不符")

    case_dir = rank_dir / f"bs_{BS}"
    if mode == "accuracy":
        return [f"[ACCURACY] 全组报告 {case_dir / 'accuracy.json'}"]

    path = case_dir / "performance.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    host = report["host_e2e"]
    kernel_ms = report["rank_max_mean_us"] / 1000
    host_ms = host["rank_max_mean_us"] / 1000
    throughput = host["global_input_tokens_per_second"]
    return [
        f"[PERF] 每轮 {report['windows_per_forward']} 窗，最慢 rank 的 kernel 耗时均值 {kernel_ms:.3f} ms",
        f"[PERF] 最慢 rank 的同步端到端均值 {host_ms:.3f} ms，全组吞吐量 {throughput:,.0f} token/s",
        f"[PERF] 全组报告 {path}",
    ]


def run_case(python, env, run_dir, args, mode, precision, config):
    mma_type = MMA_TYPES[precision]
    case_name = f"accuracy-{precision}" if mode == "accuracy" else "perf"
    output_dir = run_dir / case_name
    case_args = ["--config", str(config), "--bs-values", str(BS), "--input", "fp8", "--routing", args.routing]
    accuracy_args = []
    if mode == "accuracy":
        golden_dir = env.get("GOLDEN_DIR") or str(REPO_ROOT / "benchmark-golden")
        accuracy_args = ["--check-tokens", args.check_tokens, "--golden-dir", golden_dir]
        golden = [
            python, "-m", f"{MODULE}.golden", *case_args, *accuracy_args,
            "--ep-size", str(EP), "--workers", args.golden_workers, "--chunk-tokens", "128",
        ]
        # 全量 Golden 按 rank 生成，避免同时保留 16 个 rank 的参考张量。
        ranks = map(str, range(EP)) if args.check_tokens == "0" else (f"0-{EP - 1}",)
        with (run_dir / f"golden-{precision}.log").open("w", encoding="utf-8") as log:
            for rank in ranks:
                run_logged([*golden, "--ranks", rank], env, log)
        if args.dual_reference:
            accuracy_args.append("--dual-reference")

    launch = [
        python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc-per-node={EP}",
        "--max-restarts=0", "--module", f"{MODULE}.benchmark", *case_args, "--mode", mode,
        "--output-dir", str(output_dir), *accuracy_args,
    ]
    print(f"[UBX] {mode} {mma_type}", flush=True)
    with (run_dir / f"{case_name}.log").open("w", encoding="utf-8") as log:
        run_logged(launch, env, log)
        for line in report_lines(output_dir, mode, mma_type):
            print(line, flush=True)
            print(line, file=log)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    python = os.environ.get("PYTHON")
    if not python:
        # Conda 和 CANN 的动态库路径需要在重新启动 Python 前设置。
        command = 'set -e; source "$1"; shift; exec "$PYTHON" "$@"'
        os.execvp("bash", ["bash", "-c", command, "ubx-env", str(SCRIPT.with_name("ubx_env.sh")), str(SCRIPT), *argv])

    env = os.environ.copy()
    env["DEEPEP_SHMEM_ENDPOINT"] = env.get("DEEPEP_SHMEM_ENDPOINT") or "tcp://127.0.0.1:29180"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = "1"
    # 保留 CANN 设置的 PYTHONPATH，同时避免从当前目录导入仓库源码。
    env["PYTHONSAFEPATH"] = "1"
    want_w8a8 = "w8a8" in args.accuracy_precisions
    # 在独立进程检查 NPU，检查结束后释放设备上下文再启动 16 个 rank。
    subprocess.run([python, "-c", DEVICE_CHECK, str(EP), str(int(want_w8a8))], env=env, check=True)

    run_base = Path(env.get("RUN_BASE") or REPO_ROOT / "benchmark-runs")
    run_base.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"ubx-ep16-gelu-{args.routing}-", dir=run_base))
    print(f"[UBX] EP{EP} B{BS} E800 K25 H4096 I1536 GELU {args.routing}\n[UBX] 输出目录 {run_dir}", flush=True)

    for precision in args.accuracy_precisions:
        config = TARGET_CONFIG
        if precision == "w8a8":
            case = json.loads(TARGET_CONFIG.read_text(encoding="utf-8"))
            case["model"]["mma_type"] = MMA_TYPES[precision]
            config = run_dir / "target_gelu_k25_w8a8.json"
            config.write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
        run_case(python, env, run_dir, args, "accuracy", precision, config)
    if args.mode != "accuracy":
        run_case(python, env, run_dir, args, "perf", "w4a8", TARGET_CONFIG)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"[UBX] 子进程退出，状态 {error.returncode}。", file=sys.stderr)
        sys.exit(error.returncode if error.returncode >= 0 else 128 - error.returncode)
