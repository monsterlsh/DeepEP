# Copyright (c) 2026, Lu Lu

"""在非 UBX Ascend 950 POD 上运行 MegaMoE GELU 精度测试。"""

import argparse
from ipaddress import IPv4Address
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[3]
CONFIG_DIR = SCRIPT.parents[1] / "python/deep_ep_experimental/megamoe/configs"
MODULE = "deep_ep_experimental.megamoe"
B05_IP = "177.22.21.1"
B05_PYTHON = "/opt/miniconda3/envs/lsx/bin/python"
B05_CANN_ENV = "/home/cann/b112/cann-9.2.0/set_env.sh"
RANKS_PER_NODE = 8

BOOTSTRAP = r"""
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

DEVICE_CHECK = """
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
        sys.exit(f"可见设备 {rank} 属于 UBX，请使用 UBX GELU 入口。")
print(f"[950 POD] MegaMoE wheel: {_extension().__file__}", flush=True)
"""


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            f"默认在 B05（{B05_IP}）本机运行 EP8。通过 PYTHON 指定已安装 MegaMoE wheel 的 Python，"
            f"默认 {B05_PYTHON}。BASE_PYTHON 可覆盖其基础环境。"
            f"通过 CANN_ENV 指定 CANN 环境脚本，默认 {B05_CANN_ENV}。"
        ),
    )
    parser.add_argument("--ep-size", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--node-rank", type=int, default=0, help="每机 8 卡，按机器从 0 编号")
    parser.add_argument("--master-addr", type=IPv4Address, default=os.environ.get("MASTER_ADDR") or B05_IP)
    parser.add_argument("--master-port", type=int, default=os.environ.get("MASTER_PORT") or "29181")
    parser.add_argument("--shmem-port", type=int, default=os.environ.get("SHMEM_PORT") or "29180")
    parser.add_argument("--config", default="target_gelu_k25", help="配置名或 GELU JSON 文件，默认 target_gelu_k25")
    parser.add_argument("--bs", type=int, default=39280, help="每卡 token 数，默认 39280")
    parser.add_argument("--routing", choices=("balanced", "skewed", "random", "random-balanced"),
                        default=os.environ.get("ROUTING") or "balanced")
    parser.add_argument("--check-tokens", type=int, default=os.environ.get("CHECK_TOKENS") or "0",
                        help="每卡抽查的 token 数，0 为全量，默认 0")
    parser.add_argument("--golden-workers", type=int, default=os.environ.get("GOLDEN_WORKERS") or "2")
    parser.add_argument("--dual-reference", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("DUAL_REFERENCE") == "1", help="加入 NPU 级联双参考")
    parser.add_argument("--golden-dir", type=Path, default=os.environ.get("GOLDEN_DIR") or REPO_ROOT / "benchmark-golden")
    parser.add_argument("--output-dir", type=Path, help="运行目录，多机使用相同绝对路径")
    preparation = parser.add_mutually_exclusive_group()
    preparation.add_argument("--golden-only", action="store_true", help="只准备本机 CPU Golden")
    preparation.add_argument("--skip-golden", action="store_true", help="直接用已经准备的 CPU Golden 跑精度")
    parser.add_argument("--plan", action="store_true", help="打印本机参数与阶段，无 NPU 调用或文件写入")
    args = parser.parse_args(argv)

    nodes = args.ep_size // RANKS_PER_NODE
    if not 0 <= args.node_rank < nodes:
        parser.error(f"EP{args.ep_size} 要求 --node-rank 在 0..{nodes - 1}")
    if nodes > 1 and args.output_dir is None:
        parser.error("多机运行需要 --output-dir")
    if args.master_addr.is_loopback and nodes > 1:
        parser.error("多机运行需要其他节点可访问的 --master-addr")
    if not all(1024 <= port <= 65535 for port in (args.master_port, args.shmem_port)) or args.master_port == args.shmem_port:
        parser.error("torchrun 和 SHMEM 端口必须不同且在 1024..65535")
    if not 0 < args.bs <= 65536 or args.check_tokens < 0 or args.golden_workers < 1:
        parser.error("--bs 必须在 1..65536，--check-tokens 非负，--golden-workers 为正整数")
    if args.routing not in ("balanced", "skewed", "random", "random-balanced"):
        parser.error(f"不支持的 ROUTING {args.routing}")

    path = Path(args.config)
    if not path.is_file():
        path = CONFIG_DIR / f"{args.config}.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        model = config["model"]
        mma_type = model.get("mma_type", "fp8xfp4")
        if model.get("activation") != "gelu" or mma_type not in ("fp8xfp4", "fp8xfp8"):
            raise ValueError("需要 A8W4 或 A8W8 的 GELU 配置")
        if model["num_experts"] % args.ep_size:
            raise ValueError(f"全局专家数必须能被 EP{args.ep_size} 整除")
    except (OSError, KeyError, AttributeError, TypeError, ValueError) as error:
        parser.error(f"{path}: {error}")
    args.config = path.resolve()
    args.golden_dir = args.golden_dir.resolve()
    args.output_dir = args.output_dir.resolve() if args.output_dir else None
    args.base_python = os.environ.get("BASE_PYTHON") or B05_PYTHON
    args.python = os.environ.get("PYTHON") or args.base_python
    args.cann_env = os.environ.get("CANN_ENV") or B05_CANN_ENV
    return args


def case_args(args):
    return [
        "--config", str(args.config), "--bs-values", str(args.bs), "--input", "fp8", "--routing", args.routing,
        "--check-tokens", str(args.check_tokens), "--golden-dir", str(args.golden_dir),
    ]


def golden_commands(args, python):
    first = args.node_rank * RANKS_PER_NODE
    checked = min(args.check_tokens or args.bs, args.bs)
    ranks_per_batch = args.bs // checked
    command = [
        python, "-I", "-m", f"{MODULE}.golden", *case_args(args), "--ep-size", str(args.ep_size),
        "--workers", str(args.golden_workers), "--chunk-tokens", "128",
    ]
    # Bound the selected rows to at most one full rank per CPU invocation.
    for start in range(first, first + RANKS_PER_NODE, ranks_per_batch):
        last = min(start + ranks_per_batch, first + RANKS_PER_NODE) - 1
        yield [*command, "--ranks", f"{start}-{last}"]


def accuracy_command(args, python, output):
    command = [
        python, "-I", "-c", "from torch.distributed.run import main; main()", "--rdzv-backend=static",
        f"--nnodes={args.ep_size // RANKS_PER_NODE}", f"--nproc-per-node={RANKS_PER_NODE}",
        f"--node-rank={args.node_rank}", f"--master-addr={args.master_addr}", f"--master-port={args.master_port}",
        "--max-restarts=0", "--module", f"{MODULE}.benchmark", *case_args(args),
        "--mode", "accuracy", "--output-dir", str(output),
    ]
    if args.dual_reference:
        command.append("--dual-reference")
    return command


def run_logged(commands, env, path):
    with path.open("w", encoding="utf-8") as log:
        for command in commands:
            with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                status = process.wait()
            if status:
                raise subprocess.CalledProcessError(status, command)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    first = args.node_rank * RANKS_PER_NODE
    if args.plan:
        print(json.dumps(vars(args), default=str, indent=2))
        print(f"[950 POD] 本机 rank {first}-{first + RANKS_PER_NODE - 1}，SHMEM tcp://{args.master_addr}:{args.shmem_port}")
        return 0

    # Source CANN before restarting in the target Python, so its libraries are visible to the loader.
    if not os.environ.get("DEEPEP_950POD_BOOTSTRAPPED"):
        env = dict(os.environ, DEEPEP_950POD_BOOTSTRAPPED="1")
        os.execvpe("bash", ["bash", "-c", BOOTSTRAP, "950pod-env", args.cann_env, args.python,
                           str(Path(args.base_python).parent.parent / "lib"), str(SCRIPT), *argv], env)

    python = sys.executable
    env = dict(os.environ, DEEPEP_SHMEM_ENDPOINT=f"tcp://{args.master_addr}:{args.shmem_port}",
               PYTHONSAFEPATH="1", PYTHONNOUSERSITE="1", PYTHON_EXEC=python)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = "1"
    if not args.golden_only:
        subprocess.run([python, "-I", "-c", DEVICE_CHECK, str(RANKS_PER_NODE)], env=env, check=True)

    run_dir = args.output_dir
    if run_dir is None:
        run_base = Path(os.environ.get("RUN_BASE") or REPO_ROOT / "benchmark-runs").resolve()
        run_base.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="950pod-gelu-", dir=run_base))
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[950 POD] EP{args.ep_size} rank {first}-{first + RANKS_PER_NODE - 1} {args.config.stem} "
          f"B{args.bs} {args.routing}\n[950 POD] 输出目录 {run_dir}", flush=True)
    if not args.skip_golden:
        run_logged(golden_commands(args, python), env, run_dir / f"golden-node{args.node_rank}.log")
    if not args.golden_only:
        output = run_dir / "accuracy"
        run_logged([accuracy_command(args, python, output)], env, run_dir / f"accuracy-node{args.node_rank}.log")
        rank_dir = output / f"rank_{first}"
        manifest = json.loads((rank_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["backend"] != "mte" or manifest["ep_size"] != args.ep_size:
            raise ValueError(f"{rank_dir} 预期 MTE EP{args.ep_size}，实际为 {manifest['backend']} EP{manifest['ep_size']}")
        report = rank_dir / f"bs_{args.bs}" / "accuracy.json"
        if not report.is_file():
            raise FileNotFoundError(f"精度报告缺失 {report}")
        print(f"[ACCURACY] 全组报告 {report}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        print(f"[950 POD] 子进程退出，状态 {error.returncode}。", file=sys.stderr)
        sys.exit(error.returncode if error.returncode >= 0 else 128 - error.returncode)
    except (OSError, ValueError) as error:
        sys.exit(f"[950 POD] {error}")
