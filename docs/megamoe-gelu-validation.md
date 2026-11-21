# MegaMoE GELU / TopK32 验证交接手册

适用于 `feat/megamoe-topk`。目标是每卡 39,280 个 token，800 个路由专家，EP32（每卡 25 个专家），K25，另测 K32 上界。GMM1 为 4096→1536，GMM2 为 1536→4096。输入采用 FP8 E4M3 + E8M0，路由专家权重采用 FP4，输出为 BF16，共享专家数为零。路由权重乘在 GELU 之后、MXFP8 量化之前。

| 环境 | EP8 | EP16 | EP32 | manifest.backend |
| --- | --- | --- | --- | --- |
| 非 UBX Ascend 950，每机 8 卡 | 1×8 | 2×8 | 4×8 | `mte` |
| UBX，每机 16 卡 | 单机选 8 张卡 | 1×16 | 2×16 | `urma` |

后端由板型自动选择。UBX 窗口机内使用 MTE，跨机使用 HNS_1825 RoCE。同一节点使用一致的可见设备列表，每个进程绑定不同的本地设备，跨节点按 node rank 连续编号。B05、B07 各有 8 张非 UBX 卡，两机合用最多验证 EP16。

## 接手任务

从 `feat/megamoe-topk` 的待验提交为目标软件环境构建 wheel，同组节点使用同一产物。下表是 11 月 20 日单分支交接时的专项检查索引。本轮先验 UBX，再验 950。

| 索引 | 硬件与用例 | 专项检查 |
| --- | --- | --- |
| 1 | 一台空闲非 UBX 950 | 按第 2 节构建并运行 Linux 主机测试。保留完整 Git SHA、wheel SHA-256 和版本信息。 |
| 2 | 非 UBX EP8 | 第 3 节的窗口边界；再按第 4 节设 `NODES=1 LOCAL_RANKS=8`，跑 B39280、E800、K25/K32 的均衡及热点。记录生产接口实际选窗、尾窗、双参考和第 5 节的性能配对。 |
| 3 | B05+B07 或同型非 UBX EP16 | 两机各 8 卡。先跑 B257，再跑 B39280 的目标几何，覆盖 K25/K32、均衡及热点；验证跨节点结果、专家计数和配对时延。 |
| 4 | 真实 UBX EP8、单机 EP16 | 按第 2.2 节准备 UBX 编译环境并构建。EP8 跑第 3 节的 K 和 B 边界。EP16 增加热点和零接收专家的 rank，记录其计数及全组完成状态。在原 SwiGLU 支持的 EP8/16、K≤8 上配对。 |
| 5 | 真实 EP32 | UBX 两台 16 卡、非 UBX 四台 8 卡分别验收。运行每卡 B39280、E800、K25，并复测 K32；按路由分别交付全量精度、内存和端到端耗时。非 UBX 做原 SwiGLU 配对；UBX 的 EP32/K25 记录绝对性能，同配置基线留空并注明原因。 |

**交接时的证据（2026-11-20）**：B05 最新构建日志已产出 wheel，SHA-256 为 `3a07b4ae9f6e7bb7726203f4b879d1ce0f0ba0a0b86c81ab41e3caae643273ca`，本轮止于编译。第 6 节的非 UBX EP8 配对、K32 抽查及 EP16 小批次数据来自较早构建。当前生产自动选窗、大几何 EP16、两种真实 EP32 以及真实 UBX 的结果均需按上表取得。已有 wheel 只有在来源提交和摘要都能核实时才复用。

同日 Windows 工作区运行 `python -m pytest -q experiments/megamoe/tests`，322 项通过、9 项失败。其中 3 项涉及 POSIX 路径或启动命令，6 项在进入 SHMEM SDK 校验前触发了原生 wheel 的 Linux 平台检查。第 2 节包含同一套测试的 Linux 复验命令。

## 1. 可选：查看节点占用

如需记录手动实验时的机器负载，可在参与节点执行下列命令，查看 `vmstat` 后几次采样的 `id`、`wa`，以及 NPU 利用率和进程表。

```bash
hostname
vmstat 1 4
ps -eo user,pid,pcpu,comm --sort=-pcpu | head -20
npu-smi info
df -h "$PWD"
```

## 2. 编译并安装到独立环境

从远程 Linux 的功能分支仓库根目录 `/mnt/share/l00911701/DeepEP-UBX` 开始，在目标提交的干净检出中用 Bash 操作。使用 Linux aarch64、CANN 9.2.0 或更高版本，以及相互匹配的 Python、torch 和 torch_npu。以下示例使用 Python 3.11。分步构建时，按板型准备第 2.1 或 2.2 节的环境，再在同一终端执行第 2.3 节。

### 2.1 非 UBX Ascend 950 环境

把 CANN 脚本和基础 Python 的路径换成本机路径。验证环境通过 `--system-site-packages` 复用基础 Python 环境里的框架包。

```bash
set -eo pipefail
export CANN_ENV=/path/to/cann-9.2.0/set_env.sh
export BASE_PYTHON=/path/to/torch-npu-env/bin/python
source "$CANN_ENV"
"$BASE_PYTHON" -m venv --system-site-packages .venv-megamoe
source .venv-megamoe/bin/activate
export PYTHON="$VIRTUAL_ENV/bin/python"
export PATH="$VIRTUAL_ENV/bin:$(dirname "$BASE_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$BASE_PYTHON")")/lib:${LD_LIBRARY_PATH:-}"
```

### 2.2 UBX 环境

UBX 使用 Conda 环境 `py311_tpr` 和 `/usr/local/Ascend/cann-9.2.T101/set_env.sh`。确保 `conda` 在 PATH 中，且 `py311_tpr` 已安装与此 CANN 配套的 `torch` 和 `torch_npu`。在仓库根目录可直接构建并运行单机验证。

```bash
bash experiments/megamoe/scripts/build_ubx.sh
```

核对本机 CANN 安装路径时，运行 `bash experiments/megamoe/scripts/find_cann_env.sh`。默认从 `/` 扫描，也可传入一个或多个安装根目录。扫描会显示 UBX 配置路径是否可读，并把通过 Bash 语法检查的 CANN 候选输出为可复制的 `source` 命令。

构建脚本将 wheel 安装到 `py311_tpr`，打印 wheel 路径和 SHA-256。首次使用时初始化 `third-party/catlass`，后续使用同一路径上的锁定检出，保留 CMake 增量构建缓存。CATLASS 是编译时头文件，可通过 `CATLASS_ROOT` 指定匹配的已有检出。同一仓库内经 `build_ubx.sh` 启动的构建会串行使用这些产物。

`build_ubx.sh` 默认使用仓内锁定的 SHMEM 子模块。`SHMEM_ROOT` 指向外部 SDK 且其 `deepep-sdk.json` 记录的提交与 `dependencies.lock.json` 一致时，脚本沿用该路径，再执行原有的 SDK 校验。如果路径无效、清单无法读取或提交不符，脚本会提示原因并改用仓内 SDK。`SHMEM_ROOT` 指向仓内 `third-party/shmem/install/shmem` 时也走子模块流程，校验已有产物的构建身份。

仓内 SDK 的构建身份及文件校验通过后，脚本直接复用，即使已清理 SHMEM 的 CMake 中间目录。构建身份与当前环境不符时会报错并保留已有产物。需要重建时，先停止使用该 SDK 的作业，再执行下面的命令。`--rebuild-shmem` 始终选择仓内子模块并重建 SDK，即使终端已设置 `SHMEM_ROOT`。

```bash
bash experiments/megamoe/scripts/build_ubx.sh --rebuild-shmem
```

要试编官方 SHMEM 默认分支的最新提交，运行以下命令。

```bash
bash experiments/megamoe/scripts/build_ubx.sh --latest-shmem
```

`--latest-shmem` 将当前 DeepEP `HEAD` 的已提交源码复制到新的 `dist/shmem-latest-*/` 试编目录，并把官方 SHMEM 默认分支的实际提交及其头文件哈希写入试编目录的依赖锁文件。每轮在该目录独立保存源码、构建日志和 wheel。SHMEM 检出及 SDK 保存在 `dist/ubx-dependencies/shmem/`，按 SHMEM 提交和 CANN 环境缓存。`--latest-shmem` 与 `--rebuild-shmem` 互斥。

试编使用 `dist/ubx-dependencies/latest-native/` 下固定的 CMake 源码和构建目录。相同配置下重复构建或只改 Python 测试时，可直接复用已包含 CATLASS 模板的内核。Wave 和 URMA 分别跟踪各自的内核源码和头文件，修改一方的专用头只触发该内核重编。修改共享内核头或 CATLASS 头会触发两者重编，切换 SHMEM 版本或工具链会选择对应的构建缓存。

分步构建时，在已初始化 Conda 的 Bash 中运行下列命令，然后执行第 2.3 节。其他 UBX 节点可用同一段命令准备安装环境。

```bash
set -eo pipefail
conda activate py311_tpr
export CANN_ENV=/usr/local/Ascend/cann-9.2.T101/set_env.sh
source "$CANN_ENV"
export PYTHON="$CONDA_PREFIX/bin/python"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
sed -n '/^Version=/p; /^timestamp=/p' "$ASCEND_HOME_PATH/compiler/version.info"
```

### 2.3 构建 wheel、安装并运行主机测试

UBX 和非 UBX 都使用 `DEEPEP_NPU_ARCH=Ascend950`。wheel 包含 MTE 和 URMA，MegaMoE 在运行时根据板型选择后端。

以下示例使用仓库锁定的 SHMEM 子模块。执行 `unset SHMEM_ROOT` 后，构建脚本会按需编译或复用子模块的 SDK。若使用已安装的外部 SDK，将该行替换为指向 SDK 安装根目录的 `export SHMEM_ROOT=...`，并核对目录里有 `include/host/shmem_host_def.h`、`lib/libshmem.so` 和 `deepep-sdk.json`。

```bash
git rev-parse HEAD
git status --short --branch
cmake --version
unset SHMEM_ROOT
git submodule update --init --recursive -- third-party/shmem
depdir=$(mktemp -d)
git clone https://gitcode.com/cann/catlass "$depdir/catlass"
git -C "$depdir/catlass" checkout --detach 2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1
export CATLASS_ROOT="$depdir/catlass"
export DEEPEP_BUILD_MEGAMOE=ON DEEPEP_MEGAMOE_TIMER=ON DEEPEP_NPU_ARCH=Ascend950
export CMAKE_BUILD_PARALLEL_LEVEL=4 MAX_JOBS=4
BUILD_LOG=$(mktemp)
bash experiments/megamoe/scripts/build.sh 2>&1 | tee "$BUILD_LOG"
WHEEL=$(sed -n 's/^\[BUILD\] Installed: //p' "$BUILD_LOG" | tail -1)
sha256sum "$WHEEL"
export SHMEM_ROOT="${SHMEM_ROOT:-$PWD/third-party/shmem/install/shmem}"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
"$PYTHON" -I -c 'import torch, torch_npu; from deep_ep_experimental.megamoe import _C; print(torch.__version__, torch_npu.__version__, _C.api_protocol); assert _C.api_protocol == 13'
"$PYTHON" -m pip install -r experiments/megamoe/tests/requirements.txt
"$PYTHON" -m pytest -q experiments/megamoe/tests --tb=short
```

手动调用 `build.sh` 时，若出现 `SHMEM_ROOT must be an existing dependency directory`，可用 `declare -p SHMEM_ROOT` 查看当前终端导出的值。显式设置为空、指向不存在的目录或指向普通文件都会触发该错误。使用子模块时，在已准备好 CANN、Python 环境和 CATLASS 的原终端执行 `unset SHMEM_ROOT`，再从 `BUILD_LOG=$(mktemp)` 那行重试。`third-party/shmem/install/shmem` 是 SDK 安装产物，首次构建成功后再导出该路径。

仓内 SHMEM SDK 按 `compiler/version.info` 中的 CANN 版本和时间戳校验。若在同一检出切换 CANN 后遇到 `Existing SHMEM build identity changed`，可从独立的干净检出构建。需要在当前检出重建时，先停止使用该 SDK 的作业，确保 `SHMEM_ROOT` 已取消设置，再给 `bash experiments/megamoe/scripts/build.sh` 加 `--rebuild-shmem`。

### 2.4 在各节点验证安装

构建使用仓库锁定的 SHMEM 提交 `b41a8fa560b03ddfd9001052cd6002c6a28de809`、HNS_1825 后端和 SDK 身份文件。wheel 写入本次构建输出所示的 `dist/build-*/`。多机时将该 wheel 复制到各节点，逐台核对 SHA-256，准备匹配的 CANN、驱动和 Python 环境，再执行 `python -m pip install --force-reinstall --no-deps "$WHEEL"`（`WHEEL` 指向该节点上的实物）。Git 提交、脚本及测试配置也使用同一版本。

每个节点在实际运行环境中从仓库根进入独立运行目录，再执行后续各节。把 `RUN_BASE` 设在有足够 Golden 存储空间的文件系统上。多机间须打通 torchrun/TCPStore、SHMEM 及双参考所用的 HCCL 通信。

验证已安装的 wheel 时，先加载 DeepEP 的 SHMEM 和 `torch`、`torch_npu`，再导入 MegaMoE 原生扩展。直接导入 `_C` 时，Linux 动态加载器可能找不到 `libc10.so` 等 PyTorch 运行库。

```bash
set -eo pipefail
export REPO=$(git rev-parse --show-toplevel)
export RUN_BASE=${RUN_BASE:-/mnt/share/l00911701/megamoe-validation}
mkdir -p "$RUN_BASE"
cd "$RUN_BASE"
python -I - <<'PY'
from importlib.resources import files
from deep_ep._native import require_native

assert files("deep_ep").joinpath("_build_info.json").is_file()
require_native("MegaMoE validation")
import torch
import torch_npu
from deep_ep_experimental.megamoe import _C

assert _C.api_protocol == 13
print(_C.__file__)
PY
```

## 3. 单机精度、热点和窗口边界

在已安装 wheel 的环境中操作。以下冒烟配置对两个后端均有效。选择两个空闲端口，SHMEM 端口与 torchrun 端口分开。UBX 单机 EP16 时将 `LOCAL_RANKS` 改成 16。

```bash
export LOCAL_RANKS=8
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:29180
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MOD=deep_ep_experimental.megamoe
RUN=$(mktemp -d "$PWD/megamoe-run-XXXXXX")
CFG=gelu_smoke_k25
BS=17,255,256,257
smoke_accuracy() {
  local CFG=$1 BS=$2 ROUTE=$3 NAME=$4
  python -m "$MOD.golden" --config "$CFG" --bs-values "$BS" --ep-size "$LOCAL_RANKS" \
    --ranks "0-$((LOCAL_RANKS-1))" --input fp8 --routing "$ROUTE" --workers 2 --golden-dir "$RUN/golden"
  python -m torch.distributed.run --nnodes=1 --nproc-per-node="$LOCAL_RANKS" --master-port=29181 \
    -m "$MOD.benchmark" --mode accuracy --config "$CFG" --bs-values "$BS" --input fp8 --routing "$ROUTE" \
    --dual-reference --golden-dir "$RUN/golden" --output-dir "$RUN/$NAME-$ROUTE"
}
for ROUTE in balanced skewed; do
  smoke_accuracy "$CFG" "$BS" "$ROUTE" smoke
done
```

`balanced` 轮转目的 rank，`skewed` 将路由集中到前 K 个专家，`random` 可额外覆盖带符号权重。以下命令从同一冒烟配置派生 JSON，复用上面的 Bash 函数验证 K1/8/9/25/32 × B255/256/257。对 EP16 的热点用例，例如 E32、K8，检查排在前八个专家之外的目的 rank 本地接收计数均为零。

```bash
python - "$RUN" <<'PY'
import json, pathlib, sys
from deep_ep_experimental.megamoe.benchmark import load_config
for k in (1, 8, 9, 25, 32):
    config = load_config("gelu_smoke_k25", "255,256,257")
    config["model"]["num_topk"] = k
    (pathlib.Path(sys.argv[1]) / f"smoke-k{k}.json").write_text(json.dumps(config))
PY
for k in 1 8 9 25 32; do
  for route in balanced skewed; do
    smoke_accuracy "$RUN/smoke-k$k.json" 255,256,257 "$route" "boundary-k$k"
  done
done
```

每个 `rank_<r>/bs_<B>/accuracy.json` 都包含全组报告。`--dual-reference` 将融合算子、独立 NPU 级联和 CPU Golden 的结果按仓库既有误差预算比较。检查所有条目的 `passed`、`finite`、`counts_equal`、`cascade_counts_equal`、`checked_tokens` 及 `triangle_metrics`。默认 `--check-tokens 0` 检查全部行；抽查时 Golden 和 benchmark 要使用同一正整数，输出全量有限值与逐专家计数仍须通过。输出目录每次用新名字，Golden 可复用。精度模式另可加 `--timer` 对比打点前后的输出。

## 4. 多机及 EP32 目标配置

各节点在相同的软件环境中执行下述两段，`NODE` 根据节点分别设置为 0 到 `NODES-1`。UBX EP32 取 `NODES=2 LOCAL_RANKS=16`，950 EP32 取 `NODES=4 LOCAL_RANKS=8`。B05+B07 联调取 `NODES=2 LOCAL_RANKS=8`。主节点 IPv4 要能从所有参与节点访问。一次使用一个配置、一种路由、一个 B 和一种抽样数。

```bash
export MASTER_IP=192.0.2.10     # 替换为主节点可达的实际 IPv4
export NODE=0 NODES=2 LOCAL_RANKS=16
export DEEPEP_SHMEM_ENDPOINT="tcp://$MASTER_IP:29180"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MOD=deep_ep_experimental.megamoe
CFG=target_gelu_k25
BS=39280
ROUTE=balanced
CHECK=0
EP=$((NODES*LOCAL_RANKS))
FIRST=$((NODE*LOCAL_RANKS))
LAST=$((FIRST+LOCAL_RANKS-1))
RUN=$(mktemp -d "$PWD/megamoe-ep${EP}-XXXXXX")
python -m "$MOD.golden" --config "$CFG" --bs-values "$BS" --ep-size "$EP" --ranks "$FIRST-$LAST" \
  --input fp8 --routing "$ROUTE" --check-tokens "$CHECK" --workers 2 --chunk-tokens 128 --golden-dir "$RUN/golden"
```

单个 EP32 全量 FP32 Golden 输出合计约 19.2 GiB。可先用 `BS=257` 校验建链，再用 `BS=39280 CHECK=8` 尽早发现尾窗及路由问题，最终用 `BS=39280 CHECK=0` 出全量判定。生成 Golden 时，若主机内存较小，可分别为本地每个 rank 生成（`--ranks "$r"`）并复用同一目录。所有节点的 Golden 均就绪后，在各节点并发启动同一轮 torchrun。用 `target_gelu_k25`、`target_gelu_k32` 分别覆盖 `balanced` 与 `skewed`；每轮按自己的参数重复 Golden 和 benchmark 两段。

```bash
LAUNCH=(python -m torch.distributed.run --nnodes="$NODES" --nproc-per-node="$LOCAL_RANKS"
        --node-rank="$NODE" --master-addr="$MASTER_IP" --master-port=29181)
"${LAUNCH[@]}" -m "$MOD.benchmark" --mode accuracy --config "$CFG" --bs-values "$BS" --input fp8 \
  --routing "$ROUTE" --check-tokens "$CHECK" --dual-reference --golden-dir "$RUN/golden" --output-dir "$RUN/accuracy"
```

### 4.1 EP32 验收清单

上面的手工命令可用 UBX 统一入口代替。在节点 0（默认 `90.90.93.84`）启动，脚本通过免密 SSH 拉起节点 1（默认 `90.90.93.184`）。设置 `--node-ip NODE0 NODE1` 可覆盖地址。

```bash
export RUN_DIR=/shared/megamoe-accept/ep32-k25-$(date +%s)
python3 experiments/megamoe/scripts/run_ubx_gelu.py accuracy all \
  --ep-size 32 --config target_gelu_k25 --output-dir "$RUN_DIR"
```

逐节点运行时分别传 `--node-rank 0/1`，两台使用相同的 `--master-addr` 和 `--output-dir`。精度轮等两台的 `--stage prepare` 都成功后，再在两台执行 `--stage run`。性能轮只需 `--stage run`，用 `perf w8a8` 或 `perf w4a8` 指定精度。完整命令见 [MegaMoE 使用说明](../experiments/megamoe/README.md#ubx-gelu-精度与性能)。

脚本等待两边各自完成 16 个 rank 的 Golden，再以 `--nnodes=2 --nproc-per-node=16` 启动同一轮，结束时核对每台首个 rank 的 manifest。EP32 的 A8W4 与 A8W8 均要求窗口化 GELU 和零共享专家。A8W8 的权重占双倍显存，对称区大小以协商结果为准。默认按配置使用提前加权（GELU 后、量化前），`--weighting late` 可切回 Finalize 加权。`--output-dir` 在两台使用相同绝对路径，各节点输出盘可以独立。

前置条件：

- [ ] 两台 UBX 均 16 张可见 NPU，板型 0x44，CPU 与 NPU 空闲
- [ ] 两台安装同一 wheel，`sha256sum` 一致，`_C.api_protocol == 13`
- [ ] 两台 CANN / 驱动 / Torch / torch_npu 版本一致
- [ ] 节点 0 的 `MASTER_ADDR` 与 SHMEM endpoint 可从节点 1 访问，torchrun 端口和 SHMEM 端口互不冲突
- [ ] 主节点到节点 1 的 ssh 免密可用（`--node-ip` 模式），`REMOTE_USER` / `REMOTE_PORT` 按需设置
- [ ] 跑 A8W8 前确认 wheel 带 URMA W8A8，脚本会在设备检查阶段拦截
- [ ] 记录 Git SHA、wheel SHA-256、各机板型与设备序列号

每轮固定 B39280、`CHECK_TOKENS=0` 全量，按配置 × 路由 × 精度分别执行：

- [ ] K25 + balanced：A8W4 accuracy 与 perf
- [ ] K32 + balanced：A8W4 accuracy 与 perf
- [ ] K25 + balanced：A8W8 accuracy 与 perf
- [ ] K32 + balanced：A8W8 accuracy
- [ ] K25 + skewed：A8W4 与 A8W8 accuracy（热点路由）
- [ ] K32 + skewed：A8W4 与 A8W8 accuracy
- [ ] K25 + balanced：A8W4 `--weighting late` accuracy 与 perf（Finalize 对照组）

逐项判据：

| 检查 | 通过条件 | 位置 |
| --- | --- | --- |
| 后端、规模和精度 | `backend=urma`、`ep_size=32`、`mma_type=fp8xfp4`（W4A8）或 `fp8xfp8`（W8A8） | 两台 `rank_<r>/manifest.json` |
| 精度 | 32 个 rank 的 `passed`、`finite`、`counts_equal` 全为 true | `rank_<r>/bs_39280/accuracy.json` |
| 级联双参考 | 设 `DUAL_REFERENCE=1` 时 `cascade_*` 与 `triangle_metrics` 全部通过 | 同上 |
| 专家计数 | K25 均衡下每卡 25 个专家各接收 39,280 次，合计 982,000；K32 不整除，按 `counts_equal` 判定 | 同上 |
| 窗口 | `windows_per_forward` 与协商 W 下的满窗/尾窗一致（W256 为 153 + 112 行） | `rank_0/bs_39280/resources.json` |
| 显存 | 对称区与工作区不超过实测空闲 HBM | 同上 |
| 性能 | 记录 kernel、同步端到端均值与全组吞吐，只留基线不设门槛 | `rank_0/bs_39280/performance.json` |

产物与证据：

- [ ] 两台各自的 `golden-<精度>-node{0,1}.log`、`accuracy-<精度>-node{0,1}.log`、`perf-<精度>-node{0,1}.log`
- [ ] 协商出的 W、窗口数和 `symmetric_buffer_bytes`
- [ ] 结论写入第 6 节实测记录，并注明失败的配置与首个错误行

## 5. 性能与复核

单机时将 `LAUNCH` 设为下例的单机命令；多机沿用上一节的数组。保持板型、EP、路由、B、输入、窗口和 warmup/iters 一致。以下性能入口独立于 Golden。

```bash
# 第 4 节已设置 LAUNCH。仅从第 3 节直接转到单机性能时才初始化此数组。
if ! declare -p LAUNCH >/dev/null 2>&1; then
  LAUNCH=(python -m torch.distributed.run --nnodes=1 --nproc-per-node="$LOCAL_RANKS" --master-port=29181)
fi
"${LAUNCH[@]}" -m "$MOD.benchmark" --mode perf --config "$CFG" --bs-values "$BS" \
  --input fp8 --routing "$ROUTE" --output-dir "$RUN/perf"
# 单独采集设备阶段，避免将带 timer 的调用混入性能数字
"${LAUNCH[@]}" -m "$MOD.benchmark" --mode perf --config "$CFG" --bs-values "$BS" \
  --input fp8 --routing "$ROUTE" --timer --output-dir "$RUN/timer"
```

先查各 rank 的 `manifest.json`，核对后端、配置和每次 forward 的窗口数。UBX 按各 rank 的空闲 HBM 共同选择 W256/W128/W64。MTE 在整批无法装入 1 GiB 对称区时，按大批输入、两组权重、对称内存、满窗和尾窗 Plan 的预算共同选 W2048/W1024/W512/W256。W256 保留原有分配行为。B39280 在 W2048 下为 19 个满窗加 368 行尾窗，在 W256 下为 153 个满窗加 112 行尾窗。

在 EP32 目标几何下，主机布局计算给出的 UBX W256 单卡对称区为 K25 的 2,629,828,608 字节或 K32 的 3,365,928,960 字节。对应的本地缓冲为 380,355,072 或 486,232,064 字节，含每个 Plan 为 GELU 查找表预留的 32,768 字节，另加平台系统工作区。输入、输出、权重和框架分配器也占用显存，实机以协商出的窗口及采集到的读数为准。

| 产物 | 读数 |
| --- | --- |
| `resources.json` | `symmetric_buffer_bytes` 为 SHMEM 容量，`workspace_bytes` 为已准备的本地工作区。`torch_peak_allocated_bytes` 和 `torch_peak_reserved_bytes` 是 PyTorch 分配器峰值。设备空闲/总显存另列，避免将分配器读数当作整卡峰值。 |
| MTE 的 `profiling.json`、UBX 的 `performance.json` | `rank_max_mean_us` 为各轮最慢 rank 的 kernel 耗时均值。多窗按一次逻辑 forward 加总。`host_e2e.rank_max_mean_us` 为独立计时批次中，从公开接口调用到设备同步完成的最慢 rank 耗时均值。 |
| `kernel_samples.json`、timer 文件 | 前者记录正式性能轮的每个 kernel。独立采集的设备 timer 仅覆盖首窗。 |

集群输入吞吐量按 `EP × B × 10⁶ / host_e2e.rank_max_mean_us` 计算，单位为 token/s。

UBX 的统一入口支持 `perf w4a8 --timer` 和 `perf w8a8 --timer`。EP16 自动采 `0,4,8,12`，EP32 的第二台再采 `16,20,24,28`。诊断调用单独执行，首窗 trace 不计入正式性能统计。使用 `summarize_megamoe_overlap.py` 汇总首窗的通信包络、GMM 包络、重叠时间及未覆盖时间。公式和命令见 [UBX 设备 timer 与通信掩盖率](../experiments/megamoe/README.md#ubx-设备-timer-与通信掩盖率)。

### GELU 与 SwiGLU 的性能门槛

在同一组空闲卡上固定 EP、每卡 B、K、H、激活后的 I、输入精度和路由，使用同一份 wheel。SwiGLU 的 GMM1 输出宽度为 2I，GELU 为 I，GMM2 都是 I→H。SwiGLU 使用原有的 Finalize 加权，GELU 默认使用量化前加权；GELU 也可用 `topk_weights_type=0`（脚本 `--weighting late`）回到 Finalize 加权做对照。每种路由分别测量，均衡与热点各自给结论。

主判据是 **GELU / SwiGLU 的成对端到端时延比 ≤ 1**。每次从公开调用开始，到设备同步完成为止，先取本轮最慢 rank，再汇总配对轮次。预热、权重变换和计划准备在计时之外。用各窗 kernel 时长之和定位差异；输入检查、窗口间的专家计数累加和通信等待也会影响应用时延。

当原 SwiGLU 的完整批次超过 MTE 的 1 GiB 对称区时，应注明对照采用的原生窗口大小、满窗数及尾窗，分别报告直接公开接口和相同窗口的对照结果。UBX 原有 SwiGLU 在 EP32/K25 上没有可运行的原生基线；UBX 须在共同支持的 EP 和 K 上做成对实测，并单独报告 EP32 目标的绝对耗时。

非 UBX 950 的 K25 使用以下命令。每个 B 和路由单独出报告。`--require-no-regression` 会先写入 `paired.json`，再在配对中位数大于 1 时以非零状态退出。可在较小的 B 上追加 `--profile` 获取每窗耗时。

```bash
status=0
for route in balanced skewed; do
  for bs in 257 39280; do
    "${LAUNCH[@]}" "$REPO/experiments/megamoe/scripts/paired_benchmark.py" \
      --config target_gelu_k25 --bs "$bs" --routing "$route" \
      --warmup 2 --pairs 8 --require-no-regression --output-dir "$RUN/paired-$route-$bs" || status=1
  done
done
test "$status" -eq 0
```

非 UBX 的 K32 在 W2048 下每侧需 528 MiB 对称内存。两侧分开启动，避免超出同会话 1 GiB 上限。下例沿用第 4 节的拓扑变量，每台参与节点同时执行。选择 `auto` 验收生产选窗；如需复现旧的等窗实验，另取输出目录，将 `auto` 换为固定窗口。每种路由取 8 对独立作业，逐对交替执行顺序；每侧每次启动预热 2 次、采 3 次，再对最慢 rank 的结果取中位数。

```bash
# 让独立作业脚本里的 torchrun 使用装有本轮 wheel 的虚拟环境。
torchrun() { "$VIRTUAL_ENV/bin/python" -m torch.distributed.run "$@"; }
export -f torchrun
for route in balanced skewed; do
  NODE_RANK="$NODE" bash "$REPO/experiments/megamoe/scripts/run_isolated_mte.sh" \
    "$RUN/k32-auto-$route" "$route" auto --nnodes="$NODES" --node-rank="$NODE" \
    --nproc-per-node="$LOCAL_RANKS" --master-addr="$MASTER_IP" --master-port=29181
done
```

UBX 使用统一入口的 `compare`。先交替跑正式端到端配对作业，再自动采集两侧同一组 rank 的首窗设备 timer，最后出 `comparison.json`。当前原生 SwiGLU 后置加权的共同支持范围为 EP16、W4A8、K≤8、每卡 B≤16,384，等窗对照用 B64。EP32/K25 和 W8A8 的目标配置中，报告记录 GELU 绝对值及 SwiGLU 基线阻塞原因。详细流程见 [UBX 性能对照](../experiments/megamoe/README.md#ubx-gelu-与-swiglu-性能对照)。

```bash
UBX="$REPO/experiments/megamoe/scripts/run_ubx_gelu.py"
python3 "$UBX" compare --precision a8w4 --ep-size 16 --topk 8 --bs 64 \
  --output-dir "$RUN/ubx-ep16-k8-w4a8" --require-no-regression
python3 "$UBX" compare --ep-size 32 --output-dir "$RUN/ubx-ep32-k25"
```

出现错误时，先看第一个失败 rank 的输出及本轮 manifest。Golden 缺失通常需核对模型、EP、B、路由和抽样开关；窗口预算失败会给出 rank 与空闲 HBM。跨机检查物理节点、设备编号、SHMEM 引导地址及数据面链路。设备超时或部分 rank 已发起通信后失败，结束该轮全部 worker，确认卡已释放，再以新输出目录重启。

### 验收报告交付格式

每种拓扑、路由、K 和 B 单列一行。保存完整启动命令、节点和 rank 对应关系、软件版本、Git SHA、wheel SHA-256、每个节点的原始目录。多机使用节点本地目录时，收齐各节点的结果，配对汇总取全局 rank 0 所在节点的文件。

| 拓扑 / 后端 | E / H / I / K / 每卡 B / 路由 | 窗口 / 尾窗 / 峰值内存 | 精度（全量或每卡抽查数、通过 rank 数 / EP） | GELU / SwiGLU 端到端 ms、配对中位比 | 结论 / 原始产物路径 |
| --- | --- | --- | --- | --- | --- |

精度取各 rank 的 `manifest.json`、`resources.json`、`accuracy.json`。性能取 `paired.json` 或 `isolated_pairs.json` 的 `host_e2e.median_pair_ratio`，并保留原始轮次。正式通过要求全组完成、覆盖声明的全部行、计数与误差预算通过，且有同配置基线的各条性能用例分别满足比值 ≤ 1。未获空闲资源、缺少对应拓扑、仅完成抽查、有精度或性能失败时，分别写清发生在哪一行；无同配置基线的行写明 GELU 绝对时延及另测的对照配置。

## 6. 历次实测记录

2026-11-19 在非 UBX Ascend 950DT 上使用 CANN 9.2.0 编译协议 13 的 wheel，包含 MTE 和 UBX 两套设备库。以下第一轮构建的 SHA-256 为 `3fe3108f4a4b14cf33805809cfcf0f7bb7a2cecbe390c2f36ae1193c6d364701`。精度验收使用原有阈值。

| 机型及规模 | 配置 | 实测结果 |
| --- | --- | --- |
| B05、B07 各自 EP8 | E32、H1024、I256、K25，均衡，B17/257 | 两台机器的每个 rank 均通过全量 CPU Golden、NPU 级联及逐专家计数检查。 |
| B07 EP8 | 同上，热点，B17/255/256/257 | 每个 rank 的全量双参考均通过。曾触发舍入差异的 B17/rank6/token10 已复验通过。 |
| B05 EP8 | E32、H1024、I256、K32，均衡，B255/256/257 | 每个 rank 的全量双参考和逐专家计数均通过。 |
| B05 EP8 | E32、H1024、I256、K25，均衡，B17 / B257 | 最慢 rank 的同步端到端均值分别为 0.778 / 1.060 ms，全组约 17.5 万 / 193.9 万输入 token/s。计时使用配置中的 1 次预热和 3 次正式调用。 |
| B05 EP8 | E800、H4096、I1536、K25，均衡，B39280 | 154 个窗口；每卡抽 8 个 token 双参考通过，最大绝对误差 0.0001220703125；全量输出有限，所有专家计数一致。同步端到端 179.27 ms，约 175.3 万输入 token/s。每卡对称区 54,525,952 字节，本地工作区 725,332,992 字节。 |
| B05 EP8 | E800、H4096、I1536、K32，热点，B39280 | 154 个窗口；每卡抽 8 个 token 双参考通过，最大绝对误差 0.000244140625；全量输出有限，所有专家计数一致。同步端到端 1.326 s，约 23.7 万输入 token/s。每卡对称区 69,206,016 字节，本地工作区 727,483,392 字节。 |
| B07 EP8 | E800、H4096、I1536、K25，热点，B39280 | 154 个窗口，每卡抽 8 个 token 的双参考、全量有限值和逐专家计数均通过。同步端到端 1.051 s，全组约 29.9 万输入 token/s。 |

采用按实际到达行数选择本地工作区的构建（SHA-256 `da39837b9ceba1f3386f27b3bef4073fb37a31e9e0e08adb92ffe12194c497b7`）在 B05+B07 组成的非 UBX 双机 EP16 上通过全量双参考测试。两台机器各有 8 张卡。配置为 E32、K25、H1024、I256，均衡路由，两个 B 的 16 个 rank 均通过。B17 的同步端到端均值为 1.484 ms，B257 为 1.919 ms，对应全组约 18.3 万、214.3 万输入 token/s。

同一构建的 EP8 配对结果如下。各轮先取最慢 rank，表中报告 8 对调用的时延比中位数。B257 使用公开整批基线，B39280 的两侧均用 W256。SwiGLU 使用 Finalize 加权。

| 每卡 B | B05 均衡 GELU/SwiGLU | B07 热点 GELU/SwiGLU |
| ---: | ---: | ---: |
| 257 | 1.116 | 1.404 |
| 39280 | 1.145 | 1.256 |

B05 上另测了均衡 B39280 的等窗对照。W1024 和 W2048 的时延比分别为 1.626 和 1.623。W2048 两侧的最慢 rank 端到端均值分别为 GELU 145.87 ms、SwiGLU 89.88 ms。两侧同时放大窗口都能改善绝对时延，验收仍须在相同窗口下计算配对比值。

### 激活查表优化

原始实现的热点在激活与量化。B05、B257 的 rank 0 打点中，GELU 的 AIV1 累积激活与量化时长中位数为 508.28 µs，SwiGLU 为 77.74 µs。最后一次 GMM2 结束后，两者排空 Combine 分别用了 170.66 µs 和 3.23 µs。各发送片段的耗时接近，GELU 的激活占用 AIV1，推迟了已就绪结果的发送。

新实现利用 GMM1 输出已经舍入到 BF16 的条件，在 Plan 准备时用本机同一 CANN Erf 生成全部 65,536 种输入的 FP32 系数，按相同的 128 项页去重。设备只需存放最多 32 KiB 的页索引和系数。GELU 保留原 FP32 乘法顺序和 BF16 舍入，权重在 MXFP8 之前生效。空间不足的布局继续执行完整 Erf。B05 的独立设备探针对 9 种权重逐位核对了所有 BF16 输入，FP32 和 BF16 结果均无差异。

首次集成查表后的 B05 EP8 结果如下。配置均为 E800、K25、H4096、I1536，每一项取 2 对预热和 8 对正式样本。GELU 和 SwiGLU 用时是各轮最慢 rank 的均值，时延比是配对中位数。

| 路由 | 每卡 B | 两侧窗口 | GELU / ms | SwiGLU / ms | 时延比 |
| --- | ---: | --- | ---: | ---: | ---: |
| 均衡 | 257 | 整批 | 1.297 | 1.509 | 0.869 |
| 均衡 | 39280 | W256 | 107.05 | 148.22 | 0.723 |
| 热点 | 257 | 整批 | 3.824 | 5.321 | 0.707 |
| 热点 | 39280 | W256 | 501.36 | 834.71 | 0.601 |

B05 在均衡 B39280 下另做了全量 fused 对 NPU 级联检查。8 卡共 1,287,127,040 个输出元素，超阈值元素为 0。独立 CPU Golden 每卡抽查 8 个 token，全量逐专家计数一致。两种检查覆盖的范围不同。

### 包含两套查表实现的 wheel

构建产物 `ascend_deepep-0.1.0.dev5-cp311-cp311-linux_aarch64.whl` 的 SHA-256 为 `115db6b878d44eb2a52a6971f63a1c0d0c028c7bfda9bbb1ecc78ebd7f6e406a`。对应源码的 Linux 侧 MegaMoE 套件通过 312 项。两台非 UBX 950 在此构建上的 E800、H4096、I1536、K25、EP8 测量如下。每个场景的 8 对样本中，GELU 的端到端用时均低于 SwiGLU。

| 环境、路由 | 每卡 B | 窗口 | GELU / ms | SwiGLU / ms | 配对中位比 |
| --- | ---: | --- | ---: | ---: | ---: |
| B05 均衡 | 257 | 整批 | 1.145 | 1.329 | 0.841 |
| B07 热点 | 257 | 整批 | 3.883 | 5.414 | 0.705 |
| B05 均衡 | 39280 | W256 | 106.46 | 147.93 | 0.721 |
| B07 热点 | 39280 | W256 | 500.11 | 836.03 | 0.598 |
| B05 均衡 | 39280 | 实验，两侧 W2048 | 78.21 | 89.80 | 0.870 |

此 wheel 在 B05 用 K32 热点、B39280 复验了每卡 8 个 token 的双参考、全部专家计数和全量有限值，最大绝对误差为 0.000244140625。B07 在相同模型维度、K25 热点下完成同样检查。B07 还通过测试入口显式调用了 URMA 设备库，保留真实同机 MTE 链路和 peer 检查，K25 热点大批次的双参考也通过。该项仅覆盖在非 UBX 八卡上的同机计算及传输。

UBX 的 EP16 单机、EP32 双机以及非 UBX 的 EP32 四机仍需要对应物理资源进行实机验收。
