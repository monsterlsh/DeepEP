# MegaMoE GELU / TopK32 验收

目标用例为每卡 39,280 个 token，全局 E800、H4096、I1536、K25，另测 K32 上界。UBX 单机 16 卡用 `mte`，双机各 16 卡用 `mte_roce`，URMA 用于同配置的 A/B 掩盖率对照。非 UBX 950 每机 8 卡，B05+B07 为 EP16，非 UBX EP32 需要四台。

UBX 的运行 JSON、编译、精度、端到端性能及掩盖率命令统一维护在 [MegaMoE 使用说明](../experiments/megamoe/README.md#ubx-gelu-精度与性能)。非 UBX 精度使用 [950POD 入口](../experiments/megamoe/README.md#950pod-非-ubx-gelu-精度)，非 UBX 性能使用[配对测试说明](../experiments/megamoe/scripts/paired_benchmark.md)。

## 准备软件环境

在目标 Linux 节点准备匹配的 CANN、torch 和 torch_npu，记录 Git SHA、wheel SHA-256、驱动和框架版本。UBX 已配置的构建环境为 Conda `py311_tpr` 和 `/usr/local/Ascend/cann-9.2.T101/set_env.sh`。`build_ubx.sh` 会加载该环境、构建并安装 wheel。跨节点时使用同一个 wheel、SDK 和配置。

核对本机 CANN 安装路径时，运行 `bash experiments/megamoe/scripts/find_cann_env.sh`。默认从 `/` 扫描，也可传入一个或多个安装根目录。扫描会显示 UBX 配置路径是否可读，并把通过 Bash 语法检查的 CANN 候选输出为可复制的 `source` 命令。

构建脚本将 wheel 安装到 `py311_tpr`，打印 wheel 路径和 SHA-256。首次使用时只初始化 `third-party/catlass` 本身，后续复用同一路径上的锁定检出及 CMake 增量构建缓存。CATLASS 提供编译时头文件，也可通过 `CATLASS_ROOT` 指定匹配的已有检出。同一仓库内经 `build_ubx.sh` 启动的构建会串行使用这些产物。

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

试编使用 `dist/ubx-dependencies/latest-native/` 下固定的 CMake 源码和构建目录。三个设备库分别跟踪各自的源文件和专用头文件，共享内核头和 CATLASS 头变化时会重编三个库。切换 SHMEM 版本或工具链时使用对应的缓存。

非 UBX 已有框架环境时，加载本机 CANN 后按 [编译安装](../experiments/megamoe/README.md#2-编译安装) 准备 CATLASS 并构建 wheel。若需要独立安装目录，先用框架环境的 Python 创建 venv。以下为 B05 的基础环境。

```bash
source /home/cann/b112/cann-9.2.0/set_env.sh
BASE_PYTHON=/opt/miniconda3/envs/lsx/bin/python
"$BASE_PYTHON" -m venv --system-site-packages .venv-megamoe
source .venv-megamoe/bin/activate
export PYTHON="$VIRTUAL_ENV/bin/python"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$BASE_PYTHON")")/lib:${LD_LIBRARY_PATH:-}"
```

构建后按构建输出设置 `SHMEM_ROOT` 和 `LD_LIBRARY_PATH`。两侧安装环境均可运行以下检查。

```bash
python -I -c 'import torch, torch_npu; from deep_ep_experimental.megamoe.api import _extension; print(_extension().__file__)'
python -m pip install -r experiments/megamoe/tests/requirements.txt
python -m pytest -q experiments/megamoe/tests
```

## 单机 EP8 窗口边界

UBX 统一启动器面向 EP16 和 EP32。专项验证 EP8 的 K1/8/9/25/32 与 B255/256/257 交界时，使用以下小模型测试。先在仓库根目录激活已安装 wheel 的环境，再选择两个空闲端口。

```bash
set -e
export LOCAL_RANKS=8
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:29180
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MOD=deep_ep_experimental.megamoe
RUN=$(mktemp -d "$PWD/megamoe-run-XXXXXX")
python - "$RUN" <<'PY'
import json, pathlib, sys
from deep_ep_experimental.megamoe.benchmark import load_config
for k in (1, 8, 9, 25, 32):
    config = load_config("gelu_smoke_k25", "17,255,256,257" if k == 25 else "255,256,257")
    config["model"]["num_topk"] = k
    (pathlib.Path(sys.argv[1]) / f"smoke-k{k}.json").write_text(json.dumps(config))
PY
for cfg in "$RUN"/smoke-k*.json; do
  for route in balanced skewed; do
    args=(--config "$cfg" --input fp8 --routing "$route" --golden-dir "$RUN/golden")
    python -m "$MOD.golden" "${args[@]}" --ep-size "$LOCAL_RANKS" --ranks "0-$((LOCAL_RANKS-1))" --workers 2
    python -m torch.distributed.run --nnodes=1 --nproc-per-node="$LOCAL_RANKS" --master-port=29181 \
      -m "$MOD.benchmark" "${args[@]}" --mode accuracy --dual-reference --accuracy-cascade prefill --output-dir "$RUN/$(basename "$cfg" .json)-$route"
  done
done
```

每个 `rank_<r>/bs_<B>/accuracy.json` 都包含全组报告。上例核对 `policy=direct_dual_reference` 和 `reference_kind=prefill`。每个 rank 的融合输出须分别通过 CPU Golden 和相同输入下的 Prefill 小算子直接比对，并满足全量有限值、逐专家计数及累计计数检查。正式验收设 `--check-tokens 0`；抽查时 CPU Golden 与 benchmark 使用同一抽样数，融合对小算子仍比较所有行。矩阵中的 C01–C03 要指定 `--accuracy-cascade decode`。报告判据和小 EP 分支基线见[转测清单](megamoe-topk32-gelu-test-checklist.md#测试验证与复现方法)。

## EP32 验收清单

在节点 0 分别选 GELU 和 SwiGLU 的 EP32 运行 JSON。先用 B257 验证建链和尾窗，再对目标 B39280 跑全量。A8W4、A8W8 分开记录，K25、K32 均测 `balanced` 与 `skewed`。内存排查可先按每卡抽查 8 个 token，正式精度使用 `check_tokens=0`。EP16 单机按同一矩阵另出结果，热点路由还应核实零接收 rank 的计数。

| 检查 | 判据或读数 |
| --- | --- |
| 拓扑与用例 | 收齐所有 rank 的 `manifest.json`。核对 `ep_size`、`backend`、激活、加权位置和 `mma_type`。MTE+RoCE 应报告 `device_communication=shmem_mte_roce`。 |
| 精度与计数 | 双参考报告中所有 rank 的 `fused_vs_cpu.passed`、`fused_vs_cascade.passed`、`finite`、`counts_equal`、`cascade_counts_equal`、`cumulative_counts_equal` 与总 `passed` 全部为真。全量 EP32/K25 均衡用例每卡 25 个专家各接收 39,280 次。 |
| 内存与分窗 | 从 `resources.json` 记录 `symmetric_buffer_bytes`、`workspace_bytes`、分配器峰值及设备剩余显存。从 manifest 核对每次 forward 的窗数。例如 B39280、W256 对应 153 个满窗和 112 行尾窗。 |
| 端到端 | 使用各轮最慢 rank 的同步时延 `host_e2e.rank_max_mean_us`。集群输入吞吐为 `EP × B × 10⁶ / 时延（µs）`。kernel 总时长另列。 |
| A/B 与掩盖率 | URMA 使用同一精度、窗口、路由和采样 rank 的 `comparison.json`。成对端到端时延比中位数 A/B ≤ 1，总通信首窗掩盖率 A ≥ B。分列 Dispatch 和 Combine。MTE 系列的正式性能单列并标明后端。 |

按每种拓扑、激活、精度、K、B 和路由各保留一行结果，附运行 JSON、wheel 摘要、各节点原始目录、通过 rank 数、全量或抽查数量、窗口和性能数据。跨机失败时先查首个错误 rank 的日志、引导地址、网络绑定和实际使用的 wheel。

## 历次实测记录

下列精度结果沿用各次构建当时的参考实现和判据。当前直接双参考 Q 需另跑并保存两项逐 rank 的比对结果。

2026-11-20 B05 构建日志记录的 wheel SHA-256 为 `3a07b4ae9f6e7bb7726203f4b879d1ce0f0ba0a0b86c81ab41e3caae643273ca`。该轮记录止于编译，下列数据属于各自注明的较早构建。

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
