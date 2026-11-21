<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# MegaMoE Prefill Beta

公开入口：`deep_ep_experimental.megamoe`。W4A8 共享专家按实际单卡 B 自动选择小 BS / Prefill 策略；路由专家保留 Prefill 调度。
构建和多机测试入口位于本目录的 `scripts/`；机器地址及环境配置由使用者提供。

## GELU / TopK32 工作清单

`feat/megamoe-topk` 的 SwiGLU 和 GELU 共用一个 wheel。手动构建、Golden 生成、双参考精度检查和性能测试的命令见 [GELU 专项验证手册](../../docs/megamoe-gelu-validation.md)。

非 UBX 950 沿用 MTE 后端，UBX 窗口内的同机流量走 MTE，跨机流量走 RoCE。

阶段 A 使用同一台 UBX 上的 8 个 rank、URMA、预量化 FP8 输入、FP4 路由专家和零共享专家。构造 `SymmBuffer`、准备 W1 和调用 `fp8_fp4_mega_moe` 时均传 `activation="gelu"`。各 rank 的实际 B 相同。

| 状态 | 阶段 | 工作与验收条件 |
| --- | --- | --- |
| [x] | A · 计算 | 复用分支已有的单投影 W1。GMM1 输出按 BF16 舍入，FP32 计算 GELU 并乘当前 slot 的 `topk_weights`，再舍入、MXFP8 量化并进入 GMM2。源 rank 按原 slot 顺序用 FP32 累加，最后写 BF16。GELU 默认提前加权，显式传 `topk_weights_type=0` 时回到 Finalize 加权。 |
| [x] | A · 路由 | 支持 `1 ≤ K ≤ min(32,E)`。K1–8 保留专用归约，K9–32 复用原有六槽流水，按实际 K 遍历。进入首个窗口前，全组核对 B、代号、张量形状和路由。拒绝越界 ID、同 token 重复专家和非有限权重，允许有限的零值及负权重。 |
| [x] | A · 容量 | 对外 B 最大 65,536。所有 rank 协商 W，源 rank 至多展开 W×K 行，目的 rank 为全部源预留 EP×W×K 行。两个公开入口的尾窗都按实际行数写入原输出，每次调用按专家汇总各窗计数。W=256、B=39,280 时为 153 个满窗和 112 行尾窗。 |
| [x] | A · 本地回归 | Host C++ 执行设备侧路由与归约代码，覆盖 K1–32、空 peer、热点和非顺序回传行。CPU Golden 检查 GELU 与提前加权。API 测试覆盖满窗、尾窗、重复调用、跨 rank 拒绝非法输入和计划分配失败后的重试。 |
| [x] | B · 16 卡机内开发 | EP16 GELU 支持 K1–32。各 rank 按相同的原生分配布局估算 HBM，集体选择能放入所有卡的最大 W256/128/64。 |
| [x] | C · 32 卡跨机开发 | UBX 两个连续 16-rank 节点，跨机路由与数据分别使用 RoCE QP，完成数据写入后发布代号；带超时的等待及回收保护跨窗口复用。 |
| [x] | 非 UBX 950 开发 | 支持原 MTE 拓扑；GELU 大批次超过 1 GiB 对称区时分窗，尾窗使用对应行数的 MTE Plan。容量为 B39280 的路由与最终输出仍按原顺序对外提供。 |
| [x] | 非 UBX EP8 性能验收 | E800、H4096、I1536、K25 在 B257 和 B39280 的均衡和热点路由上通过。两侧共享输入、路由、GMM2 权重，与原 SwiGLU Finalize 加权配对。各场景均以 8 对最慢 rank 同步端到端时延的比值中位数 ≤ 1 验收。命令与窗口规则见[配对测试](scripts/paired_benchmark.md)。 |
| [ ] | UBX 实机验收 | 协议 13 wheel；EP8 的 K1/8/9/25/32 × B255/256/257、EP16 热点与空 rank、EP32 全量 B39280 E800 H4096 I1536 K25/K32。记录精度、专家计数、HBM、跨机与性能结果。 |

本地回归可从仓库根目录执行：

```shell
python -m pytest -q experiments/megamoe/tests/test_public_api.py experiments/megamoe/tests/test_urma_ep8_routing.py experiments/megamoe/tests/test_urma_ep8_reduce.py experiments/megamoe/tests/test_reference.py experiments/megamoe/tests/test_golden.py experiments/megamoe/tests/test_performance.py
```

已准备[单机 8 卡冒烟配置](python/deep_ep_experimental/megamoe/configs/stage_a_ep8_gelu_k25.json)，使用 E800、K25、H256、I256 和 B17/257。后续在设备上先生成 CPU Golden，再运行同配置的精度验证：

```shell
python -m deep_ep_experimental.megamoe.golden --config stage_a_ep8_gelu_k25 --ep-size 8 --ranks 0-7 --input fp8
# 单机 SHMEM 引导端口，请选择本机空闲端口。
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:29180
torchrun --standalone --nproc-per-node=8 -m deep_ep_experimental.megamoe.benchmark --config stage_a_ep8_gelu_k25 --mode accuracy --input fp8 --dual-reference --output-dir runs/stage-a-ep8-k25
```

性能报告按一次逻辑 forward 汇总各窗 kernel 耗时。该指标不含输入预检、主机协调及窗口间的计数累加。可选设备 timer 只采首窗。

### UBX GELU 精度与性能

在目标机器安装本次构建的 MegaMoE wheel。统一入口 `run_ubx_gelu.py` 支持单机 16 卡和双机 32 卡，默认 K25、每卡 39,280 token、均衡路由、全量精度比对。`--mode accuracy` 单独测精度，默认两种精度。`--mode perf` 单独测 GELU 性能，默认 W4A8。`--mode compare` 测 A/B 性能和掩盖率。`--precision w8a8|a8w4|all` 独立选择精度。旧的位置参数仍可用，`both` 可显式连续跑精度和 GELU 性能。

```bash
UBX=experiments/megamoe/scripts/run_ubx_gelu.py
python3 "$UBX" --mode accuracy --precision all --ep-size 16
python3 "$UBX" --mode perf --precision w8a8 --ep-size 16

# 在 90.90.93.84 上执行，自动通过 SSH 拉起 90.90.93.184。
python3 "$UBX" accuracy all --ep-size 32
python3 "$UBX" perf w8a8 --ep-size 32 --config target_gelu_k32
```

EP32 自动模式先等待两台各自完成本机 rank 的 Golden，再一起启动 32 个 rank。一侧失败时会取消另一侧的当前阶段。两台需在相同绝对路径检出源码、安装同一 wheel，并配置节点 0 到节点 1 的免密 SSH。`--node-ip NODE0 NODE1`、`--ssh-user`、`--ssh-port` 可覆盖连接参数。自定义 JSON 路径需对两台都可读。未设置 `PYTHON` 时加载 `ubx_env.sh`；显式设置时，两台都应能使用该 Python 和已配置好的 CANN 动态库环境。

手动模式在两台分别设置 `NODE_RANK=0` 和 `NODE_RANK=1`。两台的 `prepare` 均成功后，再在两台启动 `run`。性能命令只需 `run`。

```bash
UBX=experiments/megamoe/scripts/run_ubx_gelu.py
ARGS=(accuracy w4a8 --ep-size 32 --node-rank "$NODE_RANK"
      --master-addr 90.90.93.84 --output-dir /shared/ubx-ep32-k25)
python3 "$UBX" "${ARGS[@]}" --stage prepare
python3 "$UBX" "${ARGS[@]}" --stage run
```

- `--config target_gelu_k32` 选择 K32，`--weighting late` 切换到 Finalize 加权。每轮只运行一个配置。
- `CHECK_TOKENS=8` 每卡抽查 8 个 token，`CHECK_TOKENS=0` 做全量。可设置 `ROUTING=skewed` 或 `DUAL_REFERENCE=1`。
- `GOLDEN_DIR` 和 `RUN_BASE` 默认分别为仓库下的 `benchmark-golden` 和 `benchmark-runs`。默认自动分配本轮目录。显式给 `--output-dir` 时，为重跑选择新目录；双机可用共享盘或两台上的相同绝对路径。
- EP16 可复用校验通过的 Golden 和同环境的 NPU 全组通过报告。`--force-cpu` 重算 Golden，内容一致时仍可复用 NPU 报告；`--force-npu` 重测 NPU；`--force` 重跑两段。EP32 每次实测 NPU，支持 `--force-cpu` 和 `--force`。
- 精度和性能输出分别位于 `accuracy-<精度>/rank_<r>/bs_<B>/accuracy.json` 和 `perf-<精度>/rank_<r>/bs_<B>/performance.json`。手动模式和自动模式均保留各节点日志。

原 EP16 和 EP32 文件名仍可调用统一入口。双机环境和验收判据见 [GELU 验证手册](../../docs/megamoe-gelu-validation.md#41-ep32-验收清单)。

### UBX 设备 timer 与通信掩盖率

在两台 UBX 安装由当前源码构建、启用 `DEEPEP_MEGAMOE_TIMER=ON` 的 wheel 后，可在节点 0 运行以下命令。W4A8 和分窗 URMA W8A8 都支持该采集方式。

```bash
UBX=experiments/megamoe/scripts/run_ubx_gelu.py
RUN=/shared/ubx-ep32-w8a8-timer
python3 "$UBX" perf w8a8 --ep-size 32 --timer --output-dir "$RUN"
python3 experiments/megamoe/scripts/summarize_megamoe_overlap.py "$RUN" --output "$RUN/overlap.json"
```

统一入口在每台节点自动采 4 个均匀分布的 rank。EP16 采 `0,4,8,12`，EP32 的第二台再采 `16,20,24,28`。每个参与采集的 rank 约分配 672 MiB 的 timer 缓冲区。独立输出盘时，把两侧的本轮目录都交给汇总脚本。需要验证打点前后一致性时，用 `accuracy w8a8 --timer`，分别检查全量和首窗误差。

报告按同一 rank、同一首窗的设备 tick 计算。GMM 区间从 AIC 提交打点起点到对应 AIV1 FIX 完成观测点，通信区间使用有完成点的 Dispatch / Combine 包络。各类区间先取并集，掩盖率为 `通信区间与 GMM 区间的交集时长 / 通信区间时长`。总通信先合并 Dispatch 与 Combine，跨 rank 的汇总用各 rank 的重叠时间之和除以通信时间之和。缺少配对或完成点会显示覆盖缺口。这个指标是首窗的事件包络估计，包含已记录边界之间的等待和间隙。

联合分析正式 profile 和首窗掩盖率时，在仓库根目录执行：

```bash
python3 experiments/megamoe/scripts/analyze_megamoe_profile.py "$RUN" \
  --overlap "$RUN/overlap.json" --output "$RUN/analysis.json"
```

脚本读取 `performance.json` 和 `kernel_samples.json`。有完整采集配置时也可从结果树下的原始 profiler CSV 还原正式窗口。报告列出最慢 rank、首窗/稳态/尾窗和窗间空档。若保留了同轮 `urma_trace.json` 与 `communication_bandwidth.json`，还会将首窗未覆盖时间拆成首个 GMM 前、GMM 间隙和最后一个 GMM 后，并按观测结果给出排查方向。两台独立盘可同时传入本轮的两个结果目录。

### UBX GELU 与 SwiGLU 性能对照

`run_ubx_gelu.py compare` 固定 A 为 GELU 提前加权，B 为原生 SwiGLU Finalize 加权。`--precision w8a8` 或 `--precision a8w4` 单独选择精度，`--precision all`（默认）依次安排两组。A8W4 也可写作 W4A8，表示 FP8 激活与 FP4 权重。可比的 A/B 用例使用同一输入、路由和 GMM2 权重，A 的 GMM1 取 B 的 gate 权重。

1. 先跑正式同步端到端计时。可配对时默认跑 8 对独立作业，按 A/B、B/A 交替，逐轮取全组最慢 rank。
2. 正式计时全部结束后，`compare` 自动启动独立诊断作业，为每个可运行的组采集首窗 timer。两侧使用同一套 rank，每台采 4 个。
3. 生成 `comparison.json`。配对端到端时延比中位数须 **A/B ≤ 1**，总通信首窗掩盖率须 **A ≥ B**。报告同时保留 Dispatch、Combine、总通信和底层产物路径。两侧窗口序列、timer rank 和通信端点需可比，缺少有效数据时保留未完成状态。

```bash
UBX=experiments/megamoe/scripts/run_ubx_gelu.py

# 先检查目标配置和支持矩阵，不启动作业。
python3 "$UBX" --mode compare --ep-size 32 --plan

# 在主节点测 EP32/K25/B39280 的 W4A8 A/B。
python3 "$UBX" --mode compare --precision a8w4 --ep-size 32 --output-dir /shared/ubx-ep32-k25-compare

# 单独测 W8A8。
python3 "$UBX" --mode compare --precision w8a8 --ep-size 32 --output-dir /shared/ubx-ep32-k25-w8a8

# 单机测 EP16/K25/B39280 的 W4A8 A/B。
python3 "$UBX" --mode compare --precision a8w4 --ep-size 16 \
  --output-dir /shared/ubx-ep16-k25-w4a8-compare --require-no-regression
```

W4A8 的 A、B 两侧都走分窗 UBX，覆盖 EP16/EP32、K1–32、每卡 B1–65536、零共享专家。默认统一用 W256，HBM 预算不足时可用 `--window-tokens 128` 或 `--window-tokens 64` 重跑，两侧使用同一窗口大小。W8A8 的对照仍采 GELU 绝对数据，SwiGLU 标为 `BLOCKED`。`--require-no-regression` 在退化、证据不全或基线缺失时返回非零。用 `ROUTING=skewed` 或 `--config target_gelu_k32` 另起目录测其他组。

双机用独立输出盘时，把节点 1 本轮目录复制到汇总机器，再通过同一入口汇总：

```bash
python3 "$UBX" compare --report-only --output-dir /shared/ubx-ep32-k25-compare \
  --peer-output-dir /copied/node1/ubx-ep32-k25-compare
```

### 950POD 非 UBX GELU 精度

在 B05 首节点运行默认单机 8 卡精度测试，主节点地址为 `177.22.21.1`，默认测 W4A8、K25、每卡 39,280 token。脚本加载 `/home/cann/b112/cann-9.2.0/set_env.sh`，使用 `/opt/miniconda3/envs/lsx/bin/python`。请先在选用的 Python 环境安装 MegaMoE wheel。若 wheel 安装在已有 venv 中，传 `PYTHON=/绝对路径/bin/python`；`CANN_ENV` 和 `BASE_PYTHON` 可覆盖基础环境。

```bash
POD=experiments/megamoe/scripts/run_950pod_gelu_accuracy.py
python3 "$POD" --plan
CHECK_TOKENS=8 python3 "$POD" --bs 257            # 快速验证
python3 "$POD" --config target_gelu_k32           # K32 全量精度
```

B05 和 B07 双机各 8 卡时，在 B05 设置 `NODE_RANK=0`，在 B07 设置 `NODE_RANK=1`。两台使用相同的主节点地址、模型和测试参数。两台的 `--golden-only` 都完成后，再同时启动 `--skip-golden`。

```bash
POD=experiments/megamoe/scripts/run_950pod_gelu_accuracy.py
ARGS=(--ep-size 16 --node-rank "$NODE_RANK" --output-dir /shared/950pod-ep16-k25)
python3 "$POD" "${ARGS[@]}" --golden-only
python3 "$POD" "${ARGS[@]}" --skip-golden
```

四台各 8 卡对应 `--ep-size 32` 和节点序号 0–3。`--master-addr` 可指定其他主节点地址。报告位于本轮目录的 `accuracy/rank_<r>/bs_<B>/accuracy.json`。

## 1. 输入、输出与公开接口

定义：`B` 为每卡 token 数，`H` 为隐藏维度，`I` 为激活后的宽度，`K` 为 top-k，
`E` 为全局路由专家数，`L=E/EP` 为每卡路由专家数，`S` 为每卡共享专家数。

```python
# 公开调用签名；原位写入 y，返回 None。
def fp8_fp4_mega_moe(
    y, l1_weights, l2_weights, sym_buffer,
    shared_l1_weights=None, shared_l2_weights=None,
    cumulative_local_expert_recv_stats=None,
    recipe=(1, 1, 32), activation="swiglu",
    activation_clamp=None, fast_math=True,
    *, x=None, timer=None,
) -> None: ...
```

| 参数/输入 | 类型、形状与用途 |
| --- | --- |
| `y` | 调用方预分配的 BF16 `[B,H]` 输出，不能与输入重叠 |
| `sym_buffer.x / x_sf` | 默认输入：FP8 E4M3 `[B,H]` + E8M0 `[B,H/32]`，调用前填入 buffer 的前 B 行 |
| `x` | 可选 BF16 `[B,H]`，覆盖 buffer 中的 FP8 输入；内部量化为 FP8，**不是 BF16 GEMM** |
| `sym_buffer.topk_idx` | int32 `[B,K]`，全局专家 ID，同 token 不重复且不能为 -1 |
| `sym_buffer.topk_weights` | FP32 `[B,K]` 路由权重；BF16/FP8 输入均从 buffer 读取路由 |
| `l1_weights / l2_weights` | 两对 `(weight, scale)`，当前测试使用 FP4 E2M1 权重 + E8M0 scale |
| `shared_l1_weights / shared_l2_weights` | 共享专家的两对权重，必须同时传入，数量与 buffer 的 S 一致；默认不使用 |
| `cumulative_local_expert_recv_stats` | 可选 int32 `[L]`，在原值上累加专家接收 token 数 |
| `timer` | 可选 `buffer.allocate_timer()`；仅在单独诊断调用中传入 |

`recipe/fast_math` 使用签名中的默认值，`activation` 支持与 buffer 一致的 `swiglu` 或 `gelu`。
`activation_clamp` 支持非负 FP32 上限；`None` 或正无穷等价于 FLT_MAX。
MTE/URMA 均在 SwiGLU 前将 gate 上截断至 limit，up 截断至 [-limit, limit]，路由和共享专家均适用。
例如 `fp8_fp4_mega_moe(y, w1, w2, buffer, activation_clamp=7.0)`。
CPU Golden 和级联参考同步应用该限制。MTE SwiGLU 使用与级联参考一致的 FP32 sigmoid 后乘 gate 顺序，随后才进行 BF16 舍入/MX 量化。所有张量在 buffer 所在 NPU 上。
FP8 输入仍需打包到通信区，因此流水中的 `AIV_QUANT` 此时表示打包，不是重新量化。

权重在测试/推理前转换一次，转换过程不计入性能测量：

| 原始 ND 权重 | packed FP4（uint8，两个值/字节） | E8M0 scale |
| --- | --- | --- |
| GMM1（SwiGLU） | `[L,2I,H/2]`，N 方向先 gate 后 up | `[L,2I,H/64,2]` |
| GMM1（GELU） | `[L,I,H/2]`，单投影 | `[L,I,H/64,2]` |
| GMM2 | `[L,H,I/2]` | `[L,H,I/64,2]` |

调用 `w1, w2 = transform_weights_for_mega_moe(raw_w1, raw_w2)` 得到 Ascend NZ 权重。FP4 routed GMM1 一次准备两套 weight/scale：实际 B≤256 使用 `[gate128, up128]` 融合布局，B>256 使用 `[gate | up]` 配对布局。转换不进入计时区间，但额外保留一份 GMM1 权重和 scale；GMM2 不复制。
请完整保留返回的 `w1`，不要解包后重组、原地修改或自行搬运其张量；权重更新或设备变化时从原始 ND 重新转换。共享专家将首维 L 换成 S，并传入 `shared=True`，使用 planar 权重布局。
不能直接传入已经做过 CUDA 专用重排的权重。

buffer 的构造参数如下，模型参数含义与测试 JSON 相同：

```python
from deep_ep_experimental.megamoe import (
    StoreGroup, SymmBuffer, fp8_fp4_mega_moe, transform_weights_for_mega_moe,
)

# torchrun 启动后先 torch.npu.set_device(local_rank)。
group = StoreGroup.from_env()  # 主机协调用 TCPStore，算子通信用 SHMEM + MTE
buffer = SymmBuffer(
    group, num_experts=384, num_max_tokens_per_rank=4096, num_topk=6,
    hidden=7168, intermediate_hidden=3072,
    num_shared_experts=0, mma_type="fp8xfp4", activation="swiglu",
    ranks_per_node=8, timeout=120,
)
```

准备输入后依次调用 `buffer.validate_inputs(B)`（BF16 时额外传入 `x=x_bf16`）、`buffer.prepare(B)` 和算子。
执行成功后，所有 rank 按相同顺序调用 `buffer.destroy()`、`group.close()`；
异常处理及完整可运行调用示例见 [example.py](python/deep_ep_experimental/megamoe/example.py)。

MTE 约束：EP=2～128，E 可被 EP 整除且每卡最多 128 个路由专家；S=0～8；B=1～65536；
H=1024～8192 且为64的倍数，I=256～4096 且为128的倍数，1≤K≤min(32,E)。
各 rank 必须同序、同 B 调用；仅推理，不支持反向、空/不等长 B 或图捕获。
`mma_type="fp8xfp8"` 使用 FP8 E4M3 路由权重。UBX 上该精度用于零共享专家的分窗 GELU EP16 单机与 EP32 双机，传入连续 ND 权重 `[L,N,K]` 及 E8M0 scale `[L,N,K/64,2]`。启用 `DEEPEP_MEGAMOE_TIMER=ON` 的 wheel 支持该模式的设备打点。benchmark 从配置文件的 `model.mma_type` 读取矩阵乘精度。

兼容入口 `mega_moe(x, ids, weights, w1, w2, buffer, ...)` 返回 `(y, per_call_counts)`。
接口签名参照 DeepGEMM `78b6900`。本实现使用 int32 路由 ID、E8M0 输入 scale，
输入 buffer 是普通 NPU 存储，需要打包进入通信区。路由权重的位置由激活与加权模式决定，CPU 参考使用相同计算顺序。

### 共享专家策略

当 W4A8 的 S>0 时，由实际 B 选择策略，容量 `num_max_tokens_per_rank` 不参与判断：

| 项目 | B<=256 | B>256 |
| --- | --- | --- |
| 输入 | 双 AIV Quant → AIV 同步 → Route → reset → 共享输入准备 | 双 AIV Quant → reset → 共享输入准备 |
| 本卡全核屏障后 | AIC/AIV0 共享 GMM1，AIV1 count/prefix/Dispatch | AIC/AIV0 共享 GMM1，AIV1 Route/子组及跨卡同步/count/Dispatch |
| 共享 GMM1 | 完整 gate/up 输出 N192 独立任务 | N256 gate/up 配对任务 |
| 共享 GMM2 | N224 | N256 |

两条路径均使用 CATLASS 后端、M256 和 N256 共享 SwiGLU。路由专家在 B≤256 使用逻辑 N128/物理 N256 融合；B>256 使用 gate256/up256 配对任务。
Token Combine 按实际单卡 B 选择：B≤256 使用六槽，B>256 使用两槽，与是否开启共享专家无关。
小 BS 使用逐槽 epoch/count 就绪协议，每个物理 AIV 本地推进一次 epoch，
输出握手再推进一次；大 BS 使用输入/输出跨卡握手。共享 GMM1 后无需全核屏障。
共享 SwiGLU 在 AIV1 上优先于 Combine，共享 GMM2 按本核路由完成及共享激活就绪推进。
S=0 的 W4A8 小 BS 也使用上述量化完成/count 就绪协议；W8A8 使用输入/输出跨卡握手。
tiling ABI 为 224 字节。
CATLASS GMM 的布局和同步机制见 [GMM 说明](csrc/kernels/mega_moe/catlass_gmm/README.md)。

所有 rank 必须按相同顺序、相同 B 调用，切换 B 时在计时前集体调用 `prepare(B)`。
已有 `prepare` 一致性检查用于防止各 rank 选择不同握手策略。共享计算五类 timer 保留，
不添加 count 诊断事件；N192/N224 的 tile 元数据按实际 N 分块记录，JSON 格式不变。

验证边界建议覆盖 B=255/256/257/512、S=0/1/2，以及同一 buffer 下反复切换
`256 → 257 → 256` 和 epoch 回绕。CPU 协议测试不能替代 CANN 编译和 NPU 精度/性能验证。

## 路由权重位置与 ProcessGroup

通信实现称为 **MTE** 和 **URMA**，产品识别仍自动选择实现。
`SymmBuffer` 的 `topk_weights_type` 选择路由权重位置，显式取值优先，缺省时 GELU 为 1、SwiGLU 为 0。提前加权（1）是为 GELU 设计的方案，SwiGLU 的 1 仅保留为早期控制实验。

| 激活 | 值 | 路由专家计算顺序 |
| --- | --- | --- |
| SwiGLU | 0（默认） | GMM1 → clamp → SwiGLU → BF16/MX 量化 → GMM2 → 乘路由权重 → reduce |
| SwiGLU | 1（实验） | GMM1 → clamp → SwiGLU → 乘路由权重 → BF16/MX 量化 → GMM2 → reduce |
| GELU | 0 | GMM1 → BF16 → GELU → BF16/MX 量化 → GMM2 → 乘路由权重 → reduce |
| GELU | 1（默认） | GMM1 → BF16 → GELU → 乘路由权重 → BF16/MX 量化 → GMM2 → reduce |

共享专家不乘路由权重。各模式因量化位置不同不保证数值相同。
所有 rank 必须选择相同模式；它影响通信内存布局，buffer 创建后不要修改。
URMA 的提前模式把每条路由的 FP32 权重附在 Scale PUT 中，保持两次 data PUT，增加元数据传输与 Gather；Finalize 模式不传路由权重，由源 rank 在 reduce 阶段用本地权重累加。
CPU Golden 与 NPU cascade 均按同一模式及 clamp 生成参考结果。benchmark 配置 JSON 的 `model` 支持 `topk_weights_type` 和 `activation_clamp`。

```python
import torch
import torch_npu
import torch.distributed as dist
from deep_ep_experimental.megamoe import SymmBuffer, fp8_fp4_mega_moe

torch.npu.set_device(local_rank)
dist.init_process_group("hccl")
group = dist.group.WORLD
buffer = SymmBuffer(
    group, num_experts=384, num_max_tokens_per_rank=1024,
    num_topk=8, hidden=3584, intermediate_hidden=1280,
    num_shared_experts=0, ranks_per_node=group.size(), topk_weights_type=1,
)
# 填入 buffer.x/x_sf/topk_idx/topk_weights，并提前转换 w1/w2。
buffer.prepare(bs)
fp8_fp4_mega_moe(y, w1, w2, buffer, activation_clamp=7.0)
buffer.destroy()
dist.destroy_process_group()
```

接口语义参考 [ops-transformer MegaMoE](https://gitcode.com/cann/ops-transformer/blob/master/mc2/mega_moe/docs/torchapi_mega_moe.md)。

## 2. 编译安装

W4A8 在 B≤256 时使用较小 wave、rail 路由中转和打包 count。
rail 按每8个 rank 一组，EP≤8 或完整8卡分组启用，其余 EP 回退直发路由。
B>256 保留 Prefill 输入协议，所有路径保留输出跨卡同步。

要求 Linux、Python ≥3.11、Ascend950、CANN 9.2.0 及匹配的 Torch/torch_npu；框架环境由使用者准备。

```bash
git clone https://github.com/lueelu/DeepEP.git
cd DeepEP
source /path/to/cann/set_env.sh
bash experiments/megamoe/scripts/build.sh
python -m pip show ascend-deepep
```

在匹配的 Torch/torch_npu 环境中执行。实验构建入口默认开启 MegaMoE 和 timer，
自动初始化固定版本 SHMEM/CATLASS 子模块、校验或构建 SHMEM，再构建安装 wheel。
`PYTHON=/path/to/python` 可指定解释器；`DEEPEP_MEGAMOE_TIMER=OFF` 关闭设备打点。
已激活 Conda 时，入口会在 Python 启动前优先使用该环境的运行库。
已有 SDK 可用 `SHMEM_ROOT`、`CATLASS_ROOT` 指定；已有子模块须保持干净且版本与[依赖锁文件](../../dependencies.lock.json)一致。
需要重建仓内 SHMEM 时，停止使用该 SDK 的作业后传 `--rebuild-shmem`。
wheel 位于 `dist/build-*/`；运行前按构建输出设置 SHMEM 的 `LD_LIBRARY_PATH`。
基础包的 `pip wheel` 默认关闭 MegaMoE，详见[构建文档](../../docs/build.md)。

## 3. 多机测试

### 配置与启动

批量启动器需要各节点相同绝对路径的共享 checkout、匹配的 Conda/CANN 环境和 SSH 访问。确认参与节点 CPU 和 NPU 均空闲，并按实际节点指定每机 8 或 16 个 rank。
在仓库外创建机器配置，通过 `--hosts-file` 显式传入。以下地址仅用于文档示例，需替换为已分配的节点：

```json
{
  "hosts": ["192.0.2.1", "192.0.2.2", "192.0.2.3", "192.0.2.4"],
  "master": "192.0.2.1"
}
```

节点数为 EP 除以每机 rank 数。例如 EP32 在八卡 950 上用 4 个节点，在十六卡 UBX 上用 2 个节点。`master` 默认为列表首节点。主节点分配第一组连续 rank，其余节点按列表顺序分配。
在一台控制机设置远端环境后启动：

```bash
export CONDA_BASE=/path/to/conda
export CONDA_ENV=your_env
export ASCEND_ENV=/path/to/cann/set_env.sh
export HOSTS_FILE=/path/outside/checkout/hosts.json
# REMOTE_USER 默认为控制机当前用户，可按需覆盖；默认使用 SSH 密钥。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --ep-size 32 --plan
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --ep-size 32
```

入口编译一次、逐节点安装同一 wheel，再运行测试及解析 timer；不会更新 Git 分支。
`CONDA_BASE`、`CONDA_ENV`、`ASCEND_ENV` 必须显式设置，路径对应每台远端机器。
密码登录可设置 `SSH_AUTH=password` 并安装 sshpass，密码通过交互输入或 `SSHPASS` 提供。
实际机器配置、凭据和输出报告不要加入版本控制。

### 常用参数

| 参数 | 默认 / 作用 |
| --- | --- |
| `--mode perf\|accuracy` | 默认 perf 执行性能测试；accuracy 检查全量输出和专家计数 |
| `--input fp8\|bf16` | 默认 fp8；只改变输入类型，权重仍为 FP4 |
| `--config NAME或路径` | 默认 `deepseek_v4_pro`；可传模型 JSON 路径 |
| `--bs-values 96,4096` | 覆盖 JSON 中 BS 列表；每个 BS 单独启动 |
| `--routing balanced\|random\|skewed` | 默认 balanced；random/skewed 用于精度压力测试 |
| `--no-timer` | 关闭打点和解析；默认自动保存、解析主节点 rank0～7 |
| `--no-parse-timer` | 跳过自动流水解析，保留原始 timer 和耗时拆分 |
| `--ep-size 8\|16\|32\|64\|128` | 默认为 64。UBX GELU 窗口模式支持 8、16 和 32。 |
| `--ranks-per-node 8\|16` | 默认为 8。十六卡 UBX 的 EP16/32 传 16。 |
| `--check-tokens N` | 精度模式每 rank 的抽样数，默认 0 表示全量。 |
| `--hosts-file 路径 / --master-ip IP` | 必填机器列表 JSON / 覆盖列表内的主节点 |
| `--master-port / --shmem-port` | 默认29654 / 26733，必须不同且未占用 |
| `--skip-build / --wheel 路径` | 复用已安装身份匹配的唯一 wheel / 指定共享 wheel；不再编译，但仍向各机安装 |
| `--prepare-only / --plan` | 只部署并检查 / 只预览 |
| `--timeout 秒` | 每阶段、每 BS 默认7200秒 |

`balanced` 路由按全局 `[token, top-k]` 顺序轮转目标 rank，再轮转卡内 expert。
精度使用相同路由生成 CPU Golden。修改算子或包内 Python 后，需要重新构建安装，
首次运行时省略 `--skip-build`；修改文档或启动脚本无需重编算子。

以下命令使用8节点机器文件，执行 EP64：

```bash
# 全量精度测试，关闭设备打点。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --mode accuracy --bs-values 96,4096 --no-timer

# BF16 输入，性能 + 自动流水解析。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --input bf16 --bs-values 96

# EP128：将 HOSTS_FILE 改为包含16个节点的配置。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --ep-size 128
```

### 模型和共享专家配置

默认模型：H=7168、I=3072（gate+up 总宽6144）、全局384路由专家、TopK=6、S=0。
EP32/64/128 分别每卡12/6/3个路由专家；共享专家不计入384，`num_shared_experts` 表示**每卡**数量。

已提供两个配置，只有 `num_shared_experts` 不同，其余模型、BS、seed 和轮次一致：

- [`deepseek_v4_pro`](python/deep_ep_experimental/megamoe/configs/deepseek_v4_pro.json)：无共享专家，默认配置。
- [`deepseek_v4_pro_shared1`](python/deep_ep_experimental/megamoe/configs/deepseek_v4_pro_shared1.json)：每卡额外1个共享专家，不减少原有路由专家。

```bash
# 每卡1个共享专家，全量精度检查 BS96/4096。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --config deepseek_v4_pro_shared1 --mode accuracy --no-timer

# 使用相同配置执行性能测试，并自动采集、解析主节点8卡流水。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --config deepseek_v4_pro_shared1
```

需要其他数量可复制 JSON、修改 `num_shared_experts` 后使用 `--config /path/to/case.json`。
脚本自动生成对应共享权重、传给公开接口，CPU Golden 也包含共享专家；
**没有 `--shared-experts` 开关**。warmup/iters/seed 同样在 JSON 中修改。

### 测试结果

精度测试以相同输入、权重和路由生成的 CPU Golden 为参考，生成输出和专家计数的校验报告。
CPU Golden 缓存位于 `benchmark-golden/`，相同配置可复用。
性能测试根据关闭 timer 时 profiler 采集的算子 kernel Duration，生成耗时统计报告。
流水分析根据独立采集的设备 timer 数据，生成流水 JSON 和阶段耗时报告，默认解析主节点 rank0～7。

终端输出各 BS 的 `[PERF]` 耗时或 `[ACCURACY SUMMARY]` 精度摘要，结果保存至：
`benchmark-runs/ep<EP>_<时间>_<编号>/`。

| 目录内文件 | 内容 |
| --- | --- |
| `summary.txt` | 各 BS 的性能/精度汇总 |
| `logs/node_<n>_<stage>*.log` | 各节点的构建、Golden、运行、解析日志 |
| `bs_<B>/rank_0/bs_<B>/accuracy.json` | 全部 rank 的精度报告 |
| `bs_<B>/rank_<r>/bs_<B>/profiling.json` | 性能汇总；同目录保留 `kernel_samples.json` 和 `profiler/` |
| `bs_<B>/bs_<B>/timer/` | 主节点8卡原始 `.bin`、`rank_<r>_npu_time.json`、CSV 和 `timer_breakdown.txt/json` |

原始 timer 数据也可离线解析：

```bash
python -m deep_ep_experimental.megamoe.timer /实际结果目录/bs_96/bs_96/timer --ranks 0-7
```

所有 BS 执行成功后，终端输出 `[DONE] all nodes completed`；执行或报告校验失败时返回非零退出码。

## 4. 测试与验证范围

```bash
python -m pip install -r experiments/megamoe/tests/requirements.txt
python -m pytest experiments/megamoe/tests -q
```

保留公开 API、数值参考、Golden 缓存、打包、主机协调与计时解析测试。
CPU 检查不启动设备 kernel，也不作为设备验证结果。Ascend950 设备验证记录按以下维度区分：

- MegaMoE 开启/关闭、timer 开启/关闭的构建安装验证。
- BF16/FP8 输入、不同路由、共享专家、容量边界和多次调用的精度验证。
- 多流生命周期、BS 切换、初始化/销毁和多机长时间运行验证。
- 各声明 EP 规模的设备测试及同输入、同软件版本、同计时范围的性能对比。
