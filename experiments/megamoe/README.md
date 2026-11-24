<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# MegaMoE Prefill Beta

公开入口：`deep_ep_experimental.megamoe`。W4A8 共享专家按实际单卡 B 自动选择小 BS / Prefill 策略；路由专家保留 Prefill 调度。
构建和多机测试入口位于本目录的 `scripts/`；机器地址及环境配置由使用者提供。

## MegaMoE 测试

`run_megamoe.py` 从运行 JSON 读取测试参数。默认测试预设为 UBX 双机 EP32 GELU。在 `90.90.93.84` 的仓库根目录启动。需要其他机型或拓扑时，复制对应预设后直接修改 JSON。

```bash
RUN=experiments/megamoe/scripts/run_megamoe.py
CASES=experiments/megamoe/scripts/configs
python3 "$RUN" --run-config "$CASES/ubx_ep32_target.json"
```

通用预设的 `_comments` 包含字段说明和可选值。设置 `"plan": true` 或追加 `--plan` 可预览解析结果。

按验收清单测 EP4、EP8、EP16，可先列出场景，再生成一份运行 JSON。例如在 UBX 的 .84 上测 EP4 的 TopK25 GELU Finalize 加权 A8W4。两平台的每组参数、限制和多机命令见[分平台 EP 验收矩阵](../../docs/megamoe-topk32-gelu-a4w4-test-checklist.md#分平台-ep-验收矩阵)。

```bash
MATRIX=experiments/megamoe/scripts/acceptance_cases.py
python3 "$MATRIX" list --machine ubx --ep-size 4 --case K01
python3 "$MATRIX" generate --machine ubx --ep-size 4 --case K01 --phase perf --output K01-ubx-ep4-perf.json
python3 "$RUN" --run-config K01-ubx-ep4-perf.json
```

### UBX GELU 与 SwiGLU 精度和性能

在仓库根目录执行。UBX 编译使用 `build_ubx.sh`。EP32 两台各 16 卡，在两台安装同一份 wheel 和匹配的 HNS_1825 SHMEM SDK。两台使用相同的源码路径，各自选择本机的 `py311_tpr`，可在 JSON 外层用 `peer_python` 指定节点 1 的解释器绝对路径。配置节点 0 到节点 1 的免密 SSH，并打通 torchrun、SHMEM 引导端口及 NPU RoCE 数据面。

```bash
bash experiments/megamoe/scripts/build_ubx.sh
```

默认 EP32 配置为每卡 39,280 个 token、全局 800 个路由专家、TopK 25、零共享专家。GELU 的 GMM1 K/N 为 4096/1536，激活后乘路由权重再量化。SwiGLU 的 GMM1 K/N 为 4096/3072，gate 和 up 各 1536，在 Finalize 加权。两种预设的 GMM2 K/N 均为 1536/4096。

| 拓扑 | 后端 | GELU 提前加权 | SwiGLU 后置加权 |
| --- | --- | --- | --- |
| 双机 EP32 | `mte_roce` | [ubx_ep32_target.json](scripts/configs/ubx_ep32_target.json) | [ubx_ep32_target_swiglu.json](scripts/configs/ubx_ep32_target_swiglu.json) |
| 单机 EP16 | `mte` | [ubx_ep16_target.json](scripts/configs/ubx_ep16_target.json) | [ubx_ep16_target_swiglu.json](scripts/configs/ubx_ep16_target_swiglu.json) |

按 EP 选择运行 JSON。四份预设都在 `config` 下展开了原 `target_gelu_k25` 的参数。复制一份预设后可直接修改以下字段。

| 字段 | 含义 | 示例 |
| --- | --- | --- |
| `config.model.num_experts` | 全局路由专家数，须能被 EP size 整除 | `768` |
| `config.model.num_topk` | 每个 token 选择的专家数，范围 1–32 | `12` |
| `config.model.hidden` | 输入和输出隐藏维度 | `3584` |
| `config.model.intermediate_hidden` | 激活后的专家中间维度，SwiGLU 的 GMM1 输出宽度为该值的两倍 | `2048` |
| `config.model.num_shared_experts` | 共享专家数 | `0` |
| `config.bs_values` | 每卡 token 数，填一个值 | `[257]` |
| `profile_memory_peak` | 逐卡采样 HBM 已用容量。精度默认开启，性能默认关闭 | `true` / `false` |

独立显存轮可用 `python experiments/megamoe/scripts/run_megamoe.py --run-config experiments/megamoe/scripts/configs/l0/L01.json --only memory`。L01–L06 均使用这个入口。各 BS 的 `resources.json` 保存逐 rank 峰值和全组最小余量。通用 JSON 和直接 torchrun 的用法见[单 case 显存复测](../../docs/megamoe-topk32-gelu-a4w4-test-checklist.md#ubx-与-950pod-显存复测)。

`config` 还包含 `warmup`、`iters`、`seed` 和 `alignment`。外层 `routing` 选择路由数据分布。复用同一份 GELU 预设时，可在复制的 JSON 外层设 `"activation": "swiglu"` 切换到 SwiGLU Finalize 加权。外层 `topk` 和 `bs` 可覆盖内嵌值。EP32 自动拉起时脚本向两台节点传递同一份内嵌模型。

- 四份 UBX JSON 默认 `mode=both`、`precision=all`，先测精度，再测性能。单测时将 `mode` 设为 `accuracy` 或 `perf`，将 `precision` 设为 `a8w4`、`a8w8` 或 `a4w4`。也接受 `w4a8`、`w8a8`、`w4a4`。A4W4 使用 packed FP4 激活与 FP4 权重。
- JSON 中的 `topk` 可设为 1–32，`routing` 可选 `balanced`、`skewed`、`random` 或 `random-balanced`。`check_tokens=0` 全量比对，正数按每卡抽查。`dual_reference=true` 加做 NPU 级联比对。`weighting=early|late` 选择量化前或 Finalize 加权。
- 运行 JSON 可指定 `golden_dir`、`run_base`、`output_dir`、`python`、`cann_env`。相对路径以运行 JSON 所在目录为起点。
- 省略 `output_dir` 时，单项作业按实际平台、EP、后端、精度、任务、激活与加权、TopK、每卡 BS、路由和启动时间命名，例如 `ubx-ep16-mte-a8w4-perf-gelu-early-k25-bs4096-balanced-20261124-143005/`。L0 清单使用较短的 Case 目录名，例如 `L01-ubx-ep32-a8w4-memory-20261124-143005/`。实际路径见启动日志。
- EP16 的 JSON 用 `master_addr` 设置本机地址，预设为 `127.0.0.1`。EP32 的 JSON 在 `node_ip` 填两台机器的 IPv4，预设为 `["90.90.93.84", "90.90.93.188"]`。从第一台启动，脚本会通过 SSH 拉起第二台。默认取第一项作为 torchrun 和 SHMEM 的主地址，也可以在 JSON 显式设置 `master_addr`。
- EP32 手动运行时，两台的 JSON 分别设 `node_rank` 为 `0`、`1`，填相同的 `master_addr`、`output_dir`，并移除 `node_ip`。精度先在两台运行 `stage=prepare`，再在两台运行 `stage=run`。性能直接设 `stage=run`。
- EP16 可复用通过校验的 CPU Golden 和 NPU 精度报告。`force_cpu: true` 重算 Golden，`force_npu: true` 重测 NPU，`force: true` 重跑两段。EP32 的 NPU 精度每轮实测。
- 每个精度的输出在本轮的 `accuracy-<精度>` 和 `perf-<精度>` 下。检查 `rank_<r>/manifest.json` 的后端与规模，以及 `bs_<B>/accuracy.json` 的校验结果。正式性能读 MTE 系列的 `profiling.json` 或 URMA 的 `performance.json`，同步端到端字段为 `host_e2e.rank_max_mean_us`。

EP32 的 `mte_roce` 使用 `megamoe-ep32-defaults` 引入的机内 MTE、跨机 RoCE 路径，Dispatch 与 Combine 分别使用 8 和 16 个 QP，manifest 记录 `device_communication=shmem_mte_roce`。要复现原 Decode 负载，可把 [B32 模型](python/deep_ep_experimental/megamoe/configs/ep32_roce_decode.json) 或 [B96 模型](python/deep_ep_experimental/megamoe/configs/ep32_roce_bs96.json) 的内容复制到 EP32 运行 JSON 的 `config`，并设 `mode=perf`、`precision=a8w4`。[移植来源](https://github.com/DJLRY/DeepEP/commit/0c68d94f6b835970855be5661be21e5c71390c91)

### UBX GELU 与 SwiGLU 性能对照

URMA 的 `compare` 支持 EP16/EP32 的 A8W4、A8W8。A 为 GELU 提前加权，B 为 SwiGLU Finalize 加权，两侧共用输入、路由、GMM2 权重，A 的 GMM1 使用 B 的 gate 权重。复制对应的 UBX JSON，设置 `backend=urma`、`mode=compare`、`precision=a8w8`，填写新的 `output_dir`。先设 `plan: true` 查看组合和顺序，随后改回 `false`，用已启用设备 timer 的 wheel 从节点 0 运行。

默认先交替运行 8 对独立作业，逐轮取最慢 rank 的同步端到端耗时。正式测量结束后，单独为 A、B 采首窗 timer。设置 `require_no_regression: true` 时，验收要求成对端到端时延比的中位数 **A/B ≤ 1**，同时总通信掩盖率 **A ≥ B**。`comparison.json` 分别列出 Dispatch、Combine 和总通信结果。两机结果在独立盘时，收齐目录后设 `report_only: true`，在 `peer_output_dir` 数组中填写对端结果目录再运行。可用 `window_tokens` 固定两侧窗口上限为 64、128 或 256。

### 融合 MegaMoE 与 Decode 小算子性能回归

`perf` 下开启 `--compare-decode-cascade`，同一作业用同一份输入、路由和原始权重跑融合与级联。两种级联都从 Dispatch V2 开始，融合 GMM1、激活和 MX 量化，再运行 GMM2 和 Combine V2。按 `config.model.activation` 选择融合算子：

| 激活 | GMM1 融合算子 | 精度 |
| --- | --- | --- |
| `swiglu` | `torch_npu.npu_grouped_matmul_swiglu_quant_v2`，沿用 `moe_test` Decode 路径 | A8W4、A8W8、A4W4 |
| `gelu` | `cann_ops_transformer.ops.grouped_matmul_activation_quant`，使用 NZ 格式的 GMM1 权重 | A8W8、A4W4 |

两种路径都在 Combine 应用路由权重。运行需要包含本功能的 MegaMoE wheel、支持对应算子的 `torch_npu` / CANN，以及可用的 HCCL EP 组。GELU 还需要提供上述 Torch 接口的 CANN `ops-transformer`。 [CANN 融合算子说明](https://github.com/hicann/ops-transformer/blob/master/gmm/grouped_matmul_activation_quant/docs/torchapi_grouped_matmul_activation_quant.md)

在 UBX 复制 [EP16](scripts/configs/ubx_ep16_target.json) 或 [EP32](scripts/configs/ubx_ep32_target.json) 运行 JSON，设置 `mode=perf`、`precision=a8w4`、`timer=false` 和新的 `output_dir`。将内嵌 `config` 替换为 [Decode 模型 JSON](python/deep_ep_experimental/megamoe/configs/ep32_roce_decode.json) 的内容。该模型为 **H3584、I1280、E384、K8**，可将 `config.bs_values` 设为 `[96]`。

```bash
python3 experiments/megamoe/scripts/run_megamoe.py \
  --run-config /shared/megamoe/ubx-decode-perf.json --compare-decode-cascade
```

- UBX 从节点 0 执行。运行 JSON 的 `backend` 决定融合通信后端，可沿用预设的 EP16 `mte` 和 EP32 `mte_roce`，也可选 `urma`。
- 测 GELU 时，把运行 JSON 的 `precision` 设为 `a8w8` 或 `a4w4`，并在 `config.model` 中设置 `activation=gelu`、`topk_weights_type=0`。A4W4 需要原生 MTE 或 MTE+RoCE，以及支持 MXFP4 的 Decode Dispatch 和 GMM1 算子。`decode_comparison.json` 记录实际 GMM1 算子，以及级联的 GELU tanh 近似和 MegaMoE 的 GELU erf。两侧激活实现和 GMM1 舍入口径不同，这组数据用于性能对照。
- Decode 要求后加权、零共享专家、无 clamp，使用 `moe_test` 的测试范围，每卡 BS ≤ 512、TopK ≤ 16、专家数 ≤ 1024。运行配置也接受 `"compare_decode_cascade": true`。
- 950POD 在 [现有运行 JSON](scripts/configs/950pod_target.json) 中设置 `mode=perf`、`compare_decode_cascade=true`、`precision=a8w4`，把内嵌 `config` 替换为 [Decode 模型 JSON](python/deep_ep_experimental/megamoe/configs/ep32_roce_decode.json) 的内容。多机沿用每台的 `node_rank`、B05 的主地址和统一输出目录，用同一入口传入运行 JSON。
- 统一入口给两条路径提供同一份预量化 MX 输入，输入量化在测量前完成。直接用 `torchrun --module deep_ep_experimental.megamoe.benchmark` 启动时可以选择 `--input bf16`，前提是融合后端支持 BF16 输入。此时级联也会在 Dispatch 前做本地 MX 量化。回归比较时保持两次运行的 `input` 一致。

查看 `perf-w4a8/rank_0/bs_96/decode_comparison.json`（950POD 单精度目录为 `perf/`）。每种计时先取各轮最慢 rank，再对正式轮次取均值：

| 字段 | 口径 |
| --- | --- |
| `kernel_sum` | `kernel_details.csv` 中落在同一轮设备区间内的任务耗时之和。 |
| `device_span` | 同一 rank 上首个关联 kernel 开始至最后一个关联 kernel 结束，含中间空隙。 |
| `host_e2e` | 关闭 profiler 后从调用开始到设备同步完成，包含 host 下发。 |
| `speedup` | 对应口径的级联耗时 / 融合耗时，大于 1 表示融合更快。 |

### 大 case 的 Prefill 级联对照

`target_gelu_k25` 对应 **EP32、每卡 39280 token、800 专家、TopK 25、H4096、I1536**。GMM1 K/N 为 4096/1536，GMM2 K/N 为 1536/4096。`--compare-prefill-cascade` 使用 HCCL AllToAll 分发，并用独立 GMM1、GELU、MX 量化、GMM2 和反向 AllToAll 构成级联。配置的 `topk_weights_type=1` 使两侧均在 GELU 后、MX 量化前乘路由权重；改为 `0` 时均在 Finalize 加权。

UBX 复制 [EP32 大 case 运行 JSON](scripts/configs/ubx_ep32_target.json)，保留内嵌模型和两个节点地址。设置 `mode=perf`、`precision=a8w4`、`timer=false` 和新的 `output_dir`，从 `90.90.93.84` 执行：

```bash
python3 experiments/megamoe/scripts/run_megamoe.py \
  --run-config /shared/megamoe/ubx-ep32-gelu-perf.json --compare-prefill-cascade
```

将 JSON 的 `precision` 改为 `a8w8` 可测 A8W8。支持原生 A4W4 的融合后端可显式设置 `a4w4`，其中 UBX 的 A4W4 大 case 使用 `mte_roce`。950POD 使用 [运行 JSON](scripts/configs/950pod_target.json)，设置 `mode=perf`、`compare_prefill_cascade=true`、`ep_size=32` 和新的 `output_dir`。各节点填写自己的 `node_rank`，沿用相同的 B05 主地址和模型配置。不同的精度或后端使用不同的输出目录。

SwiGLU 用相同的 AllToAll 级联入口，模型设置 `activation=swiglu`、`topk_weights_type=0`。产物为 `rank_0/bs_39280/prefill_comparison.json`，沿用 Decode 的三种计时和加速比口径。级联每个源 rank 按 4096 token 分窗，专家侧按 32768 个展开行计算 FFN。单轮报告覆盖整轮 token，包括各窗路由整理、AllToAll 和输出汇总。

### UBX 设备 timer 与通信掩盖率

在 UBX 上对照 GELU 提前加权与 SwiGLU 后加权，按[UBX 掩盖率 A/B 快速开始](scripts/ubx_overlap_quickstart.md)生成运行 JSON、采集两组打点并读取结果。指南分别说明 EP16 MTE、EP32 MTE+RoCE 的本地阶段指标和 URMA 的通信完成指标。

单机 EP16 MTE 可用 `precision=a4w4`、`timer: true` 采集 W4A4 首窗，需安装支持 `mte_a4w4_timer_enabled` 的 wheel。双机 EP32 MTE+RoCE 也支持 W4A4 打点。

要在 MTE+RoCE 上采集设备 timer，在两台 UBX 安装同一份开启 `DEEPEP_MEGAMOE_TIMER` 的新 wheel。复制 EP32 预设，设置 `backend=mte_roce`、`mode=perf`、`precision=a8w4`、`timer: true`，填写 `output_dir`，从节点 0 启动。正式的 `profiling.json` 和同步端到端耗时先按关闭 timer 测量。随后独立诊断轮为默认采样 rank 记录首窗。

MTE+RoCE trace 位于 `perf-<精度>/rank_<R>/bs_<B>/roce_trace.json`，可在 Perfetto 查看。每个采样 rank 同时保存 `roce_timer.bin` 原始缓冲区和 `roce_timer.json` 事件计数、token 范围和时钟单位。各 rank 的 trace 使用各自的设备时间起点。MTE+RoCE 的 trace 包含分层通信事件，现有 URMA 掩盖率汇总使用 URMA 的完成事件。

分析 URMA 单组时，在运行 JSON 设置 `backend=urma`、`mode=perf`、`precision=a8w8`、`timer: true`，并填写 `output_dir`。

```bash
OUT=/shared/megamoe/ep16-k25-a8w8-timer
python3 "$RUN" --run-config /path/to/timer.json
python3 experiments/megamoe/scripts/summarize_megamoe_overlap.py "$OUT" --output "$OUT/overlap.json"
python3 experiments/megamoe/scripts/analyze_megamoe_profile.py "$OUT" --overlap "$OUT/overlap.json" --output "$OUT/analysis.json"
```

每台采 4 个 rank，EP16 为 0、4、8、12，EP32 再采 16、20、24、28。单个采样 rank 的 URMA timer 缓冲区约 672 MiB，MTE+RoCE 约 437 MiB，后者的原始文件与缓冲区等大。URMA 掩盖率按同一 rank、同一首窗的设备 tick 计算，公式为 **通信区间与 GMM 区间的交集时长 / 通信区间时长**。GMM 从 AIC 提交到 AIV1 观测 FIX 完成，Dispatch、Combine 各使用有完成点的通信包络。总通信先合并两段，跨 rank 再以通信时长加权。分析报告给出缺失点位、首窗与稳态差异，并在数据齐备时标出首个 GMM 前、GMM 间隙及最后一个 GMM 后的未覆盖时间。

### 950POD 非 UBX GELU 精度

在 B05 使用 [950pod_target.json](scripts/configs/950pod_target.json)。预设为 `machine=950pod`、`mode=accuracy`、`precision=a8w4`，默认单机 8 卡，`master_addr` 为 `177.22.21.1`。JSON 中的 `python` 和 `cann_env` 指向 B05 原有环境。若 wheel 安装在其他环境，修改 `python`，需要时同时设置 `base_python`。

```bash
python3 "$RUN" --run-config "$CASES/950pod_target.json"
```

将 `config.bs_values` 改为 `[257]`、`check_tokens` 改为 `8` 可快速抽样验证。将 `config.model.num_topk` 改为 `32` 可运行 K32。EP8、EP16、EP32 都可显式选择 `precision=a4w4`。零共享专家的原生融合用 `precision=all` 运行 `a8w4`、`a8w8`、`a4w4`。带共享专家的 950POD 原生融合用 `all` 运行 A8W4 和 A4W4。GELU 提前加权设置 `activation=gelu`、`topk_weights_type=1`，SwiGLU 后加权设置 `activation=swiglu`、`topk_weights_type=0`。每组使用独立的 `output_dir`。

B05 和 B07 双机各 8 卡时，在两台的 JSON 中都设 `ep_size=16`、填写相同的 `master_addr`、`output_dir` 和模型与测试参数，`node_rank` 分别设为 `0`、`1`。两台先用 `golden_only: true` 准备各自的 Golden，全部完成后设为 `false`，加上 `skip_golden: true` 并同时启动精度测试。四台各 8 卡对应 `ep_size=32` 和节点序号 0–3。报告位于本轮目录的 `accuracy/rank_<r>/bs_<B>/accuracy.json`，`precision=all` 时按精度分目录。

测 EP32 性能时，在四台 950POD 节点分别使用同一份模型和测试设置，将 `mode` 改为 `perf`、`ep_size` 改为 `32`，并填写相同的 B05 `master_addr`、两个端口及新的 `output_dir`。B05 的 `node_rank` 为 0，其他三台依次设为 1、2、3。各节点使用本机的 Python 和 CANN 路径，同时执行上面的统一入口。`perf` 直接启动每机 8 卡的 torchrun，收集 kernel 耗时、同步端到端时延和全组吞吐。每个 rank 的报告位于 `perf/rank_<r>/bs_<B>/profiling.json`，运行身份记录在同一 rank 的 `manifest.json`。`precision=all` 时按实际运行的精度写入 `perf-w4a8`、`perf-w8a8` 或 `perf-w4a4`。

精度验收矩阵、历史实测结果及原始产物判读见 [GELU 验证手册](../../docs/megamoe-gelu-validation.md)。非 UBX 的性能配对、固定窗口和隔离进程实验见[配对测试说明](scripts/paired_benchmark.md)。

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

原生 A4W4 使用 `transform_weights_for_mega_moe(raw_w1, raw_w2, activation=..., mma_type="fp4xfp4")`。两层都把 `uint8` 视为 packed bytes 转成 NZ，每个 C0 有 32 字节，即 64 个 FP4 值。SwiGLU 的 W1 保持 `[gate | up]`，两层均传入完整的返回权重对。A8W4 解量化前处理使用 C0=32 个 FP4 值的权重格式。
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
`mma_type="fp8xfp8"` 使用 FP8 E4M3 路由权重。UBX EP16/EP32 支持零共享专家的分窗 GELU 和 SwiGLU，权重为连续 ND `[L,N,K]`，E8M0 scale 为 `[L,N,K/64,2]`。直接调用 API 时，SwiGLU 的 W1 传入原始 `[gate | up]` 给 `transform_weights_for_mega_moe(..., activation="swiglu", backend="urma")`，并保留它返回的完整 W1 权重对。启用 `DEEPEP_MEGAMOE_TIMER=ON` 的 wheel 支持设备打点。benchmark 从配置文件的 `model.mma_type` 读取矩阵乘精度。

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

要求 Linux、Python ≥3.11、Ascend950、CANN 9.2.0 及匹配的 Torch/torch_npu。从当前检出的仓库根目录执行。

```bash
source /path/to/cann/set_env.sh
git -c submodule.recurse=false submodule update --init -- third-party/catlass
export CATLASS_ROOT="$PWD/third-party/catlass"
bash experiments/megamoe/scripts/build.sh
python -m pip show ascend-deepep
```

在匹配的 Torch/torch_npu 环境中执行。实验构建入口默认开启 MegaMoE 和 timer，
使用 `CATLASS_ROOT` 中的锁定版头文件，自动初始化、校验或构建 SHMEM，再构建安装 wheel。已有匹配的 CATLASS 检出时，直接将 `CATLASS_ROOT` 指向该目录即可。
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
`benchmark-runs/batch-ep<EP>-<精度>-<模式>-<激活>-k<TopK>-bs<BS>-YYYYMMDD-HHMMSS-<8位随机编号>/`。多 BS 记录最小值、最大值和个数（如 `bs96-4096-n2`），完整列表见 `case.json`。各节点用含随机编号的目录名核对共享挂载。

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
