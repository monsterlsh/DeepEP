# UBX 上比较 GELU 和 SwiGLU 的通信掩盖率

清单大用例 L01–L03 各有一份现成 JSON，可直接按 [L01 完整验收](#l01-完整验收)中的单条命令运行。下面第 1–3 节供自定义 EP 或单独分析融合 A/B 时使用。

要看 GELU 调度优化在 UBX 上的效果，先跑下面的**原生路径**。EP16 使用单机 MTE，EP32 使用双机 MTE+RoCE。两组作业会给出同一批采样 rank 的**首窗本地阶段掩盖率**，以及每卡全部 39,280 个 token 的 **timer-off 端到端时延**。文末另有 URMA 的通信完成事件 A/B 验收命令。

| 组别 | 激活与权重 | GMM1 K/N | GMM2 K/N |
| --- | --- | --- | --- |
| A | GELU 后乘路由权重，再量化，`topk_weights_type=1` | 4096/1536 | 1536/4096 |
| B | SwiGLU，在 Finalize 乘路由权重，`topk_weights_type=0` | 4096/3072 | 1536/4096 |

两组默认用 A8W4、E800、TopK25、零共享专家、每卡 BS39280、`balanced` 路由和 seed 1234。EP16 MTE 和 EP32 MTE+RoCE 均可把下面的 `PRECISION` 改成 `a4w4`，两组都使用 packed MXFP4 输入与权重。EP16 需安装含 `mte_a4w4_timer_enabled` 的新 wheel 才能采集 A4W4 打点。

## 1. 准备机器和输出目录

EP32 在 `90.90.93.84` 上操作，通过 SSH 拉起 `90.90.93.188`，每台 16 卡。两台使用相同绝对路径的 checkout，在各自的 Python 环境中安装同一份开启设备 timer 的 **UBX wheel**。EP16 只用第一台的 16 卡。复测这次优化时，wheel 源码需包含提交 `4e68b94`。构建脚本默认开启 `DEEPEP_MEGAMOE_TIMER`。[UBX 编译安装](ubx_ep32_gelu_quickstart.md#1-离线编译并安装)

主节点的 `experiments/megamoe/scripts/` 需包含配套的[原生掩盖率汇总工具](compare_ubx_native_overlap.py)。它从已保存的设备打点读取数据。

以下命令在同一个 Bash 会话运行。测 EP16 时把第一行改成 `EP=16`。EP32 的例子使用两台都能读写的共享目录。

```bash
EP=32
PRECISION=a8w4
case "$PRECISION" in
    a8w4) PERF_DIR=perf-w4a8 ;;
    a4w4) PERF_DIR=perf-w4a4 ;;
esac
cd /mnt/share/l00911701/DeepEP-UBX
SCRIPTS=experiments/megamoe/scripts
source "$SCRIPTS/ubx_env.sh"
mkdir -p "$PWD/benchmark-runs"
OUT=$(mktemp -d "$PWD/benchmark-runs/ubx-ep${EP}-${PRECISION}-overlap-$(date +%Y%m%d-%H%M%S)-XXXXXX")
echo "$OUT"
```

## 2. 生成两份原生运行 JSON，依次跑 A、B

```bash
"$PYTHON" - "$SCRIPTS" "$EP" "$OUT" "$PRECISION" <<'PY'
import json
import sys
from pathlib import Path

scripts, ep, output, precision = sys.argv[1:]
root = Path(output)
for arm, suffix in (("gelu", ""), ("swiglu", "_swiglu")):
    preset = Path(scripts) / "configs" / f"ubx_ep{ep}_target{suffix}.json"
    job = json.loads(preset.read_text(encoding="utf-8"))
    job.pop("_comments", None)
    job.update(mode="perf", precision=precision, timer=True, output_dir=str(root / arm))
    for name, preview in ((arm, False), (f"{arm}-plan", True)):
        (root / f"{name}.json").write_text(json.dumps(dict(job, plan=preview), indent=2) + "\n")
PY
```

只测自定义融合 A/B 时，执行下面的命令，再做第 3 节的汇总。

```bash
"$PYTHON" "$SCRIPTS/run_megamoe.py" --run-config "$OUT/gelu-plan.json"
"$PYTHON" "$SCRIPTS/run_megamoe.py" --run-config "$OUT/swiglu-plan.json"
"$PYTHON" "$SCRIPTS/run_megamoe.py" --run-config "$OUT/gelu.json" &&
    "$PYTHON" "$SCRIPTS/run_megamoe.py" --run-config "$OUT/swiglu.json"
```

预览中核对后端、EP、每卡 BS 和两组的 `activation`、`topk_weights_type`。每个作业先关闭 timer 测完整 forward，再单独跑首窗打点。每台采 4 个 rank，EP16 为 0、4、8、12，EP32 再加 16、20、24、28。两组使用相同输入和路由，原生预设分别生成各自形状的权重。

## 3. 汇总原生掩盖率和端到端性能

```bash
"$PYTHON" "$SCRIPTS/compare_ubx_native_overlap.py" \
    --gelu "$OUT/gelu/$PERF_DIR" --swiglu "$OUT/swiglu/$PERF_DIR" \
    --output "$OUT/native-overlap.json"
```

看表中的 Dispatch、Combine、Total 和 A−B（百分点）。**Total 的 A−B ≥ 0** 表示 A 在该首窗本地阶段的总掩盖率达到 B。在 `native-overlap.json` 中，`arms.gelu.phases.Total.overlap_rate` 和 `arms.swiglu.phases.Total.overlap_rate` 是两组比率，`gelu_minus_swiglu_pp.Total` 是百分点差。同步端到端时延越低越好，原始值在两臂 `$PERF_DIR/rank_0/bs_39280/profiling.json` 的 `host_e2e.rank_max_mean_us`，单位 µs。原生示例每臂执行一个独立作业。

要检查未掩盖的时间，查看每臂 `phases.Total.uncovered_us`。同一对象的 `communication_us` 和 `overlapping_us` 分别是分母和分子，单位均为采样 rank 累加的 µs。用 `recorded_spans.Total` 替换 `phases.Total`，可查看只统计实际打点区间的辅助口径。

- **EP16 MTE** 使用本 rank 所有 AIC 的 `AIC_GMM1`、`AIC_GMM2` 活动区间，以及 AIV 的 `AIV_DISPATCH`、`AIV_COMBINE`。原始文件在各臂的 `$PERF_DIR/bs_39280/timer/rank_<r>_npu_time.bin`。
- **EP32 MTE+RoCE** 中，Dispatch 取本地搬运阶段。异步 Combine 取 `AIV_COMBINE_ASYNC` 的行生命周期（Lifetime），从准备发送到本核观测该行 DMA 完成。RDMA 提交、网关归约和远端收到数据有独立打点。用该百分比检查本地 Combine 与 GMM 的重叠，用完整 forward 的 E2E 检查双机全程收益。各采样 rank 的 `bs_39280/roce_timer.bin` 和 `roce_trace.json` 可进一步定位。
- `case.timer_token_range_half_open` 是两臂实际比较的 token 范围。若 W256，完整 forward 有 153 个满窗和 112-token 尾窗，首窗为 `[0,256)`。要在 UBX 得到全窗掩盖率，需要逐窗保存设备打点，再按下文公式汇总。两次作业固定同一组机器和物理卡。只保存逻辑 rank 的 wheel 会在报告中显示 `physical_mapping=unverified`。

若 EP32 使用独立盘，先在主节点执行以下命令，把第二台的 `rank_16` 至 `rank_31` 收到第一台对应臂的 `$PERF_DIR/`，再运行上面的汇总命令。两台输出路径字符串应相同。

```bash
for arm in gelu swiglu; do
    dir="$OUT/$arm/$PERF_DIR"
    rsync -a --include='rank_1[6-9]/***' --include='rank_2[0-9]/***' \
        --include='rank_3[01]/***' --exclude='*' "90.90.93.188:$dir/" "$dir/"
done
```

## L01 完整验收

在主节点 `90.90.93.84` 的共享 checkout 上运行。两台各 16 卡，使用同一份开启设备 timer 的 UBX wheel，主节点需能通过 SSH 连接 `90.90.93.188`。

```bash
cd /mnt/share/l00911701/DeepEP-UBX/experiments/megamoe/scripts
python run_megamoe.py --run-config configs/l0/L01.json
```

默认先跑 GELU 提前加权、SwiGLU Finalize 加权各自的全量双参考，再测两臂融合性能、汇总首窗掩盖率，最后比较 GELU 融合与 Prefill 级联。每卡 39,280 个 token，EP32、H4096、I1536、E800、K25。只跑性能可在命令末尾加 `--only perf`，只跑精度加 `--only accuracy`，只比较 GELU 级联加 `--only prefill`，预览可加 `--plan`。L02（A8W8）和 L03（A4W4）直接替换文件名。

每次运行在仓库的 `benchmark-runs/` 下新建目录，可用 `RUN_BASE` 指定其他共享盘路径。目录名包含 Case、平台、EP、精度、运行类型和启动节点本地时间。例如 `L01-ubx-ep32-a8w4-perf-20261124-143005/` 是性能轮，`L01-ubx-ep32-a8w4-memory-20261124-151200/` 是显存轮，默认完整运行标为 `full`。同一秒启动相同配置时，从第二次起追加 `-02`、`-03`。下表的 `$OUT` 指终端打印的本次输出目录。`run-summary.json` 记录各阶段状态、输出位置及已采集的整卡 HBM 峰值。精度默认采样，正式性能默认关闭。

若在性能轮显式追加 `--profile-memory-peak`，目录标为 `perf-hbm`。单独复测 GELU、SwiGLU 融合及 GELU Prefill 对照的运行期间显存占比：

```bash
python run_megamoe.py --run-config configs/l0/L01.json --only memory
```

| 检查项 | 结果位置（相对 `$OUT`） | 如何判断 |
| --- | --- | --- |
| A、B 全量精度 | `{gelu,swiglu}-accuracy/accuracy-w4a8/rank_0/bs_39280/accuracy.json` | 每臂 32/32 rank 通过，每 rank 检查 39,280 个 token，输出有限且融合和级联的专家计数都与 Golden 一致。 |
| 融合 A/B 耗时 | `{gelu,swiglu}/perf-w4a8/rank_0/bs_39280/profiling.json` | kernel 取 `rank_max_mean_us`，同步 E2E 取 `host_e2e.rank_max_mean_us`，目标 E2E A/B ≤ 1。 |
| D / C / Total 掩盖率 | `native-overlap.json` | 比较两臂相同首窗和相同采样 rank 的本地阶段。目标 `gelu_minus_swiglu_pp.Total >= 0`，单位为百分点。 |
| A 融合与 Prefill 级联 | `gelu-prefill/perf-w4a8/rank_0/bs_39280/prefill_comparison.json` | 记录两臂耗时及级联 / 融合加速比。 |
| 整卡 HBM 峰值和占比 | `run-summary.json` 对应 job 的 `device_memory_peak_group`；逐卡明细在各臂的 `resources.json` | `max_peak_used_pct` 和 `max_peak_used_pct_rank` 给出占比最高的卡，`min_headroom_bytes` 给出观察到的最小剩余量。Prefill 对照按两臂共驻登记。HBM 读写带宽另按[转测清单](../../../docs/megamoe-topk32-gelu-a4w4-test-checklist.md#hbm-显存带宽采集)采集。 |

## B05 的 84.50% 和 58.48% 怎么算

B05 是 4 台 × 8 卡的 950POD MTE。每臂用独立诊断采集覆盖全部 20 窗（19 × 2048 + 368），每台采 4 个 rank，即 0、2、…、30。两臂合计 640 个 rank-window。每个 rank 只用自己的设备时钟，每个物理窗独立计算。

1. 合并所有 AIC 的 `AIC_GMM1`、`AIC_GMM2` 活动区间，得到计算区间并集 **G**。显式 WAIT 打点排除。
2. Dispatch 取该窗首个 `AIV_DISPATCH` 开始到最后一个结束，记为 **D**。Combine 对 `AIV_COMBINE` 同样取首尾，记为 **C**。总通信取 **D ∪ C**，重叠的通信时间只计一次。
3. 对 **X = D、C、D ∪ C** 分别计算 **掩盖率 = 100% × Σ 时长(X ∩ G) / Σ 时长(X)**。两次求和都覆盖相同 rank 和窗口。这样按通信时长加权，窗间 launch 间隙不进入分母。另报的“记录区间”口径把每阶段首尾包络改为实际打点区间并集，公式相同。

| B05 正式包，A8W4 EP32 | A · GELU 提前加权 | B · SwiGLU 后加权 |
| --- | ---: | ---: |
| Dispatch 包络掩盖率 | 87.97% | 58.39% |
| Combine 包络掩盖率 | 87.02% | 57.49% |
| **总通信包络掩盖率** | **84.50%** | **58.48%** |
| 总通信记录区间掩盖率 | 84.46% | 50.89% |
| 全 forward 同步 E2E | 107.35 ms | 127.27 ms |

以 A 为例，按采样 rank 平均后，20 窗的总通信包络为 87.289 ms，与 GMM 重叠 73.763 ms，故 `73.763 / 87.289 = 84.50%`。E2E 单独关闭 timer，使用配对权重、冷 L2、2 对预热和 8 对正式样本，逐次取 32 个 rank 的最慢值。UBX 首窗结果须在其自身后端内比较。

## 需要 URMA 通信完成口径时

UBX 的 `backend=urma, mode=compare` 会自动运行配对权重的 A/B。它根据 URMA 的提交、接收或发送完成事件计算首窗通信掩盖率。要验证 URMA 自身的通信和 E2E 门槛，复用第 1 节的环境和拓扑即可，下面的命令可独立于原生两臂执行。

```bash
"$PYTHON" - "$SCRIPTS" "$EP" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

scripts, ep, output = sys.argv[1:]
root = Path(output)
preset = Path(scripts) / "configs" / f"ubx_ep{ep}_target.json"
job = json.loads(preset.read_text(encoding="utf-8"))
job.pop("_comments", None)
job.update(backend="urma", mode="compare", precision="a8w4", pairs=8, window_tokens=256,
           require_no_regression=True, output_dir=str(root / "urma"))
for name, preview in (("urma-plan", True), ("urma-run", False)):
    (root / f"{name}.json").write_text(json.dumps(dict(job, plan=preview), indent=2) + "\n")
PY
"$PYTHON" "$SCRIPTS/run_megamoe.py" --run-config "$OUT/urma-plan.json"
"$PYTHON" "$SCRIPTS/run_megamoe.py" --run-config "$OUT/urma-run.json"
```

每种精度先交替跑 8 对 timer-off 作业（AB、BA 交替），再为 A、B 各采一次首窗。A 的 GMM1 取 B 的 gate 权重，两臂共用 GMM2。需要 A8W8 时将生成配置中的 `precision` 改为 `a8w8`，`all` 会依次测两种精度。每次正式运行使用新输出目录。

报告位于 `$OUT/urma/comparison.json`。在 `cases[]` 中按 `precision` 找对应结果。

| 字段 | 如何读 |
| --- | --- |
| `timer.{gelu,swiglu}.overlap.phases.{Dispatch,Combine,Total}.overlap_rate` | 对应阶段的首窗比率，乘 100 转成百分数 |
| `overlap_comparison.gelu_minus_swiglu` | A−B，乘 100 转成百分点。≥ 0 为总掩盖率通过 |
| `latency_comparison.median_pair_ratio` | 每轮取最慢 rank，每作业取中位数，再取 8 对 A/B 时延比的中位数。≤ 1 通过 |
| `status` | `pass` 表示两个门槛均通过。`regression` 查看对应比较项，`incomplete` 查看缺失 rank、trace 或完成打点 |

URMA 的 GMM 从 AIC 提交匹配到同 tile 的 AIV1 FIX 完成观测。通信取已观测完成的负载包络。Total 同样先做区间并集，再与计算求交，跨 rank 按通信时长加权。

EP32 独立盘首次汇总时，主节点缺少第二台的 rank，严格验收会返回非零。把节点 1 的 `$OUT/urma/comparison/` 原样复制到本机另一目录，例如 `$OUT/node1/comparison/`。在原运行 JSON 外层添加 `"report_only": true`、`"peer_output_dir": ["/实际绝对路径/node1"]` 后再运行，使用已保存的 `comparison-plan.json` 重建报告。
