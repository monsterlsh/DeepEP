# MegaMoE TopK32 GELU A4W4 转测验收清单

转测目标 · 核对旧路径无回退，验收 TopK32、GELU 加权、A4W4 和 EP32 大用例，并建立融合与小算子级联的性能及显存基线。

共 94 个对照组。一个编号对应一组配置。该组的分支、激活 A/B、融合和级联分别留结果，计时重复轮次与独立采集轮次按同一编号归档。

## 测试验证与复现方法

1. **定位 Case**。先读编号所在小节的固定配置，再用该行补齐拓扑、每卡 BS、TopK、量化精度、激活和加权位置。按该行的测试方法链接确定需运行的分支或对照臂。记录实际模型 JSON、路由和 seed。
2. **固定运行环境**。按 UBX 或 950POD 方法选机器和入口。保留实际源码提交、构建补丁或源码归档、wheel SHA256、CANN / torch_npu 版本，以及逻辑 rank 和物理卡映射。同一 Case 的各臂用相同机器、配置和采集口径，输出目录按 Case、分支或臂、采集阶段分开。
3. **先核对执行计划**。对 `run_megamoe.py` 入口先用 `"plan": true` 生成预览。UBX 的模型在 `config`，950POD 在 `spec`。核对 EP、后端、精度、每卡 BS、K、激活、加权和对照类型。950POD 还要逐节点核对 `node_rank`。直接用 `torchrun` 的方法先核对模型 JSON、节点参数、可用 NPU 数和端口。计划通过后，使用新目录执行正式作业。
4. **独立采集和验收**。先跑 Q 精度，再跑 P 的 timer-off 正式性能和该 Case 要求的级联对照。另跑 C 的 timer 采集和 M 的单臂显存采集。汇总全部节点产物，核对退出码、`manifest.json` 的实际配置和预期 rank 数。Q 逐 rank 检查通过、有限值、专家计数和双参考结果，再按 [采集口径与结果填写](#采集口径与结果填写) 回填各臂结果和原始文件路径。
5. **保留待测项**。缺少入口、计时口径、设备资源或 `groupsize` 参数定义时，记下 Case、未完成指标和具体原因。该 Case 的其余可运行项继续采集。

### 可执行复现示例

下面选 **K01（UBX 单机 EP16）** 或 **K37（950POD 四机 EP32）**。两组均为每卡 BS4096、H7168、I3072、E384、K25、S0、A8W4、GELU 后加权和均衡路由。在对应机器的仓库根目录执行，`WORK` 改为本轮新目录。K01 在 UBX 节点 0 执行。K37 每节点设自己的 `NODE_RANK`（0–3），使用相同的绝对 `WORK` 路径。

```bash
CASE=K01
NODE_RANK=0
WORK=/shared/megamoe/transfer/${CASE}-run01
RUN=experiments/megamoe/scripts/run_megamoe.py

python3 - "$CASE" "$NODE_RANK" "$WORK" <<'PY'
import json
import sys
from pathlib import Path

case, node, directory = sys.argv[1:]
node = int(node)
preset, nodes = {
    "K01": ("ubx_ep16_target.json", 1),
    "K37": ("950pod_target.json", 4),
}[case]
assert 0 <= node < nodes
root = Path(directory).resolve()
root.mkdir(parents=True, exist_ok=True)
presets = Path("experiments/megamoe/scripts/configs")
job = json.loads((presets / preset).read_text(encoding="utf-8"))
job.pop("_comments", None)
job.update(precision="a8w4", routing="balanced", check_tokens=0)
job["config"]["model"].update(
    hidden=7168, intermediate_hidden=3072, num_experts=384, num_topk=25,
    num_shared_experts=0, activation="gelu", topk_weights_type=0, mma_type="fp8xfp4",
)
job["config"]["bs_values"] = [4096]
if case == "K37":
    job.update(ep_size=32, node_rank=node)

phases = ("accuracy", "perf", "prefill", "timer") if case == "K01" else (
    "golden", "accuracy", "perf", "prefill",
)
for phase in phases:
    accuracy = phase in ("golden", "accuracy")
    args = dict(job, mode="accuracy" if accuracy else "perf",
                dual_reference=accuracy, output_dir=str(root / phase))
    if case == "K37" and accuracy:
        args.update(golden_dir=str(root / "golden-cache"),
                    golden_only=phase == "golden", skip_golden=phase == "accuracy")
    if phase == "prefill":
        args["compare_prefill_cascade"] = True
    if case == "K01":
        args["timer"] = phase == "timer"
        args["force_npu"] = phase == "accuracy"
    for plan, suffix in ((True, "-plan"), (False, "")):
        path = root / f"{phase}-node{node}{suffix}.json"
        path.write_text(json.dumps(dict(args, plan=plan), indent=2) + "\n", encoding="utf-8")
PY

for phase in accuracy perf prefill; do
    python3 "$RUN" --run-config "$WORK/$phase-node$NODE_RANK-plan.json" || exit
done
if [ "$CASE" = K01 ]; then
    python3 "$RUN" --run-config "$WORK/timer-node$NODE_RANK-plan.json" || exit
else
    python3 "$RUN" --run-config "$WORK/golden-node$NODE_RANK-plan.json" || exit
fi
```

上述命令只生成配置和运行预览。K37 先在四台节点各执行如下命令。等全部节点预生成 Golden 成功，再同步启动正式精度，避免 CPU Golden 的耗时差影响分布式建链。

```bash
python3 "$RUN" --run-config "$WORK/golden-node$NODE_RANK.json"
```

核对后执行对应的正式 JSON。K01 直接执行下列命令，K37 四台节点按同一阶段顺序同时执行。

```bash
for phase in accuracy perf prefill; do
    python3 "$RUN" --run-config "$WORK/$phase-node$NODE_RANK.json" || exit
done
if [ "$CASE" = K01 ]; then
    python3 "$RUN" --run-config "$WORK/timer-node$NODE_RANK.json" || exit
fi
```

K37 用 [N 方法](#n-直接启动融合) 另采 timer。两组按 [显存复测方法](#ubx-与-950pod-显存复测) 采 M。归档 `accuracy.json`、`profiling.json`、`prefill_comparison.json`、原始 timer 和设备显存日志。其他 Case 按自己的表行和方法链接复用这套执行顺序。

| 验证范围 | Case 编号 | 组数 | 完成判据 |
| --- | --- | --- | --- |
| [仓基准回归](#仓基准回归) | R01–R07 | 7 | 在共同支持的口径下确认时延、带宽和掩盖率无回退。 |
| TopK 与激活 | K01–K54 | 54 | 覆盖 K25 和 K32、三种激活加权组合，以及三种量化精度。 |
| [EP32 大用例](#ep32-大用例) | L01–L06 | 6 | 目标配置通过精度。比较 A/B 的 E2E 与总掩盖率，并测 GELU Prefill 级联。 |
| 豆包用例 | D01–D24 | 24 | 固定平台，对两档 groupsize、四档 BS、三种精度记录性能、显存及精度。 |
| Decode 回归 | C01–C03 | 3 | 验证 SwiGLU 和 GELU 两条 Decode 级联各自的算子路径及性能。 |

### 开测前确认

先跑 R01–R07，再做覆盖和大用例。锁定三条分支的实际 commit、构建源码状态、wheel SHA256、CANN、torch_npu、物理卡映射和链路。A4W4 使用包含 88936f1 或对应改动的 wheel。级联比较入口也要包含在测试 wheel 中。

两侧共用模型几何、输入、路由、随机种子和采集范围。同一环境各方法使用独立输出目录。所有表中数值栏初始为待测。已知暂缺的入口或打点按“采集口径与结果填写”标记。

## UBX 测试方法

[UBX EP16 运行预设](../experiments/megamoe/scripts/configs/ubx_ep16_target.json) · [UBX EP32 运行预设](../experiments/megamoe/scripts/configs/ubx_ep32_target.json) · [掩盖率采集](../experiments/megamoe/scripts/ubx_overlap_quickstart.md)

U16 为单机 EP16 MTE。U32 为双机 EP32 MTE+RoCE。按各 case 所在表的“方法”列选择。

### U16 与 U32 启动准备

U16 在 90.90.93.84 使用 16 卡，复制 ubx_ep16_target.json，保留 backend=mte。U32 在 90.90.93.84 发起，两台各 16 卡，节点列表为 90.90.93.84 和 90.90.93.184。复制 ubx_ep32_target.json，将 backend 保持为 mte_roce。双机复用相同路径、wheel 和软件环境，确认 SSH 及训练、SHMEM 和数据面链路可用。

```bash
bash experiments/megamoe/scripts/build_ubx.sh
RUN=experiments/megamoe/scripts/run_megamoe.py
python3 "$RUN" --run-config "$RUN_JSON"
```

### U 原生融合

从 case 填写运行 JSON 的 precision、output_dir、routing 和内嵌 config。将 H、I、E、K、S 依次填入 config.model 的 hidden、intermediate_hidden、num_experts、num_topk、num_shared_experts。实际 BS 写入 config.bs_values，并设置 activation 和 topk_weights_type。去掉会覆盖这些值的旧字段。plan=true 可先核对预览，正式执行时设为 false。性能设 mode=perf、timer=false。精度另用 mode=accuracy、dual_reference=true，正式精度 check_tokens=0。

### T 通信打点

复制同一 case 的性能 JSON，换输出目录并设 mode=perf、timer=true。每台取 4 个 rank，U16 取 0、4、8、12，U32 加 16、20、24、28。按实际后端读取首窗事件，报告注明本地阶段或通信完成口径。U16 MTE 的 A8W8 先测其他指标，掩盖率栏标待 timer。对 L 组的 UBX A/B，A_DIR 和 B_DIR 指两臂包含 rank 子目录的性能目录。

```bash
python3 experiments/megamoe/scripts/compare_ubx_native_overlap.py \
  --gelu "$A_DIR" --swiglu "$B_DIR" --output "$OVERLAP_JSON"
```

### P Prefill 性能级联

与 U 方法使用同一份 case 配置，另建运行 JSON，设 mode=perf、timer=false、dual_reference=false。GELU 采用 Prefill 对照，小算子路径使用 HCCL AllToAll。对照报告包含融合及级联两臂。

```bash
python3 "$RUN" --run-config "$PREFILL_JSON" --compare-prefill-cascade
```

## 950POD 测试方法

[950POD 运行预设](../experiments/megamoe/scripts/configs/950pod_target.json) · [同窗对照说明](../experiments/megamoe/scripts/paired_benchmark.md)

P32 为四台各 8 卡的 EP32，B05 主地址 177.22.21.1。P8 为 B05 单机 8 卡。

### P32 与 P8 启动准备

默认沿用 B05 的 CANN b112 环境和已安装 MegaMoE wheel 的 Python。P32 四台的 node_rank 依次为 0 至 3，各自填写本机 Python 和 CANN 路径。四台协调使用相同的模型、主地址、端口和目录结构。P8 只在 B05 启动。

```bash
source /home/cann/b112/cann-9.2.0/set_env.sh
PY=/opt/miniconda3/envs/lsx/bin/python
export PATH="$(dirname "$PY"):$PATH"
RUN=experiments/megamoe/scripts/run_megamoe.py
```

### W 配置文件入口

复制 950pod_target.json，按 case 填 model、BS、EP、precision、node_rank 和 output_dir。用 mode=perf 或 mode=accuracy 分开启动，后者设 dual_reference=true。对 GELU Prefill 另开 perf 作业并加 --compare-prefill-cascade。P8 Decode 另开 perf 作业并加 --compare-decode-cascade。

```bash
python3 "$RUN" --run-config "$RUN_JSON"
python3 "$RUN" --run-config "$CASCADE_JSON" --compare-prefill-cascade
python3 "$RUN" --run-config "$DECODE_JSON" --compare-decode-cascade
```

### N 直接启动融合

用于旧分支原配回归、950POD 的 A8W4/A8W8 SwiGLU 原生融合，以及 950POD 单独采 timer。CFG 为只包含 schema_version、model、bs_values、seed、warmup、iters 等的模型 JSON。P32 设 NNODES=4、NODE_RANK=0 至 3、MASTER=177.22.21.1。P8 设 NNODES=1、NODE_RANK=0、MASTER=127.0.0.1。当前分支设置 model.mma_type，A8W4、A8W8、A4W4 分别为 fp8xfp4、fp8xfp8、fp4xfp4。前两种设 INPUT=fp8，由 950POD 自动选择后端。A4W4 设 INPUT=fp4，并给 benchmark 加 --backend mte。R 组复用历史基线时沿用原输入类型、seed、预热与迭代数。

```bash
export DEEPEP_SHMEM_ENDPOINT="tcp://${MASTER}:29180"
MOD=deep_ep_experimental.megamoe
"$PY" -m torch.distributed.run --nnodes="$NNODES" --nproc-per-node=8 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER" --master-port=29181 \
  --max-restarts=0 --module "$MOD.benchmark" --config "$CFG" --input "$INPUT" \
  --routing balanced --mode perf --output-dir "$OUT"
```

## 分窗压测与显存采集

### G 950POD A8W4 配对分窗

当 SwiGLU 整批对称区超限时，两臂按相同窗口跑完该 Case 的全部 token。CFG、BS 取对应行，在四台各 8 卡上同时执行。该方法的 P 记录整轮同步 E2E 和吞吐。设备掩盖率需要另采同配置 timer。

```bash
export DEEPEP_SHMEM_ENDPOINT="tcp://${MASTER}:29180"
"$PY" -m torch.distributed.run --nnodes=4 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER" --master-port=29181 \
  experiments/megamoe/scripts/paired_benchmark.py --config "$CFG" --bs "$BS" \
  --routing balanced --pairs 8 --warmup 2 --output-dir "$OUT"
```

### UBX 与 950POD 显存复测

每台先记录空载，再同步采集每卡数据，区分权重和 workspace 准备后的常驻点、稳定执行时的峰值及退出前快照。使用同一批物理卡和一致采样周期。各臂单独运行。当前级联性能工具的两臂共驻显存仅供共驻诊断，级联单臂峰值待独立入口。

```bash
while true; do date -Iseconds; npu-smi info; sleep 1; done > "$HBM_LOG"
```

结束后停止采样，归档每节点原始日志。短 Decode 用例用重复批次采样，标明定频采样峰值。PyTorch allocated/reserved 与设备显存分列。

### 留档字段

按 Case 编号、分支或 A/B、融合或级联、perf/accuracy/timer/memory 归档。保留运行 JSON、commit 与补丁、wheel SHA256、采集范围、rank 和原始文件。

## 采集口径与结果填写

Case 表的 **P / C / Q / M** 分别对应性能、通信、精度、显存。在一个 Case 下按分支、A/B 或融合/级联逐臂填值并附原始报告。

| 栏位 | 每个 Case 按臂填写 | 验收和记录方法 |
| --- | --- | --- |
| P 性能 | kernel 总时延、设备跨度、同步 E2E（µs），输入吞吐（token/s）。级联另记加速比。 | 关闭 timer 做正式计时。以每轮最慢 rank 汇总，固定正式轮次。加速比按级联耗时 / 融合耗时。 |
| C 通信 | Dispatch、Combine、总掩盖率（%），有效带宽（GB/s）及字节和时间口径。 | 独立打点。仅比较相同后端、事件范围和采样 rank 的结果。分子为通信与计算交集，分母为通信时长。总通信先合并两段区间，再计算占比。级联有可比独立 trace 时另记。 |
| M 显存 | 每卡常驻和设备峰值（GiB），汇总峰值 rank。另记 PyTorch allocated/reserved 峰值、workspace、对称缓冲区。 | 融合、级联、A/B 和各分支分别起独立进程。当前原生 benchmark 可追加 --perf-scope host，关闭 profiler 和 timer。结束剩余量填作结束快照。 |
| Q 精度 | 通过 rank 数、校验 token 数、最大误差、专家计数及最终结论。 | 独立运行 accuracy。当前融合与 CPU Golden、独立 NPU 精度级联按 --dual-reference 误差预算比较。所有 rank 的 passed、finite、counts_equal 通过。保存三方指标。 |

### 950POD 设备数据采集

Q 先按以下命令预生成 Golden。RANKS 是本节点的全局 rank 范围，EP 是总 rank 数。将 N 命令改为 --mode accuracy，追加 --dual-reference --golden-dir "$GOLDEN" --check-tokens 0。旧分支按自身已提供的参数执行。

```bash
"$PY" -m "$MOD.golden" --config "$CFG" --ep-size "$EP" --ranks "$RANKS" \
  --input "$INPUT" --routing balanced --golden-dir "$GOLDEN" --check-tokens 0
```

950POD 的 C 使用 N 方法追加 --timer 和 --timer-ranks。P32 每节点取 4 卡，传入 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30。归档 timer_breakdown.json 和原始打点。以 wheel 能力为准。MTE A8W8 记录待 timer。双窗或首窗指标注明实际 token 范围。

### 未齐备指标

K 组有 18 个 SwiGLU Prefill 用例待补同激活的级联性能入口。D 组需确认 groupsize 256/512 在生产路径中的真实参数。级联单臂显存采集需独立入口，当前双臂对照报告的共驻进程显存标为待采。旧分支缺同口径的计时、带宽或占比时保留原 report，明示缺测。

各表的 P、C、Q、M 按本节字段逐臂填写。P 附 profiling.json、performance.json 或级联对照 JSON，Q 附 accuracy.json，M 附 resources.json 和设备采样日志，C 附汇总和原始 timer。相同口径的值并列，暂缺的数据填待测或对应原因。

**逐臂结果登记模板**

| Case / 分支或臂 | Kernel / 设备跨度 / E2E（µs） | 吞吐（token/s） | 带宽（GB/s） / Dispatch、Combine、总掩盖率 | 精度和计数 | 常驻 / 设备峰值（GiB）及峰值 rank | 原始产物 |
| --- | --- | --- | --- | --- | --- | --- |
| 待填 | 待测 | 待测 | 待测 | 待测 | 待测 | 待填 |

## 仓基准与 Decode 用例

R 组全部使用 P32 和 N 方法。C 组使用 P8 和 W 方法。BS 均为每卡 token 数。

### 仓基准回归

V4 Pro 取 H7168、I3072、E384、K6。V4.1 取 H5120、I2304、E384、K6。全部按分支原配执行 SwiGLU A8W4。S 是每卡共享专家数。R 组精度栏记当前分支结果。

| Case | 模型或 BS | 变量和精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 |
| --- | --- | --- | --- | --- | --- |
| `R01` | V4 Pro · BS96 | S0 · A8W4 | [N](#n-直接启动融合) · main / roce / 当前 | P 待测 · C 待测 | Q 待测 · M 待测 |
| `R02` | V4 Pro · BS96 | S1 · A8W4 | [N](#n-直接启动融合) · main / roce / 当前 | P 待测 · C 待测 | Q 待测 · M 待测 |
| `R03` | V4 Pro · BS4096 | S0 · A8W4 | [N](#n-直接启动融合) · main / roce / 当前 | P 待测 · C 待测 | Q 待测 · M 待测 |
| `R04` | V4 Pro · BS4096 | S1 · A8W4 | [N](#n-直接启动融合) · main / roce / 当前 | P 待测 · C 待测 | Q 待测 · M 待测 |
| `R05` | V4.1 · BS8192 | S1 · A8W4 | [N](#n-直接启动融合) · roce / 当前 | P 待测 · C 待测 | Q 待测 · M 待测 |
| `R06` | V4.1 · BS12288 | S1 · A8W4 | [N](#n-直接启动融合) · roce / 当前 | P 待测 · C 待测 | Q 待测 · M 待测 |
| `R07` | V4.1 · BS16384 | S1 · A8W4 | [N](#n-直接启动融合) · roce / 当前 · 核对分窗 | P 待测 · C 待测 | Q 待测 · M 待测 |

roce 表示 feat/roce-support。两个旧分支的运行包和日志分别归档。R07 先核实旧分支 BS16384 的原生容量和可比分窗。旧分支缺同步 E2E 时，用同口径独立调用计时补齐。

### Decode 级联

固定 BS96、H3584、I1280、E384、K8、S0、后加权。用 P8 的 W 方法加 --compare-decode-cascade。Q 仍按当前分支双参考流程另测。

| Case | 模型或 BS | 变量和精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 |
| --- | --- | --- | --- | --- | --- |
| `C01` | BS96 · P8 | SwiGLU · A8W4 | [W](#w-配置文件入口) · 融合 / Decode 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `C02` | BS96 · P8 | SwiGLU · A8W8 | [W](#w-配置文件入口) · 融合 / Decode 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |
| `C03` | BS96 · P8 | GELU · A8W8 | [W](#w-配置文件入口) · 融合 / Decode 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |

C03 的级联 GMM1 使用 CANN GroupedMatmulActivationQuant 与 GELU tanh。融合 MegaMoE 使用 GELU erf。性能和精度记录注明实际近似及舍入口径。

## TopK 和激活 UBX EP16

统一 BS4096、H7168、I3072、E384、S0。balanced 为性能主跑。精度另查 K32 的热点路由与零接收 rank，记录在同一 Case。

| Case | TopK | 激活和加权 | 精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 |
| --- | --- | --- | --- | --- | --- | --- |
| `K01` | 25 | GELU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K02` | 25 | GELU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |
| `K03` | 25 | GELU 后加权 | A4W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K04` | 25 | GELU 提前加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K05` | 25 | GELU 提前加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |
| `K06` | 25 | GELU 提前加权 | A4W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K07` | 25 | SwiGLU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，级联待入口 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K08` | 25 | SwiGLU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，级联待入口 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |
| `K09` | 25 | SwiGLU 后加权 | A4W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，级联待入口 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K10` | 32 | GELU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K11` | 32 | GELU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |
| `K12` | 32 | GELU 后加权 | A4W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K13` | 32 | GELU 提前加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K14` | 32 | GELU 提前加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |
| `K15` | 32 | GELU 提前加权 | A4W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K16` | 32 | SwiGLU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，级联待入口 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K17` | 32 | SwiGLU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，级联待入口 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |
| `K18` | 32 | SwiGLU 后加权 | A4W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，级联待入口 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |

加权模式 GELU 提前=1、GELU 后=0、SwiGLU 后=0。GELU 的 GMM1 N 为 3072，SwiGLU 为 6144。

## TopK 和激活 UBX EP32

统一 BS4096、H7168、I3072、E384、S0。balanced 为性能主跑。精度另查 K32 的热点路由与零接收 rank，记录在同一 Case。

| Case | TopK | 激活和加权 | 精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 |
| --- | --- | --- | --- | --- | --- | --- |
| `K19` | 25 | GELU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K20` | 25 | GELU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K21` | 25 | GELU 后加权 | A4W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K22` | 25 | GELU 提前加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K23` | 25 | GELU 提前加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K24` | 25 | GELU 提前加权 | A4W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K25` | 25 | SwiGLU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，级联待入口 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K26` | 25 | SwiGLU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，级联待入口 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K27` | 25 | SwiGLU 后加权 | A4W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，级联待入口 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K28` | 32 | GELU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K29` | 32 | GELU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K30` | 32 | GELU 后加权 | A4W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K31` | 32 | GELU 提前加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K32` | 32 | GELU 提前加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K33` | 32 | GELU 提前加权 | A4W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K34` | 32 | SwiGLU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，级联待入口 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K35` | 32 | SwiGLU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，级联待入口 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K36` | 32 | SwiGLU 后加权 | A4W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，级联待入口 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |

加权模式 GELU 提前=1、GELU 后=0、SwiGLU 后=0。GELU 的 GMM1 N 为 3072，SwiGLU 为 6144。

## TopK 和激活 950POD EP32

统一 BS4096、H7168、I3072、E384、S0。balanced 为性能主跑。精度另查 K32 的热点路由与零接收 rank，记录在同一 Case。

| Case | TopK | 激活和加权 | 精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 |
| --- | --- | --- | --- | --- | --- | --- |
| `K37` | 25 | GELU 后加权 | A8W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K38` | 25 | GELU 后加权 | A8W8 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |
| `K39` | 25 | GELU 后加权 | A4W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K40` | 25 | GELU 提前加权 | A8W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K41` | 25 | GELU 提前加权 | A8W8 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |
| `K42` | 25 | GELU 提前加权 | A4W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K43` | 25 | SwiGLU 后加权 | A8W4 | [G](#g-950pod-a8w4-配对分窗) · 融合同窗 · 级联待入口 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K44` | 25 | SwiGLU 后加权 | A8W8 | [N](#n-直接启动融合) · 融合待同窗 · 级联待入口 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |
| `K45` | 25 | SwiGLU 后加权 | A4W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合，级联待入口 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K46` | 32 | GELU 后加权 | A8W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K47` | 32 | GELU 后加权 | A8W8 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |
| `K48` | 32 | GELU 后加权 | A4W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K49` | 32 | GELU 提前加权 | A8W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K50` | 32 | GELU 提前加权 | A8W8 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |
| `K51` | 32 | GELU 提前加权 | A4W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K52` | 32 | SwiGLU 后加权 | A8W4 | [G](#g-950pod-a8w4-配对分窗) · 融合同窗 · 级联待入口 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `K53` | 32 | SwiGLU 后加权 | A8W8 | [N](#n-直接启动融合) · 融合待同窗 · 级联待入口 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |
| `K54` | 32 | SwiGLU 后加权 | A4W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合，级联待入口 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |

加权模式 GELU 提前=1、GELU 后=0、SwiGLU 后=0。GELU 的 GMM1 N 为 3072，SwiGLU 为 6144。

SwiGLU 的 A8W4/A8W8 在 BS4096 的整批对称区超限。A8W4 可用 G 方法先测计时。A8W8 的同窗版本，以及按完整 BS 的精度验证，待对应入口。

## EP32 大用例

所有用例均为 BS39280、H4096、I1536、E800、K25、S0，balanced。UBX 与 950POD 各做三种精度。

A 为 GELU 提前加权，B 为 SwiGLU 后加权。A 的 GMM1 K/N 为 4096/1536，B 为 4096/3072，两者 GMM2 都是 1536/4096。各 case 运行原生融合 A/B，另以 A 的模型运行 GELU Prefill 性能对照。

| Case | 模型或 BS | 变量和精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 |
| --- | --- | --- | --- | --- | --- |
| `L01` | U32 | A8W4 | [U](#u-原生融合) / [T](#t-通信打点) / [P](#p-prefill-性能级联) · 融合 A/B + A 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `L02` | U32 | A8W8 | [U](#u-原生融合) / [T](#t-通信打点) / [P](#p-prefill-性能级联) · 融合 A/B + A 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `L03` | U32 | A4W4 | [U](#u-原生融合) / [T](#t-通信打点) / [P](#p-prefill-性能级联) · 融合 A/B + A 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `L04` | P32 | A8W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) / [G](#g-950pod-a8w4-配对分窗) · 融合 A/B + A 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `L05` | P32 | A8W8 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 A/B + A 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 |
| `L06` | P32 | A4W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 A/B + A 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 |

### 大用例验收

A、B 的正式性能轮均处理每卡全部 39280 token。精度分别生成对应激活的 Golden。Q 填两臂的精度、计数和实际校验 token 数，完整记录各自窗口与显存。

验收比较同一后端、同一窗口和同一采样 rank 的总掩盖率。目标 A ≥ B，同时报告 A/B 的端到端耗时比及 Dispatch、Combine 各自百分点差。端到端按 A/B ≤ 1 检查。950POD A8W4 优先给出结论。

950POD 的 A8W4 使用 G 方法做同窗 A/B。A8W8 的 SwiGLU 整批超限，B 臂待 A8W8 同窗入口。GELU 可先跑。首窗百分比单列，E2E 与吞吐按整轮 token 数。

## 豆包用例 groupsize 256

全部使用 U32，H3584、I1280、E384、K8、S0。转测固定 GELU 提前加权、balanced、同一套物理卡。

使用 U / P / T 方法。先确认 groupsize 的生产配置键和生效路径，保存参数及实际值证据，然后只变更表内 BS、精度和 groupsize。固定两种实现的统计范围。

| Case | 每卡 BS | groupsize | 精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 |
| --- | --- | --- | --- | --- | --- | --- |
| `D01` | 1024 | 256 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D02` | 1024 | 256 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D03` | 1024 | 256 | A4W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D04` | 2048 | 256 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D05` | 2048 | 256 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D06` | 2048 | 256 | A4W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D07` | 4096 | 256 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D08` | 4096 | 256 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D09` | 4096 | 256 | A4W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D10` | 8192 | 256 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D11` | 8192 | 256 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D12` | 8192 | 256 | A4W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |

### 需归档的分组证据

若 groupsize 指外层 token 窗口，记录实际窗长与尾窗，并先确认 512 能在所选 U32 路径生效。现有 stage_mte_windows.py 的 512 为单机 950 MTE 实验。其他含义由需求方给出具体参数。确认前状态填写待配置。

## 豆包用例 groupsize 512

全部使用 U32，H3584、I1280、E384、K8、S0。转测固定 GELU 提前加权、balanced、同一套物理卡。

使用 U / P / T 方法。先确认 groupsize 的生产配置键和生效路径，保存参数及实际值证据，然后只变更表内 BS、精度和 groupsize。固定两种实现的统计范围。

| Case | 每卡 BS | groupsize | 精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 |
| --- | --- | --- | --- | --- | --- | --- |
| `D13` | 1024 | 512 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D14` | 1024 | 512 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D15` | 1024 | 512 | A4W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D16` | 2048 | 512 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D17` | 2048 | 512 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D18` | 2048 | 512 | A4W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D19` | 4096 | 512 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D20` | 4096 | 512 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D21` | 4096 | 512 | A4W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D22` | 8192 | 512 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D23` | 8192 | 512 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |
| `D24` | 8192 | 512 | A4W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 |

### 需归档的分组证据

若 groupsize 指外层 token 窗口，记录实际窗长与尾窗，并先确认 512 能在所选 U32 路径生效。现有 stage_mte_windows.py 的 512 为单机 950 MTE 实验。其他含义由需求方给出具体参数。确认前状态填写待配置。
