# MegaMoE TopK32 GELU 转测验收清单

转测目标 · 核对旧路径无回退，验收 TopK32、GELU 加权和 EP32 大用例，并建立融合与小算子级联的时延、通信掩盖率、显存占用及 HBM 读写带宽基线。

本分支验收 **A8W4 和 A8W8，共 291 个对照组**。每组由平台、EP 和模型配置标识。保留的用例沿用原编号，K、D 编号有间隔。每组分别归档分支、激活 A/B、融合与级联的结果。交接时先看[分平台 EP 验收矩阵](#分平台-ep-验收矩阵)和[当前验收进度](#当前验收进度)。

## 测试验证与复现方法

使用本分支源码为目标平台重编 wheel，在参与的节点安装同一份构建。CPU Golden 的缓存键包含参考实现源码指纹。分支切换后，分阶段测精度时先用本分支生成 Golden，然后用相同的 EP、模型和路由分别运行融合与级联。L 组可按 [L01 分阶段精度命令](../experiments/megamoe/scripts/ubx_overlap_quickstart.md#l01-单独生成-golden-和验证精度)执行。

**Q 在每个 rank 做两次直接比对。** C01、C02、C03 使用 Decode 小算子（Dispatch/Combine V2），其余 R、K、L、D 用例均使用 Prefill 小算子（AllToAll）。L 的 GELU、SwiGLU 两臂各用自己的 Golden 和 Prefill。矩阵生成的精度配置会选择相应实现；手写 JSON 用 `dual_reference=true` 和 `accuracy_cascade=decode|prefill`，直接启动 benchmark 用 `--dual-reference --accuracy-cascade decode|prefill`。每行均比较同一融合输出与 CPU Golden、同一融合输出与所选小算子，小算子收到相同输入、路由 ID、路由权重和模型权重。

正式 Q 取 `check_tokens=0`。在 `rank_0/bs_<BS>/accuracy.json` 的全组报告中核对 rank 集合、`policy=direct_dual_reference`、`reference_kind` 及逐 rank 的 `checked_tokens=total_tokens=BS`。只有 `fused_vs_cpu.passed` 和 `fused_vs_cascade.passed` 都为真，全量输出有限，融合及小算子的每专家接收计数等于 Golden，融合重复调用的累计计数正确，才记该 rank 通过。两对分别用同一混合误差阈值（超 0.01 的元素占比 ≤5%，最大混合误差 ≤0.1）。`cascade_vs_cpu` 仅作诊断。抽查时 CPU 仅覆盖所选 token，融合对小算子仍比较所有 token；正式 Q 需全 rank 全量通过。分阶段的两个独立 CPU 检查仅用于定位，最终以同一作业的直接比对验收。

1. **定位 Case**。先读编号所在小节的固定配置，再用该行补齐拓扑、每卡 BS、TopK、量化精度、激活和加权位置。按该行的测试方法链接确定需运行的分支或对照臂。记录实际模型 JSON、路由和 seed。
2. **固定运行环境**。按 UBX 或 950POD 方法选机器和入口。保留实际源码提交、构建补丁或源码归档、wheel SHA256、CANN / torch_npu 版本，以及逻辑 rank 和物理卡映射。同一 Case 的各臂用相同机器、配置和采集口径，输出目录按 Case、分支或臂、采集阶段分开。
3. **先核对执行计划**。对 `run_megamoe.py` 入口先用 `"plan": true` 生成预览。UBX 的模型在 `config`，950POD 在 `spec`。核对 EP、后端、精度、每卡 BS、K、激活、加权和对照类型。950POD 还要逐节点核对 `node_rank`。直接用 `torchrun` 的方法先核对模型 JSON、节点参数、可用 NPU 数和端口。计划通过后，使用新目录执行正式作业。
4. **独立采集和验收**。先跑 Q 精度，再跑 P 的 timer-off 正式性能和该 Case 要求的级联对照。另跑 C 的通信 timer、M 的单臂显存占用和 H 的显存读写带宽采集。汇总全部节点产物，核对退出码、`manifest.json` 的实际配置和预期 rank 数。Q 逐 rank 检查通过、有限值、专家计数和双参考结果，再按 [采集口径与结果填写](#采集口径与结果填写) 回填各臂结果和原始文件路径。
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
    if accuracy:
        args["accuracy_cascade"] = "prefill"
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

K37 用 [N 方法](#n-直接启动融合) 另采 timer。两组按[显存占用方法](#ubx-与-950pod-显存复测)采 M，按[HBM 带宽方法](#hbm-显存带宽采集)采 H。归档 `accuracy.json`、`profiling.json`、`prefill_comparison.json`、原始 timer 和设备显存日志。其他 Case 按自己的表行和方法链接复用这套执行顺序。

### 分平台 EP 验收矩阵

UBX 和 950POD 各有 40 个模型配置，每个配置分别登记 EP4、EP8、EP16，共 240 组。另有 51 个 EP32 对照组。下表 K、D 的编号范围只含逐组表中列出的 A8W4、A8W8 用例。相同编号换平台或 EP 后单独归档，比较时固定每卡 BS、全局专家数、路由、精度和加权阶段。

EP4、EP8、EP16 的全部 240 组均须登记当前分支融合与 `feat/roce-support` 同配置融合的性能对照状态，含两平台的 R、C、K、L、D。L 的两个激活臂分别登记。用例受限时逐项记录“不可比”或“待测”及依据。EP32 的 51 组保留本组既定验收项目，本轮新增的分支基线要求只覆盖小 EP。取数方法见[小 EP 分支性能对照](#小-ep-分支性能对照)。

| 场景 | EP4 / EP8 / EP16 每平台每种 EP 的用例 | 每种 EP 两平台合计 | 原 EP32 | 合计 | 验收重点 |
| --- | --- | ---: | ---: | ---: | --- |
| 仓基准 | R01–R07 | 14 | 7 | 49 | 核对同平台同 EP 的分支时延、带宽和掩盖率。新拓扑需重建基线。 |
| Decode | C01–C03 | 6 | 0 | 18 | 每卡 BS96，融合与 Decode 级联。 |
| TopK 与激活 | UBX K01–K17，950POD K37–K53 | 24 | 24 | 96 | K25/K32 × 三种激活加权组合 × 两种精度。UBX EP32 沿用 K19–K35。 |
| 大用例 | UBX L01、L02，950POD L04、L05 | 4 | 4 | 16 | 固定每卡 BS39280、H4096/I1536、E800/K25，比较同 EP 的 A/B 与 GELU Prefill。 |
| 豆包 | D01–D23 | 32 | 16 | 112 | 两档 groupsize、四档 BS、两种精度，分别记录生效证据。 |
| **合计** | | **80** | **51** | **291** | |

| 申请的 EP | UBX | 950POD（B05） | 每台的默认 timer 逻辑卡号 |
| --- | --- | --- | --- |
| 4 | .84 单机 4 卡，MTE | os1 单机 4 卡，MTE | 0、1、2、3 |
| 8 | .84 单机 8 卡，MTE | os1 单机 8 卡，MTE | 0、2、4、6 |
| 16 | .84 单机 16 卡，MTE | os1、os2 各 8 卡，MTE | UBX 为 0、4、8、12；950POD 每机为 0、2、4、6 |
| 32 | .84、.188 各 16 卡，MTE+RoCE | os1、os2、os7、os8 各 8 卡，MTE | 每台沿用各自 EP16 的局部卡号 |

**单组复现。** 在仓库根目录用[矩阵入口](../experiments/megamoe/scripts/acceptance_cases.py)列出所选拓扑，再生成这一组的运行 JSON。下面的 K01 是 UBX 上每卡 BS4096、K25、GELU Finalize 加权、A8W4 的融合性能。预览通过后在 .84 上执行。

```bash
MATRIX=experiments/megamoe/scripts/acceptance_cases.py
RUN=experiments/megamoe/scripts/run_megamoe.py
python3 "$MATRIX" list --machine ubx --ep-size 4
python3 "$MATRIX" generate --machine ubx --ep-size 4 --case K01 --phase perf --output K01-ubx-ep4-perf.json
python3 "$RUN" --run-config K01-ubx-ep4-perf.json --plan
python3 "$RUN" --run-config K01-ubx-ep4-perf.json
```

**950POD 复现。** L 组从 os1 运行一次，按 JSON 中的节点顺序启动。例如在 B05 跑 EP16 的 L04 融合 A/B、设备 timer 与 Prefill 对照，用：

```bash
python3 "$MATRIX" generate --machine 950pod --ep-size 16 --case L04 --phase perf --output L04-pod-ep16-perf.json
python3 "$RUN" --run-config L04-pod-ep16-perf.json --plan
python3 "$RUN" --run-config L04-pod-ep16-perf.json
```

普通 950POD 多机用例在各节点并行启动。例如 K37 EP16 用 os1 和 os2，每个节点将 `NODE_RANK` 分别设为 0 和 1，`OUT` 取两机相同的本轮新共享路径。单机 EP4/EP8 只启动一份。运行小 EP 前按本次源代码构建并在所有参与节点安装同一份对应平台的 wheel，L04、L05 的预设 Python 要指向该环境。

```bash
# os1 用 0，os2 用 1
NODE_RANK=0
OUT=/mnt/share/l00911701_megamoe_b05/runs/K37-ep16-perf-run01
CFG=K37-ep16-perf-node${NODE_RANK}.json
python3 "$MATRIX" generate --machine 950pod --ep-size 16 --case K37 --phase perf \
  --node-rank "$NODE_RANK" --results "$OUT" --output "$CFG"
python3 "$RUN" --run-config "$CFG"
```

`list` 输出每组的实际参数、可运行阶段 `phases` 和限制 `limitation`。用 `--phase accuracy` 跑全量双参考，`perf` 测性能，`memory` 专测峰值，普通用例的 `timer`、`decode`、`prefill` 按该行可用阶段生成。L 组性能阶段自动编排适用的对照与打点。K32 的热点与零接收 rank 用 `accuracy-skewed` 另测。配置文件每个阶段独立命名。

**待测结果。** 每组按照 `平台 / EP / Case / 臂` 填 Q（全 rank 双参考）、P（融合与适用级联的 kernel、同步 E2E、全组吞吐）、C（Dispatch、Combine、总掩盖率及采样 rank）、M（设备峰值 GiB、占比、最小余量）、H（HBM 读/写 GB/s 和采样覆盖）。性能、timer、显存和 HBM 带宽使用独立采集轮。历史结果只计入原平台、原 EP。缩小 EP 会增加每卡专家数及权重显存，比较时记录实选窗口和每卡专家数。

**已知边界。** 240 个小 EP 组合中有 140 组可生成实机阶段计划。L 组在 EP4 时每卡有 200 个专家，超过 native 上限 128，两平台四组保留为阻塞。EP8、EP16 分别是每卡 100、50 个专家。96 组小 EP 豆包用例在落实生产 `groupsize` 配置键和实际生效证据前保留为待接入。普通 MTE 的 A8W8 可以跑时延和精度，设备掩盖率待 timer 支持。950POD 小 EP 上的 SwiGLU 整批超过对称区容量时使用分窗。L04 的同窗性能见 `paired/paired.json`，独立两臂的实际窗口不同时，首窗掩盖率标为不可比。

### 小 EP 分支性能对照

以同一平台的 R01 EP8 为例，比较两个提交各自构建的融合算子。先在目标节点加载相同的 CANN、Torch 和 torch_npu 环境。下例在 950POD 的单机八卡运行，`OUT` 指本轮新的绝对目录。`FRAMEWORK_PY` 指已加载框架的 Python。基线 SHA 以本轮实际解析值为准。

```bash
set -euo pipefail
CURRENT=$(pwd -P)
OUT=$(mktemp -d /tmp/megamoe-R01-ep8-XXXXXX)
FRAMEWORK_PY=$(command -v python3)
git fetch origin feat/roce-support
BASE_SHA=$(git rev-parse FETCH_HEAD)
git worktree add --detach "$OUT/roce-src" "$BASE_SHA"
for arm in current roce; do
    src="$CURRENT"; if [ "$arm" = roce ]; then src="$OUT/roce-src"; fi
    git -C "$src" rev-parse HEAD > "$OUT/$arm.commit"
    git -C "$src" status --short > "$OUT/$arm.status"
    "$FRAMEWORK_PY" -m venv --system-site-packages "$OUT/$arm-env"
    py="$OUT/$arm-env/bin/python"
    (cd "$src" && PYTHON="$py" bash experiments/megamoe/scripts/build.sh) 2>&1 | tee "$OUT/$arm.build.log"
    wheel=$(sed -n 's/^\[BUILD\] Installed: //p' "$OUT/$arm.build.log")
    sha256sum "$wheel" > "$OUT/$arm.wheel.sha256"
done

python3 "$CURRENT/experiments/megamoe/scripts/acceptance_cases.py" generate \
    --machine 950pod --ep-size 8 --case R01 --phase perf --output "$OUT/job.json"
jq '.config' "$OUT/job.json" > "$OUT/current.model.json"
# 仅 R01 的这两个字段等于旧实现的固定语义，旧 schema 不接受字段名。
jq -e 'select(.model.activation=="swiglu" and .model.mma_type=="fp8xfp4")
       | del(.model.activation, .model.mma_type)' "$OUT/current.model.json" > "$OUT/roce.model.json"
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:29180
cd "$OUT"
for arm in current roce; do
    # 默认使用各源码树的锁定 SDK，采用外部 SDK 时按该侧构建日志设置 SHMEM_ROOT。
    src="$CURRENT"; if [ "$arm" = roce ]; then src="$OUT/roce-src"; fi
    ( unset PYTHONPATH
      export LD_LIBRARY_PATH="${SHMEM_ROOT:-$src/third-party/shmem/install/shmem}/lib:${LD_LIBRARY_PATH:-}"
      py="$OUT/$arm-env/bin/python"
      "$py" -I -c 'import sys,torch,torch_npu,deep_ep_experimental.megamoe._C as c; from pathlib import Path; print(c.__file__); assert Path(c.__file__).resolve().is_relative_to(sys.prefix)'
      "$py" -m torch.distributed.run --nnodes=1 --nproc-per-node=8 --master-port=29181 \
          --module deep_ep_experimental.megamoe.benchmark --config "$OUT/$arm.model.json" \
          --input fp8 --routing balanced --mode perf --output-dir "$OUT/$arm-run"
    )
    p="$OUT/$arm-run/rank_0/bs_96/profiling.json"
    jq '{kernel_us:.rank_max_mean_us, e2e_us:.host_e2e.rank_max_mean_us,
         token_s:.host_e2e.global_input_tokens_per_second, rounds, recorded_rounds}' "$p"
done
```

UBX 用相应提交的 UBX 构建环境；各侧分别安装由该提交生成的 wheel，包括多机的每一个节点。保留构建日志、wheel SHA256、各 rank 的实际包路径、`manifest.json` 及 `profiling.json` 或 `performance.json`。同一物理拓扑按顺序运行，固定每卡 BS、H/I/E/K、共享专家、激活、权重位置、量化、输入、路由与 seed，核对实际权重及路由生成方式。两边保持相同 warmup、iters、L2 刷新、生产参数和采样范围。L 组逐臂核对，D 组还要核对实际生效的 groupsize。

按指标填两侧原值、单位、原始路径和加速比。时延加速比为基线时延 / 当前时延，吞吐比为当前吞吐 / 基线吞吐。同步 E2E 从公共 forward 开始到设备同步完成，每轮取最慢 rank，再对正式轮求平均。两分支默认通信后端不同也可在相同完整 forward 和同步口径下填写产品侧 E2E 与吞吐，同时列出两边实际 backend 和窗数。kernel、timer、掩盖率等仅在覆盖的设备工作、时间区间及 rank 聚合方式一致时计算比值。上例的旧版 MTE `profiling.json` 只有 kernel 统计的情况会显示空 E2E；独立按上述同步方法补采前，E2E 填“待测”。

基线 `8a2e4736` 的旧入口默认 SwiGLU A8W4，UBX 产品自动选择 URMA（TopK 上限 8），950POD 走 MTE。每个用例依据实际基线 SHA 的能力核对。无法执行相同 GELU、TopK32、精度、容量或 groupsize 时，在该行填“不可比”、基线 SHA、配置与代码或失败日志，保留当前分支实测值。跨配置的运行、当前分支 wheel 的重复测量或单独的小算子计时均无法产生该行的分支融合基线。

#### 小 EP 逐组结果登记

每格对应一个独立用例。实测后将“待测”替换为本轮报告路径及已有的 Q/P/C/M/H 数值或缺项。标“原表”的格沿用下方原编号行的进度。L 按两臂记录，C 另记 Decode 对照，K 另记 Prefill 对照。

| UBX 编号 / 950POD 编号 | 每卡 BS · K · 激活和加权 · 精度 | UBX EP4 | UBX EP8 | UBX EP16 | 950POD EP4 | 950POD EP8 | 950POD EP16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | 96 · K6 · SwiGLU后加权 · A8W4 · S0 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| R02 | 96 · K6 · SwiGLU后加权 · A8W4 · S1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| R03 | 4096 · K6 · SwiGLU后加权 · A8W4 · S0 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| R04 | 4096 · K6 · SwiGLU后加权 · A8W4 · S1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| R05 | 8192 · K6 · SwiGLU后加权 · A8W4 · S1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| R06 | 12288 · K6 · SwiGLU后加权 · A8W4 · S1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| R07 | 16384 · K6 · SwiGLU后加权 · A8W4 · S1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C01 | 96 · K8 · SwiGLU后加权 · A8W4 | 待测 | 待测 | 待测 | 待测 | 原表 | 待测 |
| C02 | 96 · K8 · SwiGLU后加权 · A8W8 | 待测 | 待测 | 待测 | 待测 | 原表 | 待测 |
| C03 | 96 · K8 · GELU后加权 · A8W8 | 待测 | 待测 | 待测 | 待测 | 原表 | 待测 |
| K01 / K37 | 4096 · K25 · GELU后加权 · A8W4 | 待测 | 待测 | 原表 | 待测 | 待测 | 待测 |
| K02 / K38 | 4096 · K25 · GELU后加权 · A8W8 | 待测 | 待测 | 原表 | 待测 | 待测 | 待测 |
| K04 / K40 | 4096 · K25 · GELU提前 · A8W4 | 待测 | 待测 | 原表 | 待测 | 待测 | 待测 |
| K05 / K41 | 4096 · K25 · GELU提前 · A8W8 | 待测 | 待测 | 原表 | 待测 | 待测 | 待测 |
| K07 / K43 | 4096 · K25 · SwiGLU后加权 · A8W4 | 待测 | 待测 | 原表 | 待测 | 待测 | 待测 |
| K08 / K44 | 4096 · K25 · SwiGLU后加权 · A8W8 | 待测 | 待测 | 原表 | 待测 | 待测 | 待测 |
| K10 / K46 | 4096 · K32 · GELU后加权 · A8W4 | 待测 | 待测 | 原表 | 待测 | 待测 | 待测 |
| K11 / K47 | 4096 · K32 · GELU后加权 · A8W8 | 待测 | 待测 | 原表 | 待测 | 待测 | 待测 |
| K13 / K49 | 4096 · K32 · GELU提前 · A8W4 | 待测 | 待测 | 原表 | 待测 | 待测 | 待测 |
| K14 / K50 | 4096 · K32 · GELU提前 · A8W8 | 待测 | 待测 | 原表 | 待测 | 待测 | 待测 |
| K16 / K52 | 4096 · K32 · SwiGLU后加权 · A8W4 | 待测 | 待测 | 原表 | 待测 | 待测 | 待测 |
| K17 / K53 | 4096 · K32 · SwiGLU后加权 · A8W8 | 待测 | 待测 | 原表 | 待测 | 待测 | 待测 |
| L01 / L04 | 39280 · K25 · A/B · A8W4 | E200 超限 | 待测 | 待测 | E200 超限 | 待测 | 待测 |
| L02 / L05 | 39280 · K25 · A/B · A8W8 | E200 超限 | 待测 | 待测 | E200 超限 | 待测 | 待测 |
| D01 | 1024 · K8 · GELU提前 · A8W4 · GS256 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D02 | 1024 · K8 · GELU提前 · A8W8 · GS256 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D04 | 2048 · K8 · GELU提前 · A8W4 · GS256 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D05 | 2048 · K8 · GELU提前 · A8W8 · GS256 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D07 | 4096 · K8 · GELU提前 · A8W4 · GS256 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D08 | 4096 · K8 · GELU提前 · A8W8 · GS256 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D10 | 8192 · K8 · GELU提前 · A8W4 · GS256 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D11 | 8192 · K8 · GELU提前 · A8W8 · GS256 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D13 | 1024 · K8 · GELU提前 · A8W4 · GS512 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D14 | 1024 · K8 · GELU提前 · A8W8 · GS512 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D16 | 2048 · K8 · GELU提前 · A8W4 · GS512 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D17 | 2048 · K8 · GELU提前 · A8W8 · GS512 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D19 | 4096 · K8 · GELU提前 · A8W4 · GS512 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D20 | 4096 · K8 · GELU提前 · A8W8 · GS512 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D22 | 8192 · K8 · GELU提前 · A8W4 · GS512 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |
| D23 | 8192 · K8 · GELU提前 · A8W8 · GS512 | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS | 待 GS |

### 开测前确认

先跑 R01–R07，再做覆盖和大用例。锁定各参与分支的实际 commit、构建源码状态、wheel SHA256、CANN、torch_npu、物理卡映射和链路。级联比较入口也要包含在测试 wheel 中。

两侧共用模型几何、输入、路由、随机种子和采集范围。同一环境各方法使用独立输出目录。配置预览仅用于核对参数。各行分别标注已测臂、阻塞原因和待补指标，历史版本数据单列。

## 当前验收进度

本分支共 291 组，均待本分支 wheel 的实机验收。下表保留已有构建的实测记录供对照。950POD 有 6 组可核对的融合性能（K37、K46、L04、L05、C01、C02），其中 K37 有旧口径 balanced 全量双参考，K37、K46、L04、L05 有正式性能轮的设备级 HBM 观察值。另有 C03 的性能转录和级联失败日志，以及 R01–R04 的历史分支对照。

| 环境与范围 | 组数 | 已有结果 | 待验收事项 |
| --- | ---: | --- | --- |
| UBX EP16 · K01–K17 | 12 | 本轮无可绑定到编号的实机报告 | 全矩阵、K32 热点及零接收 rank |
| UBX EP32 · K19–K35、L01/L02、D01–D23 | 30 | 本轮无可绑定到编号的实机报告 | 跨机融合、级联、大用例 A/B；豆包 16 组先落实生产 groupsize |
| 950POD EP32 · R01–R07 | 7 | R01–R04 有历史分支对照 | 本轮同版本/同机回归；R05–R07 待 V4.1 真机数据 |
| 950POD EP32 · K37–K53 | 12 | K37/K46 有融合 P、Torch 显存和 HBM 观察值，K37 的旧口径 balanced Q 通过 | K37 新口径全量 Q、修复包 P/C/M/H、K32 压测、Prefill 对照及其余 10 组实测 |
| 950POD EP32 · L04、L05 | 2 | 两组均有 A 臂融合 P、Torch 显存及 HBM 观察值，L04 历史同窗 A/B 达标 | 本轮同窗 A/B 性能与掩盖率、两臂全量 Q 和 GELU Prefill |
| 950POD EP8 · C01–C03 | 3 | C01/C02 有融合 P 和 Torch 显存；C03 有用户回传值 | C03 融合原始报告、Decode 对照、三组 Q 和短时 HBM 复采 |
| 两平台扩展拓扑 · 去重后新增 | 225 | 尚无新增拓扑的实机报告 | 按矩阵逐组执行，记录 P/C/Q/M/H 或对应阻塞 |

UBX 先前回传的 EP16 Dispatch 39.76%、Combine 81.48% 缺少可核对的完整 profile 与采样配置，单列为观察记录，暂不计入验收结果。

转测优先补三项。先在 K37 修复 wheel 上复采融合 P/C/M/H，保留 W2048 × 2；再用同一构建和相同窗口复测 L04 的 A/B，核对两个验收门槛；随后补齐 K37 Prefill 和 C03 Decode 级联阻塞的性能对照。UBX 以对应 U16/U32 后端单独归档。

### 950POD 本轮融合性能和显存占用

首轮使用 `c44fb5a`、CANN b112、Python 3.11，wheel SHA256 为 `8781423ec17987ecd0a40a9675bcf88aede02e5613d35299e7a25850f02065d3`。EP32 使用 B05 os1/os2/os7/os8，EP8 使用单机。warmup=1，正式 3 轮。kernel 取后 2 轮中每轮最慢 rank 的均值，多窗先累加各窗 kernel 时延；同步 E2E 取全部 3 轮每轮最慢 rank 的均值。吞吐为全组输入 token/s。显存栏是性能进程所有 rank 的最大值，以 GiB 计。各臂常驻显存和设备峰值仍待独立采样。

| Case / 融合臂 | 每卡 BS / 窗口数 | kernel（ms） | 同步 E2E（ms） | 吞吐（token/s） | Torch 峰值 allocated / reserved（GiB） | workspace / 对称区（GiB） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K37 · GELU 后 · A8W4 | 4096 / 2 | 37.080 | 45.368 | 2,889,070 | 0.906 / 1.137 | 0.316 / 0.703 |
| K46 · GELU 后 · A8W4 | 4096 / 2 | 48.513 | 57.288 | 2,287,934 | 0.907 / 1.137 | 0.316 / 0.895 |
| L04 · A · A8W4 | 39280 / 20 | 102.039 | 109.545 | 11,474,394 | 2.139 / 2.346 | 1.276 / 0.406 |
| L05 · A · A8W8 | 39280 / 20 | 148.136 | 155.924 | 8,061,365 | 1.538 / 1.578 | 0.531 / 0.406 |
| C01 · SwiGLU 后 · A8W4 | 96 / 1 | 0.884 | 1.224 | 627,415 | 0.942 / 1.389 | 0.152 / 0.006 |
| C02 · SwiGLU 后 · A8W8 | 96 / 1 | 0.754 | 2.393 | 320,987 | 1.014 / 1.027 | 0.135 / 0.006 |

首轮产物根目录为 B05 共享盘 `/mnt/share/l00911701_megamoe_b05/megamoe-topk32`，下文记为 `$BASE`。K、L 行来自 `$BASE/runs/<Case>/perf/perf/rank_*/bs_<BS>/{profiling,resources}.json`。C01/C02 来自 `$BASE/runs/<Case>/perf-n/rank_*/bs_96/`。K37 表中 P/M 均为精度修复前的数据。L04、L05 表中均为 GELU 提前加权融合臂，还需本轮 SwiGLU 和 GELU Prefill 同配置结果。

C03（GELU 后加权 A8W8）首轮交接记录为 kernel 271 µs、E2E 1.167 ms、吞吐约 658,000 token/s。现有 C03 目录只有 Decode 对照失败日志，上述数值保留为转录，待补独立融合原始报告及最慢 rank 口径。C03 的 Decode 在 `aclnnMoeDistributeDispatchV4` 报 561002（`Get WinSize failed`）。K37 的 Prefill 在收发计数 `ReduceSum` 报 507899，日志同时记录驱动内存分配失败。两项级联性能、融合对级联加速比均待补。原日志为 `$BASE/runs/C03/perf/perf-w8a8-node0.log` 和 `$BASE/runs/K37/prefill/perf-w4a8-node0.log`。

### 950POD 精度结果

各行均为历史 Q，使用 balanced 路由及当时的级联误差预算。此前 32/32 rank 通过，输出全为有限值，融合和旧精度级联的专家计数均与 Golden 一致。K37 的新口径 Q 仍待以相同配置重跑融合对 CPU 和融合对 Prefill 两项直接比较。

| Case | 版本 / 实际窗口 | 每卡 / 全组校验 token | 修复前失败 rank | 融合对 CPU 最大绝对误差 | bad_elements |
| --- | --- | ---: | --- | ---: | ---: |
| K37 | 修复 wheel · W2048 × 2 | 4,096 / 131,072 | 21–31（11 卡） | 0.000244140625 | 0 |
| K37 边界复测 | 修复 wheel · W257 × 1 | 257 / 8,224 | 30–31（2 卡） | 0.00048828125 | 0 |

K37 修复 wheel 基于 `c44fb5a` 加 A8W4 Dispatch 补丁，SHA256 为 `238551544141378c1169fa19e30da04c12bfa75f37de5a0a5d151ace252208e3`。完整 K37 在 os1/os2/os3/os8 跑，BS257 在 os1/os2/os7/os8 跑。四节点均正常退出。修复包这轮只采 Q，BS257 是 K37 的边界验证，计数沿用 K37 一个编号。两次汇总分别在 `$BASE/fixed-wheel-ihXs7t90/K37-accuracy-_ch2k2vc/verified-summary.json` 和 `K37s-accuracy-ksx7_pdj/verified-summary.json`（同一父目录）。工作区内可查看[完整 K37 核验报告](../benchmark-runs/b05-k37-dispatch-fix-ihXs7t90/K37/verified-summary.json)和[BS257 核验报告](../benchmark-runs/b05-k37-dispatch-fix-ihXs7t90/BS257/verified-summary.json)。

### 正式性能轮内的 HBM 带宽

下表为首轮融合臂**每卡设备级 HBM 观测带宽**。每个 rank 独立选取正式轮中首窗 kernel 开始到末窗 kernel 结束的设备时间段，仅保留完整落在段内的硬件采样区间。相邻样本约隔 20 ms。按卡汇总 4 个 HBM 分片，先对每张卡按有效时长加权，再对有样本的卡等权。读写合计按表中读、写两列相加。设备侧流量包含同卡并发活动，覆盖情况随用例列出。

| 融合臂 | 读（GB/s） | 写（GB/s） | 读写合计（约 GB/s） | 有样本 rank | 有样本 rank × 正式轮 | 有效 / 剔除异常采样窗 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K37 · 修复前 | 56.89 | 65.62 | 122.51 | 27/32 | 42/64 | 42 / 1 |
| K46 | 62.50 | 72.04 | 134.54 | 32/32 | 62/64 | 77 / 1 |
| L04 · A | 200.00 | 227.43 | 427.43 | 32/32 | 64/64 | 231 / 5 |
| L05 · A | 137.03 | 165.02 | 302.05 | 32/32 | 64/64 | 369 / 7 |
| C01 / C02（各 EP8） | 待复采 | 待复采 | 待复采 | 各 0/8 | 各 0/16 | 各 0 / 0 |

C01/C02 的单轮 kernel 仅约 0.3–1.1 ms，当前采样周期内没有完整窗。C03 缺少独立融合 profile。K 组的缺样 rank 或缺样轮不补零；复测重点是延长稳定采集段、补齐覆盖。L04、L05 覆盖全部卡和正式轮，已剔除的异常窗见表。正式验收待独立采集并补足所需各臂。

### 历史分支对照

**仓基准 R01–R04。** 下表为 B07 EP32、balanced 的历史对照，斜线顺序为 main `a012a97d` / 基于 `88936f1a` 的 fix-v2。时延为每轮最慢 rank。掩盖率采用实际打点区间并集；通信带宽取报告的跨 rank 有效 payload 带宽。R02/R03 的 E2E 有上升，R04 的 Dispatch 有效带宽有下降，需用本轮构建重测后判定。B07 全部逐臂指标见 `benchmark-runs/950pod-a8w4-main/report/comparison.csv`，B05 对照在 `/mnt/share/l00911701_megamoe_b05/main-regression.RlQBDIgG/verified-forward-report/comparison.csv`。

| Case | E2E（ms） | kernel（ms） | 总掩盖率（%） | Dispatch 通信带宽（GB/s） | Combine 通信带宽（GB/s） |
| --- | ---: | ---: | ---: | ---: | ---: |
| R01 | 4.609 / 4.057 | 0.308 / 0.309 | 89.319 / 89.356 | 59.96 / 59.89 | 39.92 / 40.19 |
| R02 | 3.736 / 4.534 | 0.328 / 0.330 | 99.940 / 99.870 | 62.45 / 60.76 | 38.46 / 38.32 |
| R03 | 8.496 / 8.985 | 4.681 / 4.678 | 92.905 / 92.862 | 132.95 / 130.32 | 103.02 / 102.04 |
| R04 | 8.788 / 8.282 | 5.061 / 5.064 | 99.892 / 99.923 | 147.82 / 132.99 | 97.76 / 97.43 |

B07 的 `feat/roce-support`（`8a2e4736`）另有 R01、R03 数据。按 E2E / kernel / 总掩盖率，R01 为 3.602 ms / 0.309 ms / 89.217%，R03 为 8.807 ms / 4.673 ms / 93.068%。原始目录为 `/mnt/share/l00911701_megamoe_b05/roce-regression.i6dM3Uw9/{b07-s0,b07-s0-4096}`。R05–R07 的 V4.1 参数已确定，待实测。

**950POD 大用例 A/B。** L04 历史对照使用 B05 四机、每卡 BS39280、K25，A 与 B 使用相同的窗口。下表取全部窗口、每台 4 个偶数 rank 的实际打点区间并集。时延列是同步 E2E 的均值，比值列是 8 对正式样本的 A/B 时延比中位数。

| Case / 历史版本 | 每臂窗口 | A / B E2E（ms） | A/B 比 | A / B Dispatch（%） | A / B Combine（%） | A / B 总掩盖率（%） | 当时的判定 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| L04 · A8W4 · `4e68b947` | 19 × 2048 + 368 | 107.349 / 127.269 | 0.8444 | 87.937 / 97.068 | 87.201 / 17.572 | 84.462 / 50.893 | 两项达标，总掩盖率 +33.569 pp |

L04 历史 Q 为 A 臂 32/32、每卡抽 64 token，最大绝对误差 0.000244140625。L04 在本分支的全量双参考与同窗 A/B 均待补。L04 的依据为 `benchmark-runs/b05-ep32-a8w4-bs39280-fix/release/{acceptance,strict-result}.json`。

## UBX 测试方法

[UBX EP16 运行预设](../experiments/megamoe/scripts/configs/ubx_ep16_target.json) · [UBX EP32 运行预设](../experiments/megamoe/scripts/configs/ubx_ep32_target.json) · [掩盖率采集](../experiments/megamoe/scripts/ubx_overlap_quickstart.md)

U4、U8、U16 分别取 .84 上 4、8、16 张卡使用本机 MTE。U32 为双机 EP32 MTE+RoCE。小 EP 的 JSON、采样卡号见[分平台 EP 验收矩阵](#分平台-ep-验收矩阵)。

### U16 与 U32 启动准备

U16 在 90.90.93.84 使用 16 卡，复制 ubx_ep16_target.json，保留 backend=mte。U32 在 90.90.93.84 发起，两台各 16 卡，节点列表为 90.90.93.84 和 90.90.93.188。复制 ubx_ep32_target.json，将 backend 保持为 mte_roce。双机复用相同路径、wheel 和软件环境，确认 SSH 及训练、SHMEM 和数据面链路可用。

```bash
bash experiments/megamoe/scripts/build_ubx.sh
RUN=experiments/megamoe/scripts/run_megamoe.py
python3 "$RUN" --run-config "$RUN_JSON"
```

### U 原生融合

从 case 填写运行 JSON 的 precision、output_dir、routing 和内嵌 config。将 H、I、E、K、S 依次填入 config.model 的 hidden、intermediate_hidden、num_experts、num_topk、num_shared_experts。实际 BS 写入 config.bs_values，并设置 activation 和 topk_weights_type。去掉会覆盖这些值的旧字段。plan=true 可先核对预览，正式执行时设为 false。性能设 mode=perf、timer=false。精度另用 mode=accuracy、dual_reference=true，正式精度 check_tokens=0，C01–C03 的 accuracy_cascade=decode，其他用例为 prefill。

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

P4/P8 为 B05 单机 4/8 卡，P16 为两台各 8 卡，P32 为四台各 8 卡。B05 主地址为 177.22.21.1。

### P32 与 P8 启动准备

默认沿用 B05 的 CANN b112 环境和已安装 MegaMoE wheel 的 Python。P32 四台的 node_rank 依次为 0 至 3，各自填写本机 Python 和 CANN 路径。四台协调使用相同的模型、主地址、端口和目录结构。P8 只在 B05 启动。

```bash
source /home/cann/b112/cann-9.2.0/set_env.sh
PY=/opt/miniconda3/envs/lsx/bin/python
export PATH="$(dirname "$PY"):$PATH"
RUN=experiments/megamoe/scripts/run_megamoe.py
```

### W 配置文件入口

复制 950pod_target.json，按 case 填 model、BS、EP、precision、node_rank 和 output_dir。用 mode=perf 或 mode=accuracy 分开启动，后者设 dual_reference=true，C01–C03 另设 accuracy_cascade=decode，其余设 prefill。对 GELU Prefill 另开 perf 作业并加 --compare-prefill-cascade。P8 Decode 另开 perf 作业并加 --compare-decode-cascade。

```bash
python3 "$RUN" --run-config "$RUN_JSON"
python3 "$RUN" --run-config "$CASCADE_JSON" --compare-prefill-cascade
python3 "$RUN" --run-config "$DECODE_JSON" --compare-decode-cascade
```

### N 直接启动融合

用于旧分支原配回归、950POD 的 A8W4/A8W8 SwiGLU 原生融合，以及 950POD 单独采 timer。CFG 为只包含 schema_version、model、bs_values、seed、warmup、iters 等的模型 JSON。P32 设 NNODES=4、NODE_RANK=0 至 3、MASTER=177.22.21.1。P8 设 NNODES=1、NODE_RANK=0、MASTER=127.0.0.1。当前分支设置 model.mma_type，A8W4、A8W8 分别为 fp8xfp4、fp8xfp8。两种精度均使用 INPUT=fp8，由 950POD 自动选择后端。R 组复用历史基线时沿用原输入类型、seed、预热与迭代数。

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

每个 Case 的 M 栏填写 **整卡采样峰值（GiB）、峰值占用率（%）、对应 rank 和有效卡数**。UBX、950POD 的精度作业默认采样，正式性能、级联计时与 timer 作业默认关闭。部署包含 `device_memory.py` 的 wheel 到参与节点，再用同版启动脚本。对比同一批物理卡，并记录运行前空载情况，整卡读数包含同卡其他进程。

从仓库根目录独立复测 L01 的显存，替换 JSON 即可用于 L02、L04、L05：

```bash
python experiments/megamoe/scripts/run_megamoe.py --run-config experiments/megamoe/scripts/configs/l0/L01.json --only memory
```

K、C、D 等其他 Case 复制各自的性能 JSON，在外层设置 `"mode": "perf"`、`"perf_scope": "memory"`、`"profile_memory_peak": true`、`"timer": false` 并指定新的输出目录，保留该 Case 的模型、拓扑和融合或级联开关。直接 torchrun 运行 benchmark 时对应参数为 `--mode perf --perf-scope memory --profile-memory-peak`。显存轮执行预热及正式次数的 forward，跳过性能计时和 profiler。

每个 rank 在共享初始化及各 BS 执行期间，按 **10 ms** 间隔并在阶段首尾读取设备总量和剩余量。峰值占用率取该卡采到的最大已用字节 / 设备总字节 × 100%。短于采样间隔的尖峰可能漏采，可用 `case_periodic_sample_count` 核对该 BS 执行期间取得的周期样本数。同一进程测试多个 BS 时，共同的初始化峰值计入各 BS，前一 BS 留存的张量和分配器缓存也可能影响后续读数。需要独立比较时，每次只配置一个 BS。

| 回填项 | 数据位置及口径 |
| --- | --- |
| 每卡峰值和阶段 | `rank_<r>/bs_<B>/resources.json` 的 `device_memory_peak`。读 `peak_used_gib`、`peak_used_pct`、`sample_count`。普通 benchmark 的 `setup_peak_used_bytes` 和 `case_peak_used_bytes` 分别表示准备阶段和本 BS 执行阶段的采样最高值。 |
| 全组最紧的卡 | 同文件 `device_memory_peak_group`。记录 `max_peak_used_bytes`、`max_peak_used_pct` 和各自对应的 rank，同时记录 `min_headroom_bytes` 与 `rank_count`。L0 还把该对象和原始报告路径写到 `run-summary.json` 的对应 job。 |
| 显存归属 | 单独的 GELU、SwiGLU 融合作业分别回填。双参考、融合与级联对照、950POD 配对基准按报告的共驻范围填写。原有 `device_free_after_case_bytes` 继续表示结束快照，Torch 的 allocated/reserved 单独记录。 |

需要设备侧采样日志时，可在每个节点额外留存：

```bash
while true; do date -Iseconds; npu-smi info; sleep 1; done > "$HBM_LOG"
```

结束后停止采样，归档每节点原始日志。短 Decode 用例用重复批次采样，标明定频采样峰值。PyTorch allocated/reserved 与设备显存分列。

### HBM 显存带宽采集

两平台各 EP 均按 Case、构建和臂单独采集，保留每个 rank 的 profiler 数据、设备 ID、时钟域、采样周期、正式轮及 kernel 名。HBM 读写以十进制 GB/s 表示。M 的显存峰值以 GiB 表示，C 的有效带宽按通信 payload/通信时长计算。

1. 在独立输出目录运行同参数的正式性能配置，timer 关闭。核对正式轮数、每轮窗口数及 `profiling.json`。对融合臂取同一设备同一轮的首末融合 kernel，级联取该臂一次 forward 的完整设备跨度。核对参与的 rank 数等于该组 EP。
2. 检查各 rank 的 `ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db` 是否有 `HBM` 及同设备时钟的任务时间。用相邻采样区间积分，先汇总该设备各 HBM 分片，再对有效卡汇总，并登记分母、样本数和异常处理。缺少时间戳或轮次对应关系时，全程 `hbm_*.csv` 的 Average 仅能作为整段 profile 的观察记录。[CANN 显存读写速率字段](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0084.html)
3. 重采时先检查本机 `msprof --help` 支持的设备级 `--sys-hardware-mem` 和 `--sys-hardware-mem-freq`；B05 b112 的 HBM 采样最高 100 Hz。短 Decode 可在同一臂安排连续稳定的重复批次，记录起止和迭代次数，以批次报告设备带宽。需要单个 kernel 的主存访问数据时，另开 PMU profiling，B05 可选 `AiCMetrics.Memory` 并检查实际导出的主存读写字段，按 AI Core 边界单列。UBX 以该机 CANN 版本的实采字段和分片数为准。[CANN 系统硬件采样参数](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta2/devaids/Profiling/atlasprofiling_16_0012.html)

B05 b112 重采示例。每台节点设置独立的 `HBM_OUT`，在空闲且已分配的卡上运行设备采样，使采样段覆盖同机同臂的连续正式 forward。UBX 每台按实际设备号填写，不能沿用 B05 的 8 卡列表。

```bash
msprof --sys-devices=0,1,2,3,4,5,6,7 --sys-period=60 \
  --sys-hardware-mem=on --sys-hardware-mem-freq=100 --output="$HBM_OUT"
```

<details>
<summary>复算 B05 的 HBM 结果（离线，只读已归档的 profiler DB）</summary>

在共享盘上执行以下示例。K/L 用例替换 Case 名，C01/C02 将 `PERF_ROOT` 指到对应的 `perf-n`。此批导出数据的 `bandwidth` 为 B/s，时间戳为 ns，四个分片编号为 0–3。对这次 b112 数据，按每计数 32 B 还原的单分片增量有两簇，正常样本最大约 5.17×10⁷，异常样本最小约 1.37×10¹¹。示例以两簇间的 2³⁶ 标记并剔除异常采样窗，表中保留剔除数；更换板型或计数器时需重新核对。

```bash
PERF_ROOT=/mnt/share/l00911701_megamoe_b05/megamoe-topk32/runs/K37/perf/perf
python3 - "$PERF_ROOT" <<'PY'
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
world = json.loads((root / "rank_0/manifest.json").read_text())["ep_size"]
rates, rounds_hit, kept, rejected = [], 0, 0, 0
for rank in range(world):
    case, = (root / f"rank_{rank}").glob("bs_*")
    perf = json.loads((case / "profiling.json").read_text())
    width = json.loads((case / "resources.json").read_text())["windows_per_forward"]
    rounds, first = perf["recorded_rounds"], perf["first_selected_iteration"]
    dbfile, = case.glob("profiler/*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db")
    with sqlite3.connect(dbfile.resolve().as_uri() + "?mode=ro", uri=True) as db:
        kernels = db.execute("""SELECT t.startNs, t.endNs FROM TASK t
            JOIN COMPUTE_TASK_INFO c USING(globalTaskId) JOIN STRING_IDS s ON s.id=c.name
            WHERE instr(s.value,'_Z15mega_moe_kernel')>0 ORDER BY t.startNs""").fetchall()
        assert len(kernels) == width * rounds
        samples = defaultdict(dict)
        for ts, direction, part, bw in db.execute("""SELECT h.timestampNs,s.value,h.hbmId,h.bandwidth
            FROM HBM h JOIN STRING_IDS s ON s.id=h.type ORDER BY h.timestampNs"""):
            samples[ts][direction, part] = bw
    times = sorted(samples)
    elapsed, sums = 0, [0, 0]
    for j in range(first, rounds):
        lo, hi = kernels[j*width][0], kernels[(j+1)*width-1][1]
        hits = 0
        for a, b in zip(times, times[1:]):
            if not lo <= a < b <= hi:
                continue
            v = samples[b]
            dt = b - a
            assert set(v) == {(d, i) for d in ("read", "write") for i in range(4)}
            if max(v.values()) * dt / 32e9 >= 2**36:
                rejected += 1
                continue
            elapsed += dt
            for k, d in enumerate(("read", "write")):
                sums[k] += sum(v[d, i] for i in range(4)) * dt
            hits += 1
        kept += hits
        rounds_hit += int(hits > 0)
    if elapsed:
        rates.append([s / elapsed / 1e9 for s in sums])
gbps = [round(statistics.mean(col), 2) for col in zip(*rates)] if rates else None
print(root, "read/write GB/s", gbps, "ranks", len(rates), "/", world,
      "rank-rounds", rounds_hit, "/", world*(rounds-first), "samples", kept, "excluded", rejected)
PY
```

</details>

### 留档字段

按 Case 编号、分支或 A/B、融合或级联以及 perf/accuracy/timer/memory/hbm-bw 分开归档。保留运行 JSON、commit 与补丁、wheel SHA256、采集范围、rank 和原始文件。

## 采集口径与结果填写

Case 表的 **P / C / Q / M / H** 分别对应性能、通信、精度、显存占用、HBM 显存读写带宽。在一个 Case 下按分支、A/B 或融合/级联逐臂填值并附原始报告。

| 栏位 | 每个 Case 按臂填写 | 验收和记录方法 |
| --- | --- | --- |
| P 性能 | kernel 总时延、设备跨度、同步 E2E（µs），输入吞吐（token/s）。级联另记加速比。 | 关闭 timer 做正式计时。kernel 取 `profiling.json.rank_max_mean_us`，同步 E2E 取 `host_e2e.rank_max_mean_us`，吞吐取 `host_e2e.global_input_tokens_per_second`。分别记录各自的正式轮数。加速比按级联耗时 / 融合耗时。 |
| C 通信 | Dispatch、Combine、总掩盖率（%），有效带宽（GB/s）及字节和时间口径。 | 独立打点。仅比较相同后端、事件范围和采样 rank 的结果。分子为通信与计算交集，分母为通信时长。总通信先合并两段区间，再计算占比。级联有可比独立 trace 时另记。 |
| M 显存 | 每卡常驻量、整卡采样峰值（GiB、%）、峰值 rank、最小余量和有效卡数。另记 PyTorch allocated/reserved 峰值、workspace、对称缓冲区。 | 按[显存复测](#ubx-与-950pod-显存复测)读取各 Case 的 resources.json。分臂比较使用独立作业，级联对照的共驻值单列。 |
| H 显存带宽 | 记录单卡 HBM 读、写及同一时段读写合计（GB/s），注明有效 rank 数、正式轮次及采样范围。 | 从设备 HBM 计数器取数。分臂汇总有效卡均值，附单卡值；容量、通信 payload 带宽另填 M、C。[采集方法](#hbm-显存带宽采集) |
| Q 精度 | 参考路径、通过 rank 数、实查 token 数、两项直接比对、专家计数及结论。 | 在直接双参考作业中分别登记 `fused_vs_cpu` 和 `fused_vs_cascade` 的 `passed`、`bad_elements`、`max_abs_error`、`max_mixed_error`。检查每个 rank 的 `passed`、`finite`、`counts_equal`、`cascade_counts_equal`、`cumulative_counts_equal` 和 `checked_tokens`。另存 `cascade_vs_cpu` 的诊断值。 |

### 950POD 设备数据采集

Q 先按以下命令预生成 Golden。RANKS 是本节点的全局 rank 范围，EP 是总 rank 数。将 N 命令改为 `--mode accuracy`，追加 `--dual-reference --accuracy-cascade "$KIND" --golden-dir "$GOLDEN" --check-tokens 0`。C01–C03 取 `KIND=decode`，其他用例取 `KIND=prefill`。

```bash
"$PY" -m "$MOD.golden" --config "$CFG" --ep-size "$EP" --ranks "$RANKS" \
  --input "$INPUT" --routing balanced --golden-dir "$GOLDEN" --check-tokens 0
```

950POD 的 C 使用 N 方法追加 --timer 和 --timer-ranks。P32 每节点取 4 卡，传入 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30。归档 timer_breakdown.json 和原始打点。以 wheel 能力为准。MTE A8W8 记录待 timer。双窗或首窗指标注明实际 token 范围。

### 未齐备指标

K 组的 SwiGLU Prefill 对照按平台、EP、TopK 和精度分别实测。首轮 wheel 的 Prefill 入口仅支持 GELU，复测时记录含 SwiGLU 入口的 wheel 和对照报告。D 组需确认 groupsize 256/512 在生产路径中的真实参数。级联单臂显存采集需独立入口，当前双臂对照报告的共驻进程显存标为待采。旧分支缺同口径的计时、带宽或占比时保留原 report，明示缺测。

各表的 P、C、Q、M、H 按本节字段逐臂填写。P 附 profiling.json、performance.json 或级联对照 JSON，Q 附 accuracy.json，M 附 resources.json 和设备采样日志，C 附汇总和原始 timer，H 附带 rank 与采样时间的原始 profiler 数据。相同口径的值并列，暂缺的数据填待测或对应原因。

**逐臂结果登记模板**

| Case / 分支或臂 | Kernel / 设备跨度 / E2E（µs） | 吞吐（token/s） | 通信有效带宽（GB/s） / Dispatch、Combine、总掩盖率 | 精度和计数 | 常驻 / 设备峰值（GiB）及峰值 rank | HBM 读 / 写 / 合计（GB/s）及覆盖 | 构建与原始产物 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 待填 | 待测 | 待测 | 待测 | 待测 | 待测 | 待采 | 待填 |

## 仓基准与 Decode 用例

下表的原始 R 组使用 P32 和 N 方法，原始 C 组使用 P8，Decode 对照用 W 方法，首轮 C01/C02 的独立融合数据用 N 方法。扩展拓扑沿用表内每卡 BS、模型和激活，按[分平台 EP 验收矩阵](#分平台-ep-验收矩阵)生成运行 JSON。

### 仓基准回归

V4 Pro 取 H7168、I3072、E384、K6。V4.1 取 H5120、I2304、E384、K6。全部按分支原配执行 SwiGLU A8W4。S 是每卡共享专家数。R 组精度栏记当前分支结果。

| Case | 模型或 BS | 变量和精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 | H 显存带宽 |
| --- | --- | --- | --- | --- | --- | --- |
| `R01` | V4 Pro · BS96 | S0 · A8W4 | [N](#n-直接启动融合) · 当前融合，另附历史分支结果 | [历史对照](#历史分支对照)有 P/C · 本轮待测 | Q 待测 · M 待测 | 待采 |
| `R02` | V4 Pro · BS96 | S1 · A8W4 | [N](#n-直接启动融合) · 当前融合，另附历史分支结果 | [历史对照](#历史分支对照)有 P/C · 本轮待测 | Q 待测 · M 待测 | 待采 |
| `R03` | V4 Pro · BS4096 | S0 · A8W4 | [N](#n-直接启动融合) · 当前融合，另附历史分支结果 | [历史对照](#历史分支对照)有 P/C · 本轮待测 | Q 待测 · M 待测 | 待采 |
| `R04` | V4 Pro · BS4096 | S1 · A8W4 | [N](#n-直接启动融合) · 当前融合，另附历史分支结果 | [历史对照](#历史分支对照)有 P/C · 本轮待测 | Q 待测 · M 待测 | 待采 |
| `R05` | V4.1 · BS8192 | S1 · A8W4 | [N](#n-直接启动融合) · 当前融合 | P 待测 · C 待测 | Q 待测 · M 待测 | 待采 |
| `R06` | V4.1 · BS12288 | S1 · A8W4 | [N](#n-直接启动融合) · 当前融合 | P 待测 · C 待测 | Q 待测 · M 待测 | 待采 |
| `R07` | V4.1 · BS16384 | S1 · A8W4 | [N](#n-直接启动融合) · 当前融合 · 核对分窗 | P 待测 · C 待测 | Q 待测 · M 待测 | 待采 |

EP32 的 R 组保留 main、feat/roce-support 的历史结果供追溯。EP4/8/16 的 R 组按本轮基线另测；R07 先核实旧分支 BS16384 的原生容量和分窗。

### Decode 级联

固定 BS96、H3584、I1280、E384、K8、S0、后加权。用 P8 的 W 方法加 --compare-decode-cascade。Q 单独运行 mode=accuracy、dual_reference=true、accuracy_cascade=decode，并登记融合对 CPU、融合对 Decode 的两项结果。

| Case | 模型或 BS | 变量和精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 | H 显存带宽 |
| --- | --- | --- | --- | --- | --- | --- |
| `C01` | BS96 · P8 | SwiGLU · A8W4 | [W](#w-配置文件入口) · 融合 / Decode 级联 | N 方法[融合实测](#950pod-本轮融合性能和显存占用) · Decode/C 待测 | Q 待测 · M 有 Torch 峰值，设备峰值待采 | 0/8 卡有完整窗，待复采 |
| `C02` | BS96 · P8 | SwiGLU · A8W8 | [W](#w-配置文件入口) · 融合 / Decode 级联 | N 方法[融合实测](#950pod-本轮融合性能和显存占用) · Decode 待测 · C 待 timer | Q 待测 · M 有 Torch 峰值，设备峰值待采 | 0/8 卡有完整窗，待复采 |
| `C03` | BS96 · P8 | GELU · A8W8 | [W](#w-配置文件入口) · 融合 / Decode 级联 | 融合[转录待核](#950pod-本轮融合性能和显存占用) · Decode 561002 · C 待 timer | Q 待测 · M 待分臂采 | 待采 |

C03 的级联 GMM1 使用 CANN GroupedMatmulActivationQuant 与 GELU tanh。融合 MegaMoE 使用 GELU erf。性能和精度记录注明实际近似及舍入口径。

## TopK 和激活 UBX EP16

统一 BS4096、H7168、I3072、E384、S0。balanced 为性能主跑。精度另查 K32 的热点路由与零接收 rank，记录在同一 Case。

| Case | TopK | 激活和加权 | 精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 | H 显存带宽 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `K01` | 25 | GELU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K02` | 25 | GELU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 | 待采 |
| `K04` | 25 | GELU 提前加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K05` | 25 | GELU 提前加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 | 待采 |
| `K07` | 25 | SwiGLU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，SwiGLU 级联待实测 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K08` | 25 | SwiGLU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，SwiGLU 级联待实测 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 | 待采 |
| `K10` | 32 | GELU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K11` | 32 | GELU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 | 待采 |
| `K13` | 32 | GELU 提前加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K14` | 32 | GELU 提前加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 | 待采 |
| `K16` | 32 | SwiGLU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，SwiGLU 级联待实测 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K17` | 32 | SwiGLU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，SwiGLU 级联待实测 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 | 待采 |

加权模式 GELU 提前=1、GELU 后=0、SwiGLU 后=0。GELU 的 GMM1 N 为 3072，SwiGLU 为 6144。

## TopK 和激活 UBX EP32

统一 BS4096、H7168、I3072、E384、S0。balanced 为性能主跑。精度另查 K32 的热点路由与零接收 rank，记录在同一 Case。

| Case | TopK | 激活和加权 | 精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 | H 显存带宽 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `K19` | 25 | GELU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K20` | 25 | GELU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K22` | 25 | GELU 提前加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K23` | 25 | GELU 提前加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K25` | 25 | SwiGLU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，SwiGLU 级联待实测 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K26` | 25 | SwiGLU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，SwiGLU 级联待实测 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K28` | 32 | GELU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K29` | 32 | GELU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K31` | 32 | GELU 提前加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K32` | 32 | GELU 提前加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K34` | 32 | SwiGLU 后加权 | A8W4 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，SwiGLU 级联待实测 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K35` | 32 | SwiGLU 后加权 | A8W8 | [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 融合，SwiGLU 级联待实测 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |

加权模式 GELU 提前=1、GELU 后=0、SwiGLU 后=0。GELU 的 GMM1 N 为 3072，SwiGLU 为 6144。

## TopK 和激活 950POD EP32

统一 BS4096、H7168、I3072、E384、S0。balanced 为性能主跑。精度另查 K32 的热点路由与零接收 rank，记录在同一 Case。

| Case | TopK | 激活和加权 | 精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 | H 显存带宽 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `K37` | 25 | GELU 后加权 | A8W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | 修复前 [融合实测](#950pod-本轮融合性能和显存占用)，修复包 P/C 待测 · Prefill 阻塞 | [旧口径全量 Q](#950pod-精度结果) 32/32，新 Q 待测 · M 有旧包 Torch 峰值，修复包待采 | [HBM 观察](#正式性能轮内的-hbm-带宽)（旧包 27/32 卡） |
| `K38` | 25 | GELU 后加权 | A8W8 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 | 待采 |
| `K40` | 25 | GELU 提前加权 | A8W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K41` | 25 | GELU 提前加权 | A8W8 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 | 待采 |
| `K43` | 25 | SwiGLU 后加权 | A8W4 | [G](#g-950pod-a8w4-配对分窗) · 融合同窗 · SwiGLU 级联待实测 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K44` | 25 | SwiGLU 后加权 | A8W8 | [N](#n-直接启动融合) · 融合待同窗 · SwiGLU 级联待实测 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 | 待采 |
| `K46` | 32 | GELU 后加权 | A8W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | [融合实测](#950pod-本轮融合性能和显存占用) · 级联/C 待测 | Q 待测 · M 有 Torch 峰值，设备峰值待采 | [HBM 观察](#正式性能轮内的-hbm-带宽)（62/64 卡轮） |
| `K47` | 32 | GELU 后加权 | A8W8 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 | 待采 |
| `K49` | 32 | GELU 提前加权 | A8W4 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K50` | 32 | GELU 提前加权 | A8W8 | [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 / Prefill 级联 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 | 待采 |
| `K52` | 32 | SwiGLU 后加权 | A8W4 | [G](#g-950pod-a8w4-配对分窗) · 融合同窗 · SwiGLU 级联待实测 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `K53` | 32 | SwiGLU 后加权 | A8W8 | [N](#n-直接启动融合) · 融合待同窗 · SwiGLU 级联待实测 | P 待测 · C 待 timer | Q 待测 · M 待分臂采 | 待采 |

加权模式 GELU 提前=1、GELU 后=0、SwiGLU 后=0。GELU 的 GMM1 N 为 3072，SwiGLU 为 6144。

SwiGLU 的 A8W4/A8W8 在 BS4096 的整批对称区超限。A8W4 可用 G 方法先测计时。A8W8 的同窗版本，以及按完整 BS 的精度验证，待对应入口。

## EP32 大用例

所有用例均为 BS39280、H4096、I1536、E800、K25、S0，balanced。UBX 与 950POD 各做 A8W4 和 A8W8。

L01、L02、L04、L05 的运行 JSON 见下表。在对应平台从仓库根目录使用[统一入口](../experiments/megamoe/scripts/run_megamoe.py)执行，例如 UBX 节点 0 上运行 `python3 experiments/megamoe/scripts/run_megamoe.py --run-config experiments/megamoe/scripts/configs/l0/L01.json`。追加 `--plan` 查看计划，使用 `--only perf` 或 `--only accuracy` 可分别运行性能或精度。L04、L05 在 950POD 环境运行。

UBX 从 `90.90.93.84` 启动，两台各自在本机 Python 环境安装同一份 UBX wheel。若需固定 `.188` 的解释器，在 L01、L02 JSON 外层填写该节点解释器绝对路径 `peer_python`。B05 从 `177.22.21.1` 启动，默认使用 os1/os2/os7/os8 和预设中已归档的 K37 修复环境。测试新构建时将 wheel 安装到选用的共享环境。各节点需能访问相同路径的代码和输出，并允许主节点通过 SSH 拉起。

自动输出目录按 `benchmark-runs/<Case>-<平台>-ep<卡数>-<精度>-<运行类型>-YYYYMMDD-HHMMSS/` 命名，时间取启动节点本地时间。例如 `L04-950pod-ep32-a8w4-perf-20261124-143005/`。运行类型与 `--only` 一致，默认完整验收标为 `full`。同秒同名时追加 `-02`、`-03`。用 `RUN_BASE` 更换父目录，或在 JSON 中用 `output_dir` 指定完整路径。按本轮目录的 `run-summary.json` 回填结果，UBX 首窗对照见 `native-overlap.json`。只跑 GELU 融合与 Prefill 级联可用 `--only prefill`。

L04 的融合性能使用同窗 A/B 配对，另存 `paired/paired.json`。L04、L05 的 SwiGLU 全量双参考和 L05 的 SwiGLU 融合性能受整批对称区容量限制。950POD 的全窗掩盖率仍需两臂逐窗补采。预设会执行可运行的阶段，完成后如仍有上述缺项，清单运行状态为 `partial`，退出码为 3。其他运行故障会在具体阶段记录 `failed`。独立设备显存采样和 HBM 计数器采集仍按下方验收口径回填。

A 为 GELU 提前加权，B 为 SwiGLU 后加权。A 的 GMM1 K/N 为 4096/1536，B 为 4096/3072，两者 GMM2 都是 1536/4096。各 case 运行原生融合 A/B，另以 A 的模型运行 GELU Prefill 性能对照。

| Case | 模型或 BS | 变量和精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 | H 显存带宽 |
| --- | --- | --- | --- | --- | --- | --- |
| `L01` | U32 | A8W4 | [L01.json](../experiments/megamoe/scripts/configs/l0/L01.json) · [U](#u-原生融合) / [T](#t-通信打点) / [P](#p-prefill-性能级联) · 融合 A/B + A 级联 · [完整命令](../experiments/megamoe/scripts/ubx_overlap_quickstart.md#l01-完整验收) | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `L02` | U32 | A8W8 | [L02.json](../experiments/megamoe/scripts/configs/l0/L02.json) · [U](#u-原生融合) / [T](#t-通信打点) / [P](#p-prefill-性能级联) · 融合 A/B + A 级联 | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `L04` | P32 | A8W4 | [L04.json](../experiments/megamoe/scripts/configs/l0/L04.json) · [N](#n-直接启动融合) / [W](#w-配置文件入口) / [G](#g-950pod-a8w4-配对分窗) · 融合 A/B + A 级联 | A [融合实测](#950pod-本轮融合性能和显存占用)，本轮 B/级联/C 待测 · [历史对照](#历史分支对照) | 历史 A 抽样通过 · 本轮 A/B 全量 Q 待测，A 有 Torch 峰值 | A [HBM 观察](#正式性能轮内的-hbm-带宽)，B/级联待采 |
| `L05` | P32 | A8W8 | [L05.json](../experiments/megamoe/scripts/configs/l0/L05.json) · [N](#n-直接启动融合) / [W](#w-配置文件入口) · 融合 A/B + A 级联 | A [融合实测](#950pod-本轮融合性能和显存占用) · B/级联待测 · C 待 timer | 本轮 Q 待测 · A 有 Torch 峰值 | A [HBM 观察](#正式性能轮内的-hbm-带宽)，B/级联待采 |

### 大用例验收

A、B 的正式性能轮均处理每卡全部 39280 token。精度分别生成对应激活的 Golden。Q 填两臂的精度、计数和实际校验 token 数，完整记录各自窗口与显存。

验收比较同一后端、同一窗口和同一采样 rank 的总掩盖率。目标 A ≥ B，同时报告 A/B 的端到端耗时比及 Dispatch、Combine 各自百分点差。端到端按 A/B ≤ 1 检查。950POD A8W4 优先给出结论。

950POD 的 A8W4 使用 G 方法做同窗 A/B。A8W8 的 SwiGLU 在历史整批路径超限，新构建须复核 B 臂实际窗口与 A 相同。首窗百分比单列，E2E 与吞吐按整轮 token 数。

## 豆包用例 groupsize 256

全部使用 U32，H3584、I1280、E384、K8、S0。转测固定 GELU 提前加权、balanced、同一套物理卡。

使用 U / P / T 方法。先确认 groupsize 的生产配置键和生效路径，保存参数及实际值证据，然后只变更表内 BS、精度和 groupsize。固定两种实现的统计范围。

| Case | 每卡 BS | groupsize | 精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 | H 显存带宽 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `D01` | 1024 | 256 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `D02` | 1024 | 256 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `D04` | 2048 | 256 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `D05` | 2048 | 256 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `D07` | 4096 | 256 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `D08` | 4096 | 256 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `D10` | 8192 | 256 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `D11` | 8192 | 256 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |

### 需归档的分组证据

若 groupsize 指外层 token 窗口，记录实际窗长与尾窗，并先确认 512 能在所选 U32 路径生效。现有 stage_mte_windows.py 的 512 为单机 950 MTE 实验。其他含义由需求方给出具体参数。确认前状态填写待配置。

## 豆包用例 groupsize 512

全部使用 U32，H3584、I1280、E384、K8、S0。转测固定 GELU 提前加权、balanced、同一套物理卡。

使用 U / P / T 方法。先确认 groupsize 的生产配置键和生效路径，保存参数及实际值证据，然后只变更表内 BS、精度和 groupsize。固定两种实现的统计范围。

| Case | 每卡 BS | groupsize | 精度 | 执行方法和对照 | P 性能与 C 通信 | Q 精度与 M 显存 | H 显存带宽 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `D13` | 1024 | 512 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `D14` | 1024 | 512 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `D16` | 2048 | 512 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `D17` | 2048 | 512 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `D19` | 4096 | 512 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `D20` | 4096 | 512 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `D22` | 8192 | 512 | A8W4 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |
| `D23` | 8192 | 512 | A8W8 | U32 · [U](#u-原生融合) / [P](#p-prefill-性能级联) / [T](#t-通信打点) · 先确认 GS | P 待测 · C 待测 | Q 待测 · M 待分臂采 | 待采 |

### 需归档的分组证据

若 groupsize 指外层 token 窗口，记录实际窗长与尾窗，并先确认 512 能在所选 U32 路径生效。现有 stage_mte_windows.py 的 512 为单机 950 MTE 实验。其他含义由需求方给出具体参数。确认前状态填写待配置。
