# MegaMoE GELU 与原 SwiGLU 的性能对照

脚本可直接配合当前分支协议 13 的已安装 wheel 使用。默认 GELU 在激活后乘路由权重，SwiGLU 在 Finalize 乘权重。两侧使用相同的 EP、每卡 B、E、K、H 和激活后宽度 I。输入 FP8 位码、E8M0 scales、路由 ID、路由权重及 GMM2 权重一致。GELU 使用 SwiGLU 的 gate 投影，因此 GMM1 的 N 分别为 I 和 2I。

## 非 UBX 950

在已加载 CANN 和 wheel 的空闲机器上，从仓库根目录执行。以下示例分别测 B257 的整批调用与 B39280 的实际分窗调用。所有 rank 的 CPU/NPU 放置和环境变量应相同。

```bash
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:29180
torchrun --standalone --nproc-per-node=8 experiments/megamoe/scripts/paired_benchmark.py \
  --config gelu_smoke_k25 --bs 257 --routing balanced --pairs 8 --warmup 2 \
  --profile --require-no-regression --output-dir runs/paired-b257

torchrun --standalone --nproc-per-node=8 experiments/megamoe/scripts/paired_benchmark.py \
  --config target_gelu_k25 --bs 39280 --routing balanced --pairs 8 --warmup 2 \
  --require-no-regression --output-dir runs/paired-b39280-ep8
```

EP32 目标配置需要四台八卡 950。在每台节点设置相同的 `MASTER_ADDR` 为主节点可达 IPv4，按 0 至 3 分配 `NODE_RANK`，每次运行选择新的输出目录和空闲端口。

```bash
export DEEPEP_SHMEM_ENDPOINT="tcp://${MASTER_ADDR}:29180"
torchrun --nnodes=4 --node-rank="$NODE_RANK" --nproc-per-node=8 \
  --master-addr="$MASTER_ADDR" --master-port=29181 \
  experiments/megamoe/scripts/paired_benchmark.py --config target_gelu_k25 \
  --bs 39280 --routing balanced --pairs 8 --warmup 2 --require-no-regression \
  --output-dir runs/paired-b39280-ep32
```

每种激活均有自己的对称内存区域，两个区域由同一个 SHMEM 会话分配。两侧能原生整批执行时，按各自的公共接口调用。原 SwiGLU 的整批对称区超过 1 GiB 时，适配器在相同输入上按 GELU 的实际窗口调用原算子，满窗和尾窗各自复用一个 Plan。B39280 在 W256 下为 153 个满窗和 112 行尾窗，在 W2048 下为 19 个满窗和 368 行尾窗。适配器计入内核启动、专家计数累加和流同步；SwiGLU 的输入切片在计时前建立。GELU 的公开接口检查均计入端到端时间。

### W512 至 W2048 实验

`stage_mte_windows.py` 用同一个已安装 wheel 运行单机实验。每组分别设置相同的外层 W，然后调用该 wheel 原有的 GELU 与 SwiGLU 路径。实验在基准进程构造 buffer 时调整每个对称区的容量，按实际窗长创建 Plan。GELU 沿用公共调用；SwiGLU 在 W>256 时使用原始 Prefill 布局，按尾窗的实际长度选择布局。B39280 在 W512、W1024、W2048 下分别需要 77、39、20 次 launch，最后一窗均为 368 行。

```bash
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:29180
python experiments/megamoe/scripts/stage_mte_windows.py --nproc-per-node 8 \
  --windows 256 1024 2048 --output-dir runs/950-outer-windows
```

默认对 balanced 与 skewed 两个路由各运行全部 W，逐组串行启动 torchrun。每组仍有 8 个交替配对和 2 个预热配对。`--plan` 仅打印所有命令；分批手动调度后可用相同参数加 `--summary-only` 汇总已完成的作业。结果 `window_study.json` 保留每个相同 W 的两侧延迟，并按路由给出两个激活各自最快的 W。跨 W 各自最优比值适合选型，同 W 的重复配对数据用于判断该具体配置的性能。实验结果需在选定生产策略后用真实公共接口复测。

下表以 EP32、E800、H4096、I1536、K25、A8W4、每卡 B39280 为例。取 28 AIC、56 AIV、248 KiB UB，用 host `MakeTiling`、`WorkspaceInfo` 及 `CalcPeermemLeastSize` 计算；GELU 每个 wave 按 4 轮 GMM1 tile 预算。单 Plan 已含 50 MiB 固定预留；每份 Plan 还要加上本机 CANN `GetLibApiWorkSpaceSize()`。尾窗使用独立 Plan，在该配置与对应长窗占用相等。

| 外层 W | GELU 对称区 MiB | SwiGLU 对称区 MiB | GELU 单 Plan MiB | SwiGLU 单 Plan MiB |
|---:|---:|---:|---:|---:|
| 256 | 54 | 54 | 292.108 | 127.682 |
| 512 | 106 | 104 | 296.063 | 205.231 |
| 1024 | 208 | 208 | 303.974 | 212.555 |
| 2048 | 416 | 416 | 319.794 | 227.204 |

W2048 的 A/B 对称区合计为 832 MiB。单独执行 GELU 时，全量输入与路由 165.724 MiB、输出 306.875 MiB、准备好的模型权重 159.375 MiB。合并两份 Plan 与单个对称区，单卡已知常驻量约 1687.562 MiB，另需 CANN 系统工作区、路由排序临时张量、运行时分配及框架缓存。`paired.json` 的 `layout.prepared_plan_bytes` 给出按实际设备准备的两侧 scratch，`resources_by_rank` 记录独立于计时区间的 Torch 分配峰值和设备剩余内存。

## UBX 共同支持的配置

原 UBX SwiGLU 支持机内 EP8/16、K≤8、每卡 B≤16384。以下示例在一台 UBX 上使用 E32、K8、B257。设置 `EP=8` 或 `16`。两个公共接口各自在独立的进程中初始化其实际通信后端。脚本对两次独立进程的输入和权重使用同一生成规则。每一对使用相邻独占作业，逐对交替先后顺序。每个作业预热两次，采三次同步端到端延迟。结果只适用于测量的拓扑和配置。UBX EP32/K25 没有原 SwiGLU 同配置基线。

```bash
set -e
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:29180
EP=16
ROOT="runs/ubx-isolated-ep$EP-k8-b257"
for pair in {0..7}; do
  arms=(gelu swiglu)
  if (( pair % 2 )); then arms=(swiglu gelu); fi
  for arm in "${arms[@]}"; do
    torchrun --standalone --nproc-per-node="$EP" experiments/megamoe/scripts/paired_benchmark.py \
      --config gelu_smoke_k25 --topk 8 --bs 257 --isolated-arm "$arm" --routing balanced \
      --pairs 3 --warmup 2 --output-dir "$ROOT/$pair/$arm"
  done
done
python experiments/megamoe/scripts/summarize_isolated_pairs.py "$ROOT" --pairs 8 --require-no-regression
```

950 的 K32、H4096、W2048 下每个实现单独需要 528 MiB 对称区。两份合计超过单会话的 1 GiB 上限，可以交替启动独立进程作业。以下命令每侧各运行 8 次，保存每个作业的原始轮次、所用窗口及汇总时延比。把 `balanced` 改为 `skewed` 测热点，使用新输出目录。

```bash
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:29180
bash experiments/megamoe/scripts/run_isolated_mte.sh runs/950-k32-w2048-balanced balanced 2048
```

## 指标与判定

- Rank 0 的 `paired.json` 保留每个正式配对的执行顺序、全部 rank 的单次耗时、最慢 rank 耗时、GELU/SwiGLU 比值及分执行顺序的中位数。端到端计时从调用开始至设备同步完成。每次测量前同组同步。默认使用自然稳态缓存，`--cold-l2` 在每一侧独立清缓存，清缓存时间在计时区间外。
- 主判据使用每对两个最慢 rank 延迟的比值，再取所有配对的中位数。小于等于 1 表示本次样本的 GELU 无劣化。`--require-no-regression` 在报告落盘后对该判据返回退出码，至少需要 7 对样本且各侧预热 2 次。全组吞吐按 `EP × B / 各轮最慢 rank 平均秒数` 计算。可分别比较 balanced、skewed 或 random 路由，每种路由单独出报告。
- `--profile` 在无 profiler 的端到端测量之后重复同样顺序的配对。每次逻辑调用聚合所有窗口的 MegaMoE kernel duration，输出 `kernel_sum` 及每个 rank 的 `kernel_samples.json`。内核合计不含主机开销、窗口间空隙和独立的计数累加 kernel。
- 独立进程模式的 `isolated_pairs.json` 先逐轮取所有 rank 最慢值，再取每个作业内的中位数，最后以同一对相邻作业计算 GELU/SwiGLU。汇总时校验先后顺序、模型、路由、软硬件和每个 rank 的设备。保留原始轮次及作业路径。
- `--swiglu-weight-mode 1` 可单独作提前加权控制实验。默认后置加权的原 SwiGLU 构成用户端到端比较的基线。
