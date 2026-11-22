# UBX EP32 GELU 临时操作说明

本例在 UBX 双机各 16 卡上测 GELU A8W4 性能，后端为 `mte_roce`。使用每卡 39280 token、全局 800 个路由专家、TopK 25。GMM1 K/N 为 4096/1536，GMM2 为 1536/4096。

| 节点 | IPv4 | 手动启动时的 `node_rank` |
| --- | --- | ---: |
| 主节点 | `90.90.93.84` | 0 |
| 第二台 | `90.90.93.184` | 1 |

仓库目录为 `/mnt/share/l00911701/DeepEP-UBX`。两台之间需要互通 torchrun TCP 29500、SHMEM TCP 29180 和 NPU RoCE 数据面。

## 1. 离线编译并安装

将同一版 DeepEP checkout 拷到两台。提前准备匹配的 CANN、`py311_tpr` Conda 环境和构建依赖，以及 [依赖锁文件](../../../dependencies.lock.json) 指定的 CATLASS 和 HNS_1825 SHMEM SDK。`SHMEM_ROOT` 指向包含 `deepep-sdk.json` 的 SDK 安装目录。UBX 默认加载 `/usr/local/Ascend/cann-9.2.T101/set_env.sh`，路径不同时先设置 `UBX_CANN_SETUP`。

在 **90.90.93.84** 编译一次。把下面两个路径换成本机已经准备好的离线依赖。

```bash
cd /mnt/share/l00911701/DeepEP-UBX
export CATLASS_ROOT=/path/to/catlass
export SHMEM_ROOT=/path/to/shmem/install/shmem
PIP_NO_INDEX=1 bash experiments/megamoe/scripts/build_ubx.sh
```

脚本会在本机安装 wheel，打印 `[BUILD] Installed:` 对应的 `dist/build-*/` 下的 wheel 路径及 SHA-256。在 **90.90.93.184** 安装同一份 wheel。第二台能从共享目录读取 wheel 时，直接填构建输出的路径。使用独立磁盘时，先拷入 wheel 再安装。

```bash
cd /mnt/share/l00911701/DeepEP-UBX
source experiments/megamoe/scripts/ubx_env.sh
WHEEL=/path/to/ascend_deepep-实际文件名.whl
sha256sum "$WHEEL"
"$PYTHON" -m pip install --no-index --no-deps --force-reinstall "$WHEEL"
```

## 2. 生成 JSON 并启动

### 方案 A：用 `node_ip` 从主节点自动启动

适用于 **90.90.93.84** 可以免交互 SSH 到 **90.90.93.184** 的环境。两台使用同一份 wheel、相同绝对路径的 checkout 和 Python。SSH 登录名或端口有差异时，在运行 JSON 外层填写 `ssh_user` 或 `ssh_port`。

启动器会在各自节点加载本机的 CANN 和 Python 运行库环境。默认的 CANN 路径不同于预设时，可在 JSON 外层填写两台都可读的 `cann_env`。

以下命令只在 **90.90.93.84** 执行。脚本会通过 SSH 拉起第二台。JSON 使用 `node_ip` 按顺序指定主、从节点，默认取第一项为 `master_addr`。如果沿用手动启动的 JSON，改用自动启动前删除其中的 `node_rank` 和 `stage`。

```bash
cd /mnt/share/l00911701/DeepEP-UBX
source experiments/megamoe/scripts/ubx_env.sh

OUT=/tmp/megamoe/ep32-gelu-a8w4-01
RUN_CONFIG="$OUT/gelu-perf-auto.json"
mkdir -p -- "$OUT"

"$PYTHON" - "$OUT" "$RUN_CONFIG" <<'PY'
import json
from pathlib import Path
import sys

output_dir, run_config = sys.argv[1:]
preset = Path("experiments/megamoe/scripts/configs/ubx_ep32_target.json")
settings = json.loads(preset.read_text(encoding="utf-8"))
settings.update(
    mode="perf",
    precision="a8w4",
    node_ip=["90.90.93.84", "90.90.93.184"],
    output_dir=output_dir,
    timer=False,
)
Path(run_config).write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

"$PYTHON" experiments/megamoe/scripts/run_megamoe.py --run-config "$RUN_CONFIG"
```

每次重跑换一个新的 `OUT`。在主节点的 `$OUT` 下可以查看两台的调度日志 `launch-run-perf-w4a8-node0.log` 和 `launch-run-perf-w4a8-node1.log`。

### 方案 B：在两台节点分别手动启动

适用于 SSH 自动拉起不方便的环境。若第二台的 checkout 在其他位置，将该节点命令里的 `cd` 换成实际路径。

以下命令两台都执行。第一台设置 `NODE_RANK=0`，第二台设置 `NODE_RANK=1`。两台使用相同的 `OUT` 绝对路径字符串，可以各自写本机磁盘。每次重跑换一个新的 `OUT`。

```bash
cd /mnt/share/l00911701/DeepEP-UBX
source experiments/megamoe/scripts/ubx_env.sh

NODE_RANK=0  # 90.90.93.184 上改成 1
OUT=/tmp/megamoe/ep32-gelu-a8w4-01
RUN_CONFIG="$OUT/gelu-perf-node${NODE_RANK}.json"
mkdir -p -- "$OUT"

"$PYTHON" - "$NODE_RANK" "$OUT" "$RUN_CONFIG" <<'PY'
import json
from pathlib import Path
import sys

node_rank, output_dir, run_config = sys.argv[1:]
preset = Path("experiments/megamoe/scripts/configs/ubx_ep32_target.json")
settings = json.loads(preset.read_text(encoding="utf-8"))
settings.pop("node_ip", None)
settings.update(
    mode="perf",
    precision="a8w4",
    stage="run",
    node_rank=int(node_rank),
    master_addr="90.90.93.84",
    master_port=29500,
    shmem_port=29180,
    output_dir=output_dir,
    timer=False,
)
Path(run_config).write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

"$PYTHON" experiments/megamoe/scripts/run_megamoe.py --run-config "$RUN_CONFIG"
```

生成的配置保留预设的 `backend=mte_roce`、`ep_size=32`、`config.model.activation=gelu` 和模型参数。手动模式通过 `node_rank` 区分两台，生成配置时会移除用于 SSH 自动启动的 `node_ip`。性能测试直接跑 `stage=run`。

要换量化精度，修改运行 JSON 外层的 `precision`，可选 `a8w4`、`a8w8`、`a4w4`。要调整每卡 token 数，修改 `config.bs_values`。手动启动时，两台的负载和精度使用相同的值。

## 3. 查看性能结果

使用示例 `OUT` 时，先看每台第一个 rank 的汇总。

| 节点 | 性能 JSON | 启动日志 |
| --- | --- | --- |
| `90.90.93.84` | `$OUT/perf-w4a8/rank_0/bs_39280/profiling.json` | `$OUT/perf-w4a8-node0.log` |
| `90.90.93.184` | `$OUT/perf-w4a8/rank_16/bs_39280/profiling.json` | `$OUT/perf-w4a8-node1.log` |

- `host_e2e.rank_max_mean_us` 是逐轮取最慢 rank 后求均值的同步端到端时延，单位 µs。
- `host_e2e.global_input_tokens_per_second` 是全组输入吞吐。
- 顶层 `rank_max_mean_us` 是每次 forward 的 kernel 耗时汇总。各 rank 的 `manifest.json` 可用于核对后端、EP 和模型参数。

需要首窗设备 trace 时，使用新输出目录并在运行 JSON 设置 `timer: true`。手动启动时两台都设置。正式性能数据先采集，随后单独采集诊断数据，文件写入相应采样 rank 的 `bs_39280/roce_trace.json`。[timer 说明](../README.md#ubx-设备-timer-与通信掩盖率)

## 4. 跑 GELU A8W4 精度

以第 2 节生成的 JSON 为模板，复制一份精度配置，并在外层修改以下字段。模型仍使用 EP32 GELU、每卡 39280 token。示例 `check_tokens: 8` 表示各 rank 抽查 8 个 token 的 CPU Golden，适合先验证建链和精度流程。全量精度将其设为 `0`，并换一个新的输出目录。

```json
{
  "mode": "accuracy",
  "precision": "a8w4",
  "check_tokens": 8,
  "dual_reference": false,
  "timer": false,
  "output_dir": "/tmp/megamoe/ep32-gelu-a8w4-accuracy-01"
}
```

- **方案 A 自动启动**。保留 `node_ip`，使用第 2 节方案 A 生成的配置结构，将新文件保存为仓库根目录下的 `gelu-accuracy.json`。只在 `90.90.93.84` 执行下方命令。启动器先让两台各自生成本机 rank 的 CPU Golden，再拉起 32 个 rank 跑 NPU 精度。
- **方案 B 手动启动**。两台分别从第 2 节方案 B 的本机配置复制出 `gelu-accuracy.json`，保留各自的 `node_rank`，使用相同的模型、`check_tokens`、`master_addr` 和 `output_dir`。先将两台的 `stage` 都设为 `prepare`，分别执行下方命令。等两台都完成后，将同一份 JSON 的 `stage` 改为 `run`，再在两台执行一次。两阶段复用同一 `golden_dir`，默认使用仓库下的 `benchmark-golden`。

```bash
cd /mnt/share/l00911701/DeepEP-UBX
source experiments/megamoe/scripts/ubx_env.sh
"$PYTHON" experiments/megamoe/scripts/run_megamoe.py --run-config ./gelu-accuracy.json
```

## 5. 查看精度结果

精度通过后日志会打印 `[ACCURACY] 全组报告` 路径。以上面的 `output_dir` 为例，在 `90.90.93.84` 查看 `accuracy-w4a8/rank_0/bs_39280/accuracy.json`，在 `90.90.93.184` 查看 `accuracy-w4a8/rank_16/bs_39280/accuracy.json`。每份报告都是包含全组 32 个 rank 的 JSON 数组，误差检查失败时也会保留报告。

| 字段 | 怎么看 |
| --- | --- |
| `passed` | 每个 rank 都为 `true` 时精度通过。 |
| `checked_token_count` / `total_tokens` | CPU Golden 的实际比对行数与本 rank 的总 token 数。示例为 8 / 39280，全量比对为 39280 / 39280。 |
| `finite` | 全量输出都是有限值。 |
| `counts_equal` / `cumulative_counts_equal` | 逐专家计数与 Golden 一致，累加后的计数也正确。 |
| `bad_elements`、`max_abs`、`relative_rmse` | 用于定位超阈值元素和误差大小，最终结论看 `passed`。 |

在 `90.90.93.84` 可以用下面的命令检查是否收齐 32 个 rank，并列出失败项。

```bash
OUT=/tmp/megamoe/ep32-gelu-a8w4-accuracy-01
"$PYTHON" - "$OUT/accuracy-w4a8/rank_0/bs_39280/accuracy.json" <<'PY'
import json
from pathlib import Path
import sys

rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
failed = [row for row in rows if row["passed"] is not True]
complete = sorted(row["rank"] for row in rows) == list(range(32))
print(f"通过 {len(rows) - len(failed)}/{len(rows)} 个 rank，32 个 rank 齐全={complete}")
for row in failed:
    print({key: row.get(key) for key in (
        "rank", "checked_token_count", "finite", "counts_equal", "cumulative_counts_equal",
        "bad_elements", "max_abs", "relative_rmse",
    )})
raise SystemExit(0 if complete and not failed else 1)
PY
```

如果没有生成报告，先看 `$OUT/golden-w4a8-node0.log`、`golden-w4a8-node1.log` 里的 Golden 阶段，再查看 `accuracy-w4a8-node0.log`、`accuracy-w4a8-node1.log` 里的 NPU 错误。自动启动时，主节点的 `$OUT` 下还有两台的 `launch-prepare-accuracy-*` 和 `launch-run-accuracy-*` 调度日志。
