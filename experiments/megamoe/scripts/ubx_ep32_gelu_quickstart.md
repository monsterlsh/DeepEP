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

## 3. 查看结果

使用示例 `OUT` 时，先看每台第一个 rank 的汇总。

| 节点 | 性能 JSON | 启动日志 |
| --- | --- | --- |
| `90.90.93.84` | `$OUT/perf-w4a8/rank_0/bs_39280/profiling.json` | `$OUT/perf-w4a8-node0.log` |
| `90.90.93.184` | `$OUT/perf-w4a8/rank_16/bs_39280/profiling.json` | `$OUT/perf-w4a8-node1.log` |

- `host_e2e.rank_max_mean_us` 是逐轮取最慢 rank 后求均值的同步端到端时延，单位 µs。
- `host_e2e.global_input_tokens_per_second` 是全组输入吞吐。
- 顶层 `rank_max_mean_us` 是每次 forward 的 kernel 耗时汇总。各 rank 的 `manifest.json` 可用于核对后端、EP 和模型参数。

需要首窗设备 trace 时，使用新输出目录并在运行 JSON 设置 `timer: true`。手动启动时两台都设置。正式性能数据先采集，随后单独采集诊断数据，文件写入相应采样 rank 的 `bs_39280/roce_trace.json`。[timer 说明](../README.md#ubx-设备-timer-与通信掩盖率)
