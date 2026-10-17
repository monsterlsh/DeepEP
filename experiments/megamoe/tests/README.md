<!--
Copyright (c) 2026, Lu Lu
Modified by SimpleBright_Man 2026
-->

# MegaMoE 单机 EP8：W4A8 / W8A8 验收与性能基线

本方案用于 `0.1.0.dev6`，在 Linux、8 张空闲 Ascend950、CANN 9.2.0 和匹配的
Torch/torch_npu 环境中执行。CPU 测试验证参考、入口和参数协议；真实设备编译、精度、
通信同步与性能需按下列命令验收。W8A8 目前支持 S=0、FP8/BF16 输入、BF16 输出，
不支持共享专家和设备 timer。性能使用 profiler，不需要 `--timer`。

EP8 表示 8 个 rank；`num_experts` 是全局专家数，需可被 8 整除。`smoke` 配置为
H=1024、I=256、E=8、TopK=2，每卡一个专家。正式入口为包内 `benchmark` / `golden`。

## 1. 构建与安装检查

先将分支 `megamoe-a8w8-ep8` 同步到服务器，在该分支 checkout 中执行：

```bash
cd /path/to/DeepEP-gitcode
source /path/to/cann/set_env.sh
# 先激活已经安装匹配 torch/torch_npu 的环境。
MEGAMOE_PYTHON=$(command -v python)
export DEEPEP_BUILD_MEGAMOE=ON
PYTHON="$MEGAMOE_PYTHON" bash experiments/megamoe/scripts/build.sh
export LD_LIBRARY_PATH="${SHMEM_ROOT:-$PWD/third-party/shmem/install/shmem}/lib:${LD_LIBRARY_PATH:-}"
if [[ -n ${CONDA_PREFIX:-} ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:26733
export OMP_NUM_THREADS=1
RUN_DIR=$(mktemp -d /tmp/deepep-ep8.XXXXXX)
cd "$RUN_DIR"

env -u PYTHONPATH "$MEGAMOE_PYTHON" - <<'PY'
import json
from importlib.metadata import version
from importlib.resources import files
import torch
import torch_npu
import deep_ep
from deep_ep_experimental.megamoe import fp8_fp8_mega_moe
from deep_ep._native import require_native
print("version:", version("ascend-deepep"))
print("package:", deep_ep.__file__)
print("framework:", torch.__version__, torch_npu.__version__)
print("visible NPUs:", torch.npu.device_count())
info = json.loads(files("deep_ep").joinpath("_build_info.json").read_text())
assert info["megamoe"]["enabled"]
assert torch.npu.device_count() == 8
require_native("EP8 preflight")
PY
```

确认版本为 `0.1.0.dev6`、包路径为当前环境的 `site-packages`。SHMEM 端口需空闲。
从源码目录外启动，并移除本次命令的 `PYTHONPATH`，避免源码遮蔽 wheel。
以下 Bash 函数用同一解释器调用启动器的 `main()`，避免 `python -m torch.distributed.run`
在框架提前导入该模块时出现 runpy 重复导入警告。

```bash
run_ep8() {
  env -u PYTHONPATH "$MEGAMOE_PYTHON" -c 'from torch.distributed.run import main; main()' \
    --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
    --module deep_ep_experimental.megamoe.benchmark "$@"
}
```

## 2. 首轮 W8A8 全量精度

```bash
env -u PYTHONPATH "$MEGAMOE_PYTHON" -m deep_ep_experimental.megamoe.golden \
  --config smoke --mma-type fp8xfp8 --ep-size 8 --ranks 0-7 \
  --bs-values 96 --input fp8 --routing balanced \
  --golden-dir "$RUN_DIR/golden"

run_ep8 --config smoke --mma-type fp8xfp8 \
  --mode accuracy --bs-values 96 --input fp8 --routing balanced \
  --check-tokens 0 --golden-dir "$RUN_DIR/golden" \
  --output-dir "$RUN_DIR/w8_accuracy"
```

通过标准：进程退出码为 0，终端输出 `all 8 ranks passed`；
`w8_accuracy/rank_0/bs_96/accuracy.json` 内 8 个 rank 的 `passed`、`counts_equal`、
`cumulative_counts_equal` 都为 true，且 `mma_type` 为 `fp8xfp8`。
`--check-tokens 0` 检查全部 token。缓存区分权重类型、输入、路由、EP、BS、seed 和参考实现，
修改参考代码后需重新运行 Golden；不要复制 W4 缓存作为 W8 参考。

## 3. 边界、路由与 W4 回归

首轮通过后，在同一个 shell 中执行；相同进程内切换 BS 可覆盖计划重建和同步。

```bash
(
for mma in fp8xfp4 fp8xfp8; do
  for input in fp8 bf16; do
    for routing in balanced random skewed; do
      case_name="${mma}_${input}_${routing}"
      batches=1,17,96,255,256,257,512,4096
      env -u PYTHONPATH "$MEGAMOE_PYTHON" -m deep_ep_experimental.megamoe.golden \
        --config smoke --mma-type "$mma" --ep-size 8 --ranks 0-7 \
        --bs-values "$batches" --input "$input" --routing "$routing" \
        --golden-dir "$RUN_DIR/golden" || exit $?
      run_ep8 --config smoke --mma-type "$mma" --mode accuracy \
        --bs-values "$batches" --input "$input" --routing "$routing" \
        --check-tokens 0 --golden-dir "$RUN_DIR/golden" \
        --output-dir "$RUN_DIR/$case_name" || exit $?
    done
  done
done
)
```

每个 case 均应全 rank 通过。`random` 包含负路由权重，`skewed` 包含本地专家无 token 的 rank。
循环在子 shell 内遇错立即返回非零退出码，不关闭当前交互 shell；完整矩阵的 CPU Golden 可能耗时较长。
任一失败先保存该目录和终端日志，按原命令换新 `--output-dir` 复现，暂不继续性能结论。
此外，可将模型换为自定义 JSON，覆盖 E=16/64（每卡多个专家）、TopK=6、不同 H/I；
W8 保持 `num_shared_experts=0`。`deepseek_v4_pro` 在 EP8 下每卡 48 个专家，
应先确认 HBM/主机内存余量，再进行全模型 Golden 和设备测试。

## 4. 性能基线

精度通过后，同配置比较两种权重，并为后续 kernel 优化保留原始数据。
每个 BS 独立启动，避免容量差异影响不同 BS 的资源占用；每种配置独立重复 3 次。

```bash
(
for mma in fp8xfp4 fp8xfp8; do
  for bs in 96 4096; do
    for repeat in 1 2 3; do
      run_ep8 --config smoke --mma-type "$mma" --mode perf \
        --bs-values "$bs" --input fp8 --routing balanced \
        --output-dir "$RUN_DIR/perf_${mma}_bs${bs}_r${repeat}" || exit $?
    done
  done
done
)
```

默认 warmup=10、iters=20，统计后 10 轮；要增加轮次，复制配置 JSON 并统一修改
`warmup` / `iters`，随后用绝对路径传给 `--config`。W4/W8 使用各自确定性权重，
精度分别对拍同源 Golden；性能保持模型、输入、路由、软件版本和计时范围一致。
默认 rank 对齐前置张量占 12,000,000,000 字节，不计入 kernel Duration；
若需缩小，可在两种配置中使用完全一致的 `alignment`，并记录该变化。

```bash
"$MEGAMOE_PYTHON" - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys
for path in sorted(Path(sys.argv[1]).glob("perf_*/rank_0/bs_*/profiling.json")):
    r = json.loads(path.read_text())
    print(path.parents[2].name, r["mma_type"],
          "rank-min mean(us)=", r["mean_us"],
          "rank-max mean(us)=", r["rank_max_mean_us"],
          "rounds=", r["rounds"], "/", r["recorded_rounds"])
PY
```

`mean_us` 是逐轮最快 rank 的均值；`rank_max_mean_us` 是逐轮最慢 rank 的均值，
二者均来自融合 kernel 的 profiler Duration。保留每卡的 `kernel_samples.json`、
`profiler/`、`resources.json` 和 `manifest.json`，比较卡间差异和波动，
不能仅凭一次 smoke 的平均耗时声明全模型加速。

## 5. CPU 回归

在仓库目录执行：

```bash
python -m pip install -r experiments/megamoe/tests/requirements.txt
python -m pytest experiments/megamoe/tests -q
```

重点覆盖 W8 权重位模式/scale、Golden 缓存隔离与独立参考对拍、API 的两种输入与 BS 切换、
混合 dtype/错误布局拒绝、共享专家/timer 前置检查和启动参数透传。
CPU 通过不能替代上述 Ascend950 设备验收。
