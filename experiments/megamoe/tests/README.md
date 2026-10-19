<!--
Copyright (c) 2026, Lu Lu
Modified by SimpleBright_Man 2026
-->

# MegaMoE 单机 EP8：A8W4 / A8W8 / A4W8 验证

CPU 和 NPU 验证的完整说明、环境依赖和报告格式统一维护在
[`validation/a8w8`](../../../validation/a8w8/README.md)。原 `run_ep8.sh` 保留为公共 NPU runner 的 EP8 入口，
旧参数仍可使用，默认 quick 现在覆盖三种模式。

版本以当前分支根目录 `VERSION` 为准：fix 分支为 `0.1.0.dev5+fix.megamoe.a8w8`，EP8 分支为 `0.1.0.dev5+megamoe.a8w8.ep8`。需要 Linux、8 张空闲 Ascend950、
CANN 9.2.0 和匹配的 Torch/torch_npu。实际设备编译、跨卡精度与性能须在服务器验收。

```bash
cd /path/to/DeepEP
conda activate YOUR_NPU_ENV
source /path/to/cann/set_env.sh

# 三模式：BF16 和对应的预量化输入；构建后执行精度与性能测试。
bash experiments/megamoe/scripts/run_ep8.sh --build --suite quick

# 先预览 A4W8 计划，再执行。
bash experiments/megamoe/scripts/run_ep8.sh --mma-type fp4xfp8 --dry-run
bash experiments/megamoe/scripts/run_ep8.sh --mma-type fp4xfp8 --suite full --skip-perf

# 新建的结果目录必须位于源码目录之外。
bash experiments/megamoe/scripts/run_ep8.sh --suite full \
  --output-dir /data/megamoe/ep8-full-dev5-run1
```

| 模式 | 激活输入 | 权重 | 中间激活 | 限制 |
| --- | --- | --- | --- | --- |
| A8W4 `fp8xfp4` | BF16 / FP8 E4M3 | FP4 E2M1 | MXFP8 | 公共验收使用 S=0、timer OFF |
| A8W8 `fp8xfp8` | BF16 / FP8 E4M3 | FP8 E4M3 | MXFP8 | 不支持共享专家或设备 timer |
| A4W8 `fp4xfp8` | BF16 / packed FP4 E2M1 | FP8 E4M3 | MXFP4 | 不支持共享专家或设备 timer |

A4W8 在输入和 SwiGLU 后均按 FP4 精度量化，再无损展开为 FP8 存储、通信和计算；
没有原生混合位宽 MMAD 或 FP4 通信压缩的性能承诺。预量化输入的 Python 展开耗时
不包含在融合 kernel 的 profiler Duration 中。A4W8 的 `full_call_profiling.json` 和报告中的 `full_call_device` 用 NPU events 覆盖完整设备调用，packed 输入时包含展开；它不计入 Python 主机端提交耗时。
所有模式输出 BF16，路由权重在 GMM2 后应用。

默认 quick 应有 36 份精度报告、18 份性能报告；full 应有 300 和 18。
仅选择一种 `--mma-type` 时，quick 为 12/6，full 为 100/6；`--skip-perf` 时性能数量为 0。
检查退出码为 0、`summary.json` 的 `passed=true` 和所有 rank 的精度/专家计数。
CPU Golden 及测试不能替代 NPU 验证，也不能证明两个分支在不同随机权重下逐元素相同。
