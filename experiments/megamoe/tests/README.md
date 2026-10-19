<!--
Copyright (c) 2026, Lu Lu
Modified by SimpleBright_Man 2026
-->

# MegaMoE 单机 EP8：W4A8 / W8A8 验收与性能基线

Linux 服务器上的设备验收统一使用 [`run_ep8.sh`](../scripts/run_ep8.sh)，无需逐条生成
Golden、启动 8 个 rank 或汇总结果。本方案对应 `0.1.0.dev7+megamoe.a8w8.ep8`，需要 8 张空闲 Ascend950、
CANN 9.2.0 和匹配的 Torch/torch_npu。CPU 测试不能替代设备编译、精度、通信同步和性能验收。

W8A8 支持 S=0、FP8/BF16 输入、BF16 输出，不支持共享专家和设备 timer。
脚本使用 profiler 计时。EP8 表示 8 个 rank；全局专家数 E 必须可被 8 整除。

## 服务器执行

先同步 `megamoe-a8w8-ep8` 分支，激活已安装 Torch/torch_npu 的环境，并加载 CANN。
首次执行加 `--build`，构建并安装当前 checkout 的 wheel，然后自动进行 A8W8 首轮验收：

```bash
cd /path/to/DeepEP-gitcode
conda activate YOUR_ENV
source /path/to/cann/set_env.sh
bash experiments/megamoe/scripts/run_ep8.sh --build
```

首轮通过后执行完整矩阵；已安装当前版本时无需重新构建：

```bash
bash experiments/megamoe/scripts/run_ep8.sh --suite full
```

常用选项如下；可用 `--help` 查看完整帮助。

| 选项 | 用途 |
| --- | --- |
| `--build` | 先构建安装当前源码；默认不构建，安装版本必须与 checkout 的 `VERSION` 一致 |
| `--suite quick\|full` | 默认 `quick`；覆盖范围见下表 |
| `--skip-perf` | 只生成 Golden 和执行精度验收 |
| `--output-dir /absolute/new/path` | 指定源码目录外的全新结果目录；已有目录会被拒绝 |
| `--python /path/to/python` | 指定解释器；默认使用环境变量 `PYTHON`，否则使用当前 `python` |
| `--dry-run` | 只打印计划，不构建、不创建文件、不导入 Torch、不启动设备测试 |

例如，先预览完整精度计划，再将结果保存到一个新目录：

```bash
bash experiments/megamoe/scripts/run_ep8.sh --suite full --skip-perf --dry-run
bash experiments/megamoe/scripts/run_ep8.sh --suite full --skip-perf \
  --output-dir /data/megamoe/ep8-full-dev7-run1
```

脚本默认使用 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`、
`DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:26733` 和 `OMP_NUM_THREADS=1`，允许提前用环境变量覆盖；
可见设备必须恰好为 8 张，SHMEM 端口必须空闲。脚本补充 SHMEM 和当前 Conda 的动态库路径，
其中 SHMEM 默认目录为 `third-party/shmem/install/shmem`，可通过 `SHMEM_ROOT` 指定。

测试从源码目录外启动并清除子进程的 `PYTHONPATH`，避免源码遮蔽已安装包而缺少
`deep_ep/_build_info.json`。启动器直接调用 `torch.distributed.run.main()`，避开该模块
被框架提前导入时的 runpy 重复导入警告。构建、预检或任一用例失败都会返回非零退出码，
已生成的日志和报告会保留；修复后选择新的结果目录重新执行。

## 自动验收覆盖

`smoke` 配置为 H=1024、I=256、E=8、TopK=2，每卡一个专家。每个精度用例都先生成
匹配其权重类型、输入、路由和 BS 的独立 Golden，再启动 EP8 对拍；不会复用 W4 缓存代替 W8 参考。

| 套件 | 精度覆盖 | 性能覆盖 |
| --- | --- | --- |
| `quick` | W8，FP8/BF16 两种输入，balanced 路由，BS=96/120/121/128/129/257，全部 token | W8，BS=96/4096，各独立重复 3 次 |
| `full` | W4/W8，两种输入，balanced/random/skewed 三种路由，BS=1/17/96/119/120/121/127/128/129/255/256/257/512/4096，全部 token；另加下表的 W8 原生形状边界 | W4/W8，BS=96/4096，每种权重和 BS 各独立重复 3 次 |

性能用例统一使用 FP8 输入和 balanced 路由，且只在精度阶段全部成功后启动。
`--skip-perf` 可跳过性能阶段。`random` 覆盖负路由权重，`skewed` 覆盖本地专家无 token 的 rank。
同一用例进程内切换 BS，覆盖计划重建和同步。smoke/balanced 每专家接收 `2*BS` 个 token；
BS120/121 对应 M240/M242，BS127/128/129 对应 M254/M256/M258，覆盖 A8W8 GMM1
128 行 UB 子块的边界与跨子块同步。

完整套件额外生成以下配置，均使用 W8、两种输入、balanced 路由和 S=0：

| 用例 | H / I / E / TopK | BS | 精度采样与目的 |
| --- | --- | --- | --- |
| `tail` | 1088 / 384 / 16 / 6 | 17/121/257 | 全部 token；GMM1 K64 尾、N128 尾及 GMM2 K128 尾 |
| `wide` | 8192 / 4096 / 8 / 1 | 17/257 | 全部 token；H/I 宽度上界 |
| `topk32` | 1024 / 256 / 64 / 32 | 17/129 | 每 rank 抽查 17 个 token；TopK 上界 |
| `route_ub` | 8192 / 256 / 32 / 1 | 16384 | 每 rank 抽查 17 个 token；compact route 的 UB 预算收缩 |

所有精度用例均核对全部专家接收计数，抽样只限制输出精度对拍的 token 数。
完整矩阵的 CPU Golden 可能耗时较长；`wide`、`route_ub` 的内存需求高于 smoke，
不要同时运行多组设备测试。

## 查看结果

未指定 `--output-dir` 时，脚本创建 `/tmp/deepep-ep8.XXXXXX` 并打印实际路径。
成功时打印 `EP8 validation passed`；将下方路径替换为本次目录：

```bash
cat /tmp/deepep-ep8.XXXXXX/summary.txt
```

| 文件 | 内容 |
| --- | --- |
| `summary.txt`、`summary.json` | 精度通过情况、性能数据和预期报告数量核对 |
| `logs/*.log` | 构建、环境预检、Golden、各测试用例与汇总的完整输出 |
| `accuracy_*/rank_0/bs_*/accuracy.json` | 8 个 rank 的精度及专家计数报告 |
| `perf_*/rank_0/bs_*/profiling.json` | 对应 BS 和重复轮次的性能统计 |
| `golden/` | 本次精度参考缓存 |

通过标准是脚本退出码为 0、`summary.json` 的 `passed` 为 true，且全部预期报告存在。
每个 `accuracy.json` 的 8 个 rank 均须满足 `passed`、`counts_equal`、
`cumulative_counts_equal` 为 true；终端也应显示 `all 8 ranks passed`。
quick 共 12 份精度报告和 6 份性能报告；full 共 184 份精度报告和 12 份性能报告。
指定 `--skip-perf` 时，性能报告数量应为 0。

性能单位为微秒：`mean_us` 是逐轮最快 rank 的均值，`rank_max_mean_us` 是逐轮最慢 rank
的均值，均取自融合 kernel 的 profiler Duration。默认 warmup=10、iters=20，统计后 10 轮。
保留各卡的 `kernel_samples.json`、`profiler/`、`resources.json` 和 `manifest.json`，
用于分析卡间差异和波动。W4/W8 的性能比较应保持模型、输入、路由、软件版本和计时范围一致；
一次 smoke 耗时不能代表全模型加速。

默认 rank 对齐前置张量占 12,000,000,000 字节，不计入 kernel Duration。
关闭设备 timer 时，报告中的 `timer_pass: false` 属于预期行为。

## 尚未自动化的原生格式负例

上述 Bash 套件验证合法输入的完整执行路径，尚未自动化直接调用 `Plan.forward` 的错误物理布局测试。
该项需要已准备好的 EP8 用例：所有 rank 同时将 BF16 输入或输出转换为
`torch_npu.npu_format_cast(tensor, 29)`，直接调用 `buffer._plan.forward(...)`，
应在 kernel launch 前报 `requires plain ND/NCHW storage`，随后恢复原张量再次验证精度。
只调用 Python 权重检查不能替代这项 C++ 边界检查。

原生约束为 W8 权重 ND2、W4 权重 NZ29/50，其余张量为线性 ND2/NCHW0；
4D scale 的默认 NCHW0 和 ND2 均应通过。CPU 负例覆盖了部分拒绝逻辑，
但不代表此项真实设备验收已经完成。

## CPU 回归

CPU 回归独立于设备脚本，在仓库目录执行：

```bash
python -m pip install -r experiments/megamoe/tests/requirements.txt
python -m pytest experiments/megamoe/tests -q
```

覆盖 W8 权重位模式/scale、Golden 缓存隔离与参考对拍、API 两种输入与 BS 切换、
混合 dtype/错误布局拒绝、共享专家/timer 前置检查和启动参数透传。
`test_native_policies.py` 使用 C++17 编译器编译真实生产头，验证发送 UB 预算、
MX scale 缓存策略及 GMM1 子块约束；可用 `CXX=/path/to/g++` 指定编译器。
缺少编译器时该项会 skip，验收时须确认其实际通过。
