# A8W8 旧版 GMM1 UB 容量诊断

本分支从 `b6ab7636679e9dd87dbb1da66464a33a024b38e1` 创建，保留该提交的设备计算和
GMM1 M256 布局。新增验证脚本、默认关闭的检测编译选项及独立版本
`0.1.0.dev5+ub.probe.b6ab763`。当前其他分支可能已包含 M128 修复，不应只凭分支名判定实现。

目标环境为 Linux、单机 8 张 Ascend950、仓库锁定的 CANN 9.2.0、匹配的 Torch/torch_npu、
Python 3.11/3.12、主机 C++17 编译器。依赖准备参见同目录 `README.md`。
这里不自动 push，也不自动切换、回退任何已有分支。

## 检查内容及证据范围

固定 smoke 模型：H1024、I256、E8、TopK2、S0，EP8，balanced 路由。
每专家接收 `2*BS` 行。N256、BF16 中间结果每行占 512 字节，旧 up 缓冲区起点为
128 KiB。两份输入的地址范围是本工具的分析对象；这不是整个算子的动态 UB 占用统计。

| 每 rank BS | 每专家行数 | M256 逻辑块数 | 旧布局最大结束位置（不包含） | 按 248 KiB 的超限量 |
| --- | --- | --- | --- | --- |
| 64 | 128 | 1 | 192 KiB | 0 |
| 120 | 240 | 1 | 248 KiB | 0 |
| 121 | 242 | 1 | 249 KiB | 1 KiB |
| 128 | 256 | 1 | 256 KiB | 8 KiB |
| 4096 | 8192 | 32 | 256 KiB | 8 KiB |

BS4096 增加块数和复用次数，单个块的 UB 峰值不因此扩大 32 倍。上表仅为地址计算；
不能据此声称已经在真机复现了错误，也不能据此判定其他 UB 缓冲区安全。

设备运行会编译独立的 `ub_platform.cpp`，通过与 MegaMoE 相同的
`PlatformAscendCManager::GetCoreMemSize(UB)` 获取目标容量。查询失败就报告未完成，
不会悄悄套用 248 KiB。源码路径无法识别时同样拒绝猜测布局。

## 先进行 CPU 地址分析

在本分支仓库目录执行，只需 Python 标准库，不导入 Torch、不访问 NPU、不编译算子：

```bash
PYTHON=python bash validation/a8w8/run_ub_probe.sh --static-only \
  --output-dir /data/megamoe-results/ub-static-01
```

默认 `--ub-bytes 253952` 是明确标注的假设容量；只对 `--static-only` 有效。
默认五个 BS 会预测超限，**退出码 1 是预期诊断结果**，不是脚本运行故障。
报告中的 `device_ub_illegal_access_reported=false` 表示未收集到设备报告。

## 真机精度、边界和性能

先激活独立的 NPU Python 环境、加载 CANN。外部 `SHMEM_ROOT`/`CATLASS_ROOT` 应为绝对路径。
所有结果目录必须位于源码仓库外且尚不存在，重复测试换新目录。

```bash
# 只打印计划，不创建文件，不编译、不访问设备
PYTHON=python bash validation/a8w8/run_ub_probe.sh --build --sanitize --perf --dry-run

# 推荐首轮：普通编译，分别验证五个 BS 的 FP8/BF16 输入及所有 rank/token
PYTHON=python bash validation/a8w8/run_ub_probe.sh --build \
  --output-dir /data/megamoe-results/ub-accuracy-01

# 用户指定的大 BS；精度通过的用例才进行普通 wheel 性能测试，每项独立启动三次
PYTHON=python bash validation/a8w8/run_ub_probe.sh --build \
  --bs-values 128,4096 --inputs fp8 --perf --repeats 3 \
  --output-dir /data/megamoe-results/ub-large-perf-01
```

`PYTHON` 决定整个脚本及构建使用的解释器。默认 `python3`。
`--repo /absolute/checkout` 可指定另一份待测源码；运行前会检查安装版本、源码指纹、
timer OFF 和插桩标记，避免测到环境里的旧包。脚本从报告目录启动已安装的包。
默认可见设备 `0,1,2,3,4,5,6,7`；可设 `ASCEND_RT_VISIBLE_DEVICES`，但必须恰好 8 卡。
默认 SHMEM 地址 `tcp://127.0.0.1:26733`，可设 `DEEPEP_SHMEM_ENDPOINT` 使用空闲端口。

普通精度和检测精度使用 warmup=0，两次实际调用用于验证非零初值累计计数；不申请性能
对齐张量。性能使用 warmup=10、iters=20，统计后 10 次，默认对齐张量每卡额外约 12 GB，
其前置计算不计入 kernel Duration。关注 `rank_max_mean_us`；未通过精度的性能标为未执行。
容量计算超限的配置即使精度通过，性能也标记为 `diagnostic_only_due_to_capacity_violation`，
只供诊断，不作为正确实现的性能基线。

每个 BS/输入独立启动，异常和超时保留日志。脚本只终止本次启动的进程组。
设备执行失败后用新的进程检查 8 卡基本操作；检查失败则停止后续设备启动，避免连续失败。
`--timeout` 默认每次 Golden/设备启动 1800 秒，`--build-timeout` 默认每次构建 7200 秒。
不会重置卡、终止其他任务或自动修复设备。

## msSanitizer 检测

使用与服务器 CANN/设备匹配的 msSanitizer，加载其环境并确保 `mssanitizer` 在 PATH。
以下命令先跑普通精度，再构建插桩 wheel，对每个 rank 单独包裹 memcheck：

```bash
PYTHON=python bash validation/a8w8/run_ub_probe.sh --build --sanitize \
  --bs-values 64,120,121,128,4096 --inputs fp8,bf16 \
  --output-dir /data/megamoe-results/ub-memcheck-01
```

若要直接捕获旧内核的非法访问，可跳过普通设备调用，直接构建并执行检测版：

```bash
PYTHON=python bash validation/a8w8/run_ub_probe.sh --sanitize-only \
  --bs-values 128,4096 --inputs fp8 \
  --output-dir /data/megamoe-results/ub-memcheck-direct-01
```

`--sanitize-only` 会重新构建检测版并生成 Golden，不能与 `--perf` 一起使用。

诊断编译通过 `DEEPEP_UB_PROBE_SANITIZER=ON`，仅向 MegaMoE 设备库的编译/链接增加
`-g --cce-enable-sanitizer`，保留 O2 和原计算代码。标记写入 wheel 的 `_build_info.json`，
参与构建缓存身份。普通构建默认 OFF，不继承检测标记。检测使用每个 rank 独立日志、
`--tool=memcheck --kernel-name=mega_moe_kernel`，关闭该检测进程的 PyTorch NPU 内存池。

**检测版不能用来比较正常性能。** 同时指定 `--perf --sanitize` 时，先完成普通版性能，
再重新编译并安装检测版。命令结束后环境中保留检测版（报告会记录）；恢复正常 wheel：

```bash
PYTHON=python DEEPEP_BUILD_MEGAMOE=ON DEEPEP_MEGAMOE_TIMER=OFF \
  DEEPEP_UB_PROBE_SANITIZER=OFF bash experiments/megamoe/scripts/build.sh
```

普通版与检测版 pip 版本号相同，通过 build metadata 的 `ub_probe_sanitizer` 区分。
诊断分支整体 pip 版本与其他分支不同。不支持检测编译的旧 CANN、缺少工具、缺少某 rank
日志等情况都报告未完成。目标 `--repo` 若没有本分支的诊断编译支持，只能跑普通测试。

依据 [msSanitizer 使用指南](https://github.com/Ascend/mssanitizer/blob/master/docs/zh/user_guide/mssanitizer_user_guide.md)
及 [全量检测编译选项](https://github.com/Ascend/mssanitizer/blob/master/docs/zh/user_guide/compile_option_config.md)：
普通未插桩检测仅覆盖部分 GM 搬运；完整检测需要额外编译选项。Ascend950 的 SIMD VF 指令
感知、SHMEM 相关场景以及 blockDim 等仍受工具版本限制。因此脚本不会把零报错解释为
“UB 一定安全”，也不会为了适配工具擅自减少内核 blockDim 或修改通信实现。

## 报告与退出码

`summary.txt`/`summary.json` 分开记录容量超限、精度、异常、超时和检测证据。
`capacity.json` 保存布局源码指纹、目标容量来源、逐 BS 的地址边界。
`preflight_regular.json`/`preflight_sanitizer.json` 保存安装包身份和设备信息。
`commands.json`/`logs/` 保存命令、退出码和日志；`*_memcheck/rank_N.log` 是原始检测报告。
`accuracy_*`、`perf_*`、`sanitize_*` 含各 rank 的 manifest、精度或性能报告。

| 退出码 | 含义 |
| --- | --- |
| 0 | 所请求流程完成，未发现容量超限/用例失败；不等于证明所有设备内存访问安全 |
| 1 | 容量计算超限，或出现精度/执行/报告/检测错误；看各字段区分原因 |
| 2 | 环境、构建、平台查询、工具或必要报告缺失，诊断未完整完成 |

仅同一条 ERROR 块明确包含 MegaMoE、UB 和 illegal read/write 时，才标记
`device_ub_illegal_access_reported=true`；其他错误单独保留。进一步归因到 GMM1，仍须核对
该报告的调用栈/PC。`capacity_exceeded=true` 和设备没有报错可以同时出现。

旧版失败、修复版通过支持修复有效；两个分支还有其他差异，不能把全部性能变化归因于 M128。
本目录脚本未改变旧内核，便于后续制作仅包含 UB 修复的严格对照。

## 本地脚本回归

不依赖 pytest、Torch、CANN 或 NPU：

```bash
python -B -m unittest discover -s validation/a8w8 -p test_ub_probe.py -v
```

覆盖旧源码容量边界、BS4096 分块、平台容量变化、错误报告拒绝、漏 rank、非有限耗时、
检测证据归因和无副作用的 dry-run。CPU 回归不代表 CANN 编译或真机测试已经通过。
