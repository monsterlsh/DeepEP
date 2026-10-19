# MegaMoE A8W4 / A8W8 / A4W8：CPU 与 NPU 验证

本目录在 `fix-megamoe-a8w8` 和 `megamoe-a8w8-ep8` 两个分支中提供相同的测试入口。
两个分支均提供 `--mma-type`，旧的 `validation/a8w8/` 路径保留以兼容已有命令。
默认同时测试以下三种模式，所有模式使用 block32 E8M0 scale、BF16 输出：

| 模式 | `--mma-type` | 两层激活 / 权重精度 | 输入路径 |
| --- | --- | --- | --- |
| A8W4 | `fp8xfp4` | MXFP8 E4M3 / MXFP4 E2M1 | BF16 或预量化 FP8 |
| A8W8 | `fp8xfp8` | MXFP8 E4M3 / MXFP8 E4M3 | BF16 或预量化 FP8 |
| A4W8 | `fp4xfp8` | MXFP4 E2M1 / MXFP8 E4M3 | BF16 或预量化 packed FP4 |

A4W8 的输入和 SwiGLU 中间激活均先按 MXFP4 精度量化，再无损展开为 FP8 存储、通信和 MMAD；
它不减少当前 FP8 通信格式的字节数。BF16 输入由 native 路径量化；packed FP4 输入由 Python
运行时在设备上展开。该展开属于调用端到端路径，不计入融合 kernel 的 profiler 数字；
A4W8 另外记录包含该展开的 `full_call_device` 设备事件时长。
A8W8/A4W8 不支持共享专家和设备 timer；A8W4 保留其已有支持，本目录设备矩阵统一 S=0、timer=OFF。
CPU 回归不能替代 NPU 编译、精度、跨卡同步和性能验证。新增模式的真实 NPU 编译、精度和性能
需在服务器执行以下命令验收，文档中的用例数量是计划数量，不表示设备测试已经通过。

| 文件 | 用途 |
| --- | --- |
| `run_cpu.sh` | 从当前源码运行 CPU 回归，保存日志、JUnit、元数据和汇总 |
| `requirements-cpu.txt` | CPU 最小依赖，不安装 Torch、torch_npu 或 CANN |
| `run_npu.sh` | 构建安装、环境预检、CPU Golden、NPU 精度及性能测试 |
| `check_fp4_expand.py` | 单卡或 CPU 检查 FP4 全部字节展开；可将内部格式告警视为失败 |
| `test_runners.py` | 不使用 NPU 的脚本回归，检查两种分支接口和报告完整性 |

## 1. 拉取两个分支

在服务器上手动执行。CPU 测试不需要初始化子模块：

```bash
git clone --branch fix-megamoe-a8w8 --single-branch \
  https://github.com/monsterlsh/DeepEP.git DeepEP-fix
git clone --branch megamoe-a8w8-ep8 --single-branch \
  https://github.com/monsterlsh/DeepEP.git DeepEP-ep8
```

下面的命令适用于两个目录。每次运行使用新的结果目录；结果放在源码目录之外，
避免切分支时混用报告或从源码误导入包。两个分支的设备测试依次执行。

### 安装版本与分支对应关系

两个分支统一使用 `0.1.0.dev5` 基线，通过分支标识区分 Python 包版本。`setup.py` 从根目录 `VERSION` 读取版本，
因此 wheel 文件名、`pip show` 和测试预检显示的版本保持一致：

| 分支 | `pip show ascend-deepep` 的 Version |
| --- | --- |
| `fix-megamoe-a8w8` | `0.1.0.dev5+fix.megamoe.a8w8` |
| `megamoe-a8w8-ep8` | `0.1.0.dev5+megamoe.a8w8.ep8` |

`+` 后是 Python 包规范支持的本地版本标识，分支名中的连字符规范化为点号。
发行包名为 `ascend-deepep`，Python 导入名为 `deep_ep`。
安装后使用同一个解释器查询，避免查到其他虚拟环境：

```bash
python -I -m pip show ascend-deepep
python -I -c 'from importlib.metadata import version; print(version("ascend-deepep"))'
```

每次切换分支后运行 `bash validation/a8w8/run_npu.sh --build`。构建脚本使用
`python -I -m pip install --no-user --force-reinstall --no-deps` 安装本次生成的确切 wheel，
支持两个分支版本互相替换，以及从历史 EP8 dev7 版本切换到统一的 dev5 基线。
NPU 预检同时校验完整版本字符串和源码指纹；同一分支再次修改源码后也会检查 wheel 是否需要重建。

### 版本统一对功能和安装的影响

统一版本基线本身不回退源码，不改变算法、Python API、内核协议或构建开关。
两个分支原有的实现差异仍然保留；相同 `dev5` 不表示它们的功能完全相同。
`VERSION` 参与构建元数据和源码指纹计算，因此修改它后必须重新生成并安装 wheel，
旧 wheel 会被版本或源码指纹校验拒绝。

历史 dev7 在包管理器的版本排序中高于 dev5；两个 dev5 后缀之间也有排序。
不要依赖 `pip install --upgrade ascend-deepep` 来选择分支，应使用上面的 `--build`
入口强制安装当前源码生成的 wheel。两个版本仍属于同一个发行包，后安装的版本替换前一个。

### 已加 `--build` 仍报版本不匹配

CPU 测试运行源码，不检查 NPU wheel 是否已安装，因此 CPU 通过与安装版本不匹配可以同时发生。
若源码带有分支后缀，而安装显示无后缀的 `0.1.0.dev5`、历史版本或另一分支后缀，
先在服务器本地检查，不需要导出日志：

```bash
git branch --show-current
cat VERSION
python -I -m pip --version
python -I -m pip show ascend-deepep
python -I - <<'PY'
from importlib.metadata import distributions
import sys
print("Python:", sys.executable)
for item in distributions(name="ascend-deepep"):
    print(item.version, getattr(item, "_path", item.locate_file("")))
PY
```

旧构建脚本的 pip 安装没有隔离 `PYTHONPATH`。当它包含源码目录时，源码 `.egg-info`
可能遮蔽环境中已安装的版本，导致 pip 没有卸载旧 `.dist-info`。此时可以出现新代码与旧版本
元数据并存，日志含 `Not uninstalling ... outside environment` 或 `No files were found to uninstall`。
新版安装步骤使用 `-I` 隔离源码路径、`--no-user` 禁止装入预检不可见的用户目录，
并在安装结束时检查版本以及同一安装目录的重复元数据。离线源配置仍生效。

如果本次构建已经生成了版本正确的 wheel，且源码仍与构建时一致，可以直接重新安装它，
不必再次编译。若已更新源码（包括拉取本次构建脚本修复），应使用新版脚本加 `--build`
重新构建，否则源码指纹校验会拒绝旧 wheel：

```bash
# 使用本次 logs/build.log 中的完整 wheel 路径，不要用 dist/**/*.whl 通配混装两个分支。
WHEEL=/absolute/path/to/DeepEP/dist/build-XXXX/ascend_deepep-0.1.0.dev5+megamoe.a8w8.ep8-cp311-cp311-linux_aarch64.whl
python -I -m pip install --no-user --force-reinstall --no-deps "$WHEEL"
python -I -m pip show ascend-deepep

# 已确认安装版本正确后，不加 --build，避免旧版构建脚本再次污染安装。
bash validation/a8w8/run_npu.sh --suite quick --python "$(python -I -c 'import sys; print(sys.executable)')"
```

上面的 wheel 名称只是示例，Python ABI、平台和路径以本次构建实际输出为准。
若仍不匹配，检查当前环境的 pip `user`、`target`、`prefix` 配置及重复元数据；
不要为绕过报错临时修改 `VERSION` 或跳过版本校验。两分支应依次安装、依次测试。

## 2. CPU 测试

推荐 Linux、Python 3.11 或 3.12、Bash、Git；两个分支均需要 C++17 编译器
（如 `g++`），用于实际编译原生 FP4 量化策略头。可设置 `CXX=/path/to/g++`。
依赖固定为 NumPy 1.26.4；不要用 Python 3.13/3.14 安装这份固定依赖。

```bash
cd /path/to/DeepEP-ep8   # 另一分支换为 DeepEP-fix
python3.11 -m venv .venv
source .venv/bin/activate

# 首次执行显式安装依赖；默认不会运行 pip。
bash validation/a8w8/run_cpu.sh --install \
  --output-dir /data/megamoe-results/ep8-cpu-01

# 此后直接运行，默认 full。
bash validation/a8w8/run_cpu.sh \
  --output-dir /data/megamoe-results/ep8-cpu-02
```

`full` 运行 `experiments/megamoe/tests` 和本目录脚本回归，覆盖三种精度模式的 Python API、FP4/FP8
位模式、block32 scale、两层量化参考结果、Golden 缓存隔离、路由/计数、生命周期、构建和启动计划。
FP4 测试检查低/高 nibble 顺序、舍入边界、无损展开及 BF16/预量化输入路径。
两个分支均检查 FP4 量化、发送 UB 预算和 GMM1 子块策略；EP8 分支另有 L2 策略的 C++ 测试。
全套测试不初始化 NPU。若环境中没有 CPU PyTorch，依赖它的少量主机协议测试会 skip，
跳过数及原因保存在报告中；需要这些测试时使用已经装有 CPU PyTorch 的环境再跑。

Windows 可使用 Git Bash 和 `--suite portable`，它明确排除依赖 Linux 路径和 Unix shell
测试桩的启动/打包检查，不等于 Linux `full` 验收。Linux 服务器应使用默认 `full`。

```bash
bash validation/a8w8/run_cpu.sh --suite portable --dry-run
bash validation/a8w8/run_cpu.sh --suite portable --output-dir /tmp/deepep-cpu-portable-01
```

通过标准：退出码为 0，`summary.json` 中 `passed` 为 true；同时检查 skip/deselected
是否符合本次范围。依赖缺失、编译失败或 pytest 失败均返回非零，不会因为 `tee` 记录日志而变成成功。

## 3. NPU 环境与构建依赖

目标环境：**Linux、Ascend950、CANN 9.2.0、匹配的 Torch/torch_npu、Python ≥3.11**。
建议 Python 3.11/3.12。使用已准备好的 NPU 框架环境，不要复用上面未安装框架的纯 CPU venv。
还需要 Git、Bash、CMake、C++ 编译器和 GNU `timeout`。

```bash
cd /path/to/DeepEP-ep8
conda activate YOUR_NPU_ENV
source /path/to/cann/set_env.sh

# 有网络时初始化锁定的依赖。--build 也会检查/初始化未准备的子模块。
git submodule update --init --recursive -- third-party/shmem third-party/catlass

# 检查框架，不会在此命令中分配 NPU。
python -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__)'
```

依赖版本以仓库 `dependencies.lock.json` 为准：SHMEM 为
`4dd80e2129da42bd773982c2bb3758af1ed0bba8`，CATLASS 为
`2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1`。
不要用依赖仓库的最新分支替代这些版本。若使用预先准备的依赖，可以设置：

```bash
export SHMEM_ROOT=/path/to/validated/shmem/install/shmem
export CATLASS_ROOT=/path/to/pinned/catlass
```

SHMEM SDK 必须包含匹配的 `deepep-sdk.json`、头文件和库；CATLASS 需要保留其 Git 版本信息。
构建会校验版本和内容，详见仓库 `docs/build.md`、`scripts/record_shmem_sdk.py`。

## 4. NPU 测试：先 quick，再 full

默认单机 8 卡（EP8）。首次测试以及每次换分支后都使用 `--build`，
在当前环境构建并重新安装对应 wheel：

```bash
# 只预览：不构建、不导入 Torch、不生成报告、不启动设备。
bash validation/a8w8/run_npu.sh --build --suite quick --dry-run

# 首轮：三模式精度和融合 kernel 性能；A4W8 另测完整调用的设备时长。
bash validation/a8w8/run_npu.sh --build --suite quick \
  --output-dir /data/megamoe-results/ep8-npu-quick-01

# 完整矩阵；已安装当前源码对应 wheel 时可以省略 --build。
bash validation/a8w8/run_npu.sh --suite full \
  --output-dir /data/megamoe-results/ep8-npu-full-01

# 只看精度，可跳过 profiler 性能阶段。
bash validation/a8w8/run_npu.sh --suite full --skip-perf \
  --output-dir /data/megamoe-results/ep8-npu-accuracy-01

# 单独回归 A4W8；A8W4/A8W8 分别换成 fp8xfp4/fp8xfp8。
bash validation/a8w8/run_npu.sh --suite quick --mma-type fp4xfp8 \
  --output-dir /data/megamoe-results/ep8-a4w8-quick-01
```

在 `DeepEP-fix` 中执行相同命令，将结果路径中的 `ep8` 换为 `fix`，并重新加 `--build`。
不要把两个分支安装在同一环境后并发测试；后安装的 wheel 会替换前一个。
脚本在源码目录之外运行，清除测试子进程的 `PYTHONPATH`，检查安装版本和对应源码，
避免测到旧 wheel 或源码遮蔽安装包。

两卡环境可用 EP2；显式选择两张空闲卡：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
  bash validation/a8w8/run_npu.sh --ep-size 2 --build --skip-perf \
  --output-dir /data/megamoe-results/fix-ep2-01
```

EP 表示进程/rank 数，不是全局专家数。默认 smoke 配置 H=1024、I=256、E=8、TopK=2、S=0，
EP8 每卡一个专家，EP2 每卡四个专家。EP2 不能代替 EP8 验证。

| 套件 | 精度覆盖 | 性能覆盖 |
| --- | --- | --- |
| `quick` | 三种模式，BF16/各自预量化输入，balanced，BS=96/120/121/128/129/257，全部 token | 三种模式，BS=96/4096，各独立重复 3 次 |
| `full` | 三种模式，BF16/各自预量化输入，balanced/random/skewed，14 个 BS；每模式均有 tail、wide、topk32、route_ub 边界 | 三种模式，BS=96/4096，各独立重复 3 次 |

默认 `--mma-type all`：quick 应生成 36 份精度报告、18 份性能报告；full 为 300 和 18。
单独选择一个模式：quick 为 12 和 6；full 为 100 和 6。
包含 A4W8 且未跳过性能时，6 个 A4W8 性能用例各自还有 `full_call_device` 统计和逐 rank 的
`full_call_profiling.json`。这是同一性能用例的另一种计时口径，不增加上述性能用例数量。
`--skip-perf` 时性能报告应为 0。完整矩阵中的 wide 和 route_ub 占用更多内存、CPU Golden 耗时更长。
两个分支都覆盖这些 UB 和原生形状边界，仍需分别在目标设备验收；测试失败应保留日志分析。
每个测试先生成本分支匹配的 Golden，再检查所有 rank 的输出、专家计数和累计计数。
汇总同时检查每种精度、每条输入路径的报告数量和 manifest；缺少 A4W8 或误跑成 FP8 输入不会算通过。
同时检查 A4W8 完整调用设备时长的文件、rank 数、有效数值和展开标识；缺失或与输入类型不符也会失败。
报告仅代表所列公开调用路径，不能替代所有原生格式负例的设备测试。
旧版本仅有 `--weights` CLI 时，脚本只允许显式选择 `fp8xfp4`/`fp8xfp8`；
选择 `all` 或 `fp4xfp8` 会在构建前提示升级，不会把 A4W8 映射成 A8W8。

可通过 `--python /path/to/python` 指定解释器。使用 `--help` 查看完整参数。
`ASCEND_RT_VISIBLE_DEVICES` 的卡数必须等于 EP；默认 SHMEM 地址为
`tcp://127.0.0.1:26733`，可通过 `DEEPEP_SHMEM_ENDPOINT` 改为本机空闲端口。
分布式用例设置超时，超时退出也算失败；保留原始日志，再检查运行中的设备作业。

## 5. 内网离线依赖

在与目标服务器 **同操作系统、架构、Python 版本** 的可联网环境准备 wheelhouse，
随后手动复制到内网。下面只下载 CPU 和构建依赖，不包含 Torch/torch_npu/CANN：

```bash
python -m pip download -d /path/to/wheelhouse \
  -r validation/a8w8/requirements-cpu.txt -r requirements-build.txt
```

内网 CPU 安装：

```bash
bash validation/a8w8/run_cpu.sh --install --wheelhouse /path/to/wheelhouse \
  --output-dir /data/megamoe-results/cpu-offline-01
```

内网 NPU 构建：先准备匹配的框架/CANN 和完整的锁定子模块或外部依赖目录，再设置 pip 离线来源。
仅 clone 主仓库不会自动携带 SHMEM/CATLASS 源码；pip wheelhouse 也不包含这些 C++ 依赖。

```bash
export PIP_NO_INDEX=1
export PIP_FIND_LINKS=/path/to/wheelhouse
bash validation/a8w8/run_npu.sh --build \
  --output-dir /data/megamoe-results/npu-offline-01
```

若子模块已准备完整，构建不需要重新下载它们；如果未准备，需先通过内网镜像或手动迁移补齐。

## 6. 收集结果与比较两个分支

### A4W8 展开时的 internal-format 告警

`Cannot create tensor with internal format while allow_internal_format=False, tensor will be created with base format`
表示 torch_npu 的兼容分配路径使用基础存储格式；这条告警本身不说明 FP4 精度或计算结果错误。
部分版本即使请求 ND 也会进入该告警分支，见
[torch_npu 分配实现](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/aten/common/TensorFactories.cpp)。
是否通过仍按下文的退出码、`passed=true` 和完整报告判断。

packed FP4 展开现使用 256×2 字节查表及 `index_select(out=...)` 写入连续缓冲区，
避免原先的高级索引、设备侧拆 nibble 和跨步写入，保留所有 FP4 编码（包括负零）。
不修改业务进程的 `allow_internal_format`，也不屏蔽告警。具体框架/CANN 组合是否还触发兼容路径，
须在目标 NPU 验证；本地 CPU 的逐字节测试不能证明目标 NPU 无告警。

更新并重新构建安装后，可单独运行以下检查，无需启动 EP8 或准备模型数据。
脚本只在自己的进程内设置 `allow_internal_format=False`，用非默认 NPU stream 测试全部
256 种打包字节、输出缓冲区别名、容量切片、负零和相邻行保护。

```bash
# -I 确保使用已安装的 wheel；--strict-warning 将这一条告警视为失败，其他告警照常显示。
ASCEND_RT_VISIBLE_DEVICES=0 python -I validation/a8w8/check_fp4_expand.py --device npu --strict-warning
# 已安装 wheel 的 CPU 版本检查（需要 CPU 可导入的 torch 和包环境）。
python -I validation/a8w8/check_fp4_expand.py --device cpu

# 完整算子回归：同一分支源码变化也需要 --build，以通过源码指纹校验。
bash validation/a8w8/run_npu.sh --build --mma-type fp4xfp8 --suite full
```

### 结果与性能口径

每次执行结束后保留完整结果目录。CPU 查看 `summary.json`、JUnit 和 pytest 日志；
NPU 查看 `summary.txt`、`summary.json`、元数据、`logs/`、`accuracy_*/`、`perf_*/` 和 `golden/`。
任一阶段报错/超时、报告缺失、rank 数错误或计数错误都不算通过。
NPU 成功要求退出码为 0、汇总 `passed=true` 且报告数量完整。

```bash
cat /data/megamoe-results/ep8-npu-quick-01/summary.txt
tar -czf /data/megamoe-results/ep8-npu-quick-01.tar.gz \
  -C /data/megamoe-results ep8-npu-quick-01
```

比较性能时使用相同硬件、EP、模型形状、输入类型、路由、框架/CANN、预热和计时口径。
`mean_us` 是逐轮最快 rank 耗时的均值，`rank_max_mean_us` 是逐轮最慢 rank 耗时的均值。
默认 profiler 前置对齐张量约占每卡 12 GB，不计入融合 kernel Duration。
这些数值仅统计融合 MegaMoE kernel Duration，不是完整 API 端到端时延；尤其不包括 A4W8 的
packed FP4 输入在 Python 运行时展开的设备操作。所有报告中的设备 timer 均关闭。
不同精度会改变数值误差与设备工作量，不能据融合 kernel 数字断言完整 A4W8 调用一定更快。

A4W8 的 `profiling.json.full_call_device` 和同目录 `full_call_profiling.json` 使用调用方流的 NPU
事件计时，覆盖一次公开调用的设备执行，包括 packed FP4 输入的展开和融合 kernel。
`includes_fp4_expansion=true` 对应 `--input fp4`，BF16 输入则为 false（BF16 量化在 native 内执行）。
该字段不是 Python 主机墙钟时长，不能代表包含数据准备、权重转换、编译安装等阶段的业务端到端耗时。
汇总分别显示 `PERF fused-kernel` 和 `PERF full-call-device`，前者保留原来的比较口径，后者让
A4W8 的输入展开开销可见。`summary.json` 中的 `expected_full_call` 为 6 或 0。

**两分支的 FP8 随机权重生成方法不同，相同 seed 不代表相同权重。**
本套件分别验证各分支与其自身 Golden 的一致性，不能据此宣称两套 NPU 输出逐元素相同。
严格的跨分支数值对拍还需要共享同一批真实输入、权重、scale 和路由。
