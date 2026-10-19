# MegaMoE A8W8：CPU 与 NPU 验证

本目录在 `fix-megamoe-a8w8` 和 `megamoe-a8w8-ep8` 两个分支中提供相同的测试入口。
脚本自动适配前者的 `--weights fp8` 和后者的 `--mma-type fp8xfp8`。
这里的 A8W8 是 MXFP8 E4M3 激活与权重，配 E8M0 scale，支持 BF16/FP8 输入、BF16 输出；
不支持 A8W8 共享专家和设备 timer。CPU 回归不能替代 NPU 编译、精度、跨卡同步和性能验证。

| 文件 | 用途 |
| --- | --- |
| `run_cpu.sh` | 从当前源码运行 CPU 回归，保存日志、JUnit、元数据和汇总 |
| `requirements-cpu.txt` | CPU 最小依赖，不安装 Torch、torch_npu 或 CANN |
| `run_npu.sh` | 构建安装、环境预检、CPU Golden、NPU 精度及性能测试 |
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

两个分支使用不同的、带分支标识的 Python 包版本。`setup.py` 从根目录 `VERSION` 读取版本，
因此 wheel 文件名、`pip show` 和测试预检显示的版本保持一致：

| 分支 | `pip show ascend-deepep` 的 Version |
| --- | --- |
| `fix-megamoe-a8w8` | `0.1.0.dev5+fix.megamoe.a8w8` |
| `megamoe-a8w8-ep8` | `0.1.0.dev7+megamoe.a8w8.ep8` |

`+` 后是 Python 包规范支持的本地版本标识。发行包名为 `ascend-deepep`，Python 导入名为 `deep_ep`。
安装后使用同一个解释器查询，避免查到其他虚拟环境：

```bash
python -m pip show ascend-deepep
python -c 'from importlib.metadata import version; print(version("ascend-deepep"))'
```

每次切换分支后运行 `bash validation/a8w8/run_npu.sh --build`。构建脚本使用
`pip install --force-reinstall --no-deps` 安装本次生成的 wheel，允许从 dev7 切回 dev5。
NPU 预检同时校验完整版本字符串和源码指纹；同一分支再次修改源码后也会检查 wheel 是否需要重建。

## 2. CPU 测试

推荐 Linux、Python 3.11 或 3.12、Bash、Git；`megamoe-a8w8-ep8` 还需要 C++17 编译器
（如 `g++`），用于实际编译内核策略头。可设置 `CXX=/path/to/g++`。
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

`full` 运行 `experiments/megamoe/tests` 和本目录脚本回归，覆盖 Python API、FP8 位模式、
scale、Golden 缓存、路由/计数、生命周期、构建和启动计划。
新分支还包含发送 UB 预算、GMM1 子块和 L2 策略的 C++ 测试。
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

# 首轮：A8W8 精度通过后执行性能测试。
bash validation/a8w8/run_npu.sh --build --suite quick \
  --output-dir /data/megamoe-results/ep8-npu-quick-01

# 完整矩阵；已安装当前源码对应 wheel 时可以省略 --build。
bash validation/a8w8/run_npu.sh --suite full \
  --output-dir /data/megamoe-results/ep8-npu-full-01

# 只看精度，可跳过 profiler 性能阶段。
bash validation/a8w8/run_npu.sh --suite full --skip-perf \
  --output-dir /data/megamoe-results/ep8-npu-accuracy-01
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
| `quick` | A8W8，FP8/BF16，balanced，BS=96/120/121/128/129/257，全部 token | A8W8，BS=96/4096，各独立重复 3 次 |
| `full` | A8W4/A8W8，FP8/BF16，balanced/random/skewed，14 个 BS；另有 tail、wide、topk32、route_ub 边界 | A8W4/A8W8，BS=96/4096，各独立重复 3 次 |

quick 应生成 12 份精度报告、6 份性能报告；full 为 184 和 12。
`--skip-perf` 时性能报告应为 0。完整矩阵中的 wide 和 route_ub 占用更多内存、CPU Golden 耗时更长。
这些边界用例可能暴露同事分支尚未包含的 UB 修复问题，测试失败应保留日志分析。
每个测试先生成本分支匹配的 Golden，再检查所有 rank 的输出、专家计数和累计计数。
报告仅代表所列公开调用路径，不能替代所有原生格式负例的设备测试。

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
所有报告中的设备 timer 均关闭。

**两分支的 FP8 随机权重生成方法不同，相同 seed 不代表相同权重。**
本套件分别验证各分支与其自身 Golden 的一致性，不能据此宣称两套 NPU 输出逐元素相同。
严格的跨分支数值对拍还需要共享同一批真实输入、权重、scale 和路由。
