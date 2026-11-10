# 空间索引实验 V2

以 [正式规格](SPEC_V2.md) 为准。见 [实验进度](PROGRESS.md) 和 [数据来源](LICENSES.md)。原交接包只含规范、说明、进度和默认配置。本目录的 M1 回放使用 C++20，数据处理与统计使用 Python 3.12、PyArrow 和 psutil。

## 构建和测试

从本目录执行。当前 Windows 路径含中文，MinGW Makefiles 生成后的依赖路径解析失败。已实测相对路径构建脚本，使用 Anaconda 下的 Boost 头文件和 libspatialindex C DLL。参数 `-LsiRoot` 可指定不同的依赖目录。

```powershell
./scripts/build_mingw.ps1 -LsiRoot D:/anaconda/Library
$env:PATH = 'D:/anaconda/Library/bin;' + $env:PATH
python -m unittest discover -s tests -v
```

CMakeLists 同时提供常规构建入口，可设置 `LSI_INCLUDE` 和 `LSI_LIBRARY`。MinGW + 中文路径环境以以上已验证的构建命令为准。

## 获取并处理真实轨迹

官方来源和许可见 LICENSES.md。下载脚本可获取官方 ZIP 或登记已经手工下载的原始 ZIP。校验检查成员结构、ZIP CRC 和 SHA-256。可通过 `--sha256` 比较发布方或独立来源提供的预期摘要。原始数据、规范化数据和轨迹保存在本地的忽略目录。

```powershell
python scripts/download.py tdrive --local-archive 'C:/path/to/tdrive.zip'
python scripts/prepare.py tdrive 'C:/path/to/tdrive.zip' data/processed/tdrive
python scripts/run_m1.py configs/m1_tdrive.json build/spatial_bench.exe results/m1_tdrive --source data/processed/tdrive
```

Porto 可用同样流程，将 `tdrive` 替换为 `porto`，配置使用 `configs/m1_porto.json`。预处理选择配置对应的 UTC 时间窗口，读取官方完整行程 ZIP，排除缺失标记行程。原有采样点的时间由行程起始 epoch 加 15 秒步长确定。T-Drive 保存源时间字符串，回放将其按北京本地时间转换为 epoch。规范化输出固定字段 `entity_id,timestamp,x,y,event_type`，并生成 `metadata.json` 和 `quality.json`。同一车辆使用稳定哈希归到一个集合。

## 合成冒烟复现

```powershell
python -X utf8 scripts/run_m1.py configs/m1_smoke.json build/spatial_bench.exe results/m1_synthetic_rerun
```

使用新的结果目录。每条轨迹保留 `events.parquet`、等价的 C++ 回放文本及 SHA-256。查询是固定种子生成的矩形/圆形，真实轨迹配置也使用人工生成查询。测试前用独立验证轨迹为每种索引搜索四组参数，然后锁定参数。每组候选参数在验证轨迹上运行五个独立计时进程，按在线耗时中位数选型。每个配对测试种子独立启动计时和 oracle 校验进程。Windows 计时使用 QueryPerformanceCounter。结果包含 `metrics.csv`、`tuning.csv`、`summary.json`、配置及环境记录。在线计时包含查询和所有动态变更，初始构建、轨迹载入和全量正确性校验分别处理。峰值 RSS 为计时进程的工作集峰值，包含公共回放数据。两套 R*-Tree 使用公开查询接口，访问节点数留空。

## M2 区域混合索引

S0、S1、S2 使用相同的 16×16 稀疏粗网格和点查询语义。固定策略与四种经典索引共享回放。通过 `AdaptiveRegionalIndex::switch_mode` 在单线程事件边界完成模式切换。首轮计时、逐区域剖析和正确性校验分别执行。

```powershell
./scripts/build_mingw.ps1 -Build build_m2_fair
$env:PATH = 'D:/anaconda/Library/bin;' + $env:PATH
python -X utf8 -m unittest discover -s tests -v
python -X utf8 scripts/run_m2.py 2k configs/m2.json build_m2_fair/spatial_bench.exe results/m2_rerun/controlled_2k
python -X utf8 scripts/run_m2.py tdrive configs/m2.json build_m2_fair/spatial_bench.exe results/m2_rerun/tdrive --source data/processed/tdrive
python -X utf8 scripts/run_m2.py 20k configs/m2.json build_m2_fair/spatial_bench.exe results/m2_rerun/controlled_20k
python -X utf8 scripts/analyze_m2.py results/m2_rerun/controlled_20k results/m2_rerun/analysis
```

T-Drive 复用既有的本地标准化数据。受控组使用纯合成对象轨迹，热点规则、阶段、配置和种子保存在清单中。默认实际事件落在本地忽略目录 `data/processed/m2_traces`。复现时使用新结果目录。图表脚本需要 Matplotlib。
