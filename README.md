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

## M3 长时序区域选型

S3 阈值选型与 S4 代价贪心在 1,200 步（4 阶段 × 300 步）动态负载上在线选择区域模式。两策略共享 50 步决策窗口、2 窗口冷却、每窗口 4 次重建与 6,000 实体重建预算、同一索引库；决策只读取当窗之前已观测的负载，切换、局部维护和策略自身开销全部计入在线时间。实体聚集区与查询关注区分开设置：两个稠密力群从不被查询，两个查询热点位于稀疏区，构成可被选型利用的异质性。

```powershell
./scripts/build_mingw.ps1 -Build build_m3
$env:PATH = 'D:/anaconda/Library/bin;' + $env:PATH
python -X utf8 scripts/m3_workload.py configs/m3.json data/processed/m3_traces
python -X utf8 scripts/calibrate_m3.py configs/m3.json build_m3/spatial_m3.exe data/processed/m3_traces results/m3/calibration --cpu 0
python -X utf8 scripts/run_m3.py configs/m3.json results/m3/calibration/selected.json results/m3/calibration/selector_final.txt build_m3/spatial_m3.exe results/m3/final --cpu 0
python -X utf8 scripts/analyze_m3.py results/m3/final results/m3/calibration results/m3/analysis
```

轨迹由固定种子生成器产出，保存在本地忽略目录；标定只用训练与验证轨迹，测试种子在参数冻结后只读一次。`run_m3.py` 为每个方法/种子分别启动校验、计时和剖析进程，S3/S4 的决策哈希必须在计时与剖析之间一致。`analyze_m3.py` 产出逐种子对照、分阶段成本、决策最优性、回本和离线上下界与图。详见 [M3 报告](M3_REPORT.md)。

C++ 二进制在含中文的绝对路径下会因窄字符路径转换失败，轨迹与配置文件请使用相对路径。

## M4 想定驱动的双轨回放与前馈验证

地形来自 Moving AI Lab 公开 2D 基准（DAO 地图集，`https://movingai.com/benchmarks/dao/dao-map.zip`，研究用途，引用 Sturtevant 2012）。想定定义名义路线、阶段表与周期查询；名义推导器只读想定与地图，输出未来 200 步的区域负载；实际执行器用独立种子注入延迟、速度噪声、上报噪声、改道与阶段偏移，写出所有索引共同重放的事件文件。防泄漏测试要求同想定换实际种子后名义负载表逐字节不变。

```powershell
./scripts/build_mingw.ps1 -Build build_m3
$env:PATH = 'D:/anaconda/Library/bin;' + $env:PATH
python -X utf8 scripts/m4_build.py configs/m4.json data/processed/m4 --splits train validation test
python -X utf8 scripts/m4_build.py configs/m4.json data/processed/m4 --leakage
python -X utf8 scripts/m4_actuals.py configs/m4.json data/processed/m4 --actuals 3 --splits test validation
python -X utf8 scripts/m4_accuracy.py configs/m4.json data/processed/m4 results/m4/accuracy
python -X utf8 scripts/m4_calibrate.py configs/m4.json build_m3/spatial_m3.exe data/processed/m4 results/m4/calibration --cpu 0
python -X utf8 scripts/run_m4.py configs/m4.json results/m4/calibration/selected.json results/m4/calibration/selector_final.txt build_m3/spatial_m3.exe data/processed/m4 results/m4/run --cpu 0
python -X utf8 scripts/m4_analyze.py configs/m4.json results/m4/run results/m4/accuracy data/processed/m4/loads results/m4/analysis
python -X utf8 scripts/m4_export_viz.py configs/m4.json data/processed/m4 build_m3/spatial_m3.exe results/m4/calibration/selector_final.txt results/m4/viz --scenario test-0-brc100d --actual 0 --method greedy-scenario
```

新方法名（报告与 CSV 使用完整名称）：Fixed Regional Scan、Fixed Local Fine Grid、Fixed Local Quadtree、Threshold-Based Adaptive Regional Index Selection、Cost-Aware Greedy Regional Index Selection，以及两者各加 `(Scenario)` 的想定前馈变体。C++ 仍接受旧缩写用于 M1–M3 复现。详见 [M4 报告](M4_REPORT.md)。

## M4 第二阶段：负载语义修复与不确定性感知前馈

索引按“查询外包盒触达区域”统计查询、按“跨区新旧各一次”统计更新、按“最后上报位置”统计实体。名义推导与离线标签必须使用同一语义：`m4_loadstats.py` 用独立 Python 实现复现索引语义，并通过 `spatial_m3 loadprobe` 在每条实际轨迹上逐窗口核对。旧 Python 语义标签保留在 `data/processed/m4/loads_legacy_python_semantics/`。

```powershell
python -X utf8 scripts/m4_loadstats.py data/processed/m4 build_m3/spatial_m3.exe
python -X utf8 scripts/m4_uncertainty.py configs/m4.json data/processed/m4 results/m4/uncertainty
python -X utf8 scripts/m4_accuracy.py configs/m4.json data/processed/m4 results/m4/accuracy_v2
python -X utf8 scripts/run_m4.py configs/m4.json results/m4/calibration/selected.json results/m4/calibration/selector_final.txt build_m3/spatial_m3.exe data/processed/m4 results/m4/run_v2 --cpu 0
python -X utf8 scripts/run_m4_phase2.py configs/m4.json results/m4/calibration/selected.json results/m4/calibration/selector_final.txt build_m3/spatial_m3.exe data/processed/m4 results/m4/phase2 --cpu 0
```

新增完整方法名：History-Load Cost-Aware Greedy Regional Index Selection、Nominal Point Feed-Forward Regional Index Selection、Footprint-Calibrated Feed-Forward Regional Index Selection、Uncertainty-Aware Feed-Forward Regional Index Selection、Risk-Constrained Uncertainty-Aware Feed-Forward Regional Index Selection。详见 [M4 第二阶段报告](M4_PHASE2_REPORT.md)。

