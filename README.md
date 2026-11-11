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

## M4 第三阶段：风险度量、实体退出与同场全方法对照

风险度量包含纯期望成本、分位成本惩罚与 CVaR（α=0.8）；样本量 8/32/64/128 的稳定性审计见 `scripts/m4_risk_audit.py`。`.m3bin` 支持实体删除事件（`D`）。新留出地图 `den011d`（测试）、`orz200d`（验证）、`den101d`（训练）已冻结种子。

```powershell
python -X utf8 scripts/m4_risk_audit.py configs/m4.json data/processed/m4 results/m4/calibration results/m4/phase3/risk_audit
python -X utf8 scripts/run_m4_phase3.py configs/m4.json results/m4/calibration/selected.json results/m4/calibration/selector_final.txt build_m3/spatial_m3.exe data/processed/m4 results/m4/phase3_v2 --samples-tag _m32
python -X utf8 scripts/run_m4_final.py
python -X utf8 scripts/run_m4_exits.py
```

完整方法名：Uncertainty-Aware Feed-Forward Regional Index Selection、Quantile-Cost-Penalized Feed-Forward Regional Index Selection、CVaR-Risk-Constrained Feed-Forward Regional Index Selection、Historical-Error-Feedback Robust Feed-Forward Regional Index Selection。详见 [M4 第三阶段报告](M4_PHASE3_REPORT.md)。

## M4.4：交互式强化学习环境与 Double DQN

```powershell
python -X utf8 scripts/m4_rl_env_test.py
python -X utf8 scripts/m4_dqn.py configs/m4.json data/processed/m4 build_m3/spatial_m3.exe results/m4/phase4/dqn --seeds 11 22 --episodes 40
python -X utf8 scripts/m4_dqn_eval.py configs/m4.json data/processed/m4 build_m3/spatial_m3.exe results/m4/phase4/dqn results/m4/phase4/eval
python -X utf8 scripts/run_m4_phase4_compare.py
python -X utf8 scripts/m4_offline_ref.py configs/m4.json data/processed/m4 results/m4/calibration results/m4/phase4/offline
```

`spatial_m3 rlserve FINE QUAD CPU TRACE VERIFY` 提供持久化交互回放；`replayplan FINE QUAD CPU TRACE PLAN` 对学习策略做同路径计时。完整方法名：Historical-Load Double DQN Regional Index Selection、Scenario-Forward-Enhanced Double DQN Regional Index Selection。详见 [M4.4 报告](M4_PHASE4_REPORT.md)。

## M4.5：成本审计、共享预算调度器与小规模学习验证

```powershell
python -X utf8 scripts/m4_rl_env_test.py
python -X utf8 scripts/m4_small_learnability.py results/m4/calibration data/processed/m4 results/m4/phase5/small
python -X utf8 scripts/m4_offline_ref.py configs/m4.json data/processed/m4 results/m4/calibration results/m4/phase5/offline
python -X utf8 scripts/m4_beam_ref.py results/m4/calibration data/processed/m4 results/m4/phase5/offline --width 16
python -X utf8 scripts/m4_dqn.py configs/m4.json data/processed/m4 build_m3/spatial_m3.exe results/m4/phase5/dqn --seeds 11 22 33 --episodes 60
python -X utf8 scripts/m4_dqn_eval.py configs/m4.json data/processed/m4 build_m3/spatial_m3.exe results/m4/phase5/dqn results/m4/phase5/eval
python -X utf8 scripts/run_m4_phase5_compare.py
```

`step <actions>;<benefits>` 让调度器按预计收益执行合法候选；`rejected` 返回拒绝原因。成本口径为查询+基础更新+局部维护+切换，构建只作诊断。详见 [M4.5 报告](M4_PHASE5_REPORT.md)。

## M4.6：负载感知自适应分区

`PartitionIndex` 使用 64×64 子单元、叶子矩形划分与叶内 SCAN/FINE_GRID/LOCAL_QUAD；`spatial_m3 partition FINE QUAD CPU TRACE PLAN VALIDATE` 回放分区计划（`L x0 y0 x1 y1 level` + `M window leaf mode benefit`）。

```powershell
python -X utf8 scripts/m4_partition_experiment.py configs/m4.json data/processed/m4 build_m3/spatial_m3.exe results/m4/calibration results/m4/phase6/ablation
python -X utf8 scripts/run_m4_phase6_compare.py
```

结论：固定 16×16 分区优于自适应分区与 32×32；详见 [M4.6 报告](M4_ADAPTIVE_PARTITION_RL_REPORT.md)。

## M4.7：分区语义修复与在线选型闭环

```powershell
python -X utf8 scripts/m4_partition_plans.py configs/m4.json data/processed/m4 results/m4/phase6/ablation/plans results/m4/phase7/plans
python -X utf8 scripts/run_m4_phase7_compare.py
```

`partition_greedy FINE QUAD CPU TRACE PLAN CONFIG PRIOR VALIDATE` 在分区叶子上做在线成本贪心（先验或历史负载），切换成本使用叶子当前实体数，决策时间计入在线时间。详见 [M4.7 报告](M4_PARTITION_ONLINE_REPORT.md)。

## M4.8：PartitionIndex 上的 Double DQN 试点

```powershell
python -X utf8 scripts/m4_dqn_partition.py configs/m4.json data/processed/m4 build_m3/spatial_m3.exe results/m4/phase8/plans results/m4/phase8/dqn --seeds 11 22 33 --episodes 40
python -X utf8 scripts/m4_dqn_partition_eval.py configs/m4.json data/processed/m4 build_m3/spatial_m3.exe results/m4/phase7/plans results/m4/phase8/dqn results/m4/phase8/eval
python -X utf8 scripts/run_m4_phase8_compare.py
```

`partition_rlserve FINE QUAD CPU TRACE PLAN VERIFY` 提供可变叶子数的交互回放；`partition_greedy` 的四项成本已互斥。锚点漂移超阈值批次自动标记无效。详见 [M4.8 报告](M4_PARTITION_RL_PILOT_REPORT.md)。

## M4.9：强化学习实验修复

```powershell
python -X utf8 scripts/m4_dqn_partition.py configs/m4.json data/processed/m4 build_m3/spatial_m3.exe results/m4/phase8/plans results/m4/phase9/dqn --seeds 11 22 33 --episodes 40
python -X utf8 scripts/m4_dqn_partition_eval.py configs/m4.json data/processed/m4 build_m3/spatial_m3.exe results/m4/phase7/plans results/m4/phase9/dqn results/m4/phase9/eval_uniform
python -X utf8 scripts/run_m4_phase9_compare.py
```

窗口对齐、独立验证、动作收益、特征/推理计时与“逐轨迹 + 哨兵”锚点已修复；M4.8 标签以 [M4.9 报告](M4_PARTITION_RL_CORRECTED_REPORT.md) 为准。

## M4.10：RL 选型诊断与双向锚点

```powershell
python -X utf8 scripts/m4_phase10_tests.py
python -X utf8 scripts/m4_phase10_diagnostics.py build_m3/spatial_m3.exe results/m4/phase9/dqn results/m4/phase7/plans data/processed/m4/traces results/m4/phase10/diag --scenario test-0-brc100d
python -X utf8 scripts/run_m4_phase9_compare.py
```

锚点改为双向漂移（max/min），DQN 在线时间补计特征生成；逐叶决策日志与选型质量见 [M4.10 报告](M4_PARTITION_RL_DIAGNOSTIC_REPORT.md)。

## M4.11：反事实收益测量

```powershell
python -X utf8 scripts/m4_counterfactual_test.py
python -X utf8 scripts/m4_counterfactual.py build_m3/spatial_m3.exe results/m4/phase8/plans data/processed/m4/traces results/m4/phase11/cf --windows 12 --top 2 --repeats 2 --horizons 1 2 4 --noise-ms 2.0
```

`partition_cf FINE QUAD CPU TRACE PLAN WINDOW LEAF MODE_A MODE_B HORIZON VERIFY` 在相同状态下测量“保持 vs 切换”的真实窗口成本；噪声底约 2 ms，低于该值的标签标记为不确定。详见 [M4.11 报告](M4_COUNTERFACTUAL_SELECTION_REPORT.md)。

## M4.12：反事实测量 v2

```powershell
python -X utf8 scripts/m4_counterfactual_test.py
python -X utf8 scripts/m4_counterfactual.py build_m3/spatial_m3.exe results/m4/phase8/plans data/processed/m4/traces results/m4/phase12/cf --windows 12 --top 3 --pairs 5 --null-pairs 3 --horizons 1 2 4
```

`partition_cf FINE QUAD CPU TRACE PLAN WINDOW LEAF MODE_A MODE_B HORIZON VERIFY PROFILE ORDER` 输出完整墙钟与叶子局部分量，两臂状态摘要必须一致；标签按配对 t 区间分为可靠正/负、可重复近零、无法分辨。详见 [M4.12 报告](M4_COUNTERFACTUAL_V2_REPORT.md)。

## M4.13：受控可分性与三时域负载

```powershell
python -X utf8 scripts/m4_controlled.py build_m3/spatial_m3.exe results/m4/phase13/controlled
python -X utf8 scripts/m4_stratified_cf.py build_m3/spatial_m3.exe results/m4/phase8/plans data/processed/m4/traces data/processed/m4/loads results/m4/phase13/stratified
```

`partition_cf` 输出 `obs_prev_*`（w−1 观测）、`prior_*`（w 名义预测）、`actual_future_*`（仅离线）三组负载；受控场景给出 QUAD 的可靠正/负收益。详见 [M4.13 报告](M4_COUNTERFACTUAL_SEPARABILITY_REPORT.md)。

## H0：同构空间分区（新分支 research/scenario-aware-partition）

```powershell
git switch research/scenario-aware-partition
./scripts/build_mingw.ps1 -Build build_m3
python -X utf8 scripts/homogeneous_partition_baselines.py
python -X utf8 scripts/homogeneous_partition_experiment.py
```

`spatial_m3 partition FINE QUAD CPU TRACE PLAN VALIDATE STATIC_MODE` 的 STATIC_MODE=2 把所有叶固定为同一 PR Quadtree；H0 对照全图 vs 4×4/8×8/16×16/32×32 vs 非均匀分区。结论与复现见 [H0 报告](H0_HOMOGENEOUS_PARTITION_REPORT.md)。

## H0.1：轻量同构分区

```powershell
python -X utf8 scripts/homogeneous_partition_v2_experiment.py
python -X utf8 scripts/h01_loads.py build_m3/spatial_m3.exe results/homogeneous_partition/controlled
```

`hpartition FINE QUAD CPU TRACE PLAN VALIDATE PROFILE` 使用 `HomogeneousPartitionIndex`（每叶同一 Quad、全局 id→leaf、代际路由）；1×1 为直接 Quad 的 1.043×。受控负载显示小查询热点/更新密集/迁移阶段分区有净收益，大范围查询/均匀阶段变差。详见 [H0.1 报告](H01_LIGHTWEIGHT_PARTITION_REPORT.md)。

