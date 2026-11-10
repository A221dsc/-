# M1 首轮空间索引基线报告

2026-11-10。本轮完成 T-Drive 原生时间戳试跑和独立合成回放。按 SPEC_V2，本轮对应 M0 的 D1 数据部分，以及 M1 的四套经典基线与矩形/圆形校验。

## 数据与复现口径

- 用户提供的 T-Drive ZIP 为 153,217,794 B，成员结构和 CRC 已核验。本地 SHA-256 为 `3baded659dc7536b8806de1cbb1e3987f04f122d27f6a8a3eced715836e02fc2`。本机直连微软公布的 OneDrive 下载链接收到 HTTP 403。Porto 首次传输超时，续传失败。[Microsoft Research 与数据说明](LICENSES.md)
- 全量读取 17,662,984 条记录。剔除 30,445 条无效记录，合并 1,330,669 条同时间同位置重复记录和 9,459 条同时间异位置记录，保留最后一条有效记录。最终保留 16,292,411 条、10,300 辆。训练、验证、测试分别有 6,184 / 2,044 / 2,072 辆，ID 集合两两无交集。[数据审计](results/logs/tdrive_audit.json)
- T-Drive 记录保留源时间戳，WGS84 经纬度以固定参考点投影到本地东、北米制坐标。测试使用北京时间 2008-02-03 08:00–09:00，07:30–08:00 的最后有效点构成 1,092 个初始对象。测试窗口实际收到 1,290 辆车的 13,966 条 GPS 记录，其中 13,752 次位置更新、214 次首次插入。保持最新已知位置的回放模型使在索引中的峰值对象数达到 1,306。
- 每组用 60 个一分钟事件窗口，按 GPS 源时间戳重放。每个种子生成 3,600 条矩形及 3,600 条圆形查询，两个形状交替，查询面积采用 0.01%、0.1%、1% 三个等级。查询中心均匀采样于局部坐标 ±35 km 的方形区域，GPS 不插值。该方形以外的 679 条测试 GPS 记录及 56 个初始对象仍参与索引。PR Quadtree 在数据超出当前根边界时扩展根节点并重建，成本随触发操作计入，当前未单列。
- 每个算法的四个候选参数各在独立车辆验证轨迹上计时五次，按在线时间中位数选择。测试种子为 101、203、307、409、503，GPS 更新轨迹相同，查询流不同。置信区间按配对种子重采样，仅反映该 GPS 窗口下的查询流变化。最低测试中位数参照用于描述性比较。[实验配置](results/m1_tdrive/config.json) · [调参记录](results/m1_tdrive/tuning.csv)

## T-Drive 实测

以下为五个独立进程的逐列中位数。在线耗时由完整回放墙钟计量，包含查询、位置更新和插入。初始构建、轨迹载入和 oracle 校验分别记录。RSS 为包含公共轨迹的进程峰值工作集。

| 索引 | 参数 | 构建 ms | 查询合计 ms | P50 / P95 / P99 μs | 位置更新 / 秒 | 在线总耗时 ms | 含构建总耗时 ms | 峰值 RSS MiB |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Uniform Grid | 1,000 m | 0.298 | 4.053 | 0.2 / 1.7 / 3.5 | 10,832,611 | 5.786 | 6.104 | 8.074 |
| PR Quadtree | 容量 32 | 0.384 | 2.459 | 0.2 / 0.9 / 1.8 | 3,872,930 | 6.583 | 6.955 | 8.309 |
| Boost R*-Tree | 容量 16 | 1.596 | 3.606 | 0.4 / 1.1 / 2.3 | 1,548,405 | 13.239 | 14.912 | 8.180 |
| libspatialindex R*-Tree | 容量 8 | 9.388 | 41.873 | 3.5 / 13.9 / 38.2 | 58,180 | 284.297 | 293.576 | 8.598 |

相对 Uniform Grid，PR Quadtree、Boost 和 libspatialindex 的配对在线时间比中位数分别为 1.11、2.29、48.16，描述性 95% bootstrap 区间分别为 [1.08, 1.47]、[2.19, 2.42]、[46.57, 51.42]。本轮无区域切换、辅助索引维护或模型决策，其相关指标为零。[逐种子 CSV](results/m1_tdrive/metrics.csv) · [统计 JSON](results/m1_tdrive/summary.json)

四种索引的候选数中位数依次为 47,145、204,540、33,797、33,797。Grid 计 206,023 次网格单元检查，Quad 计 263,232 次节点检查。两套 R*-Tree 的公开查询接口未暴露访问节点数，CSV 留空。Grid 在线成本主要来自查询，Quad 本组查询合计最快，但动态变更分项为 3.606 ms。Boost 和 libspatialindex 动态变更分项分别为 9.070 ms 和 238.339 ms。当前两个适配器以删除旧点再插入新点实现移动，libspatialindex 通过内存存储模式的 C API 接入。没有将库内部算法、对象存储和适配层开销分开归因。

## 正确性与运行产物

- T-Drive 四算法 × 五种子 × 7,200 次查询，共 144,000 次测试查询全部与独立暴力扫描一致。合成组每个种子包含 2,000 峰值对象、21,600 次位置更新、12 次插删、5,760 条查询，合计 115,200 次测试查询全部一致。Grid、Quad、Boost 和 libspatialindex 的在线耗时中位数依次为 5.214、8.399、24.588、540.505 ms。[T-Drive 运行日志](results/logs/tdrive-run.log) · [合成指标](results/m1_synthetic/metrics.csv)
- 四种算法各执行五个固定种子的 2,500 次随机混合操作，另测空索引、闭边界、零半径、重合点、删除和跨根更新。圆形浮点边界回归用例已验证修复前失败和修复后通过。Python 预处理测试覆盖乱序、重复时间戳、无效字段、投影和车辆组隔离。[C++ 测试日志](results/logs/build-and-cpp-tests.log) · [Python 测试日志](results/logs/python-tests.log)
- 代码位于 `src/core`、`src/indexes`、`src/workload`、`src/validation`、`src/benchmark` 和 `scripts`。规范化 CSV 和质量报告保存在本地 `data/processed/tdrive`。同一轨迹的 Parquet 与文本回放副本保存在本地 `data/processed/traces`，对应哈希登记在结果目录。[T-Drive 轨迹清单](results/m1_tdrive/trace_manifest.json)
- Windows 11、i7-13650HX、16,869,351,424 B RAM，单线程固定逻辑 CPU 0。GCC 13.2.0、Boost 1.82.0、libspatialindex 1.9.3，`-std=c++20 -O3 -DNDEBUG`。构建采用相对路径脚本，计时使用 Windows QueryPerformanceCounter。数据缓存为先载入内存再计时，未清空 OS 文件缓存。[锁定环境](environment.lock) · [运行环境与源码哈希](results/m1_tdrive/environment.json)

## 复现

在 V2 项目根目录用 PowerShell 执行。结果目录名可替换为一个新的名字。

```powershell
./scripts/build_mingw.ps1 -Build build_final -LsiRoot D:/anaconda/Library
$env:PATH = 'D:/anaconda/Library/bin;' + $env:PATH
python -X utf8 -m unittest discover -s tests -v
$archive = 'C:/path/to/T-drive Taxi Trajectories.zip'
python -X utf8 scripts/download.py tdrive --local-archive $archive --sha256 3baded659dc7536b8806de1cbb1e3987f04f122d27f6a8a3eced715836e02fc2
python -X utf8 scripts/prepare.py tdrive $archive data/processed/tdrive
python -X utf8 scripts/run_m1.py configs/m1_tdrive.json build_final/spatial_bench.exe results/m1_tdrive_rerun --source data/processed/tdrive
python -X utf8 scripts/run_m1.py configs/m1_smoke.json build_final/spatial_bench.exe results/m1_synthetic_rerun
```

## 后续验收

补充 Porto 原生轨迹对照，覆盖不同时间窗口并记录空间分布差异。完善 R*-Tree 节点计数和 PR 扩根成本的遥测。按 SPEC_V2 接续区域三态的正确性与无策略基准，另为扇形、线段与 GE-Tree 建立专项验证。待完整方法、经典对照和非强化学习选型器均已测量，再计算相对同组最强基线的加速比。
