# H3-P0 审计数据摘要（results/h3_p0_audit）

由 `tests/h3_p0_audit.cpp` 生成；每个 CSV 是一次独立运行的结果。
控制台日志未逐行保存，CSV 为权威数据。复现：`bash scripts/run_h3_p0_audit.sh`（含日志）。

方法：A=GLOBAL_H3（既有基线）；A1=FLAT_POINTS；A2=DENSE_BUCKETS；A3=CSR_PACKED；
A4=SLOT_PACKED；A4'=SLOT_PACKED_NOSORT（去掉结果排序，仅作相位归因）。

## 4 km / res 9 / clustered / 20k 实体 / 20k 查询

| 方法 | seed 101 (ms) | seed 202 (ms) | seed 303 (ms) | 均值 | µs/查询 |
|---|---:|---:|---:|---:|---:|
| GLOBAL_H3 | 1989.3 | 2370.7 | 1888.1 | 2082.7 | 104.14 |
| FLAT_POINTS | 1676.9 | 2145.2 | 1588.9 | 1803.6 | 90.18 |
| DENSE_BUCKETS | 1628.5 | 2069.1 | 1559.7 | 1752.4 | 87.62 |
| CSR_PACKED | 1655.6 | 2022.2 | 1629.3 | 1769.0 | 88.45 |
| SLOT_PACKED | 1611.6 | 1971.0 | 1538.5 | 1707.0 | 85.35 |
| SLOT_PACKED_NOSORT | 297.8 | 357.1 | 282.4 | 312.4 | 15.62 |

- 候选覆盖生成：24.9 ms（1.2% of A）
- `candidate_cells` = ['526236', '577668', '569061']；`candidates` = ['77290455', '88645256', '70375326']（所有方法逐位相同）

## 20 km / res 10 / uniform / 20k 实体 / 10k 查询

| 方法 | seed 101 (ms) | seed 202 (ms) | seed 303 (ms) | 均值 | µs/查询 |
|---|---:|---:|---:|---:|---:|
| GLOBAL_H3 | 200.4 | 199.3 | 206.5 | 202.1 | 20.21 |
| FLAT_POINTS | 173.8 | 181.2 | 174.7 | 176.6 | 17.66 |
| DENSE_BUCKETS | 139.9 | 130.7 | 131.5 | 134.0 | 13.40 |
| CSR_PACKED | 187.8 | 121.9 | 125.4 | 145.0 | 14.50 |
| SLOT_PACKED | 131.1 | 123.2 | 119.6 | 124.6 | 12.46 |
| SLOT_PACKED_NOSORT | 111.9 | 108.8 | 125.2 | 115.3 | 11.53 |

- 候选覆盖生成：78.3 ms（38.7% of A）
- `candidate_cells` = ['1745272', '1760465', '1754086']；`candidates` = ['1021514', '1029533', '1026530']（所有方法逐位相同）

## 扣除公共常数后的分区代价（索引工作，ms）

`index_only = bench_total(seed) - sort_est(seed) - cover(seed)`；sort_est 由同种子 A4 与 A4' 之差给出。
相减法误差约 ±3% 的总量，因此只报告符号与量级，不报告方法排名。

**4 km**

| 方法 | 索引工作 | 相对 GLOBAL |
|---|---:|---:|
| `GLOBAL_H3` | 635.1 | 1.000 |
| `FIXED_K8_QUERY_TRAIN` | 791.5 | 1.246 |
| `FIXED_K8_MOVEMENT_TRAIN` | 799.5 | 1.259 |
| `CAPACITY_512_128` | 825.2 | 1.299 |
| `FIXED_K8_ENTITY` | 925.8 | 1.458 |
| `FIXED_K8_UNIFORM` | 1055.5 | 1.662 |
| **A+ SLOT_PACKED_NOSORT（实测）** | **287.5** | **0.453** |

**20 km**

| 方法 | 索引工作 | 相对 GLOBAL |
|---|---:|---:|
| `GLOBAL_H3` | 74.7 | 1.000 |
| `FIXED_K8_UNIFORM` | 115.0 | 1.539 |
| `FIXED_K8_QUERY_TRAIN` | 173.0 | 2.316 |
| `FIXED_K8_ENTITY` | 179.3 | 2.400 |
| `FIXED_K8_MOVEMENT_TRAIN` | 191.2 | 2.560 |
| `CAPACITY_512_128` | 202.8 | 2.714 |
| **A+ SLOT_PACKED_NOSORT（实测）** | **37.0** | **0.495** |
