# H3-RTM Correctness Results

**Date**: 2026-11-16
**Status**: correctness gate PASSED for the worker-sharded substrate
**Code**: `tests/h3_rtm_correctness.cpp`, `src/h3/h3_management.*`
**Log**: `results/h3_rtm/correctness.log`

## 1. What is being verified

`WorkerShardedIndex` is the execution substrate for H3-RTM:

- 固定 H3 分辨率；
- 每个 H3 cell 只属于一个连通管理区域；
- 每个管理区域只映射到一个 worker；
- 每个 worker 持有其区域内的 `H3Index → vector<EntityId>` 桶；
- 实体坐标只在全局 `points_` 中存一份；
- 查询按候选 H3 单元路由到 worker，读取桶，再用原始坐标精确过滤；
- 更新只移动实体 id，不复制坐标。

验证目标：

1. 每个实体恰好属于一个 worker 桶；
2. 每个 H3 单元都有 worker；
3. 插入、跨 worker 移动、删除重插后，查询结果与 `ArrayOracle` 完全一致；
4. 不重复、不遗漏；
5. 8 线程只读查询结果一致；
6. 不同 \(m/T\) 与不同映射下都成立。

## 2. 配置

| 地图 | H3 | workers | \(m\) | entities | queries | updates | parallel check |
|---|---|---:|---:|---:|---:|---:|---|
| 4 km | res 9 | 4 | 4, 6, 8 | 5,000 | 1,000 | 2,000 | — |
| 4 km | res 9 | 8 | 8, 12, 16 | 5,000 | 1,000 | 2,000 | 8 threads |
| 20 km | res 10 | 8 | 16 | 5,000 | 1,000 | 2,000 | 8 threads |

- 分区由 `build_connected_assignment` 生成，连通性由生成树切割保证。
- 映射使用非恒等映射 `((region*3)+1) mod workers`，避免只测“区域 i → worker i”。
- 每 100 次更新做一次全量 `validate` + 查询对照。
- 更新后再次跑全部 1,000 条查询。
- 8 线程并行查询在 T=8 配置上执行，结果与串行/ArrayOracle 对照。

## 3. Results

日志：

```text
4km/res9/t4/m1.000000: workers=4 regions=4 entities=5000 queries=1000 updates=2000 buckets=233 passed
4km/res9/t4/m1.500000: workers=4 regions=6 entities=5000 queries=1000 updates=2000 buckets=233 passed
4km/res9/t4/m2.000000: workers=4 regions=8 entities=5000 queries=1000 updates=2000 buckets=233 passed
4km/res9/t8/m1.000000: workers=8 regions=8 entities=5000 queries=1000 updates=2000 buckets=233 passed
4km/res9/t8/m1.500000: workers=8 regions=12 entities=5000 queries=1000 updates=2000 buckets=233 passed
4km/res9/t8/m2.000000: workers=8 regions=16 entities=5000 queries=1000 updates=2000 buckets=233 passed
20km/res10/t8/m2: workers=8 regions=16 entities=5000 queries=1000 updates=2000 buckets=4636 passed
H3-RTM worker-sharded correctness suite passed
```

**0 mismatch。**

## 4. What was checked

- `WorkerShardedIndex::validate`：
  - `cell_to_worker` 覆盖全部 H3 单元；
  - 所有 worker 的实体计数之和等于 `points_`；
  - 每个 id 只出现一次；
  - 每个实体的坐标重新计算 H3 cell 后，确实在对应 cell 桶中；
  - 所有点都被索引。
- 查询：
  - 初始 1,000 条；
  - 每 100 次更新后；
  - 全部更新后 1,000 条；
  - 8 线程并行只读查询；
  - 全部与 `ArrayOracle` 排序结果一致。
- 更新：
  - 同 worker / 跨 worker 移动；
  - 删除后重插；
  - 重复坐标（同一 cell 多实体）。

## 5. What is NOT yet verified

1. **慢尺度增量的分裂/合并 + worker 映射**：本次验证的是“固定分区 + 固定映射 + 动态实体”的 worker-sharded substrate；P0 的容量树分裂/合并是单索引测试，尚未与 worker 迁移联合测试。
2. **并发写**：只验证了并发只读查询；并发更新、锁、跨 worker 迁移协议尚未实现。
3. **真实线程级执行**：正确性验证是串行/并行查询，性能 pilot 是 execution-model simulation；真实任务队列、合并、锁等待尚未测。
4. **真实想定**：仍为合成 workload。
5. **全球/跨日期变更线**：只测局部投影地图。

## 6. Decision

- **正确性关卡通过**：H3-RTM 的 worker-sharded 查询/更新语义在 7 组配置、4 km/20 km、T=4/8、m/T=1.0–2.0 上均 0 mismatch。
- 可以进入 M3（粒度/消息成本敏感性）和 B2（真实线程级）实验。
- 但不能把“正确性通过”写成“性能有效”；性能仍以 pilot 的 execution-model 结果为准，需 B2 复现。

## 7. Reproduce

```bash
bash scripts/run_h3_rtm_pilot.sh
```

该脚本先跑 correctness，再跑 performance pilot。
