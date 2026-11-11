# H0 索引与分区代码审计

2026-11-11。审计对象：`src/core/index.hpp`、`src/indexes/quad.cpp`、`grid.cpp`、`regional.cpp/.hpp`、`partition.cpp/.hpp`、`src/validation/oracle.cpp`、`scripts/m4_scenario.py`、`scripts/m4_partition_experiment.py` 及 M4.6/M4.7/M4.14 报告与结果。

## 1. 索引实现是否相同

| 项目 | 结论 |
| --- | --- |
| 全图 PR Quadtree | `make_quad(capacity, world)`，类 `Quad`（`quad.cpp`），桶容量分裂的四叉树 |
| 分区内 LOCAL_QUAD | `PartitionIndex::rebuild_leaf` 调用 `make_quad(quad_capacity_, leaf.box)`，**同一个 `Quad` 类**，只是边界换成本叶盒子 |
| 分区内 FINE_GRID | `make_grid(fine_cell_)`，与 PR Quadtree 不同实现（H0 不使用） |
| 旧 `AdaptiveRegionalIndex` | 16×16 固定网格 + 每区域模式，是另一套容器，不是 `PartitionIndex` |

**关键结论**：全图 PR Quadtree 与分区 LOCAL_QUAD 是同一实现、同一容量参数、同一分裂/合并规则；H0 的“同构分区”对照成立。旧 M4 的“分区 vs 全图”对比混用了 `AdaptiveRegionalIndex`、`PartitionIndex` 与 `Quad` 三种容器，不能直接归因于分区。

## 2. Quad 的动态维护语义

- `add`：节点条目超过容量 `cap` 且深度 < 28 时分裂为 4 子节点；同一坐标的重复点不分裂。
- `drop`：删除后若节点 `size <= cap/2` 则合并子节点（动态收缩）。
- `insert`：点落在根盒子外时扩张根盒子并重建整棵树（H0 的叶子盒子由分区覆盖，正常不触发）。
- `query`：按盒子相交剪枝；叶子节点逐条 `refine`（计算候选与命中）。
- `nodes()`（H0 新增）：统计树的节点数，用于内存对照。

## 3. PartitionIndex 的路由开销

- 查询：查询盒 → 64×64 子格范围（最多 4096 次循环）→ `touched` 叶去重（线性查找）→ 每叶 `aux->query` → 拼接结果（叶间互斥，无需全局去重）。
- 更新：`leaf_of` 查路由表 → 旧叶 `ids`/`aux` 摘除 → 新叶 `ids`/`aux` 插入；**两侧叶子都要维护**。
- 实体记录：`objects_` 全局表（id → 点/叶/槽位）+ 每叶 `ids` 向量 + 每叶 `aux` 的 `points_` 映射。

**核心发现**：同一实体在 `objects_`、`ids`、`aux.points_` 中重复存储；每次更新同时维护三处。这是 H0 中 1×1 分区（475 ms）比直接 `Quad`（248.6 ms）慢 1.9 倍的主因，也是分区数增加后更新成本（380→513 ms）持续上升的原因。

## 4. 内存结构

- 分区元数据：每叶 `Box` + `level` + `ids` 向量 + `aux` 指针 + 模式。
- 路由表：64×64 `leaf_of_`（4 KB 级）。
- 索引：每叶 `Quad` 的节点（H0 实测 2970→4716 节点，随分区数增长 59%）。
- 重复：`objects_` 与 `aux.points_` 双份点记录。
- 进程峰值：9 条轨迹均为 80.1 MB，主要由 20,000 实体 + 24 万事件的轨迹文件与事件缓冲决定，索引差异被掩盖；真实索引内存需按节点数或隔离进程测量。

## 5. 构建与跨区移动

- 构建：`set_partition` → 设置各叶模式（静态模式下先建空 aux）→ 初始 20,000 实体逐个 `set`（叶查找 + ids + aux 插入）。
- 跨区移动：旧叶摘除（ids 收缩 + aux erase）+ 新叶插入（ids + aux set），没有事务与回滚。
- 冷却/预算：仅由计划 M 行调度器执行；H0 的静态模式不受预算限制（初始化即固定）。

## 6. 历史结论的混杂因素

1. **容器不同**：M4.6 的 `AdaptiveRegionalIndex`（固定 16×16 + 每区模式）与 `PartitionIndex`（子格叶 + 每叶 aux）不是同一容器，早期“自适应更差”同时包含容器差异。
2. **策略不同**：M4.7 做了同策略消融（adaptive 211.7/214.6 vs uniform 214.7），提示差异很小；但仍是不同容器与不同分区几何。
3. **模式选择**：M4.14 的异构对照（查询叶 QUAD、更新叶 SCAN）比较的是不同模式组合，不是“同索引不同分区”。
4. **缺少同构对照**：此前从未在“所有叶子同一种索引、只改分区”的条件下比较过不同分区粒度；H0 首次完成。

## 7. 审计结论

- 同构分区对照在实现上成立（同一 `Quad` 类）。
- 但 `PartitionIndex` 的簿记开销显著（重复存储 + 路由 + 两侧维护），任何分区收益必须先抵消这部分固定成本。
- 因此 H0 的正确解读是：**在现有 PartitionIndex 容器下**，分区没有带来收益；不能直接断言“空间分区”这一思想无效，但也不能在未优化容器前宣称有效。
- 后续若继续分区研究，第一优先是消除重复存储/路由开销（例如实体只存一份、叶内索引直接持有实体），否则分区收益会被容器成本淹没。
