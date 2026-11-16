# Experiment Plan

**Problem**: 固定 H3 分辨率、统一实体索引、有限线程下，管理分区怎样组织、怎样映射到线程，才能降低作战仿真的端到端执行成本？
**Method Thesis**: 把“空间管理分区”和“分区→线程映射”解耦；分区粒度 \(m\) 不必等于线程数 \(T\)。用未来负载选择连通分区与映射，快尺度重映射、慢尺度重分区。
**Date**: 2026-11-16

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: 对固定 \(T\)，\(m>T\) 的连通区域 + 负载/交互感知映射，能比 \(m=T\) 固定映射和全局动态调度降低交互密集负载的 makespan/p99 | 这是“分区粒度≠线程数”的核心；若成立，现有“一区一线程”是次优 | 3 seeds、2 maps、4/8 threads；D 比 A/B/C 的 makespan 或 p99 低 ≥5%，且不是由 fanout 增加造成 | B1 |
| C2: 用训练/想定负载得到的映射，能保留 oracle 映射的大部分收益 | 证明“前馈可用”，而不是只能事后知道未来 | D 相对 E（oracle）的遗憾 ≤ 50%；预测误差增大时单调退化 | B1, B3 |
| Anti-claim: 收益只来自更多区域/更多元数据，而不是映射 | 防止“细分即收益”的假阳性 | C（m>T 固定映射）不优于 B/A；D 的收益必须在 C 之上 | B1 |

## Paper Storyline

- **Main paper must prove**: C1（粒度—映射解耦有效）和 C2（训练负载可近似 oracle）。
- **Appendix can support**: 参数敏感性、不同 \(m/T\)、不同 H3 分辨率、fanout/消息成本分解。
- **Experiments intentionally cut**: 真实线程级 PDES、GBDT/RL 预测、全球地图、跨日期变更线、复杂边界转移。只有 B1 出现正信号才启动 B2/B3。

## Experiment Blocks

### Block 1: 粒度—映射解耦的 execution-model pilot（MUST-RUN）

- **Claim tested**: C1、C2、anti-claim。
- **Why this block exists**: 在写真实线程调度器之前，用同一候选单元/分区数据，快速筛掉“分区粒度无关”或“映射无收益”的假设。
- **Dataset / task**:
  - 4 km × 4 km / H3 res 9 / 241 cells / 20k entities / clustered。
  - 20 km × 20 km / H3 res 10 / 34,429 cells / 20k entities / uniform。
  - 5,000 training queries + 5,000 test queries + 5,000 updates per map/seed。
  - 3 seeds（101/202/303），4/8 threads。
- **Compared systems**:
  - **A GLOBAL_DYNAMIC**: 全图 H3，查询为一个任务，动态分配到最空线程。
  - **B M_EQ_T_FIXED**: \(m=T\) 连通分区，固定映射 \(i\bmod T\)。
  - **C M_GT_T_FIXED**: \(m=4T\) 连通分区，固定 round-robin 映射（无智能映射）。
  - **D M_GT_T_FAST**: \(m=4T\)，用 training 查询/移动负载 + 交互矩阵做贪心/局部搜索映射。
  - **E M_GT_T_ORACLE**: \(m=4T\)，用 test 负载做 mapping（oracle 上界，不是在线方法）。
- **Execution model**:
  - 查询任务按候选 H3 单元分到其管理区域；每个区域子任务在映射线程上执行。
  - 子任务服务时间 = `region_overhead + cell_cost × 该区域候选单元数`。
  - 跨线程子任务增加 `msg_cost × (distinct_workers-1)`；多区域增加 `merge_cost × (regions-1)`。
  - 全局 A 的服务时间 = `cell_cost × 总候选单元数`，无区域 fanout。
  - 更新：同区域一个子任务；跨区域两个子任务 + msg cost。
  - `cell_cost` 用 P0 全局实测值标定：4 km ≈ 3.48 µs/candidate；20 km ≈ 0.094 µs/candidate。`msg/merge/region` 是显式参数，做敏感性分析。
  - 这是 **execution-model simulation**，不是实际线程墙钟；用于筛选假设，结论必须标注。
- **Metrics**:
  - makespan（主）、p50/p95/p99 latency、cross-worker messages、average/max region fanout、worker imbalance、mapping objective、D 相对 E 的遗憾。
- **Success criterion**:
  - D 在至少一个交互密集负载上比 A/B/C 的 makespan 低 ≥5%，p99 不退化 >5%；C 不优于 A/B；D 相对 E 遗憾 ≤50%。
  - 若 D 不优于 B/C，C1 不成立，停止 B2。
- **Failure interpretation**:
  - \(m>T\) 无收益：分区粒度应等于线程数；写负结果。
  - 只有 E 有收益：在线预测不可用；需改进预测或放弃前馈。
  - 只有 C 有收益：收益来自细分而非映射；退化为“区域粒度研究”，不是本方法。
- **Table / figure target**: 主表 Table 1（A–E × 2 maps × 4/8 threads）；图 1（m/T trade-off）。
- **Priority**: MUST-RUN.

### Block 2: 实际线程级调度（CONDITIONAL）

- **Claim tested**: B1 的模拟收益能否在真实线程/队列上复现。
- **Why**: execution-model 可能忽略锁、缓存、任务提交和合并开销。
- **Compared systems**: A、B、D、E；真实线程池 + 分区/全局任务队列。
- **Metrics**: 实际墙钟、p99、锁等待、上下文切换、缓存 miss（如可得）。
- **Success criterion**: D 的真实墙钟仍比 A/B 低 ≥5%，且模拟与实测排序一致。
- **Priority**: 只有 B1 通过才运行。

### Block 3: 预测误差与鲁棒性（CONDITIONAL）

- **Claim tested**: C2 在预测误差下的退化边界。
- **Setup**: 对 training 负载做扰动（热点位置/强度/迁移方向），误差 0/20/50%；比较 D、E、历史/反应式映射。
- **Success criterion**: D 在 20% 误差内仍优于 B；50% 误差下可退回反应式而不崩。
- **Priority**: 只有 B1/B2 通过才运行。

### Block 4: 真实想定回放（CONDITIONAL）

- **Claim tested**: 在现有想定/轨迹上是否成立。
- **Setup**: 接现有想定生成器/真实轨迹；至少一个未见地图。
- **Priority**: NICE-TO-HAVE，但论文前必须做。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 候选覆盖/分区/映射 sanity | 1 map, 1 seed, 1k ops | assignment 覆盖全部 cell；无空区域；无非法映射 | < 5 min CPU | 低 |
| M1 | 复现 A/B 基线 | 2 maps, 3 seeds | A/B 结果稳定；B 不出现异常加速 | < 20 min CPU | 低 |
| M2 | C/D/E 主实验 | 2 maps, 3 seeds, 4/8 threads | D vs A/B/C；D vs E | < 60 min CPU | 中 |
| M3 | 决策 | 汇总 | 通过则做 B2；否则写负结果 | < 10 min | 中 |

## Compute and Data Budget

- 全部 CPU；无 GPU。
- B1：约 2 maps × 3 seeds × 5 methods × 2 thread counts = 60 个 simulation run；每个 < 1 s，总计分钟级。
- 数据：现有合成 workload + P0 标定；真实想定回放后置。
- 最大瓶颈：不是算力，而是“模型是否代表真实执行”。

## Risks and Mitigations

- **模型≠真实线程**：B1 只做筛选；B2 必须真实线程复现。
- **H3 装饰性**：承认算法可迁移到其他网格；论文卖点是“粒度—映射解耦”，不是 H3 原生层级。
- **映射 solver 过强/过弱**：固定贪心+局部搜索，报告 objective 和 oracle gap；不做重型优化。
- **基线不公平**：A 用动态调度；B/C/D/E 同候选覆盖、同服务时间参数；报告 fanout/messages。
- **负结果**：C1 不成立时明确写“分区粒度应等于线程数”，停止后续。

## Final Checklist

- [ ] B1 覆盖 A/B/C/D/E 和 anti-claim
- [ ] C1/C2 有明确成功/失败阈值
- [ ] 模拟结论标注“execution-model”
- [ ] B2 只在 B1 通过后运行
- [ ] 真实想定回放列入条件计划

---

## Revision after M2 pilot（2026-11-16）

Pilot results: [PILOT_RESULTS.md](PILOT_RESULTS.md).

- 原 H1（“固定 \(m>T\) 优于 \(m=T\)”）**被否决**：20 km/8 下固定 \(m>T\) 更慢；4 km/T8 也没有稳定收益。
- 新 H1'（“自适应选择 \(m\) + fanout-aware 映射优于固定 \(m=T\)”）初步成立：F 在 3/4 设置优于 B（−10.2%、−4.9%、−2.4%、+1.0%）。
- 最稳健的效应是 **空间管理分区 vs hash 分片**：B 比 A 低 2.2%–89.5%（20 km 最明显），机制是 fanout/messages 从 7.74/37,939 降到 1.30/2,959。
- 下一步改为 M3：\(m/T\) 与 `msg_cost` 敏感性；通过后再做 B2 真实线程。

修订后的 claim：
- **C1'**：对固定 \(T\)，自适应选择 \(m\in[T,2T]\) 与映射，能比固定 \(m=T\) 降低 makespan/p99；不是“\(m>T\) 一定更好”。
- **C2'**：training 负载选择 \(m\)/映射可接近 oracle（pilot gap 0.9%–2.5%）。
- **C0**：hash 分片全局 H3 在空间 fanout 上明显劣于空间管理分区（pilot 强效应）。

---

## Correctness gate after R010（2026-11-16）

Results: [CORRECTNESS_RESULTS.md](CORRECTNESS_RESULTS.md).

- `WorkerShardedIndex` 在 4 km/res9（T=4/8，m=1.0/1.5/2.0）和 20 km/res10（T=8，m=2.0）共 7 组配置上通过。
- 5,000 实体、1,000 查询、2,000 更新；初始/更新后/8 线程并行查询均与 `ArrayOracle` 一致，**0 mismatch**。
- 已验证唯一归属、cell→worker 覆盖、worker 计数、跨 worker 移动、删除重插、并行只读。
- 未验证：慢尺度增量分裂/合并与 worker 迁移的联合、并发写、真实线程队列、真实想定。
- 因此 B1 正确性关卡通过，允许进入 M3；但性能结论仍必须由 B2 真实线程复核。
