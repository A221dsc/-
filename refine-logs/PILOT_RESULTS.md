# H3-RTM Pilot Results (Block B1)

**Date**: 2026-11-16
**Status**: execution-model pilot complete
**Data**: `results/h3_rtm/pilot.csv`, `results/h3_rtm/pilot.log`
**Code**: `tests/h3_rtm_pilot.cpp`
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`

## 1. What was run

- 4 km × 4 km / H3 res 9 / 241 cells / 20k entities / clustered。
- 20 km × 20 km / H3 res 10 / 34,429 cells / 20k entities / uniform。
- 5,000 training queries + 5,000 test queries + 5,000 updates per map/seed。
- 3 seeds（101/202/303），4/8 workers。
- Methods:
  - **A_HASH**: 逻辑全局 H3 索引，按 cell hash 分片到 workers。
  - **B_M_EQ_T_FIXED**: \(m=T\) 连通空间分区，固定映射。
  - **C_Mx_FIXED**: \(m=1.25T/1.5T/2T\)，固定 round-robin 映射。
  - **D_Mx_FAST**: 同上，用 training 负载 + fanout-aware 贪心/局部搜索映射。
  - **F_ADAPTIVE_TRAIN**: 在 \(\{T,1.25T,1.5T,2T\}\) 中用 **training makespan** 选 \(m\) 和映射；不偷看 test。
  - **G_ADAPTIVE_ORACLE**: 用 test 选择 \(m\) 和 oracle 映射，仅作上界。
- Execution model:
  - 查询按候选 H3 单元落到区域；每个区域子任务在映射 worker 上执行；
  - 子任务成本 = `region_overhead + cell_cost × cells`；
  - 跨 worker 子任务加 `msg_cost`，多区域加 `merge_cost`；
  - `cell_cost` 用 P0 全局实测标定（4 km 3.48 µs/cell，20 km 0.0936 µs/cell）；
  - `msg_cost=5 µs`，`merge_cost=1 µs`，`region_overhead=0.5 µs`。
  - **这是 execution-model simulation，不是实际线程墙钟。**

## 2. Main results（3-seed mean makespan，µs）

| Map | T | A_HASH | B_M_EQ_T | F_ADAPTIVE | G_ORACLE | F vs B | B vs A | F vs G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 km | 4 | 212,084 | 207,396 | **186,152** | 184,515 | **−10.2%** | −2.2% | +0.9% |
| 4 km | 8 | 238,276 | 147,341 | **140,081** | 141,315 | **−4.9%** | −38.2% | −0.9% |
| 20 km | 4 | 121,450 | 35,078 | **34,250** | 33,405 | **−2.4%** | −71.1% | +2.5% |
| 20 km | 8 | 215,246 | **22,578** | 22,794 | 22,578 | +1.0% | −89.5% | +1.0% |

Mechanism metrics（3-seed mean）：

| Map | T | Method | avg fanout | cross messages | imbalance |
|---|---:|---|---:|---:|---:|
| 4 km | 8 | A_HASH | 6.77 | 32,209 | 0.485 |
| 4 km | 8 | B_M_EQ_T | 2.49 | 14,882 | 0.917 |
| 4 km | 8 | F_ADAPTIVE | 2.69 | 16,871 | **0.195** |
| 20 km | 8 | A_HASH | 7.74 | 37,939 | 0.049 |
| 20 km | 8 | B_M_EQ_T | **1.30** | **2,959** | 0.479 |
| 20 km | 8 | F_ADAPTIVE | 1.30 | 3,046 | 0.479 |

## 3. Findings

### F1: Hash-sharded global H3 is a weak execution baseline

A_HASH has near-perfect load balance but high fanout：

- 20 km / 8 workers：A fanout 7.74，B/F 1.30；A messages 37,939，B 2,959；A makespan 215 ms vs B 22.6 ms。
- 4 km / 8 workers：A fanout 6.77，B 2.49；A 238 ms vs B 147 ms。

**解释**：空间查询/交互在 hash 分片下被打散到多个 worker；空间管理分区把邻近 cell 放在同一区域，降低 fanout 和跨 worker 消息。这是当前最稳健的效应。

### F2: 固定 \(m>T\) 不是普适的；H1 原假设被否定

- 4 km / 4 workers：B 207 ms；固定 \(m=2T\) 的 D 183 ms，但固定 \(m=1.25T\) 的 C 185 ms；差异不稳定。
- 20 km / 8 workers：B 22.6 ms；固定 \(m=2T\) 的 C/D 25.5–31.9 ms，明显更慢。
- 4 km / 8 workers：C 固定 \(m>T\) 142–147 ms，B 140 ms；固定细分没有稳定收益。

**结论**：H1（“\(m>T\) 一定更好”）不成立。**分区粒度必须随负载/地图选择**，不能固定为 \(T\) 或 \(2T\)。

### F3: 自适应 \(m\) + fanout-aware 映射有稳定小收益

F_ADAPTIVE 在 3/4 设置优于 B：

- 4 km / 4：−10.2%
- 4 km / 8：−4.9%
- 20 km / 4：−2.4%
- 20 km / 8：+1.0%（基本持平）

F 相对 oracle G 的遗憾为 0.9%–2.5%（4 km / 8 的负值来自 oracle 映射启发式不完美，不作为收益）。

**结论**：H1'（“自适应选择 \(m\) + 映射”优于固定 \(m=T\)）在 pilot 中成立，但收益中等，且 20 km / 8 基本持平。

### F4: 映射是 \(m>T\) 的必要条件

固定 \(m>T\) 的 C 经常比 B 差，而使用 training fanout-aware 映射的 D/F 可以追平或超过 B。说明：

> “细分”本身没有收益；收益来自 **细分 + 把共现区域放到同一 worker**。

## 4. Decision

| Gate | Result |
|---|---|
| 正确性/可运行性 | 通过（pilot 全流程跑通） |
| H1：固定 \(m>T\) 优于 \(m=T\) | **否决** |
| H1'：自适应 \(m\) + 映射优于固定 \(m=T\) | **初步通过（3/4 设置）** |
| C2：training selector 接近 oracle | 初步通过（遗憾约 1%–2.5%） |
| 是否进入 B2 真实线程 | **有条件进入**：先补 4/8/16 workers 和消息成本敏感性；B2 必须复现 F>B 与 B≫A |

## 5. Next experiments

1. **M3：粒度/消息成本敏感性**
   - \(m/T \in \{1.0,1.25,1.5,2.0\}\)；
   - `msg_cost` 1/5/20 µs；
   - 4/8/16 workers；
   - 检查 F 的收益是否只在 msg_cost=5 时成立。
2. **B2：真实线程级**
   - 实现真实 worker 队列/分区任务；
   - 比较 A/B/F 的墙钟、p99、锁等待；
   - 必须复现模拟排序，否则以真实结果为准。
3. **B3：预测误差**
   - 扰动 training 热点/迁移；
   - 检查 F 相对 B 的退化边界。
4. **真实想定**
   - 接现有想定/轨迹，至少一个未见地图。

## 6. Limitations

- execution-model simulation，不是真实线程墙钟；
- A_HASH 是分布式 hash 分片基线，不是单机全局 H3（单机 P0 已测）；
- 合成 workload，未接真实想定；
- mapping 使用 training 分布，test 独立但同生成过程；
- `msg_cost`、`merge_cost`、`region_overhead` 是显式参数，需要敏感性分析；
- oracle G 的映射启发式不完美，负遗憾不代表真实上界。
