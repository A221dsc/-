# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | sanity: cover/assignment/mapping | A/B/C/D/E/F/G | 2 maps, 1 seed | coverage, empty regions, fanout | MUST | DONE | pilot runs without empty regions; partitions connected |
| R002 | M1 | baseline A/B | A_HASH, B_M_EQ_T_FIXED | 2 maps, 3 seeds | makespan, p95 | MUST | DONE | B ≫ A on 20 km; weaker on 4 km/T4 |
| R003 | M2 | fixed m>T vs m=T | C, D | 2 maps, 3 seeds | makespan, p99, messages | MUST | DONE | fixed m>T not robust; rejected |
| R004 | M2 | adaptive m + mapping | F, G | 2 maps, 3 seeds | oracle gap | MUST | DONE | F 3/4 better than B; gap 0.9–2.5% |
| R005 | M2 | anti-claim: more regions only | B, C, D | 2 maps, 3 seeds | D-C delta | MUST | DONE | mapping is necessary; fixed m>T often worse |
| R006 | M3 | decision | all | — | gate | MUST | DONE | H1 rejected; H1' conditional pass |
| R007 | M3 | msg cost sensitivity | A/B/F | 2 maps, 3 seeds | makespan vs msg cost | MUST | TODO | next |
| R010 | M0 | **worker-sharded correctness** | WorkerShardedIndex | 2 maps, 7 configs | mismatch, invariants, parallel read | MUST | **DONE** | 0 mismatch; see `CORRECTNESS_RESULTS.md` |
| R008 | B2 | real threads | A/B/F | conditional | wall clock | MUST | BLOCKED | after M3 |
| R009 | B3 | forecast error | F/B/G | conditional | gap vs error | NICE | BLOCKED | |
| R011 | B2 | parallel writes / migration protocol | WorkerShardedIndex | conditional | correctness, locks | MUST | TODO | not implemented |
