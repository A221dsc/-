"""H3-E0: per-tick makespan lower bound and gap decomposition for H3-P0-v2 logs.

Every quantity is computed PER TIME STEP and only then aggregated.  No
cross-step max/mean/CV is used as a bound.

For one tick t with K workers, let
    W_t  = sum_w (query_ns + update_ns - read_wait_ns - write_wait_ns)   (pure work)
    L_t  = sum_w (read_wait_ns + write_wait_ns)                          (lock wait)
    busy_w, execute_t, frame_t = prepare + dispatch + execute + merge + barrier

Work-conserving lower bound on the execution window:
    LB_exec_t = W_t / K
Exact additive decomposition of the execution window (identity, no fitting):
    execute_t - LB_exec_t = start_skew_t + tail_t + lock_t
      start_skew_t = execute_t - max_w busy_w      (workers not yet started)
      tail_t       = max_w busy_w - mean_w busy_w  (straggler beyond perfect balance)
      lock_t       = L_t / K                       (lock wait amortised over K)
because sum_w busy_w = W_t + L_t by construction.

Frame lower bound:  LB_frame_t = fixed_t + LB_exec_t
      fixed_t = prepare_t + dispatch_t + merge_t + barrier_t
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_profiles(batch: Path) -> list[dict]:
    timings = list(csv.DictReader((batch / "timings.csv").open(encoding="utf-8")))
    runs = [r for r in timings if r["experiment"] == "profile"]
    out = []
    for run in runs:
        path = batch / "profiles" / f"{int(run['sequence']):04d}.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        ticks: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            ticks[int(row["tick"])].append(row)
        out.append(dict(scenario=run["scenario"], method=run["method"],
                        workers=int(run["workers"]), seed=run["seed"], ticks=ticks))
    return out


def tick_terms(rows: list[dict]) -> dict:
    first = rows[0]
    k = len(rows)
    f = lambda r, name: float(r[name])
    work = sum(f(r, "query_ns") + f(r, "update_ns") - f(r, "read_wait_ns") - f(r, "write_wait_ns")
               for r in rows)
    lock = sum(f(r, "read_wait_ns") + f(r, "write_wait_ns") for r in rows)
    busy = [f(r, "busy_ns") for r in rows]
    execute = f(first, "execute_ns")
    fixed = f(first, "prepare_ns") + f(first, "dispatch_ns") + f(first, "merge_ns") + f(first, "barrier_tail_ns")
    frame = execute + fixed
    mean_busy = sum(busy) / k
    max_busy = max(busy)
    start_skew = execute - max_busy
    tail = max_busy - mean_busy
    lock_term = lock / k
    lb_exec = work / k
    return dict(k=k, work=work, lock=lock, execute=execute, fixed=fixed, frame=frame,
                lb_exec=lb_exec, lb_frame=fixed + lb_exec,
                start_skew=start_skew, tail=tail, lock_term=lock_term,
                identity_residual=(execute - lb_exec) - (start_skew + tail + lock_term),
                eta=lb_exec / execute if execute else 0.0,
                eta_frame=(fixed + lb_exec) / frame if frame else 0.0,
                tasks=sum(f(r, "query_tasks") + f(r, "update_tasks") for r in rows),
                candidates=sum(f(r, "candidates") for r in rows))


def summarize(entry: dict) -> dict:
    terms = [tick_terms(rows) for _, rows in sorted(entry["ticks"].items())]
    n = len(terms)
    s = lambda key: sum(t[key] for t in terms)
    frame = s("frame")
    return dict(scenario=entry["scenario"], method=entry["method"], workers=entry["workers"],
                ticks=n,
                frame_ms=frame / 1e6,
                execute_ms=s("execute") / 1e6,
                fixed_pct=100 * s("fixed") / frame,
                work_pct=100 * s("work") / frame,
                lb_frame_ms=s("lb_frame") / 1e6,
                eta_frame=s("lb_frame") / frame,
                gap_pct=100 * (1 - s("lb_frame") / frame),
                start_skew_pct=100 * s("start_skew") / frame,
                tail_pct=100 * s("tail") / frame,
                lock_pct=100 * s("lock_term") / frame,
                eta_exec=s("lb_exec") / s("execute"),
                eta_exec_median=statistics.median(t["eta"] for t in terms),
                eta_exec_min=min(t["eta"] for t in terms),
                identity_max_residual=max(abs(t["identity_residual"]) for t in terms),
                tasks_per_tick=s("tasks") / n,
                candidates_per_task=s("candidates") / max(1.0, s("tasks")))


def main(argv: list[str]) -> None:
    batch = Path(argv[1] if len(argv) > 1 else "results/h3_p0_v2/run-full")
    entries = load_profiles(batch)
    rows = [summarize(e) for e in entries]
    print(f"{'scn':5s} {'w':>2s} {'method':<16s} {'frame_ms':>8s} {'fixed%':>6s} {'etaFrame':>8s} "
          f"{'gap%':>5s} {'skew%':>6s} {'tail%':>6s} {'lock%':>6s} {'etaExec':>7s} {'min':>5s} "
          f"{'tasks/t':>7s} {'cand/task':>9s}")
    for r in sorted(rows, key=lambda x: (x["scenario"], x["workers"], x["method"])):
        print(f"{r['scenario']:5s} {r['workers']:2d} {r['method']:<16s} {r['frame_ms']:8.1f} "
              f"{r['fixed_pct']:6.1f} {r['eta_frame']:8.3f} {r['gap_pct']:5.1f} "
              f"{r['start_skew_pct']:6.1f} {r['tail_pct']:6.1f} {r['lock_pct']:6.1f} "
              f"{r['eta_exec']:7.3f} {r['eta_exec_min']:5.2f} {r['tasks_per_tick']:7.0f} "
              f"{r['candidates_per_task']:9.0f}")
        if r["identity_max_residual"] > 1.0:
            print(f"  WARNING identity residual {r['identity_max_residual']:.1f} ns")


if __name__ == "__main__":
    main(sys.argv)
