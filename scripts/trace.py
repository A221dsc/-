"""Materialize the specification's event table and the equivalent C++ replay file."""
import csv
import json
import math
import random
from collections import Counter
from functools import lru_cache
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from prepare import json_file, sha

FIELDS = [
    ("trace_id",pa.string()), ("step",pa.int64()), ("timestamp_ms",pa.int64()),
    ("event_seq",pa.int64()), ("event_type",pa.string()), ("entity_id",pa.int64()),
    ("old_x_m",pa.float64()), ("old_y_m",pa.float64()), ("new_x_m",pa.float64()),
    ("new_y_m",pa.float64()), ("query_id",pa.int64()), ("query_type",pa.string()),
    ("q_params_json",pa.string()), ("source_label",pa.string()), ("rng_seed",pa.int64()),
]


def queries(config, rng, step, timestamp):
    x0,y0,x1,y1 = config["world_bounds_m"]
    area = (x1-x0)*(y1-y0)
    scales = config["query_area_fractions"]
    for i in range(config["queries_per_step"]):
        x,y = rng.uniform(x0,x1),rng.uniform(y0,y1)
        fraction = scales[(i//2) % len(scales)]
        if i % 2:
            yield step,timestamp,"C",None,(x,y,math.sqrt(area*fraction/math.pi))
        else:
            half = math.sqrt(area*fraction)/2
            yield step,timestamp,"R",None,(x-half,y-half,x+half,y+half)


def synthetic(config, seed):
    rng = random.Random(seed)
    query_rng = random.Random(seed ^ 0xA72B)
    x0,y0,x1,y1 = config["world_bounds_m"]
    current = {i: (rng.uniform(x0,x1),rng.uniform(y0,y1))
               for i in range(1,1+config["synthetic_entities"])}
    initial = {i:(0,xy) for i,xy in current.items()}
    events = []
    for step in range(config["synthetic_steps"]):
        stamp = 1000*(step+1)
        for ident in rng.sample(list(current),config["synthetic_movers_per_step"]):
            x,y = current[ident]
            current[ident] = (min(x1,max(x0,x+rng.gauss(0,250))),
                              min(y1,max(y0,y+rng.gauss(0,250))))
            events.append((step,stamp,"U",ident,current[ident]))
        if step % 6 == 0:
            ident = min(current)
            del current[ident]
            events.append((step,stamp,"D",ident,()))
            ident = config["synthetic_entities"]+1+step
            current[ident] = rng.uniform(x0,x1),rng.uniform(y0,y1)
            events.append((step,stamp,"I",ident,current[ident]))
        events.extend(queries(config,query_rng,step,stamp))
    return initial,events,config["synthetic_steps"],{"trajectory_records": 0}


@lru_cache(maxsize=2)
def read_native_window(source, split, dataset, warmup_stamp, start_stamp, end_stamp):
    tz = timezone(timedelta(hours=8 if dataset == "TDRIVE" else 0))

    def epoch(stamp):
        return int(datetime.fromisoformat(stamp).replace(tzinfo=tz).timestamp()) * 1000

    warmup,start,end = map(epoch, (warmup_stamp,start_stamp,end_stamp))
    initial, records = {}, []
    prefix = 0
    with (source / f"{split}.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            raw_stamp = row["timestamp"]
            if not warmup_stamp <= raw_stamp < end_stamp:
                continue
            stamp = epoch(raw_stamp)
            ident = int(row["entity_id"])
            point = float(row["x"]),float(row["y"])
            if stamp < start:
                prefix += 1
                if ident not in initial or stamp > initial[ident][0]:
                    initial[ident] = stamp,point
            else:
                records.append((stamp,ident,point))
    if not initial or not records:
        raise ValueError(f"{split} window has no initial snapshots or measured GPS samples")
    records.sort()
    return initial, records, prefix, start, end


def native(config, seed, source, split):
    initial, records, prefix, start, end = read_native_window(
        source, split, config["dataset"], *(config[k] for k in ("warmup_start","start","end"))
    )
    width = config["event_window_seconds"] * 1000
    if not (start < end and width > 0 and (end-start) % width == 0):
        raise ValueError("invalid window configuration")
    active = set(initial)
    events = []
    offset = 0
    rng = random.Random(seed ^ 0xA72B)
    steps = (end-start)//width
    for step in range(steps):
        limit = start+(step+1)*width
        while offset < len(records) and records[offset][0] < limit:
            stamp,ident,point = records[offset]
            events.append((step,stamp,"U" if ident in active else "I",ident,point))
            active.add(ident)
            offset += 1
        events.extend(queries(config,rng,step,limit-1))
    return initial,events,steps,{
        "trajectory_records": len(records), "warmup_trajectory_records": prefix,
        "presence_model": "latest prior sample initializes entity, last known position retained through this window",
    }


def write_trace(config, seed, split, source, out, generated=None):
    trace_id = f'{config["experiment_id"]}_{split}_{seed}'
    dest = out / trace_id
    dest.mkdir(parents=True, exist_ok=True)
    if generated is not None:
        initial,events,steps,meta = generated
    elif config["dataset"] == "SYNTHETIC":
        initial,events,steps,meta = synthetic(config,seed)
    else:
        initial,events,steps,meta = native(config,seed,source,split)

    current = {}
    rows = []
    query_count = 0
    counts = Counter()
    bounds = config["world_bounds_m"]

    def record(step, stamp, op, ident, point):
        nonlocal query_count
        counts[op] += 1
        old = current.get(ident)
        new = qid = qtype = params = None
        if op in ("R","C"):
            query_count += 1
            qid,qtype,params = query_count,("RECT" if op == "R" else "CIRCLE"),json.dumps(point)
        elif op in ("P","S"):
            params = json.dumps([ident,*point])
            ident = old = None
        elif op == "D":
            del current[ident]
        else:
            current[ident] = new = point
        rows.append((trace_id,step,stamp,len(rows),
                     {"B":"BUILD","U":"UPDATE","I":"INSERT","D":"DELETE","R":"QUERY","C":"QUERY","P":"PHASE_END","S":"MODE_SWITCH"}[op],
                     ident,*(old or (None,None)),*(new or (None,None)),qid,qtype,params,
                     "SYNTHETIC_QUERY" if qtype else config["dataset_variant"],seed))

    path = dest / "trace.txt"
    with path.open("w", encoding="ascii", newline="\n") as file:
        print("V2M1",*bounds,len(initial),len(events),file=file)
        for ident,(stamp,point) in sorted(initial.items()):
            print(ident,*point,file=file)
            record(-1,stamp,"B",ident,point)
        for step,stamp,op,ident,point in events:
            print(op,*(() if ident is None else (ident,)),*point,file=file)
            record(step,stamp,op,ident,point)
    parquet = dest / "events.parquet"
    pq.write_table(pa.Table.from_arrays(list(zip(*rows)),schema=pa.schema(FIELDS)),parquet)
    meta.update({
        "trace_id": trace_id, "seed": seed, "split": split, "steps": steps,
        "counts": dict(counts), "initial_entities": len(initial),
        "trace_sha256": sha(path), "events_sha256": sha(parquet),
        "query_source": meta.get("query_source","seeded synthetic RECT/CIRCLE, uniform world-space centers"),
        "dataset_variant": config["dataset_variant"],
    })
    json_file(dest / "trace.json",meta)
    return path,meta
