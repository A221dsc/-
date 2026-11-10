"""Replays M2 native and controlled traces through the same seven indexes."""
import argparse
import csv
import json
import platform
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import psutil
import pyarrow

from prepare import json_file, sha
from regional_workload import controlled
from run_m1 import MEASUREMENT, invoke, summarize, write_csv
from trace import write_trace

EXTRA = ["coarse_update_ms", "local_update_ms", "switch_ms", "local_updates", "switches"]
REGION_FIELDS = ("phase_index region_id entities mode query_count object_updates local_updates "
                 "local_builds switches query_ms local_update_ms local_build_ms switch_ms").split()


def create_trace(args, conf, profile, seed, split):
    synthetic = args.group != "tdrive"
    config = dict(conf)
    config["experiment_id"] = f"m2_{args.group}_{profile}"
    data = None
    if synthetic:
        config["dataset"] = "SYNTHETIC"
        config["dataset_variant"] = "SYNTHETIC_CONTROLLED"
        size = 2000 if args.group == "2k" else 20000
        data = controlled(config,seed,size,"reference" if profile == "switch" else profile,
                          diagnostic=profile == "switch")
    return write_trace(config,seed,split,args.source,args.traces,generated=data)


def run_measure(args, path, algorithm, param, profile=False):
    cmd = [str(args.binary.resolve()),"profile_m2" if profile else "measure_m2",
           algorithm,str(param),str(args.cpu),str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{cmd} failed: {result.stderr.strip()}")
    head,*lines = result.stdout.strip().splitlines()
    stats = dict(zip(MEASUREMENT+EXTRA,head.split(","),strict=True))
    stats["all_mutations_ms"] = stats["base_update_ms"]
    if stats["coarse_update_ms"] != "":
        stats["base_update_ms"] = stats["coarse_update_ms"]
    return stats, [dict(zip(REGION_FIELDS,line.split(","),strict=True)) for line in lines]


def make_row(args, name, meta, algorithm, param, truth, observed):
    checks,peak = map(int,truth)
    counts = meta["counts"]
    assert checks == int(observed["query_count"]) == counts["R"]+counts["C"]
    assert counts.get("U",0) == int(observed["update_count"])
    assert counts.get("I",0)+counts.get("D",0) == int(observed["insert_delete_count"])
    return {
        "group":args.group,"profile":name,"trace_id":meta["trace_id"],"seed":meta["seed"],
        "dataset_variant":meta["dataset_variant"],"baseline":algorithm,"parameter":param,
        "entity_count":peak,"initial_entities":meta["initial_entities"],"steps":meta["steps"],
        "trajectory_records":meta["trajectory_records"],"trace_sha256":meta["trace_sha256"],
        **observed,"correctness_pass":True,"profile_online_ms":"",
        "profile_query_ms":"","profile_mutations_ms":"",
        "total_with_build_ms":float(observed["build_ms"])+float(observed["total_wall_ms"]),
        "query_per_sec":checks*1000/float(observed["query_wall_ms"]),
        "exact_refinements":observed["candidate_count"],
    }


def write_environment(args):
    def version(*command):
        return subprocess.check_output(command,text=True).splitlines()[0]
    json_file(args.output/"environment.json", {
        "system":platform.platform(),"processor":platform.processor(),
        "physical_cores":psutil.cpu_count(False),"logical_cores":psutil.cpu_count(),
        "ram_bytes":psutil.virtual_memory().total,"python":platform.python_version(),
        "pyarrow":pyarrow.__version__,"compiler":version("g++","--version"),
        "flags":"-std=c++20 -O3 -DNDEBUG","cpu":args.cpu,"threads":1,
        "deps":{"boost":"1.82.0","libspatialindex":"1.9.3"},
        "git_commit":version("git","rev-parse","HEAD"),"binary_sha256":sha(args.binary),
        "config_sha256":sha(args.output/"config.json"),"input_config_sha256":sha(args.config),
        "cache_state":"warm process memory input; no OS cache flush",
        "region_profile":"separate execution; all inner region timers except explicit switch timing disabled in primary ranking",
        "files_sha256":{p.as_posix():sha(p) for d in ("scripts","src","tests")
                        for p in sorted(Path(d).rglob("*"))
                        if p.is_file() and "__pycache__" not in p.parts},
    })


def tune(args, config, profile, manifest):
    path,meta = create_trace(args,config,profile,config["validation_seed"],"validation")
    manifest.append(meta)
    trials = [(algo,param) for algo,params in config["parameter_search"].items() for param in params]
    validated = {(algo,param):invoke(args.binary,"validate",algo,param,args.cpu,path)
                 for algo,param in trials}
    times = defaultdict(list)
    records = []
    for repeat in range(config["validation_repeats"]):
        random.Random(repeat ^ 0x3BBC).shuffle(trials)
        for algo,param in trials:
            observed,_ = run_measure(args,path,algo,param)
            records.append({"repeat":repeat,**make_row(args,profile,meta,algo,param,
                                                       validated[algo,param],observed)})
            times[algo,param].append(float(observed["total_wall_ms"]))
        print(f"{args.group} validation repeat {repeat+1} done",flush=True)
    write_csv(args.output/"tuning.csv",records)
    return {algo:min(params,key=lambda p:statistics.median(times[algo,p]))
            for algo,params in config["parameter_search"].items()}


def run_group(args, config, name, selected, manifest, regional_file, region_writer):
    rows = []
    seeds = config["test_seeds"] if name != "switch" else [config["test_seeds"][0]]
    for seed in seeds:
        path,meta = create_trace(args,config,name,seed,"test")
        manifest.append(meta)
        order = list(selected)
        random.Random(seed ^ 0xFEBE).shuffle(order)
        for algo in order:
            param = selected[algo]
            truth = invoke(args.binary,"validate",algo,param,args.cpu,path)
            measured,_ = run_measure(args,path,algo,param)
            row = make_row(args,name,meta,algo,param,truth,measured)
            rows.append(row)
            if algo.startswith("S"):
                profiled,regions = run_measure(args,path,algo,param,profile=True)
                for field in ("query_count","candidate_count","result_count","switches"):
                    assert measured[field] == profiled[field]
                assert sum(int(r["local_updates"]) for r in regions if r["phase_index"] != "-1") == int(profiled["local_updates"])
                assert sum(int(r["switches"]) for r in regions) == int(measured["switches"])
                if algo == "S0_ALL_SCAN" and name != "switch":
                    assert int(profiled["local_updates"]) == float(profiled["local_update_ms"]) == 0
                for field in ("base_update_ms","coarse_update_ms","local_update_ms","local_updates"):
                    row[field] = profiled[field]
                row["profile_online_ms"] = profiled["total_wall_ms"]
                row["profile_query_ms"] = profiled["query_wall_ms"]
                row["profile_mutations_ms"] = profiled["all_mutations_ms"]
                for r in regions:
                    phase = int(r["phase_index"])
                    seconds = meta.get("phase_window_seconds",meta["steps"]*60) if phase >= 0 else 0
                    labels = meta.get("phase_names",["native"])
                    if phase == -1:
                        # Full local initial build time includes all initial insertions.
                        r["local_build_ms"] = r["local_update_ms"]
                        r["local_updates"] = r["object_updates"] = 0
                        r["local_update_ms"] = 0
                    region_writer.writerow({
                        "group":args.group,"profile":name,"seed":seed,"baseline":algo,
                        "trace_id":meta["trace_id"],
                        "phase":labels[phase] if phase >= 0 else "initial_build",
                        "window_seconds":seconds,
                        "queries_per_replay_second":int(r["query_count"])/seconds if seconds else "",
                        **r,
                    })
                regional_file.flush()
        print(f"{args.group} {name} seed {seed} passed {len(order)} algorithms",flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("group", choices=["tdrive","2k","20k"])
    p.add_argument("config",type=Path)
    p.add_argument("binary",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--source",type=Path)
    p.add_argument("--traces",type=Path,default=Path("data/processed/m2_traces"))
    p.add_argument("--cpu",type=int,default=min(psutil.Process().cpu_affinity()))
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.group == "tdrive":
        metadata = json.loads((args.source/"metadata.json").read_text(encoding="utf-8"))
        assert metadata["dataset"] == "TDRIVE"
        for split in ("validation","test"):
            assert sha(args.source/f"{split}.csv") == metadata["output_sha256"][split]
        native = json.loads(Path("configs/m1_tdrive.json").read_text(encoding="utf-8"))
        config = {**config,**{k:v for k,v in native.items()
                              if k not in ("parameter_search","validation_seed","test_seeds")}}
    args.output.mkdir(parents=True)
    json_file(args.output/"config.json",config)
    write_environment(args)
    manifest = []
    main_profile = "native" if args.group == "tdrive" else "reference"
    selected = tune(args,config,main_profile,manifest)
    json_file(args.output/"selected.json",selected)
    profiles = list(config["profiles"]) if args.group == "20k" else [main_profile]
    metrics = []
    fields = ("group profile seed baseline trace_id phase window_seconds queries_per_replay_second"
              .split()+REGION_FIELDS)
    with (args.output/"regions.csv").open("w",newline="",encoding="utf-8") as regional_file:
        writer = csv.DictWriter(regional_file,fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            rows = run_group(args,config,profile,selected,manifest,regional_file,writer)
            metrics.extend(rows)
            write_csv(args.output/"metrics.csv",metrics)
            summarize(rows,args.output/f"summary_{profile}.json")
            json_file(args.output/"trace_manifest.json",manifest)
        if args.group == "2k":
            switches = run_group(args,config,"switch",{"S0_ALL_SCAN":1},manifest,regional_file,writer)
            assert int(switches[0]["switches"]) == len(config["cluster_regions_xy"])*switches[0]["steps"]
            assert float(switches[0]["switch_ms"]) > 0
            write_csv(args.output/"switch_diagnostic.csv",switches)
            json_file(args.output/"trace_manifest.json",manifest)


if __name__ == "__main__":
    main()
