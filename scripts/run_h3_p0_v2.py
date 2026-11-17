"""Run H3-P0-v2 paired workloads, retain all runs, and record resource limits."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import io
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("A+Lock", "A+Static", "A+StaticMatched", "B+")
THREAD_COUNTS = (1, 4, 8, 16, 32)
SEEDS = (12011, 14447, 17021, 19531, 21961, 24407, 26777, 29303)
SENTINEL_SEED = 8419


def command(args: list[str], timeout: int = 180) -> str:
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, timeout=timeout)
    stdout = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode:
        diagnostic = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"{args!r} exited {proc.returncode}: {diagnostic}\n{stdout}")
    return stdout


def powershell(script: str) -> str:
    prefix = "[Console]::OutputEncoding = [Text.UTF8Encoding]::new(); "
    return command(["powershell", "-NoProfile", "-Command", prefix + script], timeout=20)


def powershell_optional(script: str) -> str:
    """Environment probes must not abort a measurement run.

    WMI/CIM is denied in some sandboxes; record the failure instead of dying.
    """
    try:
        return powershell(script).strip()
    except Exception as error:  # noqa: BLE001 - probe robustness
        first = str(error).splitlines()[0] if str(error) else type(error).__name__
        return f"unavailable ({first[:160]})"


def windows_topology() -> dict:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    process = kernel.GetCurrentProcess()
    mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(ctypes.c_void_p(process), ctypes.byref(mask),
                                         ctypes.byref(system_mask)):
        raise ctypes.WinError(ctypes.get_last_error())
    size = ctypes.c_ulong(0)
    query = kernel.GetSystemCpuSetInformation
    query(None, 0, ctypes.byref(size), ctypes.c_void_p(process), 0)
    buffer = ctypes.create_string_buffer(size.value)
    if not query(buffer, size, ctypes.byref(size), ctypes.c_void_p(process), 0):
        raise ctypes.WinError(ctypes.get_last_error())
    records = []
    cursor = 0
    while cursor < size.value:
        length = int.from_bytes(buffer[cursor:cursor + 4], "little")
        kind = int.from_bytes(buffer[cursor + 4:cursor + 8], "little")
        if kind == 0:
            group = int.from_bytes(buffer[cursor + 12:cursor + 14], "little")
            cpu, core, cache, node, efficiency, flags = buffer[cursor + 14:cursor + 20]
            if group == 0 and mask.value & (1 << cpu) and (not flags & 2 or flags & 4):
                records.append(dict(cpu=cpu, group=group, core=core, llc=cache, numa=node,
                                    efficiency_class=efficiency, parked=bool(flags & 1)))
        if length == 0:
            raise RuntimeError("zero length CPU-set record")
        cursor += length
    allowed = len(records) or bin(mask.value).count("1")
    return dict(allowed_logical=allowed, physical_cores=len({r["core"] for r in records}) or allowed,
                numa_nodes=sorted({r["numa"] for r in records}), affinity_mask=hex(mask.value),
                cpu_sets=records, pinning="OS placement within inherited process affinity",
                cim=powershell_optional(
                    "Get-CimInstance Win32_Processor | "
                    "Select Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,CurrentClockSpeed | "
                    "ConvertTo-Json -Compress"),
                power_scheme=powershell_optional("powercfg /getactivescheme"))


def linux_topology() -> dict:
    affinity = sorted(os.sched_getaffinity(0))
    sys_cpu = Path("/sys/devices/system/cpu")
    cores = set()
    nodes = set()
    for cpu in affinity:
        directory = sys_cpu / f"cpu{cpu}"
        topo = directory / "topology"
        cores.add(((topo / "physical_package_id").read_text().strip(),
                   (topo / "core_id").read_text().strip()))
        nodes.update(p.name for p in directory.glob("node*"))
    quota = Path("/sys/fs/cgroup/cpu.max")
    limit = len(affinity)
    if quota.exists():
        cap, period = quota.read_text().split()
        if cap != "max":
            limit = min(limit, max(1, int(cap) // int(period)))
    return dict(allowed_logical=limit, affinity=affinity, physical_cores=len(cores),
                numa_nodes=sorted(nodes), pinning="OS placement within inherited process affinity",
                lscpu=command(["lscpu"]))


def frequency_sample() -> str:
    if sys.platform == "win32":
        return powershell_optional(
            "Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation | "
            "Where-Object Name -eq '_Total' | "
            "Select ProcessorFrequency,PercentofMaximumFrequency,PercentProcessorPerformance,"
            "PercentProcessorUtility | ConvertTo-Json -Compress")
    frequencies = []
    for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_cur_freq"):
        frequencies.append(int(path.read_text().strip()))
    return json.dumps({"scaling_cur_freq_khz": frequencies})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--sanitizer-logs", type=Path, required=True,
                        help="directory holding the tsan/asan logs or the native gate log")
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_csv(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _clean(text: str) -> bool:
    return not any(word in text for word in
                   ("FAIL ", "WARNING: ThreadSanitizer", "AddressSanitizer: ",
                    "runtime error:", "Assertion", "munmap_chunk"))


def confirm_gate(folder: Path, source: str) -> str:
    """Certify the measured source fingerprint.

    Preferred: TSan + ASan logs produced from exactly this fingerprint.
    Fallback: the native debug-gate log (checked STL + 8/16/32-worker stress +
    exhaustive answer validation), used only where sanitizers cannot run.
    """
    sanitizer = True
    for mode in ("tsan", "asan-ubsan"):
        path = folder / f"{mode}.log"
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        if not (f"SOURCE_FINGERPRINT {source}" in text
                and "SANITIZER GATE PASS" in text
                and "PASS B+ workers=8" in text and _clean(text)):
            sanitizer = False
            break
    if sanitizer:
        return "tsan+asan"
    path = folder / "native-gate.log"
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if (f"SOURCE_FINGERPRINT {source}" in text and "NATIVE GATE PASS" in text
            and "PASS B+ workers=8" in text and _clean(text)):
        return "native-debug-stress"
    raise RuntimeError(
        "no gate certifies this source fingerprint "
        f"{source}: sanitizer logs are absent, stale, or failed; "
        f"run scripts/check_h3_p0_v2_native_gate.sh {folder}")


def run_suite(args: argparse.Namespace) -> None:
    binary = args.binary.resolve(strict=True)
    out = args.out.resolve()
    if (out / "timings.csv").exists():
        raise FileExistsError("timings.csv already exists; choose a new output directory")
    source_manifest = binary.parent / "source.manifest"
    source_hash = digest(source_manifest)
    binary_hash = digest(binary)
    if binary_hash != (binary.parent / "bench.sha256").read_text().strip():
        raise RuntimeError("binary differs from the reported build")
    gate = confirm_gate(args.sanitizer_logs, source_hash)
    out.mkdir(parents=True, exist_ok=True)
    topology = windows_topology() if sys.platform == "win32" else linux_topology()
    threads = [n for n in THREAD_COUNTS if n <= topology["allowed_logical"]]
    if not threads:
        raise RuntimeError("no permitted worker count")
    timer = "steady_clock"
    if sys.platform == "win32":
        frequency = ctypes.c_longlong()
        ctypes.WinDLL("kernel32").QueryPerformanceFrequency(ctypes.byref(frequency))
        timer = f"QueryPerformanceCounter ({frequency.value} Hz), high-resolution waitable timer for arrivals"
    meta = dict(host=platform.platform(), python=sys.version, topology=topology, threads=threads,
                skipped_threads=sorted(set(THREAD_COUNTS) - set(threads)), resolution=9,
                h3=(ROOT / "third_party/h3-4.1.0/VERSION").read_text().strip(),
                compiler=command([os.environ.get("CXX", "g++"), "--version"]),
                flags="-std=c++20 -O3 -DNDEBUG -pthread -Wall -Wextra",
                source_base=command(["git", "rev-parse", "HEAD"]).strip(),
                binary_sha256=binary_hash, source_sha256=source_hash, quick=args.quick,
                methods=list(METHODS), seeds=list(SEEDS[:1] if args.quick else SEEDS),
                timer=timer, gate=gate, worker_mode="persistent native threads",
                sentinel_rule="flag ratios outside [0.95, 1.05]; retain all observations")
    (out / "environment.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "source.manifest").write_bytes(source_manifest.read_bytes())
    with (out / "timings.csv").open("x", encoding="utf-8", newline="") as raw, \
         (out / "frequency.jsonl").open("x", encoding="utf-8") as samples:
        sequence = 0
        writer = None
        expected = {}
        entities, ticks, warmup = (4000, 12, 3) if args.quick else (32000, 72, 8)

        def run(kind, block, scenario, method, count, seed, *, period=0,
                profile=False, oracle=False, n=ticks, population=entities, warm=warmup):
            nonlocal sequence, writer
            sequence += 1
            prof = str(out / "profiles" / f"{sequence:04d}.csv") if profile else "-"
            if profile:
                (out / "profiles").mkdir(exist_ok=True)
            args = [str(binary), scenario, method, str(count), str(seed), str(population),
                    str(n), str(warm), str(period), prof, str(int(oracle))]
            lines = list(csv.DictReader(io.StringIO(command(args))))
            if len(lines) != 1:
                raise RuntimeError(f"unexpected result for {args!r}: {lines!r}")
            row = dict(sequence=sequence, experiment=kind, block=block, **lines[0])
            if writer is None:
                writer = csv.DictWriter(raw, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            raw.flush()
            key = (scenario, seed, population, n, warm)
            signature = tuple(row[key] for key in
                              ("queries", "updates", "candidate_cells", "bucket_lookups",
                               "candidates", "hits", "signature"))
            if signature != expected.setdefault(key, signature):
                raise RuntimeError(f"method-dependent query output in run {sequence}")
            return row

        def sentinel(block, position):
            frequency = frequency_sample()
            run(f"sentinel_{position}", block, "A", "A+Lock", min(4, max(threads)),
                SENTINEL_SEED, n=36, population=16000, warm=8)
            samples.write(json.dumps(dict(block=block, position=position, reading=frequency)) + "\n")
            samples.flush()

        seeds = SEEDS[:1] if args.quick else SEEDS
        print(f"workers={threads}; skipped={meta['skipped_threads']}; gate={gate}", flush=True)
        for block, seed in enumerate(seeds):
            sentinel(block, "before")
            order = [(s, t, m) for s in "ABC" for t in threads for m in METHODS]
            order += [(s, 1, "A+") for s in "ABC"]
            random.Random(seed).shuffle(order)
            for scenario, count, method in order:
                run("closed", block, scenario, method, count, seed)
            sentinel(block, "after")
            print(f"closed block {block + 1}/{len(seeds)} done", flush=True)

        primary_threads = min(8, max(threads))
        for scenario in "ABC":
            calibration = run("calibration", 0, scenario, "A+Lock", primary_threads, 3907)
            baseline = int(calibration["wall_us"]) / ticks
            # Scheduled arrival timestamps do not depend on an individual method's progress.
            periods = [max(100, math.ceil(f * baseline / 100) * 100) for f in (.60, 1.60)]
            for block, seed in enumerate(seeds[:3]):
                order = [(period, method) for period in periods for method in METHODS]
                random.Random(seed).shuffle(order)
                for period, method in order:
                    run("open", block, scenario, method, primary_threads, seed, period=period)
            configs = [(t, m) for t in threads for m in METHODS]
            random.Random(9000 + ord(scenario)).shuffle(configs)
            for count, method in configs:
                run("profile", 0, scenario, method, count, seeds[0], profile=True)
            print(f"W8-{scenario} open loop and profiles done", flush=True)

        for scenario in "ABC":
            for method in METHODS:
                run("oracle", 0, scenario, method, primary_threads, seeds[0],
                    oracle=True, n=9, population=4000, warm=3)
        print(f"retained {sequence} runs; all matched work and result signatures", flush=True)
    (out / "complete.json").write_text(json.dumps(dict(runs=sequence,
        timings_sha256=digest_csv(out / "timings.csv")), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_suite(parse_args())
