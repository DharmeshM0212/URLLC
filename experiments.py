from typing import List, Tuple, Optional, Dict
import numpy as np
import os
import csv

from core import (
    Producer, ProducerParams, Scheduler, SchedulerParams, FECEncoder, FECParams,
    IIDParams, GEChannelParams, GEParams, SimParams, run_config, DEFAULT_SEED
)


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def save_csv(path: str, rows: List[Dict]):
    if not rows: return
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def append_csv(path: str, rows: List[Dict]):
    if not rows: return
    ensure_dir(os.path.dirname(path))
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)

def run_baseline_iid(outdir: str):
    sim = SimParams(total_time_s=60.0, deadline_ms=20.0, channel_mode="iid")
    prod = Producer(ProducerParams(rate_hz=20.0, payload_bytes=64))
    sched = Scheduler(SchedulerParams(mode="best_effort"))
    fec = FECEncoder(FECParams(n=1, k=1, payload_bytes=64, paths=1))
    iidp = IIDParams()
    m = run_config(sim, prod, sched, fec, iid_params=iidp, seed=DEFAULT_SEED)
    row = dict(config="IID_baseline_best_effort_n1k1", **m)
    save_csv(os.path.join(outdir, "baseline_iid.csv"), [row])
    return row

def compare_tdma_vs_best_effort_iid(outdir: str):
    rows = []
    sim = SimParams(total_time_s=60.0, deadline_ms=20.0, channel_mode="iid")
    prod = Producer(ProducerParams(rate_hz=20.0, payload_bytes=64))
    for mode in ["best_effort", "tdma"]:
        sched = Scheduler(SchedulerParams(mode=mode))
        fec = FECEncoder(FECParams(n=1, k=1, payload_bytes=64, paths=1))
        m = run_config(sim, prod, sched, fec, iid_params=IIDParams(), seed=DEFAULT_SEED+1)
        rows.append(dict(config=f"IID_{mode}_n1k1", **m))
    save_csv(os.path.join(outdir, "tdma_vs_be_iid.csv"), rows)
    return rows

def compare_fec_iid(outdir: str):
    rows = []
    sim = SimParams(total_time_s=60.0, deadline_ms=20.0, channel_mode="iid")
    prod = Producer(ProducerParams(rate_hz=20.0, payload_bytes=64))
    for (mode, n, k) in [("best_effort",1,1),("best_effort",3,1),("tdma",1,1),("tdma",3,1),("tdma",4,2)]:
        sched = Scheduler(SchedulerParams(mode=mode))
        fec = FECEncoder(FECParams(n=n, k=k, payload_bytes=64, paths=1))
        m = run_config(sim, prod, sched, fec, iid_params=IIDParams(), seed=DEFAULT_SEED+2)
        rows.append(dict(config=f"IID_{mode}_n{n}k{k}", **m))
    save_csv(os.path.join(outdir, "fec_iid.csv"), rows)
    return rows

def ge_dualpath_variants(outdir: str):
    rows = []
    sim = SimParams(total_time_s=60.0, deadline_ms=20.0, channel_mode="ge")
    prod = Producer(ProducerParams(rate_hz=20.0, payload_bytes=64))
    gep = GEChannelParams(ge=GEParams())
    for (mode, paths, n, k) in [
        ("best_effort",1,3,1),
        ("best_effort",2,3,1),
        ("tdma",1,3,1),
        ("tdma",2,3,1),
        ("tdma",2,4,2),
    ]:
        sched = Scheduler(SchedulerParams(mode=mode))
        fec = FECEncoder(FECParams(n=n, k=k, payload_bytes=64, paths=paths))
        m = run_config(sim, prod, sched, fec, ge_params=gep, seed=DEFAULT_SEED+3)
        rows.append(dict(config=f"GE_{mode}_paths{paths}_n{n}k{k}", **m))
    save_csv(os.path.join(outdir, "ge_dualpath.csv"), rows)
    return rows

def sweep_min_overhead(outdir: str, target_success=0.999, target_p99_ms=20.0):
    results = []
    sim = SimParams(total_time_s=60.0, deadline_ms=20.0, channel_mode="ge")
    prod = Producer(ProducerParams(rate_hz=20.0, payload_bytes=64))
    gep = GEChannelParams(ge=GEParams())
    grid_modes = ["best_effort", "tdma"]
    grid_paths = [1, 2]
    grid_nk = [(1,1),(2,1),(3,1),(3,2),(4,2),(5,2),(5,3),(6,3)]
    for mode in grid_modes:
        for paths in grid_paths:
            for (n,k) in grid_nk:
                sched = Scheduler(SchedulerParams(mode=mode))
                fec = FECEncoder(FECParams(n=n, k=k, payload_bytes=64, paths=paths))
                m = run_config(sim, prod, sched, fec, ge_params=gep, seed=DEFAULT_SEED+4)
                mrow = dict(config=f"{mode}_p{paths}_n{n}k{k}", mode=mode, paths=paths, n=n, k=k, **m)
                results.append(mrow)

    
    save_csv(os.path.join(outdir, "sweep_raw.csv"), results)

    
    feasible = [r for r in results if (r["deadline_success"] >= target_success and r["p99_ms"] <= target_p99_ms)]
    feasible.sort(key=lambda x: (x["overhead"], -x["deadline_success"], x["p99_ms"]))
    best = feasible[0] if feasible else None

    
    if best:
        save_csv(os.path.join(outdir, "sweep_best.csv"), [best])
    save_csv(os.path.join(outdir, "sweep_feasible.csv"), feasible)
    return results, feasible, best


def plot_tradeoffs(outdir: str, results: List[Dict], target_p99_ms=20.0):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    
    xs = [r["overhead"] for r in results]
    ys = [r["p99_ms"] for r in results]
    ensure_dir(outdir)
    plt.figure()
    plt.scatter(xs, ys)
    plt.axhline(target_p99_ms)
    plt.xlabel("Bandwidth Overhead (≈ n/k)")
    plt.ylabel("P99 Latency (ms)")
    plt.title("P99 vs Overhead (all configs)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "p99_vs_overhead.png"), dpi=160)

    
    baseline = [r for r in results if r["config"]=="best_effort_p1_n1k1"]
    if not baseline:
        
        for r in results:
            if r["mode"]=="best_effort" and r["paths"]==1 and r["n"]==1 and r["k"]==1:
                baseline = [r]; break
    if baseline:
        base_fail = 1.0 - baseline[0]["deadline_success"]
        ovs = []
        for r in results:
            if base_fail > 0:
                ovs.append(max(0.0, 100.0 * (base_fail - (1.0 - r["deadline_success"])) / base_fail))
            else:
                ovs.append(0.0)
        plt.figure()
        plt.scatter(xs, ovs)
        plt.xlabel("Bandwidth Overhead (≈ n/k)")
        plt.ylabel("Overshoot Reduction vs Baseline (%)")
        plt.title("Process Impact Proxy vs Overhead")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "overshoot_vs_overhead.png"), dpi=160)



def end_to_end_demo(outdir: str, n_runs=1000, det_p95_target_ms=40.0, net_deadline_ms=20.0, total_deadline_ms=50.0):
    from dsp import TraceParams, DetectorParams, simulate_trace_and_endpoint, detect_endpoint
    from core import transmit_one, GEChannel, GEChannelParams, GEParams

    rows = []
    
    setups = [
        dict(name="BASE (best_effort,1path,n1k1)", mode="best_effort", paths=1, n=1, k=1),
        dict(name="ROBUST (tdma,2paths,n4k2)",     mode="tdma",       paths=2, n=4, k=2),
    ]

    
    for setup in setups:
        
        tp = TraceParams()
        dp = DetectorParams()

        
        sched = Scheduler(SchedulerParams(mode=setup["mode"]))
        ch1 = GEChannel(GEChannelParams(ge=GEParams()), DEFAULT_SEED+42)
        ch2 = GEChannel(GEChannelParams(ge=GEParams()), DEFAULT_SEED+99) if setup["paths"]==2 else None
        fec = FECEncoder(FECParams(n=setup["n"], k=setup["k"], payload_bytes=64, paths=setup["paths"]))

        det_delays, net_lats, total_ok = [], [], 0
        det_ok, net_ok = 0, 0

        for r in range(n_runs):
            t, x, ep = simulate_trace_and_endpoint(tp, rng_seed=DEFAULT_SEED + r)
            det_t = detect_endpoint(t, x, dp)
            if det_t is None:
                continue
            det_delay = max(0.0, det_t - ep)
            det_ok += 1
            det_delays.append(det_delay)

            ok_net, net_lat, _, _ = transmit_one(det_t, fec, sched, ch1, ch2, deadline_ms=net_deadline_ms)
            if ok_net:
                net_ok += 1
                net_lats.append(net_lat)

            if (ok_net and (det_delay + net_lat) <= total_deadline_ms):
                total_ok += 1

        row = dict(
            config=setup["name"],
            overhead=fec.overhead,
            det_success = det_ok / n_runs,
            det_p95_ms = float(np.percentile(det_delays, 95)) if det_delays else float("nan"),
            det_p99_ms = float(np.percentile(det_delays, 99)) if det_delays else float("nan"),
            net_success = (net_ok / max(1, det_ok)),
            net_p95_ms = float(np.percentile(net_lats, 95)) if net_lats else float("nan"),
            net_p99_ms = float(np.percentile(net_lats, 99)) if net_lats else float("nan"),
            total_success = total_ok / n_runs
        )
        rows.append(row)

    save_csv(os.path.join(outdir, "end_to_end.csv"), rows)
    return rows
