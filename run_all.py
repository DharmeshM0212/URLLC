import os
import json
import pprint   
from experiments import (
    run_baseline_iid, compare_tdma_vs_best_effort_iid, compare_fec_iid,
    ge_dualpath_variants, sweep_min_overhead, plot_tradeoffs, end_to_end_demo
)

def main():
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)

    print("== Step 1: Baseline (IID) ==")
    base = run_baseline_iid(outdir)
    pprint.pprint(base)

    print("\n== Step 2: TDMA vs Best-effort (IID) ==")
    rows2 = compare_tdma_vs_best_effort_iid(outdir)
    for r in rows2: print(r["config"], r["deadline_success"], r["p99_ms"])

    print("\n== Step 3: FEC on IID ==")
    rows3 = compare_fec_iid(outdir)
    for r in rows3: print(r["config"], r["deadline_success"], r["p99_ms"], r["overhead"])

    print("\n== Step 4: GE burst + Dual-path ==")
    rows4 = ge_dualpath_variants(outdir)
    for r in rows4: print(r["config"], r["deadline_success"], r["p99_ms"], r["overhead"])

    print("\n== Steps 5 & 6: Sweep + Tradeoff Plots ==")
    results, feasible, best = sweep_min_overhead(outdir, target_success=0.999, target_p99_ms=20.0)
    print("Feasible count:", len(feasible))
    if best:
        print("Recommended:", best["config"], "| overhead", best["overhead"], "| success", best["deadline_success"], "| p99", best["p99_ms"])
    plot_tradeoffs(outdir, results, target_p99_ms=20.0)

    print("\n== Step 7: End-to-end DSP + Network Demo ==")
    rows7 = end_to_end_demo(outdir, n_runs=1000)
    for r in rows7:
        print(r["config"], "| overhead", r["overhead"], "| det_p95", r["det_p95_ms"], "| net_p99", r["net_p99_ms"], "| e2e success", r["total_success"])

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(dict(
            baseline=base,
            best_config=best if best else None,
            end_to_end=rows7
        ), f, indent=2)

if __name__ == "__main__":
    main()
