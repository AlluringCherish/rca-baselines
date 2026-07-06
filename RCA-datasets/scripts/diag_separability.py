"""Diagnose whether container fault types are separable by per-family SPECIFIC markers.

For each labeled (task, GT component, GT reason) compute candidate discriminative features:
  - thr_run / thr_peak : cpu_cfs_throttled sustained run length & peak (CPU-load specific marker)
  - mem_util          : memory_working_set / spec_memory_limit, sustained (memory-load marker)
  - read_fold/run     : fs_reads_MB sustained fold-change over baseline & run
  - write_fold/run    : fs_writes_MB sustained fold-change & run
  - cpuuse_fold       : cpu_usage_seconds sustained fold
Then print feature distributions grouped by GT reason -> see what separates.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/khmin/RCA/RCA-datasets/scripts")
from shape_signature import component_signals, shape_features
from pec_agent_planner_analyzer.tasks import load_task_bundles
from pec_agent_planner_analyzer.dataset_tools.zscore_filter import load_minute_metric

TARGET = {"container cpu load", "container memory load",
          "container read i/o load", "container write i/o load"}
CFGS = [("configs/dataset_openrca_market_cb1.yaml", "cb1"),
        ("configs/dataset_openrca_market_cb2.yaml", "cb2")]


def pod_sum_series(sub, substr):
    g = sub[sub.kpi_name.astype(str).str.lower().str.contains(substr, regex=False)]
    if g.empty:
        return None
    return g.groupby("timestamp").value.sum().sort_index()


def feat_run_fold(ser):
    if ser is None:
        return 0, 0.0, 0.0
    f = shape_features(ser.values)
    if f is None:
        return 0, 0.0, 0.0
    base = max(f["base"], 1e-9)
    fold = f["sustained_level"] / base if base > 0 else f["sustained_level"]
    return f["sustained_run"], float(f["sustained_level"] - f["base"]), float(fold)


def main():
    recs = []
    for cfg, sh in CFGS:
        for b in load_task_bundles(dataset_config=cfg):
            gts = [(a.get("reason") or "").lower() for a in b.ground_truth["answers"]]
            if not any(r in TARGET for r in gts):
                continue
            w = load_minute_metric(b.task.task_dir)
            for a in b.ground_truth["answers"]:
                r = (a.get("reason") or "").lower()
                if r not in TARGET:
                    continue
                sub = component_signals(w, a.get("component"))
                # cpu throttle
                thr = pod_sum_series(sub, "cpu_cfs_throttled_seconds")
                thr_run, thr_abs, _ = feat_run_fold(thr)
                thr_peak = float(thr.max()) if thr is not None else 0.0
                # cpu usage
                cu = pod_sum_series(sub, "cpu_usage_seconds")
                cu_run, _, cu_fold = feat_run_fold(cu)
                # memory util = working_set / spec_memory_limit
                ws = pod_sum_series(sub, "memory_working_set")
                lim = pod_sum_series(sub, "spec_memory_limit")
                mem_util = 0.0
                ws_run, ws_abs, ws_fold = feat_run_fold(ws)
                if ws is not None and lim is not None and lim.max() > 0:
                    mem_util = float(ws.max() / lim.max())
                # read / write MB
                rd = pod_sum_series(sub, "fs_reads_mb")
                rd_run, _, rd_fold = feat_run_fold(rd)
                wr = pod_sum_series(sub, "fs_writes_mb")
                wr_run, _, wr_fold = feat_run_fold(wr)
                recs.append(dict(sh=sh, idx=b.metadata["idx"], reason=r,
                                 thr_run=thr_run, thr_peak=thr_peak,
                                 cu_run=cu_run, cu_fold=cu_fold,
                                 ws_run=ws_run, ws_fold=ws_fold, mem_util=mem_util,
                                 rd_run=rd_run, rd_fold=rd_fold,
                                 wr_run=wr_run, wr_fold=wr_fold))
    df = pd.DataFrame(recs)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"n={len(df)}\n")
    cols = ["thr_run", "thr_peak", "cu_fold", "ws_fold", "mem_util", "rd_fold", "wr_fold"]
    print("=== MEDIAN feature by GT reason ===")
    print(df.groupby("reason")[cols].median().round(2))
    print("\n=== count of tasks with throttle sustained run>=3 (cpu marker) by reason ===")
    df["thr_sustained"] = df["thr_run"] >= 3
    print(df.groupby("reason")["thr_sustained"].agg(["sum", "count"]))
    df.to_csv("/tmp/sig_feats.csv", index=False)
    print("\nsaved /tmp/sig_feats.csv")


if __name__ == "__main__":
    main()
