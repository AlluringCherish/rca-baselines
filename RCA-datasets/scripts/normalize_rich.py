"""Re-normalize RCAeval RE2 from the RICH metrics.csv (434 cols) instead of the
reduced simple_metrics.csv (8 KPIs). The reduced version drops the metrics that
discriminate disk / network / socket faults; this restores them at SERVICE level
(to match the service-granular ground truth). Output -> telemetry/rcaeval_rich/
(originals untouched, for A/B).

Usage: python normalize_rich.py [split ...]   (default: ob ss tt)
"""
import os, sys, csv, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE_ROOT = Path("/home/khmin/RCA/Datasets_old/RCAEval/RE2")
DS = Path("/home/khmin/RCA/RCA-datasets")
TASKS = DS / "tasks" / "rcaeval"
OUT_ROOT = DS / "telemetry" / "rcaeval_rich"
ORIG_ROOT = DS / "telemetry" / "rcaeval"
SPLIT_DIR = {"ob": "RE2-OB", "ss": "RE2-SS", "tt": "RE2-TT"}
CAND = {  # service answer-candidates per split (container -> longest-prefix service)
    "ob": ["checkoutservice","currencyservice","emailservice","productcatalogservice","recommendationservice"],
    "ss": ["carts","catalogue","orders","payment","user"],
    "tt": ["ts-auth-service","ts-order-service","ts-route-service","ts-train-service","ts-travel-service"],
}
# raw metric suffix -> (clean kpi name, is_counter)
METRIC_MAP = {
    "container-cpu-usage-seconds-total": ("cpu", True),
    "container-memory-usage-bytes": ("mem", False),
    "container-fs-writes-bytes-total": ("disk_write", True),
    "container-fs-reads-bytes-total": ("disk_read", True),
    "container-network-receive-bytes-total": ("net_recv", True),
    "container-network-transmit-bytes-total": ("net_send", True),
    "container-network-receive-packets-dropped-total": ("net_recv_drop", True),
    "container-network-transmit-packets-dropped-total": ("net_send_drop", True),
    "container-network-receive-errors-total": ("net_recv_err", True),
    "container-network-transmit-errors-total": ("net_send_err", True),
    "container-sockets": ("sockets", False),
    "istio-latency-95": ("latency_p95", False),
    "istio-latency-99": ("latency_p99", False),
    "istio-request-total": ("throughput", True),
}
SUM_KINDS = {"cpu","mem","disk_write","disk_read","net_recv","net_send","net_recv_drop",
             "net_send_drop","net_recv_err","net_send_err","sockets","throughput"}  # latency=mean
_COUNTER_KPIS = {kpi for (kpi, is_counter) in METRIC_MAP.values() if is_counter}


def service_of(container: str, cands: list[str]) -> str | None:
    best = None
    for c in cands:
        if container == c or container.startswith(c + "-") or container.startswith(c + "_"):
            if best is None or len(c) > len(best):
                best = c
    return best


def split_col(col: str) -> tuple[str, str]:
    # "<container>_<metric-with-dashes>" -> (container, metric)
    comp, _, metric = col.rpartition("_")
    return comp, metric


def build_runs(split: str):
    base = SOURCE_ROOT / SPLIT_DIR[split]
    runs = []
    for case in sorted(os.listdir(base)):
        cp = base / case
        if not cp.is_dir():
            continue
        for run in sorted(os.listdir(cp)):
            rp = cp / run
            inj = rp / "inject_time.txt"
            if inj.exists():
                runs.append((int(inj.read_text().strip()), rp))
    return runs


def normalize_task(task_id, run_path, cands):
    mp = run_path / "metrics.csv"
    if not mp.exists():
        return None
    df = pd.read_csv(mp)
    tcol = df.columns[0]
    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy()
    minute = (t // 60 * 60).astype("int64")
    # accumulate per (service, kpi): list of per-sample series aggregated
    out = {}  # (service,kpi) -> np.array values aligned to samples
    for col in df.columns[1:]:
        comp, metric = split_col(col)
        if metric not in METRIC_MAP:
            continue
        svc = service_of(comp, cands)
        if svc is None:
            continue
        kpi, is_counter = METRIC_MAP[metric]
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if is_counter:
            d = np.diff(v, prepend=v[:1])
            d[d < 0] = 0.0  # counter reset
            val = d
        else:
            val = v
        key = (svc, kpi)
        if key not in out:
            out[key] = np.zeros(len(v))
            out[key][:] = np.nan
        cur = out[key]
        out[key] = np.where(np.isnan(cur), np.nan_to_num(val, nan=0.0),
                            cur + np.nan_to_num(val, nan=0.0)) if kpi in SUM_KINDS else \
                   np.nanmean(np.vstack([cur, val]), axis=0)
    # build long minute frame. counter-rate KPIs -> minute MAX (preserve short
    # bursts like a disk write spike); gauges -> minute MEAN.
    rows = []
    for (svc, kpi), val in out.items():
        agg = "max" if kpi in _COUNTER_KPIS else "mean"
        s = pd.DataFrame({"minute": minute, "value": val}).groupby("minute", as_index=False)["value"].agg(agg)
        for _, r in s.iterrows():
            rows.append((int(r["minute"]), svc, kpi, float(r["value"])))
    if not rows:
        return None
    long = pd.DataFrame(rows, columns=["timestamp", "cmdb_id", "kpi_name", "value"]).sort_values(
        ["timestamp", "cmdb_id", "kpi_name"])
    # write
    outdir = OUT_ROOT / task_id / "normalized"
    outdir.mkdir(parents=True, exist_ok=True)
    long.to_csv(outdir / "minute_metric.csv", index=False)
    # copy log/trace/error from original normalized so task_dir is complete
    odir = ORIG_ROOT / task_id / "normalized"
    for f in ["minute_log.csv", "minute_trace.csv", "error_logs.csv", "manifest.json"]:
        if (odir / f).exists():
            shutil.copy(odir / f, outdir / f)
    # metadata.json
    om = ORIG_ROOT / task_id / "metadata.json"
    if om.exists():
        shutil.copy(om, OUT_ROOT / task_id / "metadata.json")
    return len(long)


def main():
    splits = sys.argv[1:] or ["ob", "ss", "tt"]
    for split in splits:
        runs = build_runs(split)
        q = list(csv.DictReader((TASKS / split / "query.csv").open()))
        done = 0
        for row in q:
            tid = row["task_id"]
            st = int(datetime.strptime(row["incident_start"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
            en = int(datetime.strptime(row["incident_end"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
            match = next((rp for inj, rp in runs if st <= inj <= en), None)
            if match is None:
                continue
            n = normalize_task(tid, match, CAND[split])
            if n:
                done += 1
        print(f"{split}: normalized {done}/{len(q)} tasks -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
