"""Scale-robust shape features per container component, for fault-type classification."""
import sys
import numpy as np
sys.path.insert(0, "/home/khmin/RCA/RCA-datasets/scripts")
from shape_signature import component_signals, shape_features


def _series(sub, substr):
    g = sub[sub.kpi_name.astype(str).str.lower().str.contains(substr, regex=False)]
    if g.empty:
        return None
    return g.groupby("timestamp").value.sum().sort_index()


def _rf(ser):
    """(sustained_run, sustained_rise_abs, peak) for a series."""
    if ser is None:
        return 0, 0.0, 0.0
    f = shape_features(ser.values)
    if f is None:
        return 0, 0.0, 0.0
    return f["sustained_run"], float(f["sustained_level"] - f["base"]), float(f["peak"])


def extract(w, component, time_lo=None, time_hi=None):
    sub = component_signals(w, component)
    if time_lo is not None:
        sub = sub[(sub.timestamp >= time_lo) & (sub.timestamp < time_hi)]
    thr_run, _, thr_peak = _rf(_series(sub, "cpu_cfs_throttled_seconds"))
    cu_run, cu_rise, _ = _rf(_series(sub, "cpu_usage_seconds"))
    ws = _series(sub, "memory_working_set")
    lim = _series(sub, "spec_memory_limit")
    ws_run, ws_rise, _ = _rf(ws)
    mem_util = float(ws.max() / lim.max()) if (ws is not None and lim is not None and lim.max() > 0) else 0.0
    rd_run, rd_rise, _ = _rf(_series(sub, "fs_reads_mb"))
    wr_run, wr_rise, _ = _rf(_series(sub, "fs_writes_mb"))
    return dict(thr_run=thr_run, thr_peak=thr_peak, cu_run=cu_run, cu_rise=cu_rise,
                ws_run=ws_run, ws_rise=ws_rise, mem_util=mem_util,
                rd_run=rd_run, rd_rise=rd_rise, wr_run=wr_run, wr_rise=wr_rise)
