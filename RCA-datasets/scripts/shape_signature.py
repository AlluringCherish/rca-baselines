"""Deterministic shape/signature classifier for container fault TYPE (reason).

Idea (from shapelet / segment-shape literature): a genuine resource-load fault shows a
SUSTAINED plateau in its responsible signal, whereas incidental side-effects show a
single-sample BLIP. We compute, per KPI family (cpu / memory / read-IO / write-IO),
MAD-normalized SUSTAINED elevation over the incident window and pick the strongest
sustained family -> reason. Blips are discounted (they are side-effects).

Scope: market_cb1 / market_cb2 container faults. Network excluded (no telemetry marker).
"""
import numpy as np
import pandas as pd

# KPI-name substrings -> family. Order matters (checked in this order).
# memory uses working_set / rss (NOT cache/usage which include page-cache inflated by writes).
FAMILY_PATTERNS = {
    "cpu":   ["container_cpu_cfs_throttled", "container_cpu_usage_seconds", "container_cpu_load_average"],
    "memory": ["container_memory_working_set", "container_memory_rss"],
    "read":  ["container_fs_reads_mb", "container_fs_reads.", "container_fs_read_seconds", "container_fs_sector_reads"],
    "write": ["container_fs_writes_mb", "container_fs_writes.", "container_fs_write_seconds", "container_fs_sector_writes"],
}
FAMILY_TO_REASON = {
    "cpu": "container cpu load",
    "memory": "container memory load",
    "read": "container read i/o load",
    "write": "container write i/o load",
}


def _kpi_family(kpi: str):
    k = kpi.lower()
    for fam, pats in FAMILY_PATTERNS.items():
        for p in pats:
            if p in k:
                return fam
    return None


def shape_features(values):
    """Shape descriptor of one series over the incident window."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 6:
        return None
    q = max(3, n // 4)
    base = float(np.median(v[:q]))
    mad = float(np.median(np.abs(v[:q] - base))) or (0.05 * (np.max(v) - base) if np.max(v) > base else 0.0)
    peak = float(np.max(v))
    rise = peak - base
    if rise <= 0 or mad <= 0:
        return dict(base=base, peak=peak, rise=max(rise, 0.0), sustained_run=0,
                    sustained_level=base, strength=0.0, peak_strength=0.0, shape="flat")
    sustained_thr = base + 0.3 * rise
    half_thr = base + 0.5 * rise
    above = v > sustained_thr
    # longest consecutive run above sustained threshold
    run = best = 0
    for a in above:
        run = run + 1 if a else 0
        best = max(best, run)
    plateau_vals = v[v > half_thr]
    sustained_level = float(np.median(plateau_vals)) if len(plateau_vals) else base
    strength = (sustained_level - base) / mad           # MAD units, sustained part
    peak_strength = rise / mad
    shape = "plateau" if best >= 3 else ("blip" if best <= 1 else "short")
    return dict(base=base, peak=peak, rise=rise, sustained_run=best,
                sustained_level=sustained_level, strength=strength,
                peak_strength=peak_strength, shape=shape)


def component_signals(w, component):
    """All rows for a component INCLUDING its pods (svc -> svc-0/1/2), container metrics only."""
    cm = w.cmdb_id.astype(str)
    base = component
    # if component is a base service, include its numbered pods; if it's a pod, use exact + base
    mask = (cm == base) | cm.str.match(rf"^{base}-\d+$")
    sub = w[mask]
    return sub


# Rule thresholds (from separability diagnostic on market cb1/cb2)
THR_RUN, THR_PEAK = 3, 50      # cpu throttle sustained marker
IO_RUN, IO_RISE = 3, 20.0      # sustained read/write (MB)
MEM_RUN, MEM_UTIL = 3, 0.5     # sustained memory


def classify_features(f):
    """Deterministic fault-type rule from container shape features (see sig_features.extract).

    Priority: throttle(cpu/write) -> sustained IO(read/write) -> memory -> cpu fallback.
    Returns (reason, branch_tag).
    """
    throttled = f["thr_run"] >= THR_RUN and f["thr_peak"] >= THR_PEAK
    # sustained IO, OR a very large IO burst even if short (a multi-GB read/write in 1-2 min
    # is an injected IO fault, not memory page-cache drift)
    BIG_IO = 500.0
    rd = (f["rd_run"] >= IO_RUN and f["rd_rise"] >= IO_RISE) or f["rd_rise"] >= BIG_IO
    wr = (f["wr_run"] >= IO_RUN and f["wr_rise"] >= IO_RISE) or f["wr_rise"] >= BIG_IO
    mem = f["ws_run"] >= MEM_RUN or f["mem_util"] >= MEM_UTIL
    if throttled:
        # CPU stress throttles; heavy WRITE also throttles -> disambiguate by sustained write
        if wr and f["wr_rise"] > 50:
            return "container write i/o load", "throttle+write"
        return "container cpu load", "throttle"
    # no sustained throttle: IO faults first (reads inflate working_set via page cache)
    if rd and (not wr or f["rd_rise"] >= f["wr_rise"]):
        return "container read i/o load", "read"
    if wr:
        return "container write i/o load", "write"
    if mem:
        return "container memory load", "memory"
    return "container cpu load", "cpu-fallback"


def classify_component(w, component, time_lo=None, time_hi=None):
    """Convenience: extract features then classify. Returns (reason, branch, features)."""
    import sys
    sys.path.insert(0, "/home/khmin/RCA/RCA-datasets/scripts")
    from sig_features import extract
    f = extract(w, component, time_lo, time_hi)
    reason, branch = classify_features(f)
    return reason, branch, f
