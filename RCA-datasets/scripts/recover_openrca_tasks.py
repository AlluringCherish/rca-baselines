"""Recover missing OpenRCA task telemetry by slicing the original per-date dumps
into per-task windows (same format as existing task_7-* dirs), then normalize.

Only Bank idx 8/37/92 are missing (audited). Source: Datasets_old/openrca/Bank/telemetry/<YYYY_MM_DD>/.
"""
import sys, datetime as dt
from pathlib import Path
import pandas as pd

SRC = Path("/home/khmin/RCA/Datasets_old/openrca/Bank/telemetry")
DST_ROOT = Path("/home/khmin/RCA/RCA-datasets/telemetry/openrca/openrca_bank")
QUERY = Path("/home/khmin/RCA/RCA-datasets/tasks/openrca/Bank/query.csv")
FILES = [("metric", "metric_app.csv"), ("metric", "metric_container.csv"),
         ("trace", "trace_span.csv"), ("log", "log_service.csv")]
MISSING = [8, 37, 92]


def ts_seconds(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    return v.where(v < 1e12, v / 1000.0)  # ms -> s for comparison only


def slice_one(idx: int) -> None:
    q = pd.read_csv(QUERY)
    start = dt.datetime.strptime(str(q.iloc[idx]["incident_start"]).strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    end = dt.datetime.strptime(str(q.iloc[idx]["incident_end"]).strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ws, we = start.timestamp(), end.timestamp()
    date_dir = SRC / start.strftime("%Y_%m_%d")
    dst = DST_ROOT / f"task_7-{idx}"
    print(f"\n--- task_7-{idx}  window {start:%Y-%m-%d %H:%M}~{end:%H:%M}  src={date_dir.name} ---")
    if not date_dir.is_dir():
        print(f"  !! source date dir missing: {date_dir}"); return
    for sub, fname in FILES:
        srcf = date_dir / sub / fname
        if not srcf.exists():
            print(f"  [skip] no source {sub}/{fname}"); continue
        df = pd.read_csv(srcf)
        tcol = "timestamp"
        sec = ts_seconds(df[tcol])
        sl = df[(sec >= ws) & (sec <= we)].copy()
        outdir = dst / sub
        outdir.mkdir(parents=True, exist_ok=True)
        sl.to_csv(outdir / fname, index=False)
        print(f"  {sub}/{fname}: {len(sl)} rows  (src {len(df)})")


def normalize_one(idx: int) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    import normalize_openrca_metric as nm, normalize_openrca_trace as nt, normalize_openrca_log as nl
    dst = DST_ROOT / f"task_7-{idx}"
    m = nm.normalize_task(dst)
    t = nt.normalize_task(dst, duration_unit="ms")
    lg = nl.normalize_task(dst)
    print(f"  normalized task_7-{idx}: metric={m} trace={t} log={lg}")


if __name__ == "__main__":
    for i in MISSING:
        slice_one(i)
    print("\n=== normalize ===")
    for i in MISSING:
        normalize_one(i)
