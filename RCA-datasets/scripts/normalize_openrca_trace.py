"""Normalize OpenRCA trace data into minute_trace.csv (long format).

trace_span.csv columns: timestamp(s or ms), cmdb_id, parent_id, span_id, trace_id, duration(ms)
No statusCode / operationName available -> only duration & row_count features.

Per (minute, component): trace.row_count, trace.duration_mean, trace.duration_p95
duration: ms -> s

Output: <task>/normalized/minute_trace.csv + manifest.json (version 0)

Usage:
    python scripts/normalize_openrca_trace.py --dataset openrca_bank
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TELEMETRY_ROOT = Path("/home/khmin/RCA/RCA-datasets/telemetry/openrca")
MANIFEST_VERSION = 0
TRACE_COLS = ["timestamp", "cmdb_id", "trace_name", "value"]


DURATION_DIVISOR = {"ms": 1000.0, "us": 1_000_000.0}
_NORMAL_STATUS = {"0", "ok", "200", "201", "202", "204", "301", "302", "304"}


def _to_epoch_seconds(values: pd.Series) -> pd.Series:
    ts = pd.to_numeric(values, errors="coerce")
    return ts.where(ts < 1e12, ts / 1000.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--duration-unit", choices=["ms", "us"], default="ms",
                   help="raw duration unit (bank=ms, market=us)")
    args = p.parse_args()

    ds_dir = TELEMETRY_ROOT / args.dataset
    tasks = sorted([d for d in ds_dir.iterdir() if d.is_dir() and d.name.startswith("task_")])
    for task in tasks:
        n = normalize_task(task, args.duration_unit)
        print(f"  {task.name}: {n} rows")
    print(f"Done. {len(tasks)} tasks.")


def _telecom_edges(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["minute"] = (df["startTime"] // 1000 // 60) * 60
    df["dur_s"] = pd.to_numeric(df["elapsedTime"], errors="coerce") / 1000.0
    df["is_error"] = (~df["success"].astype(str).str.lower().isin(["true", "1"])).astype(float)
    df["ct"] = df["callType"].astype(str).str.lower()

    id2svc = dict(zip(df["id"], df["cmdb_id"]))
    df["caller"] = df["pid"].map(id2svc).fillna("root")
    df["callee"] = df["cmdb_id"]

    edge = df.groupby(["minute", "caller", "callee", "ct"]).agg(
        row_count=("dur_s", "count"),
        error_count=("is_error", "sum"),
        duration_mean=("dur_s", "mean"),
        duration_p95=("dur_s", lambda x: np.percentile(x, 95)),
    ).reset_index()
    edge["error_rate"] = (edge["error_count"] / edge["row_count"]).fillna(0)

    rows = []
    for _, r in edge.iterrows():
        ts, caller, callee, ct = int(r["minute"]), r["caller"], r["callee"], r["ct"]
        feats = [("row_count", r["row_count"]), ("error_rate", r["error_rate"]),
                 ("duration_mean", r["duration_mean"]), ("duration_p95", r["duration_p95"])]
        if caller == callee:
            for feat, val in feats:
                rows.append({"timestamp": ts, "cmdb_id": callee, "trace_name": f"trace.self.{ct}.{feat}", "value": round(float(val), 6)})
        else:
            for feat, val in feats:
                rows.append({"timestamp": ts, "cmdb_id": callee, "trace_name": f"trace.from_{caller}.{ct}.{feat}", "value": round(float(val), 6)})
                if caller != "root":
                    rows.append({"timestamp": ts, "cmdb_id": caller, "trace_name": f"trace.to_{callee}.{ct}.{feat}", "value": round(float(val), 6)})
    return rows


def _is_error(code) -> bool:
    s = str(code).strip().lower()
    if s in _NORMAL_STATUS:
        return False
    if s and s[0].isdigit():
        return s[0] in ("4", "5")  # http 4xx/5xx
    return s not in ("ok", "")  # non-numeric, non-ok -> error


def normalize_task(task_dir: Path, duration_unit: str = "ms") -> int:
    trace_path = task_dir / "trace" / "trace_span.csv"
    out_dir = task_dir / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _empty():
        pd.DataFrame(columns=TRACE_COLS).to_csv(out_dir / "minute_trace.csv", index=False)
        _update_manifest(out_dir / "manifest.json", {"trace": 0})
        return 0

    if not trace_path.exists():
        return _empty()

    df = pd.read_csv(trace_path)
    if df.empty:
        return _empty()

    # Telecom format: callType, startTime(ms), elapsedTime(ms), success, traceId, id, pid, cmdb_id, ...
    if {"callType", "startTime", "elapsedTime", "id", "pid", "cmdb_id"}.issubset(df.columns):
        rows = _telecom_edges(df)
        result = pd.DataFrame(rows, columns=TRACE_COLS).sort_values(["timestamp", "cmdb_id", "trace_name"])
        result.to_csv(out_dir / "minute_trace.csv", index=False)
        _update_manifest(out_dir / "manifest.json", {"trace": int(len(result))})
        return len(result)

    parent_col = "parent_id" if "parent_id" in df.columns else ("parent_span" if "parent_span" in df.columns else None)
    if parent_col is None or "span_id" not in df.columns:
        return _empty()

    df["minute"] = ((_to_epoch_seconds(df["timestamp"]) // 60) * 60).astype("int64")
    df["dur_s"] = df["duration"] / DURATION_DIVISOR[duration_unit]
    has_status = "status_code" in df.columns
    df["is_error"] = df["status_code"].map(_is_error).astype(float) if has_status else 0.0

    span2svc = dict(zip(df["span_id"], df["cmdb_id"]))
    df["caller"] = df[parent_col].map(span2svc).fillna("root")
    df["callee"] = df["cmdb_id"]

    edge = df.groupby(["minute", "caller", "callee"]).agg(
        row_count=("dur_s", "count"),
        error_count=("is_error", "sum"),
        duration_mean=("dur_s", "mean"),
        duration_p95=("dur_s", lambda x: np.percentile(x, 95)),
    ).reset_index()
    edge["error_rate"] = (edge["error_count"] / edge["row_count"]).fillna(0)

    rows = []
    for _, r in edge.iterrows():
        ts, caller, callee = int(r["minute"]), r["caller"], r["callee"]
        feats = [("row_count", r["row_count"]), ("duration_mean", r["duration_mean"]), ("duration_p95", r["duration_p95"])]
        if has_status:
            feats.insert(1, ("error_rate", r["error_rate"]))
        if caller == callee:
            for feat, val in feats:
                rows.append({"timestamp": ts, "cmdb_id": callee, "trace_name": f"trace.self.{feat}", "value": round(float(val), 6)})
        else:
            for feat, val in feats:
                rows.append({"timestamp": ts, "cmdb_id": callee, "trace_name": f"trace.from_{caller}.{feat}", "value": round(float(val), 6)})
                if caller != "root":
                    rows.append({"timestamp": ts, "cmdb_id": caller, "trace_name": f"trace.to_{callee}.{feat}", "value": round(float(val), 6)})

    result = pd.DataFrame(rows, columns=TRACE_COLS).sort_values(["timestamp", "cmdb_id", "trace_name"])
    result.to_csv(out_dir / "minute_trace.csv", index=False)
    _update_manifest(out_dir / "manifest.json", {"trace": int(len(result))})
    return len(result)


def _update_manifest(path: Path, rows: dict) -> None:
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            pass
    er = existing.get("rows", {})
    er.update(rows)
    path.write_text(json.dumps({"version": MANIFEST_VERSION, "rows": er}, indent=2) + "\n")


if __name__ == "__main__":
    main()
