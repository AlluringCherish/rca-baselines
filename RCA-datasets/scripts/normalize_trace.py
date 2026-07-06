"""Read RE2 source traces.csv (raw spans) → minute_trace.csv (long, minute-aggregated).

Per (minute, service): row_count, error_count, duration_mean, duration_p95, duration_max

Usage:
    python scripts/normalize_trace.py
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_ROOT = Path("/home/khmin/RCA/Datasets_old/RCAEval/RE2")
DATASET_ROOT = Path("/home/khmin/RCA/RCA-datasets")
TELEMETRY_ROOT = DATASET_ROOT / "telemetry" / "rcaeval"
TASKS_ROOT = DATASET_ROOT / "tasks" / "rcaeval"

SPLITS = {
    "ob": "RE2-OB",
    "ss": "RE2-SS",
    "tt": "RE2-TT",
}

MANIFEST_VERSION = 0

TRACE_FEATURES = ["trace.row_count", "trace.error_count", "trace.duration_mean", "trace.duration_p95", "trace.duration_max"]


def main() -> None:
    total = 0
    for split, source_dir_name in SPLITS.items():
        mapping = build_mapping(split, source_dir_name)
        for task_id, source_path in mapping:
            normalize_one(task_id, source_path)
            total += 1
    print(f"Done. Normalized {total} tasks.")


def build_mapping(split: str, source_dir_name: str) -> list[tuple[str, Path]]:
    source_base = SOURCE_ROOT / source_dir_name
    sources: list[tuple[str, str, int, Path]] = []
    for case in sorted(os.listdir(source_base)):
        case_path = source_base / case
        if not case_path.is_dir():
            continue
        for run in sorted(os.listdir(case_path)):
            if not run.isdigit():
                continue
            run_path = case_path / run
            inject_file = run_path / "inject_time.txt"
            inject_ts = int(inject_file.read_text().strip()) if inject_file.exists() else 0
            sources.append((case, run, inject_ts, run_path))

    query_path = TASKS_ROOT / split / "query.csv"
    rows = list(csv.DictReader(query_path.open()))
    mapping: list[tuple[str, Path]] = []
    for row in rows:
        task_id = row["task_id"]
        start = datetime.strptime(row["incident_start"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        end = datetime.strptime(row["incident_end"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        start_ts, end_ts = int(start.timestamp()), int(end.timestamp())
        matched = None
        for case, run, inject_ts, run_path in sources:
            if start_ts <= inject_ts <= end_ts:
                matched = run_path
                break
        if matched is None:
            raise ValueError(f"No source match for {task_id}")
        mapping.append((task_id, matched))
    return mapping


def normalize_one(task_id: str, source_path: Path) -> None:
    traces_path = source_path / "traces.csv"
    if not traces_path.exists():
        print(f"  SKIP {task_id}: no traces.csv")
        return

    df = pd.read_csv(traces_path, usecols=["spanID", "parentSpanID", "serviceName", "startTimeMillis", "duration", "statusCode"])
    if df.empty:
        print(f"  SKIP {task_id}: empty traces.csv")
        return

    df["timestamp"] = (df["startTimeMillis"] // 60000) * 60
    df["duration_sec"] = df["duration"] / 1_000_000
    df["is_error"] = df["statusCode"].fillna(0).astype(float).ne(0).astype(float)

    # Resolve caller via parentSpanID -> parent's serviceName
    span2svc = dict(zip(df["spanID"], df["serviceName"]))
    df["caller"] = df["parentSpanID"].map(span2svc).fillna("root")
    df["callee"] = df["serviceName"]

    # Aggregate per edge (caller -> callee) per minute
    edge = df.groupby(["timestamp", "caller", "callee"]).agg(
        row_count=("duration_sec", "count"),
        error_count=("is_error", "sum"),
        duration_mean=("duration_sec", "mean"),
        duration_p95=("duration_sec", lambda x: np.percentile(x, 95)),
    ).reset_index()
    edge["error_rate"] = (edge["error_count"] / edge["row_count"]).fillna(0)

    rows_out: list[dict] = []
    for _, r in edge.iterrows():
        ts = r["timestamp"]
        feats = [
            ("row_count", r["row_count"]),
            ("error_rate", r["error_rate"]),
            ("duration_mean", r["duration_mean"]),
            ("duration_p95", r["duration_p95"]),
        ]
        for feat, val in feats:
            # callee-centric: who called me
            rows_out.append({"timestamp": ts, "cmdb_id": r["callee"], "trace_name": f"trace.from_{r['caller']}.{feat}", "value": val})
            # caller-centric: who I called
            rows_out.append({"timestamp": ts, "cmdb_id": r["caller"], "trace_name": f"trace.to_{r['callee']}.{feat}", "value": val})

    result = pd.DataFrame(rows_out).sort_values(["timestamp", "cmdb_id", "trace_name"])
    result = result[["timestamp", "cmdb_id", "trace_name", "value"]]

    out_dir = TELEMETRY_ROOT / task_id / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "minute_trace.csv", index=False)

    # --- trace_error_summary.csv: per (minute, caller, callee, status_code) ---
    err_cols = ["timestamp", "caller", "callee", "status_code", "count", "mean_duration"]
    errors = df[df["statusCode"].fillna(0).astype(float) != 0].copy()
    if not errors.empty:
        error_agg = errors.groupby(["timestamp", "caller", "callee", "statusCode"]).agg(
            count=("duration_sec", "count"),
            mean_duration=("duration_sec", "mean"),
        ).reset_index()
        error_agg.columns = err_cols
        error_agg["status_code"] = error_agg["status_code"].astype(int)
        error_agg["mean_duration"] = error_agg["mean_duration"].round(6)
        error_agg = error_agg.sort_values(["timestamp", "callee", "caller"])
        error_agg.to_csv(out_dir / "trace_error_summary.csv", index=False)
        trace_error_count = int(len(error_agg))
    else:
        pd.DataFrame(columns=err_cols).to_csv(out_dir / "trace_error_summary.csv", index=False)
        trace_error_count = 0

    manifest_path = out_dir / "manifest.json"
    existing = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except Exception:
            pass
    existing_rows = existing.get("rows", {})
    existing_rows["trace"] = int(len(result))
    existing_rows["trace_error_summary"] = trace_error_count
    manifest = {"version": MANIFEST_VERSION, "rows": existing_rows}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"  {task_id}: {len(result)} rows")


if __name__ == "__main__":
    main()
