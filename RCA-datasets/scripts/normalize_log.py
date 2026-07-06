"""Read RE2 source logs.csv → minute_log.csv (aggregated) + error_logs.csv (raw error/warning).

Usage:
    python scripts/normalize_log.py
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

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
    logs_path = source_path / "logs.csv"
    if not logs_path.exists():
        print(f"  SKIP {task_id}: no logs.csv")
        return

    df = pd.read_csv(logs_path, usecols=["timestamp", "container_name", "message", "level", "error"], low_memory=False)
    if df.empty:
        _write_empty(task_id)
        return

    df["ts_sec"] = df["timestamp"] // 1_000_000_000
    df["ts_minute"] = (df["ts_sec"] // 60) * 60
    df["level_lower"] = df["level"].fillna("").str.lower()
    df["error_str"] = df["error"].fillna("")
    df["is_error"] = (df["level_lower"].isin(["error", "warning"])) | (df["error_str"] != "")

    out_dir = TELEMETRY_ROOT / task_id / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- minute_log.csv: per (minute, component) aggregated counts ---
    agg_rows: list[dict] = []
    for (ts_min, comp), g in df.groupby(["ts_minute", "container_name"]):
        agg_rows.append({"timestamp": ts_min, "cmdb_id": comp, "log_name": "log.row_count", "value": len(g)})
        agg_rows.append({"timestamp": ts_min, "cmdb_id": comp, "log_name": "log.error_count", "value": int(g["is_error"].sum())})

    minute_log = pd.DataFrame(agg_rows).sort_values(["timestamp", "cmdb_id", "log_name"])
    minute_log = minute_log[["timestamp", "cmdb_id", "log_name", "value"]]
    minute_log.to_csv(out_dir / "minute_log.csv", index=False)

    # --- error_logs.csv: raw error/warning messages ---
    errors = df[df["is_error"]].copy()
    if not errors.empty:
        error_out = errors[["ts_sec", "container_name", "message", "error_str"]].copy()
        error_out.columns = ["timestamp", "cmdb_id", "message", "error"]
        error_out = error_out.sort_values("timestamp")
        error_out.to_csv(out_dir / "error_logs.csv", index=False)
        error_count = len(error_out)
    else:
        pd.DataFrame(columns=["timestamp", "cmdb_id", "message", "error"]).to_csv(out_dir / "error_logs.csv", index=False)
        error_count = 0

    # --- manifest update ---
    manifest_path = out_dir / "manifest.json"
    existing = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except Exception:
            pass
    existing_rows = existing.get("rows", {})
    existing_rows["log"] = int(len(minute_log))
    existing_rows["error_logs"] = error_count
    manifest = {"version": MANIFEST_VERSION, "rows": existing_rows}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"  {task_id}: minute_log={len(minute_log)} rows, error_logs={error_count} rows")


def _write_empty(task_id: str) -> None:
    out_dir = TELEMETRY_ROOT / task_id / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["timestamp", "cmdb_id", "log_name", "value"]).to_csv(out_dir / "minute_log.csv", index=False)
    pd.DataFrame(columns=["timestamp", "cmdb_id", "message", "error"]).to_csv(out_dir / "error_logs.csv", index=False)
    print(f"  {task_id}: empty")


if __name__ == "__main__":
    main()
