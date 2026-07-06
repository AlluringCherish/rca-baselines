"""Read RE2 source simple_metrics.csv (wide) → minute_metric.csv (long, minute-averaged).

Usage:
    python scripts/normalize_metric.py

Reads from /home/khmin/RCA/Datasets_old/RCAEval/RE2/
Writes to  telemetry/rcaeval/{task_id}/normalized/minute_metric.csv + manifest.json
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
            raise ValueError(f"No source match for {task_id} window={row['incident_start']}~{row['incident_end']}")
        mapping.append((task_id, matched))
    return mapping


def normalize_one(task_id: str, source_path: Path) -> None:
    sm_path = source_path / "simple_metrics.csv"
    if not sm_path.exists():
        print(f"  SKIP {task_id}: no simple_metrics.csv at {source_path}")
        return

    wide = pd.read_csv(sm_path)
    if wide.empty or "time" not in wide.columns:
        print(f"  SKIP {task_id}: empty or missing time column")
        return

    long = wide.melt(id_vars=["time"], var_name="col", value_name="value")
    long = long.dropna(subset=["value"])

    parts = long["col"].str.rsplit("_", n=1)
    long["cmdb_id"] = parts.str[0]
    long["kpi_name"] = parts.str[1]

    long["timestamp"] = (long["time"] // 60) * 60
    long = long.drop(columns=["time", "col"])

    minute_avg = (
        long.groupby(["timestamp", "cmdb_id", "kpi_name"], as_index=False)["value"]
        .mean()
        .sort_values(["timestamp", "cmdb_id", "kpi_name"])
    )
    minute_avg = minute_avg[["timestamp", "cmdb_id", "kpi_name", "value"]]

    out_dir = TELEMETRY_ROOT / task_id / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "minute_metric.csv"
    minute_avg.to_csv(out_path, index=False)

    manifest_path = out_dir / "manifest.json"
    existing = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except Exception:
            pass
    existing_rows = existing.get("rows", {})
    existing_rows["metric"] = int(len(minute_avg))
    manifest = {
        "version": MANIFEST_VERSION,
        "rows": existing_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"  {task_id}: {len(minute_avg)} rows -> {out_path}")


if __name__ == "__main__":
    main()
