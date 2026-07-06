"""Normalize OpenRCA log data into minute_log.csv + error_logs.csv.

Log categories (detected from log_name) extract different signals:
  - access log  (localhost_access_log / apache_access_log / *-envoy_gateway):
        HTTP status code -> error if status >= 400
  - application log (*-service_application):
        severity -> error if severity in {error, warn, warning, fatal, err}
  - gc log (gc):
        GC pause seconds -> log.gc_pause_max

Per (minute, component) features:
  - log.row_count
  - log.error_count
  - log.gc_pause_max   (only when gc logs present)

Also writes error_logs.csv: raw error/warning entries (timestamp, cmdb_id, message).

Output: <task>/normalized/minute_log.csv, error_logs.csv + manifest.json (version 0)

Usage:
    python scripts/normalize_openrca_log.py --dataset openrca_bank
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

TELEMETRY_ROOT = Path("/home/khmin/RCA/RCA-datasets/telemetry/openrca")
MANIFEST_VERSION = 0
LOG_COLS = ["timestamp", "cmdb_id", "log_name", "value"]
ERR_COLS = ["timestamp", "cmdb_id", "message"]

_STATUS_RE = re.compile(r'HTTP/[\d.]+"?\s+(\d{3})')
_SEVERITY_RE = re.compile(r"severity:\s*(\w+)", re.IGNORECASE)
_SEVERITY_BRACKET_RE = re.compile(r"\b(info|debug|warn|warning|error|err|fatal|trace)\b", re.IGNORECASE)
_GC_SECS_RE = re.compile(r"real=([\d.]+)\s*secs")
ERROR_SEVERITIES = {"error", "err", "warn", "warning", "fatal"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    args = p.parse_args()

    ds_dir = TELEMETRY_ROOT / args.dataset
    tasks = sorted([d for d in ds_dir.iterdir() if d.is_dir() and d.name.startswith("task_")])
    for task in tasks:
        n_log, n_err = normalize_task(task)
        print(f"  {task.name}: log={n_log} rows, errors={n_err}")
    print(f"Done. {len(tasks)} tasks.")


def _category(log_name: str) -> str:
    n = str(log_name).lower()
    if n == "gc":
        return "gc"
    if "access" in n or "envoy" in n:
        return "access"
    if "application" in n or "service" in n:
        return "application"
    return "other"


def _classify_row(category: str, value: str) -> tuple[bool, float | None]:
    """Return (is_error, gc_pause_seconds)."""
    text = str(value)
    if category == "access":
        m = _STATUS_RE.search(text)
        if m:
            return int(m.group(1)) >= 400, None
        return False, None
    if category == "application":
        m = _SEVERITY_RE.search(text) or _SEVERITY_BRACKET_RE.search(text)
        if m:
            return m.group(1).lower() in ERROR_SEVERITIES, None
        return False, None
    if category == "gc":
        m = _GC_SECS_RE.search(text)
        return False, float(m.group(1)) if m else None
    return False, None


def normalize_task(task_dir: Path) -> tuple[int, int]:
    log_dir = task_dir / "log"
    out_dir = task_dir / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    if log_dir.is_dir():
        for csv_path in sorted(log_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            if not df.empty and {"timestamp", "cmdb_id", "log_name", "value"}.issubset(df.columns):
                frames.append(df)

    if not frames:
        pd.DataFrame(columns=LOG_COLS).to_csv(out_dir / "minute_log.csv", index=False)
        pd.DataFrame(columns=ERR_COLS).to_csv(out_dir / "error_logs.csv", index=False)
        _update_manifest(out_dir / "manifest.json", {"log": 0, "error_logs": 0})
        return 0, 0

    df = pd.concat(frames, ignore_index=True)
    df["minute"] = (df["timestamp"].astype("int64") // 60) * 60
    df["category"] = df["log_name"].map(_category)

    flags = df.apply(lambda r: _classify_row(r["category"], r["value"]), axis=1)
    df["is_error"] = [f[0] for f in flags]
    df["gc_pause"] = [f[1] for f in flags]

    rows = []
    for (minute, comp), g in df.groupby(["minute", "cmdb_id"]):
        rows.append({"timestamp": minute, "cmdb_id": comp, "log_name": "log.row_count", "value": float(len(g))})
        rows.append({"timestamp": minute, "cmdb_id": comp, "log_name": "log.error_count", "value": float(g["is_error"].sum())})
        gc_vals = g["gc_pause"].dropna()
        if not gc_vals.empty:
            rows.append({"timestamp": minute, "cmdb_id": comp, "log_name": "log.gc_pause_max", "value": float(gc_vals.max())})

    minute_log = pd.DataFrame(rows, columns=LOG_COLS).sort_values(["timestamp", "cmdb_id", "log_name"])
    minute_log.to_csv(out_dir / "minute_log.csv", index=False)

    errs = df[df["is_error"]]
    err_out = errs[["timestamp", "cmdb_id", "value"]].copy()
    err_out.columns = ERR_COLS
    err_out = err_out.sort_values("timestamp")
    err_out.to_csv(out_dir / "error_logs.csv", index=False)

    _update_manifest(out_dir / "manifest.json", {"log": int(len(minute_log)), "error_logs": int(len(err_out))})
    return len(minute_log), len(err_out)


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
