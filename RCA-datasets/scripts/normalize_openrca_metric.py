"""Normalize OpenRCA metric data into minute_metric.csv (long format).

Per task: read metric/*.csv, unify into timestamp,cmdb_id,kpi_name,value.
- metric_container.csv / metric_node.csv / ... : already long (timestamp,cmdb_id,kpi_name,value)
- metric_app.csv (wide: timestamp,rr,sr,cnt,mrt,tc) -> melt to long with app.* kpi names
- ms -> s conversion for response-time metrics (app.mrt)

Output: <task>/normalized/minute_metric.csv + manifest.json (version 0)

Usage:
    python scripts/normalize_openrca_metric.py --dataset openrca_bank
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

TELEMETRY_ROOT = Path("/home/khmin/RCA/RCA-datasets/telemetry/openrca")
MANIFEST_VERSION = 0

LONG_COLS = ["timestamp", "cmdb_id", "kpi_name", "value"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="e.g. openrca_bank")
    args = p.parse_args()

    ds_dir = TELEMETRY_ROOT / args.dataset
    tasks = sorted([d for d in ds_dir.iterdir() if d.is_dir() and d.name.startswith("task_")])
    total = 0
    for task in tasks:
        n = normalize_task(task)
        total += 1
        print(f"  {task.name}: {n} rows")
    print(f"Done. {total} tasks.")


def normalize_task(task_dir: Path) -> int:
    metric_dir = task_dir / "metric"
    frames: list[pd.DataFrame] = []
    if metric_dir.is_dir():
        for csv_path in sorted(metric_dir.glob("*.csv")):
            frame = _load_metric_csv(csv_path)
            if frame is not None and not frame.empty:
                frames.append(frame)

    if frames:
        result = pd.concat(frames, ignore_index=True)
        result = result.dropna(subset=["timestamp", "value"])
        # ms -> s if timestamp looks like epoch-millis (13 digits), then floor to minute
        ts = pd.to_numeric(result["timestamp"], errors="coerce")
        ts = ts.where(ts < 1e12, ts / 1000.0)
        result["timestamp"] = ((ts // 60) * 60).astype("int64")
        result = (
            result.groupby(["timestamp", "cmdb_id", "kpi_name"], as_index=False)["value"]
            .mean()
            .sort_values(["timestamp", "cmdb_id", "kpi_name"])
        )
        result = result[LONG_COLS]
    else:
        result = pd.DataFrame(columns=LONG_COLS)

    out_dir = task_dir / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "minute_metric.csv", index=False)

    _update_manifest(out_dir / "manifest.json", {"metric": int(len(result))})
    return len(result)


_MESH_RE = re.compile(r"(.+?)\.(source|destination)\.(.+)")


def _load_metric_csv(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    if df.empty:
        return None
    cols = set(df.columns)

    # Telecom long format: itemid, name(=kpi), bomc_id, timestamp(ms), value, cmdb_id
    if {"name", "timestamp", "value", "cmdb_id"}.issubset(cols) and "kpi_name" not in cols:
        out = df.rename(columns={"name": "kpi_name"})[["timestamp", "cmdb_id", "kpi_name", "value"]].copy()
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return _convert_units(out)

    # Telecom app wide format: serviceName, startTime(ms), avg_time, num, succee_num, succee_rate
    if {"serviceName", "startTime"}.issubset(cols):
        value_cols = [c for c in df.columns if c not in ("serviceName", "startTime")]
        long = df.melt(id_vars=["serviceName", "startTime"], value_vars=value_cols,
                       var_name="kpi_name", value_name="value")
        long = long.rename(columns={"serviceName": "cmdb_id", "startTime": "timestamp"})
        long["kpi_name"] = "app." + long["kpi_name"].astype(str)
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        return _convert_units(long)

    # Mesh (istio) format: cmdb_id = pod.{source|destination}.peer.service
    # Drop istio_agent_* (proxy self-state noise); fold edge info into kpi_name; cmdb_id = pod
    if "mesh" in path.name.lower() and {"timestamp", "cmdb_id", "kpi_name", "value"}.issubset(cols):
        out = df[df["kpi_name"].astype(str).str.startswith("istio_agent")].index
        m = df.drop(index=out).copy()
        parsed = m["cmdb_id"].astype(str).str.extract(_MESH_RE)
        # parsed[0]=pod, [1]=direction, [2]=peer.service
        m["pod"] = parsed[0].fillna(m["cmdb_id"])
        edge = (parsed[1].fillna("") + "." + parsed[2].fillna("")).str.strip(".")
        m["kpi_name"] = "mesh." + edge + "." + m["kpi_name"].astype(str)
        m["kpi_name"] = m["kpi_name"].str.replace(r"^mesh\.\.", "mesh.", regex=True)
        m = m.rename(columns={"cmdb_id": "_orig", "pod": "cmdb_id"})
        out_df = m[["timestamp", "cmdb_id", "kpi_name", "value"]].copy()
        out_df["value"] = pd.to_numeric(out_df["value"], errors="coerce")
        return _convert_units(out_df)

    # Container format with node.pod cmdb_id (market): unify to pod, fold node into kpi_name.
    # Only when cmdb_ids actually look like "node-N.pod" (bank container has no node prefix).
    if "container" in path.name.lower() and {"timestamp", "cmdb_id", "kpi_name", "value"}.issubset(cols):
        node_prefixed = df["cmdb_id"].astype(str).str.match(r"node-\d+\.")
        if node_prefixed.mean() > 0.5:
            c = df.copy()
            split = c["cmdb_id"].astype(str).str.split(".", n=1, expand=True)
            c["kpi_name"] = "container." + split[0] + "." + c["kpi_name"].astype(str)
            c["cmdb_id"] = split[1]
            out_df = c[["timestamp", "cmdb_id", "kpi_name", "value"]].copy()
            out_df["value"] = pd.to_numeric(out_df["value"], errors="coerce")
            return _convert_units(out_df)

    # Runtime format: cmdb_id = service.ts:port -> service, port folded into kpi prefix
    if "runtime" in path.name.lower() and {"timestamp", "cmdb_id", "kpi_name", "value"}.issubset(cols):
        r = df.copy()
        r["cmdb_id"] = r["cmdb_id"].astype(str).str.split(".", n=1).str[0]
        r["kpi_name"] = "runtime." + r["kpi_name"].astype(str)
        out_df = r[["timestamp", "cmdb_id", "kpi_name", "value"]].copy()
        out_df["value"] = pd.to_numeric(out_df["value"], errors="coerce")
        return _convert_units(out_df)

    # Already long format
    if {"timestamp", "cmdb_id", "kpi_name", "value"}.issubset(cols):
        out = df[["timestamp", "cmdb_id", "kpi_name", "value"]].copy()
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return _convert_units(out)

    # App/service wide format: component column is 'tc' (bank) or 'service' (market)
    comp_col = "tc" if "tc" in cols else ("service" if "service" in cols else None)
    if comp_col and "timestamp" in cols:
        value_cols = [c for c in df.columns if c not in ("timestamp", comp_col)]
        long = df.melt(
            id_vars=["timestamp", comp_col],
            value_vars=value_cols,
            var_name="kpi_name",
            value_name="value",
        )
        long = long.rename(columns={comp_col: "cmdb_id"})
        if comp_col == "service":
            # market: 'adservice-grpc' -> cmdb_id='adservice', protocol folded into kpi
            split = long["cmdb_id"].astype(str).str.rsplit("-", n=1, expand=True)
            proto = split[1].where(split[1].isin(["grpc", "http"]), "")
            base = split[0].where(split[1].isin(["grpc", "http"]), long["cmdb_id"])
            long["cmdb_id"] = base
            long["kpi_name"] = ("app." + proto + ".").where(proto != "", "app.") + long["kpi_name"].astype(str)
        else:
            long["kpi_name"] = "app." + long["kpi_name"].astype(str)
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        return _convert_units(long)

    print(f"    SKIP unknown format: {path.name} cols={list(df.columns)}")
    return None


def _convert_units(df: pd.DataFrame) -> pd.DataFrame:
    # ms -> s for response-time metrics
    ms_mask = df["kpi_name"].astype(str).str.lower().str.contains("mrt|responsetime|resptime|latency_ms|_milliseconds|avg_time")
    df.loc[ms_mask, "value"] = df.loc[ms_mask, "value"] / 1000.0
    return df


def _update_manifest(path: Path, rows: dict) -> None:
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            pass
    existing_rows = existing.get("rows", {})
    existing_rows.update(rows)
    path.write_text(json.dumps({"version": MANIFEST_VERSION, "rows": existing_rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
