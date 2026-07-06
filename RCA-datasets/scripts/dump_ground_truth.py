"""Materialize ground truth to explicit files (so GT is managed/auditable, not
only computed at runtime). For every task, dumps the enriched answers
(component/reason/datetime) + whether each falls in the incident window.

Outputs per dataset: <raw_dataset_path>/ground_truth.csv
And a consolidated: tasks/ground_truth_all.json  (keyed by "<split>:<idx>")
"""
import csv, json, datetime as dt
from pathlib import Path
from pec_agent_planner_analyzer.tasks import load_task_bundles

DS = Path("/home/khmin/RCA/RCA-datasets")
CONFIGS = {
    "ob": "configs/dataset_rcaeval_re2_ob.yaml", "ss": "configs/dataset_rcaeval_re2_ss.yaml",
    "tt": "configs/dataset_rcaeval_re2_tt.yaml", "bank": "configs/dataset_openrca_bank.yaml",
    "market_cb1": "configs/dataset_openrca_market_cb1.yaml", "market_cb2": "configs/dataset_openrca_market_cb2.yaml",
    "telecom": "configs/dataset_openrca_telecom.yaml",
}


def in_window(dtstr, ws, we, off):
    try:
        g = dt.datetime.strptime(dtstr, "%Y-%m-%d %H:%M:%S") - dt.timedelta(hours=off)
        return ws - dt.timedelta(minutes=2) <= g <= we + dt.timedelta(minutes=2)
    except Exception:
        return None


def main():
    allgt = {}
    for split, cfg in CONFIGS.items():
        bundles = load_task_bundles(dataset_config=cfg)
        if not bundles:
            continue
        raw = Path(bundles[0].dataset_cfg["raw_dataset_path"])
        rows = []
        for b in bundles:
            off = int(b.metadata.get("gt_tz_offset_hours", 0))
            ws = dt.datetime.strptime(b.task.incident_start, "%Y-%m-%d %H:%M:%S")
            we = dt.datetime.strptime(b.task.incident_end, "%Y-%m-%d %H:%M:%S")
            ans = b.ground_truth["answers"]
            allgt[f"{split}:{b.metadata['idx']}"] = {
                "task_id": b.task.task_id, "window": [b.task.incident_start, b.task.incident_end],
                "tz_offset_hours": off, "answers": ans,
            }
            for ai, a in enumerate(ans, 1):
                rows.append({"idx": b.metadata["idx"], "task_id": b.task.task_id, "answer_no": ai,
                             "component": a["component"], "reason": a["reason"], "datetime": a["datetime"],
                             "in_window": in_window(a["datetime"], ws, we, off)})
        out = DS / raw / "ground_truth.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["idx", "task_id", "answer_no", "component", "reason", "datetime", "in_window"])
            w.writeheader(); w.writerows(rows)
        oob = sum(1 for r in rows if r["in_window"] is False)
        print(f"{split:12} -> {out}  ({len(rows)} answers, out-of-window={oob})")
    (DS / "tasks" / "ground_truth_all.json").write_text(json.dumps(allgt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nconsolidated -> tasks/ground_truth_all.json ({len(allgt)} tasks)")


if __name__ == "__main__":
    main()
