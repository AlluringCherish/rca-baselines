"""Visualize every task in an eval run dir: refresh GT (corrected) in trace.json,
then render viz_<task>.html into each task dir via visualize_openrca."""
import json, glob, os, subprocess, sys
from pathlib import Path
from pec_agent_planner_analyzer.tasks import load_task_bundles

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/khmin/RCA/logs/runs/s1_baseline__no-memory__test70")
DSROOT = Path("/home/khmin/RCA/RCA-datasets")
VIZ = "/home/khmin/RCA/pec_rca_agent/scripts/visualize_openrca.py"
# dataset -> (config, telemetry subdir, tz offset, split-for-bundle)
DMAP = {
    "rcaeval_re2_ob": ("configs/dataset_rcaeval_re2_ob.yaml", "telemetry/rcaeval", 0, "ob"),
    "rcaeval_re2_ss": ("configs/dataset_rcaeval_re2_ss.yaml", "telemetry/rcaeval", 0, "ss"),
    "rcaeval_re2_tt": ("configs/dataset_rcaeval_re2_tt.yaml", "telemetry/rcaeval", 0, "tt"),
    "openrca_bank": ("configs/dataset_openrca_bank.yaml", "telemetry/openrca/openrca_bank", 8, "bank"),
    "openrca_market_cb1": ("configs/dataset_openrca_market_cb1.yaml", "telemetry/openrca/openrca_market_cb1", 8, "market_cb1"),
    "openrca_market_cb2": ("configs/dataset_openrca_market_cb2.yaml", "telemetry/openrca/openrca_market_cb2", 8, "market_cb2"),
    "openrca_telecom": ("configs/dataset_openrca_telecom.yaml", "telemetry/openrca/openrca_telecom", 8, "telecom"),
}

# corrected GT map per (dataset, idx)
traces = sorted(glob.glob(str(RUN / "*" / "trace.json")))
by_split = {}
for tj in traces:
    t = json.load(open(tj))
    by_split.setdefault(t["dataset"], set()).add(t["idx"])
gtmap = {}
for ds, idxs in by_split.items():
    cfg = DMAP[ds][0]
    for b in load_task_bundles(dataset_config=cfg, indices=sorted(idxs)):
        gtmap[(ds, b.metadata["idx"])] = b.ground_truth

ok = fail = 0
for tj in traces:
    t = json.load(open(tj)); ds = t["dataset"]; idx = t["idx"]; tid = t["task_id"]
    cfg, telsub, off, _ = DMAP[ds]
    # refresh GT
    if (ds, idx) in gtmap:
        t["ground_truth"] = gtmap[(ds, idx)]
        json.dump(t, open(tj, "w"), ensure_ascii=False, indent=2, default=str)
    task_dir = telsub + "/" + tid
    outdir = os.path.dirname(tj)
    if not (DSROOT / task_dir).is_dir():
        print(f"  [skip] no telemetry {task_dir}"); fail += 1; continue
    r = subprocess.run(["python3", VIZ, "--task-dir", task_dir, "--trace-json", tj,
                        "--dataset-config", cfg, "--idx", str(idx), "--gt-tz-offset", str(off),
                        "--out", outdir], cwd=DSROOT, capture_output=True, text=True)
    if r.returncode == 0:
        ok += 1
    else:
        fail += 1; print(f"  [FAIL] {ds} {tid}: {r.stderr.strip()[-200:]}")
print(f"\nDONE: {ok} viz generated, {fail} failed. (run dir: {RUN})")
