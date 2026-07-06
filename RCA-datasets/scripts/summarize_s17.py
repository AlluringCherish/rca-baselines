"""Summarize the s17 (SIGNATURE-operator) experiment vs s16 baseline on the same
market container-fault tasks. Reports: reason accuracy on container-fault GT answers,
FULL/correct, and how often the agent actually CALLED the SIGNATURE tool.
"""
import json, glob, itertools
from collections import Counter
from pec_agent_planner_analyzer.tasks import load_task_bundles, parse_scoring_points
from pec_agent_planner_analyzer.evaluate import _score_one, _safe_candidate, evaluate_planner_output
import pandas as pd

DMAP = {"openrca_market_cb1": "configs/dataset_openrca_market_cb1.yaml",
        "openrca_market_cb2": "configs/dataset_openrca_market_cb2.yaml"}
SH = {"openrca_market_cb1": "cb1", "openrca_market_cb2": "cb2"}
CF = {"container cpu load", "container memory load", "container read i/o load", "container write i/o load"}
DIM = ["component", "reason", "time"]
_c = {}


def load(ds):
    if ds not in _c:
        bs = load_task_bundles(dataset_config=DMAP[ds])
        q = pd.read_csv(bs[0].dataset_cfg["raw_dataset_path"] + "/query.csv")
        _c[ds] = ({b.metadata["idx"]: b for b in bs}, q)
    return _c[ds]


def npreds(fa):
    return [fa] if isinstance(fa, dict) else [p for p in (fa or []) if isinstance(p, dict)]


def reqs(idx, b, q):
    return [[d for d in DIM if d in sa] for sa in parse_scoring_points(q.iloc[idx]["scoring_points"])]


def analyze(run_dir, label, count_sig_calls=False):
    files = glob.glob(run_dir + "/*/trace.json")
    # restrict to the 61 market container-fault tasks
    sample = {(s["split"], s["idx"]) for s in json.load(open("/tmp/market_container.json"))}
    cf_total = cf_hit = 0           # reason accuracy on container-fault GT answers (component-localized)
    full = correct = ntask = 0
    sig_calls = sig_tasks = 0
    for tj in files:
        t = json.load(open(tj)); ds = t["dataset"]
        if ds not in DMAP:
            continue
        split = {"openrca_market_cb1": "market_cb1", "openrca_market_cb2": "market_cb2"}[ds]
        if (split, t["idx"]) not in sample:
            continue
        bm, q = load(ds); b = bm[t["idx"]]; w = b.eval_weights
        gts = b.ground_truth["answers"]; preds = npreds(t.get("final_answer"))
        rq = reqs(t["idx"], b, q); ntask += 1
        # FULL / correct
        _, d = evaluate_planner_output({"final_answer": t.get("final_answer")}, b.ground_truth, weights=w)
        if abs(d["total_score"] - 1.0) < 1e-6:
            full += 1
        # container-fault reason accuracy: for each container-fault GT, is there a pred that
        # localizes it AND gets reason right?
        for gt in gts:
            if (gt.get("reason") or "").lower() not in CF:
                continue
            cf_total += 1
            ok = any(_score_one(_safe_candidate(p), gt, position=1, weights=w)["component_ok"]
                     and _score_one(_safe_candidate(p), gt, position=1, weights=w)["reason_ok"]
                     for p in preds)
            cf_hit += ok
        # SIGNATURE call count from the trace steps
        if count_sig_calls:
            steps = t.get("steps") or t.get("trace") or t.get("actions") or []
            txt = json.dumps(t)
            n = txt.count("SIGNATURE")
            if n:
                sig_tasks += 1
                sig_calls += 1
    print(f"=== {label}  (tasks={ntask}) ===")
    print(f"  container-fault REASON accuracy (localized & reason-correct): {cf_hit}/{cf_total} = {cf_hit/cf_total:.0%}")
    print(f"  FULL: {full}/{ntask} = {full/ntask:.0%}")
    if count_sig_calls:
        print(f"  SIGNATURE appeared in {sig_tasks}/{ntask} task traces")
    return cf_hit, cf_total, full, ntask


def main():
    print("Experiment: SIGNATURE operator (s17) vs baseline (s16/v3_fulleval), same 61 market container-fault tasks\n")
    analyze("/home/khmin/RCA/logs/runs/v3_fulleval", "BASELINE s16 (v3_fulleval)")
    print()
    analyze("/home/khmin/RCA/logs/runs/market_container_s17", "s17 + SIGNATURE", count_sig_calls=True)


if __name__ == "__main__":
    main()
