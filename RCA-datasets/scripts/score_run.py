"""Score a run dir by the required-dim criterion (correct/partial/wrong) + FULL.

Usage: python3 scripts/score_run.py <run_dir> [<run_dir> ...]
correct = all scoring_points-required (answer x dim) targets hit; partial = >=1 but
not all; wrong = 0. RCAeval requires component+reason+time. FULL = total_score==1.0.
"""
import json, glob, sys, itertools
from collections import Counter
import pandas as pd
from pec_agent_planner_analyzer.tasks import load_task_bundles, parse_scoring_points
from pec_agent_planner_analyzer.evaluate import _score_one, _safe_candidate, evaluate_planner_output

DMAP = {"openrca_market_cb1": "configs/dataset_openrca_market_cb1.yaml", "openrca_bank": "configs/dataset_openrca_bank.yaml",
        "openrca_market_cb2": "configs/dataset_openrca_market_cb2.yaml", "openrca_telecom": "configs/dataset_openrca_telecom.yaml",
        "rcaeval_re2_ob": "configs/dataset_rcaeval_re2_ob.yaml", "rcaeval_re2_ss": "configs/dataset_rcaeval_re2_ss.yaml",
        "rcaeval_re2_tt": "configs/dataset_rcaeval_re2_tt.yaml"}
RC = {"rcaeval_re2_ob", "rcaeval_re2_ss", "rcaeval_re2_tt"}
DIM = ["component", "reason", "time"]
_cache = {}


def load(ds):
    if ds not in _cache:
        bs = load_task_bundles(dataset_config=DMAP[ds])
        q = None if ds in RC else pd.read_csv(bs[0].dataset_cfg["raw_dataset_path"] + "/query.csv")
        _cache[ds] = ({b.metadata["idx"]: b for b in bs}, q)
    return _cache[ds]


def npreds(fa):
    if isinstance(fa, dict):
        return [fa]
    if isinstance(fa, list):
        return [p for p in fa if isinstance(p, dict)]
    return []


def req_of(ds, idx, b, q):
    if ds in RC:
        return [["component", "reason", "time"] for _ in b.ground_truth["answers"]]
    return [[d for d in DIM if d in sa] for sa in parse_scoring_points(q.iloc[idx]["scoring_points"])]


def classify(b, preds, reqs, w):
    gt = b.ground_truth["answers"]
    total = sum(len(reqs[k]) if k < len(reqs) else 3 for k in range(len(gt)))
    M = {}
    for k, g in enumerate(gt):
        rq = reqs[k] if k < len(reqs) else DIM
        for pi, p in enumerate(preds):
            M[(k, pi)] = sum(1 for d in rq if _score_one(_safe_candidate(p), g, position=pi + 1, weights=w)[f"{d}_ok"])
    best = 0
    if preds:
        best = max((sum(M.get((k, pm[k]), 0) for k in range(len(pm)))
                    for pm in itertools.permutations(range(len(preds)), min(len(gt), len(preds)))), default=0)
    return "correct" if (total > 0 and best == total) else ("partial" if best > 0 else "wrong")


def score_run(rd):
    g = {"RCAeval": Counter(), "OpenRCA": Counter(), "TOTAL": Counter()}
    full = Counter()
    for tj in glob.glob(rd + "/*/trace.json"):
        t = json.load(open(tj)); ds = t["dataset"]
        if ds not in DMAP:
            continue
        bm, q = load(ds); b = bm[t["idx"]]
        c = classify(b, npreds(t.get("final_answer")), req_of(ds, t["idx"], b, q), b.eval_weights)
        _, d = evaluate_planner_output({"final_answer": t.get("final_answer")}, b.ground_truth, weights=b.eval_weights)
        f = 1 if abs(d["total_score"] - 1.0) < 1e-6 else 0
        key2 = "RCAeval" if ds in RC else "OpenRCA"
        for key in (key2, "TOTAL"):
            g[key][c] += 1; full[key] += f
    return g, full


if __name__ == "__main__":
    for rd in sys.argv[1:]:
        name = rd.rstrip("/").split("/")[-1]
        g, full = score_run(rd)
        print(f"\n### {name}")
        for k in ["RCAeval", "OpenRCA", "TOTAL"]:
            c = g[k]; n = sum(c.values())
            if n:
                print(f"  {k:8} n={n:2} correct={c['correct']:2}({c['correct']/n:3.0%}) "
                      f"partial={c['partial']:2}({c['partial']/n:3.0%}) wrong={c['wrong']:2}({c['wrong']/n:3.0%}) | FULL={full[k]}/{n}")
