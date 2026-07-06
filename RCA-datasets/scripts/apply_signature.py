"""Application experiment: use the signature classifier as a REASON-corrector.

Realistic setting: take the agent's own predictions on market tasks (full eval run).
For each predicted answer whose reason is a container resource/IO fault, recompute the
reason with the signature classifier RUN ON THE AGENT'S PREDICTED COMPONENT (no GT peeking).
Measure reason-correctness (vs GT, via the official scorer) BEFORE vs AFTER substitution.
"""
import sys, json, glob
from collections import Counter
sys.path.insert(0, "/home/khmin/RCA/RCA-datasets/scripts")
from shape_signature import classify_component
from pec_agent_planner_analyzer.tasks import load_task_bundles
from pec_agent_planner_analyzer.dataset_tools.zscore_filter import load_minute_metric
from pec_agent_planner_analyzer.evaluate import evaluate_planner_output, _safe_candidate, _score_one

DMAP = {"openrca_market_cb1": "configs/dataset_openrca_market_cb1.yaml",
        "openrca_market_cb2": "configs/dataset_openrca_market_cb2.yaml"}
SH = {"openrca_market_cb1": "cb1", "openrca_market_cb2": "cb2"}
CONTAINER_FAULTS = {"container cpu load", "container memory load",
                    "container read i/o load", "container write i/o load"}
_c = {}


def load(ds):
    if ds not in _c:
        _c[ds] = {b.metadata["idx"]: b for b in load_task_bundles(dataset_config=DMAP[ds])}
    return _c[ds]


def reason_ok(pred_comp, pred_reason, gt_answers, w):
    """Is (pred_comp, pred_reason) reason-correct for some GT answer it localizes?"""
    for gt in gt_answers:
        sc = _score_one(_safe_candidate({"component": pred_comp, "reason": pred_reason}), gt, position=1, weights=w)
        if sc["component_ok"] and sc["reason_ok"]:
            return True
    return False


def main():
    files = glob.glob("/home/khmin/RCA/logs/runs/v3_fulleval/*/*/trace.json") + \
            glob.glob("/home/khmin/RCA/logs/runs/v3_fulleval/*/trace.json")
    flipped_good = flipped_bad = unchanged = 0
    rows = []
    cache_w = {}
    for tj in files:
        t = json.load(open(tj)); ds = t["dataset"]
        if ds not in DMAP:
            continue
        b = load(ds)[t["idx"]]; w_key = (ds, t["idx"])
        fa = t.get("final_answer"); preds = fa if isinstance(fa, list) else [fa]
        preds = [p for p in preds if isinstance(p, dict)]
        gt = b.ground_truth["answers"]; w = b.eval_weights
        wdf = None
        for p in preds:
            pr = (p.get("reason") or "").lower(); comp = p.get("component")
            if pr not in CONTAINER_FAULTS or not comp:
                continue
            if wdf is None:
                wdf = cache_w.get(w_key) or load_minute_metric(b.task.task_dir); cache_w[w_key] = wdf
            new_reason, branch, feats = classify_component(wdf, comp)
            if new_reason is None or new_reason == pr:
                unchanged += 1; continue
            was_ok = reason_ok(comp, pr, gt, w)
            now_ok = reason_ok(comp, new_reason, gt, w)
            if now_ok and not was_ok:
                flipped_good += 1; tag = "FIX "
            elif was_ok and not now_ok:
                flipped_bad += 1; tag = "BREAK"
            else:
                unchanged += 1; tag = "neutral"
            if tag != "neutral":
                rows.append((tag, SH[ds], t["idx"], comp, pr, new_reason, branch))
    print("=== Signature reason-corrector applied to agent predictions (market full eval) ===")
    print(f"container-fault predictions changed by classifier:")
    print(f"  R fixed (wrong->right): {flipped_good}")
    print(f"  R broken (right->wrong): {flipped_bad}")
    print(f"  net R change: {flipped_good - flipped_bad:+d}\n")
    for tag, sh, idx, comp, old, new, br in sorted(rows):
        print(f"  {tag:5} {sh}-{idx:<3} {comp:24} {old:24} -> {new:24} [{br}]")


if __name__ == "__main__":
    main()
