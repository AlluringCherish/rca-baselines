"""Build INDEX.txt (scoring detail + viz links) for an eval run dir.

Usage: python3 scripts/make_index.py <run_dir>
Reuses the same format as logs/test70_scoring_detail.txt: tasks sorted
correct->partial->wrong, each with required-dim marks, GT vs PRED, and the
viz_*.html link (run viz_all_run.py first so the links exist).
"""
import json, glob, itertools, sys
from pathlib import Path
import pandas as pd
from pec_agent_planner_analyzer.tasks import load_task_bundles, parse_scoring_points
from pec_agent_planner_analyzer.evaluate import _score_one, _safe_candidate, evaluate_planner_output

DMAP = {"openrca_market_cb1": "configs/dataset_openrca_market_cb1.yaml", "openrca_bank": "configs/dataset_openrca_bank.yaml",
        "openrca_market_cb2": "configs/dataset_openrca_market_cb2.yaml", "openrca_telecom": "configs/dataset_openrca_telecom.yaml",
        "rcaeval_re2_ob": "configs/dataset_rcaeval_re2_ob.yaml", "rcaeval_re2_ss": "configs/dataset_rcaeval_re2_ss.yaml",
        "rcaeval_re2_tt": "configs/dataset_rcaeval_re2_tt.yaml"}
RC = {"rcaeval_re2_ob", "rcaeval_re2_ss", "rcaeval_re2_tt"}
DIM = ["component", "reason", "time"]
SHORT = {"openrca_market_cb1": "cb1", "openrca_market_cb2": "cb2", "openrca_bank": "bank", "openrca_telecom": "tel",
         "rcaeval_re2_ob": "ob", "rcaeval_re2_ss": "ss", "rcaeval_re2_tt": "tt"}
_c = {}


def load(ds):
    if ds not in _c:
        bs = load_task_bundles(dataset_config=DMAP[ds])
        q = None if ds in RC else pd.read_csv(bs[0].dataset_cfg["raw_dataset_path"] + "/query.csv")
        _c[ds] = ({b.metadata["idx"]: b for b in bs}, q)
    return _c[ds]


def npreds(fa):
    return [fa] if isinstance(fa, dict) else [p for p in (fa or []) if isinstance(p, dict)]


def reqs(ds, idx, b, q):
    if ds in RC:
        return [DIM[:] for _ in b.ground_truth["answers"]]
    return [[d for d in DIM if d in sa] for sa in parse_scoring_points(q.iloc[idx]["scoring_points"])]


def classify(b, preds, rq, w):
    gt = b.ground_truth["answers"]
    total = sum(len(rq[k]) if k < len(rq) else 3 for k in range(len(gt)))
    M = {}
    for k, g in enumerate(gt):
        r = rq[k] if k < len(rq) else DIM
        for pi, p in enumerate(preds):
            M[(k, pi)] = sum(1 for d in r if _score_one(_safe_candidate(p), g, position=pi + 1, weights=w)[f"{d}_ok"])
    best = max((sum(M.get((k, pm[k]), 0) for k in range(len(pm)))
                for pm in itertools.permutations(range(len(preds)), min(len(gt), len(preds)))), default=0) if preds else 0
    return ("correct" if (total > 0 and best == total) else ("partial" if best > 0 else "wrong")), best, total


def main(run):
    run = run.rstrip("/")
    name = Path(run).name
    base = f"http://localhost:8800/logs/runs/{name}"
    rows = []
    for tj in sorted(glob.glob(run + "/*/trace.json")):
        t = json.load(open(tj)); ds = t["dataset"]
        if ds not in DMAP:
            continue
        bm, q = load(ds); b = bm[t["idx"]]; preds = npreds(t.get("final_answer"))
        _, d = evaluate_planner_output({"final_answer": t.get("final_answer")}, b.ground_truth, weights=b.eval_weights)
        pa = d["candidate_score"]["answers"]; rq = reqs(ds, t["idx"], b, q)
        cls, best, total = classify(b, preds, rq, b.eval_weights)
        vdir = Path(tj).parent; viz = sorted(vdir.glob("viz_*.html"))
        vurl = f"{base}/{vdir.name}/{viz[0].name}" if viz else "(no viz — run viz_all_run.py)"
        rows.append((ds, f"{SHORT[ds]}-{t['idx']}", cls, best, total, b, preds, pa, rq, vurl))
    order = {"correct": 0, "partial": 1, "wrong": 2}
    rows.sort(key=lambda r: (order[r[2]], r[0], r[1]))
    n = {"correct": 0, "partial": 0, "wrong": 0}
    for r in rows:
        n[r[2]] += 1
    out = Path(run) / "INDEX.txt"
    with out.open("w", encoding="utf-8") as f:
        f.write(f"채점 상세 — run: {name}\n")
        f.write("기준: OpenRCA=scoring_points 요구 dim / RCAeval=3 dim 전부. correct=전부, partial=1+, wrong=0\n")
        f.write("VIZ 서버: python3 -m http.server 8800 --directory /home/khmin/RCA\n")
        f.write(f"\n총 {sum(n.values())}: correct={n['correct']} partial={n['partial']} wrong={n['wrong']}\n" + "=" * 100 + "\n")
        for ds, nm, cls, best, total, b, preds, pa, rq, vurl in rows:
            f.write(f"\n[{cls.upper():7}] {nm:8} ({best}/{total})   VIZ: {vurl}\n")
            for i, gt in enumerate(b.ground_truth["answers"]):
                a = pa[i] if i < len(pa) else {}
                p = preds[i] if i < len(preds) and isinstance(preds[i], dict) else {}
                r = rq[i] if i < len(rq) else DIM
                marks = ("c" if a.get("component_ok") else ".") + ("r" if a.get("reason_ok") else ".") + ("t" if a.get("time_ok") else ".")
                f.write(f"    req{r} [{marks}]\n")
                f.write(f"        GT  : {gt.get('component')} / {gt.get('reason')} / {gt.get('datetime') or gt.get('time')}\n")
                f.write(f"        PRED: {p.get('component')} / {p.get('reason')} / {p.get('time')}\n")
    print(f"wrote {out}  (correct={n['correct']} partial={n['partial']} wrong={n['wrong']})")


if __name__ == "__main__":
    for rd in sys.argv[1:]:
        main(rd)
