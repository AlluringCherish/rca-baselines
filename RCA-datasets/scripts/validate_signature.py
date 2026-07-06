"""Validate the shape/signature RULE classifier IN ISOLATION (GT component given).

Reports: overall accuracy, confusion matrix, and 5-fold CV (rule is fixed, so CV shows
stability across splits). Compares against majority-class baseline.
Uses cached features from /tmp/sig_feats2.csv if present, else recomputes.
"""
import sys, os
from collections import Counter, defaultdict
import pandas as pd
sys.path.insert(0, "/home/khmin/RCA/RCA-datasets/scripts")
from shape_signature import classify_features
from sig_features import extract

TARGET = {"container cpu load", "container memory load",
          "container read i/o load", "container write i/o load"}
LABELS = ["container cpu load", "container memory load", "container read i/o load", "container write i/o load"]
SHORT = lambda s: s.replace("container ", "").replace(" i/o load", "-io").replace(" load", "")


def build_df():
    cache = "/tmp/sig_feats2.csv"
    if os.path.exists(cache):
        return pd.read_csv(cache)
    from pec_agent_planner_analyzer.tasks import load_task_bundles
    from pec_agent_planner_analyzer.dataset_tools.zscore_filter import load_minute_metric
    recs = []
    for cfg, sh in [("configs/dataset_openrca_market_cb1.yaml", "cb1"), ("configs/dataset_openrca_market_cb2.yaml", "cb2")]:
        for b in load_task_bundles(dataset_config=cfg):
            if not any((a.get("reason") or "").lower() in TARGET for a in b.ground_truth["answers"]):
                continue
            w = load_minute_metric(b.task.task_dir)
            for a in b.ground_truth["answers"]:
                r = (a.get("reason") or "").lower()
                if r not in TARGET:
                    continue
                f = extract(w, a.get("component")); f.update(sh=sh, idx=b.metadata["idx"], reason=r, comp=a.get("component"))
                recs.append(f)
    df = pd.DataFrame(recs); df.to_csv(cache, index=False)
    return df


def main():
    df = build_df()
    preds, branches = [], []
    for _, row in df.iterrows():
        r, br = classify_features(row)
        preds.append(r); branches.append(br)
    df["pred"] = preds; df["branch"] = branches; df["ok"] = df.pred == df.reason
    n = len(df); acc = df.ok.mean()
    maj = df.reason.value_counts().iloc[0] / n
    print(f"=== RULE classifier (GT component given) — n={n} ===")
    print(f"accuracy = {acc:.1%}   (majority-class baseline = {maj:.1%})\n")
    print("Confusion (rows=GT, cols=pred):")
    conf = defaultdict(Counter)
    for _, row in df.iterrows():
        conf[row.reason][row.pred] += 1
    print("  GT\\PRED        " + " ".join(f"{SHORT(l):>8}" for l in LABELS))
    for gt in LABELS:
        print(f"  {SHORT(gt):14}" + " ".join(f"{conf[gt][p]:>8}" for p in LABELS) + f"   (n={sum(conf[gt].values())})")
    # per-class recall
    print("\nper-class recall:")
    for gt in LABELS:
        tot = sum(conf[gt].values()); hit = conf[gt][gt]
        print(f"  {SHORT(gt):12} {hit}/{tot} = {hit/tot:.0%}" if tot else f"  {SHORT(gt)}: n/a")
    # 5-fold CV (deterministic split by row order hashed)
    print("\n5-fold CV accuracy (fixed rule, split by idx):")
    df = df.reset_index(drop=True)
    folds = [df[df.index % 5 == k] for k in range(5)]
    accs = [f.ok.mean() for f in folds]
    print("  folds:", [f"{a:.0%}" for a in accs], f" mean={sum(accs)/5:.1%}")
    # wrong cases
    print("\n--- wrong cases ---")
    for _, row in df[~df.ok].iterrows():
        print(f"  {row.sh}-{int(row.idx):<3} {row.comp:24} GT={SHORT(row.reason):8} -> {SHORT(row.pred):8} [{row.branch}]"
              f"  thr_run={int(row.thr_run)} thr_peak={row.thr_peak:.0f} ws_run={int(row.ws_run)} mem_util={row.mem_util:.2f}"
              f" rd_run={int(row.rd_run)} rd_rise={row.rd_rise:.0f} wr_run={int(row.wr_run)} wr_rise={row.wr_rise:.0f}")


if __name__ == "__main__":
    main()
