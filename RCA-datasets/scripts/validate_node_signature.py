"""Node-level SIGNATURE classifier — prototype + isolated validation.

Node fault types (market cb1/cb2): node cpu load, node cpu spike, node memory consumption,
node disk read/write i/o consumption, node disk space consumption.
Markers (from diagnostic): no throttle equivalent; use system.* families.
Validate on GT node (reason classification isolated) -> confusion + 5-fold CV.
"""
import numpy as np, pandas as pd, itertools
from collections import Counter, defaultdict
from pec_agent_planner_analyzer.tasks import load_task_bundles
from pec_agent_planner_analyzer.operators.base import long_frame, window_ts

TARGET = ["node cpu load", "node cpu spike", "node memory consumption",
          "node disk read i/o consumption", "node disk write i/o consumption",
          "node disk space consumption"]
SHORT = {"node cpu load": "cpu-load", "node cpu spike": "cpu-spike", "node memory consumption": "mem",
         "node disk read i/o consumption": "disk-rd", "node disk write i/o consumption": "disk-wr",
         "node disk space consumption": "disk-sp"}


def _shape(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    n = len(v)
    if n < 6:
        return 0, 0.0
    q = max(3, n // 4); base = np.median(v[:q]); peak = v.max(); rise = peak - base
    if rise <= 0:
        return 0, 0.0
    run = best = 0
    for a in (v > base + 0.3 * rise):
        run = run + 1 if a else 0; best = max(best, run)
    lvl = np.median(v[v > base + 0.5 * rise]) if (v > base + 0.5 * rise).any() else base
    return best, float(lvl - base)


SIGS = {"cpu": "system.cpu.user", "mem": "system.mem.pct_usage", "rd": "system.io.rkb_s",
        "ws": "system.io.w_s", "util": "system.io.util", "await": "system.io.await",
        "dpct": "system.disk.pct_usage"}


def features(df, node):
    sub = df[df.cmdb_id.astype(str) == str(node)]
    f = {}
    for k, sig in SIGS.items():
        g = sub[sub.signal.astype(str).str.lower() == sig.lower()]
        ser = g.groupby("timestamp").value.mean().sort_index() if not g.empty else None
        run, rise = _shape(ser.values) if ser is not None else (0, 0.0)
        f[k + "_run"] = run; f[k + "_rise"] = rise
    return f


def classify(f):
    # CLEAR specific markers FIRST (nodes have noisy incidental IO that else steals cpu/mem).
    # 1) memory: mem.pct_usage clearly up (specific; other faults keep mem ~0)
    if f["mem_rise"] >= 15:
        return "node memory consumption"
    # 2) cpu LOAD: cpu.user clearly high & sustained (disk/mem faults keep cpu.user ~2-3)
    if f["cpu_rise"] >= 30 or (f["cpu_run"] >= 3 and f["cpu_rise"] >= 25):
        return "node cpu load"
    # 3) disk space: gradual % fill
    if f["dpct_rise"] >= 1.5:
        return "node disk space consumption"
    # 4) disk IO burst (read vs write)
    io_active = f["util_rise"] >= 15 or f["await_rise"] >= 8
    if io_active or f["rd_rise"] >= 5000 or f["ws_rise"] >= 100:
        # write marker (io.w_s) is specific; check it before read since contaminated nodes
        # (e.g. node-6) carry large background reads that would else mask a write fault
        if f["ws_rise"] >= 150:
            return "node disk write i/o consumption"
        if f["rd_rise"] >= 5000:
            return "node disk read i/o consumption"
        if f["ws_rise"] >= 100:
            return "node disk write i/o consumption"
        return "node disk read i/o consumption"
    # 5) weaker memory / cpu spike / fallback
    if f["mem_rise"] >= 8:
        return "node memory consumption"
    if f["cpu_rise"] >= 8:
        return "node cpu spike"
    return "node cpu load"


def main():
    rows = []
    for cfg in ["configs/dataset_openrca_market_cb1.yaml", "configs/dataset_openrca_market_cb2.yaml"]:
        for b in load_task_bundles(dataset_config=cfg):
            if not any((a.get("reason") or "").lower() in TARGET for a in b.ground_truth["answers"]):
                continue
            df = long_frame(b.task.task_dir); s, e = window_ts([b.task.incident_start, b.task.incident_end])
            df = df[(df.timestamp >= s) & (df.timestamp < e)]
            for a in b.ground_truth["answers"]:
                r = (a.get("reason") or "").lower()
                if r not in TARGET:
                    continue
                f = features(df, a.get("component"))
                pred = classify(f)
                rows.append((r, pred, pred == r, a.get("component"), f))
    n = len(rows); acc = sum(x[2] for x in rows) / n
    maj = Counter(x[0] for x in rows).most_common(1)[0][1] / n
    print(f"=== NODE classifier (GT node given) — n={n} ===")
    print(f"accuracy = {acc:.1%}  (majority baseline {maj:.0%})\n")
    conf = defaultdict(Counter)
    for r, p, ok, c, f in rows:
        conf[r][p] += 1
    print("  GT\\PRED      " + " ".join(f"{SHORT[l]:>9}" for l in TARGET))
    for gt in TARGET:
        print(f"  {SHORT[gt]:10}" + " ".join(f"{conf[gt][p]:>9}" for p in TARGET) + f"  ({sum(conf[gt].values())})")
    print("\nper-class recall:")
    for gt in TARGET:
        tot = sum(conf[gt].values()); print(f"  {SHORT[gt]:10} {conf[gt][gt]}/{tot} = {conf[gt][gt]/tot:.0%}")
    accs = []
    for k in range(5):
        fold = [x for i, x in enumerate(rows) if i % 5 == k]
        accs.append(sum(x[2] for x in fold) / len(fold))
    print(f"\n5-fold CV: {[f'{a:.0%}' for a in accs]} mean={sum(accs)/5:.1%}")
    print("\n--- wrong ---")
    for r, p, ok, c, f in rows:
        if not ok:
            print(f"  {c:8} GT={SHORT[r]:9}->{SHORT[p]:9} cpu(run{f['cpu_run']},{f['cpu_rise']:.0f}) mem{f['mem_rise']:.0f} "
                  f"rd{f['rd_rise']:.0f} ws{f['ws_rise']:.0f} util{f['util_rise']:.0f} await{f['await_rise']:.0f} dpct{f['dpct_rise']:.1f}")


if __name__ == "__main__":
    main()
