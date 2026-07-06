"""Data-analysis tools for Flow-of-Action, over offline CSV telemetry.

These re-implement the paper's K8s data tools (whether_is_abnormal_metric,
collect_trace, kubectl_logs, get_relevant_metric, *_analyze) on top of the
per-task telemetry CSVs. They are deterministic (no LLM) and return bounded
text. They are the only functions callable inside SOP code (run_sop).

Timezone contract (single conversion point):
  - Tool time arguments are strings '%Y-%m-%d %H:%M:%S' in the SAME timezone as
    the task instruction (UTC+8 for OpenRCA Bank/Market/Telecom, UTC for RCAEval).
  - to_epoch() converts such a local string to a real UTC epoch.
  - Every timestamp PRINTED by a tool is converted back with to_local(), so
    anomaly times are already in the scoring timezone.
"""
from __future__ import annotations

import calendar
import difflib
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


_NAME_COL = {"metric": "kpi_name", "trace": "trace_name", "log": "log_name"}


def _strip_tz(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[Zz]$", "", s)
    s = re.sub(r"[+-]\d{2}:?\d{2}$", "", s).strip()
    s = re.sub(r"\s+(UTC|utc)$", "", s).strip()
    s = s.replace("T", " ")
    s = re.sub(r"\.\d+$", "", s)  # drop fractional seconds
    return s


def _parse_struct(s: str) -> time.struct_time:
    txt = _strip_tz(s)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return time.strptime(txt, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized datetime string: {s!r}")


def _utc_str_to_epoch(s: str) -> int:
    return int(calendar.timegm(_parse_struct(s)))


def _fmt_num(x: float) -> str:
    try:
        return f"{float(x):.4g}"
    except (TypeError, ValueError):
        return str(x)


class ToolContext:
    def __init__(self, task_context: dict, config: dict, dataset_label: str) -> None:
        self.dataset_label = dataset_label
        self.task_dir = Path(task_context["task_dir"])
        self.tz_offset = int(
            config.get("instruction_tz_offset_hours", config.get("gt_tz_offset_hours", 0)) or 0
        )
        self.answer_candidates: List[str] = list(config.get("answer_candidates", []) or [])
        self.reasons: List[str] = list(config.get("reasons", []) or [])
        self.call_edges: Dict[str, List[str]] = dict(config.get("call_edges", {}) or {})
        self.kpi_inventory: Dict[str, List[str]] = dict(config.get("kpi_inventory", {}) or {})

        self.metric = self._read("metric.csv")
        self.trace = self._read("trace.csv")
        self.log = self._read("log.csv")
        self.error_logs = self._read("error_logs.csv")

        # default window (incident_start/end are UTC strings in query.csv)
        self.default_window: Optional[Tuple[int, int]] = None
        s, e = task_context.get("incident_start"), task_context.get("incident_end")
        if s and e and str(s).strip() and str(e).strip():
            try:
                self.default_window = (_utc_str_to_epoch(s), _utc_str_to_epoch(e))
            except ValueError:
                self.default_window = None

        self._series_cache: Dict[str, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]] = {}
        self._stats_cache: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    # ---- io ---------------------------------------------------------------
    def _read(self, name: str) -> Optional[pd.DataFrame]:
        path = self.task_dir / name
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if df.empty:
            return df
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        return df

    # ---- timezone ---------------------------------------------------------
    def to_epoch(self, local_str: str) -> int:
        """Local instruction-tz string -> real UTC epoch."""
        return int(calendar.timegm(_parse_struct(local_str)) - self.tz_offset * 3600)

    def to_local(self, epoch: float) -> str:
        """UTC epoch -> instruction-tz string '%Y-%m-%d %H:%M:%S'."""
        try:
            return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(float(epoch) + self.tz_offset * 3600))
        except (TypeError, ValueError, OverflowError):
            return str(epoch)

    def _resolve_window(self, start_time: Optional[str], end_time: Optional[str]) -> Tuple[int, int]:
        if start_time and end_time:
            return self.to_epoch(start_time), self.to_epoch(end_time)
        if self.default_window is not None:
            return self.default_window
        raise ValueError(
            "No incident window available. Pass start_time and end_time strings "
            "(format '%Y-%m-%d %H:%M:%S') copied from the incident description."
        )

    # ---- series access & stats -------------------------------------------
    def _source_df(self, source: str) -> Optional[pd.DataFrame]:
        return {"metric": self.metric, "trace": self.trace, "log": self.log}.get(source)

    def _build_series_cache(self, source: str) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
        if source in self._series_cache:
            return self._series_cache[source]
        cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        df = self._source_df(source)
        name_col = _NAME_COL[source]
        if df is not None and not df.empty and {"cmdb_id", name_col, "timestamp", "value"} <= set(df.columns):
            sub = df.dropna(subset=["value", "timestamp"])
            for (cmdb, name), g in sub.groupby(["cmdb_id", name_col], sort=False):
                g = g.sort_values("timestamp")
                cache[(str(cmdb), str(name))] = (g["timestamp"].to_numpy(), g["value"].to_numpy())
        self._series_cache[source] = cache
        return cache

    def _series(self, source: str, cmdb: str, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._build_series_cache(source).get((str(cmdb), str(name)))

    def _global_stats(self, source: str, cmdb: str, name: str) -> Optional[Dict[str, float]]:
        key = (source, str(cmdb), str(name))
        if key in self._stats_cache:
            return self._stats_cache[key]
        series = self._series(source, cmdb, name)
        if series is None or series[1].size == 0:
            return None
        vals = series[1].astype(float)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        std = float(np.std(vals))
        sigma = 1.4826 * mad
        p95 = float(np.percentile(vals, 95))
        p5 = float(np.percentile(vals, 5))
        # robust display scale: MAD-based, floored by a fraction of std so that
        # near-constant series with a lone spike don't produce astronomical sev.
        scale = max(sigma, 0.1 * std, 1e-9)
        stats = {
            "median": med,
            "mad": mad,
            "std": std,
            "scale": scale,
            "p95": p95,
            "p5": p5,
            "hi": max(p95, med + 3 * sigma),
            "lo": min(p5, med - 3 * sigma),
            "n": float(vals.size),
        }
        self._stats_cache[key] = stats
        return stats

    def _distinct(self, source: str, col: str) -> List[str]:
        df = self._source_df(source)
        if df is None or df.empty or col not in df.columns:
            return []
        return [str(v) for v in pd.unique(df[col].dropna())]

    def _candidate_cmdbs(self, source: str) -> List[str]:
        present = set(self._distinct(source, "cmdb_id"))
        if not present:
            return []
        cands = [c for c in self.answer_candidates if c in present]
        return cands if cands else sorted(present)

    # ---- fuzzy name resolution -------------------------------------------
    @staticmethod
    def _resolve_names(query: str, available: List[str], limit: int) -> List[str]:
        if not query:
            return available[:limit]
        q = query.strip().lower()
        exact = [n for n in available if n.lower() == q]
        if exact:
            return exact[:limit]
        subs = [n for n in available if q in n.lower()]
        if subs:
            return subs[:limit]
        close = difflib.get_close_matches(query, available, n=limit, cutoff=0.4)
        return close

    # ---- anomaly detection on one series ---------------------------------
    def _series_anomaly(
        self, source: str, cmdb: str, name: str, s_ep: int, e_ep: int
    ) -> Optional[Dict[str, object]]:
        series = self._series(source, cmdb, name)
        stats = self._global_stats(source, cmdb, name)
        if series is None or stats is None:
            return None
        ts, val = series
        mask = (ts >= s_ep) & (ts <= e_ep)
        wt, wv = ts[mask], val[mask].astype(float)
        if wv.size == 0:
            return None
        hi, lo, med = stats["hi"], stats["lo"], stats["median"]
        flags = [(int(t), float(v)) for t, v in zip(wt, wv) if v > hi or v < lo]
        if not flags:
            return None
        # noise suppression: keep if >=2 flagged points OR one strong breach (>=50% beyond threshold vs median gap)
        peak_t, peak_v = max(flags, key=lambda tv: abs(tv[1] - med))
        gap = (hi - med) if peak_v > hi else (med - lo)
        breach = abs(peak_v - (hi if peak_v > hi else lo))
        strong = gap > 0 and breach >= 0.5 * abs(gap)
        if len(flags) < 2 and not strong:
            return None
        onset_t = min(t for t, _ in flags)
        sev = min(abs(peak_v - med) / stats["scale"], 9999.9)
        return {
            "cmdb": cmdb,
            "name": name,
            "direction": "rise" if peak_v > hi else "drop",
            "onset": onset_t,
            "peak": peak_v,
            "median": med,
            "threshold": hi if peak_v > hi else lo,
            "severity": sev,
            "count": len(flags),
        }


# ============================ TOOLS ===================================== #

def make_tools(ctx: ToolContext) -> Dict[str, object]:
    """Return the tool functions bound to ``ctx`` for injection into SOP code."""

    def whether_is_abnormal_metric(
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        kpi_name: str = "",
        component: Optional[str] = None,
    ) -> str:
        s_ep, e_ep = ctx._resolve_window(start_time, end_time)
        if ctx.metric is None or ctx.metric.empty:
            return "No metric telemetry is available for this task."
        kpis = ctx._resolve_names(kpi_name, ctx._distinct("metric", "kpi_name"), limit=10)
        if not kpis:
            return (
                f"No metric KPI matched {kpi_name!r}. Use get_relevant_metric(query) "
                "to discover valid KPI names first."
            )
        if component:
            comps = ctx._resolve_names(component, ctx._distinct("metric", "cmdb_id"), limit=5)
        else:
            comps = ctx._candidate_cmdbs("metric")
        findings = []
        for c in comps:
            for k in kpis:
                a = ctx._series_anomaly("metric", c, k, s_ep, e_ep)
                if a:
                    findings.append(a)
        if not findings:
            return (
                f"No metric anomalies for KPIs {kpis} across {len(comps)} component(s) "
                f"in window {ctx.to_local(s_ep)} .. {ctx.to_local(e_ep)}."
            )
        findings.sort(key=lambda f: f["severity"], reverse=True)
        lines = [
            f"[ANOMALY] {f['cmdb']} {f['name']}: {f['direction']} to {_fmt_num(f['peak'])} "
            f"at {ctx.to_local(f['onset'])} (median {_fmt_num(f['median'])}, "
            f"thr {_fmt_num(f['threshold'])}, sev {f['severity']:.1f}, n={f['count']})"
            for f in findings[:20]
        ]
        return "\n".join(lines)

    def get_relevant_metric(query: str = "") -> str:
        names = ctx._distinct("metric", "kpi_name")
        inv = list(ctx.kpi_inventory.get("metric", []) or [])
        pool = list(dict.fromkeys(names + inv))
        if not pool:
            return "No metric KPI names available."
        if not query:
            head = pool[:30]
            return "Available metric KPIs (first 30):\n" + "\n".join(f"- {n}" for n in head)
        q = query.strip().lower()
        scored = []
        for n in pool:
            nl = n.lower()
            score = (2.0 if q in nl else 0.0) + difflib.SequenceMatcher(None, q, nl).ratio()
            scored.append((score, n))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [n for sc, n in scored[:30] if sc > 0.2]
        if not top:
            return f"No metric KPI closely matched {query!r}. First few available: " + ", ".join(pool[:15])
        return f"Metric KPIs matching {query!r} (top {len(top)}):\n" + "\n".join(f"- {n}" for n in top)

    def collect_trace(
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        service: Optional[str] = None,
    ) -> str:
        s_ep, e_ep = ctx._resolve_window(start_time, end_time)
        if ctx.trace is None or ctx.trace.empty:
            return "No trace telemetry is available for this task."
        cmdbs = ctx._distinct("trace", "cmdb_id")
        if service:
            cmdbs = ctx._resolve_names(service, cmdbs, limit=20)
        trace_names = ctx._distinct("trace", "trace_name")
        findings = []
        flagged_components = set()
        for c in cmdbs:
            for tn in trace_names:
                a = ctx._series_anomaly("trace", c, tn, s_ep, e_ep)
                if a:
                    findings.append(a)
                    flagged_components.add(c)
        if not findings:
            return (
                f"No trace anomalies across {len(cmdbs)} component(s) in window "
                f"{ctx.to_local(s_ep)} .. {ctx.to_local(e_ep)}."
            )
        findings.sort(key=lambda f: f["severity"], reverse=True)
        lines = [
            f"[TRACE] {f['cmdb']} {f['name']}: {f['direction']} to {_fmt_num(f['peak'])} "
            f"at {ctx.to_local(f['onset'])} (median {_fmt_num(f['median'])}, sev {f['severity']:.1f})"
            for f in findings[:15]
        ]
        # most-downstream faulty component via call_edges
        downstream = ""
        if ctx.call_edges:
            leafs = []
            for c in flagged_components:
                callees = [x for x in ctx.call_edges.get(c, []) if x != c]
                if not any(x in flagged_components for x in callees):
                    leafs.append(c)
            if leafs:
                downstream = "Most-downstream faulty component(s): " + ", ".join(sorted(leafs))
        else:
            downstream = "Downstream analysis unavailable (no call graph for this dataset)."
        return "\n".join(lines + ([downstream] if downstream else []))

    def get_logs(
        component: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> str:
        s_ep, e_ep = ctx._resolve_window(start_time, end_time)
        if ctx.log is None and ctx.error_logs is None:
            return (
                "Log telemetry is not available for this dataset. Rely on metrics "
                "(whether_is_abnormal_metric) and traces (collect_trace) instead."
            )
        comps = None
        if component:
            avail = ctx._distinct("log", "cmdb_id") or ctx._distinct("metric", "cmdb_id")
            comps = ctx._resolve_names(component, avail, limit=3)
        out: List[str] = []
        # aggregated log counters
        if ctx.log is not None and not ctx.log.empty:
            targets = comps if comps else ctx._candidate_cmdbs("log")[:15]
            for c in targets:
                for ln in ctx._distinct("log", "log_name"):
                    a = ctx._series_anomaly("log", c, ln, s_ep, e_ep)
                    if a:
                        out.append(
                            f"[LOG] {c} {ln}: {a['direction']} to {_fmt_num(a['peak'])} "
                            f"at {ctx.to_local(a['onset'])} (median {_fmt_num(a['median'])})"
                        )
        # raw error messages
        if ctx.error_logs is not None and not ctx.error_logs.empty and "message" in ctx.error_logs.columns:
            el = ctx.error_logs
            if "timestamp" in el.columns:
                el = el[(pd.to_numeric(el["timestamp"], errors="coerce") >= s_ep)
                        & (pd.to_numeric(el["timestamp"], errors="coerce") <= e_ep)]
            if comps and "cmdb_id" in el.columns:
                el = el[el["cmdb_id"].astype(str).isin(comps)]
            if not el.empty:
                msgs = el["message"].astype(str).str.slice(0, 200)
                counts = msgs.value_counts().head(10)
                out.append("Raw error messages (top 10 by count):")
                out.extend(f"  {n}x  {m}" for m, n in counts.items())
        if not out:
            return f"No log anomalies or error messages in window {ctx.to_local(s_ep)} .. {ctx.to_local(e_ep)}."
        return "\n".join(out[:25])

    def component_analyze(start_time: Optional[str] = None, end_time: Optional[str] = None) -> str:
        s_ep, e_ep = ctx._resolve_window(start_time, end_time)
        rows = []
        candidates = ctx.answer_candidates or ctx._candidate_cmdbs("metric")
        for c in candidates:
            m_hits, top_kpi, top_sev = 0, "", 0.0
            for k in ctx._distinct("metric", "kpi_name"):
                a = ctx._series_anomaly("metric", c, k, s_ep, e_ep)
                if a:
                    m_hits += 1
                    if a["severity"] > top_sev:
                        top_sev, top_kpi = a["severity"], k
            t_hits = 0
            for tn in ctx._distinct("trace", "trace_name"):
                if ctx._series_anomaly("trace", c, tn, s_ep, e_ep):
                    t_hits += 1
            l_hits = 0
            for ln in ctx._distinct("log", "log_name"):
                if ctx._series_anomaly("log", c, ln, s_ep, e_ep):
                    l_hits += 1
            if m_hits or t_hits or l_hits:
                rows.append((m_hits, t_hits, l_hits, c, top_kpi, top_sev))
        if not rows:
            return (
                f"No component shows anomalies in window {ctx.to_local(s_ep)} .. {ctx.to_local(e_ep)}. "
                "Consider a wider window or specific KPI checks."
            )
        rows.sort(key=lambda r: (r[0] + r[1] + r[2], r[5]), reverse=True)
        lines = [
            f"{c}  metric_anoms={m} (top: {tk or 'n/a'} sev={sev:.1f})  trace_anoms={t}  log_anoms={l}"
            for (m, t, l, c, tk, sev) in rows[:20]
        ]
        return "Component anomaly overview (ranked):\n" + "\n".join(lines)

    return {
        "whether_is_abnormal_metric": whether_is_abnormal_metric,
        "get_relevant_metric": get_relevant_metric,
        "collect_trace": collect_trace,
        "get_logs": get_logs,
        "component_analyze": component_analyze,
    }
