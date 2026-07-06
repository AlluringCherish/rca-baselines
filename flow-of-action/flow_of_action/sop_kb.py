"""SOP knowledge base for Flow-of-Action.

The default seed SOPs mirror the RCA-agent diagnosis workflow at a high level:
preprocess/thresholding -> metric fault identification -> trace localization ->
log/reason analysis -> multi-failure separation. They intentionally avoid
reason-specific benchmark rules, so the FoA run is driven by a general RCA
procedure rather than a hand-crafted SOP per answer label.

Steps reference only the data-analysis tool functions available inside SOP code
(see tools.py). ``generate_sop`` may add more SOPs at runtime for the current
task, but the built-in seed set stays compact and dataset-agnostic.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .embeddings import cosine_matrix, embed


SEED_SOPS: List[Dict[str, object]] = [
    {
        "name": "RCA-Agent Preprocess and Threshold SOP",
        "description": (
            "Establish the incident window and obtain a compact system-wide view "
            "before fault identification."
        ),
        "steps": [
            "1. component_analyze(start_time, end_time): summarize candidate components with metric, trace, and log anomaly counts inside the incident window.",
            "2. get_relevant_metric(''): list available metric KPI names so later checks use valid fields rather than guessed names.",
            "3. The answer is the observations obtained from the former steps.",
        ],
    },
    {
        "name": "RCA-Agent Metric Fault Identification SOP",
        "description": (
            "Identify component-KPI faults using compact metric anomaly scans, "
            "including earliest anomaly timestamps and severity."
        ),
        "steps": [
            "1. whether_is_abnormal_metric(start_time, end_time, '', None): scan all candidate components and KPIs for compact anomaly evidence and onset times.",
            "2. component_analyze(start_time, end_time): compare the metric evidence with trace/log anomaly counts for the same window.",
            "3. The answer is the observations obtained from the former steps.",
        ],
    },
    {
        "name": "RCA-Agent Trace Localization SOP",
        "description": (
            "When several service or container candidates are faulty, use traces "
            "to distinguish root-cause components from downstream victims."
        ),
        "steps": [
            "1. collect_trace(start_time, end_time): summarize abnormal call duration, error-rate, traffic-drop, and downstream information in the incident window.",
            "2. whether_is_abnormal_metric(start_time, end_time, '', None): confirm which trace-localized components are also faulty in metrics.",
            "3. The answer is the observations obtained from the former steps.",
        ],
    },
    {
        "name": "RCA-Agent Log-Based Reason Analysis SOP",
        "description": (
            "Use logs together with metric evidence to disambiguate the failure "
            "reason after candidate components have been narrowed."
        ),
        "steps": [
            "1. get_logs(None, start_time, end_time): summarize log anomalies and raw error messages for candidate components in the incident window.",
            "2. get_relevant_metric('error'): list error, resource, and traffic-related metric fields that may explain the logs.",
            "3. whether_is_abnormal_metric(start_time, end_time, '', None): compare log evidence against metric anomalies and onset times.",
            "4. The answer is the observations obtained from the former steps.",
        ],
    },
    {
        "name": "RCA-Agent Level and Root Selection SOP",
        "description": (
            "Select the final root-cause level and component from compact metric, "
            "trace, and log evidence without relying on healthy downstream symptoms."
        ),
        "steps": [
            "1. component_analyze(start_time, end_time): identify the strongest faulty candidates and their telemetry sources.",
            "2. collect_trace(start_time, end_time): separate faulty upstream/downstream candidates when trace telemetry is available.",
            "3. get_logs(None, start_time, end_time): check whether logs support the same component and reason as the metric/trace evidence.",
            "4. The answer is the observations obtained from the former steps.",
        ],
    },
    {
        "name": "RCA-Agent Multi-Failure Separation SOP",
        "description": (
            "Use this only when the issue indicates multiple independent failures "
            "or the evidence clearly separates distinct root-cause candidates."
        ),
        "steps": [
            "1. component_analyze(start_time, end_time): find whether multiple components show independent anomaly groups.",
            "2. whether_is_abnormal_metric(start_time, end_time, '', None): scan all KPIs and keep distinct onset times and components separate.",
            "3. collect_trace(start_time, end_time): verify whether the candidate groups are independent roots or downstream effects.",
            "4. The answer is the observations obtained from the former steps.",
        ],
    },
]


def sop_to_text(sop: Dict[str, object]) -> str:
    """Render a SOP to the text shown to the LLM (name + numbered steps)."""
    steps = sop.get("steps", [])
    body = "\n".join(str(s) for s in steps)
    return f"Name: {sop['name']}\nSteps:\n{body}"


def _embed_text(sop: Dict[str, object]) -> str:
    desc = sop.get("description", "")
    return f"{sop['name']}. {desc}".strip()


class SOPKnowledgeBase:
    """In-memory SOP store with embedding-based name search.

    A fresh instance is created per task; ``generate_sop`` adds SOPs that live
    only for the duration of that task.
    """

    def __init__(self, seed_sops: Optional[List[Dict[str, object]]] = None) -> None:
        self.sops: List[Dict[str, object]] = [dict(s) for s in (seed_sops or SEED_SOPS)]
        self._matrix = embed([_embed_text(s) for s in self.sops]) if self.sops else None

    def filter(self, predicate) -> None:
        """Keep only SOPs accepted by predicate and rebuild the match index."""
        self.sops = [s for s in self.sops if predicate(s)]
        self._matrix = embed([_embed_text(s) for s in self.sops]) if self.sops else None

    def names(self) -> List[str]:
        return [str(s["name"]) for s in self.sops]

    def get(self, name: str) -> Optional[Dict[str, object]]:
        """Resolve a SOP by exact, then case-insensitive, then substring match."""
        for s in self.sops:
            if str(s["name"]) == name:
                return s
        low = name.strip().lower()
        for s in self.sops:
            if str(s["name"]).lower() == low:
                return s
        for s in self.sops:
            if low and (low in str(s["name"]).lower() or str(s["name"]).lower() in low):
                return s
        return None

    def match(self, query: str, top_k: int = 3, threshold: float = 0.25) -> List[Tuple[Dict[str, object], float]]:
        """Return up to ``top_k`` (sop, score) pairs above ``threshold``."""
        if not self.sops or self._matrix is None:
            return []
        qvec = embed([query])[0]
        scores = cosine_matrix(qvec, self._matrix)
        order = list(scores.argsort()[::-1][:top_k])
        results: List[Tuple[Dict[str, object], float]] = []
        for i in order:
            score = float(scores[i])
            if score >= threshold:
                results.append((self.sops[int(i)], score))
        return results

    def best_score(self, query: str) -> float:
        if not self.sops or self._matrix is None:
            return 0.0
        qvec = embed([query])[0]
        return float(cosine_matrix(qvec, self._matrix).max())

    def add(self, name: str, steps: List[str], description: str = "") -> Dict[str, object]:
        """Add a runtime-generated SOP and update the embedding matrix."""
        import numpy as np

        sop: Dict[str, object] = {"name": name, "description": description, "steps": list(steps)}
        self.sops.append(sop)
        new_vec = embed([_embed_text(sop)])
        self._matrix = new_vec if self._matrix is None else np.vstack([self._matrix, new_vec])
        return sop
