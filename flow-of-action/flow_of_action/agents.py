"""LLM agent wrappers for Flow-of-Action (MainAgent chooser, ActionAgent,
JudgeAgent, ObAgent, CodeAgent, generate_sop, final answer).

JSON parsing policy (user-approved): json.loads (after fence strip) -> if it
fails, ONE re-prompt asking for valid JSON -> if it still fails, regex-extract
the required fields -> if that also fails, raise FoAParseError (surfaced, not
silently worked around).
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional, Tuple

from .api_router import get_chat_completion
from .prompt import foa_prompts as P


class FoAParseError(RuntimeError):
    pass


# --------------------------- parsing helpers --------------------------- #

def _strip_json_fence(text: str) -> str:
    text = str(text or "")
    if "```json" in text:
        m = re.search(r"```json\s*(.*?)\s*```", text, re.S)
        if m:
            return m.group(1).strip()
    if "```" in text:
        m = re.search(r"```\s*(.*?)\s*```", text, re.S)
        if m:
            return m.group(1).strip()
    return text


def _try_parse(text: str) -> Optional[dict]:
    t = _strip_json_fence(text).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", t):
        try:
            obj, _ = decoder.raw_decode(t[m.start():])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _robust_json_call(
    messages: List[dict],
    extractor: Callable[[str], Optional[dict]],
    logger,
    label: str,
    validate: Optional[Callable[[dict], bool]] = None,
) -> Tuple[dict, str]:
    def ok(o):
        return o is not None and (validate is None or validate(o))

    raw = get_chat_completion(messages=messages)
    obj = _try_parse(raw)
    if ok(obj):
        return obj, raw

    logger.warning(f"[{label}] response was not valid JSON; re-prompting once.")
    messages2 = list(messages) + [
        {"role": "assistant", "content": raw or ""},
        {"role": "user", "content": P.JSON_RETRY_SUFFIX},
    ]
    raw2 = get_chat_completion(messages=messages2)
    obj = _try_parse(raw2)
    if ok(obj):
        return obj, raw2

    logger.warning(f"[{label}] still invalid; using regex extraction (approved last resort).")
    obj = extractor(raw2) or extractor(raw)
    if ok(obj):
        return obj, raw2
    raise FoAParseError(
        f"{label}: could not parse a valid JSON object after re-prompt and regex extraction.\n"
        f"--- raw (last) ---\n{(raw2 or '')[:800]}"
    )


# ------------------------------ extractors ----------------------------- #

def _extract_action_set(raw: str) -> Optional[dict]:
    found = {}
    for a in P.ALL_ACTIONS:
        if re.search(rf"\b{re.escape(a)}\b", raw or ""):
            found[a] = "recovered from unparsed ActionAgent response"
    return found or None


def _extract_choice(raw: str) -> Optional[dict]:
    raw = raw or ""
    action = None
    m = re.search(r'"action"\s*:\s*"([^"]+)"', raw)
    if m:
        action = m.group(1).strip()
    if not action:
        for a in P.ALL_ACTIONS:
            if re.search(rf"\b{re.escape(a)}\b", raw):
                action = a
                break
    if not action:
        return None
    mi = re.search(r'"action_input"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    action_input = mi.group(1) if mi else ""
    return {"analysis": "", "action": action, "action_input": action_input}


def _extract_judgement(raw: str) -> Optional[dict]:
    m = re.search(r'judgement"?\s*:?\s*"?\s*(yes|no)', raw or "", re.I)
    if m:
        return {"judgement": m.group(1).capitalize(), "analysis": (raw or "")[:500]}
    return None


def _decode_escapes(s: str) -> str:
    try:
        return s.encode("utf-8").decode("unicode_escape")
    except Exception:
        return s


def _extract_code(raw: str) -> Optional[dict]:
    raw = raw or ""
    m = re.search(r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.S)
    if m:
        return {"code": _decode_escapes(m.group(1))}
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.S)
    if m:
        return {"code": m.group(1).strip()}
    if "answer" in raw and "=" in raw:
        return {"code": raw.strip()}
    return None


def _extract_sop(raw: str) -> Optional[dict]:
    raw = raw or ""
    name_m = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
    quoted_steps = re.findall(r'"(\d+\.\s*[^"]+)"', raw)
    if name_m and quoted_steps:
        return {"name": name_m.group(1), "steps": quoted_steps}
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    step_lines = [l.strip().strip('",') for l in lines if re.match(r"^\"?\d+\.", l.strip())]
    if step_lines:
        name = name_m.group(1) if name_m else (lines[0][:80] if lines else "generated SOP")
        return {"name": name, "steps": step_lines}
    return None


# ------------------------------- agents -------------------------------- #

def propose_action_set(history: str, executed_calls: List[str], n: int, logger) -> Dict[str, str]:
    prompt = P.action_agent_prompt(history, executed_calls, n)
    obj, _ = _robust_json_call(
        [{"role": "user", "content": prompt}],
        _extract_action_set,
        logger,
        "ActionAgent",
        validate=lambda o: isinstance(o, dict) and len(o) > 0,
    )
    return obj


def choose_action(system_prompt: str, history: str, action_set_text: str, step: int, logger) -> Dict[str, str]:
    user = P.main_agent_user(history, action_set_text, step)
    obj, raw = _robust_json_call(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
        _extract_choice,
        logger,
        "MainAgent",
        validate=lambda o: "action" in o and str(o.get("action", "")).strip() != "",
    )
    obj.setdefault("analysis", "")
    obj.setdefault("action_input", "")
    obj["_raw"] = raw
    return obj


def judge_root_cause(history: str, candidates: str, reasons: List[str], ask_datetime: bool, logger) -> Dict[str, str]:
    prompt = P.judge_agent_prompt(history, candidates, reasons, ask_datetime)
    obj, _ = _robust_json_call(
        [{"role": "user", "content": prompt}],
        _extract_judgement,
        logger,
        "JudgeAgent",
        validate=lambda o: "judgement" in o,
    )
    return obj


def ob_hypothesis(observation: str, reasons: List[str], logger) -> str:
    prompt = P.ob_agent_prompt(observation, reasons)
    return get_chat_completion(messages=[{"role": "user", "content": prompt}]) or ""


def code_for_sop(history: str, sop_name: str, sop_text: str, logger, window_hint=None) -> str:
    prompt = P.code_agent_prompt(history, sop_name, sop_text, window_hint)
    obj, _ = _robust_json_call(
        [{"role": "user", "content": prompt}],
        _extract_code,
        logger,
        "CodeAgent",
        validate=lambda o: "code" in o and str(o.get("code", "")).strip() != "",
    )
    return str(obj["code"])


def generate_new_sop(fault_info: str, seed_examples: str, logger) -> Dict[str, object]:
    prompt = P.generate_sop_prompt(fault_info, seed_examples)
    obj, _ = _robust_json_call(
        [{"role": "user", "content": prompt}],
        _extract_sop,
        logger,
        "generate_sop",
        validate=lambda o: "name" in o and "steps" in o and isinstance(o.get("steps"), list) and len(o["steps"]) > 0,
    )
    return obj


def final_answer(objective: str, candidates: str, history: str, forced: bool, logger) -> str:
    speak = P.speak_final(objective, candidates, forced)
    messages = [
        {"role": "system", "content": "You are the MainAgent concluding a root cause analysis."},
        {"role": "user", "content": f"# DIAGNOSIS HISTORY\n{history}\n\n{speak}"},
    ]
    raw = get_chat_completion(messages=messages) or ""
    logger.debug(f"Raw Final Answer:\n{raw}")
    return _strip_json_fence(raw)
