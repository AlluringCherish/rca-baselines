"""FlowOfActionAgent: the MainAgent orchestration loop.

Implements the paper's thought -> action-set -> action -> observation loop over
the SOP flow, with ActionAgent / CodeAgent / ObAgent / JudgeAgent as helpers.
Returns (prediction, trajectory, prompt) so the run.py harness can score and
dump artifacts exactly as it does for the rca-agent baseline.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from . import agents
from .api_router import get_token_usage
from .prompt import foa_prompts as P
from .sop_kb import SEED_SOPS, SOPKnowledgeBase, sop_to_text
from .sop_runner import run_sop_code
from .tools import ToolContext, make_tools


_MAX_ENTRY_CHARS = 1500


def _truncate(text: str, limit: int = _MAX_ENTRY_CHARS) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _normalize_action(action: str) -> Optional[str]:
    a = str(action or "").strip()
    if a in P.ALL_ACTIONS:
        return a
    low = a.lower()
    for known in P.ALL_ACTIONS:
        if known.lower() == low:
            return known
    # substring fallback: prefer the LONGEST matching name so 'generate_sop'
    # does not shadow 'generate_sop_code' (and tool names resolve correctly).
    subs = [k for k in P.ALL_ACTIONS if k.lower() in low]
    if subs:
        return max(subs, key=len)
    return None


def _parse_tool_kwargs(action_input: str) -> Tuple[Optional[dict], Optional[str]]:
    """Parse a data-tool action_input into keyword arguments.

    Returns (kwargs, error). Fails loudly rather than guessing which parameter a
    bare value maps to: only an empty input (-> no args, tools use the default
    window) or a JSON object is accepted.
    """
    s = str(action_input or "").strip()
    if not s:
        return {}, None
    try:
        obj = json.loads(s)
    except Exception:
        return None, "action_input was not valid JSON"
    if not isinstance(obj, dict):
        return None, "action_input must be a JSON object of keyword arguments (not a bare value or list)"
    return {str(k): v for k, v in obj.items()}, None


def _mentions_multiple_failures(instruction: str) -> bool:
    text = str(instruction or "").lower()
    return bool(re.search(r"\b(two|multiple|more than one)\b", text) or "for each failure" in text)


def _augment_action_set(
    action_set: Dict[str, str],
    last_action: Optional[str],
    last_run_ok: Optional[bool],
    last_match_found: Optional[bool],
    judge_found: bool,
    judge_analysis: str,
) -> Dict[str, str]:
    """Constrain the ActionAgent's proposals to the current SOP-flow state."""
    allowed: Dict[str, str] = {}

    if last_action is None:
        allowed["match_sop"] = "flow rule: at the beginning, match a SOP to the fault"
    elif last_action == "match_sop":
        if last_match_found:
            allowed["generate_sop_code"] = "flow rule: a SOP was matched; translate it into code"
        else:
            allowed["generate_sop"] = "flow rule: no SOP matched; generate a new one"
    elif last_action == "generate_sop":
        allowed["generate_sop_code"] = "flow rule: a new SOP exists; translate it into code"
    elif last_action == "generate_sop_code":
        allowed["run_sop"] = "flow rule: code was generated; execute it"
    elif last_action == "run_sop":
        if last_run_ok:
            allowed["match_observation"] = "flow rule: run_sop succeeded; analyze the observation"
        else:
            allowed["generate_sop_code"] = "flow rule: run_sop errored; regenerate the correct code"
    elif last_action == "match_observation":
        allowed["match_sop"] = "flow rule: match a more specific SOP for the hypothesized anomaly class"
    else:
        allowed["match_sop"] = "flow rule: recover by matching a SOP to the fault"

    if judge_found:
        allowed = {"Speak": f"flow rule: root cause appears found: {judge_analysis[:80]}"}

    constrained: Dict[str, str] = {}
    for name, flow_reason in allowed.items():
        constrained[name] = action_set.get(name, flow_reason)
    return constrained


def _fallback_action(action_set: Dict[str, str]) -> Optional[str]:
    for action in P.FLOW_ACTIONS:
        if action in action_set:
            return action
    return next(iter(action_set), None)


class FlowOfActionAgent:
    def __init__(self, basic_prompt, dataset_label: str, config: dict, task_context: Optional[dict] = None) -> None:
        self.bp = basic_prompt
        self.dataset_label = dataset_label
        self.config = config
        self.task_context = task_context or {}
        self.ask_datetime = not dataset_label.startswith("RCAEval/")

    # -------------------------------------------------------------------
    def run(self, instruction: str, logger, max_step: int = 20, final_token_threshold: int = 700000, **_) -> Tuple[str, List[dict], List[dict]]:
        logger.info(f"Objective: {instruction}")
        ctx = ToolContext(self.task_context, self.config, self.dataset_label)
        # incident window expressed in the instruction timezone (what the tools expect),
        # injected into the CodeAgent prompt so it uses the real times, not the example's.
        self._window_hint = (
            (ctx.to_local(ctx.default_window[0]), ctx.to_local(ctx.default_window[1]))
            if ctx.default_window is not None else None
        )
        # exact (tool, args) calls already made, to soft-block identical repeats
        # of a direct tool action (whose result cannot change) — see _run_tool_action.
        self._seen_tool_calls: set = set()
        kb = SOPKnowledgeBase()
        if not _mentions_multiple_failures(instruction):
            kb.filter(lambda sop: "multi-failure" not in str(sop.get("name", "")).lower())
        background = self.bp.build_schema(self.task_context)
        candidates = self.bp.build_candidates(self.task_context)
        system_prompt = P.main_agent_system(instruction, background, candidates, ctx.reasons, self.ask_datetime)
        seed_examples = "\n\n".join(sop_to_text(s) for s in SEED_SOPS[:2])

        transcript: List[Tuple[str, str]] = []
        trajectory: List[dict] = []
        full_log: List[dict] = [{"role": "system", "content": system_prompt}]

        last_action: Optional[str] = None
        last_run_ok: Optional[bool] = None      # result of the previous run_sop (drives flow rule)
        last_match_found: Optional[bool] = None  # whether the previous match_sop found a SOP
        pending_code: Optional[str] = None
        current_sop_name: Optional[str] = None
        executed_sops = set()
        executed_calls: List[str] = []
        judge_found = False
        judge_analysis = ""
        forced = True  # becomes False only if we stop via a Judge-approved Speak

        def render_history() -> str:
            if not transcript:
                return ""
            return "\n".join(f"{label}: {text}" for label, text in transcript)

        def record(step: int, action: str, action_input: str, observation: str, code_cell: str) -> None:
            transcript.append((f"Step {step} | Action", f"{action}({_truncate(action_input, 200)})"))
            transcript.append((f"Step {step} | Observation", _truncate(observation)))
            trajectory.append({"code": code_cell, "result": observation})
            call_sig = f"{action}({_truncate(action_input, 60)})"
            if call_sig not in executed_calls:
                executed_calls.append(call_sig)

        for step in range(1, max_step + 1):
            if get_token_usage().get("input_tokens", 0) > final_token_threshold:
                logger.warning(f"Input token threshold exceeded before step {step}; forcing final answer.")
                break

            history = render_history()

            # 1) ActionAgent proposes the action set, then flow rules augment it
            action_set = agents.propose_action_set(history, executed_calls, 5, logger)
            action_set = _augment_action_set(
                action_set, last_action, last_run_ok, last_match_found, judge_found, judge_analysis
            )
            action_set_text = "\n".join(f"- {name}: {reason}" for name, reason in action_set.items())

            # 2) MainAgent chooses one action
            choice = agents.choose_action(system_prompt, history, action_set_text, step, logger)
            full_log.append({"role": "assistant", "content": choice.get("_raw", "")})
            raw_action = choice.get("action", "")
            raw_action_input = choice.get("action_input", "")
            if isinstance(raw_action_input, (dict, list)):
                action_input = json.dumps(raw_action_input)
            else:
                action_input = str(raw_action_input or "")
            action = _normalize_action(raw_action)
            logger.info(f"### Step[{step}] action={raw_action} normalized={action} input={_truncate(action_input, 120)}")

            if action is None:
                observation = (
                    f"Invalid action {raw_action!r}. Choose exactly one of: {', '.join(P.ALL_ACTIONS)}."
                )
                record(step, str(raw_action), action_input, observation, f"# Step {step}: invalid action {raw_action!r}")
                full_log.append({"role": "user", "content": observation})
                last_action = None
                continue

            if action not in action_set:
                fallback = _fallback_action(action_set)
                if fallback is None:
                    observation = (
                        f"Invalid action {raw_action!r}. No valid flow action was available."
                    )
                    record(step, str(raw_action), action_input, observation, f"# Step {step}: invalid action {raw_action!r}")
                    full_log.append({"role": "user", "content": observation})
                    last_action = None
                    continue
                logger.warning(
                    f"Action {raw_action!r} is not allowed in the current SOP-flow state; "
                    f"forcing {fallback!r}."
                )
                action = fallback
                action_input = ""

            if action == "Speak" and judge_found:
                forced = False
                logger.info("JudgeAgent approved root cause; stopping to Speak.")
                break

            # 3) dispatch
            result = self._dispatch(
                action, action_input, instruction, ctx, kb, seed_examples, candidates,
                render_history, pending_code, current_sop_name, executed_sops,
                judge_found, judge_analysis, step, logger,
            )
            observation = result["observation"]
            pending_code = result["pending_code"]
            current_sop_name = result["current_sop_name"]
            judge_found = result["judge_found"]
            judge_analysis = result["judge_analysis"]
            record(step, action, action_input, observation, result["code_cell"])
            full_log.append({"role": "user", "content": _truncate(observation)})
            last_action = action
            last_run_ok = result.get("run_ok")
            last_match_found = result.get("match_found")

        prediction = agents.final_answer(instruction, candidates, render_history(), forced, logger)
        full_log.append({"role": "assistant", "content": prediction})
        logger.info(f"Result: {prediction}")
        return prediction, trajectory, full_log

    # -------------------------------------------------------------------
    def _dispatch(
        self, action, action_input, instruction, ctx, kb, seed_examples, candidates,
        render_history, pending_code, current_sop_name, executed_sops,
        judge_found, judge_analysis, step, logger,
    ) -> Dict[str, object]:
        code_cell = f"# Step {step}: {action}({_truncate(action_input, 120)})"
        run_ok: Optional[bool] = None
        match_found: Optional[bool] = None

        if action == "match_sop":
            query = action_input.strip() or instruction
            matches = kb.match(query, top_k=3)
            match_found = bool(matches)
            if matches:
                lines = []
                for sop, score in matches:
                    tag = " (already executed)" if sop["name"] in executed_sops else ""
                    lines.append(f"- {sop['name']} (score {score:.2f}){tag}\n  {sop_to_text(sop)}")
                observation = "Matched SOPs:\n" + "\n".join(lines)
            else:
                observation = (
                    f"No SOP matched query {query[:80]!r} (best score {kb.best_score(query):.2f} below threshold). "
                    "Consider generate_sop to create a new SOP."
                )

        elif action == "generate_sop":
            fault_info = action_input.strip() or instruction
            sop = agents.generate_new_sop(fault_info, seed_examples, logger)
            kb.add(str(sop["name"]), [str(s) for s in sop["steps"]], str(sop.get("description", "")))
            observation = "Generated a new SOP:\n" + sop_to_text(sop)

        elif action == "generate_sop_code":
            sop = kb.get(action_input.strip())
            if sop is None:
                matches = kb.match(action_input.strip() or instruction, top_k=1)
                sop = matches[0][0] if matches else None
            if sop is None:
                observation = "No SOP found to generate code for. Use match_sop or generate_sop first."
            else:
                # regenerating code for a SOP whose previous code was never run
                # (run_sop clears pending_code on success) -> the model is spinning.
                redundant = pending_code is not None and current_sop_name == str(sop["name"])
                code = agents.code_for_sop(
                    render_history(), str(sop["name"]), sop_to_text(sop), logger,
                    window_hint=getattr(self, "_window_hint", None),
                )
                pending_code = code
                current_sop_name = str(sop["name"])
                code_cell = f"# Step {step}: generate_sop_code for {current_sop_name!r}\n{code}"
                nudge = (
                    "\n\nNOTE: you already generated code for this SOP and have not run it yet. "
                    "Execute it now with run_sop to get real results — regenerating again will not "
                    "help. If a KPI name looked wrong, pass a keyword like 'cpu' or an empty string "
                    "'' (which scans all KPIs), never a placeholder."
                    if redundant else ""
                )
                observation = f"Generated code for SOP '{current_sop_name}':\n{code}{nudge}"

        elif action == "run_sop":
            code = action_input if ("answer" in action_input and "=" in action_input) else pending_code
            if not code:
                observation = "No SOP code available to run. Call generate_sop_code first."
                run_ok = False
            else:
                code_cell = f"# Step {step}: run_sop\n{code}"
                obs_text, ok = run_sop_code(code, ctx)
                run_ok = ok
                if ok:
                    if current_sop_name:
                        executed_sops.add(current_sop_name)
                    # clear the pending code so a bare run_sop cannot re-run the
                    # same script indefinitely; the next run needs fresh code.
                    pending_code = None
                    current_sop_name = None
                observation = obs_text

        elif action in P.DATA_TOOL_ACTIONS:
            observation = self._run_tool_action(action, action_input, ctx)
            code_cell = f"# Step {step}: {action}({_truncate(action_input, 120)})"

        elif action == "match_observation":
            obs_query = action_input.strip() or (render_history() or instruction)
            ob = agents.ob_hypothesis(obs_query, ctx.reasons, logger)
            judge_input = render_history() + f"\nStep {step} | ObAgent hypothesis: {ob}"
            jr = agents.judge_root_cause(judge_input, candidates, ctx.reasons, self.ask_datetime, logger)
            if str(jr.get("judgement", "")).strip().lower().startswith("y"):
                judge_found = True
                judge_analysis = str(jr.get("analysis", ""))
            observation = (
                f"ObAgent: {ob}\nJudgeAgent: {jr.get('judgement', '?')} - {jr.get('analysis', '')}"
            )

        else:  # normalized to a known action but unhandled (only Speak reaches here pre-judge)
            observation = (
                "Speak is only available once the JudgeAgent has confirmed the root cause. "
                "Continue diagnosing."
            )

        return {
            "observation": observation,
            "code_cell": code_cell,
            "pending_code": pending_code,
            "current_sop_name": current_sop_name,
            "judge_found": judge_found,
            "judge_analysis": judge_analysis,
            "run_ok": run_ok,
            "match_found": match_found,
        }

    # -------------------------------------------------------------------
    def _run_tool_action(self, action: str, action_input: str, ctx: ToolContext) -> str:
        """Execute a data-analysis tool directly as an action.

        Arguments come from action_input as a JSON object. Bad arguments are
        surfaced back to the model (with the valid signature) instead of being
        guessed at, so the model corrects the call on the next step.
        """
        # identical (tool, args) repeat -> the result is deterministic and cannot
        # change, so re-running wastes a step. Return a soft nudge instead (the
        # model is still free to pick a different action, per the soft flow).
        seen = getattr(self, "_seen_tool_calls", None)
        call_key = (action, str(action_input or "").strip())
        if seen is not None and call_key in seen:
            return (
                f"You already called {action} with these exact arguments at an earlier step; "
                "its result is unchanged. Do not repeat identical calls — check a different "
                "KPI / component / time window, run match_observation to analyze what you have, "
                "or Speak if the root cause is already pinned."
            )
        kwargs, err = _parse_tool_kwargs(action_input)
        sig = P.TOOL_SIGNATURES.get(action, action)
        if err is not None:
            win = getattr(self, "_window_hint", None)
            hint = (
                f" The incident window is start_time='{win[0]}', end_time='{win[1]}'."
                if win and win[0] and win[1] else ""
            )
            return f"{action} argument error: {err}. Pass action_input as a JSON object of keyword arguments for {sig}.{hint}"
        try:
            fn = make_tools(ctx)[action]
            result = str(fn(**kwargs))
        except (KeyboardInterrupt, SystemExit, TimeoutError):
            raise
        except BaseException as exc:  # noqa: BLE001 - surface tool errors to the model
            return f"{action} call failed: {type(exc).__name__}: {exc}. Valid signature: {sig}."
        if seen is not None:
            seen.add(call_key)
        return result
