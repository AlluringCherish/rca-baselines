"""run_sop execution sandbox.

CodeAgent emits a short straight-line script whose only calls are the injected
data-analysis tool functions, ending in ``answer = ...``. We run it in a fresh
namespace with plain exec(). On error we return the traceback as the observation
so the flow can regenerate the code (no per-SOP retry cap; bounded only by the
global step/token/time budgets). The harness SIGALRM (LoopTimeout, a
TimeoutError) is re-raised so the run.py timeout still works.
"""
from __future__ import annotations

import traceback
from typing import Dict, Tuple

try:
    import tiktoken
except ImportError:  # tiktoken listed in requirements; absence handled by char estimate
    tiktoken = None

from .tools import ToolContext, make_tools

_MAX_ANSWER_TOKENS = 8192


def _truncate_tokens(text: str, limit: int = _MAX_ANSWER_TOKENS) -> str:
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model("gpt-4")
            toks = enc.encode(text)
            if len(toks) > limit:
                return enc.decode(toks[:limit]) + "\n...[truncated]"
            return text
        except Exception:
            pass
    if len(text) > limit * 4:
        return text[: limit * 4] + "\n...[truncated]"
    return text


def run_sop_code(code: str, ctx: ToolContext) -> Tuple[str, bool]:
    """Execute SOP code. Returns (observation_text, success)."""
    namespace: Dict[str, object] = dict(make_tools(ctx))
    try:
        exec(compile(code, "<sop_code>", "exec"), namespace)
    except BaseException as exc:  # noqa: BLE001 - we translate errors into observations
        if isinstance(exc, (KeyboardInterrupt, SystemExit, TimeoutError)):
            raise
        tb = traceback.format_exc(limit=6)
        return "run_sop ERROR (fix the code and regenerate):\n" + _truncate_tokens(tb, 2000), False
    if "answer" not in namespace or namespace["answer"] is None:
        return "run_sop ERROR: the code did not assign the variable `answer`.", False
    return _truncate_tokens(str(namespace["answer"])), True
