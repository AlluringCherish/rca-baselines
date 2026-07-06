"""Flow-of-Action baseline package (SOP-enhanced multi-agent RCA).

Self-contained except for the shared benchmark dataset tree at
``/data/baselines/RCA-datasets``. All agent logic, the LLM client, the scorer,
and the dataset/prompt plumbing live inside this package.
"""

from .foa_agent import FlowOfActionAgent

__all__ = ["FlowOfActionAgent"]
