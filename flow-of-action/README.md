# Flow-of-Action Baseline

An implementation of **Flow-of-Action** (SOP-enhanced LLM multi-agent system for
Root Cause Analysis, ICLR'25 submission) as a comparison baseline, run against
the shared benchmark tree at `/data/baselines/RCA-datasets` (OpenRCA Bank /
Market cloudbed-1 / Market cloudbed-2 / Telecom, and RCAEval re2-ob / re2-ss /
re2-tt).

This directory is **self-contained** except for that benchmark dataset tree:
the agent logic, LLM client, scorer, and dataset/prompt plumbing all live in
`flow_of_action/`. Nothing is imported from the sibling `rca-agent` baseline
(the shared harness pieces were copied so the two baselines are directly
comparable but independent).

## Method

Five cooperating agents driven by an SOP flow (paper §2):

- **MainAgent** — orchestrator; each step chooses one action from a proposed set.
- **ActionAgent** — proposes a set of 5 next actions following the flow rules.
- **CodeAgent** — turns a chosen SOP into runnable Python (`generate_sop_code`).
- **ObAgent** — hypothesizes the fault type from an observation.
- **JudgeAgent** — decides whether the root cause has been found (gates `Speak`).

SOP flow tools: `match_sop` (embedding search over SOP names), `generate_sop`,
`generate_sop_code`, `run_sop`, `match_observation`. Data-analysis tools run
inside SOP code over the per-task CSV telemetry, and can also be taken directly
as actions (e.g. `component_analyze`, `collect_trace`) for a single quick check
when no SOP clearly fits (paper ActionAgent rule R4 / Table 4).

Beyond the ActionAgent's LLM proposals, the action set is augmented by a
flow-based rule (paper §2.4): the next action implied by the flow — e.g.
`match_observation` after a successful `run_sop`, or `generate_sop_code` after
`generate_sop` — is always added as a candidate (never removed), so the
MainAgent still chooses but is not starved of the logical next step
(`_augment_action_set` in `flow_of_action/foa_agent.py`).

## Deltas vs. the paper (adaptations for the offline benchmark)

- **Data tools re-implemented over offline CSVs.** The paper's live-K8s tools
  (`whether_is_abnormal_metric`, `collect_trace`, `kubectl_logs`,
  `get_relevant_metric`, `pod/node/service_analyze`) are re-implemented on the
  per-task `metric.csv` / `trace.csv` / `log.csv` / `error_logs.csv` in
  `flow_of_action/tools.py`. `pod/node/service_analyze` are collapsed into
  `component_analyze`; `kubectl_logs` → `get_logs`; live-cluster tools dropped.
- **No historical-incident knowledge base.** `match_observation` is adapted to
  trigger the ObAgent (fault-type hypothesis from the observation + the dataset's
  closed reason vocabulary) followed by the JudgeAgent, instead of retrieving
  historical incidents (which the benchmark does not provide).
- **Seed SOPs are dataset-agnostic** (`flow_of_action/sop_kb.py`), restructured
  from the 4-phase RCA methodology into a two-level hierarchy: 7 general
  procedures (triage / trace / log+KPI / per-class CPU, memory, disk, network)
  and 7 more specific ones (JVM CPU, JVM OOM, network latency vs packet loss,
  disk space vs IO, multi-failure) that `match_observation -> match_sop` drills
  into (the paper's general -> specific progression); `generate_sop` adds more at
  runtime. The paper's own engineer-authored / auto-extracted SOP corpus is not
  available, so these hand-written seeds stand in for it.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local), for
  `match_sop` name similarity only. No lexical fallback — a load failure is
  raised, not silently degraded.
- **Timezone**: tool time arguments are local instruction-tz strings; a single
  conversion point (`ToolContext.to_epoch/to_local`) maps to/from UTC epochs
  (OpenRCA UTC+8, RCAEval UTC).
- **Final answer** reuses the RCA-datasets scoring contract verbatim (JSON key
  order `datetime → component → reason`, multi-answer only when the issue says so).

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY='sk-or-v1-...'
# one-time: cache the embedding model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

Model settings are read from `config.json` (OpenRouter). The current test config
uses `openai/gpt-4o-mini`; switch the `model` field for full comparison runs.

## Usage

From `/data/baselines/flow-of-action`:

```bash
python3 -m flow_of_action.run --list-datasets
python3 -m flow_of_action.run --dry-run --dataset Bank --start-idx 0
python3 -m flow_of_action.run --dataset Bank --start-idx 0 --end-idx 0 --tag foa-smoke
python3 -m flow_of_action.run --auto --tag foa-full            # all datasets
```

Key flags: `--max-step` (default 12), `--timeout` (600s
per task), `--final-token-threshold` (700000), `--num-workers`, `--sample-num`,
`--start-idx` / `--end-idx`.

Outputs mirror the rca-agent schema:

```
outputs/result/<Dataset>/agent-<tag>-<model>.csv     # predictions + scores
outputs/monitor/<Dataset>/.../{history,trajectory,prompt}/
```
