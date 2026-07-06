from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

try:
    from loguru import logger
except ImportError:
    import logging

    class _FallbackLogger:
        def __init__(self) -> None:
            self._logger = logging.getLogger("rca_agent")
            self._logger.setLevel(logging.DEBUG)

        def remove(self, *args, **kwargs) -> None:
            for handler in list(self._logger.handlers):
                self._logger.removeHandler(handler)

        def add(self, sink, colorize=False, enqueue=False, level="INFO", **kwargs) -> None:
            if hasattr(sink, "write"):
                handler = logging.StreamHandler(sink)
            else:
                Path(sink).parent.mkdir(parents=True, exist_ok=True)
                handler = logging.FileHandler(sink, encoding="utf-8")
            handler.setLevel(getattr(logging, str(level).upper(), logging.INFO))
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self._logger.addHandler(handler)

        def debug(self, message, *args, **kwargs) -> None:
            self._logger.debug(message)

        def info(self, message, *args, **kwargs) -> None:
            self._logger.info(message)

        def warning(self, message, *args, **kwargs) -> None:
            self._logger.warning(message)

        def warn(self, message, *args, **kwargs) -> None:
            self.warning(message)

        def error(self, message, *args, **kwargs) -> None:
            self._logger.error(message)

    logger = _FallbackLogger()

try:
    from nbformat import v4 as nbf
except ImportError:
    nbf = None

from .api_router import configs, get_token_usage, reset_token_usage
from .evaluate import evaluate
from .prompt import agent_prompt as ap
from .prompt.basic_prompt import make_basic_prompt
from .rca_agent import RCA_Agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "RCA-datasets"

DATASETS = {
    "Bank": "configs/dataset_openrca_bank.yaml",
    "Market/cloudbed-1": "configs/dataset_openrca_market_cb1.yaml",
    "Market/cloudbed-2": "configs/dataset_openrca_market_cb2.yaml",
    "RCAEval/re2-ob": "configs/dataset_rcaeval_re2_ob.yaml",
    "RCAEval/re2-ss": "configs/dataset_rcaeval_re2_ss.yaml",
    "RCAEval/re2-tt": "configs/dataset_rcaeval_re2_tt.yaml",
    "Telecom": "configs/dataset_openrca_telecom.yaml",
}

RESULT_COLUMNS = [
    "problem_number",
    "row_id",
    "sample_id",
    "task_index",
    "instruction",
    "prediction",
    "execution_time_sec",
    "input_tokens",
    "output_tokens",
    "groundtruth",
    "passed",
    "failed",
    "score",
    "task_dir",
]


class LoopTimeout(TimeoutError):
    pass


def _timeout_handler(signum, frame):
    raise LoopTimeout("Loop execution exceeded the time limit")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dataset_paths(dataset_root: Path, dataset: str) -> tuple[Dict[str, Any], Path, Path, Path | None]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {', '.join(DATASETS)}")
    config_path = dataset_root / DATASETS[dataset]
    config = _load_yaml(config_path)
    raw_path = dataset_root / config["raw_dataset_path"]
    query_path = raw_path / "query.csv"
    record_path = raw_path / "record.csv"
    ground_truth_path = raw_path / "ground_truth.csv"
    if not query_path.exists():
        raise FileNotFoundError(f"Missing query file: {query_path}")
    if not record_path.exists():
        raise FileNotFoundError(f"Missing record file: {record_path}")
    return config, query_path, record_path, ground_truth_path if ground_truth_path.exists() else None


def _task_context(dataset_root: Path, dataset: str, config: Dict[str, Any], idx: int, row: pd.Series) -> Dict[str, Any]:
    source_task_id = str(row.get("task_id", f"task_7-{idx}"))
    task_id = config.get("task_id_template", "{task_id}").format(
        idx=idx,
        task_id=source_task_id,
    )
    task_dir_rel = config["task_dir_template"].format(
        telemetry_root=config["telemetry_root"],
        idx=idx,
        task_id=task_id,
    )
    task_dir = dataset_root / task_dir_rel
    return {
        "dataset": dataset,
        "idx": int(idx),
        "task_id": task_id,
        "dataset_root": str(dataset_root),
        "task_dir": str(task_dir),
        "task_dir_relative": task_dir_rel,
    }


def _format_ground_truth(idx: int, record_df: pd.DataFrame, ground_truth_df: pd.DataFrame | None) -> str:
    if ground_truth_df is not None and "idx" in ground_truth_df.columns:
        rows = ground_truth_df[ground_truth_df["idx"] == idx]
        if not rows.empty:
            lines = []
            for _, row in rows.iterrows():
                answer_no = row.get("answer_no", len(lines) + 1)
                component = row.get("component", "")
                reason = row.get("reason", "")
                dt = row.get("datetime", "")
                lines.append(f"answer {answer_no}: component={component}; reason={reason}; datetime={dt}")
            return "\n".join(lines)
    if idx < len(record_df):
        row = record_df.iloc[idx]
        return "\n".join(f"{col}: {row[col]}" for col in record_df.columns if col != "description")
    return ""


def _format_scoring_points(idx: int, row: pd.Series, ground_truth_df: pd.DataFrame | None) -> str:
    if "scoring_points" in row.index and pd.notna(row.get("scoring_points")):
        return str(row["scoring_points"])

    if ground_truth_df is not None and "idx" in ground_truth_df.columns:
        rows = ground_truth_df[ground_truth_df["idx"] == idx]
        if not rows.empty:
            lines = []
            multi_answer = len(rows) > 1
            for answer_idx, (_, answer) in enumerate(rows.iterrows(), 1):
                label = f"{answer_idx}-th" if multi_answer else "only"
                component = answer.get("component", "")
                reason = answer.get("reason", "")
                if pd.notna(component) and str(component):
                    lines.append(f"The {label} predicted root cause component is {component}")
                if pd.notna(reason) and str(reason):
                    lines.append(f"The {label} predicted root cause reason is {reason}")
            return "\n".join(lines)

    raise ValueError(f"No scoring points available for task index {idx}")


def _extract_framework_answer(prediction: str) -> str:
    text = str(prediction or "")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False, indent=4)
    return text


def _run_metadata(
    dataset: str,
    idx: int,
    sample_idx: int,
    task_index: str,
    context: Dict[str, Any],
    execution_time_sec: float,
    token_usage: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "problem_number": int(idx),
        "sample_id": int(sample_idx),
        "task_index": task_index,
        "telemetry_task_id": context.get("task_id", ""),
        "execution_time_sec": execution_time_sec,
        "input_tokens": int(token_usage.get("input_tokens", 0)),
        "output_tokens": int(token_usage.get("output_tokens", 0)),
    }


def _append_eval(eval_file: Path, row: Dict[str, Any]) -> None:
    eval_file.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if eval_file.exists():
        old_df = pd.read_csv(eval_file)
        if "framework_answer" in old_df.columns and "prediction" not in old_df.columns:
            old_df = old_df.rename(columns={"framework_answer": "prediction"})
        elif "framework_answer" in old_df.columns:
            old_df = old_df.drop(columns=["framework_answer"])
        out = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out = new_df
    ordered_columns = [col for col in RESULT_COLUMNS if col in out.columns]
    ordered_columns += [col for col in out.columns if col not in ordered_columns]
    out = out[ordered_columns]
    out.to_csv(eval_file, index=False)


def _selected_indices(query_df: pd.DataFrame, start_idx: int, end_idx: int | None) -> list[int]:
    final_idx = len(query_df) - 1 if end_idx is None else min(end_idx, len(query_df) - 1)
    if final_idx < start_idx:
        return []
    return [int(idx) for idx in query_df.index if start_idx <= int(idx) <= final_idx]


def _split_indices(indices: list[int], worker_count: int) -> list[list[int]]:
    return [indices[i::worker_count] for i in range(worker_count) if indices[i::worker_count]]


def _merge_eval_shards(eval_file: Path, shard_files: list[Path]) -> None:
    frames = []
    for shard_file in shard_files:
        if shard_file.exists():
            shard_df = pd.read_csv(shard_file)
            if "framework_answer" in shard_df.columns and "prediction" not in shard_df.columns:
                shard_df = shard_df.rename(columns={"framework_answer": "prediction"})
            elif "framework_answer" in shard_df.columns:
                shard_df = shard_df.drop(columns=["framework_answer"])
            if not shard_df.empty:
                frames.append(shard_df)
    if not frames:
        return

    merged = pd.concat(frames, ignore_index=True)
    sort_cols = [col for col in ("problem_number", "row_id", "sample_id") if col in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    eval_file.parent.mkdir(parents=True, exist_ok=True)
    if eval_file.exists():
        old_df = pd.read_csv(eval_file)
        if "framework_answer" in old_df.columns and "prediction" not in old_df.columns:
            old_df = old_df.rename(columns={"framework_answer": "prediction"})
        elif "framework_answer" in old_df.columns:
            old_df = old_df.drop(columns=["framework_answer"])
        merged = pd.concat([old_df, merged], ignore_index=True)
    ordered_columns = [col for col in RESULT_COLUMNS if col in merged.columns]
    ordered_columns += [col for col in merged.columns if col not in ordered_columns]
    merged = merged[ordered_columns]
    merged.to_csv(eval_file, index=False)


def _print_result_summary(eval_file: Path) -> None:
    if not eval_file.exists():
        print(f"No result file found: {eval_file}")
        return

    df = pd.read_csv(eval_file)
    if df.empty:
        print(f"No result rows found: {eval_file}")
        return

    score = pd.to_numeric(df.get("score"), errors="coerce").fillna(0.0)
    correct_pct = (score == 1.0).mean() * 100
    partial_pct = ((score > 0.0) & (score < 1.0)).mean() * 100
    execution_time = pd.to_numeric(df.get("execution_time_sec"), errors="coerce").mean()
    input_tokens = pd.to_numeric(df.get("input_tokens"), errors="coerce").mean()
    output_tokens = pd.to_numeric(df.get("output_tokens"), errors="coerce").mean()

    print("\nFinal Summary")
    print(f"Result file: {eval_file}")
    print(f"Correct(%): {correct_pct:.2f}")
    print(f"Partial(%): {partial_pct:.2f}")
    print(f"Execution time(s): {execution_time:.2f}")
    print(f"input_tokens: {int(input_tokens) if pd.notna(input_tokens) else 0}")
    print(f"output_tokens: {int(output_tokens) if pd.notna(output_tokens) else 0}")


def _new_notebook() -> Dict[str, Any]:
    if nbf:
        return nbf.new_notebook()
    return {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def _append_code_cell(nb: Dict[str, Any], source: str) -> None:
    if nbf:
        nb.cells.append(nbf.new_code_cell(source))
    else:
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source,
        })


def _append_markdown_cell(nb: Dict[str, Any], source: str) -> None:
    if nbf:
        nb.cells.append(nbf.new_markdown_cell(source))
    else:
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source,
        })


def _dry_run(dataset_root: Path, dataset: str, start_idx: int) -> None:
    config, query_path, record_path, ground_truth_path = _dataset_paths(dataset_root, dataset)
    query_df = pd.read_csv(query_path)
    if start_idx >= len(query_df):
        raise IndexError(f"start index {start_idx} is outside query length {len(query_df)}")
    context = _task_context(dataset_root, dataset, config, start_idx, query_df.iloc[start_idx])
    payload = {
        "dataset": dataset,
        "dataset_root": str(dataset_root),
        "query_path": str(query_path),
        "record_path": str(record_path),
        "ground_truth_path": str(ground_truth_path) if ground_truth_path else None,
        "task_context": context,
        "task_dir_exists": Path(context["task_dir"]).exists(),
        "task_files": sorted(p.name for p in Path(context["task_dir"]).glob("*.csv"))
        if Path(context["task_dir"]).is_dir() else [],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_dataset_indices(
    args: argparse.Namespace,
    uid: str,
    dataset: str,
    indices: list[int],
    eval_file_override: str | None = None,
    worker_id: int | None = None,
) -> None:
    dataset_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    config, query_path, record_path, ground_truth_path = _dataset_paths(dataset_root, dataset)
    query_df = pd.read_csv(query_path)
    record_df = pd.read_csv(record_path)
    ground_truth_df = pd.read_csv(ground_truth_path) if ground_truth_path else None
    basic_prompt = make_basic_prompt(dataset, config)

    model_name = _safe_name(str(configs.get("MODEL") or "unconfigured"))
    dataset_name = _safe_name(dataset)
    eval_file = Path(eval_file_override).resolve() if eval_file_override else (
        output_root / "result" / dataset_name / f"agent-{args.tag}-{model_name}.csv"
    )
    obs_path = output_root / "monitor" / dataset_name / f"agent-{args.tag}-{model_name}" / uid
    for child in ("history", "trajectory", "prompt"):
        (obs_path / child).mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGALRM, _timeout_handler)
    logger.info(f"Using dataset: {dataset}")
    logger.info(f"Using model: {configs.get('MODEL')}")
    logger.info(f"Dataset root: {dataset_root}")
    if worker_id is not None:
        logger.info(f"Worker {worker_id} task indices: {indices}")

    selected = set(indices)
    for idx, row in query_df.iterrows():
        if int(idx) not in selected:
            continue

        instruction = row["instruction"]
        scoring_points = _format_scoring_points(idx, row, ground_truth_df)
        task_index = str(row["task_index"])
        context = _task_context(dataset_root, dataset, config, idx, row)
        if not Path(context["task_dir"]).is_dir():
            raise FileNotFoundError(f"Task telemetry directory does not exist: {context['task_dir']}")

        for sample_idx in range(args.sample_num):
            run_id = f"{uid}_#{idx}-{sample_idx}"
            nb = _new_notebook()
            nbfile = obs_path / "trajectory" / f"{run_id}.ipynb"
            promptfile = obs_path / "prompt" / f"{run_id}.json"
            logfile = obs_path / "history" / f"{run_id}.log"

            logger.remove()
            logger.add(sys.stdout, colorize=True, enqueue=True, level=args.log_level)
            logger.add(logfile, colorize=True, enqueue=True, level="DEBUG")
            logger.info("#" * 80)
            logger.info(f"{run_id}: {dataset} idx={idx} {task_index}")
            logger.info("#" * 80)

            try:
                if args.timeout > 0:
                    signal.alarm(args.timeout)
                reset_token_usage()
                started_at = time.perf_counter()
                agent = RCA_Agent(ap, basic_prompt, task_context=context)
                prediction, trajectory, prompt = agent.run(
                    instruction,
                    logger,
                    max_step=args.controller_max_step,
                    max_turn=args.controller_max_turn,
                    final_token_threshold=args.controller_final_token_threshold,
                )
                execution_time_sec = round(time.perf_counter() - started_at, 3)
                token_usage = get_token_usage()
                signal.alarm(0)

                for step in trajectory:
                    _append_code_cell(nb, step["code"])
                    _append_markdown_cell(nb, f"```\n{step['result']}\n```")
                with nbfile.open("w", encoding="utf-8") as f:
                    json.dump(nb, f, ensure_ascii=False, indent=4)
                with promptfile.open("w", encoding="utf-8") as f:
                    json.dump({
                        "messages": prompt,
                        "run_metadata": _run_metadata(
                            dataset,
                            idx,
                            sample_idx,
                            task_index,
                            context,
                            execution_time_sec,
                            token_usage,
                        ),
                    }, f, ensure_ascii=False, indent=4)

                framework_answer = _extract_framework_answer(prediction)
                passed, failed, score = evaluate(
                    framework_answer,
                    scoring_points,
                    evaluation_mode=args.evaluation_mode,
                )
                eval_row = {
                    "problem_number": idx,
                    "row_id": idx,
                    "sample_id": sample_idx,
                    "task_index": task_index,
                    "instruction": instruction,
                    "prediction": prediction,
                    "execution_time_sec": execution_time_sec,
                    "input_tokens": int(token_usage.get("input_tokens", 0)),
                    "output_tokens": int(token_usage.get("output_tokens", 0)),
                    "groundtruth": _format_ground_truth(idx, record_df, ground_truth_df),
                    "passed": "\n".join(passed),
                    "failed": "\n".join(failed),
                    "score": score,
                    "task_dir": context["task_dir_relative"],
                }
                _append_eval(eval_file, eval_row)

                logger.info(f"Framework Answer: {framework_answer}")
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Scoring Points: {scoring_points}")
                logger.info(f"Passed Criteria: {passed}")
                logger.info(f"Failed Criteria: {failed}")
                logger.info(f"Score: {score}")
            except LoopTimeout:
                signal.alarm(0)
                logger.error(f"Loop {sample_idx} exceeded the time limit and was skipped")
                continue
            except Exception:
                signal.alarm(0)
                raise


def run_dataset(args, uid: str, dataset: str) -> None:
    dataset_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    _, query_path, _, _ = _dataset_paths(dataset_root, dataset)
    query_df = pd.read_csv(query_path)
    indices = _selected_indices(query_df, args.start_idx, args.end_idx)
    if not indices:
        logger.warning(f"No task indices selected for {dataset}")
        return

    model_name = _safe_name(str(configs.get("MODEL") or "unconfigured"))
    dataset_name = _safe_name(dataset)
    eval_file = output_root / "result" / dataset_name / f"agent-{args.tag}-{model_name}.csv"
    worker_count = max(1, int(args.num_workers))

    if worker_count == 1 or len(indices) == 1:
        _run_dataset_indices(args, uid, dataset, indices, str(eval_file))
        _print_result_summary(eval_file)
        return

    worker_count = min(worker_count, len(indices))
    chunks = _split_indices(indices, worker_count)
    shard_dir = output_root / "result" / dataset_name / "shards" / uid
    shard_files = [
        shard_dir / f"agent-{args.tag}-{model_name}.worker-{worker_id:02d}.csv"
        for worker_id in range(len(chunks))
    ]

    logger.info(f"Running {dataset} with {len(chunks)} workers over {len(indices)} tasks")
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {
            executor.submit(_run_dataset_indices, args, uid, dataset, chunk, str(shard_file), worker_id): worker_id
            for worker_id, (chunk, shard_file) in enumerate(zip(chunks, shard_files))
        }
        for future in as_completed(futures):
            worker_id = futures[future]
            future.result()
            logger.info(f"Worker {worker_id} completed")

    _merge_eval_shards(eval_file, shard_files)
    logger.info(f"Merged {len(shard_files)} result shards into {eval_file}")
    _print_result_summary(eval_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone RCA-Agent on RCA-datasets.")
    parser.add_argument("--dataset", type=str, default="Market/cloudbed-1", choices=sorted(DATASETS))
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-root", type=str, default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--sample-num", type=int, default=1)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=150)
    parser.add_argument("--controller-max-step", type=int, default=25)
    parser.add_argument("--controller-max-turn", type=int, default=5)
    parser.add_argument("--controller-final-token-threshold", type=int, default=700000)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--tag", type=str, default="rca")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--evaluation-mode",
        choices=["reason", "metric"],
        default="reason",
        help="reason uses exact reason matching; metric uses metric-category reason matching",
    )
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.data_root).resolve()
    if args.list_datasets:
        for dataset in sorted(DATASETS):
            print(dataset)
        return
    if args.dry_run:
        _dry_run(dataset_root, args.dataset, args.start_idx)
        return

    uid = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    datasets = sorted(DATASETS) if args.auto else [args.dataset]
    for dataset in datasets:
        run_dataset(args, uid, dataset)


if __name__ == "__main__":
    main()
