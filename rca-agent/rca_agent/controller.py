import json
import re
from IPython.terminal.embed import InteractiveShellEmbed

from .executor import execute_act

from .api_router import get_chat_completion, get_token_usage

system = """You are the Administrator of a DevOps Assistant system for failure diagnosis. To solve each given issue, you should iteratively instruct an Executor to write and execute Python code for data analysis on telemetry files of target system. By analyzing the execution results, you should approximate the answer step-by-step.

There is some domain knowledge for you:

{background}

{agent}

The issue you are going to solve is:

{objective}

Solve the issue step-by-step. In each step, your response should follow the JSON format below:

{format}

Let's begin."""

format = """{
    "analysis": (Your analysis of the code execution result from Executor in the last step, with detailed reasoning of 'what have been done' and 'what can be derived'. Respond 'None' if it is the first step.),
    "completed": ("True" if you believe the issue is resolved, and an answer can be derived in the 'instruction' field. Otherwise "False"),
    "instruction": (Your instruction for the Executor to perform via code execution in the next step. Do not involve complex multi-step instruction. Keep your instruction atomic, with clear request of 'what to do' and 'how to do'. Respond a summary by yourself if you believe the issue is resolved. Respond a summary by yourself if you believe the issue is resolved. Respond a summary by yourself if you believe the issue is resolved.)
}
(DO NOT contain "```json" and "```" tags. DO contain the JSON object with the brackets "{}" only. Use '\\n' instead of an actual newline character to ensure JSON compatibility when you want to insert a line break within a string.)"""

summary = """Now, you have decided to finish your reasoning process. You should now provide the final answer to the issue. The candidates of possible root cause components and reasons are provided to you. The root cause components and reasons must be selected from the provided candidates.

{cand}

Recall the issue is: {objective}

Please first review your previous reasoning process to infer an exact answer of the issue. Then, summarize your final answer of the root causes using the following JSON format at the end of your response:

```json
{{
    "1": {{
        "root cause occurrence datetime": (if asked by the issue, format: '%Y-%m-%d %H:%M:%S', otherwise ommited),
        "root cause component": (if asked by the issue, one selected from the possible root cause component list, otherwise ommited),
        "root cause reason": (if asked by the issue, one selected from the possible root cause reason list, otherwise ommited),
    }}, (mandatory)
    "2": {{
        "root cause occurrence datetime": (if asked by the issue, format: '%Y-%m-%d %H:%M:%S', otherwise ommited),
        "root cause component": (if asked by the issue, one selected from the possible root cause component list, otherwise ommited),
        "root cause reason": (if asked by the issue, one selected from the possible root cause reason list, otherwise ommited),
    }}, (only if the failure number is "unknown" or "more than one" in the issue)
    ... (only if the failure number is "unknown" or "more than one" in the issue)
}}
```
(Please use "```json" and "```" tags to wrap the JSON object. You only need to provide the elements asked by the issue, and ommited the other fields in the JSON.)
Note that all the root cause components and reasons must be selected from the provided candidates. Do not reply 'unknown' or 'null' or 'not found' in the JSON. Do not be too conservative in selecting the root cause components and reasons. Be decisive to infer a possible answer based on your current observation."""

def _build_basic_prompt(bp, task_context: dict) -> tuple[str, str]:
    if hasattr(bp, "build_schema"):
        background = bp.build_schema(task_context)
    else:
        background = bp.schema

    if hasattr(bp, "build_candidates"):
        candidates = bp.build_candidates(task_context)
    else:
        candidates = bp.cand

    return background, candidates


def _strip_json_fence(text: str) -> str:
    if "```json" not in text:
        return text
    match = re.search(r"```json\s*(.*?)\s*```", text, re.S)
    return match.group(1).strip() if match else text


def _kernel_init_code(task_context: dict) -> str:
    context_json = json.dumps(task_context or {}, ensure_ascii=False)
    return "\n".join([
        "import json",
        "import os",
        "from pathlib import Path",
        "import pandas as pd",
        "pd.set_option('display.width', 427)",
        "pd.set_option('display.max_columns', 20)",
        "pd.set_option('display.max_rows', 20)",
        f"TASK_CONTEXT = json.loads({context_json!r})",
        "DATASET_ROOT = Path(TASK_CONTEXT.get('dataset_root', '.')).resolve()",
        "TASK_DIR = Path(TASK_CONTEXT.get('task_dir', DATASET_ROOT)).resolve()",
        "if TASK_DIR.exists():",
        "    os.chdir(TASK_DIR)",
    ]) + "\n"


def control_loop(
    objective: str,
    plan: str,
    ap,
    bp,
    logger,
    max_step=25,
    max_turn=5,
    task_context=None,
    max_input_tokens_before_final: int | None = None,
) -> str:
    task_context = task_context or {}
    background, candidates = _build_basic_prompt(bp, task_context)
   
    prompt = [
            {'role': 'system', 'content': system.format(objective=objective,
                                                        format=format,
                                                        agent=ap.rules, 
                                                        background=background)},
            {'role': 'user', 'content': "Let's begin."}
        ]

    history = []
    trajectory = []
    observation = "Let's begin."
    status = False
    kernel = InteractiveShellEmbed()
    kernel.run_cell(_kernel_init_code(task_context))

    def finalize_from_current_prompt(reason: str):
        usage = get_token_usage()
        logger.warning(
            f"{reason}. input_tokens={usage.get('input_tokens', 0)}, "
            f"output_tokens={usage.get('output_tokens', 0)}"
        )
        kernel.reset()
        prompt.append({'role': 'user', 'content': summary.format(objective=objective, cand=candidates)})
        answer = get_chat_completion(
            messages=prompt,
        )
        logger.debug(f"Raw Final Answer:\n{answer}")
        prompt.append({'role': 'assistant', 'content': answer})
        answer = _strip_json_fence(answer)
        return answer, trajectory, prompt

    def input_token_threshold_exceeded() -> bool:
        if not max_input_tokens_before_final or max_input_tokens_before_final <= 0:
            return False
        return get_token_usage().get("input_tokens", 0) > max_input_tokens_before_final

    for step in range(max_step):
        if input_token_threshold_exceeded():
            return finalize_from_current_prompt(
                f"Input token threshold exceeded before step {step + 1}; forcing final answer"
            )
        
        note = [{'role': 'user', 'content': f"Continue your reasoning process for the target issue:\n\n{objective}\n\nFollow the rules during issue solving:\n\n{ap.rules}.\n\nResponse format:\n\n{format}"}]
        attempt_actor = []
        response_raw = ""
        try:
            response_raw = get_chat_completion(
                messages=prompt + note,
            )
            response_raw = _strip_json_fence(response_raw)
            logger.debug(f"Raw Response:\n{response_raw}")
            if '"analysis":' not in response_raw or '"instruction":' not in response_raw or '"completed":' not in response_raw:
                logger.warning("Invalid response format. Please provide a valid JSON response.")
                prompt.append({'role': 'assistant', 'content': response_raw})
                prompt.append({'role': 'user', 'content': "Please provide your analysis in requested JSON format."})
                continue
            response = json.loads(response_raw)
            analysis = response['analysis']
            instruction = response['instruction']
            completed = response['completed']
            logger.info('-'*80 + '\n' + f"### Step[{step+1}]\nAnalysis: {analysis}\nInstruction: {instruction}" + '\n' + '-'*80)

            if completed == "True":
                kernel.reset()
                prompt.append({'role': 'assistant', 'content': response_raw})
                prompt.append({'role': 'user', 'content': summary.format(objective=objective,
                                                                                cand=candidates)})
                answer = get_chat_completion(
                    messages=prompt,
                )
                logger.debug(f"Raw Final Answer:\n{answer}")
                prompt.append({'role': 'assistant', 'content': answer})
                answer = _strip_json_fence(answer)
                return answer, trajectory, prompt

            if input_token_threshold_exceeded():
                prompt.append({'role': 'assistant', 'content': response_raw})
                return finalize_from_current_prompt(
                    f"Input token threshold exceeded after controller step {step + 1}; forcing final answer"
                )

            code, result, status, new_history = execute_act(instruction, background, history, attempt_actor, kernel, logger)
            if not status:
                logger.warn(f'Self-Correction failed.')
                observation = "The Executor failed to execute the instruction. Please provide a new instruction."
            observation = f"{result}"
            history = new_history
            trajectory.append({'code': f"# In[{step+1}]:\n\n{code}", 'result': f"Out[{step+1}]:\n```\n{result}```"})
            logger.info('-'*80 + '\n' + f"Step[{step+1}]\n### Observation:\n{result}" + '\n' + '-'*80)
            prompt.append({'role': 'assistant', 'content': response_raw})
            prompt.append({'role': 'user', 'content': observation})

        except Exception as e:
            logger.error(e)
            prompt.append({'role': 'assistant', 'content': response_raw})
            prompt.append({'role': 'user', 'content': f"{str(e)}\nPlease provide your analysis in requested JSON format."})
            if 'context_length_exceeded' in str(e):
                logger.warning("Token length exceeds the limit.")
                kernel.reset()
                return "Token length exceeds. No root cause found.", trajectory, prompt

    logger.warning("Max steps reached. Please check the history.")
    kernel.reset()
    final_prompt = {'role': 'user', 'content': summary.format(objective=objective,
                                                                    cand=candidates).replace('Now, you have decided to finish your reasoning process. ', 'Now, the maximum steps of your reasoning have been reached. ')}
    if prompt[-1]['role'] == 'user':
        prompt[-1]['content'] = final_prompt['content']
    else:
        prompt.append({'role': 'user', 'content': final_prompt['content']})
    answer = get_chat_completion(
        messages=prompt,
    )
    logger.debug(f"Raw Final Answer:\n{answer}")
    prompt.append({'role': 'assistant', 'content': answer})
    answer = _strip_json_fence(answer)
    return answer, trajectory, prompt
