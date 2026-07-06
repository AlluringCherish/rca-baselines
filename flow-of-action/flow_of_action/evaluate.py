import os
import re
import argparse


def _metric_reason(reason: str):
    text = re.sub(r"[^a-z0-9]+", " ", str(reason or "").lower()).strip()
    compact = text.replace(" ", "")
    tokens = text.split()

    if "cpu" in tokens:
        return "cpu"
    if "memory" in tokens or "mem" in tokens:
        return "mem"
    if "disk" in tokens or "diskio" in compact:
        return "disk"
    if any(alias in tokens for alias in ("socket", "socker", "sockt", "sock")):
        return "socket"
    if any(term in tokens for term in ("latency", "delay", "loss")):
        return "delay_or_loss"
    return None


def _reason_matches(predicted_reason: str, expected_reason: str, evaluation_mode: str) -> bool:
    if predicted_reason == expected_reason:
        return True

    if evaluation_mode == "reason":
        return False
    if evaluation_mode != "metric":
        raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")

    predicted_metric = _metric_reason(predicted_reason)
    expected_metric = _metric_reason(expected_reason)
    return predicted_metric is not None and predicted_metric == expected_metric

def evaluate(prediction:str, scoring_points:str, evaluation_mode: str = "reason"):
    """
    Evaluate single JSON-like prediction with corresponding scoring points
        args:
            prediction: str, the prediction (JSON-like string)
            scoring_points: str, the scoring points string
    """

    import itertools

    predict_pattern = (
        r'{\s*'
        r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
        r'(?:"root cause component":\s*"(.*?)")?,?\s*'
        r'(?:"root cause reason":\s*"(.*?)")?\s*}'
    )

    predict_matches = re.findall(predict_pattern, prediction)


    predict_results = []
    
    for match in predict_matches:
        datetime_str, component, reason = match
        predict_results.append({
            "root cause occurrence datetime": datetime_str,
            "root cause component": component,
            "root cause reason": reason
        })



    component_pattern = r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)"
    reason_pattern = r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)"
    time_pattern = r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes \(i.e., <=1min\) of ([^\n]+)"

    components = re.findall(component_pattern, scoring_points)
    reasons = re.findall(reason_pattern, scoring_points)
    times = re.findall(time_pattern, scoring_points)

    scoringpoints_length = max(len(components),len(reasons),len(times))
    socres_num = len(components)+len(reasons)+len(times)

    def time_difference(time1_str,time2_str):
        from datetime import datetime
        time_format = "%Y-%m-%d %H:%M:%S"
        
        try:
            time1 = datetime.strptime(time1_str, time_format)
            time2 = datetime.strptime(time2_str, time_format)
        except ValueError:
            return False
        
        time_difference = abs(time1 - time2)
        if time_difference.total_seconds() <= 60:
            return True
        else:
            return False

    scores_get = 0
    passing_criteria = []
    failing_criteria = []

    if socres_num == 0:
        return passing_criteria, failing_criteria, 0.0

    criteria = []
    for i in range(scoringpoints_length):
        criteria.append({
            "root cause component": components[i] if len(components) == scoringpoints_length else None,
            "root cause reason": reasons[i] if len(reasons) == scoringpoints_length else None,
            "root cause occurrence datetime": times[i] if len(times) == scoringpoints_length else None,
        })

    def score_one(pred, expected):
        current_score = 0
        current_passing = []
        if expected["root cause component"] is not None:
            if pred["root cause component"] == expected["root cause component"]:
                current_score += 1
                current_passing.append(expected["root cause component"])
        if expected["root cause reason"] is not None:
            if _reason_matches(pred["root cause reason"], expected["root cause reason"], evaluation_mode):
                current_score += 1
                current_passing.append(expected["root cause reason"])
        if expected["root cause occurrence datetime"] is not None:
            if time_difference(expected["root cause occurrence datetime"], pred["root cause occurrence datetime"]):
                current_score += 1
                current_passing.append(expected["root cause occurrence datetime"])
        return current_score, current_passing

    if predict_results and criteria:
        match_count = min(len(criteria), len(predict_results))
        best_sore = -1
        for expected_indices in itertools.permutations(range(len(criteria)), match_count):
            for pred_indices in itertools.permutations(range(len(predict_results)), match_count):
                current_score = 0
                current_passing = []
                for expected_idx, pred_idx in zip(expected_indices, pred_indices):
                    one_score, one_passing = score_one(predict_results[pred_idx], criteria[expected_idx])
                    current_score += one_score
                    current_passing.extend(one_passing)
                if current_score > best_sore:
                    best_sore = current_score
                    passing_criteria = current_passing
        scores_get = max(best_sore, 0)
    
    failing_criteria = list(set(components+reasons+times)-set(passing_criteria))
    
    final_score = scores_get/socres_num
    bin_score = round(final_score,2)
    return passing_criteria, failing_criteria, bin_score


def file_evaluate(
    prediction_file:str,
    query_file:str,
    report_file:str,
    evaluation_mode: str = "reason",
):
    """
    Evaluate a prediction file of certain dataset with corresponding query file and save the evaluation results to a csv file
        args:
            prediction_file: str, the path of the prediction file (csv, with at least one fields: 'prediction')
            query_file: str, the path of a specific dataset recorded labels (csv)
            report_file: str, the path of the evaluation file (csv)
    """ 
    import pandas as pd

    pred_df = pd.read_csv(prediction_file)
    query_df = pd.read_csv(query_file)
    
    # If prediction file has 'row_id' column, sort by it to align with query file order
    if 'row_id' in pred_df.columns:
        pred_df = pred_df.sort_values('row_id').reset_index(drop=True)
    
    eval_df = pd.DataFrame(columns=["query", "answer", "groundtruth", "passed", "failed", "score", "task_index"])

    if len(pred_df) != len(query_df):
        raise ValueError("The length of prediction file and record file should be the same")

    for idx in range(len(pred_df)):
        prediction = pred_df.loc[idx, "prediction"]
        scoring_points = query_df.loc[idx, "scoring_points"]
        passing_criteria, failing_criteria, score = evaluate(
            prediction,
            scoring_points,
            evaluation_mode=evaluation_mode,
        )
        instruction = query_df.loc[idx, "instruction"]
        task_index = query_df.loc[idx, "task_index"]
        new_row = pd.DataFrame({
            "query": [instruction], 
            "answer": [prediction], 
            "groundtruth": [scoring_points], 
            "passed": [passing_criteria], 
            "failed": [failing_criteria], 
            "score": [score], 
            "task_index": [task_index]
        })
        eval_df = pd.concat([eval_df, new_row], ignore_index=True)


    if os.path.exists(report_file):
        eval_df.to_csv(report_file, mode='a', header=False, index=False)
    else:
        if not os.path.exists(os.path.dirname(report_file)):
            os.makedirs(os.path.dirname(report_file))
        eval_df.to_csv(report_file, index=False)


def report(report_file):
    """
    Visualize the final result of a report after evaluation
        args:
            report_file: str, report after evaluation
    """
    import pandas as pd

    scores = {
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }
    nums = {
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }
    
    partial_scores = {
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }

    df = pd.read_csv(report_file)
    # By default, task_1-3 is easy, task_4-6 is middle, task_7 is hard. For DIY task specifications, you should change this line to modify the difficulty:
    df["difficulty"] = df["task_index"].apply(lambda x: "easy" if int(x.split('_')[1]) <= 3 else "middle" if int(x.split('_')[1]) <= 6 else "hard")
    
    # Calculate strict scores (score == 1.0)
    scores['easy'] += len(df[(df["score"]==1.0) & (df["difficulty"]=="easy")])
    scores['middle'] += len(df[(df["score"]==1.0) & (df["difficulty"]=="middle")])
    scores['hard'] += len(df[(df["score"]==1.0) & (df["difficulty"]=="hard")])
    
    # Calculate partial scores as in the paper: partially correct but not fully solved.
    partial_scores['easy'] += len(df[(df["score"]>0) & (df["score"]<1.0) & (df["difficulty"]=="easy")])
    partial_scores['middle'] += len(df[(df["score"]>0) & (df["score"]<1.0) & (df["difficulty"]=="middle")])
    partial_scores['hard'] += len(df[(df["score"]>0) & (df["score"]<1.0) & (df["difficulty"]=="hard")])

    nums['easy'] += len(df[df["difficulty"]=="easy"])
    nums['middle'] += len(df[df["difficulty"]=="middle"])
    nums['hard'] += len(df[df["difficulty"]=="hard"])

    print("Strict Accuracy (Score == 1.0):")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12}")
    print(f"{'Class':<12}{'Total(#)':<12}{'Correct(#)':<12}{'Accuracy(%)':<12}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12}")
    for key in scores.keys():
        accuracy = scores[key] / nums[key] if nums[key] > 0 else 0
        print(f"{key:<12}{nums[key]:<12}{scores[key]:<12}{accuracy:.2%}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12}")
    total_accuracy = sum(scores.values()) / sum(nums.values()) if sum(nums.values()) > 0 else 0
    print(f"{'Total':<12}{sum(nums.values()):<12}{sum(scores.values()):<12}{total_accuracy:.2%}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12}")
    
    print("\nPartial Accuracy (0.0 < Score < 1.0):")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12}")
    print(f"{'Class':<12}{'Total(#)':<12}{'Correct(#)':<12}{'Accuracy(%)':<12}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12}")
    for key in partial_scores.keys():
        accuracy = partial_scores[key] / nums[key] if nums[key] > 0 else 0
        print(f"{key:<12}{nums[key]:<12}{partial_scores[key]:<12}{accuracy:.2%}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12}")
    total_accuracy = sum(partial_scores.values()) / sum(nums.values()) if sum(nums.values()) > 0 else 0
    print(f"{'Total':<12}{sum(nums.values()):<12}{sum(partial_scores.values()):<12}{total_accuracy:.2%}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12}")
    



if __name__ == '__main__':
    """
    Evaluate a list of prediction files with corresponding query files, save the evaluation results, and display the statistic results.
        args:
            p: list, a list of prediction files to evaluate
            q: list, a list of query files to evaluate
            r: str, report file to save
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, nargs='+', help="a list of prediction files to evaluate")
    parser.add_argument("-q", type=str, nargs='+', help="a list of query files to evaluate")
    parser.add_argument("-r", type=str, help="evaluation file to save")
    parser.add_argument(
        "--evaluation-mode",
        choices=["reason", "metric"],
        default="reason",
        help="reason uses exact reason matching; metric uses metric-category reason matching",
    )
    args = parser.parse_args()

    if len(args.p) != len(args.q):
        raise ValueError("The length of prediction files, query files and evaluation files should be the same")
    if os.path.exists(args.r):
        os.remove(args.r)
    
    for i in range(len(args.p)):
        try:
            file_evaluate(args.p[i], args.q[i], args.r, evaluation_mode=args.evaluation_mode)
        except Exception as e:
            print(f"Error when evaluating the file {args.p[i]}: {e}")
            continue
    
    report(args.r)
