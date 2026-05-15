import os
import json
import argparse
import re
import sys
from pathlib import Path
import numpy as np


def scorer(dataset, predictions, answers):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        if dataset not in ["qa_1", "qa_2"]:
            prediction =  sorted(x.strip() for x in prediction.split(",")) # get list numbers and sort
            ground_truths = sorted(x.strip() for x in ground_truths) # get list numbers and sort
            overlap = set(prediction) & set(ground_truths) # calculate overlapped numbers
            total_score += len(overlap) / len(ground_truths)
        else:
            total_score += max([1.0 if g.lower() in prediction.lower() else 0.0 for g in ground_truths])
    print(total_score)
    return round(100 * total_score / len(predictions), 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="")
    args = parser.parse_args()
    save_dir = os.path.join("/home/xwei/", args.path)

    root = Path(save_dir)
    scores = dict()
    all_tasks = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery"]
    task_files = {}
    for task in all_tasks:
        for p in root.rglob("*.jsonl"):
            name = p.name
            if name == f"{task}.jsonl":
                task_files[task] = str(p)
                break
    for task in all_tasks:  
        predictions, answers, lengths = [], [], []
        if task not in task_files:
            print(task)
            continue
        with open(task_files[task], "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["outputs"])
        score = scorer(task, predictions, answers)
        scores[task] = score
    out_path = os.path.join(save_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print(scores.keys())
    print(scores.values())