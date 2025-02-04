import numpy as np
from pathlib import Path
import itertools
import argparse
import json
from pathlib import Path
import gzip
from typing import Optional
import sys
import matplotlib.pyplot as plt
import os

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"

def gunzip_json(path: Path) -> Optional[dict]:
    """
    Reads a .json.gz file, but produces None if any error occurs.
    """
    try:
        with gzip.open(path, "rt") as f:
            return json.load(f)
    except Exception as e:
        return None

def gzip_json(path: Path, data: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f)

def for_file(path: Path):
    if path.suffix == ".gz":
        data = gunzip_json(path)
    else:
        with open(path, 'r') as f:
            data = json.load(f)

    if data is None:
        return None
    n = len(data["results"])
    token_infos         = data["tokens_info"][0]
    cumulative_logprob  = token_infos["cumulative_logprob"]
    token_logprobs      = token_infos["token_logprobs"]
    token_ids           = token_infos["len"]
    c = len([True for r in data["results"] if r["status"]
            == "OK" and r["exit_code"] == 0])
    return {
        "name": data["name"], 
        "n": n,
        "c": c,
        "temperature": data["temperature"] if "temperature" in data else 0.2,
        "cumulative_logprob": cumulative_logprob,
        "token_ids": token_ids,
        "token_logprobs": token_logprobs
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dirs", type=str,  help="Directories with results. ", nargs="+")
    args = parser.parse_args()

    results_dict = {}

    run_dirs = [x[0] for x in os.walk(args.dirs[0])]
    run_dirs.sort()

    for d in run_dirs:
        if d == args.dirs[0]:
            continue
        run = d.split("/runs/", 1)[1]
        run = run.replace('/', '')

        results = [for_file(p) for p in itertools.chain(
            Path(d).glob("*.results.json"), Path(d).glob("*.results.json.gz"))]
        results = [r for r in results if r is not None]
        temperatures = set(r["temperature"] for r in results)

        temperature = list(temperatures)[0]
        num_problems = len(results)
        print(f"\nRun: {run}")
        print(f"Temperature: {temperature}")
        print(f"Num Problems: {num_problems}")

        pass_log_value_list = []
        fail_log_value_list = []
        prob_value_list = []
        is_correct = []

        for r in results:
            token_count = len(r["token_ids"])

            cumulative_logprob = r["cumulative_logprob"]

            avg_prob = np.round(np.exp(cumulative_logprob / token_count), 2) 

            prob_value_list.append(avg_prob)
            if r["c"] == 0:
                fail_log_value_list.append(avg_prob)
                is_correct.append(0)
            if r["c"] == 1:
                pass_log_value_list.append(avg_prob)
                is_correct.append(1)

        prob_value_list     = np.array(prob_value_list)
        pass_log_value_list = np.array(pass_log_value_list)
        fail_log_value_list = np.array(fail_log_value_list)
        is_correct          = np.array(is_correct)
        correct_count = np.count_nonzero(is_correct == 1)
        print(f"Korrekt: {correct_count}")

        # Brechnet Wahrscheinlichleit P(Korrekt| Score in bin 0-0.09, 0.1-0.19, ..., 0.9 )
        total_bin_count = []
        correct_bin_count = []
        # conf(S_i)
        average_bin_confidence = []
        for bin in np.arange(0, 1, 0.1):
            bin_left = np.round(bin,1)
            bin_right = np.round(bin+0.1, 1)

            if bin_left == 0.9:
                correct_bin_count.append(np.count_nonzero((bin_left <= pass_log_value_list) & (pass_log_value_list <= bin_right)))
                bin_count   = np.count_nonzero((bin_left <= prob_value_list) & (prob_value_list <= bin_right))
                bin_sum     = np.sum(prob_value_list, where=(bin_left <= prob_value_list) & (prob_value_list <= bin_right))
                total_bin_count.append(bin_count)
                average_bin_confidence.append(np.divide(bin_sum, bin_count, where=np.array(bin_count)!=0))
            else:
                correct_bin_count.append(np.count_nonzero((bin_left <= pass_log_value_list) & (pass_log_value_list < bin_right)))
                bin_count   = np.count_nonzero((bin_left <= prob_value_list) & (prob_value_list < bin_right))
                bin_sum     = np.sum(prob_value_list, where=(bin_left <= prob_value_list) & (prob_value_list < bin_right))
                total_bin_count.append(bin_count)
                average_bin_confidence.append(np.divide(bin_sum, bin_count, where=np.array(bin_count)!=0))
       

        # Wahrscheinlichkeit, dass Code korrekt ist in den jeweiligen bins
        # corr(S_i)
        P_correct = np.divide(np.array(correct_bin_count), np.array(total_bin_count), where=np.array(total_bin_count)!=0)

        # Expected Calibration Error
        ece = 0
        for corr_s_i, conf_s_i, s_i_count in zip(P_correct, average_bin_confidence, total_bin_count):
            ece += ((abs(s_i_count)/abs(num_problems))*abs(corr_s_i-conf_s_i))

        ece = np.round(ece, 2)
        print(f"ECE: {ece}")
        
        # Brier Score
        brier_score = 0
        for predicted, result in zip(prob_value_list, is_correct):
            brier_score += (predicted - result)**2

        brier_score = np.round((1/num_problems) * brier_score, 2)

        print(f"Brier Score: {brier_score}")

        results_dict[run] = {
            "temperature": temperature,
            "num problems": num_problems,
            "correct": correct_count,
            "ECE": ece,
            "Brier Score": brier_score
        }

    result_json = json.dumps(results_dict, indent=4)
 
    with open("result.json", "w") as outfile:
        outfile.write(result_json)

if __name__ == "__main__":
    main()
