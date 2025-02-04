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

CHART_DIR = BASE_DIR+"charts_multipl_e/"


def create_histogram(data, path, run, typ):
    fig, ax = plt.subplots()  
    ax.hist(data, range=(0, 1.0))
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    plt.title(run+' # '+typ, fontsize=8)
    plt.savefig(path)
    print(f"Histogram saved: {path}")

def generate_calibration_bar_chart(y, x, path, run):
    plt.title(run+' # Reliability chart', fontsize=8)
    plt.bar(y, x, width = 0.1)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Confidence')
    plt.ylabel('Correct')
    plt.savefig(path)

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

    for d in args.dirs:
        run = d.split("/runs/", 1)[1]
        if not os.path.isdir(CHART_DIR+run):
            os.makedirs(CHART_DIR+run)
        run = run.replace('/', '')
        results = [for_file(p) for p in itertools.chain(
            Path(d).glob("*.results.json"), Path(d).glob("*.results.json.gz"))]
        results = [r for r in results if r is not None]
        name = d.split("/")[-1] if d.split("/")[-1] != "" else d.split("/")[-2]
        temperatures = set(r["temperature"] for r in results)

        temperature = list(temperatures)[0]
        num_problems = len(results)
        
        print(f"Temperature: {temperature}")
        print(f"Num Problems: {num_problems}")

        pass_log_value_list = []
        fail_log_value_list = []
        prob_value_list = []

        for r in results:
            token_count = len(r["token_ids"])

            cumulative_logprob = r["cumulative_logprob"]

            avg_prob = np.round(np.exp(cumulative_logprob / token_count), 2) 

            prob_value_list.append(avg_prob)
            if r["c"] == 0:
                fail_log_value_list.append(avg_prob)
            if r["c"] == 1:
                pass_log_value_list.append(avg_prob)

        pass_log_value_list = np.array(pass_log_value_list)
        fail_log_value_list = np.array(fail_log_value_list)

        # Brechnet Wahrscheinlichleit P(Korrekt| Score in bin 0-0.09, 0.1-0.19, ..., 0.9 )
        total_bin_count = []
        correct_bin_count = []
        for bin in np.arange(0, 1, 0.1):
            bin_left = np.round(bin,1)
            bin_right = np.round(bin+0.1, 1)

            if bin_left == 0.9:
                correct_bin_count.append(np.count_nonzero((bin_left <= pass_log_value_list) & (pass_log_value_list <= bin_right)))
                total_bin_count.append(np.count_nonzero((bin_left <= prob_value_list) & (prob_value_list <= bin_right)))
            else:
                correct_bin_count.append(np.count_nonzero((bin_left <= pass_log_value_list) & (pass_log_value_list < bin_right)))
                total_bin_count.append(np.count_nonzero((bin_left <= prob_value_list) & (prob_value_list < bin_right)))

        # Wahrscheinlichkeit, dass Code korrekt ist in den jeweiligen bins
        P_correct = np.divide(np.array(correct_bin_count), np.array(total_bin_count), where=np.array(total_bin_count)!=0)
        print(P_correct)
        print(prob_value_list)
        generate_calibration_bar_chart(np.arange(0.05, 1, 0.1), P_correct, CHART_DIR+run+"/calibration_bar_chart.png", run)

        create_histogram(prob_value_list, CHART_DIR+run+'/Probabilities.png',run ,'Total Probabilities')
        create_histogram(pass_log_value_list, CHART_DIR+run+'/Probabilities_pass.png',run ,'Pass Probabilities')
        create_histogram(fail_log_value_list, CHART_DIR+run+'/Probabilities_fail.png',run ,'Fail Probabilities')

if __name__ == "__main__":
    main()
