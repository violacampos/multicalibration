import numpy as np
from pathlib import Path
import itertools
import argparse
import json
from pathlib import Path
import gzip
from typing import Optional
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts_multipl_e/"

binning_type = 'linear'

DEBUG = False

def generate_calibration_bar_chart(y, x, path, run, width, bar_colors="blue"):
    plt.title(run+' # Reliability chart', fontsize=7)
    plt.plot(y, x, color=bar_colors)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Confidence')
    plt.ylabel('Correct')
    plt.savefig(path)
    plt.close()

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
        "top_p": data["top_p"],
        "cumulative_logprob": cumulative_logprob,
        "token_ids": token_ids,
        "token_logprobs": token_logprobs
    }


def create_unform_grid(m):
    return np.round(np.arange(0.0, 1+(1/m), 1/m), 2)   


def round_model_to_grid(probs, grid):
    # Round model to the grid (assign values to bin edges)
    bin_assignment = []    

    for f_x in probs:             
        bin_assignment.append(np.round(grid[np.argmin(np.abs(f_x - grid))], 2))         
    
    return np.array(bin_assignment)

def calculate_correctnes_per_bin(probs, y, assignments, grid):
    prob_correct = np.array([np.divide(len(probs[(assignments == i) & (y == 1)]), len(probs[(assignments == i)])) for i in grid])
    prob_correct[np.isnan(prob_correct)] = 0
    return prob_correct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dirs", type=str,  help="Directories with results. ", nargs="+")
    args = parser.parse_args()

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
        top_p = set(r["top_p"] for r in results)

        temperature = list(temperatures)[0]
        top_p = list(top_p)[0]
        num_problems = len(results)
        print(f"\nRun: {run}")
        print(f"Temperature: {temperature}")
        print(f"Num Problems: {num_problems}")

        pass_probs = []
        fail_probs = []
        predicted_probs = []
        is_correct = []

        for r in results:
            token_count = len(r["token_ids"])           
            cumulative_logprob = r["cumulative_logprob"]
            avg_prob = np.round(np.exp(cumulative_logprob / token_count), 2) 

            predicted_probs.append(avg_prob)
            if r["c"] == 0:
                fail_probs.append(avg_prob)
                is_correct.append(0)
            if r["c"] == 1:
                pass_probs.append(avg_prob)
                is_correct.append(1)

        predicted_probs     = np.array(predicted_probs)
        pass_probs          = np.array(pass_probs)
        fail_probs          = np.array(fail_probs)
        is_correct          = np.array(is_correct)
        correct_count = np.count_nonzero(is_correct == 1)
        if DEBUG: print(f"Korrekt: {correct_count}") 

        # Split in train and test
        train_probs, test_probs, train_y, test_y = train_test_split(predicted_probs, is_correct, test_size=0.2)
        if DEBUG: print(f"Training values: {train_probs}")
        if DEBUG: print(f"Test values: {test_probs}")

        # uniform grid 1/m
        uniform_grid = create_unform_grid(10)   
        train_bin_assignments = round_model_to_grid(train_probs, uniform_grid)

        # Calculated the mean correcteness of the assigned bins
        if DEBUG: print(f"TRAIN Assigned Bins: {train_bin_assignments}")
        train_correct_per_bin = calculate_correctnes_per_bin(train_probs, train_y, train_bin_assignments, uniform_grid)

        if DEBUG: print(f"Uniform Grid: {uniform_grid}")
        if DEBUG: print(f"TRAIN Correct per bin: {train_correct_per_bin}")     

        # Calculate correcteness bias in the given bin
        delta_p_f_ =  uniform_grid - train_correct_per_bin
 

        if DEBUG: print(f"TEST Preditions: {test_probs}")    
        # Test calibration
        test_bin_assignment = round_model_to_grid(test_probs, uniform_grid)
        if DEBUG: print(f"TEST Assigned Bins: {test_bin_assignment}")
        test_correct_per_bin = calculate_correctnes_per_bin(test_probs, test_y, test_bin_assignment, uniform_grid)
        if DEBUG: print(f"TEST Correct per bin: {test_correct_per_bin}")
        if DEBUG: print(f"TRAIN Delta_p(f): {delta_p_f_}")  
        f_dach = np.clip(test_correct_per_bin + delta_p_f_, 0, 1)
        if DEBUG: print(f"TEST Corrected Values: {f_dach}")
        if binning_type == 'linear':
            chart_range = np.arange(0, 1.1, 0.1)
            bar_width = 0.1

        generate_calibration_bar_chart(chart_range, f_dach, CHART_DIR+run+"/histogramm_binninb_calibration_bar_chart_"+binning_type+".png", run, bar_width)


if __name__ == "__main__":
    main()
