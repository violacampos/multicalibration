import numpy as np
from pathlib import Path
import itertools
import argparse
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from tools import calibration_scores
from tools import data
from tools import create_charts

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts_multipl_e/"

binning_type = 'linear'

DEBUG = False
OUTPUTS = True

np.seterr(divide='ignore', invalid='ignore')

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

    results, temperature, top_p = data.load_multipl_e_run(args.dirs[0])

    run_dirs = [x[0] for x in os.walk(args.dirs[0])]
    run_dirs.sort()

    for d in run_dirs:
        # if main dir is in list just continue
        if d == args.dirs[0]:
            continue

        # Get run name
        run = d.split("/runs/", 1)[1]

        # load the data from the run directory
        run = run.replace('/', '')
        results, temperature, top_p, num_samples = data.load_multipl_e_run(d)

        if OUTPUTS: print(f"\nRun: {run}")
        if OUTPUTS: print(f"Temperature: {temperature}")
        if OUTPUTS: print(f"Num Problems: {num_samples}")

        probs, is_correct = data.proability_and_correctness_for_samples(results)
        correct_count = np.count_nonzero(is_correct == 1)

        if OUTPUTS: print(f"Korrekt: {correct_count}") 

        # Split in train and test
        train_probs, test_probs, train_y, test_y = train_test_split(probs, is_correct, test_size=0.2)
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

        create_charts.calibration_comparision_chart(chart_range, f_dach, train_correct_per_bin ,CHART_DIR+run+"/histogramm_binning_calibration_chart_"+binning_type+".png", run)

if __name__ == "__main__":
    main()
