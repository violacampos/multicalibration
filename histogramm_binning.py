import numpy as np
from pathlib import Path
import itertools
import argparse
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from tools import data, create_charts, binning
from tools.hb_calibration import hb_calibration
from tabulate import tabulate

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts_multipl_e/"

binning_type = 'quantil'
binning_step_size = 0.1

DEBUG = False
OUTPUTS = False

np.seterr(divide='ignore', invalid='ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dirs", type=str,  help="Directories with results. ", nargs="+")
    args = parser.parse_args()

    table_print = []

    run_dirs = [x[0] for x in os.walk(args.dirs[0])]
    run_dirs.sort()

    for d in run_dirs:
        # if main dir is in list just continue
        if d == args.dirs[0] and ("humaneval" not in d and "mbpp" not in d):
            continue
        run_entry = []
        # Get run name
        run = d.split("/runs/", 1)[1]
        run_entry.append(run)

        # create directory for chart generation
        if not os.path.isdir(CHART_DIR+run):
            os.makedirs(CHART_DIR+run)

        # load the data from the run directory
        run = run.replace('/', '')
        results, temperature, top_p, num_samples = data.load_multipl_e_run(d)
        run_entry.append(num_samples)
        if OUTPUTS: print(f"\nRun: {run}")
        if OUTPUTS: print(f"Temperature: {temperature}")
        if OUTPUTS: print(f"Num Problems: {num_samples}")

        probs, is_correct = data.proability_and_correctness_for_samples(results)
        correct_count = np.count_nonzero(is_correct == 1)

        if binning_type == 'linear':
            # uniform grid 1/m
            grid = binning.create_unform_grid(10)   
        elif binning_type == 'quantil':
            # get quantils for step size n
            bin_edges = binning.create_qunatil_grid(probs, binning_step_size)
            # get the middle of the bins for hb
            grid = np.array(((bin_edges[1:]-bin_edges[:-1])/2)+bin_edges[:-1]) 
       
        if OUTPUTS: print(f"Korrekt: {correct_count}") 
        
        # Create calibration object
        hb = hb_calibration(grid, OUTPUTS, DEBUG)

        # Split in train and test
        train_probs, test_probs, train_y, test_y = train_test_split(probs, is_correct, test_size=0.33, random_state=42)
        if DEBUG: print(f"Training values: {train_probs}")
        if DEBUG: print(f"Test values: {test_probs}")

        # Calculates the deltas for the bins
        fit_correct_per_bin, train_scores = hb.fit(train_probs, train_y)

        # Uses the deltas to calculate the corrected values
        f_dach, test_scores = hb.predict(test_probs, test_y)

        train_scores = list(list(train_scores.values())[0].values())
        test_scores = list(list(test_scores.values())[0].values())
        score_difference = np.array(test_scores) - np.array(train_scores)

        for d in score_difference:
            run_entry.append(d)
        
        for d in train_scores:
            run_entry.append(d)

        if binning_type == 'linear':
            chart_range = np.arange(0, 1.1, 0.1)
        elif binning_type == 'quantil':
            chart_range = grid

        create_charts.calibration_comparision_chart(chart_range, f_dach, fit_correct_per_bin, CHART_DIR+run+"/histogramm_binning_calibration_chart_"+binning_type+".png", run)
        table_print.append(run_entry)
        
    table_print = tabulate(table_print, headers=['Run', 'Num_samples', 'ECE_diff', 'MSE_diff', 'brier_ref_diff', 'skill_score_diff', 'ECE_train', 'MSE_train', 'brier_ref_train', 'skill_score_train'], tablefmt='orgtbl')
    print(table_print)

    with open('results/histogramm_binning_'+binning_type+'_results.txt', 'w') as f:
        f.write(table_print)

if __name__ == "__main__":
    main()
