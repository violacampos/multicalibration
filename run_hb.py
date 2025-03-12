import numpy as np
from pathlib import Path
import itertools
import argparse
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from tools import data, binning
from tools.hb_calibration import hb_calibration
from tools.create_charts import chart_creator
from tabulate import tabulate
import matplotlib.pyplot as plt

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts/"

binning_type = 'linear'
binning_step_size = 0.1
m = 10

all_lang = True
use_train_test_split = True

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

    if all_lang == True:
        run_dirs = [run_dirs[0]]

    for d in run_dirs:
        # if main dir is in list just continue
        if d == args.dirs[0] and ("humaneval" not in d and "mbpp" not in d):
            continue
        run_entry = []
        
        # Get run name
        run = d.split("/runs/", 1)[1]
        run_entry.append(run)

        # create directory for chart generation
        if not os.path.isdir(CHART_DIR+run+'/hb/'+binning_type):
            os.makedirs(CHART_DIR+run+'/hb/'+binning_type)

        # load the data from the run directory
        run = run.replace('/', '')
        results, temperature, top_p, num_samples = data.load_multipl_e_run(d)
        run_entry.append(num_samples)
        print(f"\nRun: {run}")
        save_dir = CHART_DIR+run+'/hb/'+binning_type+'/'
        if OUTPUTS: print(f"Temperature: {temperature}")
        if OUTPUTS: print(f"Num Problems: {num_samples}")

        # probs -> confidence of the model
        # is_correct -> label 1: is correct, 0: is not correct
        probs, is_correct, _, _ = data.proability_and_correctness_for_samples(results)
        correct_count = np.count_nonzero(is_correct == 1)

        if use_train_test_split:
        # split in train and test
            train_probs, test_probs, train_y, test_y = train_test_split(probs, is_correct, test_size=0.33, random_state=42)
            if DEBUG: print(f"Training values: {train_probs}")
            if DEBUG: print(f"Test values: {test_probs}")
        else:
            train_probs = probs
            test_probs = probs
            train_y = is_correct
            test_y = is_correct

        # sets the type of binning
        if binning_type == 'linear':
            # uniform grid 1/m
            grid = binning.create_unform_grid(m)
            charts = chart_creator(run, binning_type, grid, save_dir, m)
        elif binning_type == 'quantil':
            # get quantils for step size n
            bin_edges = binning.create_qunatil_grid(train_probs, binning_step_size)
            # get the middle of the bins for hb
            grid = np.array(((bin_edges[1:]-bin_edges[:-1])/2)+bin_edges[:-1]) 
            charts = chart_creator(run, binning_type, grid, save_dir, bin_edges=bin_edges)
        
        if OUTPUTS: print(f"Korrekt: {correct_count}") 
        
        # Create calibration object and calculates the deltas
        hb = hb_calibration(grid, OUTPUTS, DEBUG).fit(train_probs, train_y)

        # calculate scores for the uncalibrated test set
        total_uncalibrated, correctness_uncalibrated, scores_uncalibrated = hb.calib_score(test_probs, test_y, set_b_ref=True)

        # Uses the deltas to calculate the corrected values
        corrected_probs = hb.predict(test_probs) 

        # calculate scores for the calibrated test set
        total_calibrated, correctness_calibrated, scores_calibrated = hb.calib_score(corrected_probs, test_y)
        
        # extract score into lists
        scores_uncalibrated = list(list(scores_uncalibrated.values())[0].values())
        scores_calibrated = list(list(scores_calibrated.values())[0].values())
        score_difference = np.array(scores_calibrated) - np.array(scores_uncalibrated)

        # add them to the output list
        for train, test, diff in zip(scores_uncalibrated, scores_calibrated, score_difference):
            run_entry.append(train)
            run_entry.append(test)
            run_entry.append(diff)
        
        # create different charts
        charts.set_bar_colors(total_uncalibrated, total_calibrated)
        charts.calibration_info(total_uncalibrated, correctness_uncalibrated, total_calibrated, correctness_calibrated)
            
        table_print.append(run_entry)
        
    table_print = tabulate(table_print, headers=['Run', 
                                                 'Num_samples', 
                                                 'ECE', 
                                                 'ECE_calib',
                                                 'ECE_diff', 
                                                 'ASCE', 
                                                 'ASCE_calib',
                                                 'ASCE_diff',
                                                 'MSE',
                                                 'MSE_calib', 
                                                 'MSE_diff',
                                                 'brier_ref', 
                                                 'brier_ref_calib', 
                                                 'brier_ref_diff', 
                                                 'skill_score', 
                                                 'skill_score_calib',
                                                'skill_score_diff'], tablefmt='orgtbl')
    print(table_print)

    with open('results/histogramm_binning_'+binning_type+'_results.txt', 'w') as f:
        f.write(table_print)

if __name__ == "__main__":
    main()
