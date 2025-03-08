import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from tools import data, groups, binning
from tools.calibration_scores import score
from tools.create_charts import chart_creator
from tools.LR_calibration import LR_calibration
from tabulate import tabulate

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts_multipl_e/"

DEBUG = False
OUTPUTS = True

binning_type = 'linear'
binning_step_size = 0.1
m = 10

use_train_test_split = False
control_exp = False

all_lang = True

np.seterr(divide='ignore', invalid='ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dirs", type=str,  help="Directories with results. ", nargs="+")
    args = parser.parse_args()

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

        # load the data from the run directory
        run = run.replace('/', '')
        results, temperature, top_p, num_samples = data.load_multipl_e_run(d)
        
        # create directory for chart generation
        save_dir = CHART_DIR+run+'/group_lr/'+binning_type+'/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        probs, is_correct, programs, prompts = data.proability_and_correctness_for_samples(results)
        correct_count = np.count_nonzero(is_correct == 1)

        # Define group matrix
        groups_w = []

        # Check group memebership
        for program, prompt in zip(programs, prompts):
            groups_w.append(groups.check_groups(prompt, program))

        groups_w = np.array(groups_w)
        
        if OUTPUTS: print(f"Run: {run}")
        if OUTPUTS: print(f"Gruppen Anzahl: {groups_w.sum(axis=0)}")

        # check if we split the data or use the whole dataset for evaluation
        if use_train_test_split:
            # Split in train and test
            train_X, test_X, train_y, test_y, train_groups, test_groups = train_test_split(probs, is_correct, groups_w, test_size=0.33, random_state=42)
            if DEBUG: print(f"Training values: {train_X}")
            if DEBUG: print(f"Test values: {test_X}")
        else:
            train_X = probs
            test_X = probs
            train_y = is_correct
            test_y = is_correct
            train_groups = groups_w
            test_groups = groups_w

        # sets the type of binning
        if binning_type == 'linear':
            # uniform grid 1/m
            grid = binning.create_unform_grid(m)
            charts = chart_creator(run, binning_type, grid, save_dir, m)
        elif binning_type == 'quantil':
            # get quantils for step size n
            bin_edges = binning.create_qunatil_grid(test_X, binning_step_size)
            # get the middle of the bins for hb
            grid = np.array(((bin_edges[1:]-bin_edges[:-1])/2)+bin_edges[:-1]) 
            charts = chart_creator(run, binning_type, grid, save_dir, bin_edges=bin_edges)

        alpha = 0.01

        score_calibration = score(grid, OUTPUTS, DEBUG)        
        
        assigned_bins = binning.round_model_to_grid(test_X, grid)
        deltas, gasce = score_calibration.gasce(assigned_bins, test_y, test_groups)

        print(f"Deltas: {deltas}")
        print(f"GASCE: {gasce}")

        p_group = test_groups.sum(axis=0) / num_samples
        
        print(f"P(X)=1: {p_group}")

        print(gasce * p_group)

        max_error = np.argmax(gasce*p_group)

        print(f"Optimizing Group {max_error}")

        while max_error > alpha:

            
        
        # Charts
        """charts.set_bar_colors(total_uncalibrated, total_calibrated)
        charts.calibration_info(total_uncalibrated, correctness_uncalibrated, total_calibrated, correctness_calibrated)
        """
        #print(f"Group Lamdas: {lr_calib.reg.coef_}")

if __name__ == "__main__":
    main()
