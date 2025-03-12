import numpy as np
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tools import data, binning
from tools.create_charts import chart_creator
from tools.ighb_calibration import IGHB_calibration
from tools.groups import groups
from tabulate import tabulate

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts/"

DEBUG = False
OUTPUTS = True

binning_type = 'linear'
binning_step_size = 0.1
m = 10

use_train_test_split = True

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
        save_dir = CHART_DIR+run+'/ighb/'+binning_type+'/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        probs, is_correct, programs, prompts = data.proability_and_correctness_for_samples(results)
        correct_count = np.count_nonzero(is_correct == 1)

       
        # Define group matrix
        groups_w = groups(programs, prompts).create_groups()
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
        
        ighb = IGHB_calibration(grid, alpha, OUTPUTS, DEBUG).fit(test_X, test_y, test_groups)
        total_uncalibrated, correctness_uncalibrated, scores_uncalibrated = ighb.calib_score(test_X, test_y, test_groups, set_b_ref=True)
        
        calibrated_conf = test_X
        
        while ighb.max_error > ighb.alpha:  

            if DEBUG: print(f"Max Error: {ighb.max_error}")
            calibrated_conf = ighb.predict(calibrated_conf, test_groups)

            ighb = ighb.fit(calibrated_conf, test_y, test_groups)
                
            
        total_calibrated, correctness_calibrated, scores_calibrated = ighb.calib_score(calibrated_conf, test_y, test_groups)
        # Charts
        charts.set_bar_colors(total_uncalibrated, total_calibrated)
        charts.calibration_info(total_uncalibrated, correctness_uncalibrated, total_calibrated, correctness_calibrated)
    

if __name__ == "__main__":
    main()
