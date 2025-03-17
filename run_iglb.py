import numpy as np
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tools import data, binning
from tools.create_charts import chart_creator
from tools.iglb_calibration import IGLB_calibration
from tools.groups import groups
from tabulate import tabulate

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts/"

DEBUG = False
OUTPUTS = True

binning_type = 'linear'
binning_step_size = 0.1
m = 10
alpha = 0.001

use_calib_val_split = True

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
        save_dir = CHART_DIR+run+'/iglb/'+binning_type+'/'
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
        if use_calib_val_split:
            # Split in train and test
            calib_X, val_X, calib_y, val_y, calib_groups, val_groups = train_test_split(probs, is_correct, groups_w, test_size=0.33, random_state=42)
            if DEBUG: print(f"Calib values: {calib_X}")
            if DEBUG: print(f"val values: {val_X}")
        else:
            calib_X = probs
            val_X = probs
            calib_y = is_correct
            val_y = is_correct
            calib_groups = groups_w
            val_groups = groups_w

        # sets the type of binning
        if binning_type == 'linear':
            # uniform grid 1/m
            grid = binning.create_unform_grid(m)
            charts = chart_creator(run, binning_type, grid, save_dir, m)
        elif binning_type == 'quantil':
            # get quantils for step size n
            bin_edges = binning.create_qunatil_grid(calib_X, binning_step_size)
            # get the middle of the bins for hb
            grid = np.array(((bin_edges[1:]-bin_edges[:-1])/2)+bin_edges[:-1]) 
            charts = chart_creator(run, binning_type, grid, save_dir, bin_edges=bin_edges)
        
        # Create object and calculate first deltas and so on
        iglb = IGLB_calibration(grid, alpha, OUTPUTS, DEBUG).fit(calib_X, calib_y, calib_groups)
        total_uncalibrated, correctness_uncalibrated, scores_uncalibrated = iglb.calib_score(calib_X, calib_y, calib_groups, set_b_ref=True)
        
        while True: 
            # Calculate mse for f_t
            mse_f_t = iglb.score_calibration.mse(val_X, val_y, len(val_y))

            # Assign bins an calculate the probability for each bin,group and tau combination
            assigned_bins = binning.round_model_to_grid(calib_X, grid)   
            P_S_p_g = iglb.get_P_S_p_g(assigned_bins, calib_groups) 
            
            # get the tau, bin, group for which the probality * deltas_squared maximises
            tau, bin, group = np.unravel_index((P_S_p_g*iglb.deltas_square).argmax(), iglb.deltas.shape)
            if iglb.debug: print(f"Max delta in: Tau {tau}, Bin {bin}, Group {group}")

            # First break if probability is smaller then alpha
            if P_S_p_g[tau, bin, group] < alpha:
                break

            if DEBUG: print(f"Max Error: {iglb.max_error}")
            # get the calibrated confidences for the calibration subset
            calib_X = iglb.predict(calib_X, calib_groups, assigned_bins, tau, bin, group)

            # get the calibrated confidences for the validation subset to calculate MSE
            assigned_bins_val = binning.round_model_to_grid(val_X, grid)   
            val_X = iglb.predict(val_X, val_groups, assigned_bins_val, tau, bin, group)

            # Second Break if MSE of the new model is greater or equal to the model before
            mse_h_t_plus_1 = iglb.score_calibration.mse(val_X, val_y, len(val_y))
            print(f"MSE h_t+1: {mse_h_t_plus_1} >= MSE f_t{mse_f_t}")
            if mse_h_t_plus_1 >= mse_f_t:
                break
            
            # Set the new model for the next iteration
            iglb = iglb.fit(calib_X, calib_y, calib_groups)
                
        print(f"GASCE: {iglb.gasce}\n")

        total_calibrated, correctness_calibrated, scores_calibrated = iglb.calib_score(calib_X, calib_y, calib_groups)
        
        # Charts
        charts.set_bar_colors(total_uncalibrated, total_calibrated)
        charts.calibration_info(total_uncalibrated, correctness_uncalibrated, total_calibrated, correctness_calibrated)
    

if __name__ == "__main__":
    main()
