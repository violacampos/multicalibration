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
import streamlit as st
import pickle 
from sklearn.model_selection import KFold
import json

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts/"

DEBUG = False
OUTPUTS = True

binning_type = 'linear'
binning_step_size = 0.1

alpha = 0.01
m = np.ceil(1/alpha)
grid = []

use_train_test_split = False
use_k_fold = True

all_lang = True
load_scc_results = True

save_table = False

np.seterr(divide='ignore', invalid='ignore')

def main(extern=False):
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

        # Get run name
        run = d.split("/runs/", 1)[1]

        # load the data from the run directory
        run = run.replace('/', '')
        results, temperature, top_p, num_samples = data.load_multipl_e_run(d)
        
        # create directory for chart generation
        save_dir = CHART_DIR+run+'/ighb/'+binning_type+'/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        probs, is_correct, programs, prompts, languages, names = data.proability_and_correctness_for_samples(results)

        # Keep in mind that elixir has the extension ex for concating the results
        # also wierd results for go -> folder was namen go_test.go
        if load_scc_results:
            scc_infos = data.load_scc_data(run, languages, names)
            groups_w = groups(programs, prompts).create_groups(scc=scc_infos)
        else:
            # Define group matrix
            groups_w = groups(programs, prompts).create_groups()
            
        if OUTPUTS: print(f"Run: {run}")
        if OUTPUTS: print(f"Gruppen Anzahl: {groups_w.sum(axis=0)}")

        if use_k_fold:
            kf = KFold(n_splits=5)
            kf.get_n_splits(probs)
            print(kf)
            history = {}
            for i, (train_index, test_index) in enumerate(kf.split(probs)):
                print(f"Fold {i}:")

                train_X = probs[train_index]
                test_X = probs[test_index]
                train_y = is_correct[train_index]
                test_y = is_correct[test_index]
                train_groups = groups_w[train_index]
                test_groups = groups_w[test_index]

                # get the grid for binning type and the chartmaker obj        
                grid, chartmaker = binning.get_grid_and_chartmaker(run, binning_type, save_dir, m, train_X, binning_step_size)
            
                ighb = IGHB_calibration(grid, m, alpha, OUTPUTS, DEBUG).fit(train_X, train_y, train_groups)

                # Calculate values for uncalibrated test set
                scores_uncalibrated = ighb.score_obj.calc_all_new(  test_X, 
                                                                    test_y, 
                                                                    groups=test_groups,
                                                                    deltas=ighb.get_deltas(test_X, test_y, test_groups), 
                                                                    set_brier_ref=True)
                
                total_group_uncalibrated, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = ighb.score_obj.get_total_and_correctness(test_X, test_y, test_groups) 

                temp_group_correctness = correctness_group_uncalibrated
                temp_bin_correctness = correctness_bin_uncalibrated

                # set the conf to calibrate on
                calibrated_conf = train_X
                history_item = {}
                while ighb.max_error > ighb.alpha:  
                    if DEBUG: print(f"Max Error: {ighb.max_error}")
                    # get new better calibrated confidences
                    calibrated_conf = ighb.predict(calibrated_conf, train_groups, is_correct=train_y)
                    
                    # calculate the corrected values for the test set
                    test_X = ighb.predict(test_X, test_groups, test=True, is_correct=test_y)

                    # Calculate some metrics on the UNcorrected values
                    curr_group_total, curr_group_correctness, curr_bin_total, curr_bin_correctness = ighb.score_obj.get_total_and_correctness(test_X, test_y, test_groups)
                    history_item[len(ighb.changes)] = [temp_group_correctness, curr_group_correctness, ighb.changes[-1], curr_group_total, temp_bin_correctness, curr_bin_correctness] #chartmaker.map_correctness_to_eleven_bins(
                    temp_group_correctness = curr_group_correctness
                    temp_bin_correctness = curr_bin_correctness
                    
                    # fit the model on the corrected confidences
                    ighb = ighb.fit(calibrated_conf, train_y, train_groups)
                        

                # Calculate values for calibrated test set
                scores_calibrated = ighb.score_obj.calc_all_new(test_X, 
                                                                test_y, 
                                                                groups=test_groups,
                                                                deltas=ighb.get_deltas(test_X, test_y, test_groups))
                
                total_group_calibrated, correctness_group_calibrated, total_bin_calibrated, correctness_bin_calibrated = ighb.score_obj.get_total_and_correctness(test_X, test_y, test_groups) 

                # Add entry for the run in the score table
                ighb.score_obj.add_to_score_table(run, scores_uncalibrated, scores_calibrated)

                history_item["score"] = ighb.score_obj.score_table
                history[i] = history_item
                ighb = None

            with open('ighb_history_kfold.pkl', 'wb') as f:
                pickle.dump(history, f)
        else:
            
            # check if we split the data or use the whole dataset for evaluation
            if use_train_test_split:
                # split in 60% train, 20% validation and 20% test
                train_X, test_X, train_y, test_y, train_groups, test_groups = train_test_split(probs, is_correct, groups_w, test_size=0.2, random_state=42)

                train_X, val_X, train_y, val_y, train_groups, val_groups = train_test_split(train_X, train_y, train_groups, test_size=0.25, random_state=42)
            else:
                train_X = probs
                test_X = probs
                train_y = is_correct
                test_y = is_correct
                train_groups = groups_w
                test_groups = groups_w
        
            # get the grid for binning type and the chartmaker obj        
            grid, chartmaker = binning.get_grid_and_chartmaker(run, binning_type, save_dir, m, train_X, binning_step_size)
        
            ighb = IGHB_calibration(grid, m, alpha, OUTPUTS, DEBUG).fit(train_X, train_y, train_groups)

            # Calculate values for uncalibrated test set
            scores_uncalibrated = ighb.score_obj.calc_all_new(  test_X, 
                                                                test_y, 
                                                                groups=test_groups,
                                                                deltas=ighb.get_deltas(test_X, test_y, test_groups), 
                                                                set_brier_ref=True)
            
            total_group_uncalibrated, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = ighb.score_obj.get_total_and_correctness(test_X, test_y, test_groups) 

            temp_group_correctness = correctness_group_uncalibrated
            temp_bin_correctness = correctness_bin_uncalibrated

            # set the conf to calibrate on
            calibrated_conf = train_X
            history = {}
            while ighb.max_error > ighb.alpha:  

                if DEBUG: print(f"Max Error: {ighb.max_error}")
                # get new better calibrated confidences
                if use_train_test_split:
                    calibrated_conf = ighb.predict(calibrated_conf, train_groups, is_correct=train_y)
                else:
                    calibrated_conf = ighb.predict(calibrated_conf, train_groups, test=True, is_correct=train_y)
                
                # calculate the corrected values for the test set
                if use_train_test_split:
                    test_X = ighb.predict(test_X, test_groups, test=True, is_correct=test_y)
                else:
                    test_X = calibrated_conf

                # Calculate some metrics on the UNcorrected values
                curr_group_total, curr_group_correctness, curr_bin_total, curr_bin_correctness = ighb.score_obj.get_total_and_correctness(test_X, test_y, test_groups)
                history[len(ighb.changes)] = [temp_group_correctness, curr_group_correctness, ighb.changes[-1], curr_group_total, temp_bin_correctness, curr_bin_correctness] #chartmaker.map_correctness_to_eleven_bins(
                temp_group_correctness = curr_group_correctness
                temp_bin_correctness = curr_bin_correctness
                
                # fit the model on the corrected confidences
                ighb = ighb.fit(calibrated_conf, train_y, train_groups)
                    

            # Calculate values for calibrated test set
            scores_calibrated = ighb.score_obj.calc_all_new(test_X, 
                                                            test_y, 
                                                            groups=test_groups,
                                                            deltas=ighb.get_deltas(test_X, test_y, test_groups))
            
            total_group_calibrated, correctness_group_calibrated, total_bin_calibrated, correctness_bin_calibrated = ighb.score_obj.get_total_and_correctness(test_X, test_y, test_groups) 

            # Add entry for the run in the score table
            ighb.score_obj.add_to_score_table(run, scores_uncalibrated, scores_calibrated)

            history["score"] = ighb.score_obj.score_table

            if extern:
                return total_bin_calibrated, correctness_bin_calibrated, scores_calibrated
            #else:
                # Charts
                #chartmaker.calibration_info(total_uncalibrated, correctness_uncalibrated, total_calibrated, correctness_calibrated)
            
            if use_train_test_split:
                with open('ighb_history.pkl', 'wb') as f:
                    pickle.dump(history, f)
            else:
                with open('ighb_history_no_split.pkl', 'wb') as f:
                    pickle.dump(history, f)
            # display score table for all runs
            ighb.score_obj.display_score_table()
            
            # saves the score table
            if save_table:
                with open('results/ighb_'+binning_type+'_results.txt', 'w') as f:
                    f.write(ighb.score_obj.printable_table)

if __name__ == "__main__":
    main()
