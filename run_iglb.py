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
import pickle

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts/"

DEBUG = False
OUTPUTS = True

binning_type = 'linear'
binning_step_size = 0.1
#m = 10
epsilon = 0.01
m = 10

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
        save_dir = CHART_DIR+run+'/iglb/'+binning_type+'/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        probs, is_correct, programs, prompts, languages, names = data.proability_and_correctness_for_samples(results)
       
        if load_scc_results:
            scc_infos = data.load_scc_data(run, languages, names)
            groups_w = groups(programs, prompts).create_groups(scc=scc_infos)
        else:
            # Define group matrix
            groups_w = groups(programs, prompts).create_groups()
        
        if OUTPUTS: print(f"Run: {run}")
        if OUTPUTS: print(f"Gruppen Anzahl: {groups_w.sum(axis=0)}")

        # split in 60% train, 20% validation and 20% test
        train_X, test_X, train_y, test_y, train_groups, test_groups = train_test_split(probs, is_correct, groups_w, test_size=0.2, random_state=42)

        train_X, val_X, train_y, val_y, train_groups, val_groups = train_test_split(train_X, train_y, train_groups, test_size=0.25, random_state=42)

        #calib_X, val_X, calib_y, val_y, calib_groups, val_groups = train_test_split(probs, is_correct, groups_w, test_size=0.33, random_state=42)
        
        # get the grid for binning type and the chartmaker obj        
        grid, chartmaker = binning.get_grid_and_chartmaker(run, binning_type, save_dir, m, train_X, binning_step_size)

        # Create object and calculate first deltas and so on
        iglb = IGLB_calibration(grid, epsilon, m, OUTPUTS, DEBUG).fit(train_X, train_y, train_groups)
        scores_uncalibrated = iglb.score_obj.calc_all_new(  test_X, 
                                                            test_y, 
                                                            groups=test_groups, 
                                                            set_brier_ref=True)
        total_group_uncalibrated, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = iglb.score_obj.get_total_and_correctness(test_X, test_y, test_groups) 
        
        temp_group_correctness = correctness_group_uncalibrated
        temp_bin_correctness = correctness_bin_uncalibrated
        history = {}

        while True: 
            # Calculate mse for f_t
            mse_f_t = iglb.score_obj.mse(val_X, val_y, len(val_y))

            # Assign bins an calculate the probability for each bin,group and tau combination
            assigned_bins = binning.round_model_to_grid(train_X, grid)   
            P_S_p_g = iglb.get_P_S_p_g(assigned_bins, train_groups) 
            
            # get the tau, bin, group for which the probality * deltas_squared maximises
            tau, bin, group = np.unravel_index((P_S_p_g*iglb.deltas_square).argmax(), iglb.deltas.shape)
            if iglb.debug: print(f"Max delta in: Tau {tau}, Bin {bin}, Group {group}")

            # First break if probability is smaller then alpha
            if P_S_p_g[tau, bin, group] < epsilon:
                break

            if DEBUG: print(f"Max Error: {P_S_p_g[tau, bin, group]}")
            # get the calibrated confidences for the calibration subset
            train_X = iglb.predict(train_X, train_groups, assigned_bins, tau, bin, group)

            # get the calibrated confidences for the test subset
            assigned_bins_test = binning.round_model_to_grid(test_X, grid)   
            test_X = iglb.predict(test_X, test_groups, assigned_bins_test, tau, bin, group, test=True, is_correct=test_y)

            # get the calibrated confidences for the validation subset to calculate MSE
            assigned_bins_val = binning.round_model_to_grid(val_X, grid)   
            val_X = iglb.predict(val_X, val_groups, assigned_bins_val, tau, bin, group)
            
            curr_group_total, curr_group_correctness, curr_bin_total, curr_bin_correctness = iglb.score_obj.get_total_and_correctness(test_X, test_y, test_groups)
            history[len(iglb.changes)] = [temp_group_correctness, curr_group_correctness, iglb.changes[-1], curr_group_total, temp_bin_correctness, curr_bin_correctness] 
            temp_group_correctness = curr_group_correctness
            temp_bin_correctness = curr_bin_correctness    

            # Second Break if MSE of the new model is greater or equal to the model before
            mse_h_t_plus_1 = iglb.score_obj.mse(val_X, val_y, len(val_y))
            if mse_h_t_plus_1 >= mse_f_t:
                if OUTPUTS: print(f"MSE h_t+1: {mse_h_t_plus_1} >= MSE f_t: {mse_f_t}")
                break
            
            # Set the new model for the next iteration
            iglb = iglb.fit(train_X, train_y, train_groups)
                
        #if OUTPUTS: print(f"GASCE: {iglb.gasce}\n")
        scores_calibrated = iglb.score_obj.calc_all_new(test_X, 
                                                        test_y, 
                                                        groups=test_groups)
        total_group_calibrated, correctness_group_calibrated, total_bin_calibrated, correctness_bin_calibrated = iglb.score_obj.get_total_and_correctness(test_X, test_y, test_groups) 
        total_group, correctness_group, average_group_confidence = iglb.score_obj.get_correctness_per_group(test_X, test_y, test_groups) 
                
        # Add entry for the run in the score table
        iglb.score_obj.add_to_score_table(run, scores_uncalibrated, scores_calibrated)
        history["score"] = iglb.score_obj.score_table

        with open('iglb_history.pkl', 'wb') as f:
            pickle.dump(history, f)

        if extern:
            return [total_bin_calibrated, 
                    correctness_bin_calibrated, 
                    correctness_group, 
                    average_group_confidence,
                    total_group, 
                    scores_calibrated]
        else:
            print()
            # Charts
            # chartmaker.calibration_info(total_bin_uncalibrated, correctness_bin_uncalibrated, total_bin_calibrated, correctness_bin_calibrated)

    # display score table for all runs
    iglb.score_obj.display_score_table()
    
    # saves the score table
    if save_table:
        with open('results/iglb_'+binning_type+'_results.txt', 'w') as f:
            f.write(iglb.score_obj.printable_table)
if __name__ == "__main__":
    main()
