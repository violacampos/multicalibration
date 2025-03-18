import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from tools import data, calibration_scores, binning
from tools.groups import groups
from tools.create_charts import chart_creator
from tools.lr_calibration import lr_calibration
from tabulate import tabulate

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts/"

DEBUG = False
OUTPUTS = True

binning_type = 'linear'
binning_step_size = 0.1
m = 10

use_train_test_split = True
control_exp = False

all_lang = True
save_table = True

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
        save_dir = CHART_DIR+run+'/group_lr/'+binning_type+'/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        probs, is_correct, programs, prompts = data.proability_and_correctness_for_samples(results)

        # Define group matrix
        groups_w = groups(programs, prompts).create_groups()
        groups_w = np.array(groups_w)
        
        if OUTPUTS: print(f"Run: {run}")
        if OUTPUTS: print(f"Gruppen Anzahl: {groups_w.sum(axis=0)}")

        # Set the parameters for the linear regression
        X = np.column_stack([groups_w])

        # Control experiment to check which groups helps the model to make correct predictions
        if control_exp:
            y = is_correct
        else:
            y = is_correct - probs

        # check if we split the data or use the whole dataset for evaluation
        if use_train_test_split:
            # Split in train and test
            train_X, test_X, train_y, test_y, train_probs, test_probs, train_label, test_label, train_groups, test_groups = train_test_split(X, y, probs, is_correct, groups_w, test_size=0.33, random_state=42)
            if DEBUG: print(f"Training values: {train_X}")
            if DEBUG: print(f"Test values: {test_X}")
        else:
            train_X = X
            test_X = X
            train_y = y
            test_y = y
            train_probs = probs
            test_probs = probs
            train_label = is_correct
            test_label = is_correct
            train_groups = groups_w
            test_groups = groups_w
       
        # get the grid for binning type and the chartmaker obj        
        grid, chartmaker = binning.get_grid_and_chartmaker(run, binning_type, save_dir, m, train_probs, binning_step_size)

        # Train the linear regression on the train data split
        lr = lr_calibration(grid, OUTPUTS, DEBUG).fit(train_X, train_y)

        # Calculate scores on uncalibrated test set
        total_uncalibrated, correctness_uncalibrated, scores_uncalibrated = lr.score_obj.calc_all_new(test_probs, 
                                                                                                      test_label, 
                                                                                                      groups=test_groups,
                                                                                                      deltas=lr.get_deltas(test_probs, test_label), 
                                                                                                      set_brier_ref=True)
       
        # Make predictions
        predictions = lr.predict(test_X)

        # use predictions to calibrate
        calibrated_predictions = predictions + test_probs

        # Calculate scores on uncalibrated test set
        total_calibrated, correctness_calibrated, scores_calibrated = lr.score_obj.calc_all_new(calibrated_predictions, 
                                                                                                test_label, 
                                                                                                groups=test_groups,
                                                                                                deltas=lr.get_deltas(calibrated_predictions, test_label))
       
        if OUTPUTS: print(f"Group Lamdas: {lr.reg.coef_}")

        # Add entry for the run in the score table
        lr.score_obj.add_to_score_table(run, scores_uncalibrated, scores_calibrated)
        
        # only return values when called from other script
        if extern:
            return total_calibrated, correctness_calibrated, scores_calibrated
        else:
            # Charts
            chartmaker.calibration_info(total_uncalibrated, correctness_uncalibrated, total_calibrated, correctness_calibrated)
    
    # display score table for all runs
    lr.score_obj.display_score_table()

    # saves the score table
    if save_table:
        with open('results/lr_'+binning_type+'_results.txt', 'w') as f:
            f.write(lr.score_obj.printable_table)

if __name__ == "__main__":
    main()
