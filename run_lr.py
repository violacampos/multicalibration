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
load_scc_results = True

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

        # Set the parameters for the linear regression
        X = np.column_stack([groups_w])

        # Control experiment to check which groups helps the model to make correct predictions
        if control_exp:
            y = is_correct
        else:
            y = is_correct - probs

        # check if we split the data or use the whole dataset for evaluation
        if use_train_test_split:
            # split in 60% train, 20% validation and 20% test
            train_X, test_X, train_y, test_y, train_probs, test_probs, train_label, test_label, train_groups, test_groups = train_test_split(X, y, probs, is_correct, groups_w, test_size=0.2, random_state=42)

            train_X, val_X, train_y, val_y, train_probs, val_probs, train_label, val_label, train_groups, val_groups = train_test_split(train_X, train_y, train_probs, train_label, train_groups, test_size=0.25, random_state=42)

            #train_X, test_X, train_y, test_y, train_probs, test_probs, train_label, test_label, train_groups, test_groups = train_test_split(X, y, probs, is_correct, groups_w, test_size=0.33, random_state=42)
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
        scores_uncalibrated = lr.score_obj.calc_all_new(test_probs, 
                                                        test_label, 
                                                        groups=test_groups,
                                                        set_brier_ref=True)
        #total_group_uncalibrated, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = lr.score_obj.get_total_and_correctness(test_probs, test_label, test_groups)
        
        # Make predictions
        predictions = lr.predict(test_X)

        # use predictions to calibrate
        calibrated_predictions = predictions + test_probs

        # Calculate scores on uncalibrated test set
        scores_calibrated = lr.score_obj.calc_all_new(  calibrated_predictions, 
                                                        test_label, 
                                                        groups=test_groups)
        total_group_calibrated, correctness_group_calibrated, total_bin_calibrated, correctness_bin_calibrated = lr.score_obj.get_total_and_correctness(calibrated_predictions, test_label, test_groups) 
        total_group, correctness_group, average_group_confidence = lr.score_obj.get_correctness_per_group(calibrated_predictions, test_label, test_groups) 
        
        if OUTPUTS: print(f"Group Lamdas: {lr.reg.coef_}")

        print(correctness_bin_calibrated)

        # Add entry for the run in the score table
        lr.score_obj.add_to_score_table(run, scores_uncalibrated, scores_calibrated)
        
        # only return values when called from other script
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
            #chartmaker.calibration_info(total_uncalibrated, correctness_uncalibrated, total_calibrated, correctness_calibrated)
    
    # display score table for all runs
    lr.score_obj.display_score_table()

    # saves the score table
    if save_table:
        with open('results/lr_'+binning_type+'_results.txt', 'w') as f:
            f.write(lr.score_obj.printable_table)

if __name__ == "__main__":
    main()
