import numpy as np
from pathlib import Path
import itertools
import argparse
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from tools import data, binning
from tools.hb_calibration import hb_calibration
from tabulate import tabulate
import matplotlib.pyplot as plt

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts/"

binning_type = 'linear'
binning_step_size = 0.1
m = 10

all_lang = True
use_train_test_split = True
save_table = True

DEBUG = False
OUTPUTS = True

np.seterr(divide='ignore', invalid='ignore')

def main(extern=False):
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
        
        # Get run name
        run = d.split("/runs/", 1)[1]

        # create directory for chart generation
        if not os.path.isdir(CHART_DIR+run+'/hb/'+binning_type):
            os.makedirs(CHART_DIR+run+'/hb/'+binning_type)

        # load the data from the run directory
        run = run.replace('/', '')
        results, temperature, top_p, num_samples = data.load_multipl_e_run(d)
               
        print(f"\nRun: {run}")
        save_dir = CHART_DIR+run+'/hb/'+binning_type+'/'
        if OUTPUTS: print(f"Temperature: {temperature}")
        if OUTPUTS: print(f"Num Problems: {num_samples}")

        # probs -> confidence of the model
        # is_correct -> label 1: is correct, 0: is not correct
        probs, is_correct, _, _ = data.proability_and_correctness_for_samples(results)
        correct_count = np.count_nonzero(is_correct == 1)

        if use_train_test_split:
            # split in 60% train, 20% validation and 20% test
            train_probs, test_probs, train_y, test_y = train_test_split(probs, is_correct, test_size=0.2, random_state=42)

            train_probs, val_probs, train_y, val_y = train_test_split(train_probs, train_y, test_size=0.25, random_state=42)

            # train_probs, test_probs, train_y, test_y = train_test_split(probs, is_correct, test_size=0.33, random_state=42)
            if DEBUG: print(f"Training values: {train_probs}")
            if DEBUG: print(f"Test values: {test_probs}")
        else:
            train_probs = probs
            test_probs = probs
            train_y = is_correct
            test_y = is_correct

        # get the grid for binning type and the chartmaker obj        
        grid, chartmaker = binning.get_grid_and_chartmaker(run, binning_type, save_dir, m, train_probs, binning_step_size)
               
        # Create calibration object and calculates the deltas
        hb = hb_calibration(grid, OUTPUTS, DEBUG).fit(train_probs, train_y)

        # calculate scores for the uncalibrated test set
        total_uncalibrated, correctness_uncalibrated, scores_uncalibrated = hb.score_obj.calc_all_new(test_probs, 
                                                                                                      test_y, 
                                                                                                      deltas=hb.get_deltas(test_probs, test_y),
                                                                                                      set_brier_ref=True)

        # Uses the deltas to calculate the corrected values
        corrected_probs = hb.predict(test_probs) 

        # calculate scores for the calibrated test set
        total_calibrated, correctness_calibrated, scores_calibrated = hb.score_obj.calc_all_new(corrected_probs, 
                                                                                                test_y,
                                                                                                deltas=hb.get_deltas(corrected_probs, test_y))
                       
        # Add entry for the run in the score table
        hb.score_obj.add_to_score_table(run, scores_uncalibrated, scores_calibrated)
        
        # only return values when script is called from another script
        if extern:
            return grid, correctness_uncalibrated, correctness_calibrated, total_calibrated, scores_calibrated
        else:
            # Charts
            chartmaker.calibration_info(total_uncalibrated, correctness_uncalibrated, total_calibrated, correctness_calibrated)
        
    # display score table for all runs
    hb.score_obj.display_score_table()

    # saves the score table
    if save_table:
        with open('results/hb_'+binning_type+'_results.txt', 'w') as f:
            f.write(hb.score_obj.printable_table)



if __name__ == "__main__":
    main()
