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
m = 10

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
        if not os.path.isdir(CHART_DIR+run+'/hb/'+binning_type):
            os.makedirs(CHART_DIR+run+'/hb/'+binning_type)

        # load the data from the run directory
        run = run.replace('/', '')
        results, temperature, top_p, num_samples = data.load_multipl_e_run(d)
        run_entry.append(num_samples)
        if OUTPUTS: print(f"\nRun: {run}")
        if OUTPUTS: print(f"Temperature: {temperature}")
        if OUTPUTS: print(f"Num Problems: {num_samples}")

        # probs -> confidence of the model
        # is_correct -> label 1: is correct, 0: is not correct
        probs, is_correct, _, _ = data.proability_and_correctness_for_samples(results)
        correct_count = np.count_nonzero(is_correct == 1)
       
        # split in train and test
        train_probs, test_probs, train_y, test_y = train_test_split(probs, is_correct, test_size=0.33, random_state=42)
        if DEBUG: print(f"Training values: {train_probs}")
        if DEBUG: print(f"Test values: {test_probs}")

        # sets the type of binning
        if binning_type == 'linear':
            # uniform grid 1/m
            grid = binning.create_unform_grid(m)
            bar_width = 1/m 
        elif binning_type == 'quantil':
            # get quantils for step size n
            bin_edges = binning.create_qunatil_grid(train_probs, binning_step_size)
            # get the middle of the bins for hb
            grid = np.array(((bin_edges[1:]-bin_edges[:-1])/2)+bin_edges[:-1]) 
            bar_width = np.array(bin_edges[1:] - bin_edges[:-1])
       
        if OUTPUTS: print(f"Korrekt: {correct_count}") 
        
        # Create calibration object
        hb = hb_calibration(grid, OUTPUTS, DEBUG)

        # Calculates the deltas for the bins
        # fit_correct_per_bin -> correctness per bin
        # train_scores -> collection of scores on the training dataset
        fit_correct_per_bin, fit_total_per_bin, train_scores = hb.fit(train_probs, train_y)

        # Uses the deltas to calculate the corrected values
        # f_dach -> corrected correctness values for each bin
        # test_scores -> collection of scores on the test dataset
        f_dach, test_total_per_bin, test_scores = hb.predict(test_probs, test_y)

        # extract score into lists
        train_scores = list(list(train_scores.values())[0].values())
        test_scores = list(list(test_scores.values())[0].values())
        score_difference = np.array(test_scores) - np.array(train_scores)

        # add them to the output list
        for train, test, diff in zip(train_scores, test_scores, score_difference):
            run_entry.append(train)
            run_entry.append(test)
            run_entry.append(diff)
        
        colors_fit = []
        colors_test = []
        # define chart ranges for display reasons
        if binning_type == 'linear':
            chart_range = np.arange(0, 1.1, 0.1)
            total_bin_count_norm = (fit_total_per_bin-np.min(fit_total_per_bin))/(np.max(fit_total_per_bin)-np.min(fit_total_per_bin))
            for x in total_bin_count_norm:
                colors_fit.append((0.0, 0.0, 1.0, x))
            total_bin_count_norm = (test_total_per_bin-np.min(test_total_per_bin))/(np.max(test_total_per_bin)-np.min(test_total_per_bin))
            for x in total_bin_count_norm:
                colors_test.append((0.0, 0.0, 1.0, x))
        elif binning_type == 'quantil':
            chart_range = grid            
            colors_fit = ['tab:blue']
            colors_test = ['tab:blue']



        # create different charts
        create_charts.calibration_comparision_chart(chart_range, f_dach, fit_correct_per_bin, CHART_DIR+run+'/hb/'+binning_type+"/calibration_chart.png", run)
        create_charts.calibration_bar_chart(chart_range, fit_correct_per_bin, CHART_DIR+run+'/hb/'+binning_type+"/calibration_bar_chart_train.png", run, bar_width, colors_fit, fit_total_per_bin)
        create_charts.calibration_bar_chart(chart_range, f_dach, CHART_DIR+run+'/hb/'+binning_type+"/calibration_bar_chart_f_dach.png", run, bar_width, colors_test, test_total_per_bin)
            
        table_print.append(run_entry)
        
    table_print = tabulate(table_print, headers=['Run', 
                                                 'Num_samples', 
                                                 'ECE_train', 
                                                 'ECE_test',
                                                 'ECE_diff', 
                                                 'ASCE_train', 
                                                 'ASCE_test',
                                                 'ASCE_diff',
                                                 'MSE_train',
                                                 'MSE_test', 
                                                 'MSE_diff',
                                                 'brier_ref_train', 
                                                 'brier_ref_test', 
                                                 'brier_ref_diff', 
                                                 'skill_score_train', 
                                                 'skill_score_test',
                                                'skill_score_diff'], tablefmt='orgtbl')
    print(table_print)

    with open('results/histogramm_binning_'+binning_type+'_results.txt', 'w') as f:
        f.write(table_print)

if __name__ == "__main__":
    main()
