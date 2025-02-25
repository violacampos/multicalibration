import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from tools import data, groups, calibration_scores, binning, create_charts
from tabulate import tabulate
import re
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"

DEBUG = False
OUTPUTS = True

np.seterr(divide='ignore', invalid='ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dirs", type=str,  help="Directories with results. ", nargs="+")
    args = parser.parse_args()

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

        # load the data from the run directory
        run = run.replace('/', '')
        results, temperature, top_p, num_samples = data.load_multipl_e_run(d)

        probs, is_correct, programs, prompts = data.proability_and_correctness_for_samples(results)
        correct_count = np.count_nonzero(is_correct == 1)

        # Define group matrix
        groups_w = []

        for program, prompt in zip(programs, prompts):
            groups_w.append(groups.check_groups(prompt, program))

        groups_w = np.array(groups_w)
        
        print(run)
        print(f"Gruppen summen: {groups_w.sum(axis=0)}")

        X = np.column_stack([groups_w])
        y = is_correct - probs


        # Split in train and test
        train_X, test_X, train_y, test_y, train_probs, test_probs, train_correct, test_correct, train_groups, test_groups = train_test_split(X, y, probs, is_correct, groups_w, test_size=0.33, random_state=42)
        if DEBUG: print(f"Training values: {train_X}")
        if DEBUG: print(f"Test values: {test_X}")

        # Train the linear regression on the train data split
        reg = LinearRegression().fit(train_X, train_y)
        print(reg.score(X, y))

        # 
        predictions = reg.predict(test_X)
        calibrated_predictions = predictions + test_probs

        # Calculate different scores
        gcu = np.round(np.array([np.mean(train_correct[(col == 1)] -  train_probs[(col == 1)]) for col in train_groups.T]), 2)
        gcu[np.isnan(gcu)] = 0

        print(f"Train: Group conditional unbiasednes: {gcu}")

        mse_train = calibration_scores.mse(train_probs, train_correct, len(train_probs))
        print(f"Train: MSE {mse_train}")

        grid = binning.create_unform_grid(10)
        bin_assignements = binning.round_model_to_grid(train_probs, grid)
        train_tot, train_corr, avg = binning.bin_round_probabilities(bin_assignements, train_probs, train_correct, grid)

        asce_train = calibration_scores.asce(train_corr, avg, train_tot,  len(train_probs))
        print(f"Train: ASCE {asce_train}")

        ece_train = calibration_scores.asce(train_corr, avg, train_tot,  len(train_probs))
        print(f"Train: ECE {ece_train}")

        # group conditional unbiasednes
        gcu = np.round(np.array([np.mean(test_correct[(col == 1)] -  calibrated_predictions[(col == 1)]) for col in test_groups.T]), 2)
        gcu[np.isnan(gcu)] = 0

        print(f"Test: Group conditional unbiasednes: {gcu}")

        mse_test = calibration_scores.mse(calibrated_predictions, test_correct,  len(test_probs))
        print(f"Test: MSE {mse_test}")

        bin_assignements = binning.round_model_to_grid(calibrated_predictions, grid)
        test_tot, test_corr, avg = binning.bin_round_probabilities(bin_assignements, calibrated_predictions, test_correct, grid)

        asce_test = calibration_scores.asce(test_corr, avg, test_tot,  len(test_probs))
        print(f"Test: ASCE: {asce_test}")
        
        ece_test = calibration_scores.asce(test_corr, avg, test_tot,  len(test_probs))
        print(f"Test: ECE {ece_test}")
        

        # Charts
        colors_fit = []
        colors_test = []

        total_bin_count_norm = (train_tot-np.min(train_tot))/(np.max(train_tot)-np.min(train_tot))
        for x in total_bin_count_norm:
            colors_fit.append((0.0, 0.0, 1.0, x))
        total_bin_count_norm = (test_tot-np.min(test_tot))/(np.max(test_tot)-np.min(test_tot))
        for x in total_bin_count_norm:
            colors_test.append((0.0, 0.0, 1.0, x))

        chart_range = np.arange(0, 1.1, 0.1)
        bar_width = 0.1

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(run+' # Calibration Charts', fontsize=14)
        create_charts.calibration_bar_chart(axs[0, 0], 'Training set calibration', chart_range, train_corr, bar_width, colors_fit, train_tot)
        create_charts.calibration_bar_chart(axs[0, 1], 'Test set calibration', chart_range, test_corr, bar_width, colors_test, test_tot)
        
        create_charts.count_distribution(axs[1, 0],'Traing set distribution', chart_range, train_tot, bar_width)
        create_charts.count_distribution(axs[1, 1],'Test set distribution', chart_range, test_tot, bar_width)
        
        plt.savefig(BASE_DIR+"calibration_infos.png")
        plt.close() 

        create_charts.calibration_comparision_chart(chart_range[test_tot != 0], chart_range[train_tot != 0], test_corr[test_tot != 0], train_corr[train_tot != 0], BASE_DIR+"/calibration_comparison.png", run)
        

        print(reg.coef_)

            




if __name__ == "__main__":
    main()
