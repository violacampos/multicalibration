import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from tools import data, groups, calibration_scores, binning, create_charts
from tabulate import tabulate
import re
from sklearn.linear_model import LinearRegression

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

        groups_w = []

        for program, prompt in zip(programs, prompts):
            """plines = len(program.splitlines()) # Programm Lines
            print(f"Programm lines: {plines}")
            plen = len(program)
            print(f"Programm length: {plen}")
            contains_import = 1 if "import" in program else 0
            print(f"Import: {contains_import}")
            prompt_greater_then_500 = 1 if len(prompt) >= 500 else 0
            print(f"prompt_greater_then_500: {prompt_greater_then_500}")
            has_examples = 1 if "example" in prompt else 0
            print(f"has_examples: {has_examples}")
            #m = re.search("def ([a-zA-Z0-9_](\w))\w+", prog)
            input_vars = re.search("\((.*?)\)", program)
            input_var_count = len(input_vars[0].split(","))
            print(f"Input var count: {input_var_count}")
            contains_return = 1 if "return" in program else 0
            print(f"Return: {contains_return}")"""

            groups_w.append(groups.check_groups(prompt, program))

        groups_w = np.array(groups_w)
        print(run)
        #print(groups_w.sum(axis=0))

        E = np.sum(is_correct*probs) / probs.sum()
        #print(E)

        X = np.column_stack([probs, groups_w])
        y = np.clip(is_correct, 0, 1)

        # Split in train and test
        train_X, test_X, train_y, test_y, train_probs, test_probs, train_correct, test_correct, train_groups, test_groups = train_test_split(X, y, probs, is_correct, groups_w, test_size=0.33, random_state=42)
        if DEBUG: print(f"Training values: {train_X}")
        if DEBUG: print(f"Test values: {test_X}")

        # Train the linear regression on the train data split
        reg = LinearRegression().fit(train_X, train_y)
        print(reg.score(X, y))

        # 
        predictions = reg.predict(test_X)

        # group conditional unbiasednes
        gcu = np.round(np.array([np.mean(train_y[(col == 1)] -  train_probs[(col == 1)]) for col in train_groups.T]), 2)
        gcu[np.isnan(gcu)] = 0

        print(gcu)

        mse_train = calibration_scores.mse(train_probs, train_y, len(train_probs))
        print(f"mse_train: {mse_train}")

        grid = binning.create_unform_grid(10)
        bin_assignements = binning.round_model_to_grid(train_probs, grid)
        tot, corr_1, avg = binning.bin_round_probabilities(bin_assignements, train_probs, train_correct, grid)

        asce_train = calibration_scores.asce(corr_1, avg, tot,  len(train_probs))
        print(f"asce_train: {asce_train}")


        """mse_test = calibration_scores.mse(test_probs, deltas,  len(test_probs))
        print(f"mse_test: {mse_test}")
        bin_assignements = binning.round_model_to_grid(test_probs, grid)
        tot, corr_2, avg = binning.bin_round_probabilities(bin_assignements, test_probs, deltas, grid)

        asce_test = calibration_scores.asce(corr_2, avg, tot,  len(test_probs))
        print(f"asce_test: {asce_test}")

        create_charts.calibration_comparision_chart(grid, corr_2, corr_1, BASE_DIR+"test.png", run)"""

        print(reg.coef_)

            




if __name__ == "__main__":
    main()
