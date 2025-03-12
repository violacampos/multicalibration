import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from tools import data, groups, calibration_scores, binning
from tools.create_charts import chart_creator
from tools.LR_calibration import LR_calibration
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
unit_test = True

all_lang = True

np.seterr(divide='ignore', invalid='ignore')

def main():
    grid = binning.create_unform_grid(m)
    
    run_test_1(grid)
    run_test_2(grid)
    run_test_3(grid)
    run_test_4(grid)
    


def run_test_1(grid):
    probs = np.array([0.5,
                      0.5,
                      0.5,
                      0.5])

    is_correct = np.array([1,
                           0,
                           1,
                           0])

    X = np.array(   [[1, 0],
                    [0, 1],
                    [1, 0],
                    [0, 1]])
                            
    
    y = is_correct - probs

    lr_calib = LR_calibration(grid, OUTPUTS, DEBUG).fit(X, y)

    predictions = lr_calib.predict(X)
    calibrated_predictions = np.round(predictions + probs, 2)

    expected = np.array(    [1,
                            0,
                            1,
                            0])

    assert (expected == calibrated_predictions).all() , "Test 1 failed. (correct in one group, uncorrect in the other)"

def run_test_2(grid):
    probs = np.array([0.5,
                      0.5,
                      0.5,
                      0.5])

    is_correct = np.array([1,
                           0,
                           1,
                           0])

    X = np.array(   [[1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0]])
                            
    
    y = is_correct - probs

    lr_calib = LR_calibration(grid, OUTPUTS, DEBUG).fit(X, y)

    predictions = lr_calib.predict(X)
    calibrated_predictions = np.round(predictions + probs, 2)

    expected = np.array(   [0.5,
                            0.5,
                            0.5,
                            0.5])

    assert (expected == calibrated_predictions).all() , "Test 2 failed. (two correct, two uncorrect all in one group)"

def run_test_3(grid):
    probs = np.array([0.5,
                      0.5,
                      0.5,
                      0.5])

    is_correct = np.array([1,
                           1,
                           1,
                           1])

    X = np.array(   [[0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0]])
                            
    
    y = is_correct - probs

    lr_calib = LR_calibration(grid, OUTPUTS, DEBUG).fit(X, y)

    predictions = lr_calib.predict(X)
    calibrated_predictions = np.round(predictions + probs, 2)

    expected = np.array(   [1,
                            1,
                            1,
                            1])

    assert (expected == calibrated_predictions).all() , "Test 3 failed. (no group, all correct)"

def run_test_4(grid):
    probs = np.array([0,
                      0.5,
                      1])

    is_correct = np.array([1,
                           1,
                           0])

    X = np.array(   [[1, 0],
                    [0, 1],
                    [1, 1]])
                            
    
    y = is_correct - probs

    lr_calib = LR_calibration(grid, OUTPUTS, DEBUG).fit(X, y)

    predictions = lr_calib.predict(X)
    calibrated_predictions = np.round(predictions + probs, 2)

    expected = np.array(   [1,
                            1,
                            0])

    assert (expected == calibrated_predictions).all() , "Test 3 failed. (no group, all correct)"

if __name__ == "__main__":
    main()
