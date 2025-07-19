import numpy as np
import os
from tools import binning, cmd_input
from tools.hb_calibration import hb_calibration
from tools.data import data_loader
from tools.split import split
from tools.calibration_scores import score
import matplotlib.pyplot as plt

DEBUG = False
OUTPUTS = True

np.seterr(divide='ignore', invalid='ignore')

# python run_hb.py --dir ../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1 --split --use-scc --prob-method avg_logprob 

def main(extern=False):
    args = cmd_input.load_parser()

    confidence = np.array([1, 0.5, 0.5 , 0])

    label = np.array([1, 1, 0, 0])

    groups = np.array([[1,0],
                       [0,1],
                       [1,0],
                       [0,1]])

   
    # get the grid for binning type and the chartmaker obj        
    grid, chartmaker = binning.get_grid_and_chartmaker("Calibration",
                                                       args.binning_type, 
                                                       "/data/stud/2025-MA-kuschnereit/masterarbeit/tests/calib_images/", 
                                                       args.bin_count, 
                                                       extern, 
                                                       binning_step_size=1/args.bin_count)

    score_obj = score(grid, OUTPUTS, DEBUG)        
    
    # calculate scores for the uncalibrated test set
    scores_uncalibrated = score_obj.calc_all_new(confidence, 
                                                label,
                                                groups=groups, 
                                                set_brier_ref=True)
    total_group_uncalibrated, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = score_obj.get_total_and_correctness(confidence, 
                                                                                                                                                            label, 
                                                                                                                                                            groups)
    """total_group_uncalibrated, correctness_group_uncalibrated, total_bin_uncalibrated_2, correctness_bin_uncalibrated_2 = score_obj.get_total_and_correctness(confidence[groups == 0], 
                                                                                                                                                            label[groups == 0], 
                                                                                                                                                            groups[groups == 0])"""
                    

    #chartmaker.calibration_info(total_bin_uncalibrated, correctness_bin_uncalibrated, total_bin_calibrated, correctness_bin_calibrated)
    """fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title("", fontsize=16, fontweight="bold")
    w, x = 0.05, np.arange(len(chartmaker.chart_grid))
    bars = ax.bar(chartmaker.chart_grid - w/2, correctness_bin_uncalibrated, width = w, color="blue", edgecolor='black', label='Group 1')
    bars_2 = ax.bar(chartmaker.chart_grid + w/2, correctness_bin_uncalibrated_2, width = w, color="red", edgecolor='black', label='Group 2')
    ax.plot([0, 1], [0, 1], linestyle='--')


    if total_bin_uncalibrated is not None:
        ax.bar_label(bars, total_bin_uncalibrated, fontsize=14)

    if total_bin_uncalibrated_2 is not None:
        ax.bar_label(bars_2, total_bin_uncalibrated_2, fontsize=14)   
    
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))   
    plt.xlabel('Confidence', fontsize=18)
    plt.ylabel('Correct', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(chartmaker.save_dir+"calibration_groups.pdf")
    plt.close() """
    

if __name__ == "__main__":
    main()
