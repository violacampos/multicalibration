import run_hb
import run_lr
import run_ighb
import run_iglb
import numpy as np
from tabulate import tabulate
import os

from tools.create_charts import chart_creator

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts/"
m = 10

if __name__ == "__main__":

    run = "humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1"
    binning_type = "linear"

    # create directory for chart generation
    save_dir = CHART_DIR+run+'/comparison/'+binning_type+'/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    grid, uncalib_base, hb_correctness, hb_total, hb_scores = run_hb.main(extern=True)
    lr_total, lr_correctness, lr_scores = run_lr.main(extern=True)
    ighb_total, ighb_correctness, ighb_scores = run_ighb.main(extern=True)
    iglb_total, iglb_correctness, iglb_scores = run_iglb.main(extern=True)

    table_print = []
    table_print.append(["HB"]+list(list(hb_scores.values())[0].values()))
    table_print.append(["LR"]+list(list(lr_scores.values())[0].values()))
    table_print.append(["IGHB"]+list(list(ighb_scores.values())[0].values()))
    table_print.append(["IGLB"]+list(list(iglb_scores.values())[0].values()))

    table_print = tabulate(table_print, headers=['Method', 
                                                'ECE', 
                                                'ASCE', 
                                                'MSE',
                                                'brier_ref',  
                                                'skill_score'], tablefmt='orgtbl')
    print(table_print)
    charts = chart_creator(run, binning_type, grid, save_dir, m)    
    
    if len(ighb_correctness) > 11:
        new_bins = np.arange(0, 1.1, 0.1)
        bin_assignment = []
        for old_bin in run_ighb.grid:             
            bin_assignment.append(new_bins[np.argmin(np.abs(old_bin - new_bins))])

        t = []
        for nb in new_bins:
            t.append(np.sum(ighb_total[bin_assignment == nb]))
        ighb_total = np.array(t)

        t = []
        for nb in new_bins:
            t.append(np.mean(ighb_correctness[bin_assignment == nb]))
        ighb_correctness = np.array(t)


    charts.calibration_method_comp_chart(charts.chart_range,
                                         charts.chart_range[hb_total != 0],
                                         charts.chart_range[lr_total != 0],
                                         charts.chart_range[ighb_total != 0],
                                         charts.chart_range[iglb_total != 0],
                                         uncalib_base, 
                                         hb_correctness[hb_total != 0], 
                                         lr_correctness[lr_total != 0], 
                                         ighb_correctness[ighb_total != 0], 
                                         iglb_correctness[iglb_total != 0])