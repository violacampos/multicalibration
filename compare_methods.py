import run_hb
import run_lr
import run_ighb
import run_iglb
import numpy as np
from tabulate import tabulate
import os
from tools import binning
from tools.create_charts import chart_creator

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts/"
m = 10
save_table = True

if __name__ == "__main__":

    run = "humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1"
    binning_type = "linear"        

    # create directory for chart generation
    save_dir = CHART_DIR+run+'/comparison/'+binning_type+'/'
    save_dir = 'results/counter_groups_exp/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    grid, chartmaker = binning.get_grid_and_chartmaker(run, binning_type, save_dir, m)

    hb_results = run_hb.main(extern=True)    
    
    lr_results = run_lr.main(extern=True)  

    ighb_results = run_ighb.main(extern=True)
    
    iglb_results = run_iglb.main(extern=True)

    table_print = []
    table_print.append(["Uncalib"]+list(list(hb_results[10].values())[0].values()))
    table_print.append(["HB"]+list(list(hb_results[9].values())[0].values()))
    table_print.append(["LR"]+list(list(lr_results[5].values())[0].values()))
    table_print.append(["IGHB"]+list(list(ighb_results[5].values())[0].values()))
    table_print.append(["IGLB"]+list(list(iglb_results[5].values())[0].values()))

    table_print = tabulate(table_print, headers=['Method', 
                                                'ECE', 
                                                'ASCE', 
                                                'MSE',
                                                'brier_ref',  
                                                'skill_score',
                                                'GASCE'], tablefmt='orgtbl')
    print(table_print)
    
    if save_table:
        with open('results/counter_groups_exp/comparison_'+binning_type+'_results.txt', 'w') as f:
            f.write(table_print)

    chartmaker.calibration_method_comp_chart(grid,
                                         grid[hb_results[5] != 0],
                                         grid[lr_results[0] != 0],
                                         grid[ighb_results[0] != 0],
                                         grid[iglb_results[0] != 0],
                                         hb_results[0], 
                                         hb_results[4][hb_results[5] != 0], 
                                         lr_results[1][lr_results[0] != 0], 
                                         ighb_results[1] [ighb_results[0]  != 0], 
                                         iglb_results[1][iglb_results[0] != 0])

    chartmaker.group_calibration_scatter(hb_results[1], 
                                         hb_results[2], 
                                         hb_results[3],
                                         hb_results[6], 
                                         hb_results[7], 
                                         hb_results[8],
                                         lr_results[2], 
                                         lr_results[3], 
                                         lr_results[4], 
                                         ighb_results[2], 
                                         ighb_results[3], 
                                         ighb_results[4],
                                         iglb_results[2], 
                                         iglb_results[3], 
                                         iglb_results[4])
    
