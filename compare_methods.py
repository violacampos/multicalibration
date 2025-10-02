import numpy as np
import run_hb
import run_lr
import run_ighb
import run_iglb
import compute_baseline
from tabulate import tabulate
import os
from tools import binning, cmd_input
from tools.split import split
from tools.data import data_loader
import pickle

from tools.dataset import GroupConfig, LiveCodeBenchDataset

if __name__ == "__main__":
    # loads commandline parameter
    args = cmd_input.load_parser()
    
    if args.benchmark == "livecodebench":
        config = GroupConfig(add_counter=False, 
                 larger_than_median_loc=True, 
                 larger_than_median_prompt=True,
                 larger_than_median_output=True,
                 difficulty_easy=True,
                 difficulty_medium=True,
                 difficulty_hard=True)
        split_obj = LiveCodeBenchDataset(
                jsonl_path =args.data_path, 
                 split='train', 
                 group_config=config)
        
    else:

        # get run dir
        run_dirs = [x[0] for x in os.walk(args.dir[0])]
        run_dirs.sort()

        run_dir = run_dirs[0]

        # Load data and setup grid/chartmaker
        data_obj = data_loader(args, run_dir, False, "comparison")
        # Splits the loaded data
        split_obj = split(args.split, data_obj)
        


    grid, chartmaker = binning.get_grid_and_chartmaker(split_obj.run, 
                                                       args, 
                                                       split_obj.save_dir, 
                                                       False)
    
    # execute every calibration approach
    print("Baseline:")
    baseline_results = compute_baseline.main(extern=True, data_provider=split_obj, grid=grid, chartmaker=chartmaker)    

    print("Histogram binning:")
    hb_results = run_hb.main(extern=True)    
    
    print("Linear regression:")
    lr_results = run_lr.main(extern=True)  

    print("Iterative group histogram binning:")
    ighb_results = run_ighb.main(extern=True)
    
    print("Iterative group linear binning:")
    iglb_results = run_iglb.main(extern=True)

    # Collect results in table and print table
    table_print = []
    table_print.append(["Uncalib"]+list(list(baseline_results["scores_uncalibrated"].values())[0].values()))
    table_print.append(["HB"]+list(list(hb_results["scores_calibrated"].values())[0].values()))
    table_print.append(["LR"]+list(list(lr_results["scores_calibrated"].values())[0].values()))
    table_print.append(["IGHB"]+list(list(ighb_results["scores_calibrated"].values())[0].values()))
    table_print.append(["IGLB"]+list(list(iglb_results["scores_calibrated"].values())[0].values()))

    table_print = tabulate(table_print, headers=['Method', 
                                                'ECE', 
                                                'ASCE', 
                                                'MSE',
                                                'brier_ref',  
                                                'skill_score',
                                                'GASCE'], tablefmt='orgtbl')
    print(table_print)

    if args.save_table:
        with open(split_obj.save_dir+'scores.txt', 'w') as f:
            f.write(table_print)

    # Create charts for comparison
    if args.save_charts:
        chartmaker.calibration_method_comp_chart(grid,
                                            grid[hb_results["total_bin_calibrated"] != 0],
                                            grid[lr_results["total_bin_calibrated"] != 0],
                                            grid[ighb_results["total_bin_calibrated"] != 0],
                                            grid[iglb_results["total_bin_calibrated"] != 0],
                                            baseline_results["correctness_bin_uncalibrated"], 
                                            hb_results["correctness_bin_calibrated"][hb_results["total_bin_calibrated"] != 0], 
                                            lr_results["correctness_bin_calibrated"][lr_results["total_bin_calibrated"] != 0], 
                                            ighb_results["correctness_bin_calibrated"][ighb_results["total_bin_calibrated"]  != 0], 
                                            iglb_results["correctness_bin_calibrated"][iglb_results["total_bin_calibrated"] != 0])
        

        
        chartmaker.calibration_method_comp_bar_chart(baseline_results["total_bin_uncalibrated"],
                                            hb_results["total_bin_calibrated"],
                                            lr_results["total_bin_calibrated"],
                                            ighb_results["total_bin_calibrated"],
                                            iglb_results["total_bin_calibrated"],
                                            baseline_results["correctness_bin_uncalibrated"], 
                                            hb_results["correctness_bin_calibrated"], 
                                            lr_results["correctness_bin_calibrated"], 
                                            ighb_results["correctness_bin_calibrated"], 
                                            iglb_results["correctness_bin_calibrated"])
        
        chartmaker.group_calibration_scatter(baseline_results["correctness_group_uncalib"], 
                                            baseline_results["average_group_confidence_uncalib"], 
                                            baseline_results["total_group_uncalib"],
                                            hb_results["correctness_group"], 
                                            hb_results["average_group_confidence"], 
                                            hb_results["total_group"],
                                            lr_results["correctness_group"], 
                                            lr_results["average_group_confidence"], 
                                            lr_results["total_group"], 
                                            ighb_results["correctness_group"], 
                                            ighb_results["average_group_confidence"], 
                                            ighb_results["total_group"],
                                            iglb_results["correctness_group"], 
                                            iglb_results["average_group_confidence"], 
                                            iglb_results["total_group"])
    
    # save the results for further analysis
    if args.save_data:      
        data = {
            "calibrated_probs_hb": hb_results["calibrated_probs"],
            "calibrated_probs_lr": lr_results["calibrated_probs"],
            "calibrated_probs_ighb": ighb_results["calibrated_probs"],                
            "calibrated_probs_iglb": iglb_results["calibrated_probs"],                
            "uncalibrated_probs": baseline_results["uncalibrated_probs"],
            "is_correct": baseline_results["is_correct"],
            "language": baseline_results["language"],
            "groups": baseline_results["groups"],
            "names": baseline_results["names"],
            "programs": baseline_results["programs"],
            "prompts": baseline_results["prompts"],
            "token_logprobs": baseline_results["token_logprobs"]
        }

        with open(split_obj.save_dir+'calibration_data/calibration.pkl', 'wb') as f:
            pickle.dump(data, f)