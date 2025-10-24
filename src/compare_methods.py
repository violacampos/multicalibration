import os

from tabulate import tabulate

import numpy as np
import run_hb
import run_lr
import run_ighb
import run_iglb
import compute_baseline
import run_platt
from tools import binning, cmd_input
from tools.create_latex_comands import print_latex_commands

import pickle

from tools.dataset import GroupConfig, HumanEvalDataset, LiveCodeBenchDataset

if __name__ == "__main__":
    # loads commandline parameter
    args = cmd_input.load_parser()
    
    
        # TODO replace hardcoded configs
        
    configs = {
        "livecodebench": GroupConfig(
            add_counter=False, 
            language=False,
            larger_than_median_loc=True,
            larger_than_median_prompt=True,
            larger_than_median_output=True,
            difficulty_easy=True,
            difficulty_medium=True,
            difficulty_hard=True),
        "mceval": GroupConfig(
            add_counter=False, 
            language=True,
            larger_than_median_loc=True, 
            larger_than_median_prompt=True,
            larger_than_median_output=True,
            difficulty_easy=True,
            difficulty_medium=True, 
            difficulty_hard=True),
        "humaneval": GroupConfig(
            add_counter=True,
            larger_than_median_loc=True,
            larger_than_median_prompt=True,
            difficulty_easy=False,
            difficulty_medium=False,
            difficulty_hard=False,
            language=True
        )}
    
    config = configs[args.benchmark]
        
    if args.benchmark in ["livecodebench", "mceval"]:
        
        split_obj = LiveCodeBenchDataset(
                jsonl_path =args.data_path, 
                split='train', 
                benchmark=args.benchmark,
                group_config=config,
                args=args)
        
    else:

        # get run dir
        run_dirs = [x[0] for x in os.walk(args.dir[0])]
        run_dirs.sort()

        run_dir = run_dirs[0]
        
        split_obj = HumanEvalDataset(
            jsonl_path =args.data_path, 
            run_dir=run_dir, 
            group_config=config)
        
        


    grid, chartmaker = binning.get_grid_and_chartmaker(split_obj.run, 
                                                       args, 
                                                       split_obj.save_dir, 
                                                       False)
    
    # execute every calibration approach
    print("Baseline:")
    baseline_results = compute_baseline.main(extern=True, data_provider=split_obj, grid=grid, chartmaker=chartmaker)    

    print("Platt scaling:")
    platt_results = run_platt.main(extern=True, data_provider=split_obj, grid=grid, chartmaker=chartmaker)    
    
    print("Histogram binning:")
    hb_results = run_hb.main(extern=True, data_provider=split_obj, grid=grid, chartmaker=chartmaker)    
    
    print("Linear regression:")
    lr_results = run_lr.main(type='linear', extern=True, data_provider=split_obj, grid=grid, chartmaker=chartmaker)  

    print("Logistic regression:")
    logr_results = run_lr.main(type='logistic', extern=True, data_provider=split_obj, grid=grid, chartmaker=chartmaker)  

    print("Iterative group histogram binning:")
    ighb_results = run_ighb.main(extern=True, data_provider=split_obj, grid=grid, chartmaker=chartmaker)

    print("Iterative group linear binning:")
    iglb_results = run_iglb.main(extern=True, data_provider=split_obj, grid=grid, chartmaker=chartmaker)

    # Collect results in table and print table
    table_print = []
    table_print.append(["Uncalib"]+list(list(baseline_results["scores_uncalibrated"].values())[0].values()))
    table_print.append(["Platt"]+list(list(platt_results["scores_calibrated"].values())[0].values()))
    table_print.append(["HB"]+list(list(hb_results["scores_calibrated"].values())[0].values()))
    table_print.append(["LR"]+list(list(lr_results["scores_calibrated"].values())[0].values()))
    table_print.append(["LOGR"]+list(list(logr_results["scores_calibrated"].values())[0].values()))
    table_print.append(["IGHB"]+list(list(ighb_results["scores_calibrated"].values())[0].values()))
    table_print.append(["IGLB"]+list(list(iglb_results["scores_calibrated"].values())[0].values()))

    table_print = tabulate(table_print, headers=['Method', 
                                                'ECE', 
                                                'ASCE', 
                                                'MSE',
                                                'brier_ref',  
                                                'skill_score',
                                                'accuracy',
                                                'GASCE'], tablefmt='orgtbl')
    print(table_print)
    

    if args.save_table:
        with open(split_obj.save_dir+f'scores_{args.prob_method}.txt', 'w') as f:
            f.write(table_print)

    # Create charts 
    if args.save_charts:
        
        chartmaker.calibration_method_comp_bar_chart(args.prob_method,
                                            baseline_results["total_bin_uncalibrated"],
                                            platt_results["total_bin_calibrated"],
                                            hb_results["total_bin_calibrated"],
                                            lr_results["total_bin_calibrated"],
                                            logr_results["total_bin_calibrated"],
                                            ighb_results["total_bin_calibrated"],
                                            iglb_results["total_bin_calibrated"],
                                            baseline_results["correctness_bin_uncalibrated"], 
                                            platt_results["correctness_bin_calibrated"], 
                                            hb_results["correctness_bin_calibrated"], 
                                            lr_results["correctness_bin_calibrated"], 
                                            logr_results["correctness_bin_calibrated"], 
                                            ighb_results["correctness_bin_calibrated"], 
                                            iglb_results["correctness_bin_calibrated"])
        
        chartmaker.group_calibration_scatter(args.prob_method,
                                             baseline_results["correctness_group_uncalib"], 
                                            baseline_results["average_group_confidence_uncalib"], 
                                            baseline_results["total_group_uncalib"],
                                            platt_results["correctness_group"], 
                                            platt_results["average_group_confidence"], 
                                            platt_results["total_group"],
                                            hb_results["correctness_group"], 
                                            hb_results["average_group_confidence"], 
                                            hb_results["total_group"],
                                            lr_results["correctness_group"], 
                                            lr_results["average_group_confidence"], 
                                            lr_results["total_group"], 
                                            logr_results["correctness_group"], 
                                            logr_results["average_group_confidence"], 
                                            logr_results["total_group"], 
                                            ighb_results["correctness_group"], 
                                            ighb_results["average_group_confidence"], 
                                            ighb_results["total_group"],
                                            iglb_results["correctness_group"], 
                                            iglb_results["average_group_confidence"], 
                                            iglb_results["total_group"])
        
    
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