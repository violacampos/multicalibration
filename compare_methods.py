import run_hb
import run_lr
import run_ighb
import run_iglb
import compute_baseline
from tabulate import tabulate
import os
from tools import binning, cmd_input
from tools.data import data_loader
import pickle

if __name__ == "__main__":
    args = cmd_input.load_parser()

    run = args.dir[0].split('/')[-1]
    
    run_dirs = [x[0] for x in os.walk(args.dir[0])]
    run_dirs.sort()

    if args.all_lang == True:
        run_dirs = [run_dirs[0]]

    run_dir = run_dirs[0]

    data_obj = data_loader(args, run_dir, False, "comparison")
    
    grid, chartmaker = binning.get_grid_and_chartmaker(run, 
                                                       args.binning_type, 
                                                       data_obj.save_dir, 
                                                       args.bin_count, 
                                                       False)
    
    baseline_results = compute_baseline.main(extern=True)    

    hb_results = run_hb.main(extern=True)    
    
    lr_results = run_lr.main(extern=True)  

    ighb_results = run_ighb.main(extern=True)
    
    iglb_results = run_iglb.main(extern=True)

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
    # CUDA_VISIBLE_DEVICES=6 python compare_methods.py ../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1

    if args.save_table:
        with open(data_obj.save_dir+'scores.txt', 'w') as f:
            f.write(table_print)

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

        with open(data_obj.save_dir+'calibration_data/calibration.pkl', 'wb') as f:
            pickle.dump(data, f)