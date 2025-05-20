import numpy as np
import os
from tools import binning, cmd_input
from tools.hb_calibration import hb_calibration
from tools.data import data_loader
from tools.split import split

DEBUG = False
OUTPUTS = True

np.seterr(divide='ignore', invalid='ignore')

# python run_hb.py --dir ../MultiPL-E/runs/humaneval-all-keep-Qwen2.5_Coder_7B-Instruct-1.0-comp-1 --split --use-scc --prob-method avg_logprob 

def main(extern=False):
    args = cmd_input.load_parser()

    run_dirs = [x[0] for x in os.walk(args.dir[0])]
    run_dirs.sort()

    if args.all_lang == True:
        run_dirs = [run_dirs[0]]

    run_dir = run_dirs[0]

    if run_dir == args.dir[0] and ("humaneval" not in run_dir and "mbpp" not in run_dir) and args.problem == 'code-gen':
        exit()

    # Loads all the necessary data into an dict
    data_obj = data_loader(args, run_dir, extern, "hb")

    # Splits the loaded data
    split_obj = split(args.split, data_obj)

    # get the grid for binning type and the chartmaker obj        
    grid, chartmaker = binning.get_grid_and_chartmaker(data_obj.run,
                                                       args.binning_type, 
                                                       data_obj.save_dir, 
                                                       args.bin_count, 
                                                       extern, 
                                                       probs=split_obj.train_data["probs"], 
                                                       binning_step_size=1/args.bin_count)
            
    # Create calibration object and calculates the deltas
    hb = hb_calibration(grid, OUTPUTS, DEBUG).fit(split_obj.train_data["probs"], 
                                                  split_obj.train_data["is_correct"])

    # calculate scores for the uncalibrated test set
    scores_uncalibrated = hb.score_obj.calc_all_new(split_obj.test_data["probs"], 
                                                    split_obj.test_data["is_correct"],
                                                    groups=split_obj.test_groups, 
                                                    set_brier_ref=True)
    total_group_uncalibrated, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = hb.score_obj.get_total_and_correctness(split_obj.test_data["probs"], 
                                                                                                                                                            split_obj.test_data["is_correct"], 
                                                                                                                                                            split_obj.test_groups)
    total_group_uncalib, correctness_group_uncalib, average_group_confidence_uncalib = hb.score_obj.get_correctness_per_group(split_obj.test_data["probs"], 
                                                                                                                              split_obj.test_data["is_correct"], 
                                                                                                                              split_obj.test_groups)    

    # Uses the deltas to calculate the corrected values
    corrected_probs = hb.predict(split_obj.test_data["probs"]) 

    # calculate scores for the calibrated test set
    scores_calibrated = hb.score_obj.calc_all_new(  corrected_probs, 
                                                    split_obj.test_data["is_correct"],
                                                    groups=split_obj.test_groups)
    total_group_calibrated, correctness_group_calibrated, total_bin_calibrated, correctness_bin_calibrated = hb.score_obj.get_total_and_correctness(corrected_probs, 
                                                                                                                                                    split_obj.test_data["is_correct"], 
                                                                                                                                                    split_obj.test_groups) 
    
    total_group, correctness_group, average_group_confidence = hb.score_obj.get_correctness_per_group(corrected_probs, 
                                                                                                      split_obj.test_data["is_correct"], 
                                                                                                      split_obj.test_groups) 
                    
    # Add entry for the run in the score table
    hb.score_obj.add_to_score_table(data_obj.run, scores_uncalibrated, scores_calibrated)
    
    # only return values when script is called from another script
    if extern:
        return {"correctness_bin_calibrated": correctness_bin_calibrated, 
                "total_bin_calibrated": total_bin_calibrated, 
                "correctness_group": correctness_group, 
                "average_group_confidence": average_group_confidence, 
                "total_group": total_group, 
                "scores_calibrated": scores_calibrated, 
                "calibrated_probs": corrected_probs}
    else:
        # Charts
        if args.save_charts:
            chartmaker.calibration_info(total_bin_uncalibrated, correctness_bin_uncalibrated, total_bin_calibrated, correctness_bin_calibrated)
        
    # display score table for all runs
    hb.score_obj.display_score_table()

    # saves the score table
    if args.save_table:
        with open(data_obj.save_dir+'scores.txt', 'w') as f:
            f.write(hb.score_obj.printable_table)

if __name__ == "__main__":
    main()
