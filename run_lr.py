import numpy as np
import os
from tools import binning, cmd_input
from tools.lr_calibration import lr_calibration
from tools.data import data_loader
from tools.split import split

DEBUG = False
OUTPUTS = True

np.seterr(divide='ignore', invalid='ignore')

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
    data_obj = data_loader(args, run_dir, extern, "lr")

    # Splits the loaded data
    split_obj = split(args.split, data_obj)

    # Control experiment to check which groups helps the model to make correct predictions
    if args.control_exp:
        y = split_obj.train_data["is_correct"]
    else:
        y = split_obj.train_data["is_correct"] - split_obj.train_data["probs"]
    
    # get the grid for binning type and the chartmaker obj        
    grid, chartmaker = binning.get_grid_and_chartmaker(data_obj.run, 
                                                       args.binning_type, 
                                                       data_obj.save_dir, 
                                                       args.bin_count, 
                                                       extern, 
                                                       probs=split_obj.train_data["probs"], 
                                                       binning_step_size=1/args.bin_count)

    # Train the linear regression on the train data split
    lr = lr_calibration(grid, OUTPUTS, DEBUG).fit(split_obj.train_groups, 
                                                  y)

    # Calculate scores on uncalibrated test set
    scores_uncalibrated = lr.score_obj.calc_all_new(split_obj.test_data["probs"], 
                                                    split_obj.test_data["is_correct"], 
                                                    groups=split_obj.test_groups,
                                                    set_brier_ref=True)
    total_group_uncalibrated, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = lr.score_obj.get_total_and_correctness(split_obj.test_data["probs"], 
                                                                                                                                                            split_obj.test_data["is_correct"], 
                                                                                                                                                            split_obj.test_groups)
    
    # Make predictions
    predictions = lr.predict(split_obj.test_groups)

    # use predictions to calibrate
    if args.control_exp:
        calibrated_predictions = predictions
    else:
        calibrated_predictions = predictions + split_obj.test_data["probs"]

    # Calculate scores on uncalibrated test set
    scores_calibrated = lr.score_obj.calc_all_new(  calibrated_predictions, 
                                                    split_obj.test_data["is_correct"], 
                                                    groups=split_obj.test_groups)
    
    total_group_calibrated, correctness_group_calibrated, total_bin_calibrated, correctness_bin_calibrated = lr.score_obj.get_total_and_correctness(calibrated_predictions, 
                                                                                                                                                    split_obj.test_data["is_correct"], 
                                                                                                                                                    split_obj.test_groups) 
    
    total_group, correctness_group, average_group_confidence = lr.score_obj.get_correctness_per_group(calibrated_predictions, 
                                                                                                      split_obj.test_data["is_correct"], 
                                                                                                      split_obj.test_groups) 
    
    if OUTPUTS: print(f"Group Lamdas: {lr.reg.coef_}")

    # Add entry for the run in the score table
    lr.score_obj.add_to_score_table(data_obj.run, scores_uncalibrated, scores_calibrated)
    
    # only return values when called from other script
    if extern:
        return {"total_bin_calibrated": total_bin_calibrated, 
                "correctness_bin_calibrated": correctness_bin_calibrated, 
                "correctness_group": correctness_group, 
                "average_group_confidence": average_group_confidence, 
                "total_group": total_group,
                "scores_calibrated": scores_calibrated,
                "calibrated_probs": calibrated_predictions}
    else:
        if args.save_charts:
            chartmaker.calibration_info(total_bin_uncalibrated, correctness_bin_uncalibrated, total_bin_calibrated, correctness_bin_calibrated)
    
    # display score table for all runs
    lr.score_obj.display_score_table()

    # saves the score table
    if args.save_table:
        with open(data_obj.save_dir+'scores.txt', 'w') as f:
            f.write(lr.score_obj.printable_table)

if __name__ == "__main__":
    main()
