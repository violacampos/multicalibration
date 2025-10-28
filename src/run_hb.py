import numpy as np
import os
from tools import binning, cmd_input
from tools.hb_calibration import Hb_calibration
from tools.data import data_loader
from tools.split import split

DEBUG = False
OUTPUTS = True

#np.seterr(divide='ignore', invalid='ignore')

def main(data_provider, extern=False, grid=None, chartmaker=None):
    # loads commandline parameter
    args = cmd_input.load_parser()
    
    # # get run dir
    # run_dirs = [x[0] for x in os.walk(args.dir[0])]
    # run_dirs.sort()

    # run_dir = run_dirs[0]

    # if run_dir == args.dir[0] and ("humaneval" not in run_dir and "mbpp" not in run_dir) and args.problem == 'code-gen':
    #     exit()

    # # Loads all the necessary data into an dict
    # data_obj = data_loader(args, run_dir, extern, "hb")

    # # Splits the loaded data
    # split_obj = split(args.split, data_obj)

    if grid is None or chartmaker is None:
        # get the grid for binning type and the chartmaker obj
        grid, chartmaker = binning.get_grid_and_chartmaker(data_provider.run,
                                                           args,
                                                           data_provider.save_dir,
                                                           extern, 
                                                           probs=data_provider.get_train_probs(args.prob_method))
            
    # Create calibration object and calculates the deltas
    hb = Hb_calibration(grid, args).fit(data_provider.get_train_probs(args.prob_method), 
                                                  data_provider.get_train_is_correct())

    # calculate scores for the uncalibrated test set
    scores_uncalibrated = hb.score_obj.calc_all(data_provider.get_test_probs(args.prob_method), 
                                                    data_provider.get_test_is_correct(),
                                                    groups=data_provider.get_test_groups(), 
                                                    set_brier_ref=True)
    
    _, _, total_bin_uncalibrated, correctness_bin_uncalibrated = hb.score_obj.get_total_and_correctness(data_provider.get_test_probs(args.prob_method), 
                                                                                                        data_provider.get_test_is_correct(), 
                                                                                                        data_provider.get_test_groups())

    # Uses the deltas to calculate the corrected values
    corrected_probs = hb.predict(data_provider.get_test_probs(args.prob_method)) 

    # calculate scores for the calibrated test set
    scores_calibrated = hb.score_obj.calc_all(  corrected_probs, 
                                                    data_provider.get_test_is_correct(),
                                                    groups=data_provider.get_test_groups())
    
    _, _, total_bin_calibrated, correctness_bin_calibrated = hb.score_obj.get_total_and_correctness(corrected_probs, 
                                                                                                    data_provider.get_test_is_correct(), 
                                                                                                    data_provider.get_test_groups()) 
    
    total_group, correctness_group, average_group_confidence = hb.score_obj.get_correctness_per_group(corrected_probs, 
                                                                                                      data_provider.get_test_is_correct(), 
                                                                                                      data_provider.get_test_groups()) 
                    
    # Add entry for the run in the score table
    hb.score_obj.add_to_score_table(data_provider.run, scores_uncalibrated, scores_calibrated)
    
    # only return values when script is called from another script
    if extern:
        return {"correctness_bin_calibrated": correctness_bin_calibrated, 
                "total_bin_calibrated": total_bin_calibrated, 
                "correctness_group": correctness_group, 
                "average_group_confidence": average_group_confidence, 
                "total_group": total_group, 
                "scores_calibrated": scores_calibrated, 
                "calibrated_probs": corrected_probs,
                "group_names": data_provider.group_names}
    else:
        # Charts
        if args.save_charts:
            chartmaker.calibration_info(total_bin_uncalibrated, correctness_bin_uncalibrated, total_bin_calibrated, correctness_bin_calibrated)
        
    # display score table for all runs
    hb.score_obj.display_score_table()

    # saves the score table
    if args.save_table:
        with open(data_provider.save_dir+'scores.txt', 'w') as f:
            f.write(hb.score_obj.printable_table)

if __name__ == "__main__":
    main()
