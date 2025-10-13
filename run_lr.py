import numpy as np
import os
from tools import binning, cmd_input
from tools.lr_calibration import LR_calibration
from tools.data import data_loader
from tools.split import split

DEBUG = False
OUTPUTS = True

np.seterr(divide='ignore', invalid='ignore')

def main(data_provider, type='linear', extern=False, grid=None, chartmaker=None):
    args = cmd_input.load_parser()

    # run_dirs = [x[0] for x in os.walk(args.dir[0])]
    # run_dirs.sort()

    # run_dir = run_dirs[0]

    # if run_dir == args.dir[0] and ("humaneval" not in run_dir and "mbpp" not in run_dir) and args.problem == 'code-gen':
    #     exit()

    # # Loads all the necessary data into an dict
    # if args.control_exp:
    #     data_obj = data_loader(args, run_dir, extern, "lr_all")
    # else:
    #     data_obj = data_loader(args, run_dir, extern, "lr")

    # # Splits the loaded data
    # split_obj = split(args.split, data_obj)

    # Control experiment to check which groups helps the model to make correct predictions
    if args.control_exp:
        y = data_provider.get_train_is_correct()
        X = data_provider.get_train_groups()
        print(X)
        print(data_provider.get_train_probs(args.prob_method).to_numpy())
        print(np.append(X, [data_provider.get_train_probs(args.prob_method)], ))
        exit()
    else:
        #y = data_provider.get_train_is_correct() - data_provider.get_train_probs(args.prob_method)
        #X = data_provider.get_train_groups()
        y = data_provider.get_train_is_correct()
        X = np.hstack([data_provider.get_train_probs(args.prob_method).values.reshape(-1,1), data_provider.get_train_groups()]) 
    
    if grid is None or chartmaker is None:
        # get the grid for binning type and the chartmaker obj        
        grid, chartmaker = binning.get_grid_and_chartmaker(data_provider.run, 
                                                        args, 
                                                        data_provider.save_dir, 
                                                        extern, 
                                                        probs=data_provider.get_train_probs(args.prob_method))

    # Train the linear regression on the train data split
    lr = LR_calibration(grid, OUTPUTS, DEBUG, type).fit(X, y)

    # Calculate scores on uncalibrated test set
    scores_uncalibrated = lr.score_obj.calc_all(data_provider.get_test_probs(args.prob_method), 
                                                    data_provider.get_test_is_correct(), 
                                                    groups=data_provider.get_test_groups(),
                                                    set_brier_ref=True)
    total_group_uncalibrated, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = lr.score_obj.get_total_and_correctness(data_provider.get_test_probs(args.prob_method), 
                                                                                                                                                            data_provider.get_test_is_correct(), 
                                                                                                                                                            data_provider.get_test_groups())
    
    # Make predictions
    predictions = lr.predict(np.hstack([data_provider.get_test_probs(args.prob_method).values.reshape(-1,1), data_provider.get_test_groups()]) )
        #data_provider.get_test_groups())

    # use predictions to calibrate
    if args.control_exp:
        calibrated_predictions = predictions
    else:
        calibrated_predictions = predictions #+ data_provider.get_test_probs(args.prob_method)

    # Calculate scores on uncalibrated test set
    scores_calibrated = lr.score_obj.calc_all(  calibrated_predictions, 
                                                    data_provider.get_test_is_correct(), 
                                                    groups=data_provider.get_test_groups())
    
    total_group_calibrated, correctness_group_calibrated, total_bin_calibrated, correctness_bin_calibrated = lr.score_obj.get_total_and_correctness(calibrated_predictions, 
                                                                                                                                                    data_provider.get_test_is_correct(), 
                                                                                                                                                    data_provider.get_test_groups()) 
    
    total_group, correctness_group, average_group_confidence = lr.score_obj.get_correctness_per_group(calibrated_predictions, 
                                                                                                      data_provider.get_test_is_correct(), 
                                                                                                      data_provider.get_test_groups()) 
    
    if OUTPUTS: print(f"Group weights: {lr.reg.coef_}")
    if OUTPUTS: print(f"bias: {lr.reg.intercept_}")

    # Add entry for the run in the score table
    lr.score_obj.add_to_score_table(data_provider.run, scores_uncalibrated, scores_calibrated)
    
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
        with open(data_provider.save_dir+'scores.txt', 'w') as f:
            f.write(lr.score_obj.printable_table)

if __name__ == "__main__":
    main()
