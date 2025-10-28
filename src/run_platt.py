import numpy as np
import os
from tools import binning, cmd_input

from tools.platt_calibration import Platt_calibration


DEBUG = False
OUTPUTS = True

#np.seterr(divide='ignore', invalid='ignore')

def main(data_provider, type='linear', extern=False, grid=None, chartmaker=None):
    args = cmd_input.load_parser()

    

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
        X = data_provider.get_train_probs(args.prob_method) 
    
    if grid is None or chartmaker is None:
        # get the grid for binning type and the chartmaker obj        
        grid, chartmaker = binning.get_grid_and_chartmaker(data_provider.run, 
                                                        args, 
                                                        data_provider.save_dir, 
                                                        extern, 
                                                        probs=data_provider.get_train_probs(args.prob_method))

    # Train the logisitc regression on the train data split
    platt = Platt_calibration(grid, args).fit(X, y)

    # Calculate scores on uncalibrated test set
    scores_uncalibrated = platt.score_obj.calc_all(data_provider.get_test_probs(args.prob_method), 
                                                    data_provider.get_test_is_correct(), 
                                                    groups=data_provider.get_test_groups(),
                                                    set_brier_ref=True)
    (total_group_uncalibrated, 
    correctness_group_uncalibrated, 
    total_bin_uncalibrated, 
    correctness_bin_uncalibrated) = platt.score_obj.get_total_and_correctness(data_provider.get_test_probs(args.prob_method), 
                                                                                                                                                            data_provider.get_test_is_correct(), 
                                                                                                                                                            data_provider.get_test_groups())
    
    # Make predictions
    predictions = platt.predict(data_provider.get_test_probs(args.prob_method)) 
        #data_provider.get_test_groups())

    # use predictions to calibrate
    if args.control_exp:
        calibrated_predictions = predictions
    else:
        calibrated_predictions = predictions #+ data_provider.get_test_probs(args.prob_method)

    # Calculate scores on uncalibrated test set
    scores_calibrated = platt.score_obj.calc_all(  calibrated_predictions, 
                                                    data_provider.get_test_is_correct(), 
                                                    groups=data_provider.get_test_groups())
    
    (total_group_calibrated, 
     correctness_group_calibrated, 
     total_bin_calibrated, 
     correctness_bin_calibrated) = platt.score_obj.get_total_and_correctness(calibrated_predictions, 
                                                                                                                                                    data_provider.get_test_is_correct(), 
                                                                                                                                                    data_provider.get_test_groups()) 
    
    total_group, correctness_group, average_group_confidence = platt.score_obj.get_correctness_per_group(calibrated_predictions, 
                                                                                                      data_provider.get_test_is_correct(), 
                                                                                                      data_provider.get_test_groups()) 
    
    if OUTPUTS: print(f"coef: {platt.platt.coef_}")
    if OUTPUTS: print(f"bias: {platt.platt.intercept_}")

    # Add entry for the run in the score table
    platt.score_obj.add_to_score_table(data_provider.run, scores_uncalibrated, scores_calibrated)
    
    # only return values when called from other script
    if extern:
        return {"total_bin_calibrated": total_bin_calibrated, 
                "correctness_bin_calibrated": correctness_bin_calibrated, 
                "correctness_group": correctness_group, 
                "average_group_confidence": average_group_confidence, 
                "total_group": total_group,
                "scores_calibrated": scores_calibrated,
                "calibrated_probs": calibrated_predictions,
                "group_names": data_provider.group_names}
    else:
        if args.save_charts:
            chartmaker.calibration_info(total_bin_uncalibrated, correctness_bin_uncalibrated, total_bin_calibrated, correctness_bin_calibrated)
    
    # display score table for all runs
    platt.score_obj.display_score_table()

    # saves the score table
    if args.save_table:
        with open(data_provider.save_dir+'scores.txt', 'w') as f:
            f.write(platt.score_obj.printable_table)

if __name__ == "__main__":
    main()
