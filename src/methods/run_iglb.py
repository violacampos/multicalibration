import numpy as np
import os

from tools import binning, cmd_input
from methods.iglb_calibration import IGLB_calibration
from tools.create_charts import CalibrationCharts



def main(data_provider, extern=False, bins=None, plots=None):
    # loads commandline parameter
    args = cmd_input.load_parser()

    
    
    if bins is None:
        bins = binning.Binning(args.bin_count, args.binning_type)

    train_probs = data_provider.get_train_probs(args.prob_method)
    train_is_correct = data_provider.get_train_is_correct()
    train_groups = data_provider.get_train_groups()

    

    # Create object and calculate first deltas and so on
    iglb = IGLB_calibration(bins, args).fit(train_probs, 
                                       train_is_correct, 
                                       train_groups)
    
    # compute scores on test data 
    test_probs = data_provider.get_test_probs(args.prob_method)
    test_is_correct = data_provider.get_test_is_correct()
    test_groups = data_provider.get_test_groups()
    
    scores_uncalibrated = iglb.score_obj.calc_all(  test_probs, 
                                                        test_is_correct, 
                                                        groups=test_groups, 
                                                        set_brier_ref=True)
    
    _, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = iglb.score_obj.get_total_and_correctness( test_probs, 
                                                                                                                                        test_is_correct, 
                                                                                                                                        test_groups) 
    
    
    train_probs = train_probs.copy()
    test_probs = test_probs.copy()
    val_probs = data_provider.get_val_probs(args.prob_method).copy()


    while True: 
        # Calculate mse for f_t
        mse_f_t = iglb.score_obj.mse(val_probs, 
                                     data_provider.get_val_is_correct(), 
                                     len(data_provider.get_val_is_correct()))

        # Assign bins and calculate the probability for each bin,group and tau combination
        assigned_bins = bins.round_probabilities_to_grid(train_probs)   
        P_S_p_g = iglb.get_P_S_p_g(train_probs, train_groups) 
        
        # get the tau, bin, group for which the probality * deltas_squared maximises
        tau, bin, group = np.unravel_index((P_S_p_g * iglb.deltas_square).argmax(), iglb.deltas.shape)
        if iglb.debug: 
            print(f"Max delta in: Tau {tau}, Bin {bin}, Group {group}")
        
        # Break if probability is smaller then epsilon
        if P_S_p_g[tau, bin, group] < iglb.epsilon:
            break

        if iglb.debug: 
            print(f"Max Error: {P_S_p_g[tau, bin, group]}")
        # get calibrated confidences for the calibration subset
        train_probs = iglb.predict(train_probs, 
                                                     train_groups, 
                                                     assigned_bins, 
                                                     tau, 
                                                     bin, 
                                                     group)

        # get the calibrated confidences for the test subset
        assigned_bins_test = bins.round_probabilities_to_grid(test_probs)   
        test_probs= iglb.predict(   test_probs, 
                                    test_groups, 
                                    assigned_bins_test, 
                                    tau, 
                                    bin, 
                                    group, 
                                    test=True, 
                                    is_correct=test_is_correct)

        # get the calibrated confidences for the validation subset to calculate MSE
        assigned_bins_val = bins.round_probabilities_to_grid(val_probs)   
        val_probs = iglb.predict(   val_probs, 
                                    data_provider.get_val_groups(),
                                    assigned_bins_val, 
                                    tau, 
                                    bin, 
                                    group)
        
        curr_group_total, curr_group_correctness, _, curr_bin_correctness = iglb.score_obj.get_total_and_correctness(test_probs, 
                                                                                                                                  test_is_correct, 
                                                                                                                                  test_groups)
        
        # Break if MSE starts rising
        mse_h_t_plus_1 = iglb.score_obj.mse(val_probs, 
                                            data_provider.get_val_is_correct(), 
                                            len(data_provider.get_val_is_correct()))
        
        if mse_h_t_plus_1 >= mse_f_t:
            if iglb.debug: 
                print(f"MSE h_t+1: {mse_h_t_plus_1} >= MSE f_t: {mse_f_t}")
            break
        
        # Set the new model for the next iteration
        iglb = iglb.fit(train_probs, train_is_correct, train_groups)
            

    scores_calibrated = iglb.score_obj.calc_all(test_probs, 
                                                    test_is_correct, 
                                                    groups=test_groups)
    _, _, total_bin_calibrated, correctness_bin_calibrated = iglb.score_obj.get_total_and_correctness(  test_probs, 
                                                                                                        test_is_correct, 
                                                                                                        test_groups) 
    
    total_group, correctness_group, average_group_confidence = iglb.score_obj.get_correctness_per_group(test_probs, 
                                                                                                        test_is_correct, 
                                                                                                        test_groups) 
            
    # Add entry for the run in the score table
    iglb.score_obj.add_to_score_table(data_provider.run, scores_uncalibrated, scores_calibrated)


    if extern:
        return {"total_bin_calibrated":  total_bin_calibrated, 
                "correctness_bin_calibrated":  correctness_bin_calibrated, 
                "correctness_group":  correctness_group, 
                "average_group_confidence":  average_group_confidence,
                "total_group":  total_group, 
                "scores_calibrated":  scores_calibrated,
                "calibrated_probs": test_probs,
                "group_names": data_provider.group_names}
    else:
        if args.save_charts:
            if plots is None:
                plots = CalibrationCharts(
                    data_provider.run, args.binning_type, bins.grid, data_provider.save_dir
                )
            plots.calibration_info(total_bin_uncalibrated, correctness_bin_uncalibrated, total_bin_calibrated, correctness_bin_calibrated)


    # display score table for all runs
    iglb.score_obj.display_score_table()
    
    # saves the score table
    if args.save_table:
        with open(data_provider.save_dir+'scores.txt', 'w') as f:
            f.write(iglb.score_obj.printable_table)
            

