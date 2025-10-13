import numpy as np
import os
from tools import binning, cmd_input
from tools.iglb_calibration import IGLB_calibration
import pickle
from tools.data import data_loader
from tools.split import split

DEBUG = True
OUTPUTS = True

np.seterr(divide='ignore', invalid='ignore')

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
    # data_obj = data_loader(args, run_dir, extern, "iglb")

    # # Splits the loaded data
    # split_obj = split(args.split, data_obj)
    
    # if OUTPUTS: print(f"Run: {data_provider.run}")
    # if OUTPUTS: print(f"Gruppen Anzahl: {data_provider.get_train_groups().sum(axis=0)}")
    
    if grid is None or chartmaker is None:
        # get the grid for binning type and the chartmaker obj        
        grid, chartmaker = binning.get_grid_and_chartmaker(data_provider.run, 
                                                       args, 
                                                       data_provider.save_dir,  
                                                       extern, 
                                                       probs=data_provider.get_train_probs(args.prob_method))

    # Create object and calculate first deltas and so on
    iglb = IGLB_calibration(grid, 
                            args.epsilon, 
                            args.bin_count, 
                            OUTPUTS, 
                            DEBUG).fit(data_provider.get_train_probs(args.prob_method), 
                                       data_provider.get_train_is_correct(), 
                                       data_provider.get_train_groups())
    
    scores_uncalibrated = iglb.score_obj.calc_all(  data_provider.get_test_probs(args.prob_method), 
                                                        data_provider.get_test_is_correct(), 
                                                        groups=data_provider.get_test_groups(), 
                                                        set_brier_ref=True)
    
    _, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = iglb.score_obj.get_total_and_correctness( data_provider.get_test_probs(args.prob_method), 
                                                                                                                                        data_provider.get_test_is_correct(), 
                                                                                                                                        data_provider.get_test_groups()) 
    
    temp_group_correctness = correctness_group_uncalibrated
    temp_bin_correctness = correctness_bin_uncalibrated

    train_probs = data_provider.get_train_probs(args.prob_method).to_numpy(copy=True)
    test_probs = data_provider.get_test_probs(args.prob_method).to_numpy(copy=True)
    val_probs = data_provider.get_val_probs(args.prob_method).to_numpy(copy=True)

    history = {}

    while True: 
        # Calculate mse for f_t
        mse_f_t = iglb.score_obj.mse(val_probs, 
                                     data_provider.get_val_is_correct(), 
                                     len(data_provider.get_val_is_correct()))

        # Assign bins an calculate the probability for each bin,group and tau combination
        # VIOLA: removed from predict iteration, only used for robins saved changes if test==True -> TODO check
        assigned_bins = binning.round_model_to_grid(train_probs, grid)   
        P_S_p_g = iglb.get_P_S_p_g(train_probs, data_provider.get_train_groups()) 
        
        # get the tau, bin, group for which the probality * deltas_squared maximises
        tau, bin, group = np.unravel_index((P_S_p_g*iglb.deltas_square).argmax(), iglb.deltas.shape)
        if iglb.debug: print(f"Max delta in: Tau {tau}, Bin {bin}, Group {group}")
        
        # First break if probability is smaller then alpha
        if P_S_p_g[tau, bin, group] < args.epsilon:
            break

        if DEBUG: print(f"Max Error: {P_S_p_g[tau, bin, group]}")
        # get the calibrated confidences for the calibration subset
        train_probs = iglb.predict(train_probs, 
                                                     data_provider.get_train_groups(), 
                                                     assigned_bins, 
                                                     tau, 
                                                     bin, 
                                                     group)

        # get the calibrated confidences for the test subset
        assigned_bins_test = binning.round_model_to_grid(test_probs, grid)   
        test_probs= iglb.predict(   test_probs, 
                                    data_provider.get_test_groups(), 
                                    assigned_bins_test, 
                                    tau, 
                                    bin, 
                                    group, 
                                    test=True, 
                                    is_correct=data_provider.get_test_is_correct())

        # get the calibrated confidences for the validation subset to calculate MSE
        assigned_bins_val = binning.round_model_to_grid(val_probs, grid)   
        val_probs = iglb.predict(   val_probs, 
                                    data_provider.get_val_groups(),
                                    assigned_bins_val, 
                                    tau, 
                                    bin, 
                                    group)
        
        curr_group_total, curr_group_correctness, _, curr_bin_correctness = iglb.score_obj.get_total_and_correctness(test_probs, 
                                                                                                                                  data_provider.get_test_is_correct(), 
                                                                                                                                  data_provider.get_test_groups())
        
        # Add history element to track changes
        history[len(iglb.changes)] = [temp_group_correctness, curr_group_correctness, iglb.changes[-1], curr_group_total, temp_bin_correctness, curr_bin_correctness] 
        temp_group_correctness = curr_group_correctness
        temp_bin_correctness = curr_bin_correctness    

        # Second Break if MSE of the new model is greater or equal to the model before
        mse_h_t_plus_1 = iglb.score_obj.mse(val_probs, 
                                            data_provider.get_val_is_correct(), 
                                            len(data_provider.get_val_is_correct()))
        
        if mse_h_t_plus_1 >= mse_f_t:
            if OUTPUTS: print(f"MSE h_t+1: {mse_h_t_plus_1} >= MSE f_t: {mse_f_t}")
            break
        
        # Set the new model for the next iteration
        iglb = iglb.fit(train_probs, data_provider.get_train_is_correct(), data_provider.get_train_groups())
            
    #if OUTPUTS: print(f"GASCE: {iglb.gasce}\n")
    scores_calibrated = iglb.score_obj.calc_all(test_probs, 
                                                    data_provider.get_test_is_correct(), 
                                                    groups=data_provider.get_test_groups())
    _, _, total_bin_calibrated, correctness_bin_calibrated = iglb.score_obj.get_total_and_correctness(  test_probs, 
                                                                                                        data_provider.get_test_is_correct(), 
                                                                                                        data_provider.get_test_groups()) 
    
    total_group, correctness_group, average_group_confidence = iglb.score_obj.get_correctness_per_group(test_probs, 
                                                                                                        data_provider.get_test_is_correct(), 
                                                                                                        data_provider.get_test_groups()) 
            
    # Add entry for the run in the score table
    iglb.score_obj.add_to_score_table(data_provider.run, scores_uncalibrated, scores_calibrated)
    history["score"] = iglb.score_obj.score_table

    if extern:
        return {"total_bin_calibrated":  total_bin_calibrated, 
                "correctness_bin_calibrated":  correctness_bin_calibrated, 
                "correctness_group":  correctness_group, 
                "average_group_confidence":  average_group_confidence,
                "total_group":  total_group, 
                "scores_calibrated":  scores_calibrated,
                "calibrated_probs": test_probs}
    else:
        if args.save_charts:
            chartmaker.calibration_info(total_bin_uncalibrated, correctness_bin_uncalibrated, total_bin_calibrated, correctness_bin_calibrated)

    if args.save_history:
        with open(data_provider.save_dir+'history_data/iglb_history.pkl', 'wb') as f:
            pickle.dump(history, f)

    # display score table for all runs
    iglb.score_obj.display_score_table()
    
    # saves the score table
    if args.save_table:
        with open(data_provider.save_dir+'scores.txt', 'w') as f:
            f.write(iglb.score_obj.printable_table)
            
if __name__ == "__main__":
    main()
