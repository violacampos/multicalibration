import numpy as np
import os
from tools import binning, cmd_input
from tools.iglb_calibration import IGLB_calibration
import pickle
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
    data_obj = data_loader(args, run_dir, extern, "iglb")

    # Splits the loaded data
    split_obj = split(args.split, data_obj)
    
    if OUTPUTS: print(f"Run: {data_obj.run}")
    if OUTPUTS: print(f"Gruppen Anzahl: {split_obj.train_groups.sum(axis=0)}")
    
    # get the grid for binning type and the chartmaker obj        
    grid, chartmaker = binning.get_grid_and_chartmaker(data_obj.run, 
                                                       args.binning_type, 
                                                       data_obj.save_dir, 
                                                       args.bin_count, 
                                                       extern, 
                                                       probs=split_obj.train_data["probs"], 
                                                       binning_step_size=1/args.bin_count)

    # Create object and calculate first deltas and so on
    iglb = IGLB_calibration(grid, args.epsilon, args.bin_count, OUTPUTS, DEBUG).fit(split_obj.train_data["probs"], 
                                                                                    split_obj.train_data["is_correct"], 
                                                                                    split_obj.train_groups)
    
    scores_uncalibrated = iglb.score_obj.calc_all_new(  split_obj.test_data["probs"], 
                                                        split_obj.test_data["is_correct"], 
                                                        groups=split_obj.test_groups, 
                                                        set_brier_ref=True)
    
    total_group_uncalibrated, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = iglb.score_obj.get_total_and_correctness(split_obj.test_data["probs"], 
                                                                                                                                                              split_obj.test_data["is_correct"], 
                                                                                                                                                              split_obj.test_groups) 
    
    temp_group_correctness = correctness_group_uncalibrated
    temp_bin_correctness = correctness_bin_uncalibrated

    train_probs = split_obj.train_data["probs"].to_numpy()
    test_probs = split_obj.test_data["probs"].to_numpy()
    val_probs = split_obj.val_data["probs"].to_numpy()

    history = {}

    while True: 
        # Calculate mse for f_t
        mse_f_t = iglb.score_obj.mse(val_probs, 
                                     split_obj.val_data["is_correct"], 
                                     len(split_obj.val_data["is_correct"]))

        # Assign bins an calculate the probability for each bin,group and tau combination
        assigned_bins = binning.round_model_to_grid(train_probs, grid)   
        P_S_p_g = iglb.get_P_S_p_g(assigned_bins, split_obj.train_groups) 
        
        # get the tau, bin, group for which the probality * deltas_squared maximises
        tau, bin, group = np.unravel_index((P_S_p_g*iglb.deltas_square).argmax(), iglb.deltas.shape)
        if iglb.debug: print(f"Max delta in: Tau {tau}, Bin {bin}, Group {group}")

        # First break if probability is smaller then alpha
        if P_S_p_g[tau, bin, group] < args.epsilon:
            break

        if DEBUG: print(f"Max Error: {P_S_p_g[tau, bin, group]}")
        # get the calibrated confidences for the calibration subset
        train_probs = iglb.predict(train_probs, 
                                                     split_obj.train_groups, 
                                                     assigned_bins, 
                                                     tau, 
                                                     bin, 
                                                     group)

        # get the calibrated confidences for the test subset
        assigned_bins_test = binning.round_model_to_grid(test_probs, grid)   
        test_probs= iglb.predict(   test_probs, 
                                    split_obj.test_groups, 
                                    assigned_bins_test, 
                                    tau, 
                                    bin, 
                                    group, 
                                    test=True, 
                                    is_correct=split_obj.test_data["is_correct"])

        # get the calibrated confidences for the validation subset to calculate MSE
        assigned_bins_val = binning.round_model_to_grid(val_probs, grid)   
        val_probs = iglb.predict(   val_probs, 
                                    split_obj.val_groups,
                                    assigned_bins_val, 
                                    tau, 
                                    bin, 
                                    group)
        
        curr_group_total, curr_group_correctness, curr_bin_total, curr_bin_correctness = iglb.score_obj.get_total_and_correctness(test_probs, 
                                                                                                                                  split_obj.test_data["is_correct"], 
                                                                                                                                  split_obj.test_groups)
        
        # Add history element to track changes
        history[len(iglb.changes)] = [temp_group_correctness, curr_group_correctness, iglb.changes[-1], curr_group_total, temp_bin_correctness, curr_bin_correctness] 
        temp_group_correctness = curr_group_correctness
        temp_bin_correctness = curr_bin_correctness    

        # Second Break if MSE of the new model is greater or equal to the model before
        mse_h_t_plus_1 = iglb.score_obj.mse(split_obj.val_data["probs"], 
                                            split_obj.val_data["is_correct"], 
                                            len(split_obj.val_data["is_correct"]))
        if mse_h_t_plus_1 >= mse_f_t:
            if OUTPUTS: print(f"MSE h_t+1: {mse_h_t_plus_1} >= MSE f_t: {mse_f_t}")
            break
        
        # Set the new model for the next iteration
        iglb = iglb.fit(train_probs, split_obj.train_data["is_correct"], split_obj.train_groups)
            
    #if OUTPUTS: print(f"GASCE: {iglb.gasce}\n")
    scores_calibrated = iglb.score_obj.calc_all_new(test_probs, 
                                                    split_obj.test_data["is_correct"], 
                                                    groups=split_obj.test_groups)
    total_group_calibrated, correctness_group_calibrated, total_bin_calibrated, correctness_bin_calibrated = iglb.score_obj.get_total_and_correctness(test_probs, 
                                                                                                                                                      split_obj.test_data["is_correct"], 
                                                                                                                                                      split_obj.test_groups) 
    
    total_group, correctness_group, average_group_confidence = iglb.score_obj.get_correctness_per_group(test_probs, 
                                                                                                        split_obj.test_data["is_correct"], 
                                                                                                        split_obj.test_groups) 
            
    # Add entry for the run in the score table
    iglb.score_obj.add_to_score_table(data_obj.run, scores_uncalibrated, scores_calibrated)
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
        with open(data_obj.save_dir+'history_data/iglb_history.pkl', 'wb') as f:
            pickle.dump(history, f)

    # display score table for all runs
    iglb.score_obj.display_score_table()
    
    # saves the score table
    if args.save_table:
        with open(data_obj.save_dir+'scores.txt', 'w') as f:
            f.write(iglb.score_obj.printable_table)
if __name__ == "__main__":
    main()
