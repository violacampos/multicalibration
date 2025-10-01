import numpy as np
import os
from tools import binning, cmd_input
from tools.ighb_calibration import IGHB_calibration
import pickle 
from sklearn.model_selection import KFold
from tools.data import data_loader
from tools.split import split

DEBUG = False
OUTPUTS = True

np.seterr(divide='ignore', invalid='ignore')

def main(extern=False):
    # loads commandline parameter
    args = cmd_input.load_parser()

    m = args.bin_count
    
    # get run dir
    run_dirs = [x[0] for x in os.walk(args.dir[0])]
    run_dirs.sort()

    run_dir = run_dirs[0]

    if run_dir == args.dir[0] and ("humaneval" not in run_dir and "mbpp" not in run_dir) and args.problem == 'code-gen':
        exit()

    # Loads all the necessary data into an dict
    data_obj = data_loader(args, run_dir, extern, "ighb")

    # Splits the loaded data
    split_obj = split(args.split, data_obj)
        
    if OUTPUTS: print(f"Run: {data_obj.run}")
    if OUTPUTS: print(f"Gruppen Anzahl: {split_obj.train_groups.sum(axis=0)}")

    # Possible k_fold
    if args.k_fold:
        if args.split:
            exit("Can't use split while using K-Fold!")

        kf = KFold(n_splits=5)
        kf.get_n_splits(split_obj.train_data["probs"])

        history = {}
        for i, (train_index, test_index) in enumerate(kf.split(split_obj.train_data["probs"])):
            print(f"Fold {i}:")

            train_X = split_obj.train_data["probs"][train_index]
            test_X = split_obj.train_data["probs"][test_index]
            train_y = split_obj.train_data["is_correct"][train_index]
            test_y = split_obj.train_data["is_correct"][test_index]
            train_groups = split_obj.train_groups[train_index]
            test_groups = split_obj.train_groups[test_index]

            # get the grid for binning type and the chartmaker obj        
            grid, chartmaker = binning.get_grid_and_chartmaker(data_obj.run, 
                                                               args, 
                                                               data_obj.save_dir, 
                                                               extern, 
                                                               probs=train_X)

            # Fit calibrator
            ighb = IGHB_calibration(grid, m, 1/args.bin_count, OUTPUTS, DEBUG).fit(train_X, train_y, train_groups)

            # Calculate values for uncalibrated test set
            scores_uncalibrated = ighb.score_obj.calc_all(  test_X, 
                                                            test_y, 
                                                            groups=test_groups,
                                                            set_brier_ref=True)
            
            _, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = ighb.score_obj.get_total_and_correctness(test_X, test_y, test_groups) 

            temp_group_correctness = correctness_group_uncalibrated
            temp_bin_correctness = correctness_bin_uncalibrated

            # set the conf to calibrate on
            calibrated_conf = train_X
            history_item = {}
            while ighb.max_error > ighb.alpha:  
                if DEBUG: print(f"Max Error: {ighb.max_error}")
                # get new better calibrated confidences
                calibrated_conf = ighb.predict(calibrated_conf, train_groups, is_correct=train_y)
                
                # calculate the corrected values for the test set
                test_X = ighb.predict(test_X, test_groups, test=True, is_correct=test_y)

                # Calculate some metrics on the UNcorrected values
                curr_group_total, curr_group_correctness, _, curr_bin_correctness = ighb.score_obj.get_total_and_correctness(test_X, test_y, test_groups)
                history_item[len(ighb.changes)] = [temp_group_correctness, curr_group_correctness, ighb.changes[-1], curr_group_total, temp_bin_correctness, curr_bin_correctness]
                temp_group_correctness = curr_group_correctness
                temp_bin_correctness = curr_bin_correctness
                
                # fit the model on the corrected confidences
                ighb = ighb.fit(calibrated_conf, train_y, train_groups)
                    

            # Calculate values for calibrated test set
            scores_calibrated = ighb.score_obj.calc_all(test_X, 
                                                        test_y, 
                                                        groups=test_groups)
            
            _, _, total_bin_calibrated, correctness_bin_calibrated = ighb.score_obj.get_total_and_correctness(test_X, test_y, test_groups) 

            # Add entry for the run in the score table
            ighb.score_obj.add_to_score_table(data_obj.run, scores_uncalibrated, scores_calibrated)
            ighb.score_obj.display_score_table()
            
            history_item["score"] = ighb.score_obj.score_table
            history[i] = history_item
            ighb = None
        if args.save_history:
            with open(data_obj.save_dir+'history_data/ighb_history_kfold.pkl', 'wb') as f:
                pickle.dump(history, f)
    else:
        
        # get the grid for binning type and the chartmaker obj        
        grid, chartmaker = binning.get_grid_and_chartmaker(data_obj.run, 
                                                           args, 
                                                           data_obj.save_dir, 
                                                           extern, 
                                                           probs=split_obj.train_data["probs"])
    
        ighb = IGHB_calibration(grid, m, 1/m, OUTPUTS, DEBUG).fit( split_obj.train_data["probs"], 
                                                                                split_obj.train_data["is_correct"], 
                                                                                split_obj.train_groups)

        # Calculate values for uncalibrated test set
        scores_uncalibrated = ighb.score_obj.calc_all(  split_obj.test_data["probs"], 
                                                            split_obj.test_data["is_correct"], 
                                                            groups=split_obj.test_groups, 
                                                            set_brier_ref=True)
        
        _, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = ighb.score_obj.get_total_and_correctness(split_obj.test_data["probs"], 
                                                                                                                                                                  split_obj.test_data["is_correct"],
                                                                                                                                                                  split_obj.test_groups) 

        temp_group_correctness = correctness_group_uncalibrated
        temp_bin_correctness = correctness_bin_uncalibrated

        # set the conf to calibrate on
        calibrated_conf = split_obj.train_data["probs"]
        uncalibrated_conf = split_obj.test_data["probs"]
        history = {}
        while ighb.max_error > ighb.alpha:  
            #print(f"Max Error: {ighb.max_error}")
            # get new calibrated confidences
            if args.split:
                calibrated_conf = ighb.predict(calibrated_conf, 
                                               split_obj.train_groups, 
                                               is_correct=split_obj.train_data["is_correct"])
            else:
                calibrated_conf = ighb.predict(calibrated_conf, 
                                               split_obj.train_groups, 
                                               test=True, 
                                               is_correct=split_obj.train_data["is_correct"])
            
            # calculate the corrected values for the test set VIOLA check this!!
            if args.split:
                split_obj.test_data["probs"] = ighb.predict(split_obj.test_data["probs"], 
                                                            split_obj.test_groups, 
                                                            test=True, 
                                                            is_correct=split_obj.test_data["is_correct"])
            else:
                split_obj.test_data["probs"] = calibrated_conf

            # Calculate some metrics on the values
            curr_group_total, curr_group_correctness, _, curr_bin_correctness = ighb.score_obj.get_total_and_correctness(split_obj.test_data["probs"], 
                                                                                                                                      split_obj.test_data["is_correct"], 
                                                                                                                                      split_obj.test_groups)
            # Storing history
            history[len(ighb.changes)] = [temp_group_correctness, curr_group_correctness, ighb.changes[-1], curr_group_total, temp_bin_correctness, curr_bin_correctness] #chartmaker.map_correctness_to_eleven_bins(
            temp_group_correctness = curr_group_correctness
            temp_bin_correctness = curr_bin_correctness
            
            # fit the model on the corrected confidences
            ighb = ighb.fit(calibrated_conf, 
                            split_obj.train_data["is_correct"], 
                            split_obj.train_groups)
                

        # Calculate values for calibrated test set
        scores_calibrated = ighb.score_obj.calc_all(split_obj.test_data["probs"], 
                                                        split_obj.test_data["is_correct"], 
                                                        groups=split_obj.test_groups)
        
        _, _, total_bin_calibrated, correctness_bin_calibrated = ighb.score_obj.get_total_and_correctness(  split_obj.test_data["probs"], 
                                                                                                            split_obj.test_data["is_correct"], 
                                                                                                            split_obj.test_groups) 
        
        total_group, correctness_group, average_group_confidence = ighb.score_obj.get_correctness_per_group(split_obj.test_data["probs"], 
                                                                                                            split_obj.test_data["is_correct"], 
                                                                                                            split_obj.test_groups) 
    
        # Add entry for the run in the score table
        ighb.score_obj.add_to_score_table(data_obj.run, scores_uncalibrated, scores_calibrated)

        history["score"] = ighb.score_obj.score_table

        if extern:
            return {"total_bin_calibrated": total_bin_calibrated, 
                    "correctness_bin_calibrated": correctness_bin_calibrated, 
                    "correctness_group": correctness_group, 
                    "average_group_confidence": average_group_confidence, 
                    "total_group": total_group,
                    "scores_calibrated": scores_calibrated,
                    "calibrated_probs": split_obj.test_data["probs"]}
        else:
            if args.save_charts:
                chartmaker.calibration_info(total_bin_uncalibrated, correctness_bin_uncalibrated, total_bin_calibrated, correctness_bin_calibrated)
        if args.save_history:
            with open(data_obj.save_dir+'history_data/ighb_history.pkl', 'wb') as f:
                pickle.dump(history, f)

        # display score table for all runs
        ighb.score_obj.display_score_table()
        
        # saves the score table
        if args.save_table:
            with open(data_obj.save_dir+'scores.txt', 'w') as f:
                f.write(ighb.score_obj.printable_table)
                
        

if __name__ == "__main__":
    main()
