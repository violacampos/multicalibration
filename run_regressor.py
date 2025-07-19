import numpy as np
import os
from tools import binning, cmd_input
from tools.regressor_calibration import regressor_calibration
from tools.data import data_loader
from tools.split import split

DEBUG = False
OUTPUTS = True

np.seterr(divide='ignore', invalid='ignore')

def main(extern=False):
    # loads commandline parameter
    args = cmd_input.load_parser()

    # get run dir
    run_dirs = [x[0] for x in os.walk(args.dir[0])]
    run_dirs.sort()

    if args.all_lang == True:
        run_dirs = [run_dirs[0]]

    run_dir = run_dirs[0]

    if run_dir == args.dir[0] and ("humaneval" not in run_dir and "mbpp" not in run_dir) and args.problem == 'code-gen':
        exit()

    # Loads all the necessary data into an dict
    data_obj = data_loader(args, run_dir, extern, "regressor")

    # Necessary for loading the results of the evaluator LLM
    results, temperature, top_p, num_samples = data_obj.load_multipl_e_run(run_dir)

    verb_data_path_quantitativ = 'verbalized_data/quantitativ/'+data_obj.run+'/'+args.model+'/data.json'
    verb_data_quantitativ = data_obj.load_json_data(verb_data_path_quantitativ)
    probs_quantitativ, _, _, _, _, _, _ = data_obj.load_samples(results, verb_data_quantitativ, type='quantitativ')
    data_obj.data["probs_quantitativ"] = probs_quantitativ

    verb_data_path_qualitativ = 'verbalized_data/qualitativ/'+data_obj.run+'/'+args.model+'/data.json'
    verb_data_qualitativ = data_obj.load_json_data(verb_data_path_qualitativ)
    probs_qualitativ, _, _, _, _, _, _ = data_obj.load_samples(results, verb_data_qualitativ, type='qualitativ')
    data_obj.data["probs_qualitativ"] = probs_qualitativ
    

    # Splits the loaded data
    split_obj = split(args.split, data_obj)

    # Control experiment to check which groups helps the model to make correct predictions
    y = split_obj.train_data["is_correct"] - split_obj.train_data["probs"]

    # Create histograms for the token distribtion of each sample
    bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    histograms_train = np.array([np.histogram([np.round(np.exp(list(l.values())[0][0]),2) for l in sample], bins=bin_edges)[0] for sample in split_obj.train_data["token_logprobs"]])
    histograms_test = np.array([np.histogram([np.round(np.exp(list(l.values())[0][0]),2) for l in sample], bins=bin_edges)[0] for sample in split_obj.test_data["token_logprobs"]])

    # Stack input features into one array
    X = np.hstack((split_obj.train_groups, np.array(split_obj.train_data["probs"]).reshape(-1, 1), histograms_train, np.array(split_obj.train_data["probs_quantitativ"]).reshape(-1, 1),np.array(split_obj.train_data["probs_qualitativ"]).reshape(-1, 1)))
    X_test = np.hstack((split_obj.test_groups, np.array(split_obj.test_data["probs"]).reshape(-1, 1), histograms_test, np.array(split_obj.test_data["probs_quantitativ"]).reshape(-1, 1),np.array(split_obj.test_data["probs_qualitativ"]).reshape(-1, 1)))

    print(np.array(split_obj.test_groups).sum(axis=0))
    print(len(split_obj.test_data["probs"]))

    # get the grid for binning type and the chartmaker obj        
    grid, chartmaker = binning.get_grid_and_chartmaker(data_obj.run, 
                                                       args, 
                                                       data_obj.save_dir, 
                                                       extern, 
                                                       probs=split_obj.train_data["probs"])

    # Train the linear regression on the train data split
    reg = regressor_calibration(grid, args, OUTPUTS, DEBUG).fit(X,y)
    


    # Calculate scores on uncalibrated test set
    scores_uncalibrated = reg.score_obj.calc_all(split_obj.test_data["probs"], 
                                                    split_obj.test_data["is_correct"], 
                                                    groups=split_obj.test_groups,
                                                    set_brier_ref=True)
    
    total_group_uncalibrated, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = reg.score_obj.get_total_and_correctness(split_obj.test_data["probs"], 
                                                                                                                                                            split_obj.test_data["is_correct"], 
                                                                                                                                                            split_obj.test_groups)
    
    # Make predictions
    predictions = reg.predict(X_test)

    calibrated_predictions = predictions + split_obj.test_data["probs"]

    # Calculate scores on uncalibrated test set
    scores_calibrated = reg.score_obj.calc_all( calibrated_predictions, 
                                                split_obj.test_data["is_correct"], 
                                                groups=split_obj.test_groups)
    
    total_group_calibrated, correctness_group_calibrated, total_bin_calibrated, correctness_bin_calibrated = reg.score_obj.get_total_and_correctness(calibrated_predictions, 
                                                                                                                                                    split_obj.test_data["is_correct"], 
                                                                                                                                                    split_obj.test_groups) 
    
    total_group, correctness_group, average_group_confidence = reg.score_obj.get_correctness_per_group(calibrated_predictions, 
                                                                                                      split_obj.test_data["is_correct"], 
                                                                                                      split_obj.test_groups) 
    
    # Add entry for the run in the score table
    reg.score_obj.add_to_score_table(data_obj.run, scores_uncalibrated, scores_calibrated)
    
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
    reg.score_obj.display_score_table()

    # saves the score table
    if args.save_table:
        with open(data_obj.save_dir+'scores.txt', 'w') as f:
            f.write(reg.score_obj.printable_table)

if __name__ == "__main__":
    main()
