import numpy as np
import os
from sklearn.model_selection import train_test_split
from tools import data, binning, cmd_input
from tools.hb_calibration import hb_calibration
from tools.groups import groups

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts/"

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

    for d in run_dirs:
        # if main dir is in list just continue
        if d == args.dir[0] and ("humaneval" not in d and "mbpp" not in d):
            continue
        
        # Get run name
        run = d.split("/runs/", 1)[1]

        # create directory for chart generation
        if not os.path.isdir(CHART_DIR+run+'/hb/'+args.binning_type):
            os.makedirs(CHART_DIR+run+'/hb/'+args.binning_type)

        # load the data from the run directory
        run = run.replace('/', '')
        results, temperature, top_p, num_samples = data.load_multipl_e_run(d)

        verb_data = None
        if args.prob_method in ["verbalized_qual", "verbalized_quant"]:
            verb_data_path = 'verbalized_data/'+args.prob_method+'/'+run+'/verbalized_data.json'
            verb_data = data.load_json_data(verb_data_path)

        print(f"\nRun: {run}")
        save_dir = CHART_DIR+run+'/hb/'+args.binning_type+'/'
        if OUTPUTS: print(f"Temperature: {temperature}")
        if OUTPUTS: print(f"Num Problems: {num_samples}")

        # probs -> confidence of the model
        # is_correct -> label 1: is correct, 0: is not correct
        probs, is_correct, programs, prompts, languages, names, _ = data.proability_and_correctness_for_samples(results, verb_data, type=args.prob_method)
        correct_count = np.count_nonzero(is_correct == 1)

        if args.use_scc:
            scc_infos = data.load_scc_data(run, languages, names)
            groups_w = groups(programs, prompts).create_groups(scc=scc_infos)
        else:
            # Define group matrix
            groups_w = groups(programs, prompts).create_groups()

        if args.split:
            # split in 60% train, 20% validation and 20% test
            train_X, test_X, train_y, test_y, train_groups, test_groups, train_lang, test_lang = train_test_split(probs, is_correct, groups_w, languages, test_size=0.2, random_state=42)

            train_X, val_X, train_y, val_y, train_groups, val_groups, train_lang, val_lang = train_test_split(train_X, train_y, train_groups, train_lang, test_size=0.25, random_state=42)
        else:
            train_X = probs
            test_X = probs
            train_y = is_correct
            test_y = is_correct
            train_groups = groups_w
            test_groups = groups_w

        # get the grid for binning type and the chartmaker obj        
        grid, chartmaker = binning.get_grid_and_chartmaker(run, args.binning_type, save_dir, args.bin_count, train_X, 1/args.bin_count)
               
        # Create calibration object and calculates the deltas
        hb = hb_calibration(grid, OUTPUTS, DEBUG).fit(train_X, train_y)

        # calculate scores for the uncalibrated test set
        scores_uncalibrated = hb.score_obj.calc_all_new(test_X, 
                                                        test_y,
                                                        groups=test_groups, 
                                                        set_brier_ref=True)
        total_group_uncalibrated, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = hb.score_obj.get_total_and_correctness(test_X, test_y, test_groups)
        total_group_uncalib, correctness_group_uncalib, average_group_confidence_uncalib = hb.score_obj.get_correctness_per_group(test_X, test_y, test_groups)    

        # Uses the deltas to calculate the corrected values
        corrected_probs = hb.predict(test_X) 

        # calculate scores for the calibrated test set
        scores_calibrated = hb.score_obj.calc_all_new(  corrected_probs, 
                                                        test_y,
                                                        groups=test_groups)
        total_group_calibrated, correctness_group_calibrated, total_bin_calibrated, correctness_bin_calibrated = hb.score_obj.get_total_and_correctness(corrected_probs, test_y, test_groups) 
        
        total_group, correctness_group, average_group_confidence = hb.score_obj.get_correctness_per_group(corrected_probs, test_y, test_groups) 
                       
        # Add entry for the run in the score table
        hb.score_obj.add_to_score_table(run, scores_uncalibrated, scores_calibrated)
        
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
            chartmaker.calibration_info(total_bin_uncalibrated, correctness_bin_uncalibrated, total_bin_calibrated, correctness_bin_calibrated)
        
    # display score table for all runs
    hb.score_obj.display_score_table()

    # saves the score table
    if args.save_table:
        with open('results/'+run+'/hb_'+args.binning_type+'_results.txt', 'w') as f:
            f.write(hb.score_obj.printable_table)

if __name__ == "__main__":
    main()
