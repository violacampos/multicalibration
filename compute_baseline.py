import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from tools import data, binning, cmd_input
from tools.groups import groups
from tools.calibration_scores import score
import matplotlib.pyplot as plt

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts/"

DEBUG = False
OUTPUTS = True

np.seterr(divide='ignore', invalid='ignore')

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
        if not os.path.isdir(CHART_DIR+run+'/baseline/'+args.binning_type+'/'+args.prob_method):
            os.makedirs(CHART_DIR+run+'/baseline/'+args.binning_type+'/'+args.prob_method)

        # load the data from the run directory
        run = run.replace('/', '')
        results, temperature, top_p, num_samples = data.load_multipl_e_run(d)

        verb_data = None
        if args.prob_method in ["verbalized_qual", "verbalized_quant"]:
            verb_data_path = 'verbalized_data/'+args.prob_method+'/'+run+'/verbalized_data.json'
            verb_data = data.load_json_data(verb_data_path)

               
        print(f"\nRun: {run}")
        save_dir = CHART_DIR+run+'/baseline/'+args.binning_type+'/'+args.prob_method+'/'
        if OUTPUTS: print(f"Temperature: {temperature}")
        if OUTPUTS: print(f"Num Problems: {num_samples}")

        # probs -> confidence of the model
        # is_correct -> label 1: is correct, 0: is not correct
        probs, is_correct, programs, prompts, languages, names, token_logprobs = data.proability_and_correctness_for_samples(results, verb_data, type=args.prob_method)
        correct_count = np.count_nonzero(is_correct == 1)

        if args.use_scc:
            scc_infos = data.load_scc_data(run, languages, names)
            groups_w = groups(programs, prompts).create_groups(scc=scc_infos)
        else:
            # Define group matrix
            groups_w = groups(programs, prompts).create_groups()

        if args.split:
            # split in 60% train, 20% validation and 20% test
            train_X, test_X, train_y, test_y, train_groups, test_groups, train_lang, test_lang, train_names, test_names, train_prompts, test_prompts = train_test_split(probs, is_correct, groups_w, languages, names, prompts, test_size=0.2, random_state=42)

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
               
        score_obj = score(grid, OUTPUTS, DEBUG)

        # calculate scores for the uncalibrated test set
        scores_uncalibrated = score_obj.calc_all_new(test_X, 
                                                        test_y,
                                                        groups=test_groups, 
                                                        set_brier_ref=True)
        
        total_group_uncalibrated, correctness_group_uncalibrated, total_bin_uncalibrated, correctness_bin_uncalibrated = score_obj.get_total_and_correctness(test_X, test_y, test_groups)
        
        total_group_uncalib, correctness_group_uncalib, average_group_confidence_uncalib = score_obj.get_correctness_per_group(test_X, test_y, test_groups)    

        total_lang_uncalib, correctness_lang_uncalib, average_lang_confidence_uncalib = score_obj.get_correctness_per_language(probs, is_correct, languages)       

        # Add entry for the run in the score table
        score_obj.add_to_score_table(run, scores_uncalibrated, [], baseline=True)
        
        # only return values when script is called from another script
        if extern:
            return {"correctness_bin_uncalibrated": correctness_bin_uncalibrated, 
                    "total_bin_uncalibrated": total_bin_uncalibrated,
                    "correctness_group_uncalib": correctness_group_uncalib, 
                    "average_group_confidence_uncalib": average_group_confidence_uncalib,
                    "total_group_uncalib": total_group_uncalib,
                    "scores_uncalibrated": scores_uncalibrated,
                    "uncalibrated_probs": test_X,
                    "is_correct": test_y,
                    "language": test_lang,
                    "names": test_names,
                    "prompts": test_prompts,
                    "token_logprobs": token_logprobs}
        else:
            colors_uncalibrated = []
            total_bin_count_norm = (total_bin_uncalibrated-np.min(total_bin_uncalibrated))/(np.max(total_bin_uncalibrated)-np.min(total_bin_uncalibrated))
            for x in total_bin_count_norm:
                colors_uncalibrated.append((0.0, 0.0, 1.0, x))
            fig, axs = plt.subplots(1, 1, figsize=(7, 5))
            chartmaker.calibration_bar_chart(axs, 'Baseline reliability', correctness_bin_uncalibrated, colors_uncalibrated, total_bin_uncalibrated)
            plt.savefig(chartmaker.save_dir+"calibration_"+'no_split' if not args.split else ''+".png")
            plt.close() 
        
    # display score table for all runs
    score_obj.display_score_table()

    # saves the score table
    if args.save_table:
        if not os.path.isdir('results/'+run):
            os.makedirs('results/'+run)
        with open('results/'+run+'/baseline_'+args.binning_type+'_'+args.prob_method+'_'+('no_split' if not args.split else '')+'_results.txt', 'w') as f:
            f.write(score_obj.printable_table)



if __name__ == "__main__":
    main()
