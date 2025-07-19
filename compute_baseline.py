import numpy as np
import os

from tools import  binning, cmd_input
from tools.data import data_loader
from tools.split import split
from tools.calibration_scores import score
import matplotlib.pyplot as plt

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

    # if main dir is in list just continue
    if run_dir == args.dir[0] and ("humaneval" not in run_dir and "mbpp" not in run_dir) and args.problem == 'code-gen':
        exit()
    
    # Loads all the necessary data into an dict
    data_obj = data_loader(args, run_dir, extern, "baseline")

    # Splits the loaded data
    split_obj = split(args.split, data_obj)

    # get group matrix
    groups = np.array([np.array(xi) for xi in data_obj.data["groups"].values])

    print(f"Group count: {groups.sum(axis=0)}")

    # get the grid for binning type and the chartmaker obj        
    grid, chartmaker = binning.get_grid_and_chartmaker(data_obj.run, 
                                                       args, 
                                                       data_obj.save_dir, 
                                                       extern, 
                                                       probs=split_obj.train_data["probs"])
            
    score_obj = score(grid, OUTPUTS, DEBUG)

    # calculate scores for the uncalibrated test set
    scores_uncalibrated = score_obj.calc_all(   split_obj.test_data["probs"], 
                                                    split_obj.test_data["is_correct"],
                                                    groups=split_obj.test_groups, 
                                                    set_brier_ref=True)

    _, _, total_bin_uncalibrated, correctness_bin_uncalibrated = score_obj.get_total_and_correctness(split_obj.test_data["probs"], 
                                                                                                     split_obj.test_data["is_correct"], 
                                                                                                     split_obj.test_groups)

    total_group_uncalib, correctness_group_uncalib, average_group_confidence_uncalib = score_obj.get_correctness_per_group(split_obj.test_data["probs"], 
                                                                                                                           split_obj.test_data["is_correct"], 
                                                                                                                           split_obj.test_groups)    

    # Add entry for the run in the score table
    score_obj.add_to_score_table(data_obj.run, scores_uncalibrated, [], baseline=True)
    
    # only return values when script is called from another script
    if extern:
        return {"correctness_bin_uncalibrated": correctness_bin_uncalibrated, 
                "total_bin_uncalibrated": total_bin_uncalibrated,
                "correctness_group_uncalib": correctness_group_uncalib, 
                "average_group_confidence_uncalib": average_group_confidence_uncalib,
                "total_group_uncalib": total_group_uncalib,
                "scores_uncalibrated": scores_uncalibrated,
                "uncalibrated_probs": split_obj.test_data["probs"],
                "is_correct": split_obj.test_data["is_correct"],
                "groups": split_obj.test_groups,
                "language": split_obj.test_data["languages"],
                "names": split_obj.test_data["names"],
                "programs": split_obj.test_data["programs"],
                "prompts": split_obj.test_data["prompts"],
                "token_logprobs": split_obj.test_data["token_logprobs"]}
    else:
        if args.save_charts:
            _, axs = plt.subplots(1, 1, figsize=(6, 5))
            
            chartmaker.calibration_bar_chart(axs, 
                                             'Uncalibrated', 
                                             correctness_bin_uncalibrated, 
                                             chartmaker.get_bar_colors(total_bin_uncalibrated), 
                                             total_bin_uncalibrated)        
            
            plt.savefig(chartmaker.save_dir+"calibration.pdf")
            plt.close() 
        
    # display score table for all runs
    score_obj.display_score_table()

    # saves the score table
    if args.save_table:
        with open(data_obj.save_dir+'scores.txt', 'w') as f:
            f.write(score_obj.printable_table)



if __name__ == "__main__":
    main()
