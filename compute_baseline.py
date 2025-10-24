import numpy as np
import os

from tools import  binning, cmd_input
from tools.calibration_scores import score
import matplotlib.pyplot as plt

DEBUG = False
OUTPUTS = True
BIGCODEBENCH = True

np.seterr(divide='ignore', invalid='ignore')

def main(data_provider, extern=False, grid=None, chartmaker=None):

    
    # loads commandline parameter
    args = cmd_input.load_parser()
    

    if grid is None or chartmaker is None:
        # get the grid for binning type and the chartmaker obj
        grid, chartmaker = binning.get_grid_and_chartmaker(data_provider.run,
                                                           args,
                                                           data_provider.save_dir,
                                                           extern, 
                                                           probs=data_provider.get_train_probs(args.prob_method))
   
            
    score_obj = score(grid, OUTPUTS, DEBUG)

    # calculate scores for the uncalibrated test set
    scores_uncalibrated = score_obj.calc_all(data_provider.get_test_probs(args.prob_method),
                                                    data_provider.get_test_is_correct(),
                                                    groups=data_provider.get_test_groups(),
                                                    set_brier_ref=True)

    _, _, total_bin_uncalibrated, correctness_bin_uncalibrated = score_obj.get_total_and_correctness(data_provider.get_test_probs(args.prob_method),
                                                                                                     data_provider.get_test_is_correct(),
                                                                                                     data_provider.get_test_groups())

    total_group_uncalib, correctness_group_uncalib, average_group_confidence_uncalib = score_obj.get_correctness_per_group(data_provider.get_test_probs(args.prob_method),
                                                                                                                           data_provider.get_test_is_correct(),
                                                                                                                           data_provider.get_test_groups())

    # Add entry for the run in the score table
    score_obj.add_to_score_table(data_provider.run, scores_uncalibrated, [], baseline=True)
    
    # only return values when script is called from another script
    if extern:
        return {"correctness_bin_uncalibrated": correctness_bin_uncalibrated, 
                "total_bin_uncalibrated": total_bin_uncalibrated,
                "correctness_group_uncalib": correctness_group_uncalib, 
                "average_group_confidence_uncalib": average_group_confidence_uncalib,
                "total_group_uncalib": total_group_uncalib,
                "scores_uncalibrated": scores_uncalibrated,
                "uncalibrated_probs": data_provider.get_test_probs(args.prob_method),
                "is_correct": data_provider.get_test_is_correct(),
                "groups": data_provider.get_test_groups(),
                "language": data_provider.get_test_languages(),
                "names": data_provider.get_test_names(),
                "programs": data_provider.get_test_programs(),
                "prompts": data_provider.get_test_prompts(),
                "token_logprobs": data_provider.get_test_token_logprobs(),
                "group_names": data_provider.group_names}
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
        with open(data_provider.save_dir+'scores.txt', 'w') as f:
            f.write(score_obj.printable_table)



if __name__ == "__main__":
    main()
