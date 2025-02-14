import numpy as np
from pathlib import Path
import itertools
import argparse
import json
from pathlib import Path
import os
from tools import calibration_scores
from tools.np_encoder import NpEncoder
from tools import create_charts
from tools import data
from tools import binning

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts_multipl_e/"

charts = True
binning_type = 'linear' # other option is linear
binning_step_size = 0.1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dirs", type=str,  help="Directories with results. ", nargs="+")
    args = parser.parse_args()

    results_scores_dict = {}
    results_details_dict = {}

    run_dirs = [x[0] for x in os.walk(args.dirs[0])]
    run_dirs.sort()

    for d in run_dirs:
        # if main dir is in list just continue
        if d == args.dirs[0]:
            continue

        # Get run name
        run = d.split("/runs/", 1)[1]

        # create directory for chart generation
        if not os.path.isdir(CHART_DIR+run):
            os.makedirs(CHART_DIR+run)

        # load the data from the run directory
        run = run.replace('/', '')
        results, temperature, top_p, num_samples = data.load_multipl_e_run(d)

        print(f"\nRun: {run}")
        print(f"Temperature: {temperature}")
        print(f"Num samples: {num_samples}")

        probs, is_correct = data.proability_and_correctness_for_samples(results)
        
        correct_count = np.count_nonzero(is_correct == 1)
        print(f"Korrekt: {correct_count}")

        # Set binning range
        if binning_type == 'quantil':
            bin_ranges = [(np.quantile(probs, i) if (i != 0) and (i != 1) else i) for i in np.arange(0, 1+binning_step_size, binning_step_size)]   
        elif binning_type == 'linear':
            bin_ranges = np.arange(0, 1+binning_step_size, binning_step_size)

        total_per_bin, correct_per_bin, average_bin_confidence, chart_range, bar_width = binning.bin_probabilities(bin_ranges, probs, is_correct)
           
        fail_bin_count =  total_per_bin - correct_per_bin

        # Wahrscheinlichkeit, dass Code korrekt ist in den jeweiligen bins
        # corr(S_i)
        P_correct = np.divide(correct_per_bin, total_per_bin, where=total_per_bin!=0)

        # Expected Calibration Error
        ece = calibration_scores.ece(P_correct, average_bin_confidence, total_per_bin, num_samples)
        print(f"ECE: {ece}")

        # Brier Score reference 
        p_r, brier_score_ref = calibration_scores.brier_ref(correct_count, num_samples)
        print(f"Brier Score ref: {brier_score_ref}")

        # Brier Score actual
        brier_score_actual = calibration_scores.brier_actual(probs, is_correct, num_samples)
        print(f"Brier Score actual: {brier_score_actual}")

        # Skill Score
        skill_score = calibration_scores.skill_score(brier_score_ref, brier_score_actual)
        print(f"Skill Score: {skill_score}")

        results_scores_dict[run] = {
            "temperature": temperature,
            "top_p": top_p,
            "num samples": num_samples,
            "correct": correct_count,
            "ECE": ece,
            "p_r": p_r,
            "brier score ref": brier_score_ref,
            "brier score actual": brier_score_actual,
            "skill score": skill_score
        }

        results_details_dict[run] = {
            "total_bin_count": list(total_per_bin),
            "correct_bin": list(correct_per_bin),
            "fail_bin": list(fail_bin_count),
            "prob_list": list(probs),
            "P_correct": list(P_correct),
            "confidence": list(average_bin_confidence),
        }
        
        if charts == True:
            colors = []
            if binning_type == 'linear':
                total_bin_count_norm = (total_per_bin-np.min(total_per_bin))/(np.max(total_per_bin)-np.min(total_per_bin))
                for x in total_bin_count_norm:
                    colors.append((0.0, 0.0, 1.0, x))
            else:
                colors = ['tab:blue']
            create_charts.calibration_bar_chart(chart_range, P_correct, CHART_DIR+run+"/calibration_bar_chart_"+binning_type+".png", run, bar_width, colors)
            create_charts.stacked_bar_plot(chart_range, np.array(correct_per_bin), fail_bin_count, CHART_DIR+run+"/Probabilities_"+binning_type+".png", run, bar_width)

    result_scores_json = json.dumps(results_scores_dict, indent=4, cls=NpEncoder)
    result_details_json = json.dumps(results_details_dict, indent=4, cls=NpEncoder)
 
    with open("results/result_"+binning_type+"_scores.json", "w") as outfile:
        outfile.write(result_scores_json)
    
    with open("results/result_"+binning_type+"_details.json", "w") as outfile:
        outfile.write(result_details_json)

if __name__ == "__main__":
    main()
