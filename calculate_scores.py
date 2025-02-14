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

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts_multipl_e/"

charts = True
binning_type = 'quantil' # other option is linear

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

        pass_log_value_list = probs[is_correct == 1]
        correct_count = np.count_nonzero(is_correct == 1)
        
        print(f"Korrekt: {correct_count}")

        total_bin_count = []
        correct_bin_count = []
        average_bin_confidence = []
        chart_range = []
        if binning_type == 'linear':
            chart_range = np.arange(0.05, 1, 0.1)
            bar_width = 0.1
            # Brechnet Wahrscheinlichleit P(Korrekt| Score in bin 0-0.09, 0.1-0.19, ..., 0.9 )
            # conf(S_i)
            for bin in np.arange(0, 1, 0.1):
                bin_left = np.round(bin,1)
                bin_right = np.round(bin+0.1, 1)

                if bin_left == 0.9:
                    correct_bin_count.append(np.count_nonzero((bin_left <= pass_log_value_list) & (pass_log_value_list <= bin_right)))
                    bin_count   = np.count_nonzero((bin_left <= probs) & (probs <= bin_right))
                    bin_sum     = np.sum(probs, where=(bin_left <= probs) & (probs <= bin_right))
                    total_bin_count.append(bin_count)
                    average_bin_confidence.append(np.divide(bin_sum, bin_count, where=np.array(bin_count)!=0))
                else:
                    correct_bin_count.append(np.count_nonzero((bin_left <= pass_log_value_list) & (pass_log_value_list < bin_right)))
                    bin_count   = np.count_nonzero((bin_left <= probs) & (probs < bin_right))
                    bin_sum     = np.sum(probs, where=(bin_left <= probs) & (probs < bin_right))
                    total_bin_count.append(bin_count)
                    average_bin_confidence.append(np.divide(bin_sum, bin_count, where=np.array(bin_count)!=0))
        
        if binning_type == 'quantil':
            bin_ranges = [0, 
                          np.quantile(probs, 0.1),
                          np.quantile(probs, 0.2),
                          np.quantile(probs, 0.3),
                          np.quantile(probs, 0.4),
                          np.quantile(probs, 0.5),
                          np.quantile(probs, 0.6),
                          np.quantile(probs, 0.7), 
                          np.quantile(probs, 0.8),
                          np.quantile(probs, 0.9),
                          1]
            bar_width = []
            
            for bin_left, bin_right in zip(bin_ranges, bin_ranges[1:]):
                chart_range.append(((bin_right-bin_left)/2)+bin_left)
                bar_width.append(bin_right-bin_left)
                if bin_right == 1:
                    correct_bin_count.append(np.count_nonzero((bin_left <= pass_log_value_list) & (pass_log_value_list <= bin_right)))
                    bin_count   = np.count_nonzero((bin_left <= probs) & (probs <= bin_right))
                    bin_sum     = np.sum(probs, where=(bin_left <= probs) & (probs <= bin_right))
                    total_bin_count.append(bin_count)
                    average_bin_confidence.append(np.divide(bin_sum, bin_count, where=np.array(bin_count)!=0))
                else:
                    correct_bin_count.append(np.count_nonzero((bin_left <= pass_log_value_list) & (pass_log_value_list < bin_right)))
                    bin_count   = np.count_nonzero((bin_left <= probs) & (probs < bin_right))
                    bin_sum     = np.sum(probs, where=(bin_left <= probs) & (probs < bin_right))
                    total_bin_count.append(bin_count)
                    average_bin_confidence.append(np.divide(bin_sum, bin_count, where=np.array(bin_count)!=0))

           
        fail_bin_count =  np.array(total_bin_count) - np.array(correct_bin_count)

        # Wahrscheinlichkeit, dass Code korrekt ist in den jeweiligen bins
        # corr(S_i)
        P_correct = np.divide(np.array(correct_bin_count), np.array(total_bin_count), where=np.array(total_bin_count)!=0)

        # Expected Calibration Error
        ece = calibration_scores.ece(P_correct, average_bin_confidence, total_bin_count, num_samples)
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
            "total_bin_count": list(total_bin_count),
            "correct_bin": list(correct_bin_count),
            "fail_bin": list(fail_bin_count),
            "prob_list": list(probs),
            "P_correct": list(P_correct),
            "confidence": list(average_bin_confidence),
        }
        
        if charts == True:
            colors = []
            if binning_type == 'linear':
                total_bin_count_norm = (total_bin_count-np.min(total_bin_count))/(np.max(total_bin_count)-np.min(total_bin_count))
                for x in total_bin_count_norm:
                    colors.append((0.0, 0.0, 1.0, x))
            else:
                colors = ['tab:blue']
            create_charts.calibration_bar_chart(chart_range, P_correct, CHART_DIR+run+"/calibration_bar_chart_"+binning_type+".png", run, bar_width, colors)
            create_charts.stacked_bar_plot(chart_range, np.array(correct_bin_count), fail_bin_count, CHART_DIR+run+"/Probabilities_"+binning_type+".png", run, bar_width)

    result_scores_json = json.dumps(results_scores_dict, indent=4, cls=NpEncoder)
    result_details_json = json.dumps(results_details_dict, indent=4, cls=NpEncoder)
 
    with open("results/result_"+binning_type+"_scores.json", "w") as outfile:
        outfile.write(result_scores_json)
    
    with open("results/result_"+binning_type+"_details.json", "w") as outfile:
        outfile.write(result_details_json)

if __name__ == "__main__":
    main()
