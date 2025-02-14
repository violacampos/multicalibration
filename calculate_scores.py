import numpy as np
from pathlib import Path
import itertools
import argparse
import json
from pathlib import Path
import gzip
from typing import Optional
import sys
import matplotlib.pyplot as plt
import os

BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"
CHART_DIR = BASE_DIR+"charts_multipl_e/"

charts = True
binning_type = 'quantil' # other option is linear

def create_histogram(data, path, run, typ):
    fig, ax = plt.subplots()  
    ax.hist(data, range=(0, 1.0))
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    plt.title(run+' # '+typ, fontsize=7)
    plt.savefig(path)
    print(f"Histogram saved: {path}")
    plt.close()

def generate_calibration_bar_chart(y, x, path, run, width, bar_colors):
    plt.title(run+' # Reliability chart', fontsize=7)
    plt.bar(y, x, width = width, color=bar_colors, edgecolor='black')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Confidence')
    plt.ylabel('Correct')
    plt.savefig(path)
    plt.close()

def create_stacked_bar_plot(x, y1, y2, path, run, width):
    plt.bar(x, y1, color='g', width = width, edgecolor='black')
    plt.bar(x, y2, bottom=y1, color='r', width = width, edgecolor='black')
    plt.xlabel("Confidence")
    plt.ylabel("Anzahl")
    plt.legend(["Pass", "Fail"])
    plt.title(run, fontsize=7)
    plt.savefig(path)
    plt.close()

"""def prob_sample_to_recover(prob_sample, temp, top_p=1, hidden_vocab_num=None):
    prob_sample = prob_sample * top_p
    top_logprobs = len(prob_sample)

    if hidden_vocab_num is None:
        hidden_vocab_num = 1 if (1 - prob_sample.sum() > 0.001) else 0

    if hidden_vocab_num > 0:
        hidden_probs = [max(1 - prob_sample.sum(), 0) / hidden_vocab_num] * hidden_vocab_num
        prob_sample = np.concatenate([prob_sample, hidden_probs])

    prob_recover_unnorm = prob_sample ** temp
    prob_recover = prob_recover_unnorm / prob_recover_unnorm.sum()

    return prob_recover[:top_logprobs]"""

def gunzip_json(path: Path) -> Optional[dict]:
    """
    Reads a .json.gz file, but produces None if any error occurs.
    """
    try:
        with gzip.open(path, "rt") as f:
            return json.load(f)
    except Exception as e:
        return None

def gzip_json(path: Path, data: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f)

def for_file(path: Path):
    if path.suffix == ".gz":
        data = gunzip_json(path)
    else:
        with open(path, 'r') as f:
            data = json.load(f)

    if data is None:
        return None
    n = len(data["results"])
    token_infos         = data["tokens_info"][0]
    cumulative_logprob  = token_infos["cumulative_logprob"]
    token_logprobs      = token_infos["token_logprobs"]
    token_ids           = token_infos["len"]
    c = len([True for r in data["results"] if r["status"]
            == "OK" and r["exit_code"] == 0])
    return {
        "name": data["name"], 
        "n": n,
        "c": c,
        "temperature": data["temperature"] if "temperature" in data else 0.2,
        "top_p": data["top_p"],
        "cumulative_logprob": cumulative_logprob,
        "token_ids": token_ids,
        "token_logprobs": token_logprobs
    }


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
        if d == args.dirs[0]:
            continue
        run = d.split("/runs/", 1)[1]
        if not os.path.isdir(CHART_DIR+run):
            os.makedirs(CHART_DIR+run)
        run = run.replace('/', '')
        results = [for_file(p) for p in itertools.chain(
            Path(d).glob("*.results.json"), Path(d).glob("*.results.json.gz"))]
        results = [r for r in results if r is not None]
        temperatures = set(r["temperature"] for r in results)
        top_p = set(r["top_p"] for r in results)

        temperature = list(temperatures)[0]
        top_p = list(top_p)[0]
        num_problems = len(results)
        print(f"\nRun: {run}")
        print(f"Temperature: {temperature}")
        print(f"Num Problems: {num_problems}")

        pass_log_value_list = []
        fail_log_value_list = []
        prob_value_list = []
        is_correct = []

        for r in results:
            token_count = len(r["token_ids"])

            #corrected_logprob = []

            #for prob in r['token_logprobs']:
                #print(prob)
                #logprob = prob_sample_to_recover(np.array([np.exp(list(prob.values())[0][0])]), temperature, top_p)[0]
                #corrected_logprob.append(logprob)
                #print(f"Uncorrected: {np.exp(list(prob.values())[0][0])} -> Corrected: {prob_sample_to_recover(np.array([np.exp(list(prob.values())[0][0])]), temperature, top_p)[0]}")
            
            cumulative_logprob = r["cumulative_logprob"]

            #avg_prob = np.sum(np.array(corrected_logprob))/token_count
            # avg_prob = prob_sample_to_recover(np.array([np.exp(cumulative_logprob / token_count)]), temperature, top_p)[0]

            # Can be used but may be influenced by bug https://github.com/vllm-project/vllm/issues/9453
            avg_prob = np.round(np.exp(cumulative_logprob / token_count), 2) 

            prob_value_list.append(avg_prob)
            if r["c"] == 0:
                fail_log_value_list.append(avg_prob)
                is_correct.append(0)
            if r["c"] == 1:
                pass_log_value_list.append(avg_prob)
                is_correct.append(1)

        prob_value_list     = np.array(prob_value_list)
        pass_log_value_list = np.array(pass_log_value_list)
        fail_log_value_list = np.array(fail_log_value_list)
        is_correct          = np.array(is_correct)
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
                    bin_count   = np.count_nonzero((bin_left <= prob_value_list) & (prob_value_list <= bin_right))
                    bin_sum     = np.sum(prob_value_list, where=(bin_left <= prob_value_list) & (prob_value_list <= bin_right))
                    total_bin_count.append(bin_count)
                    average_bin_confidence.append(np.divide(bin_sum, bin_count, where=np.array(bin_count)!=0))
                else:
                    correct_bin_count.append(np.count_nonzero((bin_left <= pass_log_value_list) & (pass_log_value_list < bin_right)))
                    bin_count   = np.count_nonzero((bin_left <= prob_value_list) & (prob_value_list < bin_right))
                    bin_sum     = np.sum(prob_value_list, where=(bin_left <= prob_value_list) & (prob_value_list < bin_right))
                    total_bin_count.append(bin_count)
                    average_bin_confidence.append(np.divide(bin_sum, bin_count, where=np.array(bin_count)!=0))
        
        if binning_type == 'quantil':
            bin_ranges = [0, 
                          np.quantile(prob_value_list, 0.1),
                          np.quantile(prob_value_list, 0.2),
                          np.quantile(prob_value_list, 0.3),
                          np.quantile(prob_value_list, 0.4),
                          np.quantile(prob_value_list, 0.5),
                          np.quantile(prob_value_list, 0.6),
                          np.quantile(prob_value_list, 0.7), 
                          np.quantile(prob_value_list, 0.8),
                          np.quantile(prob_value_list, 0.9),
                          1]
            bar_width = []
            
            for bin_left, bin_right in zip(bin_ranges, bin_ranges[1:]):
                chart_range.append(((bin_right-bin_left)/2)+bin_left)
                bar_width.append(bin_right-bin_left)
                if bin_right == 1:
                    correct_bin_count.append(np.count_nonzero((bin_left <= pass_log_value_list) & (pass_log_value_list <= bin_right)))
                    bin_count   = np.count_nonzero((bin_left <= prob_value_list) & (prob_value_list <= bin_right))
                    bin_sum     = np.sum(prob_value_list, where=(bin_left <= prob_value_list) & (prob_value_list <= bin_right))
                    total_bin_count.append(bin_count)
                    average_bin_confidence.append(np.divide(bin_sum, bin_count, where=np.array(bin_count)!=0))
                else:
                    correct_bin_count.append(np.count_nonzero((bin_left <= pass_log_value_list) & (pass_log_value_list < bin_right)))
                    bin_count   = np.count_nonzero((bin_left <= prob_value_list) & (prob_value_list < bin_right))
                    bin_sum     = np.sum(prob_value_list, where=(bin_left <= prob_value_list) & (prob_value_list < bin_right))
                    total_bin_count.append(bin_count)
                    average_bin_confidence.append(np.divide(bin_sum, bin_count, where=np.array(bin_count)!=0))

           
        fail_bin_count =  np.array(total_bin_count) - np.array(correct_bin_count)

        # Wahrscheinlichkeit, dass Code korrekt ist in den jeweiligen bins
        # corr(S_i)
        P_correct = np.divide(np.array(correct_bin_count), np.array(total_bin_count), where=np.array(total_bin_count)!=0)

        # Expected Calibration Error
        ece = 0
        for corr_s_i, conf_s_i, s_i_count in zip(P_correct, average_bin_confidence, total_bin_count):
            ece += ((abs(s_i_count)/abs(num_problems))*abs(corr_s_i-conf_s_i))

        ece = np.round(ece, 2)
        print(f"ECE: {ece}")

        # Brier Score reference 
        p_r = correct_count / num_problems
        brier_score_ref = p_r * (1-p_r)
        print(f"Brier Score ref: {brier_score_ref}")

        # Brier Score actual
        brier_score_actual = 0
        for predicted, result in zip(prob_value_list, is_correct):
            brier_score_actual += (predicted - result)**2

        brier_score_actual = np.round((1/num_problems)*brier_score_actual, 2)
        print(f"Brier Score actual: {brier_score_actual}")

        # Skill Score
        skill_score = (brier_score_ref-brier_score_actual)/brier_score_ref
        print(f"Skill Score: {skill_score}")

        results_scores_dict[run] = {
            "temperature": temperature,
            "top_p": top_p,
            "num problems": num_problems,
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
            "prob_list": list(prob_value_list),
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
            generate_calibration_bar_chart(chart_range, P_correct, CHART_DIR+run+"/calibration_bar_chart_"+binning_type+".png", run, bar_width, colors)
            create_stacked_bar_plot(chart_range, np.array(correct_bin_count), fail_bin_count, CHART_DIR+run+"/Probabilities_"+binning_type+".png", run, bar_width)

    result_scores_json = json.dumps(results_scores_dict, indent=4)
    result_details_json = json.dumps(results_details_dict, indent=4)
 
    with open("result_"+binning_type+"_scores.json", "w") as outfile:
        outfile.write(result_scores_json)
    
    with open("result_"+binning_type+"_details.json", "w") as outfile:
        outfile.write(result_details_json)

if __name__ == "__main__":
    main()
