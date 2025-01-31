from tools.json_utils import read_json, read_jsonl
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.calibration import calibration_curve

logit_data = []

MODEL_NAME  = "Qwen2.5-Coder-7B-Instruct"
PARAMS      = "-temp-0"
DATASET     = "human-eval"
BASE_DIR    = "/data/stud/2025-MA-kuschnereit/masterarbeit/"

EVALUATED_SAMPLES_DIR = BASE_DIR+"evaluated_samples/"
GENERATED_SAMPLES_DIR = BASE_DIR+"generated_samples/"
CHART_DIR = BASE_DIR+"charts/"

def load_logits_for_task_id(task_id):
    entry = next(filter(lambda a : a['task_id'] == task_id, logit_data), None)
    return entry["logprobs"], len(entry["logprobs"]), entry["cumulative_logprob"]
 

def create_histogram(data, name, typ):
    fig, ax = plt.subplots()  
    ax.hist(data, range=(0, 1.0))
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    plt.title(MODEL_NAME+PARAMS+' '+typ)
    plt.savefig(name)
    print(f"Histogram saved: {name}")


def generate_calibration_curve(y, predicted_prob):
    prob_true, prob_pred = calibration_curve(y, predicted_prob, n_bins=5)
    
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    
    plt.title(MODEL_NAME+' Calibration Curve')
    plt.xlabel('Confidence')
    plt.ylabel('Correct')

    plt.savefig(CHART_DIR+'/'+DATASET+'/'+MODEL_NAME+"/calibration_curve.png")


def generate_calibration_bar_chart(y, x):
    plt.title(MODEL_NAME+PARAMS+' Reliability chart')
    plt.bar(y, x, width = 0.1)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Confidence')
    plt.ylabel('Correct')
    plt.savefig(CHART_DIR+'/'+DATASET+'/'+MODEL_NAME+"/calibration_bar_chart"+PARAMS+".png")

if __name__ == "__main__":
    eval_results_file_path = EVALUATED_SAMPLES_DIR+DATASET+"/samples-"+MODEL_NAME+PARAMS+"_eval_results.json"
    sample_logits_file_path = GENERATED_SAMPLES_DIR+DATASET+"/"+MODEL_NAME+"/details-"+MODEL_NAME+PARAMS+".jsonl"
    
    eval_data = read_json(eval_results_file_path)
    logit_data = read_jsonl(sample_logits_file_path)

    pass_log_value_list = []
    fail_log_value_list = []
    prob_value_list = []
    y = []

    print(f"Eval Data loaded from: {eval_results_file_path}")
    print(f"Logit Data loaded from: {sample_logits_file_path}")
    print(f"Evaluation from: {eval_data['date']}")

    for task, entry in eval_data['eval'].items():
        logprobs, token_count, cumulative_logprob = load_logits_for_task_id(task)

        avg_prob = np.round(np.exp(cumulative_logprob / token_count), 2) 
        """print(f"Average cumulative: {avg_prob}")
        joint_prob = []
        for logprob in logprobs:
            prob = np.exp(list(logprob.values())[0][0])
            joint_prob.append(prob)
        print(np.sum(np.array(joint_prob))/token_count)   
        exit()"""

        prob_value_list.append(avg_prob)
        if entry[0]['base_status'] == 'fail':
            fail_log_value_list.append(avg_prob)
            y.append(0)
        if entry[0]['base_status'] == 'pass':
            pass_log_value_list.append(avg_prob)
            y.append(1)

    pass_log_value_list = np.array(pass_log_value_list)
    fail_log_value_list = np.array(fail_log_value_list)

    # Brechnet Wahrscheinlichleit P(Korrekt| Score in bin 0-0.09, 0.1-0.19, ..., 0.9 )
    total_count = []
    positiv_count = []
    for bin in np.arange(0, 1, 0.1):
        bin_left = np.round(bin,1)
        bin_right = np.round(bin+0.1, 1)

        if bin_left == 0.9:
            positiv_count.append(np.count_nonzero((bin_left <= pass_log_value_list) & (pass_log_value_list <= bin_right)))
            total_count.append(np.count_nonzero((bin_left <= prob_value_list) & (prob_value_list <= bin_right)))
        else:
            positiv_count.append(np.count_nonzero((bin_left <= pass_log_value_list) & (pass_log_value_list < bin_right)))
            total_count.append(np.count_nonzero((bin_left <= prob_value_list) & (prob_value_list < bin_right)))

    # Wahrscheinlichkeit, dass Code korrekt ist in den jeweiligen bins
    P_correct = np.divide(np.array(positiv_count), np.array(total_count), where=np.array(total_count)!=0)
    
    generate_calibration_bar_chart(np.arange(0.05, 1, 0.1), P_correct)

    create_histogram(prob_value_list, CHART_DIR+'/'+DATASET+'/'+MODEL_NAME+'/Probabilities'+PARAMS+'.png', 'Total Probabilities')
    create_histogram(pass_log_value_list, CHART_DIR+'/'+DATASET+'/'+MODEL_NAME+'/Probabilities'+PARAMS+'_pass.png', 'Pass Probabilities')
    create_histogram(fail_log_value_list, CHART_DIR+'/'+DATASET+'/'+MODEL_NAME+'/Probabilities'+PARAMS+'_fail.png', 'Fail Probabilities')