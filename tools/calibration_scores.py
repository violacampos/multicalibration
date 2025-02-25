import numpy as np
import math
"""
    Calculates the expected Calibration error. Weighted average of the deviation from the 
    fraction of predictions that are correct and the average estimated probability.

    correctness_per_bin: Probality for the correctnes per bin
    confidence_per_bin: Average confidence of the model per bin
    total_bin_count: Count of samples per bin
    num_samples: Total count of samples in the dataset
"""
def ece(correctness_per_bin, confidence_per_bin, total_per_bin, num_samples):
    ece = 0
    for corr_s_i, conf_s_i, s_i_count in zip(correctness_per_bin, confidence_per_bin, total_per_bin):
        ece += ((abs(s_i_count)/abs(num_samples))*abs(corr_s_i-conf_s_i))

    return np.round(ece, 2)

def asce(correctness_per_bin, confidence_per_bin, total_bin_count, num_samples):
    asce = 0
    for corr_s_i, conf_s_i, bin_count in zip(correctness_per_bin, confidence_per_bin, total_bin_count):
        asce += (bin_count/num_samples)*(corr_s_i-conf_s_i)**2
    return np.round(asce, 2)

def asce_deltas(total_bin_count, num_samples, delta_p_f):
    asce = 0
    for bin_count, delta in zip( total_bin_count, delta_p_f):
        asce += (bin_count/num_samples)*(delta)**2
    return np.round(asce, 2)

"""
    Calculates the baseline score of the uncalibrated model where every prediction is in one bin.
    p_r is the average correctness in this bin.

    correct_sample_count: Correct samples in the dataset
    num_samples: Total count of samples in the dataset
"""
def brier_ref(correct_sample_count, num_samples):
    p_r = correct_sample_count / num_samples
    return np.round(p_r, 2) ,np.round(p_r * (1-p_r), 2)

"""
    Caculates the actual brier score for the given data.
    Also known as the MSE

    confidences: List of all prediction probailities
    label: label if the given sample is correct
    num_problems: Total count of samples in the dataset
"""
def mse(confidences, labels, num_problems):
    brier_score_actual = 0
    for conf, label in zip(confidences, labels):
        brier_score_actual += (label - conf)**2
    return np.round((1/num_problems)*brier_score_actual, 2)

"""
    Caculates the skill score. Perfect score is 1.0. Negativ mean worse than the baseline. Small positiv values indicate good skill

    brier_ref: Brier baseline score
    brier_actual: Actual brier score of the dataset
"""
def skill_score(brier_ref, brier_actual):
    return np.round((brier_ref-brier_actual)/brier_ref, 2)


def expeceted_variance(probs, label, bin_assignement, grid, num_samples):
    expec_var = 0.0        
    for i in grid:
        bin_probs = probs[bin_assignement == i]
        bin_labels = label[bin_assignement == i]
        if len(bin_probs) == 0:
            continue

        E = np.mean(bin_probs)
        if math.isnan(E): E = 0

        variance = E * (1 - E)**2

        weight = len(bin_labels) / num_samples
        expec_var += weight * variance

    return np.round(expec_var, 2)

