import numpy as np

"""
    Calculates the expected Calibration error. Weighted average of the deviation from the 
    fraction of predictions that are correct and the average estimated probability.

    P_correct: Probality for the correctnes per bin
    average_bin_confidence: Average confidence of the model per bin
    total_bin_count: Count of samples per bin
    num_samples: Total count of samples in the dataset
"""
def ece(P_correct, average_bin_confidence, total_bin_count, num_samples):
    ece = 0
    for corr_s_i, conf_s_i, s_i_count in zip(P_correct, average_bin_confidence, total_bin_count):
        ece += ((abs(s_i_count)/abs(num_samples))*abs(corr_s_i-conf_s_i))

    return np.round(ece, 2)

"""
    Calculates the baseline score of the uncalibrated model where every prediction is in one bin.
    p_r is the average correctness in this bin.

    correct_sample_count: Correct samples in the dataset
    num_samples: Total count of samples in the dataset
"""
def brier_ref(correct_sample_count, num_samples):
    p_r = correct_sample_count / num_samples
    return p_r, p_r * (1-p_r)

"""
    Caculates the actual brier score for the given data.
    Also known as the MSE

    prediction_prob_list: List of all prediction probailities
    is_correct: label if the given sample is correct
    num_problems: Total count of samples in the dataset
"""
def brier_actual(prediction_prob_list, is_correct, num_problems):
    brier_score_actual = 0
    for predicted, result in zip(prediction_prob_list, is_correct):
        brier_score_actual += (predicted - result)**2
    return np.round((1/num_problems)*brier_score_actual, 2)

"""
    Caculates the skill score. Perfect score is 1.0. Negativ mean worse than the baseline. Small positiv values indicate good skill

    brier_ref: Brier baseline score
    brier_actual: Actual brier score of the dataset
"""
def skill_score(brier_ref, brier_actual):
    return(brier_ref-brier_actual)/brier_ref

