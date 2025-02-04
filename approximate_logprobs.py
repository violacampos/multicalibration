import numpy as np

def prob_sample_to_recover(prob_sample, temp, top_p=1, hidden_vocab_num=None):
    prob_sample = prob_sample * top_p
    top_logprobs = len(prob_sample)

    if hidden_vocab_num is None:
        hidden_vocab_num = 1 if (1 - prob_sample.sum() > 0.001) else 0

    if hidden_vocab_num > 0:
        hidden_probs = [max(1 - prob_sample.sum(), 0) / hidden_vocab_num] * hidden_vocab_num
        prob_sample = np.concatenate([prob_sample, hidden_probs])

    prob_recover_unnorm = prob_sample ** temp
    prob_recover = prob_recover_unnorm / prob_recover_unnorm.sum()

    return prob_recover[:top_logprobs]

# Test case

# Sampling parameters
temp = 0.5
top_p = 0.9
top_logprobs = 3

# Original logits
logits = np.array([1, 0, -1, -2,])

# Compute prob_sample using softmax with temperature temp
logits_temp = logits / temp
exp_logits_temp = np.exp(logits_temp)
prob_sample = exp_logits_temp / exp_logits_temp.sum()



prob_sample /= top_p
prob_sample = prob_sample[:top_logprobs]  # Same to vllm's e**logprob
print(prob_sample)
prob_recover = prob_sample_to_recover(prob_sample, temp, top_p)

print(" prob_sample:", prob_sample)
print("prob_recover:", prob_recover)

# Ground truth probabilities
prob_gt = np.exp(logits) / np.exp(logits).sum()
print("     prob_gt:", prob_gt)