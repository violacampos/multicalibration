import numpy as np
from tools import calibration_scores

np.seterr(divide='ignore', invalid='ignore')

score_grid = np.arange(0.0, 1+1, 1)

groups = np.array([[1, 0],
                   [1, 0],
                   [0, 1],
                   [0, 1]])

assigned_bins = np.array([1.0,
                          1.0,
                          1.0,
                          1.0])
labels = np.array([1,
                   0,
                   0,
                   0])


total_per_bin, correctness_per_bin, confidence_per_bin = calibration_scores.score.bin_round_probabilities_discret(assigned_bins, labels, score_grid)

print(f"Total per Bin:\n{total_per_bin}")
print(f"Correctness per Bin:\n{correctness_per_bin}")
print(f"Confidence per Bin:\n{confidence_per_bin}")

num_samples = len(assigned_bins)

asce = 0
delta = []
for corr_s_i, conf_s_i, bin_count in zip(correctness_per_bin, confidence_per_bin, total_per_bin):
    delta.append(corr_s_i-conf_s_i)
    asce += (bin_count/num_samples)*(corr_s_i-conf_s_i)**2

delta = np.array(delta)
print(f"\nDeltas:\n{delta}")
print(f"Deltas**2:\n{delta**2}")
print(f"\nASCE:\n{asce}")
print("-----------------------------------------")

#groups = np.ones(groups.shape)
# calculate the total correct per bin
correct_per_bin_group = np.array([[np.divide(len(assigned_bins[(assigned_bins == i) & (labels == 1) & (g ==1)]), len(assigned_bins[(assigned_bins == i) & (g ==1)])) for g in groups.T] for i in score_grid])
correct_per_bin_group[np.isnan(correct_per_bin_group)] = 0



# calculate the total count per bin
total_per_bin_group = np.array([[len(assigned_bins[(assigned_bins == i) & (g ==1)]) for g in groups.T]  for i in score_grid])
total_per_bin_group[np.isnan(total_per_bin_group)] = 0

# sum the probabilities per bin
bin_sums_group = np.array([[assigned_bins[(assigned_bins == i) & (g ==1)].sum() for g in groups.T]  for i in score_grid])

# calculate the average confidence per bin
average_bin_group_confidence = np.divide(bin_sums_group, total_per_bin_group, where=np.array(total_per_bin_group)!=0)

print(f"Total per Bin/Group:\n{total_per_bin_group}")
print(f"Correctness per Bin/Group:\n{correct_per_bin_group}")
print(f"Confidence per Bin/Group:\n{average_bin_group_confidence}")


gasce = np.zeros((2))
deltas = []
for corr_bin_group, conf_bin_group, bin_count in zip(correct_per_bin_group, average_bin_group_confidence, total_per_bin_group):
    deltas.append(corr_bin_group-conf_bin_group)
    gasce += ((bin_count/num_samples)*((corr_bin_group-conf_bin_group)**2))

deltas = np.array(deltas)
print(f"\nDeltas:\n{deltas}")
print(f"Deltas**2:\n{deltas**2}")
print(f"Deltas per Bin:")
print(deltas.sum(axis=1)/2)

gasce = np.array(gasce)
print(f"\nGASCE:\n{gasce}")