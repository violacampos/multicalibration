import numpy as np
import math
from termcolor import colored

from tabulate import tabulate

from tools import binning

class score:
    def __init__(self, grid, outputs, debug):
        self.debug = debug
        self.outputs = outputs
        self.grid = grid
        self.score_grid = np.arange(0.0, 1+(1/10), 1/10)
        self.p_r = 0
        self.brier_ref_score = 0

        self.score_table = []
        self.printable_table = []

    """
        Calculates the expected Calibration error. Weighted average of the deviation from the 
        fraction of predictions that are correct and the average estimated probability.

        correctness_per_bin: Probality for the correctnes per bin
        confidence_per_bin: Average confidence of the model per bin
        total_bin_count: Count of samples per bin
        num_samples: Total count of samples in the dataset
    """
    def ece(self, correctness_per_bin, confidence_per_bin, total_per_bin, num_samples):
        ece = 0
        for corr_s_i, conf_s_i, s_i_count in zip(correctness_per_bin, confidence_per_bin, total_per_bin):
            ece += ((abs(s_i_count)/abs(num_samples))*abs(corr_s_i-conf_s_i))

        return np.round(ece, 2)

    def asce(self, correctness_per_bin, confidence_per_bin, total_bin_count, num_samples):
        asce = 0
        for corr_s_i, conf_s_i, bin_count in zip(correctness_per_bin, confidence_per_bin, total_bin_count):
            asce += (bin_count/num_samples)*(corr_s_i-conf_s_i)**2
            #
        return np.round(asce ,4)

    def asce_deltas(self, total_bin_count, num_samples, delta_p_f):
        asce = 0
        for bin_count, delta in zip(total_bin_count, delta_p_f):
            asce += (bin_count/num_samples)*(delta)**2
        return np.round(np.mean((delta_p_f)**2), 2)

    """
        Calculates the baseline score of the uncalibrated model where every prediction is in one bin.
        p_r is the average correctness in this bin.

        correct_sample_count: Correct samples in the dataset
        num_samples: Total count of samples in the dataset
    """
    def brier_ref(self, correct_sample_count, num_samples):
        p_r = correct_sample_count / num_samples
        return np.round(p_r, 2) ,np.round(p_r * (1-p_r), 2)

    """
        Caculates the actual brier score for the given data.
        Also known as the MSE

        confidences: List of all prediction probailities
        label: label if the given sample is correct
        num_problems: Total count of samples in the dataset
    """
    def mse(self, confidences, labels, num_problems):
        brier_score_actual = 0
        for conf, label in zip(confidences, labels):
            brier_score_actual += (label - conf)**2
        return np.round((1/num_problems)*brier_score_actual, 2)

    """
        Caculates the skill score. Perfect score is 1.0. Negativ mean worse than the baseline. Small positiv values indicate good skill

        brier_ref: Brier baseline score
        brier_actual: Actual brier score of the dataset
    """
    def skill_score(self, brier_ref, brier_actual):
        return np.round((brier_ref-brier_actual)/brier_ref, 2)

    def gcu(self, label, confidence, groups):
        gcu = np.round(np.array([np.mean(label[(col == 1)] -  confidence[(col == 1)]) for col in groups.T]), 2)
        gcu[np.isnan(gcu)] = 0
        return gcu

    def expeceted_variance(self, probs, label, bin_assignement, grid, num_samples):
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

    """def gasce(self, deltas):
        if deltas.shape[0] == 2 and len(deltas.shape) == 3:
            gasce = np.round(np.mean(np.mean(deltas**2, axis=1), axis=0), 4)
        elif len(deltas.shape) == 2:
            gasce = np.round(np.array(np.mean(deltas**2, axis=0)), 4)
        else:
            gasce = np.round(np.mean(deltas**2), 4)
        return gasce"""
    
    def gasce(self, assigned_bins, labels, groups, grid=None):
        if grid is None: grid = self.score_grid
        num_samples = len(assigned_bins)
        
        # calculate the total correct per bin
        correct_per_bin_group = np.array([[np.divide(len(assigned_bins[(assigned_bins == i) & (labels == 1) & (g ==1)]), len(assigned_bins[(assigned_bins == i) & (g ==1)])) for g in groups.T] for i in grid])
        correct_per_bin_group[np.isnan(correct_per_bin_group)] = 0

        # calculate the total count per bin
        total_per_bin_group = np.array([[len(assigned_bins[(assigned_bins == i) & (g ==1)]) for g in groups.T]  for i in grid])
        total_per_bin_group[np.isnan(total_per_bin_group)] = 0
        
        # sum the probabilities per bin
        bin_sums_group = np.array([[assigned_bins[(assigned_bins == i) & (g ==1)].sum() for g in groups.T]  for i in grid])

        # calculate the average confidence per bin
        average_bin_group_confidence = np.divide(bin_sums_group, total_per_bin_group, where=np.array(total_per_bin_group)!=0)
        
        gasce = 0
        deltas = []
        for corr_bin_group, conf_bin_group, bin_group_count in zip(correct_per_bin_group, average_bin_group_confidence, total_per_bin_group):
            #gasce += ((bin_count/total_per_bin_group.sum())*((corr_bin_group-conf_bin_group)**2))
            deltas.append((corr_bin_group-conf_bin_group))
            gasce += ((bin_group_count/num_samples)*((corr_bin_group-conf_bin_group)**2))

        return np.array(gasce)
    
    def calc_all_new(   self,
                        confidences,
                        labels,
                        groups=None,
                        deltas=None, # deviation per bin
                        set_brier_ref=False  # to differentiate between fit and predict
                    ):
        
        if set_brier_ref:
            prefix = 'Uncalib'
            color = 'red'
        else:
            prefix = 'Calib'
            color = 'green'

        # Assign the values in X to the corresponding bin (discretize values)
        assigned_bins = binning.round_model_to_grid(confidences, self.score_grid)
        if self.debug: print(f"TEST Assigned Bins: {assigned_bins}")

        # Calculate some metrics on the UNcorrected values
        total_per_bin, correctness_per_bin, confidence_per_bin = binning.bin_round_probabilities_discret(assigned_bins, labels, self.score_grid)

        num_samples = len(assigned_bins)
        num_correct = np.count_nonzero(labels == 1)
        
        # Get Expected Calibration Error
        ece = self.ece(correctness_per_bin, confidence_per_bin, total_per_bin, num_samples)

        # Get Mean Squared Error
        mse = self.mse(assigned_bins, labels, num_samples)

        # Get Average Squared Error
        asce = self.asce(correctness_per_bin, confidence_per_bin, total_per_bin, num_samples)

        # Get expected Variance 
        expeceted_variance = self.expeceted_variance(assigned_bins, labels, assigned_bins, self.score_grid, num_samples)

        # Output results if set
        if self.outputs: 
            print(f"{colored(prefix, color)} ECE: {ece}")
            print(f"{colored(prefix, color)} MSE: {mse}")
            print(f"{colored(prefix, color)} ASCE: {asce}") 
            print(f"{colored(prefix, color)} Expected Variance: {expeceted_variance}") 
        
        # Get Group conditional unbiasednes
        if groups is not None:
            gcu = self.gcu(labels, confidences, groups)
            if self.outputs: print(f"{colored(prefix, color)} GCU: {gcu}")

        # Set the reference Score for the Skill Score calculation
        if set_brier_ref:
            self.p_r, self.brier_ref_score = self.brier_ref(num_correct, num_samples)
            if self.outputs: 
                print(f"{colored(prefix, color)} Baseline: {self.p_r}")
                print(f"{colored(prefix, color)} Brier ref: {self.brier_ref_score}")

        # Calculate the Skill score
        skill_score = self.skill_score(self.brier_ref_score, mse)
        if self.outputs: print(f"{colored(prefix, color)} Skill Score: {skill_score}") 
        
        # create dict for better overview
        results =   {
                        prefix: {
                            "ECE": ece,
                            "ASCE": asce,
                            "MSE": mse,
                            "Brier ref": self.brier_ref_score,
                            "Skill Score": skill_score
                        } 
                    }  
                      
        gasce = self.gasce(assigned_bins, labels, groups)
        if self.outputs: print(f"{colored(prefix, color)} GASCE: {np.round(gasce, 4)}") 
        results[prefix]['GASCE'] = np.round(gasce, 4)

        return results
    
    def get_total_and_correctness(self, confidences, labels, groups):
        # Assign the values in X to the corresponding bin (discretize values)
        assigned_bins = binning.round_model_to_grid(confidences, self.score_grid)

        # Calculate some metrics on the UNcorrected values
        correctness_group = np.array([[np.divide(len(labels[(assigned_bins == i) & (labels == 1) & (g ==1)]), len(labels[(assigned_bins == i) & (g ==1)])) for g in groups.T] for i in self.score_grid])
        correctness_group[np.isnan(correctness_group)] = 0

        total_group = np.array([[len(assigned_bins[(assigned_bins == i) & (g ==1)]) for g in groups.T]  for i in self.score_grid])
        total_group[np.isnan(total_group)] = 0

        correctness_bin = np.array([np.divide(len(labels[(assigned_bins == i) & (labels == 1)]), len(labels[(assigned_bins == i)])) for i in self.score_grid])
        correctness_bin[np.isnan(correctness_bin)] = 0

        total_bin = np.array([len(assigned_bins[(assigned_bins == i)]) for i in self.score_grid])
        total_bin[np.isnan(total_bin)] = 0

        return total_group, correctness_group, total_bin, correctness_bin

    def get_total_per_group(self, confidences, labels, groups):
        # Assign the values in X to the corresponding bin (discretize values)
        assigned_bins = binning.round_model_to_grid(confidences, self.grid)

        # calculate the total count per bin
        total_group = np.array([[len(confidences[(assigned_bins == i) & (g == 1)]) for g in groups.T]  for i in self.grid])
        total_group[np.isnan(total_group)] = 0

        return total_group

    
    def add_to_score_table(self, run, uncalib_scores, calib_scores):
        
        uncalib_scores = list(list(uncalib_scores.values())[0].values())
        uncalib_gasce = uncalib_scores[-1]
        uncalib_scores = uncalib_scores[:-1]

        calib_scores = list(list(calib_scores.values())[0].values())
        calib_gasce = calib_scores[-1]
        calib_scores = calib_scores[:-1]
        
        score_difference = list(np.round(np.array(calib_scores) - np.array(uncalib_scores), 4))

        gasce_diff = np.round(np.array(calib_gasce) - np.array(uncalib_gasce), 4)

        score_difference.append(gasce_diff)
        uncalib_scores.append(uncalib_gasce)
        calib_scores.append(calib_gasce)

        self.add_entry(run, "Uncalib", uncalib_scores)
        self.add_entry(run, "Calib", calib_scores)
        self.add_entry(run, "Diff", score_difference)

    def add_entry(self, run, type, scores):
        entry = []
        
        if type == 'Uncalib':
            entry.append(run)
        else:
            entry.append('')
        entry.append(type)
        for s in scores:
            entry.append(s)

        self.score_table.append(entry)

    def display_score_table(self):
        self.printable_table = tabulate(self.score_table, headers=[ 'Run', 
                                                                    'Type',
                                                                    'ECE', 
                                                                    'ASCE', 
                                                                    'MSE',
                                                                    'brier_ref', 
                                                                    'skill_score',
                                                                    'GASCE'], tablefmt='orgtbl')
        print(self.printable_table)
