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
        return np.round(asce, 2)

    def asce_deltas(self, total_bin_count, num_samples, delta_p_f):
        asce = 0
        for bin_count, delta in zip(total_bin_count, delta_p_f):
            asce += (bin_count/num_samples)*(delta)**2
        return np.round(asce, 2)

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

    def gasce(self, deltas):
        if deltas.shape[0] == 2 and len(deltas.shape) == 3:
            gasce = np.mean(np.mean(np.mean(deltas**2, axis=1), axis=0))
        elif len(deltas.shape) == 2:
            gasce = np.mean(np.mean(deltas**2, axis=0))
        else:
            gasce = np.mean(deltas**2)
        return gasce

    def calc_all(   self, 
                    set_brier_ref, # to differentiate between fit and predict
                    confidences, # raw confidences of the model
                    labels, # indicate if sample is correct
                    assigned_bins,  # assigend bins for the given confidence
                    correctness_per_bin,  # probability that the sample is correct per bin
                    total_per_bin,  # total samples in bin
                    confidence_per_bin, # average confidence per bin
                    delta_p_f=None # deviation per bin
                    ):
        
        if set_brier_ref:
            prefix = 'Uncalib'
            color = 'red'
        else:
            prefix = 'Calib'
            color = 'green'

        num_samples = len(confidences)
        num_correct = np.count_nonzero(labels == 1)
        
        # Calculate scores for Data
        ece = self.ece(correctness_per_bin, confidence_per_bin, total_per_bin, num_samples)

        mse = self.mse(confidences, labels, num_samples)

        if delta_p_f is not None:
            asce = self.asce_deltas(total_per_bin, num_samples, delta_p_f)
        else:
            asce = self.asce(correctness_per_bin, confidence_per_bin, total_per_bin, num_samples)

        expeceted_variance = self.expeceted_variance(confidences, labels, assigned_bins, self.grid, num_samples)

        if set_brier_ref:
            self.p_r, self.brier_ref_score = self.brier_ref(num_correct, num_samples)
            if self.outputs: 
                print(f"{colored(prefix, color)} Baseline: {self.p_r}")
                print(f"{colored(prefix, color)} Brier ref: {self.brier_ref_score}")

        skill_score = self.skill_score(self.brier_ref_score, mse)
        
        if self.outputs: 
            print(f"{colored(prefix, color)} ECE: {ece}")
            print(f"{colored(prefix, color)} MSE: {mse}")
            print(f"{colored(prefix, color)} ASCE: {asce}") 
            print(f"{colored(prefix, color)} Expected Variance: {expeceted_variance}") 
            print(f"{colored(prefix, color)} Skill Score: {skill_score}\n") 

        results =   {
                        prefix: {
                            "ECE": ece,
                            "ASCE": asce,
                            "MSE": mse,
                            "Brier ref": self.brier_ref_score,
                            "Skill Score": skill_score
                        } 
                    }  

        return results
    
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
        assigned_bins = binning.round_model_to_grid(confidences, self.grid)
        if self.debug: print(f"TEST Assigned Bins: {assigned_bins}")

        # Calculate some metrics on the UNcorrected values
        total_per_bin, correctness_per_bin, confidence_per_bin = binning.bin_round_probabilities_discret(assigned_bins, labels, self.grid)

        num_samples = len(assigned_bins)
        num_correct = np.count_nonzero(labels == 1)
        
        # Get Expected Calibration Error
        ece = self.ece(correctness_per_bin, confidence_per_bin, total_per_bin, num_samples)

        # Get Mean Squared Error
        mse = self.mse(assigned_bins, labels, num_samples)

        # Get Average Squared Error
        asce = self.asce(correctness_per_bin, confidence_per_bin, total_per_bin, num_samples)

        # Get expected Variance 
        expeceted_variance = self.expeceted_variance(assigned_bins, labels, assigned_bins, self.grid, num_samples)

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
        
        # Get Group Average squared calibration error
        if deltas is not None:
            gasce = self.gasce(deltas)
            if self.outputs: print(f"{colored(prefix, color)} GASCE: {gasce}\n")
            results[prefix]['GASCE'] = gasce

        return total_per_bin, correctness_per_bin, results
    
    def add_to_score_table(self, run, uncalib_scores, calib_scores):
        
        uncalib_scores = list(list(uncalib_scores.values())[0].values())
        calib_scores = list(list(calib_scores.values())[0].values())
        score_difference = np.array(calib_scores) - np.array(uncalib_scores)

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
