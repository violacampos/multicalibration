import numpy as np
from tools import calibration_scores, binning
from termcolor import colored

class hb_calibration:
    
    def __init__(self, grid, outputs, debug):
        self.grid = grid
        self.debug = debug
        self.outputs = outputs
        self.delta_p_f_ = []
        self.f_dach = []
        self.fit_correct_per_bin = []
        self.p_r = 0
        self.brier_ref = 0


    def calculate_scores(self, 
                         is_fit, # to differentiate between fit and predict
                         confidences, # raw confidences of the model
                         labels, # indicate if sample is correct
                         num_correct, # total correct
                         num_samples, # total samples
                         assigend_bins,  # assigend bins for the given confidence
                         correctness_per_bin,  # probability that the sample is correct per bin
                         total_per_bin,  # total samples in bin
                         confidence_per_bin, # average confidence per bin
                         delta_p_f, # deviation per bin
                         ):
        if is_fit:
            prefix = 'Fit'
            color = 'cyan'
        else:
            prefix = 'Predict'
            color = 'red'
        
        # Calculate scores for Data
        ece = calibration_scores.ece(correctness_per_bin, confidence_per_bin, total_per_bin, num_samples)
        if self.outputs: print(f"{colored(prefix, color)} ECE: {ece}")

        mse = calibration_scores.mse(confidences, labels, num_samples)
        if self.outputs: print(f"{colored(prefix, color)} MSE: {mse}")

        asce = calibration_scores.asce_deltas(total_per_bin, num_samples, delta_p_f)
        if self.outputs: print(f"{colored(prefix, color)} ASCE: {asce}") 

        expeceted_variance = calibration_scores.expeceted_variance(confidences, labels, assigend_bins, self.grid, num_samples)
        if self.outputs: print(f"{colored(prefix, color)} Expected Variance: {expeceted_variance}") 

        if is_fit:
            self.p_r, self.brier_ref = calibration_scores.brier_ref(num_correct, num_samples)
            if self.outputs: print(f"{colored(prefix, color)} Baseline: {self.p_r}")
            if self.outputs: print(f"{colored(prefix, color)} Brier ref: {self.brier_ref}")

        skill_score = calibration_scores.skill_score(self.brier_ref, mse)
        if self.outputs: print(f"{colored(prefix, color)} Skill Score: {skill_score}\n") 


        results =   {
                        prefix: {
                            "ECE": ece,
                            "ASCE": asce,
                            "MSE": mse,
                            "Brier ref": self.brier_ref,
                            "Skill Score": skill_score
                        } 
                    }  

        return results

    def f_x(f_strich_x, y, bin):
        P_D = len(y[(f_strich_x == bin)  & (y == 1)]) / len(y[(f_strich_x == bin)])
        return P_D


    def fit(self, X, y):
        # Number of samples in X
        num_samples = len(X)
        # Number of correct samples
        num_correct = np.count_nonzero(y == 1)
        # Assign the values in X to the corresponding bin (discretize values)
        assigend_bins = binning.round_model_to_grid(X, self.grid)

        # Calculated the mean correcteness, total and average confidence of the confidence-scores in th assigned bins
        if self.debug: print(f"TRAIN Assigned Bins: {assigend_bins}")
        total_per_bin, correctness_per_bin, confidence_per_bin = binning.bin_round_probabilities_discret(assigend_bins, y, self.grid)
        if self.debug: print(f"Uniform Grid: {self.grid}")
        if self.debug: print(f"TRAIN Correct per bin: {correctness_per_bin}") 

        # Calculate correcteness bias in the given bin
        self.delta_p_f_ =  np.round(np.array([np.mean(y[(assigend_bins == i)] -  X[(assigend_bins == i)]) for i in self.grid]), 2)
        self.delta_p_f_[np.isnan(self.delta_p_f_)] = 0

        print(f"{colored('Deltas', 'green')}: {self.delta_p_f_}\n")
        # Calculate some scores on the given data like MSE, ECE, ASCE, ...
        scores = self.calculate_scores(True, 
                                       X, 
                                       y, 
                                       num_correct, 
                                       num_samples, 
                                       assigend_bins, 
                                       correctness_per_bin, 
                                       total_per_bin, 
                                       confidence_per_bin, 
                                       self.delta_p_f_)

        return correctness_per_bin, total_per_bin, scores

    def predict(self, X, y):
        # Number of samples in X
        num_samples = len(X)
        
        # Number of correct samples
        num_correct = np.count_nonzero(y == 1)

        if self.debug: print(f"TEST Preditions: {X}")    

        # Assign the values in X to the corresponding bin (discretize values)
        assigend_bins = binning.round_model_to_grid(X, self.grid)
        if self.debug: print(f"TEST Assigned Bins: {assigend_bins}")

        # Correct the model confidence with the calculated deltas
        X_ = np.array([bin_a+self.delta_p_f_[int(bin_a*10)] for bin_a in assigend_bins])

        # Calculate the new assigned bins with the corrected values
        assigend_bins_corrected = binning.round_model_to_grid(X_, self.grid)

        # Calculate some metrics on the corrected values
        total_per_bin, correctness_per_bin, confidence_per_bin = binning.bin_round_probabilities_discret(assigend_bins_corrected, y, self.grid)

        # Calculate the deltas on the corrected values
        deltas_test= np.round(np.array([np.mean(y[(assigend_bins_corrected == i)]) -  np.mean(X_[(assigend_bins_corrected == i)]) for i in self.grid]), 2)
        deltas_test[np.isnan(deltas_test)] = 0

        if self.outputs: print(f"{colored('Deltas Test', 'green')}: {deltas_test}\n")

        scores = self.calculate_scores(False, 
                                       X_, 
                                       y, 
                                       num_correct, 
                                       num_samples, 
                                       assigend_bins_corrected, 
                                       correctness_per_bin, 
                                       total_per_bin, 
                                       confidence_per_bin, 
                                       deltas_test)

        if self.debug: print(f"TEST Corrected Values: {correctness_per_bin}")

        return correctness_per_bin, total_per_bin, scores