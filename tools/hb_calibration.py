import numpy as np
from tools import binning
from tools.calibration_scores import score
from termcolor import colored

class hb_calibration:
    
    def __init__(self, grid, outputs, debug):
        self.grid = grid
        self.debug = debug
        self.outputs = outputs
        self.delta_p_f_ = []
        self.score_calibration = score(grid, outputs, debug)

    def fit(self, X, y):
        # Assign the values in X to the corresponding bin (discretize values)
        assigend_bins = binning.round_model_to_grid(X, self.grid)

        # Calculate correcteness bias in the given bin
        self.delta_p_f_ =  np.round(np.array([np.mean(y[(assigend_bins == i)] -  assigend_bins[(assigend_bins == i)]) for i in self.grid]), 2)
        self.delta_p_f_[np.isnan(self.delta_p_f_)] = 0

        if self.outputs: print(f"{colored('Deltas', 'green')}: {self.delta_p_f_}\n")

        return self

    def predict(self, X):
        if self.debug: print(f"TEST Preditions: {X}")    

        # Assign the values in X to the corresponding bin (discretize values)
        assigend_bins = binning.round_model_to_grid(X, self.grid)
        if self.debug: print(f"TEST Assigned Bins: {assigend_bins}")

        # Correct the model confidence with the calculated deltas
        X_ = np.array([bin_a+self.delta_p_f_[int(bin_a*10)] for bin_a in assigend_bins])

        return X_
    
    def calib_score(self, X, y, set_b_ref=False):
        # Number of samples in X
        num_samples = len(X)
        
        # Number of correct samples
        num_correct = np.count_nonzero(y == 1)

        # Assign the values in X to the corresponding bin (discretize values)
        assigend_bins = binning.round_model_to_grid(X, self.grid)
        if self.debug: print(f"TEST Assigned Bins: {assigend_bins}")

        # Calculate some metrics on the UNcorrected values
        total, correctness, confidence = binning.bin_round_probabilities_discret(assigend_bins, y, self.grid)

        # Calculate the deltas on the uncorrected values
        deltas = self.calc_deltas(y, assigend_bins)
        if self.outputs: print(f"{colored('Deltas', 'green')}: {deltas}\n")

        scores = self.score_calibration.calc_all(set_b_ref, 
                                                assigend_bins, 
                                                y, 
                                                num_correct, 
                                                num_samples, 
                                                assigend_bins, 
                                                correctness, 
                                                total, 
                                                confidence, 
                                                deltas)
        
        return total, correctness, scores
    
    def calc_deltas(self, y, assigend_bins):
        # Calculate the deltas
        deltas= np.round(np.array([np.mean(y[(assigend_bins == i)]) -  np.mean(assigend_bins[(assigend_bins == i)]) for i in self.grid]), 2)
        deltas[np.isnan(deltas)] = 0
        return deltas
