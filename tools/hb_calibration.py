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
        self.score_obj = score(grid, outputs, debug)

    def fit(self, X, y):
        # Calculate correcteness bias in the given bin
        self.delta_p_f_ = self.get_deltas(X, y)
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
        
    def get_deltas(self, X, y):
        # Assign the values in X to the corresponding bin (discretize values)
        assigend_bins = binning.round_model_to_grid(X, self.grid)
        # Calculate the deltas
        deltas= np.round(np.array([np.mean(y[(assigend_bins == i)]) -  np.mean(assigend_bins[(assigend_bins == i)]) for i in self.grid]), 2)
        deltas[np.isnan(deltas)] = 0
        return deltas
