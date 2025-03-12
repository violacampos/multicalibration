import numpy as np
from sklearn.linear_model import LinearRegression
from tools.calibration_scores import score
from tools import binning

class LR_calibration:
    
    def __init__(self, grid, outputs, debug):
        self.grid = grid
        self.debug = debug
        self.outputs = outputs
        self.reg = None
        self.score_calibration = score(grid, outputs, debug)
 
    def fit(self, X, y):
        self.reg = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        return self.reg.predict(X)

    """
        Calculates the group conditional unbiasednes
    """
    def gcu(self, label, confidence, groups):
        gcu = np.round(np.array([np.mean(label[(col == 1)] -  confidence[(col == 1)]) for col in groups.T]), 2)
        gcu[np.isnan(gcu)] = 0
        return gcu

    def calib_score(self, probs, label, groups, set_b_ref=False):
        # Number of samples in X
        num_samples = len(probs)
        
        # Number of correct samples
        num_correct = np.count_nonzero(label == 1)

        # Assign the values in X to the corresponding bin (discretize values)
        assigend_bins = binning.round_model_to_grid(probs, self.grid)
        if self.debug: print(f"TEST Assigned Bins: {assigend_bins}")

        # Calculate some metrics on the UNcorrected values
        total, correctness, confidence = binning.bin_round_probabilities_discret(assigend_bins, label, self.grid)

        if self.debug: print(f"GCU: {self.gcu(label, probs, groups)}")
        if self.debug: print(f"GASCE: {self.score_calibration.gasce(assigend_bins, label, groups)}")

        # Calculate calibrations scores
        scores = self.score_calibration.calc_all(set_b_ref, 
                                                assigend_bins, 
                                                label, 
                                                num_correct, 
                                                num_samples, 
                                                assigend_bins, 
                                                correctness, 
                                                total, 
                                                confidence)
        
        return total, correctness, scores