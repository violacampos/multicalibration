import numpy as np
from sklearn.linear_model import LinearRegression
from tools.calibration_scores import score
from tools import binning

class IGHB_calibration:
    
    def __init__(self, grid, alpha, outputs, debug):
        self.grid = grid
        self.alpha = alpha
        self.debug = debug
        self.outputs = outputs
        self.score_calibration = score(grid, outputs, debug)

        self.deltas = None
        self.deltas_square = None
        self.max_error = 0
        self.gasce = None

        
    def fit(self, X, y, groups):
        assigned_bins = binning.round_model_to_grid(X, self.grid)
        self.deltas, self.gasce = self.score_calibration.gasce(assigned_bins, y, groups)

        self.deltas_square = self.deltas**2

        if self.debug: print(f"GASCE: {self.gasce}")

        p_group = groups.sum(axis=0) / len(groups)
        
        if self.debug: print(f"P(X)=1: {p_group}")
        
        c = self.gasce*p_group
        if self.debug: print(f"While condition: {c}")
        
        self.max_error = c[np.argmax(c)]
        if self.debug: print(self.max_error)

        return self

    def predict(self, X, groups):
        assigned_bins = binning.round_model_to_grid(X, self.grid)
        bin, group = np.unravel_index(self.deltas_square.argmax(), self.deltas.shape)
        if self.debug: print(f"Max delta in: Bin {bin}, Group {group}")
        
        max_delta = self.deltas[bin, group]
        if self.debug: print(f"Max delta: {max_delta}")

        X_ = np.array([bin_a+self.deltas[bin, group] if (int(bin_a*10) == bin) and (groups[idx, group] == 1) else bin_a for idx, bin_a in enumerate(assigned_bins)])

        return X_ 

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