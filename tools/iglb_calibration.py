import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from tools.calibration_scores import score
from tools import binning

class IGLB_calibration:
    
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
        self.LS = None

        
    def fit(self, X, y, groups):
        assigned_bins = binning.round_model_to_grid(X, self.grid)

        self.deltas = self.get_deltas(assigned_bins, y, groups) 

        self.gasce = np.mean(self.score_calibration.gasce(self.deltas), axis=0)
        if self.debug: print(f"GASCE: {self.gasce}")
        
        self.deltas_square = self.deltas**2

        p_group = groups.sum(axis=0) / len(groups)
        if self.debug: print(f"P(X)=1: {p_group}")
        
        c = self.gasce*p_group
        if self.debug: print(f"While condition: {c}")
        
        self.max_error = c[np.argmax(c)]

        self.LS = self.get_LS(X, y, assigned_bins, groups)

        return self

    def predict(self, X, groups):
        assigned_bins = binning.round_model_to_grid(X, self.grid)        
        
        P_S_p_g = self.get_P_S_p_g(assigned_bins, groups) 
        tau, bin, group = np.unravel_index((P_S_p_g*self.deltas_square).argmax(), self.deltas.shape)
        if self.debug: print(f"Max delta in: Tau {tau}, Bin {bin}, Group {group}")
        
        print(self.LS[tau, bin, group])
        if self.debug: print(f"Sigmoid Coef: {self.LS[tau, bin, group].coef_}, Intercept: {self.LS[tau, bin, group].intercept_}")

        for idx, bin_a in enumerate(assigned_bins):
            if (int(bin_a*10) <= bin) and (groups[idx, group] == 1) and self.LS[tau, bin, group] is not None:
                self.LS[tau, bin, group].predict([[X[idx]]])[0]
            else:
                bin_a

        if tau == 0:
            X_ = np.array([self.LS[tau, bin, group].predict([[X[idx]]])[0] if (int(bin_a*10) <= bin) and (groups[idx, group] == 1) and self.LS[tau, bin, group] is not None else X[idx] for idx, bin_a in enumerate(assigned_bins)])
        else:
            X_ = np.array([self.LS[tau, bin, group].predict([[X[idx]]])[0] if (int(bin_a*10) >= bin) and (groups[idx, group] == 1) and self.LS[tau, bin, group] is not None else X[idx] for idx, bin_a in enumerate(assigned_bins)])

        return X_ 

    """
        Calculates the group conditional unbiasednes
    """
    def gcu(self, label, confidence, groups):
        gcu = np.round(np.array([np.mean(label[(col == 1)] -  confidence[(col == 1)]) for col in groups.T]), 2)
        gcu[np.isnan(gcu)] = 0
        return gcu
    
    def get_deltas(self, assigned_bins, y, groups):
        # Calculate correcteness bias in the given bin and group
        deltas_smaller = []
        for i in self.grid:
            temp = []
            for g in groups.T:
                temp.append(np.mean(y[(assigned_bins <= i) & (g == 1)] -  assigned_bins[(assigned_bins <= i) & (g == 1)]))
            deltas_smaller.append(temp)
        
        deltas_greater = []
        for i in self.grid:
            temp = []
            for g in groups.T:
                temp.append(np.mean(y[(assigned_bins >= i) & (g == 1)] -  assigned_bins[(assigned_bins >= i) & (g == 1)]))
            deltas_greater.append(temp)
        
        deltas = np.stack([np.array(deltas_smaller), np.array(deltas_greater)])
        deltas[np.isnan(deltas)] = 0   

        return deltas
    
    def get_P_S_p_g(self, assigned_bins, groups):
        P_S_p_g_smaller = []

        for i in self.grid:
            temp = []
            for g in groups.T:
                temp.append(len(assigned_bins[(assigned_bins <= i) & (g == 1)]) / len(assigned_bins))
            P_S_p_g_smaller.append(temp)

        P_S_p_g_greater = []

        for i in self.grid:
            temp = []
            for g in groups.T:
                temp.append(len(assigned_bins[(assigned_bins >= i) & (g == 1)]) / len(assigned_bins))
            P_S_p_g_greater.append(temp)

        P_S_p_g = np.stack([np.array(P_S_p_g_smaller), np.array(P_S_p_g_greater)]) 
        P_S_p_g[np.isnan(P_S_p_g)] = 0   

        return P_S_p_g
    
    def get_LS(self, X, is_correct, assigned_bins, groups):
        LS_smaller = []
        
        for i in self.grid:
            temp = []
            for g in groups.T:
                calibrator = LogisticRegression()
                if len(np.unique(is_correct[(X <= i) & (g == 1)])) < 2:
                    temp.append(None)
                else:
                    calibrator.fit(X[(X <= i) & (g == 1)].reshape(-1, 1), is_correct[(X <= i) & (g == 1)])
                    temp.append(calibrator) 
            LS_smaller.append(temp)

        LS_greater = []

        for i in self.grid:
            temp = []
            for g in groups.T:
                calibrator = LogisticRegression()
                if len(np.unique(is_correct[(X >= i) & (g == 1)])) < 2:
                    temp.append(None)
                else:
                    calibrator.fit(X[(X >= i) & (g == 1)].reshape(-1, 1), is_correct[(X >= i) & (g == 1)])
                    temp.append(calibrator) 
            LS_greater.append(temp)

        LS = np.stack([np.array(LS_smaller), np.array(LS_greater)]) 

        return LS

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