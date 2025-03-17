import numpy as np
from tools.calibration_scores import score
from tools import binning
from scipy.special import logit, expit
from scipy.optimize import minimize

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
        # get the assigned bins of the confidences
        assigned_bins = binning.round_model_to_grid(X, self.grid)

        # calculate deltas
        self.deltas = self.get_deltas(assigned_bins, y, groups) 

        # set the group average square calibration error
        self.gasce = np.mean(self.score_calibration.gasce(self.deltas), axis=0)
        if self.debug: print(f"GASCE: {self.gasce}")
        
        # set deltas_square for further use
        self.deltas_square = self.deltas**2     

        # set the linear scaling for every bin, group and tau combination
        self.LS = self.get_LS(X, y, groups)

        return self

    def predict(self, X, groups, assigned_bins, tau, bin, group):
        
        # get the alpha and beta values for the given tau, bin, group
        alpha_star = self.LS[tau, bin, group][0]
        beta_star = self.LS[tau, bin, group][1]
        if self.debug: print(f"Alpha: {alpha_star}, Beta: {beta_star}")

        # Set the new values with the help of the alpha and beta values for all elements in the set
        if tau == 0:
            X_ = np.array([expit(alpha_star + beta_star * logit(X[idx])) if (int(bin_a*10) <= bin) and (groups[idx, group] == 1) else X[idx] for idx, bin_a in enumerate(assigned_bins)])
        else:
            X_ = np.array([expit(alpha_star + beta_star * logit(X[idx])) if (int(bin_a*10) >= bin) and (groups[idx, group] == 1) else X[idx] for idx, bin_a in enumerate(assigned_bins)])

        return X_ 
   
    def get_deltas(self, assigned_bins, y, groups):
        # Calculate correcteness bias in the given bin, group and use smaller then
        deltas_smaller = [[np.mean(y[(assigned_bins <= i) & (g == 1)] -  assigned_bins[(assigned_bins <= i) & (g == 1)]) for g in groups.T] for i in self.grid]

        # Calculate correcteness bias in the given bin, group and use greater then
        deltas_greater = [[np.mean(y[(assigned_bins >= i) & (g == 1)] -  assigned_bins[(assigned_bins >= i) & (g == 1)]) for g in groups.T]for i in self.grid]
        
        # Stack both arrays index 0 is <= and 1 is >=
        deltas = np.stack([np.array(deltas_smaller), np.array(deltas_greater)])
        deltas[np.isnan(deltas)] = 0   

        return deltas
    
    def get_P_S_p_g(self, assigned_bins, groups):
        
        # Create sets with tau <= bin, for each bin and group
        P_S_p_g_smaller = [[len(assigned_bins[(assigned_bins <= i) & (g == 1)]) / len(assigned_bins) for g in groups.T] for i in self.grid]

        # Create sets with tau >= bin, for each bin and group
        P_S_p_g_greater = [[len(assigned_bins[(assigned_bins >= i) & (g == 1)]) / len(assigned_bins) for g in groups.T] for i in self.grid]

        # Stack both arrays index 0 is <= and 1 is >=
        P_S_p_g = np.stack([np.array(P_S_p_g_smaller), np.array(P_S_p_g_greater)]) 
        P_S_p_g[np.isnan(P_S_p_g)] = 0   

        return P_S_p_g
    
    def get_LS(self, X, is_correct, groups):
        # Get alpha and beta values for <= subsets
        LS_smaller = [[self.linear_scaling(X[(X <= i) & (g == 1)], is_correct[(X <= i) & (g == 1)]) for idx, g in enumerate(groups.T)] for i in self.grid]
        
        # Get alpha and beta values for >= subsets
        LS_greater = [[self.linear_scaling(X[(X >= i) & (g == 1)], is_correct[(X >= i) & (g == 1)]) for idx, g in enumerate(groups.T)] for i in self.grid]
        
        # Stack both arrays index 0 is <= and 1 is >=
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
    
    def linear_scaling(self, X, is_correct):
        #print(f"\nBin: {i}, Group: {g}")
        # clip the value to dont get -inf or inf
        X = np.clip(X, 1e-10, 1 - 1e-10)
        # get logits for confidences
        logit_f = logit(X)
        
        # mse function to optimize for alpha and beta
        def mse(params):
            alpha, beta = params
            transformed = expit(alpha + beta * logit_f)  # LS[f](x)
            return np.mean((transformed - is_correct) ** 2)  # calculate the MSE

        # minimize for the mse and get alpha and beta values
        result = minimize(mse, x0=[0, 1])  
        alpha_star, beta_star = result.x

        return [alpha_star, beta_star]