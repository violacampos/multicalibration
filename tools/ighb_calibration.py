import numpy as np
from sklearn.linear_model import LinearRegression
from tools.calibration_scores import score
from tools import binning

class IGHB_calibration:
    
    def __init__(self, grid, m, alpha, outputs, debug):
        self.grid = grid
        self.alpha = alpha
        self.m = m
        self.debug = debug
        self.outputs = outputs
        self.score_obj = score(grid, outputs, debug)

        self.deltas = None
        self.deltas_square = None
        self.P_S_p_g = []
        self.max_error = 0
        self.gasce = None

        self.changes = []

        
    def fit(self, X, y, groups):
        assigned_bins = binning.round_model_to_grid(X, self.grid)
        
        self.deltas = self.get_deltas(X, y, groups) 

        self.gasce = self.score_obj.gasce(assigned_bins, y, groups, grid=self.grid)
        if self.debug: print(f"GASCE: {self.gasce}")

        self.deltas_square = self.deltas**2

        self.P_S_p_g = np.array([[len(assigned_bins[(assigned_bins == i) & (g == 1)]) / len(X) for g in groups.T] for i in self.grid])
        #self.P_S_p_g = np.array([[len(assigned_bins[(assigned_bins == i) & (g == 1)]) / groups.sum() for g in groups.T] for i in self.grid])

        p_group = groups.sum(axis=0) / len(groups)
        if self.debug: print(f"P(X)=1: {p_group}")
        
        c = self.gasce*p_group
        if self.debug: print(f"While condition: {c}")
        
        self.max_error = c[np.argmax(c)]
        if self.debug: print(self.max_error)

        return self

    def predict(self, X, groups, test=False, is_correct=None):
        assigned_bins = binning.round_model_to_grid(X, self.grid)

        bin, group = np.unravel_index((self.P_S_p_g*self.deltas_square).argmax(), self.deltas.shape)
        if self.debug: print(f"Max delta in: Bin {bin}, Group {group}\n")

        max_delta = self.deltas[bin, group]
        if self.debug: print(f"Max delta: {max_delta}")

        X_ = np.array([bin_a+self.deltas[bin, group] if (bin_a == (bin/self.m)) and (groups[idx, group] == 1) else bin_a for idx, bin_a in enumerate(assigned_bins)])    

        
        ab_test = binning.round_model_to_grid(X_, self.grid)

        if test:        
            self.changes.append([bin, group, max_delta, len(ab_test[ab_test != assigned_bins]), [ab_test[ab_test != assigned_bins], groups[ab_test != assigned_bins], is_correct[ab_test != assigned_bins]], self.P_S_p_g[bin, group]])    
        else:
            """print(len(ab_test[ab_test != assigned_bins]))
            print(f"Max delta: {max_delta}")
            print(f"Max delta in: Bin {bin}, Group {group}")
            t = np.array([bin_a if (bin_a == (bin/self.m)) and (groups[idx, group] == 1) else 0 for idx, bin_a in enumerate(assigned_bins)])
            print(t[t!=0])
            print(bin/self.m)
            print(bin)
            print()"""

        return X_ 

    def get_deltas(self, X, y, groups):
        # Calculate correcteness bias in the given bin and group
        assigned_bins = binning.round_model_to_grid(X, self.grid)       

        # calculate the total correct per bin
        correct_per_bin_group = np.array([[np.divide(len(assigned_bins[(assigned_bins == i) & (y == 1) & (g ==1)]), len(assigned_bins[(assigned_bins == i) & (g ==1)])) for g in groups.T] for i in self.grid])
        correct_per_bin_group[np.isnan(correct_per_bin_group)] = 0
        
        # calculate the total count per bin
        total_per_bin_group = np.array([[len(assigned_bins[(assigned_bins == i) & (g ==1)]) for g in groups.T]  for i in self.grid])
        total_per_bin_group[np.isnan(total_per_bin_group)] = 0
        
        # sum the probabilities per bin
        bin_sums_group = np.array([[assigned_bins[(assigned_bins == i) & (g ==1)].sum() for g in groups.T]  for i in self.grid])

        # calculate the average confidence per bin
        average_bin_group_confidence = np.divide(bin_sums_group, total_per_bin_group, where=np.array(total_per_bin_group)!=0)
 
        """deltas = np.array([[np.mean(y[(assigned_bins == i) & (g == 1)] - assigned_bins[(assigned_bins == i) & (g == 1)]) for g in groups.T] for i in self.grid])
        deltas[np.isnan(deltas)] = 0   """
        
        deltas = []
        for corr_bin_group, conf_bin_group, bin_group_count in zip(correct_per_bin_group, average_bin_group_confidence, total_per_bin_group):
            #gasce += ((bin_count/total_per_bin_group.sum())*((corr_bin_group-conf_bin_group)**2))
            deltas.append((corr_bin_group-conf_bin_group))
        return np.array(deltas)