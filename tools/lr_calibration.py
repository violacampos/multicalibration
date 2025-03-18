import numpy as np
from sklearn.linear_model import LinearRegression
from tools.calibration_scores import score
from tools import binning

class lr_calibration:
    
    def __init__(self, grid, outputs, debug):
        self.grid = grid
        self.debug = debug
        self.outputs = outputs
        self.reg = None
        self.score_obj = score(grid, outputs, debug)
 
    def fit(self, X, y):
        self.reg = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        return self.reg.predict(X)
    
    def get_deltas(self, X, y):
        # Assign the values in X to the corresponding bin (discretize values)
        assigend_bins = binning.round_model_to_grid(X, self.grid)
        # Calculate the deltas
        deltas= np.round(np.array([np.mean(y[(assigend_bins == i)]) -  np.mean(assigend_bins[(assigend_bins == i)]) for i in self.grid]), 2)
        deltas[np.isnan(deltas)] = 0
        return deltas