import numpy as np
from tools import binning
from tools.calibration_scores import score
from termcolor import colored

class hb_calibration:
    
    def __init__(self, grid, args, outputs, debug):
        """
            Initilaizes a histogram binning object

            :param grid: used grid for calibration
            :param args: passed arguments from command line
            :param outputs: flag to enable optional outputs
            :param debug: flag to enable debug outputs
        """

        self.grid = grid
        self.debug = debug
        self.outputs = outputs
        self.delta_p_f_ = []
        self.m = args.bin_count
        self.score_obj = score(grid, outputs, debug)

    def fit(self, X, y):
        """
            Learns the deltas for each bin on the data X and y.

            :param X: the probabilities used for calibration
            :param y: label of correctness for each sample

            :return: Histogram binning object
        """
        # Calculate correcteness bias in the given bin
        self.delta_p_f_ = self.get_deltas(X, y)
        if self.outputs: print(f"{colored('Deltas', 'green')}: {self.delta_p_f_}\n")

        return self

    def predict(self, X):
        """
            Uses the learned deltas to adjust the probabilities X

            :param X: the probabilities to adjust

            :return: Adjusted probabilities
        """
        # Assign the values in X to the corresponding bin (discretize values)
        assigend_bins = binning.round_model_to_grid(X, self.grid)
        if self.debug: print(f"TEST Assigned Bins: {assigend_bins}")

        # Correct the model confidence with the calculated deltas
        X_ = np.array([bin_a+self.delta_p_f_[np.where(self.grid == bin_a)[0][0]] for bin_a in assigend_bins])

        return X_
        
    def get_deltas(self, X, y):
        """
            Calculates the deltas for each possible bin.

            :param X: the probabilities used for calibration
            :param y: label of correctness for each sample

            :return: deltas
        """
        # Assign the values in X to the corresponding bin (discretize values)
        assigend_bins = binning.round_model_to_grid(X, self.grid)
        
        # Calculate the deltas
        deltas= np.round(np.array([np.mean(y[(assigend_bins == i)]) -  np.mean(assigend_bins[(assigend_bins == i)]) for i in self.grid]), 2)
        deltas[np.isnan(deltas)] = 0
        
        return deltas
