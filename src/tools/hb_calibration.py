import numpy as np
from tools import binning
from tools.calibration_scores import Score
from termcolor import colored

class Hb_calibration:
    
    def __init__(self, grid:binning.Binning, args:dict):
        """
            Initializes a histogram binning object

            :param grid: used grid for calibration
            :param args: passed arguments from command line

        """

        self.bins = grid
        self.debug = args.debug
        self.outputs = args.print_info
        self.delta_p_f_ = []
        self.m = args.bin_count
        self.score_obj = Score(grid, args)

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
        assigned_bins = self.bins.round_probabilities_to_grid(X)
        if self.debug: print(f"TEST Assigned Bins: {assigned_bins}")

        # Correct the model confidence with the calculated deltas
        X_ = np.array([bin_a+self.delta_p_f_[np.where(self.bins.grid == bin_a)[0][0]] for bin_a in assigned_bins])

        return X_
        
    def get_deltas(self, X, y):
        """
            Calculates the deltas for each possible bin.

            :param X: the probabilities used for calibration
            :param y: label of correctness for each sample

            :return: deltas
        """
        # Assign the values in X to the corresponding bin (discretize values)
        assigned_bins = self.bins.round_probabilities_to_grid(X)
        
        # Calculate the deltas
        deltas= np.array([np.mean(y[(assigned_bins == i)]) -  i for i in self.bins.grid])
        deltas[np.isnan(deltas)] = 0
        
        return deltas
