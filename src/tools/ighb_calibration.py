import numpy as np
from tools.calibration_scores import score
from tools import binning, groups


class IGHB_calibration:

    def __init__(self, grid, m, alpha, outputs, debug):
        """
        Initilaizes a iterative group histogram binning object

        :param grid: used grid for calibration
        :param m: Number of bins
        :param alpha: Value to determine if the algorithm should stop
        :param outputs: flag to enable optional outputs
        :param debug: flag to enable debug outputs
        """

        assert len(grid) - 1 == m, "Grid size and m do not match"
        assert alpha == 1 / m, "Alpha and m do not match"
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
        """
        Learns deltas for each bin-group combination and calculates the max error on the data.

        :param X: Probabilities for calibration
        :param y: Label of correctness
        :param groups: Group matrix

        :return: IGHB object
        """
        
        # Get deltas and probability for every bin-group combination
        self.deltas = self.get_deltas(X, y, groups)
        self.deltas_square = self.deltas**2
        group_counts = groups.sum(axis=0)
        counts = self.get_bin_group_counts(X, groups)
        total_counts = len(X)
        # probability of bin and group membership
        self.P_S_p_g = counts / total_counts
        
        # Get GASCE
        self.gasce = self.score_obj.gasce_not_rounded(X, y, groups)
        if self.debug:
            print(f"GASCE: {self.gasce}, sum: {self.gasce.sum()}")

        # Probability of group membership
        p_group = group_counts / len(groups)
        if self.debug:
            print(f"P(g(X)=1): {p_group}")

        # Calculate effect of calibration error for each group
        c = self.gasce * p_group
        if self.debug:
            print(f"gASCE * P(g(X)=1): {c}")

        # Get max error for stop condition
        self.max_error = c[np.argmax(c)]
        if self.debug:
            print(f"Max error: {self.max_error}")
        return self

    def predict(self, X, groups, test=False, is_correct=None):
        """
        Uses the learned delta on a selected bin-group combination for adjustement.

        :param X: Probabilities for calibration
        :param groups: Group matrix
        :param test: Flag to store changes on test subset
        :param is_correct: Labels of correctness for history

        :return: adjusted probabilities
        """
        assigned_bins = binning.round_model_to_grid(X, self.grid)

        # Select the bin-group combiation with max probability for 
        # a sample to be in the bin-group combintation times the deltas squared
        bin, group = np.unravel_index(
            (self.P_S_p_g * self.deltas_square).argmax(), self.deltas.shape
        )
        if self.debug:
            print(f"Max delta in: Bin {bin}, Group {group}")

        # Get delta for the combination
        max_delta = self.deltas[bin, group]
        if self.debug:
            print(f"Max delta: {max_delta}")

        # Adjust samples in that combination
        bin_indices = np.digitize(X, self.grid) - 1
        mask = (bin_indices == bin) & (groups[:, group] == 1)
        X_ = X.copy()
        X_[mask] += self.deltas[bin, group]
        

        # Save changes on test subset # VIOLA: did not check this
        if test:
            ab_test = binning.round_model_to_grid(X_, self.grid)
            self.changes.append(
                [
                    bin,
                    group,
                    max_delta,
                    len(ab_test[ab_test != assigned_bins]),
                    [
                        ab_test[ab_test != assigned_bins],
                        groups[ab_test != assigned_bins],
                        is_correct[ab_test != assigned_bins],
                    ],
                    self.P_S_p_g[bin, group],
                ]
            )

        return X_
    

    
    def get_bin_group_counts(self, X, groups):
        """
        Returns an array of shape (n_bins, n_groups) with the absolute number of samples
        from X that fall into each bin and are in each group.

        :param X: Probabilities for calibration
        :param groups: Group matrix

        :return: 2D array of counts
        """
        # Assign each sample to a bin index (0-based)
        bin_indices = np.digitize(X, self.grid) - 1  

        n_bins = len(self.grid)
        n_groups = groups.shape[1]
        counts = np.zeros((n_bins, n_groups), dtype=int)

        for i in range(n_bins):
            for j in range(n_groups):
                # Select samples in bin i and group j
                mask = (bin_indices == i) & (groups[:, j] == 1)
                counts[i, j] = np.sum(mask)
        return counts
    
    def get_correct_per_bin_group_counts(self, X, y, groups):
        """
        Returns an array of shape (n_bins, n_groups) with the absolute number of samples
        from X that fall into each bin and are in each group.

        :param X: Probabilities for calibration
        :param y: Labels of correctness
        :param groups: Group matrix

        :return: 2D array of counts of correct samples
        """
        # Assign each sample to a bin index (0-based)
        bin_indices = np.digitize(X, self.grid) - 1  

        n_bins = len(self.grid)
        n_groups = groups.shape[1]
        counts = np.zeros((n_bins, n_groups), dtype=int)

        for i in range(n_bins):
            for j in range(n_groups):
                # Select samples in bin i and group j
                mask = (bin_indices == i) & (y == 1) & (groups[:, j] == 1)
                counts[i, j] = np.sum(mask)
        return counts
    
    def get_mean_per_bin_group(self, X, groups):
        """
        Returns an array of shape (n_bins, n_groups) with the mean value of X
        for each bin and group.

        :param X: Probabilities for calibration
        :param groups: Group matrix

        :return: 2D array of mean values
        """
        bin_indices = np.digitize(X, self.grid) - 1
        n_bins = len(self.grid)
        n_groups = groups.shape[1]
        means = np.zeros((n_bins, n_groups), dtype=float)

        for i in range(n_bins):
            for j in range(n_groups):
                mask = (bin_indices == i) & (groups[:, j] == 1)
                if np.any(mask):
                    means[i, j] = X[mask].mean()
                else:
                    means[i, j] = 0.0
        return means

    def get_deltas(self, X, y, groups):
        """
        Calculates the deltas for each bin-group combination.

        :param X: Probabilities for calibration
        :param y: Labels of correctness for history
        :param groups: Group matrix

        :return: 2D Array of deltas (between likelihood and correctness) for each bin-group combination
        """
        # round to grid
        # assigned_bins = binning.round_model_to_grid(X, self.grid)
        
        counts = self.get_bin_group_counts(X, groups)
        correct_counts = self.get_correct_per_bin_group_counts(X, y, groups)

        # correctness per bin and group
        correct_per_bin_group_ = np.divide(
            correct_counts,
            counts,
            out=np.zeros_like(correct_counts, dtype=float),
            where=(counts > 0),
        )
        
        probs_per_bin_group_ = self.get_mean_per_bin_group(X, groups)

        deltas = correct_per_bin_group_ - probs_per_bin_group_
        return deltas

        