import numpy as np
from tools.calibration_scores import score
from tools import binning


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
        # Assign probabilties to bins
        assigned_bins = binning.round_model_to_grid(X, self.grid)

        # Get deltas and probability for every bin-group combination
        self.deltas = self.get_deltas(X, y, groups)
        self.deltas_square = self.deltas**2
        group_counts = groups.sum(axis=0)
        self.P_S_p_g = np.array(
            [
                [
                    len(assigned_bins[(np.round(assigned_bins, 2) == np.round(i, 2)) & (g == 1)])
                    / len(
                        X
                    )  # group_counts[g_idx] VIOLA check this!! -> handle empty groups
                    for g_idx, g in enumerate(groups.T)
                ]
                for i in self.grid
            ]
        )

        # Get GASCE
        self.gasce = self.score_obj.gasce(assigned_bins, y, groups, grid=self.grid)
        if self.debug:
            print(f"GASCE: {self.gasce}, sum: {self.gasce.sum()}")

        # Probability of that a sample is in a group
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

        # Select the bin-group combiation with max probability for a sample to be in the bin-group combintation times the deltas squared
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
        X_ = np.array(
            [
                (
                    bin_a + self.deltas[bin, group]
                    if (np.round(bin_a, 2) == (bin / self.m))
                    and (groups[idx, group] == 1)
                    else bin_a
                )
                for idx, bin_a in enumerate(
                    assigned_bins
                )  # VIOLA: original probs vs discretized?
            ]
        )

        # Save changes on test subset
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

    def get_deltas(self, X, y, groups):
        """
        Calculates the deltas for each bin-group combination.

        :param X: Probabilities for calibration
        :param y: Labels of correctness for history
        :param groups: Group matrix

        :return: 2D Array of deltas
        """
        # round to grid
        assigned_bins = binning.round_model_to_grid(X, self.grid)

        # calculate the total correct per bin and group
        correct_per_bin_group = np.array(
            [
                [
                    np.divide(
                        len(assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3)) & (y == 1) & (g == 1)]),
                        len(assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3)) & (g == 1)]),
                        
                    )
                    for g in groups.T
                ]
                for i in self.grid
            ]
        )
        correct_per_bin_group[np.isnan(correct_per_bin_group)] = 0

        # calculate the total count per bin
        total_per_bin_group = np.array(
            [
                [len(assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3)) & (g == 1)]) for g in groups.T]
                for i in self.grid
            ]
        )
        total_per_bin_group[np.isnan(total_per_bin_group)] = 0

        # sum the probabilities per bin
        bin_sums_group = np.array(
            [
                [assigned_bins[(np.round(assigned_bins, 3) == np.round(i, 3)) & (g == 1)].sum() for g in groups.T]
                for i in self.grid
            ]
        )

        # calculate the average confidence per bin
        average_bin_group_confidence = np.divide(
            bin_sums_group,
            total_per_bin_group,
            where=np.array(total_per_bin_group) != 0,
        )

        deltas = []
        for corr_bin_group, conf_bin_group, bin_group_count in zip(
            correct_per_bin_group, average_bin_group_confidence, total_per_bin_group
        ):
            deltas.append((corr_bin_group - conf_bin_group))
        return np.array(deltas)
