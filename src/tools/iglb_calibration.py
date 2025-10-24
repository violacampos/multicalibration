import numpy as np
from tools.calibration_scores import score
from tools import binning
from scipy.special import logit, expit
from scipy.optimize import minimize


class IGLB_calibration:

    def __init__(self, grid, alpha, m, outputs, debug):
        """
        Initilaizes a iterative group linear binning object

        :param grid: used grid for calibration
        :param m: Number of bins
        :param alpha: Value to determine if the algorithm should stop
        :param outputs: flag to enable optional outputs
        :param debug: flag to enable debug outputs
        """
        assert len(grid) - 1 == m, "Grid size and m do not match"
        # assert alpha == 1 / m, "Alpha and m do not match"
        self.grid = grid
        # self.alpha = alpha
        self.debug = debug
        self.m = m
        self.outputs = outputs
        self.score_obj = score(grid, outputs, debug)

        self.deltas = None
        self.deltas_square = None
        self.LS = None

        self.changes = []

    def fit(self, X, y, groups):
        """
        Learns deltas for each bin-group combination and creates a linear scaling within each bin-group combination

        :param X: Probabilities for calibration
        :param y: Label of correctness
        :param groups: Group matrix

        :return: ILGB object
        """
        # calculate deltas
        self.deltas = self.get_deltas(X, y, groups)

        # set deltas_square for further use
        self.deltas_square = self.deltas**2

        # set the linear scaling for every bin, group and tau combination
        self.LS = self.get_LS(X, y, groups)

        return self

    def predict(
        self, X, groups, assigned_bins, tau, bin, group, test=False, is_correct=None
    ):
        """
        Uses the learned delta on a selected bin-group combination for adjustement.

        :param X: Probabilities for calibration
        :param groups: Group matrix
        :param assigned_bins: Discretized probabilities # VIOLA: TODO removed from iteration, only used for robins saved changes if test==True -> TODO check
        :param tau: Tau of the probabilites that have to be adjusted
        :param bin: Bin of the probabilites that have to be adjusted
        :param group: Group of the probabilites that have to be adjusted
        :param test: Flag to store changes on test subset
        :param is_correct: Labels of correctness for history

        :return: adjusted probabilities
        """
        # get the alpha and beta values for the given tau, bin, group
        alpha_star, beta_star = self.LS[tau, bin, group]

        if self.debug:
            print(f"Alpha: {alpha_star}, Beta: {beta_star}")

        # Set the new values with the help of the alpha and beta values for all elements in the set
        if tau == 0:
            X_ = np.array(
                [
                    (
                        expit(alpha_star + beta_star * logit(X[idx]))
                        if (bin_a <= (bin / self.m)) and (groups[idx, group] == 1)
                        else X[idx]
                    )
                    for idx, bin_a in enumerate(X)
                ]
            )
        else:
            X_ = np.array(
                [
                    (
                        expit(alpha_star + beta_star * logit(X[idx]))
                        if (bin_a >= (bin / self.m)) and (groups[idx, group] == 1)
                        else X[idx]
                    )
                    for idx, bin_a in enumerate(X)
                ]
            )

        # Save changes on test subset VIOLA: unchecked
        if test:
            ab_test = binning.round_model_to_grid(X_, self.grid)
            self.changes.append(
                [
                    tau,
                    bin,
                    group,
                    (alpha_star, beta_star),
                    len(ab_test[ab_test != assigned_bins]),
                    [
                        ab_test[ab_test != assigned_bins],
                        groups[ab_test != assigned_bins],
                        is_correct[ab_test != assigned_bins],
                    ],
                ]
            )

        return X_

    def get_deltas(self, X, y, groups):
        """
        Calculates the deltas for the different tau-bin-group combinations

        :param X: Probabilities for calibration
        :param y: Labels of correctness
        :param groups: Group matrix

        :return: 3D delta array
        """
        # get the assigned bins of the confidences
        assigned_bins = binning.round_model_to_grid(X, self.grid)

        # Calculate correcteness bias in the given bin, group and use smaller then
        deltas_smaller = [
            [
                np.mean(
                    y[(X <= i) & (g == 1)]
                    - X[(X <= i) & (g == 1)]
                )
                for g in groups.T
            ]
            for i in self.grid
        ]

        # Calculate correcteness bias in the given bin, group and use greater then
        deltas_greater = [
            [
                np.mean(
                    y[(X >= i) & (g == 1)]
                    - X[(X >= i) & (g == 1)]
                )
                for g in groups.T
            ]
            for i in self.grid
        ]

        # Stack both arrays index 0 is <= and 1 is >=
        deltas = np.stack([np.array(deltas_smaller), np.array(deltas_greater)])
        deltas[np.isnan(deltas)] = 0

        return deltas

    def get_P_S_p_g(self, probs, groups):
        """
        Calculates the probability that a sample is in the different tau-bin-group combinations

        :param probs: Sample probabilities
        :param groups: Group matrix

        :return: 3D probaility array
        """
        # Create sets with tau <= bin, for each bin and group
        P_S_p_g_smaller = [
            [
                len(probs[(probs <= i) & (g == 1)]) / len(probs)
                for g in groups.T
            ]
            for i in self.grid
        ]

        # Create sets with tau >= bin, for each bin and group
        P_S_p_g_greater = [
            [
                len(probs[(probs >= i) & (g == 1)]) / len(probs)
                for g in groups.T
            ]
            for i in self.grid
        ]

        # Stack both arrays index 0 is <= and 1 is >=
        P_S_p_g = np.stack([np.array(P_S_p_g_smaller), np.array(P_S_p_g_greater)])
        P_S_p_g[np.isnan(P_S_p_g)] = 0

        return P_S_p_g

    def get_LS(self, X, is_correct, groups):
        """
        Gets the liner scaling parameters for the different tau-bin-group combinations

        :param X: Probabilities
        :param is_correct: Labels of correctness
        :param groups: Group matrix

        :return: 3D probaility array
        """
        # Get alpha and beta values for <= subsets
        LS_smaller = [
            [
                self.linear_scaling(
                    X[(X <= i) & (g == 1)], is_correct[(X <= i) & (g == 1)]
                )
                for g in groups.T
            ]
            for i in self.grid
        ]

        # Get alpha and beta values for >= subsets
        LS_greater = [
            [
                self.linear_scaling(
                    X[(X >= i) & (g == 1)], is_correct[(X >= i) & (g == 1)]
                )
                for g in groups.T
            ]
            for i in self.grid
        ]

        # Stack both arrays index 0 is <= and 1 is >=
        LS = np.stack([np.array(LS_smaller), np.array(LS_greater)])
        return LS

    def linear_scaling(self, X, is_correct):
        """
        Learns the alpha and beta values of the linear scaling for the given probabilities and labels.

        :param X: Probabilities
        :param is_correct: Labels of correctness

        :return: List with alpha and beta
        """
        # clip the value to dont get -inf or inf
        X = np.clip(X, 1e-10, 1 - 1e-10)
        # get logits for confidences
        logit_f = logit(X)

        # mse function to optimize for alpha and beta
        def mse(params):
            alpha, beta = params
            transformed = expit(alpha + beta * logit_f)
            return np.mean((transformed - is_correct) ** 2)

        # minimize for the mse and get alpha and beta values
        result = minimize(mse, x0=[0, 1])
        alpha_star, beta_star = result.x

        return [alpha_star, beta_star]
