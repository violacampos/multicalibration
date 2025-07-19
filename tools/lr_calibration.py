from sklearn.linear_model import LinearRegression
from tools.calibration_scores import score

class lr_calibration:
    
    def __init__(self, grid, outputs, debug):
        """
            Initilaizes a linear regression object

            :param grid: used grid for calibration
            :param outputs: flag to enable optional outputs
            :param debug: flag to enable debug outputs
        """
        self.grid = grid
        self.debug = debug
        self.outputs = outputs
        self.reg = None
        self.score_obj = score(grid, outputs, debug)
 
    def fit(self, X, y):
        """
            Learns the weight and bias for the given probabilities and labels

            :param X: Probabilities
            :param y: Differnce between label and confidence

            :return: LR object
        """
        self.reg = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        """
            Uses the learned weights to predict the needed adjustments

            :param X: Probabilities

            :return: list of adjustments (or deltas)
        """
        return self.reg.predict(X)