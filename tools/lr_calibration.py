from sklearn.linear_model import LinearRegression, LogisticRegression
from tools.calibration_scores import score

class LR_calibration:
    
    def __init__(self, grid, outputs, debug, type:str):
        """
            Initilaizes a linear regression object

            :param grid: used grid for calibration
            :param outputs: flag to enable optional outputs
            :param debug: flag to enable debug outputs
            :param type: Regression type. One of 'linear' and 'logistic'
        """
        self.grid = grid
        self.debug = debug
        self.outputs = outputs
        self.reg = None
        self.type = type
        self.score_obj = score(grid, outputs, debug)
 
    def fit(self, X, y):
        """
            Learns the weight and bias for the given probabilities and labels

            :param X: Probabilities
            :param y: Differnce between label and confidence

            :return: LR object
        """
        self.reg = LinearRegression().fit(X, y) if self.type == 'linear' else LogisticRegression().fit(X,y) 
        return self

    def predict(self, X):
        """
            Uses the learned weights to predict the needed adjustments

            :param X: Probabilities

            :return: list of adjustments (or deltas)
        """
        return self.reg.predict(X)