from sklearn.linear_model import LinearRegression, LogisticRegression
from tools.calibration_scores import score

class Platt_calibration:
    
    def __init__(self, grid, outputs, debug):
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
        self.platt = None

        self.score_obj = score(grid, outputs, debug)
 
    def fit(self, X, y):
        """
            Learns the weight and bias for the given probabilities and labels

            :param X: Probabilities
            :param y: Differnce between label and confidence

            :return: LR object
        """
        X = X.values.reshape(-1,1)
        self.platt = LogisticRegression(solver='lbfgs').fit(X,y) 
        return self

    def predict(self, X):
        """
            Uses the learned weights to predict the needed adjustments

            :param X: Probabilities

            :return: list of calibrated probabilities
        """
        X = X.values.reshape(-1,1)
        return self.platt.predict_proba(X)[:,1]
    
    
    