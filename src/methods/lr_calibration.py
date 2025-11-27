from sklearn.linear_model import LinearRegression, LogisticRegression
from tools.calibration_scores import Score

class LR_calibration:
    
    def __init__(self, grid, args, type:str):
        """
            Initilaizes a linear regression object

            :param grid: used grid for calibration
            :param args: command line arguments
        """
        self.grid = grid
        self.debug = args.debug
        self.outputs = args.print_info
        self.reg = None
        self.type = type
        self.score_obj = Score(grid, args)

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
        return self.reg.predict(X).astype(float)