from sklearn.linear_model import LogisticRegression
from tools.calibration_scores import Score

class Platt_calibration:
    
    def __init__(self, bins, args):
        """
            Initilaizes a linear regression object

            :param bins: used binning object for calibration
            :param args: command line args
        """
        self.bins = bins
        self.debug = args.debug
        self.outputs = args.print_info
        self.platt = None

        self.score_obj = Score(bins, args)
 
    def fit(self, X, y):
        """
            Learns the weight and bias for the given probabilities and labels

            :param X: Probabilities
            :param y: Differnce between label and confidence

            :return: LR object
        """
        X = X.reshape(-1,1)
        self.platt = LogisticRegression(solver='lbfgs').fit(X,y) 
        return self

    def predict(self, X):
        """
            Uses the learned weights to predict the needed adjustments

            :param X: Probabilities

            :return: list of calibrated probabilities
        """
        X = X.reshape(-1,1)
        return self.platt.predict_proba(X)[:,1]
    
    
    