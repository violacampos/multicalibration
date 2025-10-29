from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from tools.calibration_scores import Score
from sklearn.linear_model import LinearRegression

class regressor_calibration:
    
    def __init__(self, grid, args):
        """
            Initilaizes a regressor object

            :param grid: used grid for calibration
            :param args: passed commandline parameter
            :param outputs: flag to enable optional outputs
            :param debug: flag to enable debug outputs
        """
        self.grid = grid
        self.debug = args.debug
        self.outputs = args.print_info
        self.reg = None
        self.regressor = args.regressor
        self.score_obj = Score(grid, args)
 
    def fit(self, X, y):
        """
            Uses the selected regressor for training

            :param X: probabailities
            :param y: difference between label and confidence

            :return: regressor object
        """
        if self.regressor == 'LR':
            self.reg = LinearRegression().fit(X, y)
        elif self.regressor == 'SVR':
            self.reg = SVR(kernel='rbf').fit(X, y)
        elif self.regressor == 'XGBoost':
            self.reg = XGBRegressor().fit(X, y)
        return self

    def predict(self, X):
        """
            Uses the learned regressor to predict the adjustments

            :param X: probabailities

            :return: adjustments for calibration
        """
        return self.reg.predict(X)