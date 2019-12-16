
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


class Regressor(BaseEstimator):
    def __init__(self):
        self.regressor = LogisticRegression(solver='lbfgs', multi_class='multinomial')

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.regressor.predict(X).astype(int)
        y_pred_ohe = np.zeros((y_pred.size, y_pred.max()+1))
        y_pred_ohe[np.arange(y_pred.size), y_pred] = 1
        return y_pred_ohe
