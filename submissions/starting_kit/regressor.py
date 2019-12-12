from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


class Regressor(BaseEstimator):
    def __init__(self):
        self.regressor = None

    def fit(self, X, y):
        self.regressor = LogisticRegression(multi_class=True)
        self.regressor.fit(X, y)

        return self

    def predict(self, X):
        y_pred = self.regressor.predict(X)
        return y_pred
