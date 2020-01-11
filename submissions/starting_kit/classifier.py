
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


class Classifier(BaseEstimator):
    def __init__(self):
        self.classifier = LogisticRegression(solver='lbfgs', max_iter=1000,
                                             multi_class='multinomial')

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.classifier.predict(X).astype(int)
        return y_pred

    def predict_proba(self, X):
        proba_pred = self.classifier.predict_proba(X)
        return proba_pred
