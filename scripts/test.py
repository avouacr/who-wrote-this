
import problem

X_df, y_array = problem.get_train_data(sep='|')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(strip_accents='ascii',
                                          max_df=0.7)

    def fit(self, X_df, y=None):
        self.vectorizer.fit(X_df['paragraph'])
        return self

    def transform(self, X_df):
        X_preprocessed = self.vectorizer.transform(X_df['paragraph'])
        return X_preprocessed


feature_extractor = FeatureExtractor()

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.clf.predict(X).astype(int)
        y_pred_ohe = np.zeros((y_pred.size, y_pred.max() + 1))
        y_pred_ohe[np.arange(y_pred.size), y_pred] = 1
        return y_pred


classifier = Classifier()



from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

clf = Pipeline(steps=[
    ('feature_extractor', feature_extractor),
    ('classifier', classifier)])

cv = problem.get_cv(X_df, y_array)

scores_Xdf = cross_val_score(clf, X_df, y_array, cv=cv, scoring='f1_micro', n_jobs=3)

print("mean: %e (+/- %e)" % (scores_Xdf.mean(), scores_Xdf.std()))