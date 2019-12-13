import os

import pandas as pd
import numpy as np

import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType

from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OrdinalEncoder

problem_title = "Who wrote this? Predicting the author of a paragraph"
_target_column_name = "author"
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow


class WhoWroteThis(FeatureExtractorRegressor):
    def __init__(
        self, workflow_element_names=["feature_extractor", "regressor",],
    ):
        super().__init__()
        self.element_names = workflow_element_names


workflow = WhoWroteThis()


# define the score (basic multiclass F1-score)
class F1Score(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="F1-score"):
        self.name = name

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        return f1_score(y_true, y_pred, average='micro')


score_types = [
    F1Score(),
]


# untested
def get_cv(X, y):
    cv = ShuffleSplit(n_splits=6, test_size=0.20, random_state=hash("UwU") % 1000)
    return cv.split(X, y)


# untested
def _read_data(path, f_name, sep='|'):
    data = pd.read_csv(os.path.join(path, "data", f_name), sep=sep, low_memory=False)
    y_array = OrdinalEncoder().fit_transform(data[_target_column_name].values[:,np.newaxis])
    X_df = data.drop(columns=[_target_column_name])
    return X_df, y_array.flatten()


# untested
def get_train_data(sep='|', path="."):
    f_name = "who_wrote_this_corpus_complete.csv"
    X_df, y_array = map(lambda x: x[: int(len(x) / 0.8)], _read_data(path, f_name,
                                                                     sep=sep))
    return X_df, y_array


# untested
def get_test_data(path="."):
    f_name = "who_wrote_this_corpus_small.csv"
    X_df, y_array = map(lambda x: x[int(len(x) / 0.8) :], _read_data(path, f_name))
    return X_df, y_array
