import os
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OrdinalEncoder

import rampwf as rw
from rampwf.workflows import FeatureExtractorClassifier
from rampwf.score_types.classifier_base import ClassifierBaseScoreType

problem_title = "Who wrote this? Predicting the author of a paragraph"
_target_column_name = "author"
_prediction_label_names = list(range(0, 10))
Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)


# An object implementing the workflow
class WhoWroteThis(FeatureExtractorClassifier):
    def __init__(self, workflow_element_names=["feature_extractor", "classifier"]):
        super().__init__()
        self.element_names = workflow_element_names


workflow = WhoWroteThis()


# define the score (basic multiclass F1-score)
class F1Score(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1

    def __init__(self, name="F1-score", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average="micro")


score_types = [F1Score()]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=6, test_size=0.20, random_state=hash("UwU") % 1000)
    return cv.split(X, y)


def _read_data(path, f_name, sep="|"):
    data = pd.read_csv(os.path.join(path, "data", f_name), sep=sep, low_memory=False)
    y_array = OrdinalEncoder().fit_transform(
        data[_target_column_name].values[:, np.newaxis]
    )
    X_df = data.drop(columns=[_target_column_name])
    return X_df, y_array.flatten()


def get_train_data(sep="|", path="."):
    f_name = "who_wrote_this_corpus_train.csv"
    X_df, y_array = _read_data(path, f_name, sep=sep)
    return X_df, y_array


def get_test_data(sep="|", path="."):
    f_name = "who_wrote_this_corpus_test.csv"
    X_df, y_array = _read_data(path, f_name, sep=sep)
    return X_df, y_array
