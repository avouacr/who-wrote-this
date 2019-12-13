
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vocabulary = None
        self.vectorizer = TfidfVectorizer(strip_accents='ascii',
                                          max_df=0.7)

    def fit(self, X_df, y=None):
        self.vectorizer.fit(X_df['paragraph'])
        return self

    def transform(self, X_df):
        X_preprocessed = self.vectorizer.transform(X_df['paragraph'])
        return X_preprocessed
