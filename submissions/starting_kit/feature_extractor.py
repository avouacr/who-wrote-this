import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from collections import Counter


def bag_of_word_vectorizer(paragraph, vocabulary=None):
    # default: empty vocabulary
    vocabulary = vocabulary or []
    # tokenize keeping only words in vocabulary
    tokens = [token for token in word_tokenize(paragraph) if token in vocabulary]
    # bag of word representation with word frequency
    tokens_count = Counter(tokens)
    n_tokens = len(tokens)
    bag_of_word_array = (
        np.array([tokens_count.get(word, 0) for word in vocabulary]) / n_tokens
    )
    return bag_of_word_array


CUSTOM_STOPWORDS = ["--", ".", ",", "!", ";", "’", ":", "?", "...", "'", "«", "»"]


class FeatureExtractor(object):
    def __init__(self):
        self.vocabulary = None
        self.column_transformer = None

    def fit(self, X_df, y):
        # build vocab
        self.vocabulary = set()
        X_df["paragraph"].map(
            lambda paragraph: map(
                lambda token: self.vocabulary.add(token), word_tokenize(paragraph)
            )
        )
        # remove stopwords from vocabulary
        ## load stopwords
        try:
            nltk.data.find("corpora/stopwords.zip")
        except LookupError:
            nltk.download("stopwords")
        french_stopwords = set(stopwords.words("french")).union(CUSTOM_STOPWORDS)
        self.vocabulary = self.vocabulary.difference(french_stopwords)

        # update column_transformer
        self.column_transformer = make_column_transformer(
            (
                FunctionTransformer(bag_of_word_vectorizer, self.vocabulary),
                ["paragraph"],
            )
        )

        return self

    def transform(self, X_df):
        return self.column_transformer.transform(X_df)
