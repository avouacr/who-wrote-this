import numpy as np
from scipy.sparse import csr_matrix, vstack

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from collections import Counter


CUSTOM_STOPWORDS = ["--", ".", ",", "!", ";", "’", ":", "?", "...", "'", "«", "»"]


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vocabulary = None
        self.column_transformer = None

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        # build vocab
        self.vocabulary = set()
        tokens_arrays = [word_tokenize(p) for p in X_df["paragraph"]]
        for p_tokens in tokens_arrays:
            for token in p_tokens:
                self.vocabulary.add(token)
        # remove stopwords from vocabulary
        ## load stopwords
        try:
            nltk.data.find("corpora/stopwords.zip")
        except LookupError:
            nltk.download("stopwords")
        french_stopwords = set(stopwords.words("french")).union(CUSTOM_STOPWORDS)
        self.vocabulary = self.vocabulary.difference(french_stopwords)

        def bag_of_word_vectorizer(x, vocabulary=None):
            # default: empty vocabulary
            vocabulary = vocabulary or []
            inter_bow_arrays = []
            inter_bow_sparse = []
            for i, paragraph in enumerate(x):
                # tokenize keeping only words in vocabulary
                tokens = [token for token in word_tokenize(paragraph) if token in vocabulary]
                # bag of word representation with word frequency
                tokens_count = Counter(tokens)
                n_tokens = len(tokens)
                bag_of_word_array = (
                        np.array([tokens_count.get(word, 0) for word in vocabulary]) / n_tokens
                )
                inter_bow_arrays.append(bag_of_word_array)

                if (i % 500 == 0) or (i == len(x) - 1):
                    inter_bow_sparse.append(csr_matrix(np.array(inter_bow_arrays)))
                    inter_bow_arrays = []

                sparse_out = vstack(inter_bow_sparse)

            return sparse_out

        X_preprocessed = bag_of_word_vectorizer(x=X_df['paragraph'],
                                                vocabulary=self.vocabulary)

        return X_preprocessed
