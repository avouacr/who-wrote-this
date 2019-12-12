import argparse

from functools import reduce

import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import json


# Custom stopwords
CUSTOM_STOPWORDS = ["--", ".", ",", "!", ";", "’", ":", "?", "...", "'", "«", "»"]


## Help function for reduce
def flatten_count(accumulator, items):
    for item in items:
        accumulator[item] = accumulator.get(item, 0) + 1
    return accumulator


# Create a function that removes given stopwords from a list of words
def make_stopwords_remover(stopwords):
    def stopwords_remover(words):
        return [word for word in words if word not in stopwords]

    return stopwords_remover


def tokenize_data(df, exclude_stopwords=True):
    # Prevent side effect
    df = df.copy()

    # Load stopwords
    try:
        nltk.data.find("corpora/stopwords.zip")
    except LookupError:
        nltk.download("stopwords")

    french_stopwords = set(stopwords.words("french")).union(CUSTOM_STOPWORDS)

    # Tokenize paragraphs
    df["tokenized"] = df["paragraph"].str.lower().map(word_tokenize)
    if exclude_stopwords:
        df["tokenized"] = df["tokenized"].map(make_stopwords_remover(french_stopwords))

    # Build vocabulary (with word count)
    words_count = reduce(flatten_count, df["tokenized"].values.tolist(), {})

    return df, words_count


def main(data_path, df_output_path, dictionary_output_path, exclude_stopwords):
    df = pd.read_csv(data_path, sep="|")
    df, words_count = tokenize_data(df, exclude_stopwords=exclude_stopwords)

    with open(dictionary_output_path, "w") as f:
        json.dump(words_count, f)

    df.to_csv(df_output_path, index=False)


def cli():
    # Load dataset
    arg_parser = argparse.ArgumentParser(description="Tokenizer CLI")

    arg_parser.add_argument("data_path", type=str)
    arg_parser.add_argument("df_output_path", type=str)
    arg_parser.add_argument("dictionary_output_path", type=str)
    arg_parser.add_argument("--exclude_stopwords", type=bool, default=True)

    args = arg_parser.parse_args()
    data_path = args.data_path
    df_output_path = args.df_output_path
    dictionary_output_path = args.dictionary_output_path
    exclude_stopwords = args.exclude_stopwords

    main(data_path, df_output_path, dictionary_output_path, exclude_stopwords)


if __name__ == "__main__":
    cli()
