import argparse
import json
from collections import Counter
from functools import reduce

import nltk
import numpy as np
import pandas as pd
import regex as re  # needed as re does not allow infinite lookbacks
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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


def extract_proper_noun(txt):
    return re.findall(r"(?<!^|\. |\.  )[A-Z][a-z]+", txt)


def tokenize_data(df, exclude_stopwords=True, exclude_proper_nouns=True, proper_noun_thresh=7):
    # Prevent side effect
    df = df.copy()

    if exclude_stopwords:
        # Load stopwords
        try:
            nltk.data.find("corpora/stopwords.zip")
        except LookupError:
            nltk.download("stopwords")

        french_stopwords = set(stopwords.words("french")).union(CUSTOM_STOPWORDS)

    if exclude_proper_nouns:
        # Get proper nouns (persons, places).
        # We get all names starting with a capital letter and that are not at the
        # beginning of a sentence. We then remove from the corpus all those that
        # appear more than proper_noun_thresh times.
        proper_nouns = df.paragraph.apply(extract_proper_noun).apply(pd.Series).stack()
        counted = Counter(proper_nouns)
        filtered_proper_nouns = [el for el in proper_nouns if counted[el] >= proper_noun_thresh]
        filtered_proper_nouns = np.unique(filtered_proper_nouns)
        filtered_proper_nouns = [x.lower() for x in filtered_proper_nouns]

    if (exclude_stopwords & exclude_proper_nouns):
        words_to_remove = set(french_stopwords).union(set(filtered_proper_nouns))
    elif exclude_stopwords:
        words_to_remove = french_stopwords
    elif exclude_proper_nouns:
        words_to_remove = filtered_proper_nouns

    # Tokenize paragraphs
    df["tokenized"] = df["paragraph"].str.lower().map(word_tokenize)

    if (exclude_stopwords | exclude_proper_nouns):
        df["tokenized"] = df["tokenized"].map(make_stopwords_remover(words_to_remove))

    # Build vocabulary (with word count)
    words_count = reduce(flatten_count, df["tokenized"].values.tolist(), {})

    return df, words_count


def main(data_path, df_output_path, dictionary_output_path, exclude_stopwords):
    df = pd.read_csv("/home/jaime/who-wrote-this/data/who_wrote_this_corpus_small.csv", sep="|")
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
    arg_parser.add_argument("--exclude_proper_nouns", type=bool, default=True)
    arg_parser.add_argument("--proper_noun_thresh", type=int, default=7)

    args = arg_parser.parse_args()
    data_path = args.data_path
    df_output_path = args.df_output_path
    dictionary_output_path = args.dictionary_output_path
    exclude_stopwords = args.exclude_stopwords

    main(data_path, df_output_path, dictionary_output_path, exclude_stopwords)


if __name__ == "__main__":
    cli()
