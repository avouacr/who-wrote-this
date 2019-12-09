"""
Build database for the Data Camp challenge using books from the Gutenberg project.
"""

import os
from numpy.random import randint, seed
import pandas as pd

from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers


def sample_paragraphs(book_id, author, n_parag, min_length):
    """Get book as text file and randomly sample a fixed number of paragraphs."""
    # Remove leading/trailing spaces
    book = strip_headers(load_etext(book_id)).strip()
    # Split paragraphs
    parag = book.split('\n\n')
    # Remove single line breaks
    parag = [x.replace('\n', ' ') for x in parag]
    # Remove paragraphs below a certain length
    parag = [p for p in parag if len(p) > min_length]
    # Exclude first/last 10 parag from sampling as they may contain remaining metadata
    parag = parag[10:-10]

    # Sample paragraphs
    seed(42)
    sample_ind = randint(0, len(parag), n_parag)

    if n_parag > len(parag):
        raise ValueError('The number of paragraphs to sample is higher than the '
                         'total number of paragraphs.')
    else:
        parag_sampled = [parag[i] for i in sample_ind]

    return zip(parag_sampled, [author]*len(parag_sampled))


def build_db(books_ref, n_parag, min_length):
    """Build dataframe with couples (paragraph, author) from the input book references."""
    all_parags = []
    for book_id, author in books_ref:
        parags = sample_paragraphs(book_id=book_id, author=author,
                                   n_parag=n_parag, min_length=min_length)
        for p in parags:
            all_parags.append(p)

    return pd.DataFrame(all_parags, columns=['paragraph', 'author'])


def export_db(df, output_dir='data', sep='|'):
    """Export database as .csv file."""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    filename = 'complete_db.csv'
    output_file = os.path.join(output_dir, filename)
    df.to_csv(output_file, sep=sep, index=False)
    print('Complete database saved as : ' + output_file)


if __name__ == '__main__':

    # List of references to sample. Format : (book_id, author)
    # book_id is the ID of the book in the Gutenberg project database
    books = [(5711, 'Zola'), (10775, 'Maupassant'), (55860, 'Balzac')]

    # Build database
    n_parag=200
    df_complete = build_db(books, n_parag=n_parag, min_length=100)
    assert(df_complete.shape[0] == len(books)*n_parag)

    # Export database
    export_db(df_complete, output_dir='data', sep='|')
