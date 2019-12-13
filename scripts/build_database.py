"""
Build database for the Data Camp challenge using books from the Gutenberg project.
"""

import os
from numpy.random import randint, seed

from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers


def sample_paragraphs(book_id, n_parag, min_length):
    """Get book as text file and randomly sample a fixed number of paragraphs."""
    # Get book as string and emove metadata
    book = load_etext(book_id)
    # Remove metadata
    book = strip_headers(book).strip()
    # Remove the character we'll choose as separator
    book = book.replace("|", " ")
    # Split paragraphs
    parag = book.split("\n\n")
    # Remove single line breaks
    parag = [x.replace("\n", " ") for x in parag]
    # Remove paragraphs below a certain length
    parag = [p for p in parag if len(p) > min_length]
    # Exclude first/last 10 parag from sampling as they may contain remaining metadata
    parag = parag[10:-10]

    # Sample paragraphs
    seed(42)
    sample_ind = randint(0, len(parag), n_parag)

    if n_parag is not None:
        if n_parag > len(parag):
            raise ValueError(
                "The number of paragraphs to sample is higher than the "
                "total number of paragraphs."
            )
        else:
            parag_sampled = [parag[i] for i in sample_ind]

    else:
        # If n_parag is None, all paragraphs are sampled
        parag_sampled = parag

    return parag_sampled


def build_db(books_ref, n_parag, min_length, file_name, output_dir="data", sep="|"):
    """Build database from book references and export as .csv."""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_file = os.path.join(output_dir, file_name)
    with open(output_file, "w") as f:
        f.write("paragraph" + sep + "author" + "\n")

    for author, book_id in books_ref:
        parags = sample_paragraphs(
            book_id=book_id, n_parag=n_parag, min_length=min_length
        )
        with open(output_file, "a") as f:
            for p in parags:
                f.write(p + sep + author + "\n")


if __name__ == "__main__":
    # Get list of references to sample. Format : (book_id, author)
    # book_id is the ID of the book in the Gutenberg project database
    with open("data/meta/book_references.txt") as f:
        book_refs = f.read().splitlines()
    book_refs = [(x.split(",")[0], int(x.split(",")[1])) for x in book_refs]

    # Build complete database and complete vocab
    build_db(
        book_refs, n_parag=None, min_length=100,
        file_name="who_wrote_this_corpus_complete.csv"
    )

    # Build small database
    build_db(
        book_refs,
        n_parag=200,
        min_length=100,
        file_name="who_wrote_this_corpus_small.csv",
    )
