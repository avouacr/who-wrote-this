import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


DATA_URLS = {
    "who_wrote_this_corpus_complete.csv": "https://drive.google.com/uc?export=download&id=1c6x1w2KC3PDpNOsUMq9V_ASMpCmJg3or",
    "who_wrote_this_corpus_small.csv": "https://drive.google.com/uc?export=download&id=1pzF0xsIEeknAJjovi4iNXFcJKzlsxWFR",
}


def main(output_dir="data"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for file_name, url in DATA_URLS.items():
        output_file = os.path.join(output_dir, file_name)
        print(f"Downloading {output_file}")

        if os.path.exists(output_file):
            print(
                "Data already downloaded, delete it if you want to download it again."
            )
            continue

        print("Downloading from {} ...".format(url))
        urlretrieve(url, filename=output_file)
        print("=> File saved as {}".format(output_file))


if __name__ == "__main__":
    test = os.getenv("RAMP_TEST_MODE", 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()
