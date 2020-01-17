import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

REPO_REV_URL = "https://gist.githubusercontent.com/GuillaumeDesforges/e9c108915285a021093594ac932550f8/raw/31997e390f0808445ab79a726d1181c1643be7d7"

FILE_NAMES = ["who_wrote_this_corpus_train.csv", "who_wrote_this_corpus_test.csv"]

DATA_URLS = {file_name: f"{REPO_REV_URL}/{file_name}" for file_name in FILE_NAMES}


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
