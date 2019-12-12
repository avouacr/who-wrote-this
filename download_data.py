import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


DATA_URL = (
    "https://drive.google.com/uc?export=download&id=1c6x1w2KC3PDpNOsUMq9V_ASMpCmJg3or"
)
OUTPUT_NAME = "who_wrote_this_corpus.csv"


def main(output_dir="data"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_file = os.path.join(output_dir, OUTPUT_NAME)

    if os.path.exists(output_file):
        print("Data already downloaded, delete it if you want to download it again.")
        return

    print("Downloading from {} ...".format(DATA_URL))
    urlretrieve(DATA_URL, filename=output_file)
    print("=> File saved as {}".format(output_file))


if __name__ == "__main__":
    test = os.getenv("RAMP_TEST_MODE", 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()
