import pooch
import zipfile
from pathlib import Path

def download_test_data():
    url = "https://github.com/pyoceans/pocean-core/releases/download"
    version = "d1.0.0"

    fname = pooch.retrieve(
        url=f"{url}/{version}/test_data.zip",
        known_hash="sha256:28be36e8e0ec90a8faf5ee3f4d52638bdeb2cbcbeac8c823de680cf84aa34940",
    )

    here = Path(__file__).resolve().parent
    print(fname)
    print(here)
    with zipfile.ZipFile(fname, "r") as zip_ref:
        zip_ref.extractall(here)

if __name__ == "__main__":
    download_test_data()