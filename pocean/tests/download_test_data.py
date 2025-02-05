import zipfile
from pathlib import Path

import pooch


def download_test_data():
    url = "https://github.com/pyoceans/pocean-core/releases/download"
    version = "2025.01"

    fname = pooch.retrieve(
        url=f"{url}/{version}/test_data.zip",
        known_hash="sha256:41180c6bc6017de935250c9e8c1bbb407507049baebd767692c4f74fb8d662a8",
    )

    here = Path(__file__).resolve().parent
    print(fname)
    print(here)
    with zipfile.ZipFile(fname, "r") as zip_ref:
        zip_ref.extractall(here)


if __name__ == "__main__":
    download_test_data()
