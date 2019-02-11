Development
============

Create a conda environment

...code bash

    $ conda create -n pocean37 python=3.7
    $ source activate pocean37
    $ conda install --file requirements.txt --file requirements-test.txt


Running tests
-------------

...code bash

    $ pytest
