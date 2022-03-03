Development
============

Create a conda environment

...code bash

    $ conda create -n pocean310 python=3.10
    $ source activate pocean310
    $ conda install --file requirements.txt --file requirements-dev.txt


Running tests
-------------

...code bash

    $ pytest
