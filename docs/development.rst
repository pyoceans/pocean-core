Development
============

Create a conda environment

.. code-block:: bash

    conda create --name pocean310 python=3.10 --file requirements.txt --file requirements-dev.txt
    conda activate pocean310

Running tests
-------------

.. code-block:: bash

    # download test datasets
    cd pocean/tests
    python download_test_data.py

    # run test suite
    pytest
