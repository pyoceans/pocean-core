#!/bin/bash
set -ev

if [ "$TRAVIS_PYTHON_VERSION" == "3.5" ]; then
    cd docs
    conda install --file requirements.txt
    sphinx-apidoc -M -f -o api ../pocean ../pocean/tests
    make html
    doctr deploy --built-docs=_site/html .
    cd ..
fi
