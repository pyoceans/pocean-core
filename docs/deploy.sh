#!/bin/bash
set -ev

cd docs
conda install --file requirements.txt
sphinx-apidoc -M -f -o api ../pocean ../pocean/tests
make html
cd ..
doctr deploy . --built-docs docs/_site/html
