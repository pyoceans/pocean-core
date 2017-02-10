#!/bin/bash
set -ev

if [ "$TRAVIS_PYTHON_VERSION" == "3.5" ]; then
    conda install --file docs/requirements.txt
    mkdocs build --clean --verbose
    doctr deploy --built-docs=_site --gh-pages-docs .
fi
