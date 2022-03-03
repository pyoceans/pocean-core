#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No version specified, exiting"
    exit 1
fi

# Set version to release
sed -i "s/^__version__ = .*/__version__ = \"$1\"/" pocean/__init__.py
sed -i "s/version = .*/version = \"$1\"/" docs/conf.py
sed -i "s/release = .*/release = \"$1\"/" docs/conf.py
echo $1 > VERSION

# Commit release
git add pocean/__init__.py
git add VERSION
git add docs/conf.py
git commit -m "Release $1"

# Tag
git tag $1

# Push to Git
echo "git push --tags origin main"
