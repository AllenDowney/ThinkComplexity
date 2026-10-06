# Requires jupyter-book<2 and ghp-import (requirements-dev.txt);
# _config.yml and _toc.yml target Jupyter Book 1.

# Build the Jupyter book version

# copy the notebooks
cp ../soln/*.ipynb .

# add tags to hide the solutions
python prep_notebooks.py

# build the HTML version; stale pages persist unless _build is removed
rm -rf _build
jb build .

# push it to GitHub
ghp-import -n -p -f _build/html
