# Build the notebook folder and zip file

# copy notebooks from soln
cp ../soln/app*.ipynb .
cp ../soln/chap*.ipynb .
cp ../soln/*.py .

# remove the solutions
python prep_notebooks.py

# pip install pytest nbmake

# run nbmake
# pytest --nbmake chap*.ipynb

# commit the changes
git add app*.ipynb
git add chap*.ipynb
git add *.py
git commit -m "Updating code and notebooks"

# build the zip file
cd ../..; zip -r ThinkComplexity.zip \
    ThinkComplexity/nb/app*.ipynb \
    ThinkComplexity/nb/chap*.ipynb \
    ThinkComplexity/nb/*.py

# add and commit it
mv ThinkComplexity.zip ThinkComplexity
cd ThinkComplexity

git add ThinkComplexity.zip
git commit -m "Updating the zip file"
git push
