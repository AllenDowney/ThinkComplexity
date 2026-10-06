# sudo apt install pandoc
# pip install notedown

python split.py book.tex

for TEXFILE in chap*.tex
do
    MDFILE=${TEXFILE%.tex}.md
    IPYNBFILE=${TEXFILE%.tex}.ipynb

    echo "pandoc --atx-headers $TEXFILE -t markdown > $MDFILE"
    pandoc --markdown-headings=atx $TEXFILE -t markdown > $MDFILE

    echo "notedown $MDFILE > $IPYNBFILE"
    notedown $MDFILE  > $IPYNBFILE
    sed -i 's/::: exercise/### Exercise/g' $IPYNBFILE
    sed -i 's/":::/"/g' $IPYNBFILE
    sed -i 's/:   //g' $IPYNBFILE
done
