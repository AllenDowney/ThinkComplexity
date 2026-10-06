# Converting chapters from book.tex to notebooks

In v3 the notebooks in `soln/` are the canonical source of each chapter. This
directory holds the 2nd edition's LaTeX (`book.tex`) and the tools for merging
a chapter's text into its notebook. The process was worked out on chapter 12
in ThinkComplexity2 (its issue #66), and `soln/chap12.ipynb` is the model.

## 1. Extract

```bash
python split/tex2md.py 2 > split/chap02_book.md
```

`tex2md.py` slices the chapter out of `book.tex` (the Preface is chapter 0),
rewrites the book's macros, and runs pandoc. The result is Markdown with
straight quotes, fenced `python` code blocks and `[^n]` footnotes; `\index`
entries are dropped. Anything that needs a decision is left as a marker:
`FIGURE[name]: caption` where a figure was, and `REF[label]` for each
cross-reference. The extract is generated, so it is not tracked.

## 2. Merge, in jupytext Markdown

Export the notebook (`jupytext --to md soln/chap02.ipynb`), walk the book text
section by section, and put the prose around the notebook's code cells. Then
`jupytext --update --to ipynb soln/chap02.md`, which keeps the outputs.

- **The notebook's code is canonical.** The book's code blocks are display
  copies, usually excerpts with docstrings trimmed. Delete the book's copy and
  keep the cell.
- **Method-by-method walk-throughs** use `%%add_method_to ClassName` (defined
  in `utils.py`): a first cell with the `class` line, class attributes and
  `__init__`, then one cell per method the book discusses, next to its prose.
  This doesn't work for methods that use zero-argument `super()`.
- **Code defined in another chapter**: don't redefine it. Refer to it, or show
  it as plain text; a fenced `python` block becomes a code cell.
- **Figures**: delete the `FIGURE[...]` marker and make sure the cell that
  draws the figure sits there. Fold the caption into the sentence that
  introduces it: "The following figure shows..." before the cell, "The
  previous figure shows..." after it.
- **Cross-references**: `REF[...]` into another chapter becomes "Chapter N".
  The paragraph saying "The code for this chapter is in `chapNN.ipynb`" goes.
- **Footnotes**: classic Jupyter doesn't render `[^n]`; make them inline
  parentheticals or links.
- **Notes that duplicate the book**: keep the book's wording. Notebook-only
  material (extra checks, exercise solutions) stays.
- **Exercises** go where the notebook has them. Solution cells start with
  `# Solution`, so `nb/prep_notebooks.py` blanks them for students.
- **Content stays as it is** during the merge; corrections are separate commits.

## 3. Layout

As in ThinkStats (e.g. `~/ThinkStats/nb/chap11.ipynb`); see `soln/chap12.ipynb`.

1. **Promo cell**: the book is available from
   [Bookshop.org](https://bookshop.org/a/98697/9781492040200) and
   [Amazon](https://amzn.to/3wPN0SJ) (affiliate links), and the
   buy-me-a-coffee line.
2. **Title and introduction**: `# Chapter title` and the chapter's
   introductory text, before any code.
3. **Colab link** to the solution notebook,
   `https://colab.research.google.com/github/AllenDowney/ThinkComplexity/blob/v3/soln/chapNN.ipynb`.
   `nb/prep_notebooks.py` rewrites it to `nb/` in the student copy.
4. **Setup cells**: imports, `download`, `utils`.
5. **Copyright cell** at the end: book title linked to the book's site,
   copyright, MIT license for code, CC BY-NC-SA 4.0 for text.

Split long text cells: every section heading starts a new markdown cell (set
`split_at_heading=true` in the notebook's jupytext metadata), and long runs of
prose get split with two blank lines. One blank line does not split a cell.

Notebooks that are nbformat 4.1/4.2 must be upgraded to 4.5 after cells are
added, with an `id` on every cell, or Jupyter reports a validation error.
Check with `nbformat.validate`.

## 4. Verify

```bash
python split/verify_merge.py 2
```

`verify_merge.py` compares the notebook with its committed version (git
`HEAD`): every function and method must have the same AST, with
`%%add_method_to` cells attached to their classes. Then it checks that every
sentence of `split/chapNN_book.md` appears in the notebook's markdown, and
prints the ones that don't; that list should match the intended edits.

Then run the notebook (`pytest --nbmake soln/chapNN.ipynb`, or leave the long
ones to CI), regenerate the student copy (the copy and `prep_notebooks.py`
steps of `nb/build.sh`), and commit one chapter per commit.
