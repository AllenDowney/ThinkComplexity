"""Extract one chapter of split/book.tex as Markdown, for merging into a notebook.

Usage (from the repo root):

    python split/tex2md.py 2 > split/chap02_book.md

The mechanical part only. It leaves markers where a person has to decide what
to write, so they are easy to find:

    FIGURE[name]: caption    where a figure was; the cell that draws it goes here
    REF[label]               a cross-reference (Figure~, Section~, Chapter~)

Index entries are dropped; they can be recovered from book.tex if needed.
"""
import re
import subprocess
import sys
from pathlib import Path

BOOK = Path(__file__).parent / 'book.tex'

# braces with one level of nesting, as in \index{Evolution@{\em Evolution}}
BRACES = r'\{(?:[^{}]|\{[^{}]*\})*\}'


def chapter_source(number):
    """Returns the LaTeX for the given chapter, numbered as in the notebooks.

    The Preface is chapter 0, so chapter 12 is the 13th \\chapter.
    """
    text = BOOK.read_text()
    starts = [m.start() for m in re.finditer(r'^\\chapter\{', text, re.M)]
    start = starts[number]
    end = starts[number + 1] if number + 1 < len(starts) else len(text)
    return text[start:end].split('\\appendix')[0]


def figure(match):
    body = match.group(0)
    name = re.search(r'figs/([\w-]+)', body).group(1)
    caption = re.search(r'\\caption\{(.*?)\}\s*\\label', body, re.S)
    caption = caption.group(1).strip() if caption else ''
    return f'\n\nFIGURE[{name}]: {caption}\n\n'


def preprocess(tex):
    tex = re.sub(r'\\index' + BRACES + r'\n?', '', tex)
    # LaTeX quotes: ``like this'' (or ``like this" in places)
    tex = tex.replace("``", '"').replace("''", '"')
    tex = re.sub(r'\\py\{([^}]*)\}',
                 lambda m: '\\texttt{' + m.group(1).replace('_', '\\_') + '}', tex)
    # lstlisting with a language becomes a fenced, labeled code block
    tex = re.sub(r'\\begin\{code\}', r'\\begin{lstlisting}[language=Python]', tex)
    tex = re.sub(r'\\begin\{stdout\}', r'\\begin{lstlisting}[language=text]', tex)
    tex = re.sub(r'\\end\{(code|stdout)\}', r'\\end{lstlisting}', tex)
    tex = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', figure, tex, flags=re.S)
    tex = re.sub(r'(Figure|Section|Chapter)~\\ref\{([^}]*)\}', r'\1 REF[\2]', tex)
    tex = re.sub(r'\\ref\{([^}]*)\}', r'REF[\1]', tex)
    return tex


def to_markdown(tex):
    result = subprocess.run(
        ['pandoc', '-f', 'latex-smart',
         '-t', 'markdown-smart',
         '--wrap=none'],
        input=tex, capture_output=True, text=True, check=True)
    md = result.stdout
    md = md.replace(r'REF\[', 'REF[').replace(r'FIGURE\[', 'FIGURE[').replace(r'\]', ']')
    # pandoc's LaTeX reader curls apostrophes and quotes even with -smart
    md = md.replace('\u2019', "'").replace('\u2018', "'")
    md = md.replace('\u201c', '"').replace('\u201d', '"')
    # drop heading anchors like {#prisoners}
    md = re.sub(r'^(#+ .*?) \{#[\w-]+\}$', r'\1', md, flags=re.M)
    # jupytext makes "``` python" blocks into code cells; "``` text" stays markdown
    md = re.sub(r'^``` \{\.(python|text) language="\w+"\}$', r'```\1', md, flags=re.M)
    return md


if __name__ == '__main__':
    print(to_markdown(preprocess(chapter_source(int(sys.argv[1])))))
