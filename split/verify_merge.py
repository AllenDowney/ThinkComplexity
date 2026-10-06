"""Checks for integrating a chapter's book text into its notebook.

Usage (from the repo root):

    python split/verify_merge.py 2

1. Code: every function and method in the old notebook (git HEAD) has the same
   AST in the new one, with %%add_method_to cells attached to their classes.
2. Text: every sentence of the extracted book text (chapNN_book.md) appears in
   the new notebook's markdown. Missing sentences are printed for review; they
   should match the edits log.
"""
import ast
import inspect
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).parents[1]
HERE = Path(__file__).parent


def cells(nb):
    return [(c['cell_type'], ''.join(c['source'])) for c in nb['cells']]


def normalize_docstrings(tree):
    """Clean docstring indentation, which changes when a method moves out of its class."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                doc = inspect.cleandoc(first.value.value)
                first.value.value = '\n'.join(l.rstrip() for l in doc.splitlines()).strip()
    return tree


def functions(nb):
    """Map 'Class.method' or 'function' to the AST dump of its definition."""
    result = {}
    for kind, src in cells(nb):
        if kind != 'code':
            continue
        lines = [l for l in src.splitlines() if not l.lstrip().startswith('!')]
        target = None
        if lines and lines[0].startswith('%%add_method_to'):
            target = lines[0].split()[1]
            lines = lines[1:]
        elif any(l.lstrip().startswith('%') for l in lines):
            lines = [l for l in lines if not l.lstrip().startswith('%')]
        tree = normalize_docstrings(ast.parse(textwrap.dedent('\n'.join(lines))))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        result[f'{node.name}.{item.name}'] = ast.dump(item)
            elif isinstance(node, ast.FunctionDef):
                name = f'{target}.{node.name}' if target else node.name
                result[name] = ast.dump(node)
    return result


def sentences(text):
    """Normalized sentences of markdown text."""
    text = re.sub(r'```.*?```', ' ', text, flags=re.S)   # code
    text = re.sub(r'^#+ .*$', ' ', text, flags=re.M)      # headings
    text = re.sub(r'^:::.*$', ' ', text, flags=re.M)      # pandoc divs
    text = re.sub(r'\[\^\d+\]:?', ' ', text)              # footnote marks
    text = re.sub(r'[*`>]', '', text)                     # emphasis, code, quotes
    text = re.sub(r'^\s*(-|\d+\.)\s+', '', text, flags=re.M)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r' ([.,;:])', r'\1', text)              # space left by a footnote mark
    parts = re.split(r'(?<=[.!?:;])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def main(number):
    name = f'chap{number:02d}'
    old = json.loads(subprocess.run(['git', 'show', f'HEAD:soln/{name}.ipynb'], cwd=REPO,
                                    capture_output=True, text=True, check=True).stdout)
    new = json.loads((REPO / 'soln' / f'{name}.ipynb').read_text())

    print('== Code')
    f_old, f_new = functions(old), functions(new)
    missing = sorted(set(f_old) - set(f_new))
    added = sorted(set(f_new) - set(f_old))
    changed = sorted(k for k in set(f_old) & set(f_new) if f_old[k] != f_new[k])
    print(f'{len(f_old)} functions/methods before, {len(f_new)} after')
    for label, names in [('missing', missing), ('added', added), ('changed', changed)]:
        print(f'  {label}: {names or "none"}')

    print('\n== Text')
    book = (HERE / f'{name}_book.md').read_text()
    notebook = ' '.join(src for kind, src in cells(new) if kind == 'markdown')
    haystack = ' '.join(sentences(notebook))
    book_sentences = sentences(book)
    absent = [s for s in book_sentences if s not in haystack]
    print(f'{len(book_sentences)} sentences in the book text; {len(absent)} not found verbatim:')
    for s in absent:
        print('  -', s[:150])


if __name__ == '__main__':
    main(int(sys.argv[1]))
