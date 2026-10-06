import nbformat as nbf
from glob import glob


def process_cell(cell):
    # get tags
    tags = cell['metadata'].get('tags', [])

    # add hide-cell tag to solutions
    if cell['cell_type'] == 'code':
        source = cell['source']

        # remove solutions
        if source.startswith('# Solution') or 'solution' in tags:
            cell['source'] = []

        # remove %%expect cell magic
        if source.startswith('%%expect'):
            t = source.split('\n')[1:]
            cell['source'] = '\n'.join(t)

    # add reference label
    for tag in tags:
        if tag.startswith('chapter') or tag.startswith('section'):
            print(tag)
            label = f'({tag})=\n'
            cell['source'] = label + cell['source']


def process_notebook(path):
    ntbk = nbf.read(path, nbf.NO_CONVERT)

    # add a hide-cell tag to the cell that defines download,
    # wherever it is in the setup cells
    for cell in ntbk.cells:
        if cell.source.startswith(r'from os.path import basename, exists'):
            tags = cell.metadata.get('tags', [])
            if 'hide-cell' not in tags:
                tags.append('hide-cell')
            cell.metadata['tags'] = tags

    # if the second element of ntbk.cells loads nb_black, remove it
    second = ntbk.cells[1]
    if second.source.startswith(r'%load_ext nb_black'):
        ntbk.cells.pop(1)

    for cell in ntbk.cells:
        process_cell(cell)

    nbf.write(ntbk, path)


# Collect a list of the notebooks in the content folder
paths = glob("*.ipynb")

for path in sorted(paths):
    print('prepping', path)
    process_notebook(path)
