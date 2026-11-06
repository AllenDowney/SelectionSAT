"""Check the notebooks without running them.

For every notebook at the top level: join the code cells and check that the
Python parses, and check that its jupytext copy in md/ has the same cells
(run `make md` to regenerate the copies). Lines that start with a magic (%)
or shell command (!) become `pass`, keeping their indentation. Nothing is
executed, so this needs no data and no PyMC.
"""

import ast
import sys
from pathlib import Path

import jupytext

ROOT = Path(__file__).resolve().parents[2]


def code(nb):
    lines = []
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        for line in cell.source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(('%', '!')):
                line = line[: len(line) - len(stripped)] + 'pass'
            lines.append(line)
        lines.append('')
    return '\n'.join(lines)


def cells(nb):
    return [(c.cell_type, c.source.strip()) for c in nb.cells]


def main():
    errors = 0
    paths = sorted(ROOT.glob('*.ipynb'))
    for path in paths:
        name = path.relative_to(ROOT)
        nb = jupytext.read(path)
        try:
            ast.parse(code(nb), filename=str(path))
        except SyntaxError as e:
            errors += 1
            print(f'{name}: {e.msg} (code line {e.lineno})')

        md_path = ROOT / 'md' / (path.stem + '.md')
        if not md_path.exists():
            errors += 1
            print(f'{name}: no copy in md/ (run make md)')
        elif cells(jupytext.read(md_path)) != cells(nb):
            errors += 1
            print(f'{name}: md/{md_path.name} is out of date (run make md)')

    print(f'Checked {len(paths)} notebooks, {errors} problems')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
