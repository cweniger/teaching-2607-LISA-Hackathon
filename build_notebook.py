"""Convert tutorial_src.py (jupytext percent format) -> tutorial_lisa_sbi.ipynb.

Cell markers:  "# %% [markdown]" starts a markdown cell (lines stripped of a
leading "# "),  "# %%" starts a code cell.  Run:  python build_notebook.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys
SRC = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'tutorial_src.py')
OUT = HERE / (SRC.stem.replace('_src', '') + '.ipynb' if len(sys.argv) > 1 else 'tutorial_lisa_sbi.ipynb')


def parse(text):
    cells = []
    kind, buf = None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return
        if kind == 'markdown':
            lines = [re.sub(r'^# ?', '', l) for l in lines]
        src = [l + '\n' for l in lines]
        src[-1] = src[-1].rstrip('\n')
        cells.append({'cell_type': kind, 'metadata': {}, 'source': src,
                      **({'outputs': [], 'execution_count': None}
                         if kind == 'code' else {})})

    for line in text.splitlines():
        m = re.match(r'^# %%( \[markdown\])?\s*$', line)
        if m:
            flush()
            kind, buf = ('markdown' if m.group(1) else 'code'), []
        else:
            buf.append(line)
    flush()
    return cells


nb = {
    'cells': parse(SRC.read_text()),
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                       'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10'},
        'accelerator': 'GPU',
        'colab': {'provenance': [], 'gpuType': 'T4'},
    },
    'nbformat': 4,
    'nbformat_minor': 4,
}
OUT.write_text(json.dumps(nb, indent=1))
print(f'wrote {OUT} ({len(nb["cells"])} cells)')
