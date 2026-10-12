"""Load the notebook's data pipeline and helper functions without running the full model comparison.

The scripts in this folder reuse exactly the code in notebooks/spx_spread_forecasting.ipynb, so any change
there carries over. Executing the pipeline (data cleaning, features, rolling HMM) takes about a minute.
"""
import contextlib
import io
import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings('ignore', message='FigureCanvasAgg is non-interactive')   # plt.show() in notebook cells

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks' / 'spx_spread_forecasting.ipynb'
CV_RUN_MARKER = 'results, win_stats, last_fit = {}'


def load(verbose=False):
    """Return the notebook namespace after Sections 0-6 (data, features, HMM, rolling_cv definition)."""
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    ns = {'display': print if verbose else (lambda *a, **k: None)}
    out = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())

    cwd = os.getcwd()
    os.chdir(NOTEBOOK.parent)                      # the notebook reads ../data/data.csv
    try:
        with out:
            for src in cells:
                if CV_RUN_MARKER in src:
                    exec(src.split(CV_RUN_MARKER)[0], ns)    # define rolling_cv, skip the model runs
                    break
                exec(src, ns)
            # helper functions that the notebook defines after the main model run
            for src in cells:
                if src.lstrip().startswith('def metrics'):
                    exec(src.split('bench = results')[0], ns)
                elif src.lstrip().startswith('def diebold_mariano'):
                    exec(src.split('y = results')[0], ns)
    finally:
        os.chdir(cwd)
    return ns
