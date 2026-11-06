# SelectionSAT

The analysis behind Allen's PyData Boston 2025 talk and the post "SAT math scores: gender difference or selection bias?" (Substack and allendowney.com blog, 2025-12-16). Question: are boys' higher SAT math scores a difference in ability, or in who takes the test? A Bayesian model of latent ability plus a selection function, fitted to College Board percentile tables for 2011–2015.

## Published work: don't break it

- The post links **`selection_bias3.ipynb` on `main` via Colab**. Never rename or move it.
- Tags: `pydata-boston-2025` and `published-2025` (`6997859`) mark the version the talk and post used.
- The modeling stack is pinned in `environment.yml` to the versions that produced the published numbers (PyMC 5, ArviZ 0.x). Upgrading is a library change: recheck the numbers against the post (observed gap 31–34 points; ~8 after accounting for selection).

## What is where

- `selection_bias.ipynb`, `selection_bias2.ipynb`, `selection_bias3.ipynb`, `selection_bias4.ipynb`: the four model versions (described in the README). Version 3 is the published one.
- `selection_bias3_loo.ipynb`: LOO / data-attribution analysis; loads the fit `selection_bias3_idata.nc` that `selection_bias3.ipynb` saves (gitignored; to be published as a release asset).
- `figs/`: figures the notebooks save (used in the talk slides).
- `md/`: jupytext copies of the notebooks. The `.ipynb` files are the source; after changing a notebook, run `make md` and commit both.
- `parse_sat_math.py`: parses the College Board PDFs into per-year CSVs; `sat_math_csv_files.zip` holds the parsed CSVs the notebooks download. `scripts/`: small helpers (`parse_raw.py`, `download_math_csv.py`).
- `utils.py`: helpers imported by every notebook except `selection_bias3.ipynb` (`decorate`, `value_counts`); most of the file is copied from an AIBM project and unused (board, step 2). The published `selection_bias3.ipynb` defines its own, so it runs on Colab alone.

Run order: `parse_sat_math.py` (only to rebuild the CSVs from the PDFs) → `selection_bias3.ipynb` (fit, slow) → `selection_bias3_loo.ipynb`.

## Data terms

The College Board PDFs and CSVs are not committed (gitignored); the README should link the College Board pages instead (board, step 4). Only the parsed counts (`sat_math_csv_files.zip`) are published.

## Environment

`mamba env create -f environment.yml` (env `SelectionSAT`); always `mamba`, not `conda`.

## Board

`PROJECT_BOARD.md` (committed). The refresh follows `~/GreenTeaPress/plans/research_repos.md`; SEO in `~/GreenTeaPress/plans/seo.md`. Ask Allen's decisions in a dialog (AskUserQuestion), in batches.
