# SelectionSAT Project Board

Numbered tasks for tracking work. Each task has a permanent number; add new tasks at the end. Update status as work progresses.

The analysis behind the PyData Boston 2025 talk and the post "SAT math scores: gender difference or selection bias?": are boys' higher SAT math scores a difference in ability, or a difference in who takes the test? A Bayesian model of latent ability plus a selection function, fitted to College Board percentile tables for 2011–2015.

**Guides:** the research-repo refresh checklist, `~/GreenTeaPress/plans/research_repos.md` (steps 0–7 below follow it); SEO in `~/GreenTeaPress/plans/seo.md`; the list of repos and their order in `~/AllenDowneyCom/PROJECT_BOARD.md` Task 17. Ask Allen's decisions in a dialog (AskUserQuestion), in batches.

### Current focus (2026-11-06)

- **Task 1, step 2:** the Colab install cell (content issue 1) and a Colab run of `selection_bias3.ipynb`; then dead code (`utils.py`) and `make check`.

---

## Task 1: Refresh (research repo)

**Status:** Steps 0–1 done 2026-11-06 (local commits, not pushed); next: step 2

### Inventory (2026-11-06, read-only)

- **GitHub:** `AllenDowney/SelectionSAT`, public, 2 stars. Description "Exploration of selection processes in SAT test-taking"; no website, no topics, **no license**. No Pages site (`gh api …/pages` 404). One branch, `main`, up to date with `origin`; no tags.
- **History:** 18 commits, 2025-08-30 to 2025-12-16 ("Updates for talk slides" 2025-11-23; "Switching to nutpie" 11-26; "Updating with nutpie" 12-16, the day of the post).
- **Tracked (11 files):** `selection_bias.ipynb`, `selection_bias2.ipynb`, `selection_bias3.ipynb`, `selection_bias3_loo.ipynb`, `parse_sat_math.py`, `utils.py`, `sat_math_csv_files.zip` (the parsed CSVs the notebooks download), `requirements.txt`, `requirements-dev.txt`, `Makefile`, `README.md`.
- **Untracked (64):** 
  - `selection_bias4.ipynb` (the "slope penalty" version the README describes as version 4) and `md/` (jupytext-style `.md` copies of all five notebooks; `plan.md` says edits were made in `md/` and converted to `.ipynb`), `convert_notebooks_to_md.py`;
  - `selection_bias3_idata.nc` (the saved fit that `selection_bias3_loo.ipynb` loads) and ~40 figure PNGs (`selection_bias01.png` … `selection_bias40.png`, `selection_bias3_loo_influence.png`), probably for the slides;
  - data: 15 College Board PDFs (math 2011–2015, plus critical reading, writing and composite tables not used by the analysis), the per-year CSVs `sat_math_2011.csv` … `2015.csv` (the zip's contents), intermediate parser outputs (`sat_math_structured_*.csv`, `sat_math_table_*.csv`, `sat_math_2014_text_parsed.csv`, `sat_math_complete.csv`), the College Board's own 2014 and 2015 group-data CSVs;
  - helpers: `download_math_csv.py`, `parse_raw.py`; `plan.md` (the LOO data-attribution plan); `.vscode/`, `.ipynb_checkpoints/`, `__pycache__/`.
- **Article and talk:** Substack and blog, 2025-12-16 (updated 2025-12-30): "SAT math scores: gender difference or selection bias?", <https://allendowney.substack.com/p/sat-math-scores-gender-difference-or-selection-bias> and <https://www.allendowney.com/blog/2025/12/16/sat-math-scores-gender-difference-or-selection-bias/>. It announces the **PyData Boston 2025** talk video and links the slides (Google Slides), the first notebook (ThinkBayes2 `examples/bread.ipynb`, the Poincaré problem), and **"the second notebook (analysis of SAT data) on Colab": `colab.research.google.com/github/AllenDowney/SelectionSAT/blob/main/selection_bias3.ipynb`**. So `selection_bias3.ipynb` on `main` is the published analysis, and its path must not change. The post also links the PyMC Labs Applied Bayesian Modeling workshop and *Think Bayes*.
- **Notebook setup:** `selection_bias3.ipynb` (140 cells, 61 with saved outputs) installs `preliz`, `empiricaldist` and `nutpie` one at a time with `%pip install` if missing, and downloads `sat_math_csv_files.zip` from `main`. `np.set_printoptions(legacy='1.25')` is already there.
- **Environment:** no `environment.yml`; the Makefile's `create_environment` runs `conda create … python pymc` (the house rule is `mamba` and an `environment.yml`), then `pip install -r requirements-dev.txt`. `lint`/`format` point at a `code/` folder and a `pyproject.toml` that don't exist. `tests` runs `pytest --nbmake *.ipynb`, which would refit every model (slow).
- **README:** a good description of the question, findings, method and the four model versions, but no link to the post, the talk, the slides or the video; install steps use `conda`; "Mathemathics" in two PDF names is the College Board's spelling.
- **Secrets:** none found (no tokens or config files).
- **Size:** `.git` 25 MB.

### Decisions (Allen, 2026-11-06)

1. **Board and planning docs: committed** (as in OvertonParadox), not gitignored. No Pages site serves the default branch, so they don't become web pages; the site (decision 7) publishes from `gh-pages`.
2. **Tags:** `pydata-boston-2025` and `published-2025`, both on `6997859` ("Updating with nutpie", 2025-12-16, the version the post's Colab link ran).
3. **Model versions: keep all four, documented** (the README already describes them). Commit `selection_bias4.ipynb`. The published `selection_bias3.ipynb` and `selection_bias3_loo.ipynb` stay at the root with the others.
4. **College Board data:** keep publishing the parsed counts (`sat_math_csv_files.zip`); **link, don't commit** the PDFs (README links the College Board's pages, after checking their terms).
5. **Figures in `figs/`; the fit (`selection_bias3_idata.nc`) as a GitHub release asset**, so `selection_bias3_loo.ipynb` can run without refitting (it should download it if missing).
6. **License: code MIT, text CC BY-NC-SA 4.0** (as the workshops). `LICENSE` for the code; the README and the notebooks' last cell say both.
7. **Site: a proper Jupyter Book site, like the other projects.** For a research repo that means mystmd (Jupyter Book 2), as in LifeExpectancy and YouWouldChooseNow, with `jb/postbuild.py`, SEO and the page-view counter. Pages: landing page, the published notebook, the LOO notebook, the model versions.

### Steps

0. Survey and decisions
   - [x] Inventory (above)
   - [x] Allen answered the decisions (2026-11-06, above)
1. Set up
   - [x] Tagged `pydata-boston-2025` and `published-2025` on `6997859`, pushed (2026-11-06)
   - [x] Triaged untracked files (2026-11-06; Allen's answers: regenerate and commit `md/`; commit `figs/`; commit the helpers in `scripts/`; delete `plan.md`). Figures moved to `figs/`; `selection_bias.ipynb`, `selection_bias3.ipynb` and `selection_bias3_loo.ipynb` now create `figs/` and save there (only those lines changed). `md/` regenerated with `convert_notebooks_to_md.py` (the old copies were stale: `md/selection_bias3.md` lacked 18 cells); `make md` regenerates it. Deleted `plan.md` and the September `selection_bias.png`, `selection_bias2.png` (copies in the session scratchpad). `.gitignore`: PDFs, College Board CSVs, the unzipped `sat_math_*.csv`, `*.nc`, clutter
   - [x] `CLAUDE.md`
   - [x] `environment.yml`, pinned to the versions in the existing `SelectionSAT` env (Python 3.13, PyMC 5.25, ArviZ 0.22, nutpie 0.16, PreliZ 0.20; `--dry-run` solves). Removed the unpinned `requirements*.txt`. Makefile: mamba targets and `make md`; removed `lint`/`format` (pointed at nothing), `requirements` and `tests` (refit every model; step 2's `make check` replaces it). README install steps use mamba
2. Code and CI
   - [ ] Colab: fix the install cell (content issue 1) and run `selection_bias3.ipynb` on Colab (`colab-run` skill); compare the key numbers with the post (the gap from ~31–34 points to ~8)
   - [ ] Dead code: `utils.py`, `parse_raw.py`, `download_math_csv.py`, `convert_notebooks_to_md.py` (keep, merge or delete)
   - [ ] CI: `make check` = pyflakes plus a syntax check of the notebooks (LifeExpectancy's `.github/scripts/check_notebooks.py`); no model fits in CI unless Allen wants a fast test
3. Text and errors
   - [ ] codespell and a link check (README, notebooks)
   - [ ] Check the post's claims against the current outputs; anything different goes in content issues
4. README for readers of the published work
   - [ ] The post, the talk (video, slides) and the Colab link first; the analysis in brief (keep the current text); what is where; data (College Board source and terms); run order (parse → fit → LOO); model versions; how to cite (the tag); license
5. Site and SEO (decision 7: yes, mystmd)
   - [ ] mystmd with LifeExpectancy's `jb/postbuild.py`; `seo.md` checklist; `robots.txt` and the Research section on `allendowney.github.io`; Allen submits the sitemap; `repo_about.sh` (description, website, topics)
6. Cross-links
   - [ ] The post already links the Colab notebook. Add the repo (or site) link to the Substack and blog posts if Allen wants (drafts; back up the blog first)
   - [ ] *Think Bayes* or the PyMC Labs workshop in the README's "next steps"
7. Close
   - [ ] Note what was done in `~/AllenDowneyCom/PROJECT_BOARD.md` Task 17

### Content issues (Allen decides)

1. **The Colab link may be broken.** The notebook installs `nutpie` alone on Colab (`%pip install nutpie`). Since ArviZ 1.x, that pulls the new ArviZ and breaks Colab's older PyMC (Allen's rule, 2026-11-04). The fix: one install cell before importing PyMC, `if "google.colab" in sys.modules: %pip install -q "pymc>=6" "arviz>=1.1" nutpie preliz empiricaldist`, then check the results still match the post (a library upgrade). If the notebook must stay on PyMC 5: pin `"arviz<1"` instead. Run on Colab before and after to confirm.
2. README: no links to the post, talk, slides, video or Colab; `conda` instructions.
3. No license on GitHub.
4. ~~`plan.md` (LOO plan)~~: done; deleted (Allen, 2026-11-06).
