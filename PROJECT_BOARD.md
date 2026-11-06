# SelectionSAT Project Board

Numbered tasks for tracking work. Each task has a permanent number; add new tasks at the end. Update status as work progresses.

The analysis behind the PyData Boston 2025 talk and the post "SAT math scores: gender difference or selection bias?": are boys' higher SAT math scores a difference in ability, or a difference in who takes the test? A Bayesian model of latent ability plus a selection function, fitted to College Board percentile tables for 2011–2015.

**Guides:** the research-repo refresh checklist, `~/GreenTeaPress/plans/research_repos.md` (steps 0–7 below follow it); SEO in `~/GreenTeaPress/plans/seo.md`; the list of repos and their order in `~/AllenDowneyCom/PROJECT_BOARD.md` Task 17. Ask Allen's decisions in a dialog (AskUserQuestion), in batches.

### Current focus (2026-11-06)

- **Task 1, step 5:** site and SEO (mystmd). Allen: check four links in a browser (step 3).
- **Task 2:** paused to assess the collected data against the College Board's terms (see Task 2).

---

## Task 1: Refresh (research repo)

**Status:** Steps 0–1 done and pushed 2026-11-06; steps 2–4 done 2026-11-06 (link check: four for Allen); next: step 5

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
   - [x] Colab: fix the install cell (content issue 1) and run `selection_bias3.ipynb` on Colab (`colab-run` skill); compare the key numbers with the post (the gap from ~31–34 points to ~8)
     - Allen first chose upgrading to PyMC 6 on Colab, then **stay on PyMC 5** (2026-11-06) after the first Colab run showed ArviZ 1.x has no `plot_posterior` (about 9 plotting cells would need porting) and that Colab ships PyMC 5.28 with ArviZ 0.22. The three `try/except %pip install` cells are now one: `if "google.colab" in sys.modules: %pip install -q nutpie "arviz<1" preliz empiricaldist`.
     - **The published notebook never ran on Colab:** the data cell used `zip_filename` without defining it (it only worked where the CSVs already existed). Fixed: the download cell defines it.
     - Install cell, after three Colab runs (each newer package upgrades the stack: nutpie pulled PyTensor 3, the latest PreliZ needs ArviZ 1): `%pip install -q nutpie "pymc<6" "arviz<1" "preliz==0.20.*" empiricaldist`.
     - [x] **Colab run passes** (2026-11-06; Colab: Python 3.13, PyMC 5.28.5, ArviZ 0.22): all 85 code cells. Numbers match the published outputs within sampling noise: hierarchical `mu` Male 507.47 vs 507.50, Female 502.08 vs 502.08; model 2 `mu` 508.84 vs 508.81; model 1 `mu` 525.97 vs 526.19. Colab runs fewer chains (2,000 posterior-predictive draws vs 4,000; lower ESS). The committed notebook keeps its published outputs; only the setup and download cells changed.
     - Reference numbers: the post (a talk announcement) says only "about 30 points"; the observed gaps the notebook prints are 31.4, 33.8, 31.5, 31.3, 31.3 (2011–2015); the "~8 points" is the README's claim, shown only in plots (`selection_bias33.png`).
     - `selection_bias3_idata.nc` (2025-11-26) is not necessarily the published fit: the `to_netcdf` line in the published notebook is commented out. Its posterior mean of `mu` is 503.0 (Male) vs 502.6 (Female), a gap of ~0.3, not ~8 (content issue 5).
   - [x] Dead code (2026-11-06): `utils.py` trimmed from 928 to 163 lines (Allen: keep the style): `AIBM_COLORS`, `configure_plot_style()` (applied on import; the figures from `selection_bias.ipynb` use it), `underride`, `value_counts`, `decorate`. The scripts stay (decision in step 1); pyflakes fixes in `parse_sat_math.py`, `scripts/parse_raw.py`, `convert_notebooks_to_md.py`. The full `utils.py` is in git history
   - [x] CI: `make check` = pyflakes on the `.py` files plus `.github/scripts/check_notebooks.py` (adapted from LifeExpectancy: notebook syntax, with indented magics kept as `pass`; and the `md/` copies match the notebooks). `.github/workflows/check.yml` runs it on push. No model fits in CI
3. Text and errors
   - [x] codespell (2026-11-06): no typos in the README, `CLAUDE.md`, `md/`, the scripts
   - [ ] Link check (2026-11-06; 21 URLs in the README and the notebooks): College Board PDFs, fairness-measures, the repo and the zip all 200; Wayback 2015 snapshot confirmed. **Allen to check in a browser:** the three aei.org links in `selection_bias2.ipynb` and the other notebooks (403 to scripts, as the checklist expects) and the Wayback 2014 math PDF (`web.archive.org/web/20160402182221/…/sat-percentile-ranks-mathematics-2014.pdf`; rate-limited, and the availability API lists no snapshot)
   - [x] Claims (2026-11-06): the post only says "about 30 points" (the notebook: 31.3–33.8). The README's "~8 points" was wrong; Allen chose the notebook's number: now "about 5 points" (mu 507.5 vs 502.1)
4. README for readers of the published work
   - [x] README rewritten (2026-11-06): the talk (video `6pwtbNVgyzg`, slides, the post, Colab link, the bread notebook) first; findings (about 5 points) and method from the old text; what is where; data (links to the five PDFs, 2014–2015 via the Internet Archive); running it (Colab, mamba, run order, `make check`); model versions (kept); how to cite (the talk; the `published-2025` tag); license; learn more (*Think Bayes*, the workshop). `LICENSE` (MIT). All new links 200
   - [ ] Decision 6 also says the notebooks' last cell states both licenses: not done yet (it would add a cell to the published notebook; ask Allen)
   - [x] License cell (code MIT, text CC BY-NC-SA 4.0, repo link) at the end of all five notebooks (Allen, 2026-11-06)
   - [ ] **Data terms** (looked up 2026-11-06): collegeboard.org/site-terms now redirects to the student legal terms (privacy.collegeboard.org/student-legal-terms). Section 8: "Our Content may not be distributed, downloaded, uploaded, reproduced, reposted, retransmitted, disseminated, sold, published, broadcast, or circulated, or otherwise used, in part or in whole, in any way whatsoever without our express written permission," and "You may not … scrape or data-mine College Board Services or Content." Section 10: non-commercial use only. No terms specific to the research reports were found, and no permission process for them (the permission form covers test materials). The repo has published parsed counts (`sat_math_csv_files.zip`) since 2025; numerical facts aren't copyrightable in the US, but the terms are contractual. Allen (2026-11-06): **keep the zip, with attribution**; no permission request
5. Site and SEO (decision 7: yes; Allen confirmed Jupyter Book 2 / MyST, 2026-11-06)
   - [x] Site in `jb/` (2026-11-06; Jupyter Book 2.1.2 / book theme): `myst.yml`, `index.md` (the talk with the video embedded, findings with the mu posterior figure, pages, data, how to cite), `versions.md`; the notebooks render in place from `../` (no copies), with their saved outputs. `postbuild.py` from LifeExpectancy (`BASE = '/SelectionSAT'`), plus `DESCRIPTIONS` for the two notebook pages (the theme gives them the project's description; a frontmatter cell would show on Colab). Social image `figs/selection_bias33.png`. Test build in a scratch copy: 4 pages, sitemap with real URLs, no localhost, canonical, counter, descriptions. Allen reviewed it ("looks good")
   - [x] Notebook headings retitled (Allen): "The SAT Math Gap: A Selection Model" (`selection_bias3.ipynb`), "Which Data Matter Most? Leave-One-Out Analysis" (`selection_bias3_loo.ipynb`)
   - [x] `environment.yml`: `jupyter-book>=2`, `ghp-import`; `jb/_build/` gitignored
   - [ ] Deploy (`cd jb && make deploy`), then push `~/AllenDowney.github.io` (robots.txt `Sitemap:` line, sitemap index regenerated, Research entry: prepared, uncommitted) and set the About box (`gh repo edit`; `~/GreenTeaPress/scripts/repo_about.sh` has the SelectionSAT line, committed locally there)
   - [ ] Allen: submit `https://allendowney.github.io/SelectionSAT/sitemap.xml` in Search Console; tick GreenTeaPress Task 22
6. Cross-links
   - [ ] The post already links the Colab notebook. Add the repo (or site) link to the Substack and blog posts if Allen wants (drafts; back up the blog first)
   - [ ] *Think Bayes* or the PyMC Labs workshop in the README's "next steps"
7. Close
   - [ ] Note what was done in `~/AllenDowneyCom/PROJECT_BOARD.md` Task 17

### Content issues (Allen decides)

1. **The Colab link was broken** (confirmed 2026-11-06: undefined `zip_filename`; fixed, see step 2). The notebook installs `nutpie` alone on Colab (`%pip install nutpie`). Since ArviZ 1.x, that pulls the new ArviZ and breaks Colab's older PyMC (Allen's rule, 2026-11-04). The fix: one install cell before importing PyMC, `if "google.colab" in sys.modules: %pip install -q "pymc>=6" "arviz>=1.1" nutpie preliz empiricaldist`, then check the results still match the post (a library upgrade). If the notebook must stay on PyMC 5: pin `"arviz<1"` instead. Run on Colab before and after to confirm.
2. README: no links to the post, talk, slides, video or Colab; `conda` instructions.
3. No license on GitHub.
4. ~~`plan.md` (LOO plan)~~: done; deleted (Allen, 2026-11-06).
5. ~~**The README's "~8 points"**~~: fixed to about 5 (Allen, 2026-11-06). Still open: the `.nc` question below. **The README's "~8 points" didn't match the published notebook.** The hierarchical model's posterior means of `mu` (published outputs, and the Colab rerun 2026-11-06) are Male 507.5, Female 502.1: a gap of about **5.4 points**. `selection_bias3_idata.nc` (2025-11-26) is a different fit (Male 503.0, Female 502.6), so it isn't the published one. Check the talk slides for the number Allen presented; fix the README (step 4); decide whether to refit and save the published fit before publishing the `.nc` (decision 5).

---

## Task 2: More data releases for the collection

**Status:** **Paused 2026-11-06** (Allen) to assess what we have, how useful more data would be, and whether it can be obtained in ways consistent with the College Board's terms. Nothing in `data/` is committed (`data/` is gitignored).

### What happened

Allen asked (2026-11-06) to look for more data, then chose all four leads: retry the 10-point tables, national bands 2006–2016, state-level comparison, redesigned SAT. Two background agents **started downloading** College Board PDFs and parsing them. Then the terms lookup (Task 1, step 4) found section 8 of the College Board's site terms: no scraping or data-mining of their content, and no reproducing or reposting it without written permission. **The state-report download was stopped partway** (149 of about 306 known state files), and Allen paused the task to check against the terms before going further. Downloaded so far: 10 national reports, 149 state reports, 9 national suite reports (about 38 MB, in `data/raw/`, gitignored).

### What we have (local only)

| File | Contents | Source PDFs |
|---|---|---|
| `data/sat_bands_national.csv` | Old SAT, national, counts by sex in score bands: 50-point bands 2005–2008 (12 bins), 100-point bands 2009–2013 and 2016 (6 bins); math, reading, writing. 2014–2015 not found | `research/cb-seniors-2005-total-group-report.pdf`, `…-2006-national-report.pdf`, `…-2007-national-report.pdf`, `Total_Group_Report_CBS_08.pdf`, `cbs-2009-national-TOTAL-GROUP.pdf`, `2010-total-group-profile-report-cbs.pdf`, `cbs2011_total_group_report.pdf`, `TotalGroup-2012.pdf`, `2013/TotalGroup-2013.pdf`, reports.collegeboard.org 2016 suite report |
| `data/sat_bands_state.csv`, `data/sat_participation_state.csv` | Old SAT, 25 states (AL–MS, with DC), 2009–2013 and 2016: counts by sex in 100-point bands; test takers and mean math by sex. No participation rate (needs graduates: NCES Digest table 226.40, public domain) | `research/ST_yy_03_03_01.pdf` (2009–2012), `research/2013/ST_13_…`, `reports.collegeboard.org/media/pdf/st16030301.pdf` |
| `data/sat_bands_suite.csv` | Redesigned SAT, national, 2018–2025: **percentages** by sex in 100-point bands (2017 has no bands by sex) | reports.collegeboard.org data archive |

Checks: Male + Female = Total in every band (where Total excludes students with no gender); bands sum to the reported test takers; 2011–2013 bands equal the 10-point tables summed. Parsers: `scripts/parse_total_group.py`, `scripts/parse_state_reports.py`, `scripts/parse_suite_reports.py` (uncommitted). **No 10-point tables for other years were found** (URL guesses 404; the Wayback index was rate-limited).

### How useful (first look)

- **State level looks most valuable.** In 2012 the math gap is smallest where nearly everyone takes the SAT: Maine 13 points (51.6% of test takers male), Delaware 21 (48.3%), both states that paid for the SAT for all students; in low-participation "ACT states" it is 40–46 (Iowa 46, Mississippi 43). Across the 25 states the gap ranges 12–62 points in each year, against about 31–37 nationally. That is a direct, model-free check of the selection explanation, and a natural extension of the model (selection varies by state with participation).
- **National bands 2005–2016:** extend the series, but at 6–12 bins the model's 10-point "spikes" can't be fitted; useful for a binned-likelihood version or a check that the estimated selection is stable over time.
- **Redesigned SAT 2018–2025:** a different test and scale, percentages only; a separate, simpler comparison (Male 11–13% vs Female 6–8% in the 700–800 math band).

### Consistent with the terms? (to decide)

- Downloading individual reports that the College Board posts for the public, for analysis, is their intended use; **bulk automated downloading** is what the no-scraping clause most plausibly covers.
- Alternatives: ask the College Board for permission or a data file (permission form, or the research office); **state education departments** post their own state's reports (e.g. New Hampshire's 2014 report on education.nh.gov), under their own terms; **NCES Digest** tables (federal, public domain) give means by sex and participation by state, though not distributions.
- Publishing: like `sat_math_csv_files.zip` (Allen: keep it, attributed), but any new data release is a separate decision.
