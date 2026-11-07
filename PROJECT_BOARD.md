# SelectionSAT Project Board

Numbered tasks for tracking work. Each task has a permanent number; add new tasks at the end. Update status as work progresses.

The analysis behind the PyData Boston 2025 talk and the post "SAT math scores: gender difference or selection bias?": are boys' higher SAT math scores a difference in ability, or a difference in who takes the test? A Bayesian model of latent ability plus a selection function, fitted to College Board percentile tables for 2011–2015.

**Guides:** the research-repo refresh checklist, `~/GreenTeaPress/plans/research_repos.md` (steps 0–7 below follow it); SEO in `~/GreenTeaPress/plans/seo.md`; the list of repos and their order in `~/AllenDowneyCom/PROJECT_BOARD.md` Task 17. Ask Allen's decisions in a dialog (AskUserQuestion), in batches.

### Current focus (2026-11-06)

- **Task 1:** the site is live. Open: Allen submits the sitemap and pastes the post updates (`planning/crosslinks_drafts.md`); the Wayback 2014 link to recheck later (step 3).
- **Task 2:** paused to assess the collected data against the College Board's terms (see Task 2).

---

## Task 1: Refresh (research repo)

**Status:** Steps 0–1 done and pushed 2026-11-06; steps 2–6 done 2026-11-06 (site live); open items for Allen in steps 3 and 6; step 7 done (release `fit-2026-11`)

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
   - [ ] Link check (2026-11-06; 21 URLs in the README and the notebooks): College Board PDFs, fairness-measures, the repo and the zip all 200; Wayback 2015 snapshot confirmed. **Allen checked in a browser (2026-11-06): the three aei.org links are good.** Still unconfirmed: the Wayback 2014 math PDF (`web.archive.org/web/20160402182221/…/sat-percentile-ranks-mathematics-2014.pdf`), 429 in Allen's browser too, most likely the Internet Archive rate-limiting this network's IP after the day's scripted requests. Recheck later or from another network; if the snapshot is gone, find another (the availability API listed none at that timestamp) or cite the College Board's 2014 group-data CSV
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
   - [x] SelectionSAT env **rebuilt from scratch** (2026-11-06): the first `mamba env update` sat for 22 minutes, most likely at the "Confirm changes" prompt (no `-y`, no terminal; mamba 2.5); the solve takes under a minute. Fresh `mamba env create -f environment.yml -y`: PyMC 5.25.1, ArviZ 0.22.0, nutpie 0.16.8, PreliZ 0.20.0, Jupyter Book 2.1.7. PreliZ 0.20 failed to import with matplotlib 3.11 (`matplotlib.style.core` is gone), so `matplotlib=3.10` is pinned. `make check` and `make build` pass in it
   - [x] **Colab risk** fixed (2026-11-06): the Colab cell pins `preliz==0.20.*`, which fails with matplotlib 3.11, so the cell now also pins `"matplotlib<3.11"`. Colab rerun passes (85 code cells)
   - [x] **Live 2026-11-06: https://allendowney.github.io/SelectionSAT/** (gh-pages via ghp-import; built with the LifeExpectancy env's Jupyter Book 2.1.2 while the SelectionSAT env update ran). All pages, sitemap and robots.txt 200. `~/AllenDowney.github.io`: `Sitemap:` line, sitemap index regenerated, Research entry; pushed. About box: website, topics, description (`gh repo edit`; the same line in `~/GreenTeaPress/scripts/repo_about.sh`, committed locally there)
   - [x] Sitemap submitted in Search Console (Allen, 2026-11-06); ticked in GreenTeaPress Task 22 (committed locally there)
6. Cross-links
   - [x] Related posts (2026-11-06, `related-articles`): The Poincaré Problem, Berkson Goes to College, Factors of Underrepresentation, Causation Collision and Confusion; on the site's landing page ("Related posts")
   - [x] Drafts for Allen: `planning/crosslinks_drafts.md` (2026-11-06): an update paragraph for the Substack and blog posts (site link, the 5-point finding, GitHub, Colab), an optional "Related" line for three older posts, an optional book line. Allen pastes; for the blog, back up first
   - [ ] Blog and Substack updates: **AllenDowneyCom Task 23** (Allen, 2026-11-06: do the WordPress edits there, with the "Related" line on all three older posts; Allen pastes the Substack paragraph)
     - [x] **Blog done 2026-11-06** (AllenDowneyCom Task 23): the update paragraph plus a *Think Bayes* line (`from=blog-sat-math`) at the end of the SAT post; the "Related" line at the end of Berkson Goes to College, Factors of Underrepresentation and The Poincaré Problem. Checked live
     - [ ] Substack: Allen pastes the update paragraph (and the *Think Bayes* line with `from=substack-sat-math`, as on the blog); then check it through the API
   - [x] *Think Bayes* and the PyMC Labs workshop in the README's "Learn more"
7. Close
   - [x] **The fit as a release asset (done 2026-11-06).** The saved `selection_bias3_idata.nc` (2025-11-26) was not the published fit and had no log-likelihood (the LOO notebook's outputs show the 2025-11-21 fit it ran against did). Refit on Unity (`unity/refit.sh`; env `SelectionSAT` built there from `environment.yml`; 7–8 minutes on 4 CPUs): the first refit matched but had no log-likelihood, so the job now runs `pm.compute_log_likelihood` before saving (doesn't change the draws). Result: mu 507.59 / 502.08 (published 507.5 / 502.1), r-hat at most 1.00, no divergences, `log_likelihood.obs_bins` (4, 1000, 5, 2). The LOO notebook run against it reproduces the published influence figure. **Release `fit-2026-11`** (Allen's choice) with the `.nc` (checksum verified after download); `selection_bias3_loo.ipynb` downloads it if missing; README and `CLAUDE.md` updated; site redeployed. Local copies: `selection_bias3_idata.nc` (the refit), `results/old/selection_bias3_idata_2025-11-26.nc`, `results/refit-2026-11-06-nolik/`
   - [ ] Note what was done in `~/AllenDowneyCom/PROJECT_BOARD.md` Task 17 (done 2026-11-06, see there; update at the end)

### Content issues (Allen decides)

1. **The Colab link was broken** (confirmed 2026-11-06: undefined `zip_filename`; fixed, see step 2). The notebook installs `nutpie` alone on Colab (`%pip install nutpie`). Since ArviZ 1.x, that pulls the new ArviZ and breaks Colab's older PyMC (Allen's rule, 2026-11-04). The fix: one install cell before importing PyMC, `if "google.colab" in sys.modules: %pip install -q "pymc>=6" "arviz>=1.1" nutpie preliz empiricaldist`, then check the results still match the post (a library upgrade). If the notebook must stay on PyMC 5: pin `"arviz<1"` instead. Run on Colab before and after to confirm.
2. README: no links to the post, talk, slides, video or Colab; `conda` instructions.
3. No license on GitHub.
4. ~~`plan.md` (LOO plan)~~: done; deleted (Allen, 2026-11-06).
5. ~~**The README's "~8 points"**~~: fixed to about 5 (Allen, 2026-11-06). ~~The `.nc` question~~: resolved by the refit and release `fit-2026-11` (step 7). **The README's "~8 points" didn't match the published notebook.** The hierarchical model's posterior means of `mu` (published outputs, and the Colab rerun 2026-11-06) are Male 507.5, Female 502.1: a gap of about **5.4 points**. `selection_bias3_idata.nc` (2025-11-26) is a different fit (Male 503.0, Female 502.6), so it isn't the published one. Check the talk slides for the number Allen presented; fix the README (step 4); decide whether to refit and save the published fit before publishing the `.nc` (decision 5).

---

## Task 2: More data releases for the collection

**Status:** **Resumed 2026-11-06 at a slow rate** (Allen: "we have half of the data. I'm not sure our downloading constitutes scraping"): `scripts/download_state_reports.sh` gets the remaining state reports (2009–2013, 2016), one request every 30 seconds, stopping on 429/403. Earlier the same day: paused, then assessed (below). Paused to assess what we have, how useful more data would be, and whether it can be obtained in ways consistent with the College Board's terms. Nothing in `data/` is committed (`data/` is gitignored).

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

### Assessment of the state data (2026-11-06)

Participation from the **NCES Digest table 226.40** (federal, public domain: 2013 edition for 2011–12 and 2012–13, 2014 edition for 2013–14, 2016 edition for 2014–15; saved as `data/nces_sat_participation.csv`; no 2015–16 column, so 2016 isn't matched). Joined with the 25 states' gaps (male minus female mean math) for 2012 and 2013: 50 state-years. Plot: `data/state_gap_vs_participation.png` (local).

- **The gap shrinks as participation grows:** correlation −0.58 (2012), −0.67 (2013); a linear fit gives about −1.4 points per 10 percentage points, from about 42 points at 5% participation to about 28 at 100%. Correlation with the male share of test takers: −0.45.
- **The universal-SAT states stand out:** Maine (93%; the state paid for the SAT for all juniors) 13 points in both years; Delaware (100%) 21–22. Low-participation "ACT states" (Iowa, Mississippi, Arkansas, Alabama, Louisiana; 3–9%) are at 40–53.
- **Outlier:** DC (83%, gap 58).
- **Caveats:** this is ecological (states differ in more than participation); in low-participation states the SAT takers are mostly applicants to selective colleges, and a larger gap in the upper tail (greater male variance) would produce the same pattern without sex-specific selection. A model with state-specific selection and the full score bands could separate the two; the means alone can't.
- **How useful:** high, for a follow-up post or a model extension, and it's a check of the talk's conclusion with independent data. **What more data would add:** the other 26 states (more College Board PDFs), which would double the points and add more full-participation states (e.g. Idaho, Michigan, Illinois after they switched to the SAT, which is after 2016 on the redesigned test).

### While the download runs (2026-11-06)

- [x] Task 2 scripts kept **local and gitignored** (Allen, 2026-11-06: publish neither the derived data nor the scripts): `scripts/parse_total_group.py`, `parse_state_reports.py`, `parse_suite_reports.py`, `download_state_reports.sh`, `nces_participation.py`. They were briefly in two local commits, removed before any push
- [x] Participation for 2009–2015, all states: `scripts/nces_participation.py` (NCES Digest 2010 table 154, 2012 table 174, 2013/2014/2016 table 226.40; 101 overlapping state-years agree exactly) → `data/nces_sat_participation.csv`. No 2015–16 (the 2017 edition switched to the redesigned SAT), so the 2016 state reports stay unmatched
- [x] 2014–2015 state reports: not on the College Board's sites (probed `pdf/sat/`, `pdf/research/2014|2015/`, and the 2016 pattern on reports.collegeboard.org: all 404). Some state education departments host their own state's report
- [x] Sketch of a state-level model: `planning/state_model.md` (latent ability by state and sex, selection by state, sex and year; separates sex-specific selection from greater male variance using high- vs low-participation states; validate by simulation; fit on Unity)

### Decisions (Allen, 2026-11-06)

- After the download: a **first look only** (the 51-state plots and the male share of takers); decide on the state model after seeing it
- Graduates by sex: **find state-reported graduates by sex** (coverage will be uneven)
- **Publish neither the derived data nor the scripts**; `data/` and the Task 2 scripts stay local

### Graduates by sex (2026-11-06)

- Found (NCES, public domain; public schools only): **2009–10** (NCES 2013-309, table 8: graduates by sex and state) and **2011–12** (Digest 2015, table 219.33). Not found as counts: 2008–09 (only in an older CCD file), 2010–11 (NCES 2014-391 has graduation rates by sex, not counts), 2012–13 (no later edition of table 219.33)
- `scripts/nces_graduates.py` (local) → `data/nces_graduates_by_sex.csv`: 101 state-years. The male share of graduates ranges from 43% to 52% across states and changed by at most 2.5 points within a state from 2010 to 2012, so for 2009, 2011 and 2013 use the nearest year's share times the total graduates (implied by NCES participation)
- Caveat: SAT takers include private-school students; the graduates are public-school only

### First look, all 51 states (2026-11-06)

Download finished: 306 state reports (51 × 2009–2013, 2016; 157 new, at one request every 30 s; no failures, no throttling). All parse and pass the checks (the parser now treats an all-blank band row as zeros: ND_13, ND_16, WY_13). Joined with NCES participation (2009–2013) and graduates by sex: 255 state-years. Local outputs: `data/state_first_look.png`, `data/state_first_look.csv`.

- **The gap falls as participation rises:** pooled correlation −0.54 (by year −0.40 to −0.61); fit 41 points at 5% participation, 28 at 100%. Means by participation: 0–10% 41.6; 10–50% 36.7; 50–80% 33.4; 80–100% 28.2.
- **Boys are underrepresented among test takers almost everywhere:** the male share of takers is about 3 points below the male share of public-school graduates at every participation level (−2.8 below 10%, −3.8 at 10–50%, −3.3 at 50–80%, −1.0 above 80%), and about zero only at full participation (Maine +0.1, Delaware +0.3).
- **What that says about the two explanations** (`planning/state_model.md`): if the gap came from greater male variance with selection on ability only, boys would be *over*represented among takers in low-participation states, where takers come from the upper tail. They're underrepresented there too, which favors sex-specific selection (boys less likely to take the SAT at a given ability). Not conclusive: ecological; graduates are public-school only; and the underrepresentation barely varies with participation (correlation 0.14), which the model would have to explain.
- Next, if Allen wants: the state model (Allen chose a first look first).
- **State model started (Allen, 2026-11-06)** in `state/` (local, gitignored, with `selection_state.ipynb`): recovers known parameters from simulated data, but the first real fits conflict. Diagnosis (Allen chose to diagnose before changing the model) in `planning/state_model.md`, "First fits and diagnosis": a bad mode with negative selection; bands alone can't separate state ability from selection, so participation is essential; with participation, one slope per sex can't fit all states (implied δ ≈ 56, s_part ≈ 0.93).

### Consistent with the terms? (to decide)

- Downloading individual reports that the College Board posts for the public, for analysis, is their intended use; **bulk automated downloading** is what the no-scraping clause most plausibly covers.
- Alternatives: ask the College Board for permission or a data file (permission form, or the research office); **state education departments** post their own state's reports (e.g. New Hampshire's 2014 report on education.nh.gov), under their own terms; **NCES Digest** tables (federal, public domain) give means by sex and participation by state, though not distributions.
- Publishing: like `sat_math_csv_files.zip` (Allen: keep it, attributed), but any new data release is a separate decision.
