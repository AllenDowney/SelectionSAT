# The SAT Math Gap: Gender Difference or Selection Bias?

Male test takers score about 30 points higher than female test takers on the math section of the SAT. Does that reflect a difference in math ability, or a difference in who takes the test? If boys with weaker math skills are less likely to take the SAT than girls with the same skills, the observed gap overstates the difference in ability.

This repository holds the Bayesian model that estimates how much of the gap selection can explain, using the score distributions the College Board published for 2011–2015.

## The talk

Presented at **PyData Boston 2025** by Allen Downey.

- **Video:** <https://www.youtube.com/watch?v=6pwtbNVgyzg>
- **Slides:** [Google Slides](https://docs.google.com/presentation/d/e/2PACX-1vQ64XmhBYSAfIKHgBIWz8Pbvbgdk-9FxUUMpAGVqouR3eAVqR6hJytfl3-ffZ1qXTNMnNuQiCHr7QWs/pub)
- **Announcement and abstract:** [SAT math scores: gender difference or selection bias?](https://allendowney.substack.com/p/sat-math-scores-gender-difference-or-selection-bias) (*Probably Overthinking It*, December 16, 2025; also on [the blog](https://www.allendowney.com/blog/2025/12/16/sat-math-scores-gender-difference-or-selection-bias/))
- **The analysis notebook:** [`selection_bias3.ipynb`](selection_bias3.ipynb), [run it on Colab](https://colab.research.google.com/github/AllenDowney/SelectionSAT/blob/main/selection_bias3.ipynb)
- **The first notebook in the talk** (the Poincaré problem, a simpler example of selection bias): [`bread.ipynb` on Colab](https://colab.research.google.com/github/AllenDowney/ThinkBayes2/blob/master/examples/bread.ipynb), from *Think Bayes*

## Findings

- **Observed difference:** in each year from 2011 to 2015, male test takers scored 31–34 points higher than female test takers.
- **Selection effects:** the model estimates that boys with low math ability are less likely to take the SAT than girls with the same ability, and that selection is stronger for boys than for girls.
- **Difference in ability:** after accounting for selection, the estimated difference in latent math ability is only about 5 points (posterior means of 507.5 for males and 502.1 for females in the hierarchical model).

So most of the observed gap could be explained by who takes the test rather than by a difference in ability.

## Method

The hierarchical Bayesian model (in [PyMC](https://www.pymc.io/)):

1. **Latent ability:** a Gaussian distribution of math ability for each sex, extending beyond the SAT's 200–800 range.
2. **Selection:** a logistic function relating potential score to the probability of taking the test, for each year and sex.
3. **Test artifacts:** the score distributions are "spiky" because of how the SAT is scored (for example, the penalty for wrong answers); the model includes a noise term for that.
4. **Pooling:** the years share the ability distributions, and each year has its own selection function.

Prior elicitation uses [PreliZ](https://preliz.readthedocs.io/), sampling uses [nutpie](https://github.com/pymc-devs/nutpie), and the notebook validates the model by recovering known parameters from simulated data.

## What is where

| | |
|---|---|
| [`selection_bias3.ipynb`](selection_bias3.ipynb) | **The analysis presented in the talk** (model version 3) |
| [`selection_bias3_loo.ipynb`](selection_bias3_loo.ipynb) | Leave-one-out analysis: which score bins most influence the fit |
| [`selection_bias.ipynb`](selection_bias.ipynb), [`selection_bias2.ipynb`](selection_bias2.ipynb), [`selection_bias4.ipynb`](selection_bias4.ipynb) | Other model versions (below) |
| [`figs/`](figs/) | Figures the notebooks save (used in the slides) |
| [`md/`](md/) | Markdown copies of the notebooks ([jupytext](https://jupytext.readthedocs.io/)), for reading diffs |
| [`sat_math_csv_files.zip`](sat_math_csv_files.zip) | The score counts parsed from the College Board reports (the notebooks download it) |
| [`parse_sat_math.py`](parse_sat_math.py) | Parses the College Board PDFs into the CSV files |
| [`scripts/`](scripts/) | Small helpers: dump a PDF's raw rows; download and unzip the CSV files |
| [`utils.py`](utils.py) | Plotting helpers and the plot style used by the other notebooks |

## Data

The data are the College Board's "SAT Percentile Ranks for Males, Females, and Total Group" tables for college-bound seniors, mathematics section, which give the number of male and female test takers at each score from 200 to 800:

- [2011](https://secure-media.collegeboard.org/digitalServices/pdf/SAT-Mathemathics_Percentile_Ranks_2011.pdf), [2012](https://secure-media.collegeboard.org/digitalServices/pdf/research/SAT-Mathemathics-Percentile-Ranks-2012.pdf), [2013](https://secure-media.collegeboard.org/digitalServices/pdf/research/SAT-Mathematics-Percentile-Ranks-2013.pdf) (College Board)
- [2014](https://web.archive.org/web/20160402182221/https://secure-media.collegeboard.org/digitalServices/pdf/sat/sat-percentile-ranks-mathematics-2014.pdf), [2015](https://web.archive.org/web/20211222094405/https://secure-media.collegeboard.org/digitalServices/pdf/sat/sat-percentile-ranks-mathematics-2015.pdf) (Internet Archive copies; no longer on the College Board's site)

("Mathemathics" is the College Board's spelling.) The PDFs are not in this repository; [`sat_math_csv_files.zip`](sat_math_csv_files.zip) holds the counts parsed from them, one CSV file per year, with columns `Score`, `Total_Number`, `Male_Number`, `Female_Number`.

## Running it

The quickest way is [Colab](https://colab.research.google.com/github/AllenDowney/SelectionSAT/blob/main/selection_bias3.ipynb): the notebook installs what it needs and downloads the data.

To run it locally, create the environment with [mamba](https://mamba.readthedocs.io/) (the modeling stack is pinned to the versions that produced the published results):

```bash
git clone https://github.com/AllenDowney/SelectionSAT.git
cd SelectionSAT
mamba env create -f environment.yml
mamba activate SelectionSAT
jupyter lab
```

Run order:

1. `python parse_sat_math.py` — only to rebuild the CSV files from the PDFs (put the PDFs in this folder first); otherwise the notebooks download the parsed data.
2. `selection_bias3.ipynb` — fits the models (several minutes).
3. `selection_bias3_loo.ipynb` — loads the saved fit, `selection_bias3_idata.nc`. To create it, uncomment the `to_netcdf` line in `selection_bias3.ipynb`.

`make check` runs the static checks that CI runs (pyflakes, and a check that every notebook's code parses and its `md/` copy is current).

## Model Versions

The repository contains four different versions of the selection bias model, each exploring different modeling choices:

### Version 1: `selection_bias.ipynb` (Baseline Model)
- **Noise parameter**: Fixed `sigma_noise = 0.05`
- **Noise detrending**: None - uses raw `ZeroSumNormal` for log_noise
- **Selection function**: `logits = beta * thetas_scaled` (no intercept parameter)
- **Scaling**: `scale = 100` (thetas scaled to range -3 to 3)
- **Characteristics**: Simplest model, serves as baseline for comparison

### Version 2: `selection_bias2.ipynb` (Alternative Selection Model)
- **Noise parameter**: Fixed `sigma_noise = 0.05`
- **Noise detrending**: Detrends log_noise in simple model by subtracting linear trend
- **Selection function**: `logits = alpha + beta * thetas_scaled` (includes intercept `alpha`)
- **Scaling**: `scale = 400` (different scaling approach)
- **Hierarchical model**: Uses `HalfNormal` prior for beta, includes Binomial likelihood for number of test-takers
- **Characteristics**: Explores alternative selection function parameterization with intercept term

### Version 3: `selection_bias3.ipynb` (Estimated Noise Parameter; the version in the talk)
- **Noise parameter**: `sigma_noise` is estimated with Gamma prior (not fixed)
- **Noise detrending**: Detrends log_noise in simple model; hierarchical model has detrending with slope penalty
- **Selection function**: `logits = beta * thetas_scaled` (no intercept parameter)
- **Scaling**: `scale = 100`
- **Characteristics**: Allows the noise magnitude to be learned from data rather than fixed

### Version 4: `selection_bias4.ipynb` (Slope Penalty Model)
- **Noise parameter**: `sigma_noise` is estimated with Gamma prior
- **Noise detrending**: Detrends log_noise with explicit slope penalty using `pm.Potential` in both simple and hierarchical models
- **Selection function**: `logits = beta * thetas_scaled` (no intercept parameter)
- **Scaling**: `scale = 100`
- **Characteristics**: Most sophisticated noise modeling with explicit penalty on slope of detrended noise

### Key Differences Summary

| Feature | Version 1 | Version 2 | Version 3 | Version 4 |
|---------|-----------|-----------|------------|-----------|
| `sigma_noise` | Fixed (0.05) | Fixed (0.05) | Estimated | Estimated |
| Noise detrending | None | Yes (simple) | Yes (both) | Yes (both, with penalty) |
| Selection intercept | No | Yes (`alpha`) | No | No |
| Scale | 100 | 400 | 100 | 100 |
| Hierarchical Binomial | No | Yes | No | No |

All versions share the same core structure: they model latent efficacy distributions, estimate selection functions, and account for test artifacts. The differences reflect exploration of modeling choices around noise handling, selection function parameterization, and whether to fix or estimate the noise magnitude parameter.


## How to cite

For the findings, cite the talk:

> Allen B. Downey, "The SAT Math Gap: Gender Difference or Selection Bias?", PyData Boston 2025. <https://www.youtube.com/watch?v=6pwtbNVgyzg>

For the code, cite this repository at the tag `published-2025`, the version used for the talk and the post: <https://github.com/AllenDowney/SelectionSAT/tree/published-2025>.

## License

The code is available under the [MIT License](LICENSE). The text and figures are available under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). The SAT data are the College Board's.

## Learn more

- [*Think Bayes*](https://allendowney.github.io/ThinkBayes2/), free online, for the Bayesian methods used here.
- The [Applied Bayesian Modeling Workshop](https://www.pymc-labs.com/workshops/applied-bayesian-modeling) from PyMC Labs.
