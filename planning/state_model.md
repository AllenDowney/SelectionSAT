# Sketch: a state-level selection model (Task 2)

2026-11-06. A plan, not yet built. Data: `data/sat_bands_state.csv` (counts by sex in 100-point math bands, old SAT, 2009–2013 and 2016), `data/sat_participation_state.csv` (test takers by sex), `data/nces_sat_participation.csv` (percent of graduates taking the SAT, 2009–2015).

## The question

Across states, the male–female gap in mean SAT math shrinks as participation rises (correlation about −0.6 in 2012–13; Maine and Delaware, near-universal, have the smallest gaps). Two explanations predict that pattern:

1. **Sex-specific selection** (the talk's model): low-ability boys are less likely to take the SAT than low-ability girls, so the gap among test takers overstates the gap in ability, and more so where fewer students take the test.
2. **Greater male variance with selection on ability only:** boys' and girls' ability distributions have similar means, but boys' is wider. Where only strong students take the SAT, the takers come from the upper tail, where boys are overrepresented, so the gap among takers is larger, even if selection is the same for both sexes.

The state data can separate these because high-participation states show nearly the whole distribution (little selection), while low-participation states show only the selected upper part.

## Model

For state *s*, sex *g*, year *y*:

- **Latent ability:** θ ~ Normal(μ_s + δ_g, σ_g). States differ in mean ability (μ_s, hierarchical); the sex difference in mean (δ) and the two standard deviations (σ_g) are shared across states (or hierarchical around a shared value, as a check).
- **Selection:** P(take | θ) = logistic(α_{s,g,y} + β_g · (θ − 500)/100). The intercepts α set each state's participation level; the slopes β_g say how strongly ability drives taking the test, by sex.
- **Likelihood:**
  - band counts: Multinomial over the six 100-point bands (200–290, …, 700–800), with probabilities proportional to ∫ Normal(θ) · P(take | θ) over each band, computed on a 10-point grid and summed into bands (at 100-point resolution the scoring "spikes" average out, so no noise term);
  - number of takers: Binomial(N_grad,s,y,g, ∫ Normal · P(take)), which ties α to the observed participation. N_grad by sex isn't in the reports: use NCES graduates by state with an assumed sex split (about 50/50), or ACS counts of 17–18-year-olds by sex, and check how sensitive the results are.
- **Scores outside 200–800:** as in the national model, latent ability extends beyond the scale; mass below 200 and above 800 piles into the end bands.

## How it separates the explanations

- **Model A (sex-specific selection):** β_male ≠ β_female, free.
- **Model B (selection on ability only):** β_male = β_female; σ_male > σ_female allowed.
- Compare with LOO (as in `selection_bias3_loo.ipynb`), and with posterior predictive checks of the gap against participation: B predicts the gap goes to δ as participation goes to 100% with the male share of takers fixed by the graduates; A predicts the male share of takers falls as participation falls. The male share of takers by state (`test_takers_male / test_takers`) is the direct evidence: under B it should be close to the male share of graduates near full participation and rise as participation falls (upper tail is male); under A it falls.
- The 2012–13 first look: the male share of takers is below 50% in every state but Maine (51.6%), and correlates with the gap at −0.45; that already looks more like A than B, but it needs the model.

## Practical

- Size: 51 states × 2 sexes × 6 years (2016 without participation) = up to 612 cells × 6 bands; small. PyMC with nutpie; fits on Unity (`unity/`), one task per model variant.
- Validate first: simulate data from both A and B with known parameters, check the fits recover them and that LOO picks the right one.
- Check against the national model: the national 2011–2013 10-point tables summed into bands equal the state bands summed over states (already checked for the national reports), so the state model's implied national δ and σ_g should be close to `selection_bias3.ipynb`'s.
- 2016 (last old-SAT cohort, tested through January 2016) has no NCES participation and a different cohort definition; leave it out or use it only as a check.

## Data gaps

- 2014 and 2015 state reports: not found on the College Board's sites (several URL patterns 404); some state education departments host their own state's report.
- 2006–2008 state reports: not found (same patterns 404).
- Graduates by sex by state: needed for the takers likelihood; see above.
