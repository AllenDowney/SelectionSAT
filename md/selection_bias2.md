---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Selection Bias

Relevant articles: 

https://www.aei.org/carpe-diem/2016-sat-test-results-confirm-pattern-thats-persisted-for-45-years-high-school-boys-are-better-at-math-than-girls/

https://www.aei.org/economics/sat-results-dont-confirm-boys-better-at-math-than-girls/

https://www.aei.org/carpe-diem/actually-50-years-of-test-scores-do-confirm-that-boys-outperform-girls-on-the-sat-math-test/

```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.random

import pymc as pm
import arviz as az
import xarray as xr
import pytensor.tensor as pt 

from utils import value_counts, decorate

from scipy.special import expit

```

```python
# Make the figures smaller to save some screen real estate
plt.rcParams["figure.dpi"] = 75
plt.rcParams["figure.figsize"] = [6, 3.5]
plt.rcParams["axes.titlelocation"] = "left"
```

## SAT Data

Data from https://fairnessmeasures.github.io/Pages/DatasetPages/SAT.html

Zehlike, Meike, Carlos Castillo, Francesco Bonchi, Ricardo Baeza-Yates, Sara Hajian, Mohamed Megahed. “Fairness Measures: Datasets and software for detecting algorithmic discrimination.” June, 2017. http://fairness-measures.org/ [bibtex]

Ultimately from:

2011 math: https://secure-media.collegeboard.org/digitalServices/pdf/SAT-Mathemathics_Percentile_Ranks_2011.pdf

2011 writing: https://secure-media.collegeboard.org/digitalServices/pdf/SAT-Writing_Percentile_Ranks_2011.pdf

2012 mathematics: https://secure-media.collegeboard.org/digitalServices/pdf/research/SAT-Mathemathics-Percentile-Ranks-2012.pdf

2011 composite: https://secure-media.collegeboard.org/digitalServices/pdf/SAT-Percentile-Ranks-Composite-CR-M-W-2011.pdf

https://secure-media.collegeboard.org/digitalServices/pdf/SAT-Percentile-Ranks-Composite-CR-M_2011.pdf

2012 reading: https://secure-media.collegeboard.org/digitalServices/pdf/research/SAT-Critical-Reading-Percentile-Ranks-2012.pdf

2012 composite: https://secure-media.collegeboard.org/digitalServices/pdf/research/SAT-Percentile-Ranks-Composite-CR-M-W-2012.pdf

https://secure-media.collegeboard.org/digitalServices/pdf/research/SAT-Percentile-Ranks-Composite-CR-M-2012.pdf

2013 composite: https://secure-media.collegeboard.org/digitalServices/pdf/research/SAT-Percentile-Ranks-Composite-CR-M-W-2013.pdf

https://secure-media.collegeboard.org/digitalServices/pdf/research/SAT-Percentile-Ranks-Composite-CR-M-2013.pdf

https://secure-media.collegeboard.org/digitalServices/pdf/research/SAT-Percentile-Ranks-Composite-CR-M-W-2013.pdf

2013 mathematics: https://secure-media.collegeboard.org/digitalServices/pdf/research/SAT-Mathematics-Percentile-Ranks-2013.pdf

2013 reading: https://secure-media.collegeboard.org/digitalServices/pdf/research/SAT-Critical-Reading-Percentile-Ranks-2013.pdf

2014: https://web.archive.org/web/20160402182221/https://secure-media.collegeboard.org/digitalServices/pdf/sat/sat-percentile-ranks-mathematics-2014.pdf

2015: https://web.archive.org/web/20211222094405/https://secure-media.collegeboard.org/digitalServices/pdf/sat/sat-percentile-ranks-mathematics-2015.pdf

2016: 



```python
def read_sat_math(year):
    df = pd.read_csv(f'sat_math_{year}.csv', index_col=0)
    df.columns = ['Total', 'Male', 'Female']
    return df
```

```python
df_map = {}
years = range(2011, 2016)
for year in years:
    df_map[year] = read_sat_math(year)
```

```python
df_map[2011].sum()
```

```python
df_map[2012].sum()
```

```python
df_map[2013].sum()
```

```python
df_map[2014].sum()
```

```python
df_map[2015].sum()
```

```python
from empiricaldist import Pmf

def make_pmf(df, col):
    pmf = Pmf(df[col])
    pmf.normalize()
    return pmf
```

```python
for year, df in df_map.items():
    pmf = make_pmf(df, 'Male')
    pmf.plot(label=year)

decorate(title='Male test-takers')
```

```python
for year, df in df_map.items():
    pmf = make_pmf(df, 'Female')
    pmf.plot(label=year)

decorate(title='Female test-takers')
```

```python
def compare_spikes(df_map, year):
    pmf_male = make_pmf(df_map[year], 'Male')
    pmf_male.plot(label='Male')
    
    pmf_female = make_pmf(df_map[year], 'Female')
    pmf_female.plot(label='Female')
    
    mean_male = pmf_male.mean()
    mean_female = pmf_female.mean()
    print(f'Male={mean_male}, Female={mean_female}, Diff={mean_male - mean_female}')
    decorate(title=year)
```

```python
compare_spikes(df_map, 2011)
```

```python
compare_spikes(df_map, 2012)
```

```python
compare_spikes(df_map, 2013)
```

```python
compare_spikes(df_map, 2014)
```

```python
compare_spikes(df_map, 2015)
```

```python
def plot_selection_function(idata, col, thetas, **options):
    thetas_xr = xr.DataArray(thetas, dims="theta", coords={"theta": thetas})

    # Extract posterior samples of alpha and beta ---
    posterior = idata['posterior']  # shape (chain, draw)

    # Compute logits and apply sigmoid ---
    #logits = posterior["alpha"] + posterior["beta"] * thetas_xr
    logits = posterior["beta"] * thetas_xr
    p = xr.apply_ufunc(expit, logits)

    # Compute median and HDI ---
    median = p.median(dim=["chain", "draw"])
    hdi_bounds = az.hdi(p, hdi_prob=0.94)
    low, high = hdi_bounds['x'].T

    # Plot ---
    plt.plot(thetas, median, **options)
    plt.fill_between(
        thetas,
        low,
        high,
        alpha=0.3,
    )    
```

## Simple Model

```python
full_thetas = np.arange(100, 910, 10)
full_thetas
```

```python
scale = 400
full_thetas_scaled = (full_thetas - 500) / scale
full_thetas_scaled
```

```python
left_mask = full_thetas < 200
right_mask = full_thetas > 800
center_mask = (full_thetas >= 200) & (full_thetas <= 800)
```

```python
thetas = full_thetas[center_mask]
thetas
```

```python
thetas_centered = thetas - 500
```

```python
thetas_scaled = thetas_centered / scale
```

```python
year = 2014
sex = 'Male'
counts_obs = df_map[year].loc[thetas, sex]
counts_obs
```

```python
N_obs = int(counts_obs.sum())
N_thetas = len(thetas)
N_full_thetas = len(full_thetas)

# Assuming for now that half of all high school seniors took the test
N_total = N_obs * 2
N_total, N_obs
```

```python
import preliz as pz

dist, plot = pz.maxent(pz.Gamma(), lower=0.01, upper=0.1, mass=0.95)
beta_params = dist.params_dict
beta_params
```

```python
dist, plot = pz.maxent(pz.Gamma(), lower=60, upper=140, mass=0.95)
sigma_params = dist.params_dict
sigma_params
```

```python
dist, plot = pz.maxent(pz.Gamma(), lower=0.04, upper=0.07, mass=0.95)
sigma_noise_params = dist.params_dict
sigma_noise_params
```

```python
sigma_noise = 0.05
```

```python
def make_model(N_obs, counts_obs):
    with pm.Model() as model:
        # Latent population score distribution
        mu = pm.Normal("mu", mu=500, sigma=30)
        sigma = pm.Gamma("sigma", **sigma_params)

        # Selection model: logistic function
        beta = pm.Gamma("beta", **beta_params)
        
        # Compute population distribution (normalized logpdf)
        log_pdf = pm.logp(pm.Normal.dist(mu=mu, sigma=sigma), full_thetas)
        
        # Generate perturbations in the center region
        # sigma_noise = pm.Gamma("sigma_noise", **sigma_noise_params)
        log_noise_raw = pm.ZeroSumNormal("log_noise_raw", sigma=sigma_noise, 
                                         shape=(len(thetas),))

        # --- Detrend log_noise ---
        # slope = (x·y) / (x·x)
        x = thetas_scaled
        slope = pm.math.dot(log_noise_raw, x) / pm.math.dot(x, x)
        log_noise = pm.Deterministic("log_noise", log_noise_raw - slope * x)

        log_noise_full = pt.zeros_like(full_thetas)
        log_noise_full = pt.set_subtensor(log_noise_full[center_mask], log_noise)

        log_selection = beta * full_thetas_scaled
        
        # add up the logits
        logits = log_pdf + log_selection + log_noise_full
        
        # convert logits to probabilities
        pdf = pm.Deterministic("pdf", pm.math.sigmoid(logits))

        # roll up the tails
        left_mass = pdf[left_mask].sum()
        right_mass = pdf[right_mask].sum()
        center_pdf = pdf[center_mask]

        center_pdf = pt.set_subtensor(center_pdf[0], center_pdf[0] + left_mass)
        center_pdf = pt.set_subtensor(center_pdf[-1], center_pdf[-1] + right_mass)

        # normalize
        test_pdf = center_pdf / pm.math.sum(center_pdf)

        # Multinomial likelihood for binned scores
        pm.Multinomial("obs_bins", n=N_obs, p=test_pdf, observed=counts_obs)
        
    return model
```

```python
model = make_model(N_obs, counts_obs)

with model:
    prior = pm.sample_prior_predictive(1000)
    idata = pm.sample(target_accept=0.95)
```

```python
#pm.summary(idata, var_names=['mu', 'sigma', 'alpha', 'beta'])
pm.summary(idata, var_names=['mu', 'sigma', 'beta'])
```

```python
pm.plot_trace(idata, var_names=['mu', 'sigma', 'beta'])
decorate()
```

```python
pm.plot_posterior(idata, var_names=['mu', 'sigma'])
decorate()
```

```python
#pm.plot_posterior(idata, var_names=['alpha', 'beta'])

pm.plot_posterior(idata, var_names=['beta'])
decorate()
```

```python
az.plot_forest(idata, var_names='log_noise', combined=True);
```

```python
log_noise = np.median(az.extract(idata)['log_noise'], axis=1)
plt.plot(thetas, log_noise);
```

```python
pop_pdf = np.median(az.extract(idata)['pdf'], axis=1)
plt.plot(full_thetas, pop_pdf);
```

```python
plot_selection_function(idata, 'sat_math', thetas_scaled)
```

## Parameter recovery

```python
def generate_with_pymc(
    N_total, true_mu, true_sigma, true_alpha, true_beta, true_log_noise, random_seed=123
):
    with pm.Model() as gen:
        # Fixed “true” parameters as constants (no uncertainty)
        mu = pm.Data("mu", true_mu)
        sigma = pm.Data("sigma", true_sigma)
        alpha = pm.Data("alpha", true_alpha)
        beta = pm.Data("beta", true_beta)
        log_noise = pm.Data("log_noise", true_log_noise)
        
        logp_vals = pm.logp(pm.Normal.dist(mu=mu, sigma=sigma), full_thetas)
        perturbed_logp = logp_vals + log_noise - pm.math.mean(log_noise)
        logp_normalizer = pm.math.logsumexp(perturbed_logp)
        pop_pdf = pm.Deterministic("pop_pdf", pm.math.exp(perturbed_logp - logp_normalizer))

        # tails → center
        left_mass  = pop_pdf[left_mask].sum()
        right_mass = pop_pdf[right_mask].sum()
        center_pdf = pop_pdf[center_mask]
        center_pdf = pt.set_subtensor(center_pdf[0],  center_pdf[0]  + left_mass)
        center_pdf = pt.set_subtensor(center_pdf[-1], center_pdf[-1] + right_mass)

        # selection
        logits = alpha + beta * thetas_scaled
        p_take = pm.Deterministic("p_take", pm.math.sigmoid(logits))

        test_unnorm = center_pdf * p_take
        prob_test = pm.Deterministic("prob_test", test_unnorm.sum())
        test_pdf = pm.Deterministic("test_pdf", test_unnorm / test_unnorm.sum())

        # generative nodes (no observed)
        N_test = pm.Binomial("N_test", n=N_total, p=prob_test)
        obs_bins = pm.Multinomial("obs_bins", n=N_test, p=test_pdf)

        prior_pred = pm.sample_prior_predictive(
            draws=1, 
            var_names=["N_test", "obs_bins", "prob_test", "test_pdf", "p_take"],
            random_seed=random_seed
        )
        return prior_pred
```

```python
N_total=1_500_000
true_mu=530
true_sigma=120
true_alpha=0.0
true_beta=0.015
true_log_noise = log_noise

prior_pred = generate_with_pymc(
    N_total, true_mu, true_sigma, true_alpha, true_beta, true_log_noise, random_seed=4
)
```

```python
N_test_draw = int(prior_pred.prior["N_test"].values.squeeze())
N_test_draw
```

```python
counts_draw = np.asarray(prior_pred.prior["obs_bins"].values.squeeze(), dtype=int)
counts_draw
```

```python
pmf = Pmf(counts_draw, thetas)
pmf.normalize()
pmf.plot(label='draw')

pmf = Pmf(counts_obs, thetas)
pmf.normalize()
pmf.plot(label='observed')

pmf.normalize()
print(pmf.mean())
```

```python
N_test_draw, N_obs
```

```python
model = make_model(N_test_draw, counts_draw)
#model = make_model(N_obs, counts_obs)

with model:
    idata = pm.sample(target_accept=0.95)
```

```python
pm.summary(idata, var_names=['mu', 'sigma', 'alpha', 'beta'])
```

```python
az.plot_posterior(idata, var_names=['mu'], ref_val=true_mu);
```

```python
az.plot_posterior(idata, var_names=['sigma'], ref_val=true_sigma);
```

```python
az.plot_posterior(idata, var_names=['alpha'], ref_val=true_alpha);
```

```python
az.plot_posterior(idata, var_names=['beta'], ref_val=true_beta);
```

## Hierarchical

```python
def get_counts_obs(year, sex):
    return df_map[year].loc[thetas, sex]
```

```python
years = [2011, 2012, 2013, 2014, 2015]
sexes = ['Male', 'Female']

data = np.array([
    [get_counts_obs(year, sex=sex) for sex in sexes]
    for year in years
])

data.shape  # (n_years, n_sexes, n_scores)
```

```python
data.sum(axis=2)
```

```python
N_obs = data.sum(axis=2)

# Assuming for now that half of all high school seniors took the test
N_total = N_obs * 2
N_total, N_obs
```

```python
import pytensor.tensor as pt 

coords = {
    'year': years,
    'sex': sexes,
    'full_score': full_thetas,  # 81
    'score': thetas_centered,     # 61
}

with pm.Model(coords=coords) as model:
    # Data
    data_pt = pm.Data("data", data, dims=("year", "sex", "score"))
    N_obs_pt = pm.Data("N_obs", N_obs, dims=("year", "sex"))
    N_total_pt = pm.Data("N_total", N_total, dims=("year", "sex"))

    # Latent population score distribution
    mu = pm.Normal("mu", mu=500, sigma=30, dims='sex')
    sigma = pm.Gamma("sigma", alpha=20, beta=0.2, dims='sex')

    # Compute population distribution (normalized logpdf)
    mu_b = mu[:, None]
    sigma_b = sigma[:, None]
    theta_b = full_thetas[None, :]

    logp_vals = pm.logp(pm.Normal.dist(mu=mu_b, sigma=sigma_b), theta_b) # sex, score
    log_noise = pm.ZeroSumNormal("log_noise", sigma=0.05, dims=['year', 'full_score'])
    # TODO: consider taking diffs of log_noise
    
    perturbed_logp = logp_vals[None, :, :] + log_noise[:, None, :]
    logp_normalizer = pm.math.logsumexp(perturbed_logp, axis=-1, keepdims=True)
    pop_pdf = pm.math.exp(perturbed_logp - logp_normalizer)
    
    # collect tail mass per (year, sex)
    left_mass = pop_pdf[:, :, left_mask].sum(axis=-1)   # shape (year, sex)
    right_mass = pop_pdf[:, :, right_mask].sum(axis=-1) # shape (year, sex)

    # slice out the center scores
    center_pdf = pop_pdf[:, :, center_mask]             # shape (year, sex, n_center)

    # add tail mass into the first and last bins
    center_pdf = pt.set_subtensor(center_pdf[:, :, 0],  center_pdf[:, :, 0]  + left_mass)
    center_pdf = pt.set_subtensor(center_pdf[:, :, -1], center_pdf[:, :, -1] + right_mass)

    # Selection model: logistic function
    alpha = pm.Normal("alpha", mu=0, sigma=0.1, dims=['year', 'sex'])
    #beta = pm.Normal("beta", mu=0, sigma=0.01, dims=['year', 'sex'])
    beta = pm.HalfNormal("beta", sigma=0.01, dims=['year', 'sex'])

    logits = alpha[:, :, None] + beta[:, :, None] * thetas_scaled[None, None, :]
    p_take = pm.Deterministic("p_take", pm.math.sigmoid(logits))

    # Test-taker score distribution
    test_unnorm = center_pdf * p_take
    prob_test = pm.math.sum(test_unnorm, axis=-1)
    test_pdf = test_unnorm / pm.math.sum(test_unnorm, axis=-1, keepdims=True)

    # Binomial likelihood for number of test-takers
    pm.Binomial("N_test", n=N_total_pt, p=prob_test, observed=N_obs_pt)

    # Multinomial likelihood for binned scores
    pm.Multinomial("obs_bins", n=N_obs_pt, p=test_pdf, observed=data_pt,
                  dims=['year', 'sex', 'score'])

    prior = pm.sample_prior_predictive(1000)
```

```python
with model:
    idata = pm.sample(target_accept=0.95)
```

```python
pm.summary(idata, var_names=['mu', 'sigma', 'alpha', 'beta'])
```

```python
az.plot_forest(idata, var_names='log_noise', combined=True);
```

```python
pm.plot_trace(idata, var_names=['mu', 'sigma', 'alpha', 'beta'])
decorate()
```

```python
pm.plot_posterior(idata, var_names=['mu', 'sigma'])
decorate()
```

```python
pm.plot_posterior(idata, var_names=['alpha'])
decorate()
```

```python
pm.plot_posterior(idata, var_names=['beta'])
decorate()
```

```python
for year in years:
    idata_one = idata.sel(year=year, sex="Male")
    plot_selection_function(idata_one, None, thetas_centered / 30)
    
decorate(xlabel='Latent efficacy', ylabel='Prob(taking test)', title='Males 2014-2015')
```

```python
for year in years:
    idata_one = idata.sel(year=year, sex="Female")
    plot_selection_function(idata_one, None, thetas_centered / 30)
    
decorate(xlabel='Latent efficacy', ylabel='Prob(taking test)', title='Females 2014-2015')
```

```python
with model:
    post_pred = pm.sample_posterior_predictive(idata)
```

```python
post_pred
```

```python
obs_bins_post_pred = az.extract(post_pred, group='posterior_predictive')['obs_bins']
obs_bins_post_pred
```

```python
obs_bins_post_pred.shape
```

```python
medians = np.median(obs_bins_post_pred, axis=-1)
medians.shape
```

```python
for i, year in enumerate(years):
    for j, sex in enumerate(sexes):
        plt.figure()
        df_map[year][sex].plot(color='gray')
        plt.plot(thetas, medians[i, j], label=f'{year} {sex}')
        decorate()
```

```python

```
