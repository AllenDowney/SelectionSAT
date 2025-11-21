# SelectionSAT

A Python project for parsing SAT Mathematics percentile rank data from PDF files and analyzing selection bias in standardized testing.

## Overview

This repository contains tools to extract and process SAT Mathematics percentile rank data from official College Board PDF reports (2011-2015) and perform Bayesian analysis of selection bias in standardized testing. The project addresses a key question in educational assessment: whether observed gender differences in SAT Mathematics scores reflect true differences in ability or selection effects where students with different ability levels have different probabilities of taking the test.

The repository includes:
- **Data extraction tools** for parsing PDF percentile rank reports into structured CSV files
- **Bayesian analysis** using hierarchical modeling to estimate selection bias effects
- **Statistical validation** through parameter recovery tests and model diagnostics


## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SelectionSAT
```

2. Create the conda environment:
```bash
make create_environment
```

3. Activate the environment:
```bash
conda activate SelectionSAT
```

4. Install additional requirements:
```bash
make requirements
```


## Analysis Summary

The `selection_bias.ipynb` notebook contains a Bayesian analysis of selection bias in SAT Mathematics scores using data from 2011-2015.

### Key Findings

**Observed Score Differences:**
- Male test-takers consistently scored 31-34 points higher than female test-takers across all years (2011-2015)

**Selection Bias Analysis:**

- **Strong evidence of selection effects**: The analysis estimates that boys with low math ability are less likely to take the SAT than girls with equivalent ability

- **Gender difference in selection**: Selection effects are substantially stronger for males than females

- **Reduced true ability gap**: When accounting for selection bias, the estimated difference in underlying math efficacy between genders drops from 31-34 points to only ~8 points

### Methodology

The analysis uses a hierarchical Bayesian model that:

1. **Models latent ability distribution**: Assumes a Gaussian distribution of true math efficacy extending beyond the SAT score range (100-900)

2. **Estimates selection functions**: Relates potential test scores to probability of taking the SAT

3. **Accounts for test artifacts**: Models the "spiky" score distributions caused by SAT scoring mechanics (multiple choice penalties, etc)

4. **Pools information across years**: Uses hierarchical modeling to share information between years while allowing for year-specific effects

### Technical Approach

- **Data**: SAT Mathematics percentile rank distributions by gender (2011-2015)

- **Modeling**: PyMC-based Bayesian inference with MCMC sampling

- **Validation**: Parameter recovery tests confirm model reliability

- **Key parameters**: 
  - `mu`: Mean of latent efficacy distribution by gender
  - `sigma`: Standard deviation of latent efficacy by gender  
  - `beta`: Selection effect strength (how much efficacy affects test-taking probability)

### Implications

The analysis suggests that a substantial portion of the observed gender gap in SAT Mathematics scores may be due to selection bias rather than true differences in mathematical ability.

The methodology provides a framework for analyzing selection bias in other standardized assessments and educational contexts.

## Data Processing

### Data Sources

The project processes official College Board SAT Mathematics percentile rank reports:

- SAT-Mathemathics_Percentile_Ranks_2011.pdf
- SAT-Mathemathics-Percentile-Ranks-2012.pdf
- SAT-Mathematics-Percentile-Ranks-2013.pdf
- sat-percentile-ranks-mathematics-2014.pdf
- sat-percentile-ranks-mathematics-2015.pdf


### Parse SAT Data

Run this script to extract data from all available PDF files:

```bash
python parse_sat_math.py
```

This will:
- Process all SAT Mathematics PDF files in the directory
- Extract score distributions by gender for each year
- Save individual CSV files for each year (e.g., `sat_math_2011.csv`)
- Display summary statistics for each dataset

### Output Format

Each CSV file contains the following columns:
- `Score`: SAT Mathematics score (200-800)
- `Total_Number`: Total number of test-takers with this score
- `Male_Number`: Number of male test-takers with this score
- `Female_Number`: Number of female test-takers with this score

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

### Version 3: `selection_bias3.ipynb` (Estimated Noise Parameter)
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


