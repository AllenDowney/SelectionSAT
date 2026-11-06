---
title: "Model Versions"
description: "Four versions of the SAT selection model, which differ in how they handle noise in the score distributions and how they parameterize the selection function."
---

The repository contains four versions of the selection bias model, each exploring different modeling choices:

### Version 1: [`selection_bias.ipynb`](https://github.com/AllenDowney/SelectionSAT/blob/main/selection_bias.ipynb) (Baseline Model)
- **Noise parameter**: Fixed `sigma_noise = 0.05`
- **Noise detrending**: None - uses raw `ZeroSumNormal` for log_noise
- **Selection function**: `logits = beta * thetas_scaled` (no intercept parameter)
- **Scaling**: `scale = 100` (thetas scaled to range -3 to 3)
- **Characteristics**: Simplest model, serves as baseline for comparison

### Version 2: [`selection_bias2.ipynb`](https://github.com/AllenDowney/SelectionSAT/blob/main/selection_bias2.ipynb) (Alternative Selection Model)
- **Noise parameter**: Fixed `sigma_noise = 0.05`
- **Noise detrending**: Detrends log_noise in simple model by subtracting linear trend
- **Selection function**: `logits = alpha + beta * thetas_scaled` (includes intercept `alpha`)
- **Scaling**: `scale = 400` (different scaling approach)
- **Hierarchical model**: Uses `HalfNormal` prior for beta, includes Binomial likelihood for number of test-takers
- **Characteristics**: Explores alternative selection function parameterization with intercept term

### Version 3: [`selection_bias3.ipynb`](https://github.com/AllenDowney/SelectionSAT/blob/main/selection_bias3.ipynb) (Estimated Noise Parameter; the version in the talk)
- **Noise parameter**: `sigma_noise` is estimated with Gamma prior (not fixed)
- **Noise detrending**: Detrends log_noise in simple model; hierarchical model has detrending with slope penalty
- **Selection function**: `logits = beta * thetas_scaled` (no intercept parameter)
- **Scaling**: `scale = 100`
- **Characteristics**: Allows the noise magnitude to be learned from data rather than fixed

### Version 4: [`selection_bias4.ipynb`](https://github.com/AllenDowney/SelectionSAT/blob/main/selection_bias4.ipynb) (Slope Penalty Model)
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


