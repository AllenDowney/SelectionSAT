---
title: "The SAT Math Gap: Gender Difference or Selection Bias?"
description: "Male test takers score about 30 points higher than female test takers on SAT math. A Bayesian model shows how much of that gap could come from who takes the test."
thumbnail: ../figs/selection_bias33.png
---

Male test takers score about 30 points higher than female test takers on the math section of the SAT. Does that reflect a difference in math ability, or a difference in who takes the test? If boys with weaker math skills are less likely to take the SAT than girls with the same skills, the observed gap overstates the difference in ability.

This site presents a Bayesian model that estimates how much of the gap selection can explain, using the score distributions the College Board published for 2011–2015. The code and data are on [GitHub](https://github.com/AllenDowney/SelectionSAT).

## The talk

Presented at **PyData Boston 2025** by [Allen Downey](https://allendowney.com).

:::{iframe} https://www.youtube-nocookie.com/embed/6pwtbNVgyzg
:width: 100%
Video of the talk.
:::

- **Slides:** [Google Slides](https://docs.google.com/presentation/d/e/2PACX-1vQ64XmhBYSAfIKHgBIWz8Pbvbgdk-9FxUUMpAGVqouR3eAVqR6hJytfl3-ffZ1qXTNMnNuQiCHr7QWs/pub)
- **Announcement and abstract:** [SAT math scores: gender difference or selection bias?](https://allendowney.substack.com/p/sat-math-scores-gender-difference-or-selection-bias) (*Probably Overthinking It*, December 16, 2025)
- **The first notebook in the talk** (the Poincaré problem, a simpler example of selection bias): [`bread.ipynb` on Colab](https://colab.research.google.com/github/AllenDowney/ThinkBayes2/blob/master/examples/bread.ipynb), from [*Think Bayes*](https://allendowney.github.io/ThinkBayes2/)

## Findings

- **Observed difference:** in each year from 2011 to 2015, male test takers scored 31–34 points higher than female test takers.
- **Selection effects:** the model estimates that boys with low math ability are less likely to take the SAT than girls with the same ability, and that selection is stronger for boys than for girls.
- **Difference in ability:** after accounting for selection, the estimated difference in latent math ability is only about 5 points (posterior means of 507.5 for males and 502.1 for females).

```{figure} ../figs/selection_bias33.png
:width: 80%
Posterior distributions of mean latent math ability for male and female test takers, 2011–2015.
```

So most of the observed gap could be explained by who takes the test rather than by a difference in ability.

## On this site

- [The Model](../selection_bias3.ipynb): the notebook presented in the talk, with the data, the model, a check that it recovers known parameters from simulated data, and the results. You can also [run it on Colab](https://colab.research.google.com/github/AllenDowney/SelectionSAT/blob/main/selection_bias3.ipynb).
- [Which Data Matter Most?](../selection_bias3_loo.ipynb): a leave-one-out analysis of which score bins most influence the fit.
- [Model Versions](versions.md): the four versions of the model and how they differ.

## Related posts

On [*Probably Overthinking It*](https://www.allendowney.com/blog/):

- [The Poincaré Problem](https://www.allendowney.com/blog/2025/09/25/the-poincare-problem/): the first example in the talk, using Bayesian methods to correct for selection bias.
- [Berkson Goes to College](https://www.allendowney.com/blog/2021/04/07/berkson-goes-to-college/): how selection in college admissions creates a misleading correlation between SAT math and verbal scores.
- [Factors of Underrepresentation](https://www.allendowney.com/blog/2021/08/25/factors-of-underrepresentation/): men at the top of the SAT math range, and math versus verbal ability, in women's underrepresentation in math-intensive fields.
- [Causation, Collision, and Confusion](https://www.allendowney.com/blog/2023/05/10/causation-collision-and-confusion/): a talk on Berkson's paradox and other errors caused by biased samples.

## Data

The data are the College Board's "SAT Percentile Ranks for Males, Females, and Total Group" tables for college-bound seniors, mathematics section, 2011–2015, which give the number of male and female test takers at each score from 200 to 800. The [README](https://github.com/AllenDowney/SelectionSAT#data) links the original reports.

## How to cite

For the findings, cite the talk: Allen B. Downey, "The SAT Math Gap: Gender Difference or Selection Bias?", PyData Boston 2025. For the code, cite the repository at the tag [`published-2025`](https://github.com/AllenDowney/SelectionSAT/tree/published-2025).

The code is available under the MIT License; the text and figures under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The SAT data are the College Board's.
