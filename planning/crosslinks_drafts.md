# Cross-link drafts (2026-11-06)

Drafts for Allen to paste; nothing here has been sent or published. After the site is live at https://allendowney.github.io/SelectionSAT/.

## 1. Substack post

[SAT math scores: gender difference or selection bias?](https://allendowney.substack.com/p/sat-math-scores-gender-difference-or-selection-bias) (2025-12-16)

Add at the end of the post:

> **Update (November 2026):** The notebooks, data and results from this talk are now collected on a site: [The SAT Math Gap](https://allendowney.github.io/SelectionSAT/). The short version: male test takers scored 31–34 points higher than female test takers in 2011–2015, but after accounting for who takes the test, the model's estimate of the difference in math ability is only about 5 points. The code is on [GitHub](https://github.com/AllenDowney/SelectionSAT), and the [notebook runs on Colab](https://colab.research.google.com/github/AllenDowney/SelectionSAT/blob/main/selection_bias3.ipynb).

Then, as on the blog (AllenDowneyCom Task 23, 2026-11-06), a second paragraph:

> The Bayesian methods in the talk are from [*Think Bayes*, second edition](https://greenteapress.com/wp/think-bayes/?from=substack-sat-math), which you can read online for free or order from [Bookshop.org](https://bookshop.org/a/98697/9781492089469) or [Amazon](https://amzn.to/334eqGo) (affiliate links).

After editing, check the live post through the API (`updated_at`) and that the slug is this one.

## 2. Blog post

[SAT math scores: gender difference or selection bias?](https://www.allendowney.com/blog/2025/12/16/sat-math-scores-gender-difference-or-selection-bias/)

The same update paragraph as the Substack post. Back up the blog first (`~/AllenDowneyCom/scripts/backup_wp.sh blog`), edit with wp-cli, clear the cache, check the page (AllenDowneyCom Tasks 15–16).

## 3. Related blog posts → the site (optional)

A one-line link from older posts a reader of this one would follow. Same procedure as above (backup, wp-cli). Suggested line at the end of each:

> **Related:** [The SAT Math Gap](https://allendowney.github.io/SelectionSAT/), a Bayesian model of how much of the gender gap in SAT math scores comes from who takes the test (PyData Boston 2025).

- [Berkson Goes to College](https://www.allendowney.com/blog/2021/04/07/berkson-goes-to-college/) (2021-04-07)
- [Factors of Underrepresentation](https://www.allendowney.com/blog/2021/08/25/factors-of-underrepresentation/) (2021-08-25)
- [The Poincaré Problem](https://www.allendowney.com/blog/2025/09/25/the-poincare-problem/) (2025-09-25), which already belongs to the same talk

## 4. Book line (optional)

The Bayesian methods are from *Think Bayes*; the selection-bias theme runs through *Probably Overthinking It* (Berkson's paradox chapter). If a book line fits the update, link the Green Tea Press book page with a `from=substack-sat-math` tag (AllenDowneyCom Task 16, calls-to-action conventions).
