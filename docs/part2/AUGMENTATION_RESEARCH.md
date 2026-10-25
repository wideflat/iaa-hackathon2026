# Data augmentation and transfer for the emerging market: research and strategy

2026-10-25 (EDT). A research agent searched the literature on request. **Caveat: the agent could not open full texts (the fetch tool failed), so every source below was read as a search snippet or abstract only.** Statements marked [S] come from such a snippet; [O] is opinion. Nothing here was reproduced by us except what the experiment section says.

## What the literature says

| Topic | Evidence | Relevance to us |
|---|---|---|
| Credibility plus machine learning from a large market to a small portfolio | [S] Dutang, Spedicato, Guibert, "Adjusting manual rates to own experience: comparing the credibility approach to machine learning", Variance 2025 (https://variancejournal.org/article/138733.pdf): models pretrained on a market portfolio and adapted to a small company beat company-only ML and were competitive with hierarchical Bühlmann-Straub | This is our structure (developed-market model, credibility-weighted level). It supports the design; it does not show a better one |
| Transfer learning in actuarial work | [S] Kim and Bauer, "Transfer learning in the actuarial domain" (https://arxiv.org/abs/2311.07597): instance, feature and parameter transfer for motor frequency with scarce labels, linked to credibility | Same |
| CANN | [S] Schelldorfer and Wüthrich 2019: a network correction on top of a GLM, trainable credibility weight | Part 1 GBM-offset net was rejected; no cross-country CANN paper was found |
| Importance weighting under covariate shift | [S] high variance, lower effective sample size, no gain in one 2025 study (https://arxiv.org/pdf/2112.10157); TableShift benchmark (https://arxiv.org/pdf/2312.07577) | We tried domain-shift weights (D67): negative |
| Synthetic rows (CTGAN, VAE, MICE) to enlarge a training set | [S] arXiv 2509.02171: on freMTPL2, GLMs fitted on synthetic data matched real-data GLMs overall but the relativities of categorical levels differed; authors advise against building a rating plan on it. [S] TabEBM (NeurIPS 2024): generic generators overfit on small tables and usually do worse than real data alone (authors' own motivation) | A generator cannot add information about 20-50 fire claims. Not worth trying |
| Tabular foundation models | [S] "Is TabPFN the silver bullet for insurance pricing?" (https://arxiv.org/pdf/2605.22892): not consistently better than a GLM or XGBoost on two motor sets; exposure and sparse counts not handled | Not worth trying as the main model |
| Heavy tails | [S] splicing a body with a GPD tail (https://arxiv.org/pdf/1608.01566); capped-mean calibration; resampling the target (SMOTE) distorts levels (https://arxiv.org/pdf/2311.11900) | Our level already uses winsorised means and the developed excess load; SMOTE would move the mean level that pooled Tweedie deviance punishes |
| Validation resampling | no source found. [O] a customer-cluster bootstrap measures noise but does not reduce it; a parametric bootstrap (simulate claims from the fitted mean with a fitted tail) can test whether a calibration estimator is biased at our claim counts, but only against the simulator's own assumptions | We already use the paired cluster bootstrap |

## Strategy for this project

Ranked by expected value and by risk of fitting our two-direction validation (the research agent's list, edited by us):
1. **Pooled training with a market flag** (the user's request): built and measured, see below.
2. **Parametric-bootstrap harness for the credibility constants** (the buildings constant 400 and the others were tuned on the same two validation directions): simulate emerging claim counts and severities from the developed model's expected values with a fitted compound Poisson-gamma/Pareto tail, apply the level estimators, and compare bias and deviance regret at our claim counts. It can reject a level method; it cannot pick the best one. Not yet run.
3. **Segment-level hierarchical (empirical-Bayes) shrinkage**: already covered by the gender factor and the rejected extra corrections (D66); expected gain small.
4. **Capped-plus-excess level**: the robust level already uses the 99% winsorised mean and the developed excess load.

Not worth trying: CTGAN/TabDDPM/TabSyn/copula augmentation, SMOTE, TabPFN as the main model, more covariate-shift weighting, choosing between models on simulated validation data, anything tuned on the public score.

## Experiment: pooled training (RESULTS 71-72)
See `docs/experiments/RESULTS.md` sections 71-72.
