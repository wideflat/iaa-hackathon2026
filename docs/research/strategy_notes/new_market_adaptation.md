# Adapting an insurance pricing model from a developed market to a new (emerging) market: methods and playbook for IAA Hackathon 2026 Part 2

## Provenance and reliability legend (read first)

- **[READ]** = I fetched and read the primary text (raw source or docs) in this session. Only the LightGBM and XGBoost source and docs fall in this class (fetched from raw.githubusercontent.com, master branch as of 2026-10-18).
- **[SNIPPET]** = I saw only a web-search result summary (search-engine-generated text, not the document). Treat these as leads. I could not open arxiv.org (proxy 403 on CONNECT; WebFetch DNS failure), variancejournal.org (403), or sciencedirect. No paper below was read in full.
- **[BG]** = my background knowledge of standard actuarial/ML practice. It is not sourced in this session and is unverified. It appears only under "Inferences" and should be checked before being quoted as fact in a submission.
- Search effort: about 13 tool calls. The literature coverage is thin. Gaps are listed explicitly per section.

---

## 1. Detecting what changed (schema/unit checks, missingness/coding, shift tests, frequency/severity/lag, exposure, inflation, documentation)

### Takeaway
Detection is a diagnostic ladder: (a) schema, units and coding audit, (b) univariate then multivariate shift tests (PSI/KS/chi-square, then a domain classifier / adversarial validation), (c) outcome-side comparisons (frequency, severity, report lag, exposure, inflation) which covariate-shift tests cannot see. The ASOP 23 / ASOP 56 framing (data selection, review, limitations, disclosure) is the natural wrapper for the change log. I found no actuarial paper that prescribes a template for a "market change log"; that format has to be designed.

### Cited Findings
- Adversarial validation recipe: drop the target, concatenate train and test, label test rows 1, fit a classifier, read ROC-AUC. AUC near 0.5 means the distributions are indistinguishable; high AUC signals shift. Feature importances of the adversarial model point at drifting features. Dropping a strongly drifting feature is a judgement call, to be checked against validation performance. A library default threshold of 0.7 is that library's choice, not a standard. [SNIPPET; blog/library sources: [KDnuggets "Adversarial Validation, Explained"](https://www.kdnuggets.com/2016/10/adversarial-validation-explained.html), [adval on PyPI](https://pypi.org/project/adval)]
- A covariate-shift workflow in a (non-peer-reviewed) library README: shift detection (KS and chi-square tests, PSI, classifier two-sample tests), then importance weighting (domain-classifier density ratios, clipping, normalisation, effective sample size), then calibration checks. [SNIPPET: [shiftstat on PyPI](https://pypi.org/project/shiftstat/)]
- Covariate shift assumes P(y|x) is unchanged while P(x) moves; density-ratio weights estimated by a domain classifier are the usual tabular approach, but density-ratio estimates degrade (high variance) in high dimension. [SNIPPET: [par.nsf.gov biblio 10343723](https://par.nsf.gov/biblio/10343723) and the shiftstat README above]
- ASOP 23 (Data Quality; adopted 1993, last revised Dec 2016) covers selecting data, reviewing it, using it, reliance on data supplied by others, documentation, and communications/disclosures. ASOP 56 (Modeling, effective 1 Oct 2020) covers model appropriateness, data quality and output validation. [SNIPPET: [Actuarial Standards Board ASOP 23](https://www.actuarialstandardsboard.org/asops/asop-no-23-revision-data-quality/), [actuary.info on ASOP 56](https://actuary.info/insights/cas-ai-primer-2026-review) (secondary; section numbers unverified)]
- A 2015 ASOP 23 exposure draft proposed that actuaries summarise unresolved questionable data in disclosures. Whether the 2016 final kept this wording is unconfirmed. [SNIPPET: ASB pages above]
- Claim-reporting delay: ignoring right-censored reporting delays underestimates event frequency. Joint delay/frequency models exist (Shikuri and Fujisawa, arXiv 2510.04421, parametric hazards for newly enrolled cohorts; Abdelrahman, Badescu, Craiu and Lin, arXiv 2409.12896, marked Cox/HMM micro-level IBNR counts; Verrall and Wüthrich 2016 and Verbelen et al. 2017 on occurrence/reporting). [SNIPPET: [arXiv 2510.04421](https://arxiv.org/html/2510.04421v2), [arXiv 2409.12896](https://arxiv.org/abs/2409.12896v1), [arXiv 1801.02935](https://arxiv.org/pdf/1801.02935)]
- Inflation is not one number: an Argentine motor-liability study (IAA presentation 2017) found a homogeneous portfolio does not imply homogeneous claims, because property damage, bodily injury and death claims, and administrative/mediation/trial routes, inflate differently. Turkish motor claims rose sharply under inflation above 50% while regulatory price caps (MTPL) limited rate increases. Duration of an inflation spike matters through wage-cost spirals (KPMG). [SNIPPET: [IAA Cuypers Panama 2017](https://actuaries.org/app/uploads/2025/07/Cuypers_PPT_Panama2017.pdf), [Fitch via reinsurancene.ws](https://www.reinsurancene.ws/challenging-times-for-turkish-insurers-fitch/), [KPMG inflation impact](https://assets.kpmg.com/content/dam/kpmgsites/ch/pdf/inflation-impact-on-insurers.pdf)]
- XGBoost documentation explicitly states that `base_margin` is the analogue of a GLM offset, with the Poisson exposure offset ln(exposure) as its worked example. [READ: [xgboost doc/tutorials/intercept.rst](https://raw.githubusercontent.com/dmlc/xgboost/master/doc/tutorials/intercept.rst), "Offset" section]

### Inferences
- **[BG]** Concrete detection checklist, in order (all cheap, all scriptable):
  1. Schema: column names/types/cardinalities vs Part 1; units (currency, sum insured, area m2 vs sq ft, vehicle age in months vs years, dates D/M/Y vs M/D/Y, 1900-01-01 or 9999 sentinels, 0 meaning missing); id uniqueness and join keys; policy start/end vs exposure fraction; negative or zero exposure; claims with report date before loss date; claims outside policy period.
  2. Missingness: per-column NA rate, per-column "pseudo-missing" values (0, -1, 999, "UNKNOWN", ""), missingness vs claim indicator (informative missingness check), missingness by sub-period (a column that "turns on" mid-history).
  3. Coding: category level overlap with Part 1 (new/unseen levels, merged levels, case/whitespace variants), ordinal scale changes, different peril taxonomy; build an explicit mapping table and log unmapped levels.
  4. Univariate shift per feature: PSI (rules of thumb 0.1 moderate / 0.25 large; **[BG]**, convention not a theorem), KS for numerics, chi-square or Jensen-Shannon for categoricals. Report together with the share of exposure affected, not just the statistic.
  5. Multivariate shift: domain classifier (LightGBM) Part 1 vs Part 2 with grouped CV; AUC plus top features; also AUC of a classifier that cannot see any rating factors but sees only calendar/exposure fields to catch convention changes.
  6. Outcome side (not detectable by covariate tests): frequency per peril per 1,000 exposure-years with Poisson/NB CIs; zero-claim share; severity quantiles (median, p90, p99, max) and cap-hit rate for your severity cap; report-lag CDF per peril (days from loss to report), censoring at the data cut-off; claim-count to premium/sum-insured ratios; seasonality.
  7. Exposure conventions: check sum of exposure vs calendar span; whether exposure is policy-years, days, months, vehicle-years or sums insured; mid-term changes; whether cancelled policies are present; whether claim records exist for policies absent from the policy file.
  8. Inflation: if the dataset has a date and amounts, fit log-severity on calendar-year dummies (controlling for risk mix) to read trend directly; compare with any CPI/repair-cost index available; keep trend a scenario parameter rather than a point estimate when there are only 2-4 calendar years. Segment by claim type where bodily-injury-like and damage-like claims exist (Argentina lesson).
  9. Is the Part 1 -> Part 2 difference covariate shift, label shift (base rate moves), or concept shift (relativities move)? Domain-classifier AUC addresses the first; a calibration plot of the Part 1 model on Part 2 (actual/expected by decile, by rating factor) addresses the second and third.
- **[BG]** Change-log structure that maps onto ASOP 23/56 language: for each item record (id, area, what changed, evidence (statistic + plot), size of impact (exposure share and estimated effect on expected cost), action taken, residual uncertainty, owner, status). Keep "assumed unchanged" items too: an explicit list of things checked and found stable is persuasive.
- Because covariate-shift weighting assumes P(y|x) stable, it is the wrong remedy for changed perils or reporting; use it only after the calibration check shows relativities hold.

### Gaps
- No actuarial paper found that gives a model-change-log template or "market comparison" protocol. All checklist content above is background knowledge.
- ASOP 23/56 provisions were not read in the original text (ASB PDFs not reached). Do not quote section numbers.
- No source found on PSI threshold provenance for insurance; the 0.1/0.25 convention is from credit-scoring practice **[BG]**.

---

## 2. Modelling responses (refit vs transfer vs recalibration; credibility; GLM offset; monotone constraints; domain adaptation for tabular; GBM fine-tuning)

### Takeaway
With a smaller, noisier dataset the best-supported family is "keep the Part 1 model as a prior and learn a small, regularised correction" (offset/base_margin/init_score, then a low-capacity model), with credibility-style shrinkage to decide how far to move from the prior, and a from-scratch simple GLM as a competing benchmark. The only directly relevant insurance paper found (Dutang, Spedicato, Guibert, Variance 2025) reports that pretraining on a broad portfolio and transferring to a small book beat a model trained on the small book alone and was competitive with hierarchical Buhlmann-Straub credibility. This is from the abstract/snippets only.

### Cited Findings
- Dutang, Spedicato and Guibert, "Adjusting Manual Rates to Own Experience: Comparing the Credibility Approach to Machine Learning," Variance vol. 18 (2025), CAS Variance Prize 2025: pretrain ML models on a broad market portfolio, apply to an individual insurer's book; combining company and market data significantly improved accuracy vs an insurer-only ML model and was competitive with hierarchical Buhlmann-Straub credibility; data are an anonymised market dataset plus a European non-life insurer. [SNIPPET: [Variance PDF](https://variancejournal.org/article/138733.pdf) (not opened), [CAS abstract](https://www.casact.org/abstract/adjusting-manual-rates-own-experience-comparing-credibility-approach-machine-learning), [CAS prize article](https://www.casact.org/article/2025-variance-prize-honors-research-credibility-theory-and-machine-learning), [HAL preprint](https://hal.univ-grenoble-alpes.fr/LJK-PS-ASAR/hal-04821310v2)]
- Shi and Shi (2022) reported that embedding layers trained by supervised learning transfer effectively to models with limited data (cited in a neural-network pricing benchmark, [arXiv 2310.12671](https://arxiv.org/pdf/2310.12671)). [SNIPPET, second-hand citation]
- A 2026 evaluation of TabPFN for motor insurance pricing found it does not consistently outperform established baselines, has much longer inference, and is sensitive to training-set size. [SNIPPET: arXiv 2601.07675 / search result list; paper not read]
- Credibility basics: credibility-weighted estimate = Z x observation + (1-Z) x other information; 1-Z is the complement of credibility; with limited data Z falls toward 0. Buhlmann-Straub adds exposure weights per observation. The complement is where the judgement lies, and common approaches may give too much credibility. [SNIPPET: [SOA STAM study note](https://www.soa.org/Files/Edu/2018/2018-stam-22-18.pdf), [CAS E-Forum "Credibility: counting the cost"](https://eforum.casact.org/article/120031-credibility-counting-the-cost/attachment/232278.pdf)]
- Nielsen and Sandqvist (ASTIN Bulletin 2000) applied Buhlmann-Straub weighting to hazard estimation so that sparse subgroups borrow from others. [SNIPPET: [ASTIN vol30 no2 p405](https://www.casact.org/library/astin/vol30no2/405.pdf)]
- Bayesian shrinkage: Venter (CAS 2018 presentation) argues Bayesian shrinkage can beat MLE for multivariate GLM estimation and that the James-Stein estimator matches Buhlmann credibility; Zhang and Miljkovic (CAS E-Forum 2018) use a Bayesian Poisson frequency model with power priors for a new territory, borrowing from a similar book and improving prediction error. [SNIPPET: [CAS annual 2018 c-27](https://casact.org/sites/default/files/presentation/annual_2018_presentations_c-27_1.pdf), [Zhang-Miljkovic](https://www.casact.org/pubs/forum/18spforumv2/03_Zhang_Miljkovic.pdf)]
- "Adapted Tree Boosting for Transfer Learning" (arXiv 2002.11982): train XGBoost on the source domain, then revise/continue those trees on target data; two strategies (batch transfer, multi-round exchange). A 2025 battery-pack paper (arXiv 2504.10658) fine-tunes a base XGBoost model by adding a few shallow trees on limited target data. I found no benchmark of GBDT warm-start vs from-scratch on small data. [SNIPPET: [arXiv 2002.11982](https://arxiv.org/pdf/2002.11982), [arXiv 2504.10658](https://arxiv.org/pdf/2504.10658)]
- LightGBM `train()` accepts `init_model` (filename or Booster) "used for continue training"; `Dataset(init_score=...)` sets per-row starting scores; if `init_model` is passed, a user-supplied init_score is overridden by the init model's predictions (library warns). [READ: [lightgbm/engine.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/engine.py), [basic.py](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/basic.py) around "The init_score will be overridden by the prediction of init_model"]
- LightGBM `Booster.refit(data, label, decay_rate=0.9, init_score=..., weight=...)` re-estimates leaf values on new data keeping tree structure; the docs state leaf_output = refit_decay_rate x old + (1 - refit_decay_rate) x new. This is a literal "shrink new-market leaf estimates toward the Part 1 leaf values" operation, a credibility-like blend. [READ: [basic.py refit](https://raw.githubusercontent.com/microsoft/LightGBM/master/python-package/lightgbm/basic.py), [Parameters.rst refit_decay_rate](https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/Parameters.rst)]
- LightGBM `path_smooth` shrinks a node's value toward its parent with weight depending on node size n: w x (n/ps)/(n/ps+1) + w_parent/(n/ps+1) (a credibility-form shrinkage inside the tree); `monotone_constraints` with `monotone_constraints_method` (basic/intermediate/advanced) and `monotone_penalty`; `linear_tree`; `boost_from_average` defaults true for regression/binary (check how it interacts with Tweedie/Poisson when an init_score is given). `tweedie_variance_power` default 1.5, allowed 1.0 to <2.0. [READ: Parameters.rst, same URL]
- XGBoost `DMatrix(base_margin=...)`: per-row starting margin (on the link scale, before the inverse link); it overrides `base_score`; must also be supplied at predict time; documented as the GLM-offset analogue; multi-target margins are matrices. [READ: [xgboost core.py set_base_margin docstring](https://raw.githubusercontent.com/dmlc/xgboost/master/python-package/xgboost/core.py), intercept.rst]

### Inferences
- **[BG]** Ladder of options, ordered by data requirement (use the cheapest that wins in validation):
  0. **Do-nothing baseline**: Part 1 model applied as-is, with only a global level correction (multiply by A/E ratio on Part 2 training data, per line/peril). Always report it; it is the benchmark for "does adaptation add value".
  1. **Recalibration** (very small data): fit on Part 2 a Poisson/Tweedie GLM with log(Part 1 prediction) as offset and a few parameters: intercept per peril, slope on log(prediction) (slope <1 means Part 1 relativities over-dispersed in the new market), maybe a handful of segment effects. This is Platt/Cox-style recalibration on the link scale.
  2. **Offset + GBM correction**: LightGBM with `init_score = log(Part1 prediction)` (plus log exposure for frequency), few shallow trees, small learning rate, heavy `min_data_in_leaf`, `lambda_l2`, early stopping; number of rounds chosen by grouped CV. Equivalent to XGBoost `base_margin`. Zero trees must be a candidate (the model can decline to move).
  3. **Credibility blend of predictions**: final = Z x new-market model + (1-Z) x Part-1-adapted model, with Z per peril/segment chosen by Buhlmann-Straub style K = sigma^2_within / tau^2_between estimated from fold-level A/E dispersion, or Z = n/(n+K) with K tuned in CV. Blend on the log scale for multiplicative models.
  4. **Refit from scratch** on Part 2 with a GLM (elastic net / splines / few interactions) or small GBM with monotone constraints and strong regularisation; use it as the "independent" component of the blend and as the fall-back if Part 1 priors are shown biased.
  5. **Tree-level transfer**: `Booster.refit` with a tuned `decay_rate` per peril as a cheap credibility-like option that keeps the structure; `init_model` continuation with a small number of extra trees.
  6. **Importance weighting** (domain-classifier density ratios, clipped, ESS reported) only if Part 1 data are reused as training rows for the target; here it re-weights Part 1 rows to look like Part 2, useful for stacking Part 1 rows as a down-weighted augmentation (source rows weight w_s << target rows). Treat as an experiment, not the default, because covariate-shift weighting fixes only P(x) differences.
- **[BG]** Practical choices when features are fewer: drop features absent or unreliable in Part 2 from the Part 1 model by retraining on Part 1 with the common feature set (so the prior is not conditioned on unavailable inputs). Check that the prior performs on Part 1 itself with the reduced feature set so you can attribute losses to feature loss vs market change.
- **[BG]** Per-peril, per-line granularity: perils that do not exist in Part 1 (e.g. flood/earthquake/ theft taxonomies) have no prior and need a pooled frequency x severity from Part 2 only, or a coarse prior from Part 1's nearest peril. Credibility Z should vary by peril since claim counts differ by orders of magnitude.
- **[BG]** Metric alignment: Tweedie D2 rewards getting the mean right across segments; a global level error and wrong power are both penalised. Tune `tweedie_variance_power` on Part 2 (the right value can differ in a market with different severity tails), and do not copy 1.5 blindly.
- **[BG]** Monotone constraints (age of vehicle, sum insured, deductible) are a cheap way to inject domain knowledge when data cannot identify the shape; expected cost is typically monotone in sum insured and deductible. Verify direction in Part 1 first.
- Evidence strength is weak: the key insurance result is one paper read at snippet level; the GBM transfer references are not insurance-specific and a TabPFN evaluation found no consistent win. Frame the choice in the write-up as "validated on our own folds", not as literature-proven.

### Gaps
- Full text of Dutang et al. (method details: which model, how the market features were used, sample sizes) not obtained. This is the most important paper to read when network access allows (arXiv/HAL/variancejournal).
- No benchmark found for GBM warm-start vs scratch on small insurance data; no source on LightGBM `init_score` with Tweedie objectives specifically (the `boost_from_average` interaction needs a quick empirical test).
- The Numerai forum claim that LightGBM `init_score` exists only in `fit()` is outdated or wrong for current versions: the current source shows `init_score` on `Dataset` and a handling path inside `Dataset` construction. [READ: basic.py]

---

## 3. Validation with little history (leave-one-period-out, grouped CV, bootstrap, rows needed, uncertainty and scenario reporting)

### Takeaway
I found no insurance-specific source on validation design for thin data; the guidance below is standard practice (background knowledge) combined with the sourced shift-detection tools. The key design principle is that the validation split must mimic the actual deployment gap (next period, unseen policyholders, same market), and every model-choice claim needs an uncertainty interval.

### Cited Findings
- Adversarial validation's stated rationale: if train/test AUC is near 0.5 local CV should track the leaderboard; if high, the validation scheme may overstate performance. One variant selects training rows most similar to the test set as the validation set. [SNIPPET: [KDnuggets](https://www.kdnuggets.com/2016/10/adversarial-validation-explained.html)]
- Delay and censoring matter for what the "latest period" labels mean: ignoring reporting delay biases recent-period frequency downward. [SNIPPET: [arXiv 2510.04421](https://arxiv.org/html/2510.04421v2)]
- Shift-weighting pipelines recommend clipping/normalising weights and reporting effective sample size. [SNIPPET: [shiftstat](https://pypi.org/project/shiftstat/)]

### Inferences
- **[BG]** Splits: (1) grouped K-fold by policyholder/customer id (and by address/region if the leak risk exists) on all Part 2 rows; (2) leave-one-period-out (LOPO) or expanding-window forward validation by underwriting/accident year, always applying the same completeness adjustment as at prediction time (with only 2-4 years, LOPO on the last year is the main "forward" test, the others are for variance); (3) combine: grouped by customer within forward folds. Report the mean and spread across folds, plus fold-wise paired differences between candidate models.
- **[BG]** Uncertainty: block bootstrap by customer (and by period if possible) of the whole pipeline (including tuning if affordable, otherwise fixed hyperparameters) gives intervals on D2 and on portfolio-level expected cost; paired bootstrap of model A minus model B D2 resolves "is adaptation better than the no-adaptation baseline". Because claims are rare and heavy-tailed, the sampling noise in D2 is dominated by a few large claims; report D2 with and without the largest claims or with capped severity.
- **[BG]** Rough sample-size logic: relative standard error of an estimated claim rate with N claims is about 1/sqrt(N) (Poisson). Resolving a 10% relative difference in a segment's frequency at about 2 sigma needs around 400 claims in that segment; 20% needs about 100. Overall model comparisons on aggregate D2 need far fewer, but per-factor relativities (e.g. 40 levels) need claim counts per level. Use this to decide which factors get their own relativity vs shrunken/grouped levels. Do a quick simulation: simulate Part 2 from the fitted model, repeat the pipeline, and measure how often the ranking of candidates is correct at the observed row/claim counts.
- **[BG]** Scenario sensitivity to report: severity trend (e.g. +/- 5 to 10 percentage points per year relative to base, wider if inflation is high), report-lag completeness factor (base and slow-settlement scenario), exposure-convention alternative (e.g. policy-days vs policy-months), cap level for severity, and Tweedie power. Present results as a small tornado table of the change in total expected cost and in D2.
- **[BG]** Guard against over-tuning: with few rows, hyperparameter search on the same folds you report from overstates performance; use nested or a locked final fold, and limit the number of candidates (the team already has a cap on submissions, which also serves as a guard).

### Gaps
- No source found on minimum claim counts or on LOPO design for pricing; numbers above are generic statistical rules of thumb.
- Nothing retrieved on how the organisers' scoring (Tweedie D2 on the hidden set) behaves under small test sizes; worth estimating by bootstrap on Part 1.

---

## 4. Emerging-market insurance characteristics reported in the literature, and thin-data pricing

### Takeaway
The sources found are few and mostly grey literature (IAA papers, practitioner articles). They agree that data scarcity, weak data quality and low actuarial capacity are the defining problems, that inflation/FX volatility and regulation (price caps) distort experience, and that Bayesian/credibility methods and conservative loadings are the standard responses. No peer-reviewed quantitative study of claim reporting patterns or exposure measurement in emerging markets was found.

### Cited Findings
- IAA Microinsurance Task Force issues paper (May 2014) identifies limited supply of actuarial capacity in inclusive insurance markets as a key issue and recommends more data gathering. [SNIPPET: [IAA paper page](https://actuaries.org/paper/addressing-the-gap-in-actuarial-services-in-inclusive-insurance-markets/), [PDF](https://actuaries.org/app/uploads/2025/04/Paper_Inclusive_Insurance_Markets.pdf)]
- Biener, "Pricing in Microinsurance Markets" (2011, working draft marked "do not quote"): pricing problem stems from limited data and a lack of suitable methods; Bayesian methods are a keyword; imprecise premiums risk insolvency if too low or unaffordability if uncertainty loadings are too high. [SNIPPET: [IAA Biener Madrid 2011](https://actuaries.org/app/uploads/2025/07/Biener_Paper_Madrid2011.pdf)]
- IAA-GAIN webinar (Sept 2025): in most low and middle income countries actuarial capability is weak or non-existent; data are lacking on at-risk populations and emerging climate risk, leading to mispriced products. [SNIPPET: [IAA event page](https://actuaries.org/event/joint-webinar-by-iaa-gain-on-precision-for-protection-actuarys-edge-in-microinsurance-pricing/)]
- Syed Danish Ali, CAS working paper (2017, not peer reviewed): commercial general insurance in emerging markets (UAE, Saudi Arabia), regulatory change pulling actuaries into pricing; calls for opening the "black box" of insurance operations. [SNIPPET: [CAS working paper](https://www.casact.org/sites/default/files/2021-02/working-paper-syed-danish-ali-2017-06.pdf)]
- Practitioner view (LatAm): data arrives late, incomplete or in siloed systems; pricing teams spend more time reconciling data than optimising rates. India: GI industry still relies on international actuarial resources; a non-tariff regime transition presentation exists. [SNIPPET: [FinTech Global](https://fintech.global/2025/08/13/insurance-pricing-in-latam-explained/), [Actuaries India](https://actuariesindia.org/sites/default/files/2022-05/Transition_Non-tariff_Regime_Karpagam_Sanka.PDF)]
- IFoA-CAS property per-risk working party: limited information from primary companies pushes actuaries toward conservative assumptions and higher premiums (US/global, not emerging-specific). [SNIPPET: via search result list; not opened]
- Inflation/FX: Turkey (CPI above 50% forecast by Fitch for 2022-23, price caps on MTPL limited repricing; motor drove 60% of the 1H2021 rise in paid claims); Argentina (claims inflate differently by type and legal route; devaluation interplay). [SNIPPET: see Section 1 sources]
- Reporting delay methods exist (Section 1) but no emerging-market specific delay data found.

### Inferences
- **[BG]** Likely differences to hypothesise for the hackathon dataset (a synthetic "emerging market" may be constructed to include them): lower or different motor frequency; different vehicle mix (older, motorcycles); theft/flood/fire shares in property; weaker address geocoding; sums insured in a different currency with large inflation steps (look for level jumps by year and deflate by an index before modelling, or include year effects); longer settlement and later reporting (so the last accident period is incomplete); policy duration/exposure recorded as counts of policies rather than days; fewer rating factors and higher missingness; more round-number amounts (estimated, not paid values).
- **[BG]** Thin-data pricing practice: (i) benchmark/manual rates from a mature market with local adjustment factors; (ii) credibility with a complement from a similar book (SOA/CAS credibility literature above); (iii) Bayesian hierarchical models with informative priors from the benchmark; (iv) explicit risk margins and loadings for parameter uncertainty; (v) conservative caps on rate movements and rapid experience monitoring. These map directly onto the offset + shrinkage plan in Section 2 and the scenario reporting in Section 3.

### Gaps
- No quantitative source for emerging-market claim frequency/severity differences, settlement patterns, or exposure measurement. Searches returned mostly microinsurance (life/health/agri) rather than general-insurance motor/property pricing.
- Not opened: IAA microinsurance pricing PDF (Biener), IAA issues paper full text, IFoA UK 2016 report; these may contain concrete data-scarcity methods.
- Searches on SOA/CAS "emerging markets" and "exposure measurement" produced nothing specific; try SOA Research Institute, CAS E-Forum, and ASTIN Colloquium proceedings if access is available.

---

## 5. What a judging panel will look for in an "adaptation" story (good change logs, uncertainty communication)

### Takeaway
I found no published judging rubric beyond the organisers' wording quoted in the task ("identify what has changed, respond to those differences, and communicate any remaining uncertainty"). The suggestions below are inferences from that wording plus the ASOP 23/56 emphasis on documentation, limitations and disclosure. Treat as hypotheses.

### Cited Findings
- ASOP 23 structure includes documentation and communications/disclosures sections; ASOP 56 is concerned with model appropriateness, data quality and output validation. [SNIPPET: [ASB](https://www.actuarialstandardsboard.org/asops/asop-no-23-revision-data-quality/), [actuary.info](https://actuary.info/insights/cas-ai-primer-2026-review)]
- Imprecise premiums create asymmetric risks (insolvency if under-priced, unaffordability if over-loaded), which is the framing used in the micro-insurance pricing literature for uncertainty communication. [SNIPPET: Biener, link in Section 4]
- The organisers' Part 2 brief: the new dataset "may contain different characteristics and practical challenges"; the task is to determine how the approach should adapt. (From the user's task statement; I did not independently fetch the official rules page.)

### Inferences
- **[BG]** Structure of a strong adaptation story: (1) "What we expected vs what we found" table with evidence and magnitude; (2) a decision for each difference (ignore / adjust / re-model / flag) with the reason and the validation result that justified it; (3) a head-to-head table: Part 1 model as-is, recalibrated, offset-GBM, scratch GLM, blend, with D2 and paired-bootstrap intervals; (4) a statement of what was deliberately not changed and why; (5) residual-uncertainty section: scenario table (trend, lag, exposure, cap), segment-level credibility weights, and where the model should not be trusted (new perils, sparse levels); (6) monitoring plan: what to check when more experience emerges, with triggers. This follows the organisers' three verbs.
- **[BG]** Good change-log entry example: "ID C7 | Claims | Report delay: median 41 days vs 9 in Part 1 for contents theft (n=212 claims); 18% of last-quarter claims likely unreported | Action: completeness factor 1.11 for last accident quarter (empirical lag CDF), scenario +/-4% | Impact: +2.3% total expected cost | Residual: lag CDF estimated from 3 years, CI 1.06-1.17."
- **[BG]** Uncertainty communication: give ranges (bootstrap 80%/95%) for portfolio total and for key segments, not just point predictions; show a calibration (A/E by decile) chart with intervals; say which conclusions are robust across scenarios and which are not; avoid false precision in the tail severity; disclose reliance on unverifiable data assumptions (ASOP 23 spirit).
- **[BG]** Avoid: claiming a large improvement from a single fold; tuning on the final fold; hiding that the do-nothing baseline was close; unexplained feature removals.

### Gaps
- No actual hackathon scoring rubric or past winning submissions retrieved. The official IAA hackathon rules were not fetched here; check them for how the written explanation is weighted vs leaderboard score.

---

## 6. Concrete playbook (LightGBM, GLMs, drift-screen toolkit)

### Takeaway
Spend the first two hours on data audit and the baselines (as-is and recalibrated), the first day on candidate adaptation models with honest validation, and the final days on uncertainty quantification, scenario analysis and the narrative. All steps below are **[BG]** design proposals, not sourced procedures, except where the LightGBM/XGBoost behaviour is cited in Section 2.

### Cited Findings
- LightGBM mechanics used in the plan: `init_score` on `Dataset` / `init_model` continuation / `Booster.refit(decay_rate)` / `path_smooth` / `monotone_constraints` / `tweedie_variance_power`. [READ: LightGBM source and Parameters.rst, see Section 2]
- XGBoost equivalent: `base_margin` as offset. [READ: xgboost intercept.rst]
- Adversarial validation and PSI/KS/chi-square shift screens as listed in Section 1. [SNIPPET sources there]

### Inferences

**First 2 hours (audit and baselines)**
1. Freeze Part 1 artefacts: model files, feature list, caps, trend and completeness factors, preprocessing code, the commit hash. Create `change_log.md` (table: id, area, finding, evidence, impact, action, residual uncertainty, status).
2. Load Part 2; run the schema/unit/coding audit script (Section 1 steps 1-3): column diff vs Part 1, dtype, cardinality, level overlap, sentinel values, date ranges, exposure sum vs calendar span, join integrity (claims to policies), duplicates. Log everything, even benign.
3. Run the existing drift-screen toolkit Part 1 vs Part 2: PSI/KS per feature, domain-classifier AUC with grouped CV and top features. Separately compute outcome-side tables per line/peril: frequency with CI, severity quantiles and cap-hit rate, report-lag CDF, zero-exposure/short-exposure counts. Do not use outcome fields as classifier features in the shift test.
4. Score the unmodified Part 1 model (restricted to available features, with missing-feature handling noted) on Part 2: A/E by line/peril, by decile, by key factors; D2 on the latest period via forward split. This is baseline 0. Add baseline 1: global per-peril multiplier (recalibration intercept only).
5. Decide the unit of analysis (customer-level groups for CV, accident year for forward splits) and write the validation protocol before looking at model results.
6. Triage: list the top 5 changes by estimated impact on expected cost; assign owners.

**First day (adaptation candidates)**
7. Candidate A: GLM with offset log(Part 1 prediction) [+ log exposure], intercept and slope per peril. Candidate B: LightGBM correction with `init_score` = log Part 1 prediction, shallow trees, low learning rate, large `min_data_in_leaf`, L2, `path_smooth`, early stopping by grouped CV, including the zero-tree option. Candidate C: scratch regularised GLM (elastic net with splines) on Part 2 only. Candidate D: scratch small LightGBM with monotone constraints and the same regularisation. Candidate E: credibility blend (log-scale weighted average of the Part 1-adapted model and the scratch model) with Z per peril from CV; optionally `Booster.refit` with a grid of `decay_rate`.
8. Fit report-lag completeness and trend on Part 2 (empirical lag CDF per peril; severity trend by calendar year or as a scenario parameter); re-run candidates with updated labels and exposures. Check exposure convention by testing alternatives (days vs months) via frequency stability across partial-year policies.
9. Tune `tweedie_variance_power` and severity cap on Part 2 only for the final stage.
10. Compare candidates on grouped-forward folds with paired differences; keep a short list (at most 3). Record in the change log each decision and the evidence.
11. Sanity checks (leak guards): customer/address identity leakage across folds, features computed using future claims, claim report dates after the cut-off.

**Final days (uncertainty, robustness, narrative)**
12. Block bootstrap (by customer) of the final pipeline (or fixed-hyperparameter pipeline if costly) for D2 intervals, portfolio mean and key-segment means; paired bootstrap against baseline 0 and baseline 1.
13. Scenario table: trend +/-, lag factor scenarios, exposure convention alternative, cap level, Tweedie power; show impact on total expected cost and D2. Credibility weights Z by peril/line reported.
14. Simulation check: simulate Part 2-sized data from the fitted model and confirm that the pipeline ranks models correctly at this sample size; if not, say that differences are within noise and prefer the simpler/more conservative candidate.
15. Choose the final model by pre-registered rule (best mean forward D2 within one standard error, then simplest); lock the final fold for a one-time check.
16. Write-up: change log (appendix), adaptation decisions, head-to-head table with intervals, residual uncertainty and monitoring plan, limitations (new perils, sparse levels, reliance on assumptions). Keep the sections aligned with the three verbs (identify, respond, communicate uncertainty).
17. Reproducibility: seeds, versions, a single `run_all` script, saved intermediate tables that back each change-log entry.

### Gaps
- Timing for each step depends on the dataset size and submission limits (not known). Dataset specifics (row counts, which factors are missing) will change which candidates are viable.
- Not verified: whether the organisers allow external or Part 1 data for training in Part 2 (affects the use of importance-weighted Part 1 rows), or whether Part 1 predictions on the Part 2 data may be reused. Check the rules page.
