# Validation strategy and leaderboard practice in noisy, heavy-tailed Kaggle competitions

**Provenance warning (read first).** In this session the egress proxy blocked direct fetching of arxiv.org, papers.nips.cc, proceedings.mlr.press, Kaggle discussion pages (SPA), discourse.aicrowd.com, oreilly.com, slideshare, towardsdatascience, ceur-ws.org and others (WebFetch failed with DNS errors; curl got HTTP 403 on CONNECT). I did NOT read any source in full. Every "Cited Finding" below comes from a WebSearch result summary (a model-written digest of the page, with short quotes), not from the page itself. Treat each as a snippet-level claim to be verified against the linked source. No numbers appear below except ones that appeared in those digests. Statements labelled "Inference" or "Background (unverified)" are my own reasoning or prior knowledge and carry no citation. I did not find a Kaggle write-up for a Tweedie or heavy-tailed insurance competition that discusses shake-up in detail.

## 1. Public-vs-private shake-up behaviour in noisy, heavy-tailed metrics (magnitudes, who moves, published analyses)

### Takeaway
The best published analysis (Roelofs et al., NeurIPS 2019, 120 Kaggle competitions) found little evidence of adaptive overfitting to the public leaderboard. Big rank changes mostly come from crowded rankings where score differences are within statistical noise. For our problem that means the main risk is noise from a small, heavy-tailed holdout and train-to-test shift, not necessarily deliberate overfitting. I could not retrieve the paper's quantitative results.

### Cited Findings
- Roelofs et al. analysed over one hundred (120) Kaggle competitions, comparing public-leaderboard rankings with final private rankings, and reported little to no sign of adaptive overfitting. Kaggle's public/private split is used as a natural experiment. — [WebSearch digest of NeurIPS 2019 paper pages](https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning); [escholarship thesis, "Measuring Generalization and Overfitting in Machine Learning"](https://escholarship.org/content/qt6j01x9mz/qt6j01x9mz_noSplash_7dbaf125dab807966668b07751835f16.pdf)
- The authors built a statistical test for classification competitions that compares submission scores with an ideal null model without overfitting. — [Roelofs et al. paper PDF (digest only)](https://papers.nips.cc/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-Paper.pdf)
- The thesis argues that distribution shift, more than overfitting, is the main concern for robust ML. — [Roelofs thesis (digest only)](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-102.pdf)
- The paper states that if many models of essentially equal performance are submitted, small variations can lead to large rank changes, but these are not problematic if each method's performance is unchanged up to statistical fluctuation. The paper reports it compared its results with the "Meta Kaggle: Competition Shake-up" notebook and found similar conclusions. — [Roelofs et al. author feedback / paper text (digest only)](https://papers.NeurIPS.cc/paper_files/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-AuthorFeedback.pdf)
- The Meta Kaggle shake-up notebook (jtrotman) measures shake-up as mean absolute rank change scaled by number of teams, plus a variant for the top 10% of public ranks. I did not open the notebook, so its results are unverified. — [Dauphine lecture slides (lec-3) as described by search digest](https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/lec-3.pdf)
- The same lecture attributes most shake-ups to crowded rankings (competitors very close together, so small private-set changes reshuffle ranks), plus adaptive overfitting, very small training sets, or non-i.i.d. data. — [Dauphine lecture slides](https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/lec-3.pdf)
- Examples of big movers: in Coleridge ("Show US the Data"), one team reportedly rose more than 1,400 places to finish 47th of 1,610 teams. The write-up attributes shake-ups generally to public-LB overfitting and weak CV. In the Persuade essay competition, teams that clustered essays by topic and over-optimised were reportedly shaken down by hundreds of places on the "truly private" set. — [nbroad Coleridge reflections](https://nbroad-me.hf.space/blog/kaggle_coleridge) (Persuade claim comes via a search digest of the same results set; unverified)
- Simulation by an AICrowd participant: a model's score on random test subsets of similar size can differ by as much as 10% in some cases; strong models likely still rank well, but models close together on the board can move a lot; conclusion "local CV is a must". Informal, not peer reviewed. — [AICrowd: Simulating leaderboard shakeups](https://discourse.aicrowd.com/t/simulating-leaderboard-shakeups-to-give-some-context/5917)
- Heavy-tailed or noisy target: in ASHRAE Great Energy Predictor III, there was extensive discussion on handling outliers in the target; top competitors removed noisy points mechanically or manually, and generalisation was a key concern. — [ASHRAE overview summary](https://book.st-hakky.com/en/data-science/ashrae-overview)
- Mania et al. (NeurIPS 2019) argue that models submitted to competitions are highly similar in their predictions, which limits how much test-set reuse overfits; the meta-review says this holds for models used in Kaggle competitions. — [Model Similarity Mitigates Test Set Overuse](https://arxiv.org/pdf/1905.12580); [NeurIPS meta-review](https://papers.neurips.cc/paper_files/paper/2019/file/48237d9f2dea8c74c2a72126cf63d933-MetaReview.html)

### Inferences
- With public score gaps of 0.001-0.004 among top teams and our own unpaired CI of about +-0.014, the top of our leaderboard is a "crowded ranking" in the Roelofs sense: expect large rank reshuffles at the private stage largely from noise, regardless of anyone's tuning.
- Because the private ~67% subset overlaps in distribution with the public ~33% (random row split, same future year), the public-private gap is dominated by sampling noise of a few large claims, not systematic shift. Shift between train years and the future test year affects both subsets equally and cannot be learned from the public score alone.
- Teams that moved up in published accounts are described as having trusted CV and made few submissions (see section 5); the evidence for "ensembles move up" is anecdotal in what I saw.

### Gaps
- No numeric shake-up magnitudes (median rank change, score-gap statistics, effect of test-set size) from Roelofs et al. or Meta Kaggle were retrievable; the paper PDF was blocked.
- No shake-up analysis of an insurance/Tweedie/heavy-tailed-loss competition (e.g. Allstate Claims Severity 2016, Allstate Claim Prediction 2011, Liberty Mutual, Porto Seguro, Travelers) was found. Searches returned only student write-ups and descriptions.
- Famous heavy-tail-driven shake-ups: none verified.

## 2. Time-forward validation for next-period extrapolation (folds, gaps, label maturity, fold weighting, adversarial validation)

### Takeaway
Sources agree that when there is temporal dependence, validation must be out-of-time, with the validation period after training, and that adversarial validation is the standard train/test-shift check. I found no source addressing label maturity, gapped splits or fold weighting specifically; those parts are my reasoning only.

### Cited Findings
- Never use random splits when temporal dependence exists; use time-based splits where validation always follows training. One competitor who dropped on the private board attributed it to a time-based split with different later-period conditions. — [Search digest of Kaggle-related blogs, e.g. Kaggle Handbook / ML Journey](https://mljourney.com/kaggle-model-selection-techniques-explained/); [Global Maksimum, "Fundamentals to Survive a Kaggle Shake-up"](https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-fundamentals-to-survive-a-kaggle-shake-up-3dec0c085bc8)
- Adversarial validation is a binary classifier that tests whether train and test come from similar distributions. — [Kaggle Q&A thread / digest](https://www.kaggle.com/discussions/questions-and-answers/113061)
- Owen Zhang's widely shared slides recommend validation be as realistic as possible, usually out-of-time, and warn that the more you tweak the model, the bigger the public-private gap (as reported by the search digest; slides not opened). — [Owen Zhang, Tips for data science competitions](https://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions)
- In the AICrowd insurance pricing game, a 2nd-place write-up reports 5-fold CV for new-business models and a train/test split by year for renewals (digest). — [AICrowd 2nd place solution](https://discourse.aicrowd.com/t/2nd-place-solution/5193)
- The Kaggle Book has a chapter "Designing good validation" with a section on choosing final submissions and states the public leaderboard is often misleading; the body text was paywalled. — [Kaggle Book ch. 6 (O'Reilly)](https://www.oreilly.com/library/view/the-kaggle-book/9781801817479/Text/Chapter_6.xhtml); [Packt chapter page](https://packtpub.com/en-jp/product/the-kaggle-book-9781801817479/chapter/designing-good-validation-8/section/designing-good-validation-8)

### Inferences
- Our two expanding-window folds (2021-22 -> 2023; 2021-23 -> 2024) match the standard "validation always after training" practice. The test year's maturity (claims reported by the evaluation date) should be matched by truncating validation-year claims to the same development age; unmatched maturity biases the level more than the ranking.
- Adversarial validation (train vs test features, ideally also year-N vs year-N+1 inside train) is cheap and gives a list of drifting features to drop or de-emphasise. Its AUC says nothing about target drift (claim frequency/severity trend), which must be handled separately.
- Weighting folds toward the most recent (2024) is defensible for a next-year target but raises variance; report both folds and the pooled paired difference.

### Gaps
- No source found on gapped/purged splits, label-maturity matching, or fold weighting in Kaggle practice (Background, unverified: purged/embargoed CV is standard in finance, e.g. Lopez de Prado).
- Adversarial validation write-ups with numbers were not retrieved.

## 3. Statistical comparison of models on the same rows (paired bootstrap, permutation tests, seed noise, sample size for Tweedie deviance)

### Takeaway
I found no authoritative source on paired resampling of Tweedie deviance differences, so the recommended procedure is general statistical practice, not something I verified. Searches confirm that Tweedie deviance on a holdout is the standard comparison yardstick in the actuarial literature and that bootstrap/Vuong-type tools exist, but nothing on required sample sizes.

### Cited Findings
- Insurance papers compare models on out-of-sample Tweedie deviance as the (large-sample) predictive loss and check train vs test deviance gaps for overfitting. — [Autocalibration and Tweedie-dominance (arXiv 2103.03635)](https://arxiv.org/pdf/2103.03635) (digest only)
- For penalised Tweedie/Poisson/logistic regressions the paired (case) bootstrap was reported to work better than the residual bootstrap for coefficient intervals; this is about coefficients, not model-vs-model score differences. — [Interval Estimation of Coefficients in Penalized Regression Models of Insurance Data (arXiv 2410.01008)](https://arxiv.org/pdf/2410.01008) (digest only)
- The Vuong test for non-nested models is used alongside deviance in a zero-inflated Tweedie study. — [Zero-Inflated Tweedie Boosted Trees with CatBoost (arXiv 2406.16206)](https://arxiv.org/pdf/2406.16206) (digest only)
- Mania et al. show that models that agree closely in their predictions limit overfitting from reuse of a test set; high model similarity is also what makes paired comparison tighter than unpaired. — [Model Similarity Mitigates Test Set Overuse](https://arxiv.org/pdf/1905.12580) (digest only)

### Inferences
- Pairing removes the shared-row component of variance. Our finding that unpaired CIs (+-0.014) cannot resolve gains under about 0.01, while paired CIs are narrower, is consistent with this. The paired difference variance depends on how similar the two models' per-row losses are; for a GLM vs GBM pair on the same rows it is often far smaller than unpaired.
- Heavy tails: a handful of rows (the 11.9M claim) contribute a big share of the deviance sum, so the paired difference d_i = loss_A,i - loss_B,i has a heavy-tailed distribution and the bootstrap CI is dominated by whether those rows are resampled. Recommended reporting (my reasoning, unverified): (a) the paired bootstrap on the full metric, clustered by customer; (b) the same with row costs capped/winsorised; (c) the same excluding the top-k claims, and a leave-one-giant-claim-out table showing the sign of the difference. A model preference that flips on one claim should be declared "unresolved".
- Permutation (sign-flip) tests on per-cluster paired differences are a reasonable alternative to the bootstrap and are exact under symmetry of d_i; with very heavy tails a trimmed-mean or median-of-clusters statistic is more stable. Check BCa vs percentile intervals (a digest suggested this too, unsourced).
- DeLong-style tests are specific to AUC/ROC and do not transfer directly to Tweedie deviance; for Gini-type ranking metrics the analogous approach is a U-statistic/bootstrap on clusters.
- Seed noise: repeat each model with several seeds and report the seed-to-seed standard deviation of the validation metric; any gain smaller than this is not a gain. This is standard practice (unverified for this metric here).
- Sample size: since gains under about 0.01 are unresolved with a few years of data, small improvements can be accepted only on structural grounds (prior plausibility, consistency across both folds and across subsets), not on the point estimate.

### Gaps
- No source with numbers for how many rows/years are needed to resolve small gains in a Tweedie D2 metric.
- No literature found specifically on paired bootstrap for deviance differences, or on cluster bootstrap for pooled D2.

## 4. Leaderboard probing for level/scale; how much can be learned from public feedback

### Takeaway
Probing the public leaderboard is a long-standing technique but contested. Theory says the overfitting from k adaptive queries scales roughly with sqrt(k/n) (per a lecture digest), and the Ladder paper shows that standard Kaggle-style feedback can be attacked. For a few low-dimensional parameters (level/scale) this is a small risk in principle, but with a 33% subset dominated by a few large claims the measured level is mostly a measure of those claims.

### Cited Findings
- Leaderboard probing (Timonin tutorial) is described as a competition-specific technique, in some cases exploiting leakage to recover public-LB labels, in other cases revealing information about the distribution of public-LB y-values. — [Timonin LB probing tutorial (nbviewer)](https://nbviewer.jupyter.org/github/Dene33/mlcourse.ai/blob/master/jupyter_english/tutorials/kaggle_leaderboard_probing_nikolai_timonin.ipynb); [How to LB probe on Kaggle](https://towardsdatascience.com/how-to-lb-probe-on-kaggle-c0aa21458bfe)
- Some hosts ban probing outright (ADDI called it an unfair practice that undermines the contest); a Zindi thread notes that limited score precision made it hard to pin down a bias with only 40 submissions. — [AICrowd ADDI thread](https://discourse.aicrowd.com/t/welcome-to-the-addi-alzheimers-detection-challenge/3896/15); [Zindi TAHMO thread](https://www.zindi.africa/competitions/tahmo-incoming-solar-radiation-prediction-challenge/discussions/33205). Check the rules of our hackathon before probing.
- Blum and Hardt (Ladder, ICML 2015): repeated submission lets participants overfit the holdout; the standard Kaggle mechanism (rounding of scores) is vulnerable to a boosting attack that inflates public scores, especially with small rounding; the Ladder only updates the shown score when a submission is significantly better than the previous best. — [The Ladder (arXiv 1502.04585)](https://arxiv.org/pdf/1502.04585); [PMLR page](https://proceedings.mlr.press/v37/blum15.html)
- A Columbia slide deck gives the public-vs-true performance gap as growing roughly O(sqrt(k/n)) for k submissions on n test points (digest; formula not verified against the paper's exact statement). — [Columbia COMS 6998 slides on leaderboards](https://www.cs.columbia.edu/~djhsu/coms6998-f17/student-work/slides/leaderboards.pdf)
- Related: "Toward a Better Understanding of Leaderboard" (arXiv 1510.03349) discusses leaderboard accuracy; content not read. — [ar5iv 1510.03349](https://ar5iv.labs.arxiv.org/html/1510.03349)
- A Kaggle-tips deck claims the fewer submissions made, the less likely public-LB overfit. — [Dauphine lecture slides](https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/lec-3.pdf)

### Inferences
- Is level probing "standard"? I found no source in which top competitors describe scale/level calibration against the public LB for a Tweedie-type metric. Probing in the literature is mostly about label leakage. Do not claim it is standard practice.
- Information budget (reasoning): each submission yields one real-valued score on the public rows. Estimating a single multiplicative scale factor from a handful of submissions uses only a few degrees of freedom, so adaptive-overfitting risk (sqrt(k/n)-type) is small compared with the sampling noise of the score. The real problem is that the public score's optimum scale is determined by the few largest claims in the 33% subset; the private 67% has different large claims, so the public-optimal scale has a large standard error relative to the true optimal scale. Shrink the public-implied scale toward the CV-implied scale (for example a precision-weighted average using a cluster-bootstrap estimate of each one's variance).
- Capped-cost robust scoring on validation is the right internal check; consider estimating the public-implied level on a capped version of what is probed only if the host metric allows it (it does not; the host metric is uncapped, so the probe necessarily includes tail noise).
- If you use public feedback, restrict to a very small number of parameters (level, maybe one tail-shape parameter), pre-register the grid, and keep all other modelling decisions CV-only.

### Gaps
- No Grandmaster statement on probing scale/mean for deviance-type metrics was found.
- The exact boosting-attack query counts and bounds from the Ladder paper were not read.

## 5. Final-submission selection (CV vs public vs blend; hedging; what worked)

### Takeaway
The anecdotal consensus in what I found is to select by CV and use the public LB as a sanity check/correlation check, with some advice to hedge between a CV-best and a diverse alternative. Evidence is anecdotal; I did not see a rigorous study of selection rules.

### Cited Findings
- A Jigsaw competitor who skipped a high-CV submission for a higher public-LB one regretted it and resolved to "trust my CV"; another write-up says the Jigsaw winner submitted only two submissions and improved the model only by CV. — [Search digest of Kaggle write-ups (Global Maksimum handbook posts)](https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-tips-tricks-to-survive-a-kaggle-shake-up-23675beed05e)
- A Playground Series S6E2 1st-place write-up ("Diversity, Selection, and Trusting the CV-LB Relation") reportedly did not pick the single highest-CV submission, because of suspected split overfitting, and instead chose from a narrow band of CV scores; title also stresses diversity. Not read in full. — [PS S6E2 1st place](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t)
- Slide deck: local CV is better than public LB "90%+ of the time" and suggests blending the two, e.g. 50/50 weighting. The 90% figure is anecdotal. — [Dauphine / tips deck via digest](https://pt.slideshare.net/slideshow/tips-and-tricks-to-win-kaggle-data-science-competitions/72909469)
- One author uses the public LB "as a lodestar but not the target" and looks for a positive CV-LB correlation; a blog gives a 0.85 correlation heuristic for trusting local validation (unverified). — [ML Journey](https://mljourney.com/kaggle-model-selection-techniques-explained/); [Kaggle journey blog](https://amontgomerie.github.io/2022/02/10/kaggle-journey.html)
- Kaggle Book chapter on validation says trusting the public LB "is not advisable because it is quite often misleading". — [Kaggle Book ch. 6](https://www.oreilly.com/library/view/the-kaggle-book/9781801817479/Text/Chapter_6.xhtml)

### Inferences
- We pick one final submission (not two), so hedging must happen inside the model, not across entries: choose the blend/ensemble with the best CV that is not an outlier in the CV band, with level shrunk between CV and public estimates. A robust ensemble of a GLM and a GBM, whose per-row losses are partly independent, is expected to have lower variance on the unseen 67% than either component; this is the logic behind "ensembles move up", which I did not see rigorously documented.
- Use the 10-per-day submissions to measure the CV-LB relationship over many candidates (a rank correlation plot), not to climb the board. If CV and public disagree by less than the paired-bootstrap CI, ignore the disagreement.
- Decision rule suggestion (my reasoning): final = argmax over candidates of robust-CV score (pooled folds, capped cost), with ties within the paired CI broken in favour of the simpler/more diverse blend; public score only used to veto candidates that are clearly worse (outside its noise band) or to calibrate level with shrinkage.

### Gaps
- Not found: a quantitative comparison of selection rules (CV vs LB vs blend) from real competitions; the Kaggle Book section on final-submission strategy body text was paywalled.
- The text of the PS S6E2 write-up was not read beyond the digest.

## 6. Practical checklists from Grandmasters relevant to our situation

### Takeaway
The recurring advice is: build validation that mimics the test split (out-of-time here), trust CV over public LB, limit tuning and submissions, check adversarial validation, ensemble diverse models, and treat score differences within noise as ties. I could not access the book texts, so this list is assembled from digests.

### Cited Findings
- Kaggle Book ch. 6 "Designing good validation" includes: probing the public LB, adversarial validation, handling different distributions between train and test, and strategies for choosing final submissions (outline only; author list not verified). — [Packt chapter outline](https://packtpub.com/en-jp/product/the-kaggle-book-9781801817479/chapter/designing-good-validation-8/section/designing-good-validation-8)
- Owen Zhang: validation as realistic as possible (out-of-time); the more you tweak, the larger the public-private gap. — [Slides](https://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions)
- Datacamp course summary: Kaggle computes the metric on the whole test set but displays only the public part; the private part decides the standings; shake-ups are linked to train/test differences and public/private differences. — [DataCamp](https://campus.datacamp.com/courses/winning-a-kaggle-competition-in-python/kaggle-competitions-process?ex=8)
- "Approaching (Almost) Any ML Problem" (Thakur): I did not retrieve any content on it in this session. Not cited.
- AICrowd pricing game 2nd place: most participants looked at Tweedie deviance rather than RMSE on the leaderboard. — [AICrowd 2nd place](https://discourse.aicrowd.com/t/2nd-place-solution/5193)
- Liberty Mutual fire-loss competition: a CAS presentation reports a Tweedie (p=1.5) GLM with public-LB Gini about 0.40 and a GLMM about 0.41, with ensembling discussed. Possibly the closest domain precedent; I did not see private scores. — [CAS RPM 2015 handout](https://casact.org/sites/default/files/presentation/rpm_2015_handouts_paper_3896_handout_2468_0.pdf)

### Inferences
- Checklist for our case (all inferred, not verified): (1) report all candidate comparisons as paired, customer-clustered bootstrap differences on both folds and flag any decision that depends on a single claim; (2) keep a seed-noise floor; (3) adversarial-validate train vs test and year-over-year; (4) cap submissions used for exploration and log CV-vs-public for all of them; (5) estimate level using CV (maturity-matched), shrink toward the public-implied level by relative variance; (6) final choice is a pre-declared rule, written down before the last submissions.

### Gaps
- Primary texts (Kaggle Book, Thakur, individual Grandmaster interviews/write-ups such as from Titericz, Deotte, Zhang) were not read.
- No Kaggle write-ups for heavy-tailed insurance targets with a private-LB postmortem were found; Allstate Claims Severity (2016) and Allstate Claim Prediction (2011) were identified as comparable but not analysed.
