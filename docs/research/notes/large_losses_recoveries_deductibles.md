# Large losses, catastrophes, recoveries (salvage/subrogation) and deductibles in loss development and ratemaking

> **Evidence-quality warning for the report writer.** In this sandbox every WebFetch call failed (DNS/proxy 403 for casact.org, battleacts.pro.wiki, actuary.org, milliman.com, variancejournal.org, metricgate.com). Every cited finding below therefore comes from **WebSearch result summaries (snippets and secondary summaries)**, not from primary text I read in full. I did not read Werner & Modlin, Friedland, the Grace paper, the Riegel paper or ASOP 39 directly. Where a claim is labelled "(snippet)" it is what the search summary said the page says. Items under "Inferences" that carry no link are standard actuarial practice from my own background knowledge and are **not source-verified in this session**; treat them as hypotheses to confirm before publication.

---

## 1. Large-loss treatment in ratemaking (capping, ELFs, loading back, choosing the cap, per-occurrence vs per-claim, heavy-tailed BI)

### Takeaway
The standard approach is to cap (limit) each loss at a threshold, develop and trend the capped losses (which are far more stable), then load back an expected excess provision via an excess ratio or excess loss factor, dividing capped losses by (1 - excess ratio). Threshold choices are a judgement (basic limit, reinsurance retention, percentile of the size distribution, % of coverage limit, benchmark). For motor third-party bodily injury the capped body and the excess tail should be treated as separate components.

### Cited Findings
- Werner & Modlin (5th ed., 2016) shock-loss handling per a BattleActs study guide: remove losses above a threshold and replace them with an average expected large-loss amount. The guide defines an excess ratio as losses above the threshold divided by non-excess losses. Candidate thresholds listed: the basic limit, reinsurance limits, a percentage of coverage, the 99th percentile of loss size, or an industry benchmark. (snippet, secondary study-guide summary) — [BattleActs Werner06 Loss & LAE](https://battleacts.pro.wiki/wiki/Werner06.LossLAE)
- NCCI (workers' compensation) large-loss procedure, described in a CAS Variance article by Daley: reported losses are capped at a dollar threshold by subtracting, claim by claim, the dollars in excess of the threshold from the state's aggregate unlimited losses. The **limited losses are then developed with limited loss development factors, trended, and divided by (1 - XS)**, where XS is the excess ratio for that threshold. (snippet) — [Variance: Catastrophes and Workers Compensation Ratemaking](https://variancejournal.org/article/84910-catastrophes-and-workers-compensation-ratemaking)
- Basic limits and ILFs: the lowest limit offered is typically the "basic limit"; higher limits are "increased limits". An ILF is the ratio of expected pure premium at limit k to expected pure premium at the basic limit, with frequency assumed independent of severity and limit. Large volatile losses are capped and excluded from the analysis. Higher-limit experience has less volume and is highly variable. (snippet, ISO/CAS presentation) — [CAS RPM 2009 Thorpe handout](https://casact.org/sites/default/files/presentation/rpm_2009_handouts_thorpe.pdf)
- Trend differs by layer: excess losses tend to rise faster than overall loss costs, so basic-limit costs rise more slowly (1981 CAS discussion note on basic limits trend factors). (snippet) — [A Note on Basic Limits Trend Factors [Discussion]](https://www.casact.org/abstract/note-basic-limits-trend-factors-discussion)
- Burn costs in excess layers are extremely volatile and highly right-skewed, which limits use of linear credibility; a 2017 CAS paper proposes Bayesian credibility on severity distributions instead. (snippet) — [An Alternative Approach to Credibility for Large Account and Excess of Loss Treaty Pricing](https://www.casact.org/abstract/alternative-approach-credibility-large-account-and-excess-loss-treaty-pricing)
- A CAS-seminar GLM design separates the tail: model frequency, capped severity (gamma), and the propensity of an excess claim (binomial), then add an excess-severity model; expected loss cost = frequency x capped severity + P(excess) x excess severity. Another CAS comparison found frequency-severity and loss-cost GLM relativities are fairly close when both use the same pre-model treatments, including loss capping and exposure adjustments. (snippet) — [CAS RPM 2011 Yan handout](https://www.casact.org/sites/default/files/presentation/rpm_2011_handouts_pm13-yan.pdf); also [CAS RPM 2012 session 4738](https://casact.org/sites/default/files/presentation/rpm_2012_handouts_session_4738_presentation_1068_0.pdf) (which of the Yan/RPM handouts holds which statement was not distinguishable from the snippets)
- Heavy-tailed severity: a 2022 paper argues actuarial severity data are heavy-tailed with outliers hard to separate from genuine claims, that plain MLE gives non-robust estimators, and proposes trimmed (T) and winsorized (W) estimators (winsorizing is closely related to capping). (snippet) — [Robust Estimation of Loss Models for Truncated and Censored Severity Data (arXiv 2202.13000)](https://arxiv.org/pdf/2202.13000)
- A Fundacion MAPFRE-indexed paper using a motor third-party liability portfolio (2004-2014) reports heavier-tailed mixtures outperforming a standard Tweedie set-up; Tweedie reserving models are compound Poisson-Gamma, which struggle with heavy tails. (snippet) — [Beyond the Tweedie reserving model](https://documentacion.fundacionmapfre.org/documentacion/bib/163311.do?format=xml)
- A 2018 spliced-distribution regression study reports spliced models beat Tweedie regression on tail fit and prediction accuracy (abstract only seen). — [Fat-tailed regression modeling with spliced distributions](https://documentacion.fundacionmapfre.org/documentacion/bib/167971.do?format=xml)

### Inferences
- (Source-verified pieces combined) Because capped losses are trended and developed separately from the excess portion, the expected cost per policy = capped expected cost / (1 - XS), or equivalently capped cost + excess load. A data scientist modelling a capped target must add back the excess component or the prediction is biased low by roughly the excess ratio.
- (Background knowledge, not verified here) Per-occurrence capping is the common ratemaking basis for liability lines where several claimants or coverages arise from one event; per-claimant (per-person) capping is used when injuries are separate. For motor BI in a dataset keyed by claim, the practical choice is dictated by whether the data identify occurrences. Cap level is usually chosen so that a stable minority (often on the order of 1-5% of claims or a basic policy limit) is excess; the NCCI/Werner lists above are the only concrete options I could verify.
- (Background knowledge) With heavy-tailed BI, a single un-capped claim can dominate cell-level pure premium, so GLM relativities fit to un-capped targets mostly track where the big claims happened to fall; capping plus a flat or credibility-weighted excess load usually stabilises relativities, at the cost of assuming the excess share is roughly uniform across rating cells (which should be tested, e.g. by vehicle/driver class).

### Gaps
- Primary text of Werner & Modlin chapter 6 (shock-loss section) not read; only a study-guide summary. Specific recommended threshold rules (e.g. exact percentile guidance) not confirmed.
- No source found giving a BI-specific cap value or a per-occurrence vs per-claim comparison for motor third-party injury.
- No source obtained quantifying how much capping biases the conditional mean in GLMs.

---

## 2. Large-loss development vs attritional development and how actuaries separate them

### Takeaway
Large losses develop differently (slower reporting, long settlement, case reserves jump), so chain-ladder factors from a blended triangle misstate both pieces. Common practice is to split at a threshold and develop separately (Riegel's bifurcation approach, two sets of factors), but threshold choice is often arbitrary and large losses not yet over the threshold distort immature years.

### Cited Findings
- Riegel (ASTIN Bulletin 2014) "bifurcation approach for attritional and large losses in chain ladder calculations": a chain-ladder-style IBNR method using claims payment data for attritional claims and claims incurred data for large losses, applied to a German motor third-party liability portfolio; a later Riegel paper analyses a quasi-additive version based on Mack's model. (snippet) — [IDEAS/RePEc: ASTIN Bulletin 44(1)](https://ideas.repec.org/a/cup/astinb/v44y2014i01p127-172_00.html); [Cambridge ASTIN Bulletin page](https://resolve-he.cambridge.org/core/journals/astin-bulletin-journal-of-the-iaa/article/bifurcation-approach-for-attritional-and-large-losses-in-chain-ladder-calculations/02C6C6739F84D9885DB780F0C6A289FF)
- Threshold selection: the threshold is usually set by the reserving actuary and is often arbitrary or externally imposed; a Milliman white paper proposes choosing a range of thresholds using chain-ladder/Mack stability diagnostics (linearity vs exposure, uncorrelated link ratios, calendar effects). (snippet) — [Milliman: Applications of data science in non-life reserving](https://milliman.com/-/media/milliman/importedfiles/ektron/applications-data-science-non-life-reserving.ashx)
- Why separation is hard: for recent accident years a higher share of eventual large losses has **not yet crossed the threshold**, whereas for old years most are already identified; an alternative split is by settlement speed (fast-reported-and-settled = attritional, slow = long development). (snippet, UCLouvain working paper) — [UCLouvain DIAL working paper](https://research.dial.uclouvain.be/bitstreams/8d7bd8a2-df50-4bca-b08e-79a7ab469134/download)
- Adams (CAS 2013): if development factors come from triangles with per-occurrence limits, removing large losses before applying the factors understates ultimate and reserve; suggests two sets of factors when large losses can be separated. (snippet) — [CAS Forum 2013 Adams](https://casact.org/sites/default/files/database/forum_13fforum_03-adams.pdf); [Adams handout](https://casact.org/sites/default/files/document/adams_1.pdf)
- A 2006 CAS Forum paper proposes projecting individual large claims separately from aggregate attritional losses and then combining for variability. (snippet) — [CAS Forum Fall 2006](https://www.casact.org/sites/default/files/database/forum_06fforum_209.pdf)
- Ground-up development factors understate the development left in an excess layer because large claims report and settle later (a general reference page, low authority). (snippet) — [MetricGate: excess loss development](https://metricgate.com/docs/excess-loss-development/)
- NCCI develops *limited* (capped) losses with *limited* LDFs, i.e. capped triangles have their own, lower and more stable, factors. (snippet) — [Variance article above](https://variancejournal.org/article/84910-catastrophes-and-workers-compensation-ratemaking)

### Inferences
- A capped-loss triangle should be developed with factors computed on the same cap; applying ground-up factors to capped data over-develops, and applying capped factors to ground-up data under-develops (follows from the Adams and NCCI findings).
- For a single-year forecast on immature data, the unobserved-large-loss problem means open attritional claims and not-yet-large claims both need development; a model trained on immature claim cost will systematically understate the heavy-tail share for recent periods unless exposure/maturity is modelled or the target is developed first.

### Gaps
- No formula-level source for estimating excess-layer development factors or for tail factors on large losses was reached.
- Riegel's paper and the Milliman diagnostics were seen only via abstract/summary.

---

## 3. Catastrophe / weather-event adjustments (removal, long-term load, single-year forecast scored on actuals)

### Takeaway
Standard practice removes actual cat losses from the experience period and substitutes an expected annual cat load (non-modelled long-term average or catastrophe model, sometimes blended), because cat years bias a short window. ASOP 39 says the presence or absence of catastrophes in the data can bias cost estimates.

### Cited Findings
- Actuaries typically strip actual historical cat experience from the experience period and substitute expected annual cat loss for the future period, from a non-modelled approach or a cat model (Milliman write-up on California wildfire ratemaking). The California regulation requires at least 20 years of data for homeowners multiple-peril fire; 20 years of unadjusted loss history can bias wildfire projections and reflects no climate-driven frequency change. (snippet) — [Milliman: wildfire catastrophe models in California ratemaking](https://www.milliman.com/en/insight/wildfire-catastrophe-models-california-ratemaking)
- A 2002 CAS presentation: general agreement that historical cat losses should be removed in favour of, or blended with, modelled expected losses in primary ratemaking; warns the model long-term average may not match expected losses for a particular year and that double counting must be avoided. (snippet) — [CAS ratemaking seminar 2002 Rollins](https://www.casact.org/sites/default/files/presentation/ratesem_2002_handouts_rollins1.pdf)
- ASOP 39 (2000): cat losses can cause extreme volatility in historical data and generally need separate and different treatment; the older common method compared actual cat to non-cat losses over a longer period and applied that ratio to expected non-cat losses. Hurricane Andrew and Northridge showed the limits of historical-only estimates. A revision to ASOP 39 ("Treatment of Catastrophe or Extreme Event Losses in Future Cost Estimates...") went through exposure drafts in Dec 2024 and Mar 2026; finalisation status was not confirmed. (snippet) — [ASB ASOP 39 page](https://www.actuarialstandardsboard.org/asops/treatment-catastrophe-losses-propertycasualty-insurance-ratemaking/); [Academy summary](https://actuary.org/resources/treatment-of-catastrophe-losses-in-property-casualty-insurance-ratemaking/); [second exposure draft Mar 2026](https://actuary.org/wp-content/uploads/2026/05/ASOP-No.-39_second-exposure-draft_March-2026.pdf)
- Catastrophe capping (CAS 1994 Forum paper): individual cat losses are capped before a state's cat provision is calculated, with the cap chosen so only very rare occurrences are censored; a provision above the capped mean may be needed to reach the company-wide cat premium. (snippet) — [CAS Forum 1994](https://casact.org/sites/default/files/database/forum_94wforum_94wf485.pdf)
- NCCI (workers' comp) also has a separate catastrophe treatment in ratemaking (Daley, Variance). (snippet) — [Variance](https://variancejournal.org/article/84910-catastrophes-and-workers-compensation-ratemaking)

### Inferences
- A model fitted to all-in experience mixes a random cat component into non-cat relativities. If the scoring year has actual cat losses (or none), error is dominated by event timing and geography, not by rating factors; the expected cost per policy should be the non-cat model prediction plus a long-term cat load allocated by exposure/peril, and a forecast scored on realised outcomes will be systematically off in either direction in a cat/no-cat year even if perfectly calibrated in expectation.
- A data scientist should flag event-tagged claims (by date/peril/region) and either exclude from training or model as a separate peril component; otherwise a heavy-tailed cluster of claims inflates certain postcodes' relativities.
- Not verified: whether a fixed "attritional vs cat" flag exists in the target dataset.

### Gaps
- No primary text read; the 10/20/30-year averaging conventions and the cat-load allocation methods (e.g. by exposure vs by modelled AAL) were not confirmed from sources.
- Flood/hail/storm-specific guidance beyond wildfire was not found.

---

## 4. Deductibles / policy excess: net vs ground-up, loss elimination ratio, leveraged trend, development of net-of-deductible losses

### Takeaway
Losses net of a deductible are a layer of the ground-up distribution; the LER = E[X ^ d]/E[X] measures the share eliminated. With inflation, a fixed deductible erodes, so net losses grow faster than ground-up (leverage); net-of-deductible triangles develop differently from ground-up and should not share factors without checking.

### Cited Findings
- LER is the percentage decrease in the insurer's expected payment from imposing a deductible; LER(d) = E[X ^ d] / E[X]; empirical estimate = sum(min(x_i, d)) / sum(x_i). LER rises with d with diminishing returns for heavy tails. (snippets from a course text and a calculator site; lower authority for the latter) — [Loss Data Analytics (arXiv 1808.06718)](https://arxiv.org/pdf/1808.06718); [MetricGate: loss elimination ratio](https://metricgate.com/docs/loss-elimination-ratio/)
- A CAS review warns that if premium relativities for deductibles and loss-elimination relativities fall out of sync, a higher-deductible rate can be inadequate; with deductible roll-ups, approaches beyond calendar-year incurred losses become more important. (snippet) — [CAS Discussion Paper Program 1980 (80dpp207)](https://www.casact.org/pubs/dpp/dpp80/80dpp207.pdf)
- Leveraged trend: trends for higher limits generally exceed basic-limit trends; excess-layer trends usually outpace total-limit trends (CAS RPM handout); the attachment point matters most for the leveraged effect on reinsured layers, with the limit tempering it. (snippet) — [CAS RPM 2009 Thorpe](https://casact.org/sites/default/files/presentation/rpm_2009_handouts_thorpe.pdf); [CAS Forum Spring 2011 Ball/Staudt](https://www.casact.org/sites/default/files/database/forum_11spforum_ball_staudt.pdf); [The Effect of Trend on Excess of Loss Coverage](https://casact.org/abstract/effect-trend-excess-loss-coverage)
- Keatinge: inflation's leveraging at the retention is partly offset by dampening at the layer's upper limit. (snippet via search summary) — [Adjusting Size of Loss Distributions for Trend [Review]](https://casact.org/abstract/adjusting-size-loss-distributions-trend-review)
- Illustrative (broker article, not actuarial): a deductible layer's losses rose about 2% when overall trend was 8%, while a $5M umbrella layer rose 20%. As quoted in the search summary; the 2% figure for a "deductible layer" is unusual and I could not check the article. — [Lockton: Is your insurance program keeping pace with loss trends?](https://global.lockton.com/us/en/news-insights/is-your-insurance-program-keeping-pace-with-loss-trends)
- Annual-aggregate-deductible (AAD) business can distort loss-development triangles and IBNR; a CAS E-Forum paper (Connor) addresses combining with non-AAD business for factor selection. Only the snippet was seen. — [CAS Forum Spring 1997](https://www.casact.org/sites/default/files/database/forum_97spforum_97spf219.pdf)
- Claims below the deductible: NAIC market-conduct guidance counts claims closed because the amount claimed was below the deductible as closed without payment; Triple-I notes CWP claims include losses below deductibles, duplicates, uncovered losses. (snippet) — [NAIC MCAS handout](https://content.naic.org/sites/default/files/legacy/mcas_training/2018/MCAS_Property_and_Casualty-Handout.pdf); [MyChesCo/Triple-I](https://www.mychesco.com/?p=670250)

### Inferences
- If the dataset records only payments net of the excess, claims below the excess are censored/absent or appear as zero-payment closed claims, so frequency in the data depends on the excess; model frequency and severity on a consistent basis (ground-up with deductible as covariate, or truncated/left-censored severity).
- Over a multi-year trend horizon, a fixed monetary excess makes net claim cost rise faster than ground-up inflation; net-of-deductible counts of claims paying anything also rise (more claims pierce the deductible). Both effects matter when training on older years and scoring a newer one.
- Net-of-deductible amounts develop later than ground-up amounts on average (a claim must first exceed the retention) - consistent with the excess-layer development statement above but not confirmed for deductibles specifically.

### Gaps
- Werner & Modlin's deductible-relativity method (including its treatment of trend/leverage) not accessed.
- No source found on whether to develop ground-up and then net down, or develop net-of-deductible triangles directly.

---

## 5. Recoveries: salvage and subrogation (recording, anticipated vs actual, development with age, bias when only closed claims record them)

### Takeaway
Salvage and subrogation (S&S) are recoveries that reduce ultimate cost; they must be tracked as separate gross/net data, are typically modelled by development, ratio-to-paid or Bornhuetter-Ferguson-type methods, and tend to arrive late. If open claims show no recoveries yet, a net-of-recovery measure is biased high on immature claims.

### Cited Findings
- Grace (CAS Proceedings 1993/94), "Estimating Salvage and Subrogation Reserves - Adapting the Bornhuetter-Ferguson Approach": proposes an adapted BF method and compares two traditional approaches; says insurers must develop methods to estimate anticipated recoveries; notes there are not sufficient data to independently determine the amount; the IRS began requiring explicit recognition for tax; statutory rules historically did not allow anticipated recoveries in reserves, but from the 1992 Annual Statement reserves may be shown net of anticipated recoveries. (snippet) — [CAS Proceedings 93271 (Grace)](https://casact.org/pubs/proceed/proceed93/93271.pdf); [abstract](https://www.casact.org/abstract/estimating-salvage-and-subrogation-reserves-adapting-bornhuetter-ferguson-approach)
- A 1988 CAS seminar discussion says S&S "have probably been mentioned as a potential distorter of loss development patterns" and that the salvage estimate is net of collection expenses. (snippet) — [CAS Loss Reserve Seminar Transactions 1988](https://www.casact.org/pubs/CLRSTrans/1988/683.pdf)
- Friedland, Ch. 14 (Recoveries), per BattleActs study guide: you must be absolutely clear whether data are net, gross or ceded; paid S&S data may be thin at early maturities; the **ratio method** ties S&S to paid claims (and ratios can be adjusted when unreasonable); S&S can also be projected with the development method. (snippet) — [BattleActs Friedland14 Recoveries](https://battleacts.pro.wiki/wiki/Friedland14.Recoveries); [Friedland text (CAS PDF, not read)](https://www.casact.org/library/studynotes/Friedland_estimating.pdf)
- Insurer 10-K XBRL data show reserves reduced by estimated salvage and subrogation recoverable (net presentation). (snippet) — [SEC XBRL R9](https://www.sec.gov/Archives/edgar/data/9346/000000934612000005/R9.htm)
- Industry scale: NAIC-based figure of $51.6B recovered through subrogation and salvage in 2021 across auto lines; an estimated ~15% of P&C claims close without a valid subrogation opportunity identified (trade article, secondary). — [PropertyCasualty360 (Jan 2026)](https://www.propertycasualty360.com/amp/2026/01/26/the-hidden-drain-on-casualty-claims/)
- Claim systems model recoveries and "recovery reserves" as separate financial objects from payments (Guidewire ClaimCenter docs). (snippet) — [Guidewire recoveries docs](https://docs.guidewire.com/cloud/is/202603/cloudapibf/cloudAPI/ClaimCenter/financials/recoveries.html)
- Practitioner page (low authority, flagged): statutory reporting allows reserves gross or net of anticipated S&S while GAAP and tax require net. — [skills.sh subrogation page](https://www.skills.sh/open-gitagent/enterprise-skills/subrogation)

### Inferences
- The search results did **not** state directly that recoveries develop with claim age or that closed-only recording biases open claims; I infer from (i) the 1988 "distorter of development patterns" remark, (ii) Friedland's note that early-maturity S&S data are thin, and (iii) the existence of ratio/BF/development methods for S&S. Reasonable conclusion: recoveries are received after payments, so at early ages net-of-recovery cost is overstated relative to ultimate and a development (or ratio-to-paid) adjustment is needed.
- If recoveries are only recorded on closed claims, open claims have a recovery of zero in the data although some will receive one later; training a model on net amounts from immature data therefore (a) biases open-claim net cost upward and (b) makes closed vs open cohorts non-comparable. Safer: model gross cost, and model recoveries separately (probability of recovery x recovery rate, perhaps by claim type/ liability/ third-party), or flag recorded-recovery status and condition on claim status/age.
- Recoveries can be sizable relative to net cost for motor property damage; for BI they are usually minor except for contribution/subrogation against liable third parties (background knowledge, unverified).

### Gaps
- No source directly confirmed the age-pattern of recoveries (e.g. share recovered by development lag) or the closed-claim-only recording bias; no numerical age-to-age factors for S&S found.
- Grace's paper and Friedland Ch. 14 not read in full; method details (adapted BF mechanics) not captured.
- Treatment of anticipated recoveries on open claims in *ratemaking* (as opposed to reserving) not found in a citable source.

---

## 6. Declined / repudiated / closed-without-payment (CWP) claims in counts and development

### Takeaway
CWP claims are a large, heterogeneous class (below-deductible, duplicates, uncovered, withdrawn, denied, no amount owed) and must be treated consistently in claim counts, otherwise frequency and average severity both shift. Regulators define and report them; I found no citable ratemaking text on how they enter claim-count development.

### Cited Findings
- Maryland insurance regulator guidance: claims closed without payment include every claim closed with no loss payment (regardless of loss/receipt date), including cases where adjusting expense was incurred but nothing paid, and claims withdrawn by the insured or denied by the insurer. (snippet) — [Maryland Insurance Administration](https://www.apps.insurance.maryland.gov/SevereEvent/About)
- NAIC MCAS data call: CWP includes claims closed because the amount claimed was below the deductible; a reopened claim is counted as a new and distinct claim; closed claims cannot exceed received claims. (snippet) — [NAIC MCAS industry elements 2017](https://content.naic.org/sites/default/files/legacy/mcas_training/A.MCAS.PC.Industry_Elements_2017_data_year.pdf); [NAIC MCAS P&C handout](https://content.naic.org/sites/default/files/legacy/mcas_training/2018/MCAS_Property_and_Casualty-Handout.pdf)
- Illinois market-conduct ratio: claims closed without payment divided by total claims closed. (snippet) — [Illinois DOI scorecard formulas](https://insuranceqa.illinois.gov/MarketConductExam/MCASscorecardratioformulas.pdf)
- Triple-I: CWP should not be read as denial; reasons include below-deductible losses, duplicate filings, uncovered losses and no amount owed; methodologies differ across insurers. (snippet) — [MyChesCo: Insurance industry defends CWP data](https://www.mychesco.com/?p=670250)
- An academic ratemaking paper models the number of reported claims and IBNR claim counts through the reporting-delay distribution, using open and closed claim information together. (snippet) — [Cambridge Core record](https://www.cambridge.org/core/product/67BDC03A87304281CD11D5AD4A5645CD)

### Inferences
- Claim counts used for frequency should be defined consistently (e.g. all reported claims including CWP, or only claims with payment). Counting CWP in the denominator of severity drags average severity down; excluding them from numerator and denominator consistently is also valid, but the choice must match between training and scoring and between mature and immature claims.
- At early ages many open claims have zero payment and will later either pay or close without payment; so the fraction of zero-cost claims shrinks (or grows, for CWP) with age. A tweedie/zero-inflated target on immature data has more structural zeros than it will finally have. This is reasoning, not a sourced statement.
- Reopened claims: NAIC counts them as new claims for conduct reporting, which differs from actuarial practice of keeping them with the original claim (background knowledge; not verified here).

### Gaps
- No actuarial (CAS/Friedland/Werner) source found on claim-count development with CWP vs paid-claim counts, or on how declined claims are handled in frequency development triangles.

---

## 7. Practical guidance: robust fitting (Tweedie, Huber, capped target, separate large-loss model) and its interaction with the above

### Takeaway
Treat capping, large-loss modelling and recoveries as explicit data decisions before choosing a loss function. Tweedie handles zero mass plus right skew but is still mean-driven, so a few huge claims dominate; capping (winsorizing) the target or modelling a separate excess component stabilises relativities, but the excess load must be added back.

### Cited Findings
- Tweedie regression models total claim amount per exposure directly, with power p in (1, 2) (scikit-learn docs, French motor third-party liability data). — [scikit-learn Tweedie regression on insurance claims](https://scikit-learn.org/1.0/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html)
- One CAS example chose p by maximum profile log-likelihood over a grid of values (p about 1.30 in that dataset), with a smooth inverted-U likelihood. (snippet) — [CAS Spring 2017 Yan](https://casact.org/sites/default/files/presentation/spring_2017_presentations_c-12_yan.pdf)
- Capping and exposure adjustments are common pre-model treatments; frequency-severity and loss-cost GLMs give fairly close relativities once these are applied; the separate capped-severity/excess-propensity/excess-severity design is documented (see Section 1). — [CAS RPM 2011 Yan](https://www.casact.org/sites/default/files/presentation/rpm_2011_handouts_pm13-yan.pdf)
- Robust alternatives: trimmed/winsorized estimators for heavy-tailed severity (arXiv 2202.13000); spliced regression and heavy-tailed mixtures reportedly beat Tweedie on tail fit (see Section 1 links). A left-censored log-Student-t regression has also been proposed for heavy-tailed censored losses. — [Hybrid, Causal, and Stochastic Field Modeling of Censored Heavy-Tailed Insurance Losses](https://fourier.or.id/index.php/FOURIER/article/view/277)
- glum documents Poisson/Gamma/Tweedie GLMs on French motor TPL claims as a worked tutorial. — [glum French motor tutorial](https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html)

### Inferences
- Huber loss (squared error near zero, linear in the tails) is a *conditional-median-leaning* robustification: it down-weights extreme claims, so it estimates something below the conditional mean for a skewed target. Using it (or a capped target) without adding an excess load gives biased-low expected ultimate cost per policy. No source found directly supporting or testing this; it is my reasoning from the definitions.
- Recommended workflow synthesising the sections above (my recommendation, not a sourced standard): (1) work in ground-up, gross-of-recovery terms where data allow; (2) separate cat/event claims; (3) cap per claim (or per occurrence) at a threshold chosen to leave a stable excess share; (4) fit Tweedie or frequency x gamma severity on the capped target; (5) model the excess portion separately (propensity x excess severity, or pooled excess ratio); (6) adjust for development using factors matched to the cap and the deductible basis; (7) model recoveries separately; (8) evaluate on a holdout with the same maturity as the scoring set and report error with and without the largest claims, since the score on realised outcomes is dominated by a few big claims.
- Because the evaluation metric (if scored on realised cost) is dominated by big claims, capping the training target lowers variance of relativities but can worsen score on a year containing a huge claim; the trade-off should be decided by whether the metric is mean-based.

### Gaps
- No source found comparing Huber loss against capping or Tweedie on insurance pure-premium data, nor on GBM-specific tuning for heavy tails.
- No benchmark numbers on bias from capping vs separate large-loss model.
