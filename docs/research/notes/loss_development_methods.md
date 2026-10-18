# Loss development: from paid and reported losses to ultimate

**Source-access caveat (read first).** casact.org, battleacts.pro.wiki, chainladder-python docs, Wikipedia, hugginsactuarial.com and variancejournal.org were all blocked by the sandbox proxy (HTTP 403 on CONNECT / DNS failure), and WebFetch failed on every URL. Everything in "Cited Findings" therefore comes from WebSearch result summaries (snippets and AI-generated digests of the pages), not from the primary texts (Friedland, Werner & Modlin, Mack, Stanard, Berquist-Sherman, Bornhuetter-Ferguson). I did not read any primary source in full. Items under "Inferences" are standard textbook formulas I am stating from background knowledge; where a snippet corroborates them I say so, otherwise treat them as "unverified against a primary text in this session". Several cited pages are study aids or vendor sites (BattleActs, MetricGate, fiveable, LossReserves.com), which I label as such.

## 1. Definitions: paid, case reserve, reported/case-incurred, ultimate, IBNR (pure IBNR vs IBNER)

### Takeaway
Ultimate loss is the fully developed total cost once all claims (including late-reported ones) are reported and settled; reported (case-incurred) = paid + case reserves; "IBNR" is used inconsistently: narrowly for unreported claims only, or broadly for pure IBNR plus development on known claims (IBNER) plus reopened claims. Always check which definition a document uses before comparing numbers.

### Cited Findings
- Ultimate loss is the total paid for a fully developed loss, i.e. paid plus outstanding reported losses plus IBNR — [IRMI](https://www.irmi.com/term/insurance-definitions/ultimate-loss) (via search digest).
- Ultimate-loss prediction is framed as predicting total losses on a set of policies "once all claims are reported and settled" — [arXiv 2406.19903 (Bayesian loss development models)](https://web3.arxiv.org/pdf/2406.19903) (via search digest).
- A loss development factor (LDF) is a multiplier on current (paid or reported) losses to estimate ultimate; there are separate incurred and paid versions; the required reserve (outstanding) is ultimate minus paid — [Wikipedia: Loss development factor](https://en.wikipedia.org/wiki/Loss_development_factor), [Sigma Actuarial](https://www.sigmaactuary.com/?p=533) (via search digest).
- Ultimates can be hard to pin down because claims may be unreported and take years to settle; informal usage of these terms conflicts across practitioners — [Sigma Actuarial](https://www.sigmaactuary.com/?p=533) (via search digest).
- Pure IBNR = losses that have occurred but not been reported; IBNER = expected future development on already-reported claims (can be negative if case reserves are redundant) — [Huggins Actuarial: How to Explain IBNR](https://hugginsactuarial.com/how-to-explain-ibnr/), [Wikipedia: IBNR](https://en.wikipedia.org/wiki/Incurred_but_not_reported) (via search digest; neither page fetched).
- Broad IBNR = case reserve development + provision for reopened claims + IBNYR (including incurred-and-reported-but-not-recorded). Narrow usage = pure IBNR only. A claim-by-claim "known claims" reserve is described as case reserves + development on known claims + reopened-claims reserve — [Huggins Actuarial](https://hugginsactuarial.com/how-to-explain-ibnr/), [CAS clrstrans 1995](https://www.casact.org/sites/default/files/old/clrstrans_1995_2a.pdf) (digest; the CAS PDF was in the result list, not read).
- Reopened claims: provision for future payments on claims closed at the valuation date that may reopen due to unforeseen circumstances (AAA-style definition) — [Huggins Actuarial / search digest](https://hugginsactuarial.com/how-to-explain-ibnr/).
- Terms IBNER and IBNYR are used inconsistently across sources — same digest.

### Inferences
- Identity (standard): Reported (case-incurred) = Paid + Case reserve (outstanding on open claims). Ultimate = Paid + Case reserve + IBNR_total, where IBNR_total = Pure IBNR (IBNYR) + IBNER (+ reopened). Unpaid = Ultimate - Paid; IBNR (broad) = Ultimate - Reported. Triangle-based methods estimate the broad quantity; they cannot split pure IBNR from IBNER unless separate count and case-reserve histories exist (see Q6).
- "Ultimate" is an estimate, valuation-date dependent, normally on a defined basis (gross/net of recoveries, with/without ALAE). Whether the data include ALAE/salvage/subrogation must be fixed before development.

### Gaps
- Did not read primary definitions (CAS Statement of Principles, ASOPs, Friedland glossary); definitions are from secondary digests.
- Could not verify numerical split proportions between pure IBNR and IBNER for any line.

## 2. Loss triangles, age-to-age factors, averages, CDFs, tail factors

### Takeaway
Triangles are accident period x development age, cumulative; link ratios C(k+1)/C(k) are averaged (simple, volume-weighted, latest-n, medial/ex-hi-lo) and then judgmentally selected; CDF = product of selected links including a tail. A tail is needed when the triangle's last age is not fully developed (e.g. workers' comp), and is estimated by Bondy-type repeating, curve fits (exponential decay, inverse power), or external benchmarks.

### Cited Findings
- Friedland's chain-ladder procedure is described as seven steps starting from compiling a development triangle through age-to-age factors, averages, selected and tail factors, cumulative factors, and projecting ultimate; core assumption is that past development patterns continue, so changes in case reserving or claim mix can distort results unless adjusted for — [Wikipedia: Chain-ladder method](https://en.wikipedia.org/wiki/Chain-ladder_method) (digest). BattleActs and chainladder-python reproduce Friedland Ch. 5 (triangles) and Ch. 7 (development technique) — [BattleActs Ch. 7](https://battleacts.pro.wiki/wiki/Friedland07.Development), [chainladder-python Ch. 7](https://chainladder-python.readthedocs.io/main/friedland/chapter_7.html) (study aids; pages not fetched).
- Volume-weighted link = sum of later-age losses / sum of earlier-age losses over accident years having both; simple average = arithmetic mean of individual ratios; latest-3 variants of both exist; "medial" average drops one high and one low (e.g. latest 5x1); selection is judgmental after comparing averages, often leaning volume-weighted — digest of [MathWorks bootstrap example](https://www.mathworks.com/help/risk/bootstrap-using-chain-ladder-method.html), [Wikipedia](https://en.wikipedia.org/wiki/Chain-ladder_method), [Actex Ch. 7 preview](https://www.actexmadriver.com/Assets/ClientDocs/prod_preview/A105RC.pdf), [MetricGate](https://metricgate.com/docs/loss-development-triangle/).
- In the ChainLadder R package, simple-average and volume-weighted factors are regression estimates with weights 1/C^2 and 1/C respectively (so the factor family is a weighted-regression family) — [ChainLadder R package](https://mages.github.io/ChainLadder/reference/chainladder.html) (digest). Bardis-Majidi-Murphy discuss a family of chain-ladder factor models for selected link ratios — [CAS paper](https://www.casact.org/sites/default/files/2021-07/Family-Chain-Ladder-Bardis-Majidi-Murphy.pdf).
- Mack (1993) distribution-free assumptions: E[C(i,k+1) | past] = f_k C(i,k); Var proportional to C(i,k) with a age-specific variance parameter; accident years independent. A recursive follow-up extends the standard-error formula to include a tail factor — [Mack 1993 abstract](https://casact.org/abstract/distribution-free-calculation-standard-error-chain-ladder-reserve-estimates) (ASTIN Bulletin 23(2):213-225), [Mack/Cambridge summary](https://resolve.cambridge.org/core/services/aop-cambridge-core/content/view/E8D207F9A4DCE30300A76780FE510437/S0515036100011764a.pdf/distribution-free-calculation-of-the-standard-error-of-chain-ladder-reserve-estimates.pdf) (digest; I did not see the original variance assumption text).
- A tail is needed when the triangle ends before claims settle; workers' comp Schedule P uses ten years though some claims run up to ~50 years — [Boor, CAS Forum 2006 / CLRS 2004](https://www.casact.org/sites/default/files/presentation/clrs_2004_handouts_boor.doc) (digest).
- Four families of tail methods: Bondy-type (repeat last link), algebraic paid/incurred relationships, curve fitting, benchmarks (the last requires external data) — [Miller, CLRS 2004](https://www.casact.org/sites/default/files/presentation/clrs_2004_handouts_miller.doc) (digest). Bondy (repeat last link) is roughly exponential decay with 50% decay between tail factors; modified Bondy ~ decay factor 2/3 — same source.
- Sherman (1984) found an inverse power curve fit age-to-age factors better than other forms; no closed-form tail for the inverse power curve is known, so practitioners multiply fitted factors to a finite cutoff age — [Variance journal article](https://variancejournal.org/article/120867-tail-factor-convergence-in-sherman-s-inverse-power-curve-loss-development-factor-model) (digest).
- Exponential decay is usually fit log-linearly on (f-1), which requires every link > 1.000 — [CAS Actuarial Review, inverse power via GLM](https://ar.casact.org/wp-content/uploads/2017/05/Estimation_of_Inverse_Power_Parameters_via_GLM.pdf) (digest).
- chainladder-python `TailCurve` supports exponential and inverse power via OLS on the link ratios and extrapolates — [chainladder-python tails](https://chainladder-python.readthedocs.io/main/user_guide/tails.html) (digest). A generalized power-transform family: lambda=1 exponential, 0.5 Clark square-root decay, 0 Sherman inverse power — [Ledger Analytics docs](https://ledger-investing-ledger-analytics.readthedocs-hosted.com/en/latest/user-guide/classical-power.html) (digest).

### Inferences
Standard formulas (background knowledge; consistent with the digests but not checked against Friedland):
- Cumulative triangle C(i,k): accident period i, age k (months). Link f(i,k) = C(i,k+1)/C(i,k).
- Volume-weighted f_k = sum_{i<=n-k} C(i,k+1) / sum_{i<=n-k} C(i,k). Simple f_k = mean_i f(i,k). Latest-n: restrict to the n most recent diagonals. Ex-hi/lo: drop max and min of the link column, then average (needs >= 3-4 points).
- Selected link f*_k. CDF to ultimate at age k: CDF_k = f*_k x f*_{k+1} x ... x f*_{last} x Tail. Percent developed = 1/CDF_k. Ultimate_i = Latest_i x CDF_{age(i)}; IBNR = Ultimate - Reported(or Paid).
- Exponential decay tail: fit ln(f_k - 1) = a + b k; project (f_k - 1) for k beyond the triangle; tail = product over future k of f_k (truncate at a cutoff age). Inverse power (Sherman): f_k = 1 + a k^(-b), fit ln(f_k - 1) = ln a - b ln k. Mind the digest warning that links <=1 break the log fit.
- Practical data-science points: use only fully observed diagonals; check the latest diagonal is a true valuation-date diagonal; for sparse columns (few points) lean on a smoothed curve; paid and incurred triangles need separate factor sets.

### Gaps
- Did not retrieve CAS Tail Factor Working Party report (2013) or Friedland Ch. 7 text for formal tail-selection guidance (the benchmark path is thin: no benchmark sources such as NCCI/ISO/Schedule P tail data were found).
- Specific rules for when an ex-hi/lo or latest-n average should be preferred came only as generic "judgment" statements; no authoritative guidance seen.

## 3. Methods: chain ladder, Bornhuetter-Ferguson, Cape Cod, expected loss ratio, Berquist-Sherman; when each is appropriate

### Takeaway
Chain ladder (development) is responsive but leveraged on immature years; expected-claims (ELR) ignores actual experience; BF blends the two (reported + a priori x percent unreported); Cape Cod derives the a priori ELR from the data using "used-up premium"; Berquist-Sherman restates the triangle when case reserve adequacy or settlement speed changed. Immature, long-tailed, or unstable-development years favour BF/Cape Cod/ELR; mature years favour chain ladder.

### Cited Findings
- BF is a weighted average of the development (chain-ladder) and expected-claims methods; Friedland Ch. 9 is the main source — [BattleActs Ch. 9](https://battleacts.pro.wiki/wiki/Friedland09.BornFerg) (study aid; digest).
- Original BF paper: Bornhuetter and Ferguson (1972), "The Actuary and IBNR" — referenced via [CAS Proceedings 1997 abstract](https://www.casact.org/abstract/bornhuetter-ferguson-principle) / [Proceedings 97482](https://casact.org/sites/default/files/database/proceed_proceed97_97482.pdf) (this is the 1997 "BF Principle" paper, not the 1972 original; not read).
- BF is most useful for recent, immature years where actual paid/reported amounts are small and unstable; commonly used for workers' comp and general liability and for lines where one large loss can distort the triangle — [CAS BF Initial Expected Loss Ratio Working Party paper](https://www.casact.org/sites/default/files/2021-02/01b-initial-expected-loss-ratio-working-party-paper.pdf), [Oracle docs](https://docs.oracle.com/en/industries/financial-services/ofs-analytical-applications/insurance-loss-modeller/8.1.2.8.0/oilmu/bornhuetter-ferguson-bf-method.html) (digests).
- Chain ladder multiplies reported by the CDF, so one large/late claim in a young year swings the ultimate; BF adds an a priori expected amount for the unreported share that does not depend on actual reported — [LossReserves.com](https://lossreserves.com/learn/bornhuetter-ferguson/) (educational site).
- The expected-claims (ELR) method relies wholly on the a priori and ignores emerging experience — [The Actuarial Club](https://theactuarialclub.com/2019/05/17/bornhuetter-ferguson-technique-if-the-development-is-maturely-immature/).
- BF depends on two judgments: the a priori expected ultimate and the percent reported; if either is poor the answer is poor, and BF is not automatically better for immature years — [ActuaryPath](https://www.actuarypath.com/concepts/bornhuetter-ferguson/).
- Cape Cod / Stanard-Buhlmann: estimates the expected loss ratio from all accident years when no credible a priori exists; ELR = sum(reported) / sum(on-level premium x percent reported) where percent reported = 1/CDF ("used-up premium") — [chainladder-python Ch. 10 (Friedland)](https://chainladder-python.readthedocs.io/main/friedland/chapter_10.html), [MetricGate](https://metricgate.com/docs/cape-cod-method/), [CAS Forum 2002 Stanard-Buhlmann](https://www.casact.org/sites/default/files/database/forum_02fforum_02ff777.pdf) (digests; MetricGate describes losses as paid in one place while Stanard's original used reported; check).
- Cape Cod is a special case of the BF family where the a priori is estimated from the data; Gluck (1997, PCAS) generalized it with a decay factor between 0 and 1 giving years further from the current one less weight; an inappropriate decay can understate variability — [CAS 2012 handout](https://casact.org/sites/default/files/presentation/clrs_2012_handouts_session_5154_handout_548_0.pdf), [Sappington & Jiang](https://www.casact.org/sites/default/files/presentation/cs25-stochasticcapecod_anoldfriendinanewsuit-fileid-254934.pdf) (digests; Gluck's exact formula not seen).
- Berquist-Sherman (CAS 1977): incurred triangle adjusted for changes in case reserve adequacy, paid triangle for changes in claim settlement rates — [CLRS 2018 Taylor presentation](https://casact.org/sites/default/files/presentation/clrs_2018_presentations_st-6_taylor.pdf), [BattleActs Ch. 13 notes](https://battleactsmain.ca/v2/assets/doc/ReservingChapter13Berquist-ShermanMethod.pdf) (digests).
- Case-adequacy test: if reserving practice is consistent, growth in average open-claim case reserves should be roughly equal to growth in average closed-claim cost; unequal rates imply a change in adequacy. Adjustment: restate average case outstanding at the latest diagonal's level, detrend back by a selected severity trend, multiply by open counts, add paid to get adjusted incurred — [CLRS 2018 Taylor](https://casact.org/sites/default/files/presentation/clrs_2018_presentations_st-6_taylor.pdf), [BattleActs Ch. 13](https://battleactsmain.ca/v2/assets/doc/ReservingChapter13Berquist-ShermanMethod.pdf), [chainladder-python BerquistSherman](https://chainladder-python.readthedocs.io/stable/library/generated/chainladder.BerquistSherman.html) (digests). Worked example in BattleActs: outstanding severity trending 5% vs paid severity 3%; adjustment strips the extra 2%.
- Paid adjustment: disposal rate = cumulative closed counts / selected ultimate counts; assume the latest diagonal reflects current settlement rates; an exponential curve is fitted between cumulative closed claims and cumulative paid; paid at other cells is re-interpolated — [ChainLadder R `BS.paid.adj`](https://mages.github.io/ChainLadder/reference/BS.paid.adj.html), [Bermuda docs](https://ledger-investing-bermuda-ledger.readthedocs-hosted.com/en/latest/api/utils/index.html), [Actuaries India (Cicci)](https://actuariesindia.org/sites/default/files/2022-06/13th%20GCA_Reserve%20Techniques_Charles%20Cicci.pdf) (digests). B-S needs claim count (open/closed) data by age.
- Paid and incurred chain ladder ultimates often diverge materially for most accident years; paid-to-incurred ratios correlate with individual development factors (motivation for Munich chain ladder); large paid vs incurred gaps signal case-adequacy issues — [Quarg & Mack, Munich Chain Ladder abstract](https://www.casact.org/abstract/munich-chain-ladder-claims-reserving-technique-closes-gap-between-paid-and-incurred-based), [MetricGate](https://metricgate.com/docs/paid-incurred-chain/).
- Bodily injury: serious claims close later than minor claims, so average closed amounts rise with age; a CAS paper proposes a claims closure model instead of pure chain ladder — [CAS abstract: Evaluating BI Liabilities Using a Claims Closure Model](https://casact.org/abstract/evaluating-bodily-injury-liabilities-using-claims-closure-model).

### Inferences
Formulas (background knowledge, standard Friedland forms):
- Chain ladder: Ult_CL = Reported x CDF. IBNR_CL = Reported x (CDF - 1).
- Expected claims (ELR): Ult_EC = Premium x ELR (or exposure x expected pure premium/trended). IBNR_EC = Ult_EC - Reported.
- BF: Ult_BF = Reported + Ult_EC x (1 - 1/CDF); IBNR_BF = Ult_EC x (1 - 1/CDF). Equivalent: Ult_BF = (1/CDF) x Ult_CL + (1 - 1/CDF) x Ult_EC, i.e. credibility = percent reported. The user's formula "BF ultimate = reported + a priori x (1 - 1/CDF)" is correct with "a priori" meaning the a priori expected ultimate.
- Cape Cod: ELR_CC = sum_i Reported_i / sum_i (OnLevelPremium_i / CDF_i); then Ult_CC,i = Reported_i + OLPrem_i x ELR_CC x (1 - 1/CDF_i). Premiums should be on-level and ELRs trended to a common cost level when pooling years (Friedland's trending of ELRs).
- Benktander/Neuhaus (iterated BF) is an intermediate between BF and CL; not verified in sources here.
- Decision guide (synthesis): mature years with stable patterns -> chain ladder (paid or incurred); immature years in long-tail lines (BI, GL, WC) -> BF or Cape Cod, using incurred percent reported (paid CDFs at young ages are huge and unstable); if no credible a priori -> Cape Cod; if triangle shows case-reserve strengthening/weakening or speed-up/slow-down of closures -> Berquist-Sherman (needs counts) or switch to methods not reliant on case reserves; when development is erratic from large losses -> cap/split large losses, BF/ELR for them. Compare paid vs incurred indications as a diagnostic.
- Chain ladder breaks down if (a) volume (exposure) changes mix across accident years, (b) case reserving or settlement practices change, (c) the latest diagonal is unrepresentative, (d) CDF is very large (leverage).

### Gaps
- No primary text seen for BF (1972), Stanard (1985), Berquist-Sherman (1977). The BF citation found is the 1997 CAS "BF Principle" paper, not the original.
- The exact Gluck generalized Cape Cod formula was not retrieved.
- Benktander/Neuhaus not sourced.
- No quantitative guidance (e.g. at what CDF to prefer BF) found.

## 4. Claim-count development vs severity development; frequency-severity; count development with only reporting delays

### Takeaway
Reported-count triangles can be developed with chain ladder exactly like dollars to get ultimate claim counts; ultimate loss = ultimate counts x ultimate severity (severity developed separately, or via closure/disposal rates). When no case reserve information exists, only the count (pure IBNR) piece is estimable from the report-lag pattern; severity must come from external/mature data or trend assumptions.

### Cited Findings
- Chain ladder applied to a reported-claim-count triangle gives development factors that are readily converted into the IBNR (reporting) delay distribution — [Prediction of RBNS and IBNR claims using claim amounts and claim counts (Verrall & Wüthrich)](https://www.bayes.citystgeorges.ac.uk/__data/assets/pdf_file/0008/368837/prediction-of-rbns-and-ibnr-claims.pdf) (digest).
- That paper separates an IBNR delay (occurrence-to-report) from an RBNS delay (report-to-settlement) and models them separately, unlike most methods — same source.
- Friedland Ch. 11 frequency-severity: development-based, and disposal-rate/closure-based variants; the disposal rate approach uses proportions of claims closed at each age with selected ultimate counts; severity in development-based version is developed separately and need not equal developed ultimate dollars / developed ultimate counts — [BattleActs Ch. 11](https://battleacts.pro.wiki/wiki/Friedland11.FreqSev), [SOA exam solution 2021 spring](https://www.soa.org:443/4a4969/globalassets/assets/files/edu/2021/spring/solutions/edu-2021-spring-solutions-girr.pdf) (digests; the exact mechanics not seen). A CAS presentation says closure-rate methods help when closure rates change but require transaction-level payment and case reserve files — [CAS Winter 2021 / presentation](https://www.casact.org/sites/default/files/2021-03/completewinter2021.pdf) (digest).
- Werner & Modlin Ch. 6 excerpt: reported, paid, and closed counts can all be used; numerator and denominator of frequency/severity must share a basis; severity on a developed basis may use losses developed to ultimate; example reported severity at 18 months = $18.75M / 7,500 claims = $2,500 — [Basic Ratemaking 5th ed. excerpt (pp. 94-109)](https://consumerwatchdog.org/wp-content/uploads/2025/04/IRH-SFG-163-2016-05-01-Basic-Ratemaking-Fifth-Edition-Pages-94-109-Excerpted.pdf) (digest).
- Count-based IBNR interval estimates can be badly off when reporting delays are highly variable; a yearly chain ladder misses weekly patterns (nowcasting paper) — [Marked Cox models for IBNR claims count (Cambridge)](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/7F8EBD9A3AE3A72A44BFC4C352295EBA/S0515036125100561a.pdf/marked_cox_models_for_ibnr_claims_count_continuous_and_discretized_approaches_with_dirichletdriven_reporting_delays.pdf), [arXiv 1909.08336](https://arxiv.org/html/1909.08336v3) (digests).

### Inferences
- Frequency-severity: Ult$ = Ult_count x Ult_severity. Ult_count = Reported_count x CDF_count (or Reported_count / completeness). Pure IBNR count = Reported_count x (CDF_count - 1). Severity side: (a) develop average reported (or paid) severity triangles, (b) trend mature-year severity forward, (c) closure-based method.
- Count development is cleaner than dollar development for IBNR-only questions because counts do not depend on case-reserve practice (but do depend on claim-reporting definitions, e.g. CWP/closed-without-payment counting and reopening). Explicitly state whether counts include zero-payment claims.
- With counts only: you can estimate ultimate counts (pure IBNR) but not ultimate dollars without severity assumptions (external average severity, trended severity from older mature years, or a-priori).

### Gaps
- Did not obtain Friedland Ch. 11 formulas directly; the mechanics are my recollection.
- No authoritative source seen on handling CWP/reopen definitions in count triangles.

## 5. Reporting-lag / report-delay distribution approaches and completeness factors (1/CDF)

### Takeaway
Completeness factor at age k = P(claim reported by age k) = 1/CDF_k on a count triangle (the chain-ladder percent-reported); a report-delay distribution is the incremental version (differences of the completeness curve), and chain ladder on counts is the non-parametric estimate of it under a multiplicative (occurrence x delay) structure.

### Cited Findings
- Chain ladder on the reported-count triangle yields factors convertible to the delay distribution of IBNR claims — [Verrall & Wüthrich](https://www.bayes.citystgeorges.ac.uk/__data/assets/pdf_file/0008/368837/prediction-of-rbns-and-ibnr-claims.pdf) (digest).
- Chain ladder on reporting delays estimates an "IBNR factor," an element of a reserve, not a reserve itself — [A multistate approach to disability insurance reserving with information delays](https://arxiv.org/pdf/2312.14324) (digest).
- Reporting-delay models include EM-based and Poisson-type approaches with the occurrence process and delay distribution modelled separately — [arXiv 1909.08336 (EM algorithm for events with reporting delay)](https://arxiv.org/html/1909.08336v3), [CRAN NetSimR PureIBNR vignette](https://cran.r-project.org/web/packages/NetSimR/vignettes/PureIBNR.html) (digests).
- A monthly reporting-delay chain ladder implementation excludes the partially observed valuation-month diagonal when estimating factors — [GitHub: ibnr-reporting-delay-chain-ladder](https://github.com/Matariranwa/ibnr-reporting-delay-chain-ladder) (secondary; amounts not counts).
- I found no authoritative source defining "completeness factor" explicitly; closest analogue is the CDF / percent developed — same search digest.

### Inferences
- Let p_d = P(reported at lag d), F_k = sum_{d<=k} p_d (completeness). Then CDF_k = 1/F_k; F_k = prod_{j>=k} 1/f_j for count link ratios. Ult_count_i = Reported_count_i(k_i) / F_{k_i}. Pure IBNR_i = Reported x (1/F - 1).
- Equivalence: chain ladder on counts is the MLE of the multinomial/Poisson multiplicative model (occurrence level x delay prob) when the delay distribution is stationary (this equivalence is standard in the stochastic reserving literature, e.g. Mack/Verrall; not verified in sources above). So a parametric delay (lognormal, mixture, discrete hazard) can replace the non-parametric factors for smoother tails and finer time grids.
- Caveats: stationary delay assumption, exposure growth (occurrence volume varying by period is handled by chain-ladder scaling; but seasonality/weekday effects should be in the model for daily data), right truncation at valuation date, with the current period only partially exposed.

### Gaps
- No primary actuarial text (e.g. Friedland count development, Brosius, Taylor 2000) retrieved for completeness-factor terminology.
- Did not verify the count chain-ladder/Poisson equivalence in a source seen this session.

## 6. What can and cannot be estimated from a single valuation snapshot (report dates + current paid/incurred only)

### Takeaway
Nothing in the sources addresses this exactly; my reasoning: a single snapshot with claim-level report dates lets you reconstruct a report-lag (count) triangle (reported-to-date by accident period x lag), so count development/pure IBNR is valid; but without historical paid/incurred snapshots you cannot build dollar development triangles or test case-reserve adequacy, so dollar-development methods require external patterns or a priori assumptions.

### Cited Findings
- Case-adequacy checks (Berquist-Sherman) need average case outstanding and open counts by age across diagonals, and paid/disposal-rate adjustment needs closed counts by age — [BattleActs Ch. 13](https://battleactsmain.ca/v2/assets/doc/ReservingChapter13Berquist-ShermanMethod.pdf), [ChainLadder R BS.paid.adj](https://mages.github.io/ChainLadder/reference/BS.paid.adj.html) (digests).
- Closure-rate methods require transaction-level payment and case reserve data — [CAS Winter 2021](https://www.casact.org/sites/default/files/2021-03/completewinter2021.pdf) (digest).
- The chain-ladder assumption is that past patterns continue; changes in case reserving or claim mix distort results unless adjusted — [Wikipedia: Chain-ladder method](https://en.wikipedia.org/wiki/Chain-ladder_method) (digest).

### Inferences
Valid with one snapshot (report date + occurrence date + current paid/incurred per claim):
- Report-lag triangle: for each accident period i and lag d (report date minus accident date), count claims reported by lag d, restricted to cells fully observed at the valuation date (lag d <= valuation - end of period i). This gives count triangle, completeness F_d, ultimate counts, pure IBNR counts (Q4/Q5).
- Frequency estimate: Ult_count / exposure by accident period (needs exposure).
- Dollar quantities on closed/paid-to-date only: Current paid and current incurred are valid snapshots; ultimate$ via ultimate counts x severity assumption, where severity comes from (a) older, mature accident periods' current paid or incurred per claim, trended, (b) external benchmarks, or (c) Expected-loss-ratio method, which needs only premium and a priori.
- BF / Cape Cod on dollars require a percent-reported/paid pattern; this must come from an external or prior-study payment/reporting pattern (not derivable from a single snapshot). If such an external pattern is given, BF with a priori ELR, or Cape Cod, can be applied to current incurred/paid. Using count completeness as a proxy for dollar percent reported is an approximation only valid if later-reported claims have the same average severity as early-reported ones, which is often false (late-reported claims may be larger, e.g. BI).
Not estimable from a snapshot alone:
- Link ratios on dollars (no earlier-age values per claim, unless you reconstruct from paid transactions: a payment history by date would allow a paid triangle; only incurred/case history is missing).
- Case reserve adequacy or IBNER (needs case estimate history); reopened-claim development; Berquist-Sherman case adjustment.
- Separation of IBNER from pure IBNR in dollars (the count piece gives pure IBNR; IBNER requires case development).
- Severity development of the same claim over time; settlement-rate changes (need close dates or history; close dates in a snapshot give closure timing for closed claims but censor open ones).
- Check: a snapshot has right-censoring; valid cells are only the upper-left triangle; using all claims' report lags without censoring biases lags short.
Methods remaining valid with only a count/report-lag triangle: chain ladder on counts, parametric/nonparametric report-delay models, frequency (count/exposure) projection; Mack-style count variance (under assumptions); and, combined with an external severity, an ultimate-dollars estimate.

### Gaps
- No source found directly discussing single-snapshot claim-level data; this section is my own reasoning from the cited method requirements and is flagged as unverified inference.
- Whether payment-date transaction data are available (which would enable a paid triangle) is dataset-specific and unknown here.

## Overall source quality summary
- Primary documents seen only as search-result pointers: Mack 1993 (ASTIN 23(2)), Friedland 2010 (CAS), Werner & Modlin Basic Ratemaking (excerpt PDF listed), Berquist-Sherman 1977, Gluck 1997, Stanard 1985 (CAS Forum 2002 article on Stanard-Buhlmann listed), Sherman 1984, Boor 2006, Miller/Joe CLRS 2004.
- Study aids/vendor sites (lower reliability): BattleActs, MetricGate, fiveable, LossReserves.com, ActuaryPath, Apex, Tesora, Actuarial Club.
- Recommended follow-up if the sandbox allowed: read Friedland Ch. 7, 9, 10, 11, 13 directly, the CAS Tail Factor Working Party report, and Mack 1993.
