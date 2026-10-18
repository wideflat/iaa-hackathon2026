# Modelling approach (current state, 2026-10-18)

The task is a **pricing** forecast: no 2025 claims exist, so we predict next year's expected cost per policy row from 2021-2024 experience. Standard pricing structure, no reserving machinery beyond a small immaturity adjustment (planned).

## Pipeline (all in `src/iaa/`)
| Step | Module | What it does |
|---|---|---|
| Load | `data.py` | Parquet loaders, ids as text, dates parsed, `year` from period start |
| Target | `target.py` | Net cost of a claim = `incurred_to_date - recoveries`, repudiated = 0; claims matched to the policy row of the same `policy_id` whose period contains `loss_date`; row target `y` and `n_claims`; exposure recomputed from dates (`days/365`, cap 1) because the supplied column is corrupted on ~2-3% of training rows |
| Features | `features.py` | Joins customer, address and same-year weather; customer age; **claim history as known at the end of the previous year** (claims lost and reported by 31 Dec of year-1: count, cost, count last year); column policy below |
| Models | `models/baselines.py`, `models/freqsev.py` | See below |
| Validation | `validation.py`, `cv.py`, `metrics.py`, `rehearsal.py` | Fixed forward-in-time folds, exact D2 and clustered bootstrap, leak alarm |
| Guards | `drift.py`, `tests/` | Train-vs-test shift and outcome-correlation screens; 35+ unit tests |
| Output | `submission.py`, `scripts/submit.py` | Validated CSV writer; guarded Kaggle submit (`--confirm` required) |

## Feature policy (what the models may and may not use)
- **Never**: ids, dates, `renewal_decision` (training only; also a consequence of claims: rows later "declined by the insurer" are 0.5-0.65% of rows but 24-52% of cost), `exposure` (used as offset only), any target-like column (`y`, `y_true`, `y_all`, `n_claims`, `pred`).
- **Excluded as outcome-generated**: `customer_service_contacts` (train correlation with claim count 0.55; in the 2025 rows only baseline contacts remain). With it CV D2 is 0.55; without it ~0.10.
- **Excluded by default as suspected noise** (to be ablated, not assumed): surname, email_domain, vehicle_colour, branch_id, campaign_code, document_preference, marketing_consent, marketing_segment, previous_insurer.
- **Included**: policy and rating factors, `written_premium` and `discount_applied` (present in the test file; circular with the incumbent price, to be ablated), customer and address attributes, yearly weather, claim-history features.

## Model family 1: direct Tweedie GBM (baseline)
LightGBM, objective Tweedie (power 1.5), label = `y / exposure` weighted by exposure, prediction = rate x exposure. lr 0.03, 31 leaves, min 400 rows per leaf, feature/bagging fraction 0.8, L2 10, 300-400 rounds.
**Rebalancing** (key finding): a Tweedie-1.5 log-link GBM does not preserve totals; it predicts 20-30% below actual even on its own training data (it is deviance-optimal there). Scaling predictions per line so they sum to the training actuals lifted fold A D2 from 0.0975 to 0.1225. Not a trend effect.

## Model family 2: frequency x severity per peril (current best)
For each of the 15 line/peril combinations (motor 5, buildings 6, contents 4):
`E[cost | row] = E[count | row] x E[capped severity | row] x (1 + excess load)`, summed over perils, then rebalanced per line.
- **Frequency**: LightGBM Poisson with `log(exposure x base rate)` offset, lr 0.05, 15 leaves, min 300, 250 rounds (GLM variant: sklearn Poisson with exposure weights).
- **Severity**: LightGBM Gamma on claim-level capped net cost, lr 0.05, 15 leaves, min 60, 200 rounds. Cost floored at 1 CU (fully recovered claims have net cost 0).
- **Large losses**: cap at the 99.9th percentile of the peril's training claims (or max when < 150 claims); excess load = sum(excess)/sum(capped) from training. Handles the single open 11.9M bodily-injury claim (next largest 1.1M).
- Severity is modelled on claims as reported: within a peril, reporting lag is independent of claim size (Spearman within +-0.03), so late-reported claims are not systematically larger.

## Claim maturity (immature 2024)
2024 claims are incomplete at the 2024-12-31 snapshot (20% open; motor bodily injury only ~55% reported at 12 months). Evidence and tooling: `data/LDF_TABLES.md`, `methodology/TARGET_DEFINITION.md`. Decision (see `workflow/DECISIONS.md` D8): do not build the target from link ratios; only the frequency model gets a small completeness adjustment for 2024 exposure by peril (planned, not yet in the submitted model). Tested alternatives: maturity-as-feature (+0.012 in a one-year-earlier rehearsal, inside the noise), closed-claims-only labels (worse).

## Known limitations
- The first GLM benchmark is unstable (buildings D2 < 0; motor calibration 344 on fold B): heavy-tailed inputs and weak regularisation. Not a valid benchmark yet.
- The submitted model is trained on 2024 as valued (immature), so its level is probably low for motor bodily injury; the leaderboard level probes agree (optimum near x1.16).
- Case-estimate development (IBNER) cannot be measured from a single valuation snapshot.
- Valuation date and maturity of the hidden 2025 answer are not stated by the organisers (questions drafted in `methodology/TARGET_DEFINITION.md` section 7).
