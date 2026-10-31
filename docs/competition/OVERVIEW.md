# Competition overview (as read from the Kaggle pages and API on 2026-10-17)

**Host**: International Actuarial Association. **Page**: kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1. Two-part hackathon; prizes are decided after Part 2 by a judging panel, not by the Part 1 leaderboard alone.

## Task
Predict, for every 2025 policy row in `test_motor`, `test_buildings`, `test_contents`, the **expected ultimate claim cost**: the final cost of all claims with a loss date in the row's period, net of the policy excess and net of recoveries, excluding declined claims. It is the risk premium, not the price charged. Predictions must be >= 0. One CSV with `row_id,prediction` covering all 626,639 test rows.

## Metric
Tweedie D2 score with power 1.5 (sklearn `d2_tweedie_score`), pooled over all test rows: the fraction of Tweedie deviance explained relative to a constant-rate baseline. Higher is better.

## Timeline and limits
| Item | Detail |
|---|---|
| Part 1 | Opens 2026-10-16, closes **2026-10-23 16:00 UTC** (also the team-merger deadline) |
| Part 2 | Separate Kaggle page, opens when Part 1 closes: a new "emerging market" dataset, same team. Task: adapt the approach, explain what changed and remaining uncertainty |
| Team | Max 3 members; same team in both parts |
| Submissions | **10 per day** (confirmed by the host on 2026-10-18; the rules text was updated, it previously said 5); one Final Submission is selected for judging (default: best public score) |
| Public / private split | Public leaderboard uses about 33% of test rows; final standings use the other 67% |
| Prizes | CAD 500 / 100 / 50, only for members or students of an IAA member association |

## Judging (after Part 2)
| Weight | Criterion |
|---|---|
| 40% | Pricing performance and adaptation: private leaderboard scores of Part 1 and Part 2 (equal weights) plus the panel's view of modelling choices, adaptation, treatment of data quality and uncertainty |
| 25% | Effective, verified use of AI across the workflow |
| 20% | Trust, validation and guardrails: tests, diagnostics, governance, reproducibility, human review |
| 15% | Communication: video demo (first 5 minutes graded), written report (first 5 pages graded), well-structured repository |

## Data (provided as Parquet)
| File | Rows | One row per | Content |
|---|---|---|---|
| `policies_motor` / `policies_buildings` / `policies_contents` | 980k / 708k / 722k | policy period | 2021-2024 rating factors, premium; `renewal_decision` (training only). **No target.** |
| `claims` | 163k | claim | peril, loss and report dates, status (closed/open/repudiated), paid, incurred, recoveries; valued at 2024-12-31 |
| `customers`, `addresses`, `weather` | 545k, 576k, 2.88M | customer / address / address-year | demographics, property and location, yearly weather (apparently 2021-2025) |
| `test_motor` / `test_buildings` / `test_contents` | 255k / 184k / 187k | policy period | 2025 rows to predict |
| `sample_submission` | 626,639 | test row | format |

Links: `customer_id` -> customers; `address_id` -> addresses and weather; `policy_id` -> claims (match a claim to the row whose period contains its `loss_date`); `row_id` -> submission.

## Rules that matter for how we work
- Competition data must not be shared or committed (data is gitignored); processing with hosted AI tools is explicitly permitted.
- No private code or data sharing outside the team; one Kaggle account per person.
- External data and tools are allowed if reasonably accessible; AutoML is allowed.
- Winners must provide reproducible code under an open-source licence and documentation.
