# Video script (5 minutes graded; target 4:30)

Screen: the repository README and `docs/part2/REPORT.md`; one slide per block. Speaker: [team].

| Time | Block | Say / show |
|---|---|---|
| 0:00-0:30 | Result | "Week 1: private rank 3 of 54, D2 0.13864. Week 2: the same engine, adapted to a market with two years of data; public score 0.16462 for the final ensemble." Show the table of section 1. |
| 0:30-1:30 | What changed | Three numbers on one slide: motor frequency +34% and cost -42%; 76% of the test book is new business; no history before 2023. Show the peril table (frequency multipliers by peril, severity scale by line). "The new market is the old market times stable peril multipliers: we measured that, we did not assume it." |
| 1:30-2:30 | Method | One picture: developed model on shared variables -> add last-year claim history -> blend three tree models, a direct Tweedie GBM and six nets (single and three-seed fits) -> level from the emerging book (counts and winsorised severity per peril, credibility-weighted, no closing to the training total) -> gender factor. "The emerging data decides how much, the developed data decides who." |
| 2:30-3:30 | How we trust it | Both-direction validation (forward and reverse) and why: two years, buildings swings 1.5x. Show the step table (0.1110 -> 0.1408 forward, 0.1272 reverse). Show the rejected list in one sentence: more than thirty ideas rejected because they helped in one direction only, plus a simulation of the level method with a known truth and a ceiling analysis: a perfect level would add at most 0.005. Show the Part 1 shake-up figure: public and private agree overall (0.997) but not inside the top group (0.20): we select by local validation. |
| 3:30-4:15 | AI in the workflow | Claude Code did EDA, code with tests (125), experiments; 11 helper sessions ran refits in parallel; a blind second agent compared the markets from raw tables and found two things we missed. Show one catch: the three unsupported statements corrected before commit. "Every number traces to a command." |
| 4:15-4:30 | Guardrails and uncertainty | Model card, monitoring triggers, leak alarm, buildings level as the main stated risk (+-20% = +-0.01), gender rating flagged for regulatory review. One sentence on the office premium: written premiums roughly equal expected claims, so loadings need a 2.5x premium. |
