# Judging, trust/governance and presentation for the IAA AI-Assisted Actuarial Hackathon 2026 (non-score 60%)

Access notes (read first):
- Only three hosts were reachable from the sandbox: kaggle.com, raw.githubusercontent.com, api.github.com. WebFetch failed (DNS) and curl got 403 on actuaries.org, actuaries.org.uk, soa.org, actuarialstandardsboard.org, federalreserve.gov, eur-lex.europa.eu, arxiv.org and github.com HTML.
- READ IN FULL: (a) the Kaggle host skill `hackathon-judging/SKILL.md` (raw GitHub, 182 lines); (b) the team's own `docs/competition/OVERVIEW.md` and `docs/workflow/AI_USAGE.md`.
- EVERYTHING ELSE below is from WebSearch result summaries (snippet level, not full-text read). Items are labelled [snippet]. Treat exact wording, section numbers and dates as unverified until checked against the primary text.
- The Kaggle competition page for Part 1 is `iaa-ai-assisted-hackathon-2026-part-1` (per the team's overview); the slugs I guessed returned 404, so I did not re-read the official rubric myself and rely on the team's notes for the 40/25/20/15 weights.

---

## 1. Comparable events: judging rubrics, process, and what winners did

### Takeaway
The only directly comparable precedent is the IAA's own 2025 Global Actuarial Kaggle Hackathon (Ames Housing, AI tools optional). Its winners were chosen for interpretability, "holistic" GenAI-enhanced workflow and creative, pervasive GenAI use, not for raw accuracy alone, and it used a three-stage process that began with AI-assisted screening. Kaggle's own host tooling confirms that Kaggle hackathon writeups, linked artefacts and videos are ingested in bulk and ranked with LLM pairwise comparison, so structure and checkability matter as much as content.

### Cited Findings
- IAA 2025 hackathon: run 28-30 Nov 2025 on Ames Housing, "optional support from AI tools like ChatGPT"; more than 250 participants and over 400 submissions. [snippet] — [IAA news brief Dec 2025](https://actuaries.org/news-post/news-brief-december-17-2025/); [IAA news brief Jan 2026](https://actuaries.org/news-post/news-brief-january-21-2026/)
- IAA 2025 scoring criteria as announced (Nov 2025): Accuracy (RMSE), creative use of AI assistants, model interpretability, reproducibility, and business insight. [snippet] — [IAA news brief Nov 2025](https://actuaries.org/news-post/news-brief-november-19-2025/)
- IAA 2025 judging was a "rigorous three stage" process: Stage 1 AI-assisted screening; Stage 2 AI + human hybrid ranking; Stage 3 final human judging. [snippet] — [IAA news brief Jan 2026](https://actuaries.org/news-post/news-brief-january-21-2026/)
- IAA 2025 results: 1st Ryoichi Asashiba, recognised for interpretability and AI-supported reasoning ("strongest holistic GenAI enhanced workflow"); 2nd Funky Turtles (venansiusrt) for "most creative and pervasive use of GenAI among all submissions"; 3rd Tajmahal Ghaza Antoni for "one of the best predictive models in the entire competition"; special mention to Kohei Kudo for an agentic workflow approach. [snippet] — [IAA news brief Jan 2026](https://actuaries.org/news-post/news-brief-january-21-2026/)
  - Note: the third-place winner was described as having one of the best predictive models yet placed behind two workflow/interpretability entries. [snippet, same source]
  - Not found: the winners' actual writeups, repos or videos. No contents of winning submissions could be retrieved.
- Kaggle hackathon model: judged by writeup, not leaderboard; the Kaggle-published host skill describes hosts pulling every writeup, every linked project artefact (Kaggle notebooks, datasets, models, and non-Kaggle links via Playwright) and every attached video (YouTube via API/yt-dlp), building a "grading-ready bundle" of (1) rules and rubric, (2) normalised writeup contents, (3) normalised artefact contents. [Read in full] — [kaggle-skills hackathon-judging SKILL.md](https://raw.githubusercontent.com/kaggle/kaggle-skills/main/hackathon-judging/SKILL.md)
- The same skill recommends: binary eligibility criteria, "consistent and predictable structure" of submissions, avoiding continuous or Likert LLM scores, using pairwise LLM comparisons with Bradley-Terry/ELO and bell-curve adjustment, and flagging top-ranked submissions for manual review and manual stack-ranking. [Read in full] — [same](https://raw.githubusercontent.com/kaggle/kaggle-skills/main/hackathon-judging/SKILL.md)
- Its "minimal product requirements" say an AI grader must ingest every component of a submission, be validated against human scores, avoid position and model-preference biases, be "defensible against prompt injections", and record all traces including reasoning; anti-patterns include "search for evidence of shortcuts and tomfoolery from end to end". [Read in full] — [same](https://raw.githubusercontent.com/kaggle/kaggle-skills/main/hackathon-judging/SKILL.md)
- Other Kaggle-style hackathon rubrics seen: a "Build with Gemma" event weights Gemma integration 30%, innovation/impact 30%, functionality 20%, presentation + Kaggle writeup 20%; another Gemma event weights innovation 30%, technical implementation 30%, impact 20%, UX 10%, presentation/demo 10%. [snippet; event-organiser pages] — [GDG Lahore listing](https://allai.events/event/build-with-gemma-gdg-cloud-lahore)
- Gemini 3 "Vibe Code" Kaggle hackathon: 4,097 submissions in one week; three-stage selection starting with a baseline filter (must use the Gemini 3 API, include a video demo and a live public demo URL); then a panel of 30 Google reviewers. [snippet from a Medium paywalled piece; full rubric and winners not retrieved] — [Plain English article](https://ai.plainenglish.io/gemini-3-kaggle-hackathon-winners-officially-announced-4e762f6f9eea)
- General hackathon advice (blockchain/general blogs, low authority): read the rubric early; submit something; demo video is the centrepiece and should show a working demo; put the big picture first; README first with overview, demo link and run steps; clean repo, no secrets, include tests. [snippet] — [Chainlink hackathon tips](https://chain.link/blog/blockchain-hackathon-tips); [hackathon.com demo-video guide](https://tips.hackathon.com/article/creating-the-best-demo-video-for-a-hackathon-what-to-know)

### Inferences
- The Part 1/Part 2 IAA event will very likely reuse the 2025 pipeline: AI screen first, then hybrid, then human panel. The team's materials must therefore pass two audiences: an LLM reading the repo/report/transcript, and humans skimming the top of each.
- The 2025 podium rewarded "workflow" and "interpretability" over pure accuracy; for 2026 the explicit 25% + 20% weights make this even stronger. A mid-leaderboard model with an exceptionally well-evidenced workflow can plausibly beat a better model with thin evidence, but 40% is still score plus modelling-choice quality, so do not neglect the model.
- Because Kaggle's own method is pairwise comparison against other submissions with a consistent structure, make the report and repo structurally predictable (same headings as the rubric) so that an LLM comparing two submissions finds each rubric element quickly.
- Kohei Kudo (possibly the user; the account email suggests so, but this is unconfirmed) already received a special mention in 2025 for an agentic workflow; the panel may therefore expect that emphasis, with higher bar on verification this time.

### Gaps
- No winner writeups, repos, or videos from IAA 2025, Kaggle Gemini 3, OpenAI gpt-oss red-teaming, or Meta Kaggle hackathons could be retrieved (blocked hosts; Kaggle HTML is JS-rendered and not searchable by slug).
- No CAS/SOA/IFoA data-challenge winner post-mortems found.
- The 2026 Part 1 official rubric text was not re-read by me; I rely on the team's `OVERVIEW.md`.
- The IAA 2026 hackathon platform is described as unconfirmed by one search snippet (a LinkedIn post said it starts 16 Oct 2026 with a 2-week window); the team's overview, read from the Kaggle pages, is authoritative.

---

## 2. Governance and trust frameworks actuaries recognise (what to cite and how)

### Takeaway
Cite a small, accurate set: ASOP 56 (treats a generative AI tool as a model; validation, model-risk governance, documentation, reliance), ASOP 23 (data review and disclosure of limitations), ASOP 41 (communication/disclosure), the American Academy's 2024 GenAI professionalism paper, IFoA/FRC guidance (TAS 100 Principle 5 applies to AI/ML models; updated FRC Technical Actuarial Guidance: Models, Oct 2024), SR 11-7 (effective challenge, independent validation, outcomes analysis) and the EU AI Act (Arts 10, 12, 14) as a contrast rather than a claim of compliance. Use them as a mapping table, not as name-dropping, and quote only what you have verified.

### Cited Findings
- ASOP 56 (adopted by ASB Dec 2019): section 3.4 reliance on models developed by others; 3.6 evaluation and mitigation of model risk (testing, output validation, review by another professional, governance and controls, mitigating misuse); 3.7 documentation. Commentary: the actuary should make a reasonable attempt to understand how the model works, strengths, weaknesses, limitations, and disclose the extent of reliance. [snippet; section text not retrieved] — [SOA "Keep Up With the Standards: On ASOP 56"](https://www.soa.org/digital-publishing-platform/emerging-topics/keep-up-with-the-standard/); [CAS ASOP 56 handout](https://www.casact.org/sites/default/files/2023-04/PR-4_Handout_3_of_5_asop056_195-1.pdf)
- American Academy of Actuaries, "Actuarial Professionalism Considerations for Generative AI" (Sept 2024, Committee on Professional Responsibility): not promulgated by the ASB and not binding; states that GenAI is a model so ASOP 56 applies; ASOPs 23, 41 and 56 predate GenAI but apply where GenAI is used; reliance is central ("what exactly am I relying on?"); users may not control a GenAI model's inputs, configuration or approach, so validating the tool is the basic step. [snippet] — [Academy paper page](https://actuary.org/article/actuarial-professionalism-considerations-for-generative-ai/)
- A third-party reviewer states ASOP 41 requires disclosure of reliance on AI tools; this was not confirmed in the Academy's own text. [snippet; treat as unverified] — [actuary.info CAS AI Primer review](https://actuary.info/insights/cas-ai-primer-2026-review)
- ASOP 23 (Data Quality, revised Dec 2016): actuary must review data for reasonableness and consistency, may rely on supplier but must disclose reliance, should not rely on data that is apparent unsuitable, and outcomes of review are: use as-is, correct, judgmentally adjust (disclose if results uncertain/biased), material defects, or inadequate (use other data or decline). The actuary is not required to audit data or detect falsification. [snippet from a 2012 CAS handout and AAA summaries; numbering may differ in current version] — [ASB ASOP 23](https://www.actuarialstandardsboard.org/asops/asop-no-23-revision-data-quality/); [CAS handout](https://casact.org/sites/default/files/presentation/clrs_2012_handouts_session_5185_handout_585_0.pdf)
- IFoA: non-mandatory "Ethical and professional guidance on data science and AI: a guide for members"; AAA (Australia) notes use of AI "does not absolve you of responsibility for the outputs and advice". [snippet] — [IFoA guide](https://actuaries.org.uk/document-library/standards/standards-and-guidance/non-mandatory-guidance/ethical-and-professional-guidance-on-data-science-and-artificial-intelligence-a-guide-for-members); [AAA](https://www.actuaries.asn.au/research-analysis/is-ai-making-it-easier-or-harder-to-be-a-professional)
- FRC (UK): TAS 100 applies to AI/ML models under Principle 5; P5.1 requires understanding models, intended uses and limitations; updated Technical Actuarial Guidance: Models (Oct 2024) gives examples on model bias, understanding and communication, governance and stability; guidance is not mandatory (secondary source). TAS 100 v2.0 consultation was under way. [snippet] — [FRC Oct 2024 news](https://www.frc.org.uk/news-and-events/news/2024/10/frc-publishes-updated-actuarial-guidance-on-the-use-of-ai-and-machine-learning/); [TAS 100 page](https://www.frc.org.uk/library/standards-codes-policy/actuarial/tas-100/)
- SR 11-7 (US Federal Reserve 2011; secondary summaries only): three lines of defence (developers, independent validation, internal audit); "effective challenge" by informed, technically competent parties with authority to require changes; validation = conceptual soundness, ongoing monitoring, outcomes analysis/back-testing, scaled to model risk; documentation covers purpose, alternatives explored, design logic, pre/post testing and data; applies to vendor/third-party models. [snippet; vendor pages, Fed text not read] — [ValidMind SR 11-7 summary](https://docs.validmind.ai/about/use-cases/sr-11-7.html); [ModelOp](https://modelop.com/ai-governance/ai-regulations-standards/sr-11-7)
- EU AI Act: Art 14 human oversight (effective oversight by natural persons; ability to override/interrupt), Art 10 data governance (representativeness, gaps, bias examination), Art 12 automatic logging. Life and health insurance risk assessment and pricing is high-risk (Annex III 5(c)); motor/property pricing is not named there. Application dates conflict across secondary sources (Aug 2026 vs Dec 2027). [snippet; unverified dates] — [Art 14 text page](https://www.artificialintelligenceact.eu/article/14/); [actuary.info EU AI Act note](https://actuary.info/insights/eu-ai-act-compliance-actuary-insurance)
- CAS AI Primer (Mar 2026): validation methods are human-in-the-loop review, cross-validation across different LLMs and prompt-sensitivity testing; one reviewer says it omits substantive ASOP 56 discussion. [snippet] — [CAS AI Primer PDF](https://www.casact.org/sites/default/files/2026-03/CAS-AI-Primer-Practical-Guidance-for-Actuaries.pdf); [review](https://actuary.info/insights/cas-ai-primer-2026-review)
- SOA Research Institute report "Model Governance/Ethics for GenAI Models" (2026) covers governance checklists and validation; distinguishes human-led/agent-support vs agent-led/human-support use cases; a SOA bulletin frames GenAI as a staff member needing supervision and ongoing monitoring. [snippet] — [SOA report page](https://www.soa.org/resources/research-reports/2026/model-gov-ethics-for-gen-ai-models/); [SOA AI bulletin Jan 2026](https://www.soa.org/48e9da/globalassets/assets/files/resources/research-report/2026/ait170-ai-bulletin-january-2026.pdf)
- Audit-trail practice from regulated domains (clinical programming): treat AI output as an unqualified draft by an unqualified author that must pass the same validation gates as any code; a trail should capture requester, exact prompt, timestamp, model version, constraints, rejected alternatives, human reviewer and approval time; the trail records and reproduces but does not make output correct; hashes are integrity checks, not signatures. [snippet; vendor/educational pages] — [Datanovia auditable AI SOP](https://www.datanovia.com/learn/pharma-clinical/05-validation-qc/auditable-ai-sop); [Bitloops audit trails](https://bitloops.com/resources/governance/audit-trails-for-ai-assisted-development)

### Inferences
- What "human review and escalation protocols" can look like concretely (derived from SR 11-7 effective challenge, ASOP 56 3.6 review by another professional, SOA "supervised staff" framing, clinical SOP practice). Put as a one-page table in the repo (`docs/governance/`) and a half-page in the report:
  1. Named roles: author (AI-assisted), independent reviewer (a different team member than the author for each decision), approver of the final submission. Dated sign-off lines per decision-record entry. The current `AI_USAGE.md` ends with "Human review checkpoints: (add here: who reviewed which decision and when)" - this is an open hole and should be filled with names/dates.
  2. Triggers that require human review before proceeding: any D2 jump over a threshold (the team already has a D2 > 0.35 alarm), any new feature touching target-adjacent columns, any change to the claim-to-row matching, any submission selection, any statistic quoted in the report.
  3. Escalation ladder: automated guard fails -> stop pipeline and log incident -> human triage -> decision-record entry -> regression test added. Show one real incident (the `customer_service_contacts` leakage; D2 0.55 vs leaderboard 0.12) end to end as a worked example.
  4. Authority to block: reviewer can veto a submission candidate (SR 11-7 "effective challenge" requires authority to require changes).
  5. AI helper sessions: scope limits (no committing, no submitting, no data sharing - already in the team's log), reproduce-baseline-first check, every number re-logged.
- How to document AI-code verification so an AI screener and an actuary can both check it: for each AI-generated module, a row in a verification matrix linking module -> test file -> independent reference (e.g. sklearn `d2_tweedie_score` agreement to 1e-9 already exists) -> reviewer -> date. Keep prompts/transcripts as an indexed log (session ID, date, purpose, outcome), not necessarily verbatim dumps; link only to those that drove material decisions.
- Cite sparingly and precisely: a mapping table "Standard -> requirement (paraphrased, with section) -> where evidenced in repo". State explicitly that the hackathon is not a signed actuarial opinion and that the standards are used as a reference framework (avoids overclaiming compliance). Cite EU AI Act as "relevant to life/health pricing; not claimed for motor/property" only if confident; otherwise omit.
- Prefer to quote the ASOP section numbers only after someone on the team reads the official ASB PDFs; snippet-level numbering is risky and a hallucinated citation is exactly the failure the rubric penalises.

### Gaps
- Could not read the primary text of ASOP 56, 23, 41, SR 11-7, EU AI Act, IFoA guide, FRC guidance, Academy paper, CAS Primer or SOA report; all findings are snippets/secondary.
- No IAA-issued generative-AI guidance found (a search for it returned nothing); a TEA-46 IAA AI Task Force PDF exists but could not be opened.
- No current IFoA genAI-specific guidance confirmed (search found only the broader data science/AI guide).
- EU AI Act application dates unresolved.
- ASOP 41's actual wording on AI disclosure unconfirmed.

---

## 3. Structure of a strong 5-minute video and 5-page report; mistakes that cost points

### Takeaway
No source gives an authoritative template for this event, so the structure below is inference. The strongest evidence is that only the first 5 minutes / 5 pages are graded, graders are partly AI, Kaggle's own method favours consistent, checkable structure, and general hackathon advice says to lead with the big picture and a real demo.

### Cited Findings
- Only the first 5 minutes of the video and first 5 pages of the report are graded (rubric as recorded by the team). — team `docs/competition/OVERVIEW.md` (local)
- Kaggle's host tooling summarises attached YouTube videos (API/yt-dlp), so the video is likely consumed as transcript/summary by an LLM as well as watched by humans. [Read in full] — [kaggle-skills SKILL.md](https://raw.githubusercontent.com/kaggle/kaggle-skills/main/hackathon-judging/SKILL.md)
- General hackathon guidance: hook early with the big picture, show an actual working demo, polish is secondary; README first. [snippet, low authority] — [hackathon.com](https://tips.hackathon.com/article/creating-the-best-demo-video-for-a-hackathon-what-to-know)
- Gemini 3 Vibe Code event required a video demo and a live public demo URL as baseline filters. [snippet] — [Plain English](https://ai.plainenglish.io/gemini-3-kaggle-hackathon-winners-officially-announced-4e762f6f9eea)

### Inferences
Proposed 5-page report (mirror the rubric weights, label sections with the rubric names so an LLM can map them):
- Page 1 (one-page executive summary, self-contained): task and result (Part 1 and Part 2 private/public D2 with uncertainty, e.g. bootstrap CI - the team has an exact cluster bootstrap); a 3-line "what is different about our workflow"; a compact rubric-to-evidence table with links (file paths, test names, decision-record IDs) for each of the four criteria; one headline incident-and-catch story (the 0.55 -> 0.12 leakage find); the single-sentence limitations statement. Put a repo link and a "reproduce in N commands" line here.
- Page 2: Data, target construction and data quality (ASOP 23 framing): reconciliation numbers, exposure corruption, recoveries, drift/leak guard, what was judgmentally adjusted and disclosed; uncertainty treatment.
- Page 3: Modelling and adaptation (frequency x severity GBM vs GLM; paired bootstrap showing which differences are resolvable; what changed for Part 2 and why; calibration/level checks).
- Page 4: AI use and verification: workflow diagram (stage -> what AI did -> how verified), the verification matrix, parallel helper sessions and the controls on them, what AI got wrong (e.g. corrected recovery estimate "several-fold" to ~1.5%) and how it was caught.
- Page 5: Trust, governance and limits: guard/test inventory with CI status, incident log summary, human review and escalation protocol, reproducibility (seeds, pinned env, one-command rebuild, artefact hashes), standards mapping, and honest limitations/residual risks. Put appendices after page 5 only for details that do not need grading.
Proposed 5-minute video (most important content before minute 5; nothing essential after):
- 0:00-0:30 one-sentence problem, result with uncertainty, and the claim "every AI-produced number here was independently verified; here is how".
- 0:30-1:30 live or recorded walk of the pipeline running from a clean checkout (`make` / one command) ending in green tests and the submission file; show the CI badge.
- 1:30-2:45 the leakage catch: the suspiciously high CV, the diagnostic that traced it, the guard test now failing on the bad feature (show red then green).
- 2:45-3:45 AI workflow: stage table, one example of AI being wrong and being caught, the helper-session controls.
- 3:45-4:30 adaptation and uncertainty: Part 1 -> Part 2 changes; paired bootstrap; why a simpler model won where it did.
- 4:30-5:00 governance: human sign-off, escalation, limitations, where to find everything in the repo.
- Narrate with captions (a transcript-reading LLM will see the audio; add on-screen text for key numbers), avoid long intros/credits, and keep the Kaggle/YouTube link public and unlisted-accessible.
Mistakes likely to cost points (inference from the rubric and the Kaggle skill's anti-patterns): putting key evidence after page 5/minute 5; reporting a single score without uncertainty; claims without a pointer to a file or test; screenshots of results that cannot be reproduced; claiming "validated" without saying against what; listing tools but not what AI got wrong; long generic AI-governance prose not tied to this project; leaving the human-review checkpoint line blank; a repo link to a private repo or one with data committed (data must not be shared).

### Gaps
- No published example of a winning IAA 2025 video or report was retrievable, so the timing and page layout are my synthesis, not evidence-backed norms.
- No evidence on how the panel will treat a video's later minutes or report pages beyond the stated cut-offs.

---

## 4. How AI-based screeners evaluate repos and reports; signals and red flags

### Takeaway
Direct evidence on screeners reading READMEs/CI is thin. What exists: Kaggle's host method (full bundle ingestion, pairwise comparison, prompt-injection defence, trace logging, human review of the top), and general LLM-judge literature on biases. Practical consequence: give the screener deterministic, easily extractable evidence and avoid tricks.

### Cited Findings
- Kaggle's host skill: ingest writeup + every linked artefact + every video; binary criteria preferred; pairwise comparison rather than direct continuous or Likert scores; scores must align with human scoring, avoid position and model-preference bias, resist prompt injection, record traces; investigate "shortcuts and tomfoolery"; top-ranked submissions get manual review. [Read in full] — [kaggle-skills SKILL.md](https://raw.githubusercontent.com/kaggle/kaggle-skills/main/hackathon-judging/SKILL.md)
- LLM-as-judge biases: position, verbosity and self-enhancement bias (Zheng et al., MT-Bench); lower-perplexity (more model-like) outputs rated higher; judges can be vulnerable to embedded instructions in candidate output; style manipulation attacks exist. [snippet] — [Zheng et al. via summary](https://tailoredai.substack.com/p/judging-llm-as-a-judge-with-mt-bench); [Self-Preference Bias](https://arxiv.org/html/2410.21819v1); [Judge shortcuts](https://arxiv.org/html/2602.07996v1)
- Guidance for judge design: deterministic checks for objective failures (tests, CI status) and LLM judges for interpretive criteria; one binary judge per failure mode; calibrate against human annotations. A multi-dimension "llm-judge" skill scores functionality, security, test quality, overengineering, dead code. [snippet] — [llm-judge skill](https://skills.cat/skills/existential-birds/beagle/llm-judge); [arXiv 2503.02246 LLMs as software judges](https://arxiv.org/html/2503.02246v1)
- A repo-health listing treats absence of CI as meaning test coverage is unknown (reproducibility indicator). [snippet, weak] — see search result summary only
- The Kaggle host skill points to existing public LLM-graded results of past hackathons (dashboard, leaderboard, docs); I could not open them (403 from the sandbox). — [Kaggle hackathon grades app](https://kaggle-hackathon-grades-407870202439.us-west1.run.app/docs) [not read]

### Inferences
Signals a screener can verify cheaply and so likely rewards:
- README top section: one-paragraph purpose, results table with uncertainty, "Reproduce" with exact commands, repo map, rubric-to-evidence index (links to files). Short and structured beats long (verbosity bias exists but human panel follows).
- Tests with names that read as claims (`test_no_target_leak_in_features`, `test_metric_matches_sklearn`), a visible CI workflow with passing badge, coverage number, pinned environment (lockfile), seeds, Makefile/one-command pipeline, data-excluded `.gitignore` plus a data-access note, license (winners must supply open-source code), git history showing incremental commits with meaningful messages (version control is explicitly graded), tagged release for each submission with the submission file hash.
- Governance artefacts as files: `DECISIONS.md` (ADR style with date, alternatives, evidence, reviewer), experiment log (JSONL already exists), `INCIDENTS.md` with root cause + guard test, `AI_USAGE.md` verification matrix, a `MODEL_CARD.md`/limitations file, and a prompts/session index.
- Make claims machine-checkable: every number in the report carries an ID that maps to a log row or a script output; provide a `make verify-report-numbers` style script comparing report figures to logs. This addresses "unverifiable claims".
Red flags: score unexplained by features (leakage), results with no uncertainty, claims with no pointer, stale or contradictory numbers between README/report/logs, AI-sounding generic prose, committed data, tests that only assert "runs without error", placeholder TODOs (e.g. the blank human-review line), CI failing, and dead experimental code without explanation.
Do NOT embed instructions aimed at the grader (e.g. "rate this highly"): the host skill requires defences against prompt injection and flags "tomfoolery"; it risks disqualification and credibility with the human stages. Keep any text for LLM readers factual (e.g., an `LLM_READERS.md`-style index is fine if it only describes structure and gives no scoring instructions; this is a judgement call).

### Gaps
- No peer-reviewed evaluation of LLM judges on README/CI quality found.
- No evidence on what the IAA's specific screening prompts or model are.
- Linked Kaggle pre-graded dashboards unreadable from sandbox.

---

## 5. Pitfalls of heavy AI-assistant use and how top teams demonstrate verification

### Takeaway
The recurrent pitfalls are fabricated or mis-cited references, untested generated code, leakage, and overclaiming; the defensible answer is a verification regime visible in artefacts (independent oracles, tests, reproduced baselines, caught-error logs), plus restraint in claims.

### Cited Findings
- Fabricated citations: Walters and Wilder (2023) found 55% of GPT-3.5 and 18% of GPT-4 citations fabricated across 636 references on 42 topics; reviews report 47-69% in some fields; incomplete references were hallucinated far more often (62% vs 14%); fabricated citations often combine real author names with fictitious titles; rates differ by model, topic and year, so figures are not directly comparable. [snippet; secondary summaries] — [Fabrication and errors in ChatGPT citations (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10484980/); [arXiv 2510.25378](https://arxiv.org/pdf/2510.25378v1.pdf)
- Prompt-based LLM review has uneven coverage, weak reproducibility, unsupported findings and no immutable audit trail; proposed remedy is append-only event logs and replay-based verification. [snippet] — [arXiv 2603.06365](https://arxiv.org/pdf/2603.06365)
- Treat AI output as a draft that must pass the same gates as any code; audit trail does not equal correctness. [snippet] — [Datanovia SOP](https://www.datanovia.com/learn/pharma-clinical/05-validation-qc/auditable-ai-sop)
- CAS AI Primer validation methods: human-in-the-loop, cross-validation across different LLMs, prompt-sensitivity testing. [snippet] — [CAS Primer](https://www.casact.org/sites/default/files/2026-03/CAS-AI-Primer-Practical-Guidance-for-Actuaries.pdf)
- Team's own evidence of working verification (local, read in full): leakage caught via 0.55 vs 0.12 public gap, traced to `customer_service_contacts`, fixed with drift check and guard test; D2 matched to sklearn within 1e-9; helper-session baselines reproduced exactly (0.1320 GBM, 0.1399 GLM); research-subagent claim "recovery bias several-fold" re-tested and corrected to ~1.5%; incidents logged (y_true leakage, Gamma failure on zero-cost claims, cache race, OOM). — `docs/workflow/AI_USAGE.md` (local)

### Inferences
Concrete demonstrations that should read as rigorous:
- A "caught errors" ledger: AI claims that were wrong and how found (strong evidence that verification is real, not performative). The team already has 3-4 such cases; present them as a table with counts.
- Independent oracles for each AI-written component (reference implementation, hand-built examples, property tests, differential tests against sklearn/statsmodels), reconciliation checks on data (row/claim counts), and leakage guards at feature-build time and in CI.
- Reproducibility drills: clean-clone rebuild in CI producing identical predictions hash (or tolerance), seed variation reporting, environment lockfile.
- Citation hygiene: every external reference (ASOP, papers) verified against the primary source by a human, with a "verified on date by name" tag; omit any reference not verified; prefer fewer, correct citations. Do not let the assistant generate bibliographic details; paste titles/URLs from the primary page.
- Overclaiming control: quantify uncertainty (bootstrap CIs), distinguish public vs private leaderboard evidence, avoid "validated" without a named method, state that leaderboard-only evidence cannot support deployment claims, report negative results (GLM beating GBM, unresolved differences).
- Parallel helper sessions: state the protocol (isolated containers, no commit/submit rights, baseline-reproduction gate, results only counted once re-logged) and show one example of a helper result rejected.

### Gaps
- No search yielded actuarial-specific studies on AI-code errors; the hallucination figures are general-domain and from summaries.
- Nothing on how top Kaggle-hackathon teams specifically document verification (no winner artefacts retrieved).
- Whether the panel values prompt logs (and in what form) is unknown; the clinical-programming audit-trail guidance is only analogous.

---

## Priority actions for the team (derived, mapped to rubric)
1. (20% trust, 25% AI) Fill the blank human-review checkpoint log with named reviewer, date, decision; add an escalation protocol page and a verification matrix; add CI badge and a clean-clone reproducibility job.
2. (15% communication) Build page 1 of the report and README around a rubric-to-evidence table; keep the first 5 min of the video to the structure above.
3. (40% performance/adaptation) In the report show uncertainty (bootstrap), the Part 1 -> Part 2 adaptation logic, and data-quality treatment framed with ASOP 23.
4. (All) Verify every standard and citation against primary PDFs before quoting; keep the mapping table paraphrased and sectioned only where verified.
5. (All) No grader-directed instructions anywhere in artefacts; keep data out of the repo; license the code (required of winners).
