# NHL V2 Phase1B — opportunity components and pregame intelligence

**Verdict: component candidates rejected; full forward engine blocked. No promotion.**

This phase implements data/source gates and tests real exposed hockey components. It does not establish a professional-analyst-beating engine. 2018–2022 estimate priors, 2023 tests fixed components, 2024–2025 are fixed burned diagnostics. B2 was itself fitted on 2018–2025; its retrospective comparisons are exposed/in-sample, not untouched confirmation. No Phase1B forward forecasts were issued.

## Integrity and preserved history

PR65 was at 1a77fc01. Merge156eb9a preserves the original926 rows byte-for-byte and appends397 to1323, including all original blobs/manifests. B2v1.1 code/model/protocol hashes and all provenance verify. PR68 integrity commits d4a7fb1 and3ee6054 remain isolated in that PR: dispatched-ref checkout/push, exact final boxscore reconciliation, actualteam quarantine, one forecast-ledger owner, and refusal to grade a branch missing upstream ledger commits. Original frozen grader/forecast probabilities are never rewritten. Manual grading must be dispatched on repaired research branch; unmerged PR65 still has its legacy workflow.

Oct7 replay remains344 grades +8 ambiguous-row quarantines. MeaningfulT90 n71 MAE .994117/CRPS .671983; T30 essentially identical. These three games are descriptive only. Independent human comparators0, certified lineups0.

Audit found all1323 stored P1..P5 survival fields shifted (P1always1). Correct NB2 evaluation is separate and unchanged; stored probabilities are not scientifically valid and must not be presented as P>=1..5. No repair to the frozen engine was attempted.

## Actual source feasibility

Historical total/EV/PP/PK TOI and SOG are complete and reconcile. Stable player skill can follow team changes; deployment uses current-team appearances only. No exact historical completion timestamp or original publication vintage exists: replay uses an explicit conservative24h completion proxy, never claims timestamp-certified forward history. Forward history requires actually observed final completion.

A deeper branch audit found285069 adjudicated official-SOG +missed +blocked total-attempt labels for2017–2023. They were copied byte-identically with SHA pins from cea38be3; no old rejected model coefficients/features/results were used or rescored. 112709Phase1A rows lack these labels; missing is never zero-imputed. Strength-specific attempts and2024onward/current attempt labels are absent. New unauthorized acquisition is disabled.

Existing T24H10/T9010/T303 decisions have complete pre-cutoff endpoint captures, but zero nonempty roster/scratch observations and zero confirmed dressed captures. No horizon meets40games/3dates/.995recall. A roster response stays observed, even if eventual candidate recall qualifies; it is not a confirmed lineup.

NHL official API automation/storage rights remain unproven; public NHL terms prohibit unauthorized automated compilation. SportsDataIO documents EV/PP/PK lines/injuries, Sportradar documents positional/PP depth and injury feeds. Neither was accessed, licensed or timing-certified; historical original revisions not demonstrated and pricing unknown/contactsales. MoneyPuck returned a data-license request in this environment; no data acquired. Official projected-lineup articles are potentially useful but mutable historical snapshots and automated access remain unqualified.

## Component outcomes (2023 exposed primary expected participants)

|Component|Reference error|Candidate error|Decision|
|---|---:|---:|---|
|TotalTOI prior10 vs recent3 (seconds)|113.651|117.012|Rejected; worse +EV guard fail|
|TotalTOI prior10 vs70/30 blend (seconds)|113.651|112.810|Rejected;0.74%<2% gate|
|Totalattempts mean vs TOI×shrunk rate|1.718727|1.736508|Rejected;1.03% worse|
|Totalattempts mean vs opponent-adjusted rate|1.718727|1.738028|Rejected;1.12% worse|
|Conversion position vs player shrinkage MAE|0.259728|0.259075|Rejected;0.25%<2% gate|
|Participation unsmoothed vs beta Brier|0.076538|0.078746|Rejected as improvement|

No failed families were composed or rescued. The implemented full strength-specific analytic PMF is unvalidated input-gated infrastructure; actual attempt counts are required, never inferred from SOG. Expected team budget normalization is tested but joint draw allocation is explicitly not certified. The research lock is immutable and NOT_READY, so the writer refuses forecasts.

## Paired SOG comparators (fixed scripted baselines, not the failed full engine)

|Period|Primaryn|B2MAE/CRPS|RecentmeanMAE/CRPS|TransparentanalystMAE/CRPS|
|---|---:|---:|---:|---:|
|2023|43278|1.055068/0.709917|1.141792/0.764196|1.094398/0.740183|
|2024|43452|1.024621/0.685797|1.109705/0.741119|1.062944/0.715204|
|2025|43328|1.019316/0.681029|1.103235/0.735872|1.059547/0.710632|

Full/actuallyplayed metrics, signedbias, interval coverage, Brier/calibration reliability forP1..P5, conditionalEV/PP/PKTOI, catastrophic misses and exposure/population counts are in phase1b_results.json. No forwarding skill claim comes from these exposed periods; no confidence interval is used to override failed practical gates.

Candidate coverage misses951/47221 dressed skaters in2023,993/47224 in2024,995/47230 in2025 (~2.0–2.1%), plus90/68/60 ambiguous membership rows quarantined before seeing outcomes. Candidate/source coverage is materially below the required99.5%. Unpaired rows85/131/107 remain explicitly counted rather than disappearing from the headline ledger.

## Error receipts and remaining hockey information

The development primary algebraic mean-absolute contributions are shot-rate/conversion/normalvariance .97675 SOG, availability .29675, TOI .17779, teamenvironment .07287. They are not independent causal effects and can cancel. Actual-TOI×pregamerates is postgame diagnostic only. Available total-attempt and conversion errors were measured independently on42192 pairedprimary rows and35288 attempt-positive conversion rows. The remaining strength-specific/current decomposition is blocked, not invented. phase1b_reporting_errata.json clarifies an older generic blocked-label template field; the specific2023 totalattempt results are present and unchanged.

The largest measured unresolved term is player shot-generation/conversion/variance. A better recent-average formula did not solve it. Genuinely new evidence should be qualified line/PP promotions, EV/PP/PK opportunity changes, authorized strength-specific attempts and source-certified player membership/callups. A fresh protocol must test these observations at their actual cutoff; no postgame deployment label masquerades as pregame knowledge.

## Prospective execution and reproducibility

The Phase0 registry and source inventory are snapshot-pinned and remain byte-identical. Phase1B updates live in phase1b_research_registry.json and phase1b_source_inventory.json, without invalidating historical hashes.

Independent professional forecast intake requires pre-cutoff author/method/PMF/evidence hashes and explicit hockey-only evidence authorization. Scripted analyst baseline bridges prior seasons and has no five-current-season-game minimum, but is not an independent human. Oct7 comparator gaps remain missing forever.

Preregistered Phase1B forward gates retain28days/15000 meaningful playergames per horizon, T90primary,2%CRPS/.05SOGMAE practical gain against strongest identical-row comparator and paired two-calendar-week moving-block evidence; independent TOI/attempt acceptance, calibration and coverage guards. Original B2 gates remain unchanged. No clean sample exists forPhase1B. No one-night promotion.

Run source audit, unit tests, develop (new outputdirectory), then diagnose only with the locked hashes. The initial developmentlock is retained; a separate integrity amendment updates snapshot/forward-writer hashes and strict forward-only history-vintage guards, leaving modelcode,priors,decisions and numerical results unchanged. Full receipt archives are deterministic CI artifacts (~24–25MB each); committed examples are the first10 chronological records per period, never handpicked favorable cases.

**Uncertified:** dressing/availability at every horizon, current EV/PP/PK line assignments, roster ownership from current announcements, goalie/coach/zone state, new API collection rights, strength-specific attempts, independent professional superiority and Phase1B forward improvement.

**No production/API/frontend/scheduler/other-sport code changed; no original forecast/engine artifact changed; no merge.**
