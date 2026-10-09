# NFL shadow dispatch incident and recovery

This repair starts from main, independently of draft research PR #64. No missed horizon is backfilled. R11 remains unresolved; operational shadow delivery is not scientific promotion.

## Proven incident

- Run [37622405399](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/37622405399), job 112795664348, logged `HTTP Error 404` during schedule acquisition. The old release URL is unavailable. The verified replacement is `https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv` (HTTP200 on 2026-10-09; exact retrieval digest in the evidence artifact).
- Run [37839965174](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/37839965174), job 113526572023, was green after 100 ticks; its final three ticks said `another dispatcher holds the lock`.
- Replaying the original binding check against authentic state `edd871f9` produced **LockError before flock**: bound host `machine:bb9ebc7b4ed8498385fd3b7f0a389054`; inspecting host `machine:f34bda38c22c4dee98a9bc8da64e12a2`; `distributed writers are unsupported (explicit rebind required)`. This empirically proves host denial, distinct from local contention. The original Actions code swallowed the exact exception; its log does not record the machine ID of that old runner.
- TB/DAL T24 at2026-10-08T00:15Z was missed; T90 at2026-10-08T22:45Z was still PLANNED after kickoff. Official published T24/T90 were empty. This is a dispatch failure, not zero eligible players.
- Preserve legacy picks and all research forecasts. PR64's Phase2A forecasts are not the publisher's input.

## Ownership and durability

Only the serialized `nfl-phase1e-shadow` workflow can use the audited Actions ownership adapter. It records immutable ownership transitions and hashes of legacy host bindings. It retains local session flock and per-store flock. Existing physical bindings are never deleted/rebound; ordinary local writers remain bound to their original host.

Every checkpoint checks remote state HEAD, verifies all existing state is unchanged or a JSONL prefix, then pushes a descendant using an explicit expected SHA lease. Remote advancement fails closed; no ledger rebase. Retry transient failures only while the expected remote SHA remains unchanged. Preserve failure diagnostics instead of claiming durability.

All captured raw CAS content is chunked/compressed with deterministic gzip and sha256 onto the state branch. Restore validates chunks and original content; cache is only an accelerator. A valid source snapshot is checkpointed before forecast generation, then forecast batches/DONE state are checkpointed. Checkpoint failure fails the workflow.

The live schedule allowlist retains consumed identity/kickoff/rest/coach metadata and a **completed-prior-game boolean** in the frozen loader's required `result` column. Numeric results, target-game scores, QB starters, odds, weather and arbitrary extra columns are excluded. The old frozen sanitizer and all frozen model code hashes remain unchanged.

## Recovery commands

After review, dispatch the shadow workflow on the repair branch (same global concurrency as main):

```
gh workflow run nfl_phase1e_shadow.yml --ref fix/nfl-shadow-dispatch-reliability -f mode=audit
gh workflow run nfl_phase1e_shadow.yml --ref fix/nfl-shadow-dispatch-reliability -f mode=run -f duration_min=0
gh workflow run nfl_new_engine_publish.yml --ref fix/nfl-shadow-dispatch-reliability
```

`audit` downloads real sources, verifies the existing immutable prefit, and durably seals source bytes; it fits nothing, generates no forecasts and earns no T24/T90 evidence. `run` checks the actual schedule, closes overdue keys, and captures only inside the original horizon window. An overdue-key repair is deliberately red and durable. Run again for a fresh healthy invocation. `readiness` uses deep source checks. Stop and inspect any explicit source/prefit/ownership error; do not remove a gate.

Publisher uses a separate main output checkout, so manually executing reviewed repair code can update the presentation JSON without merging any code. The collector does not import PR64 research. Publisher verifies batch footer, exact horizon/cutoff, pre-cutoff source manifest/blob digests, generation before kickoff and pre-cutoff prefit provenance. Missing evidence produces explicit PUBLISH_EMPTY verification errors, not a fallback.

The watchdog checks successful schedule capture, failures, open overdue keys and the cutoff window; it cannot capture forecasts. One recovery dispatch while no run is active; no unconditional self-chain. Hosted cron is still best-effort: it cannot guarantee an on-time run. Use bounded manual recovery around the next cutoff until post-review deployment and repeated live delivery are verified.

## Live acceptance still required

Next PHI/JAX kickoff: 2026-10-11T13:30Z. T24: 2026-10-10T13:30Z; T90: 2026-10-11T12:00Z. The capture begins four minutes before each cutoff; completed retrieval must be within five minutes before cutoff and never later. Generation must finish before kickoff.

For each horizon verify: original prefit existed before cutoff; real source retrieval timestamp and source hashes; meaningful immutable batch; DONE or explicit partial comparator status; durable branch SHA after capture and after forecast; read-only publisher includes exactly those IDs. Check the public raw artifact and New Engine page, not only Actions logs. Only after BOTH real horizons are present may delivery be called operationally verified. Synthetic tests are plumbing evidence, never forward forecast evidence.

## Executed verification (2026-10-09)

- Real source audit [37975520860](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/37975520860): success; 38 sources, zero optional/required omissions, no completed-game lag, Week 5 prefit 45827ae5... verified. Real full-source retrieval 18:47:05.806073Z; no model fitting or forecast generation in this audit.
- Real dispatcher [37975767701](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/37975767701): expected failure, after durably appending TB/DAL T90 MISSED_REAL_CUTOFF. Its old PLANNED record was preserved. No forecast was generated/backfilled. Healthy follow-up [37976066944](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/37976066944) succeeded and reported 0 due forecasts.
- Immutable verification at state f219fa05268ccc3644233e6d77c838a15176c8b7:95 original files verified; all forecast/baseline/comparator batches, artifacts and host bindings byte-identical; all historical JSONL bytes retained as prefixes. Physical binding remains bb9ebc...; audited Actions writer566c0b... works without replacing it.
- Full frozen-engine delivery test: real source bytes and existing prefit,100,000 draws, injected **SYNTHETIC TEST TIME ONLY**.301 PHI/JAX rows per horizon;198 T24 and230 T90 rows withP(active)>=.75. Both horizons reachedDONE; publisher reproduced both after the CAS cache was deleted, using durable backups.1test passed in249.60 s. No test forecast entered nfl-shadow-state or public docs. This is plumbing evidence, NOT subsequent live forecast evidence.
- Regression suite159 passed,1 optional integration skip,2 historical-build tests deselected. The two failures were reproduced on a clean worktree of original main 913fcae. Frozen source hashes and code_identity b77dc75b... are checked separately and unchanged. The optional full-engine test was also explicitly run as described above.
- Public read-only publisher runs37976072236/37976319056 saved diagnostic JSON on main. Chromium360×800 confirmed the actual New Engine page consumed it, showed OPERATIONAL_FAILURE/MISSED_CUTOFF, had360px document width and called no legacy endpoint. The proposed repair HTML moves the warning above the long ad-hoc section; viewport screenshot is a development artifact, not deployed HTML.

**Deployment limitation:** scheduled main still executes old code until this PR is reviewed/merged. An old main publisher did overwrite a repaired diagnostic once; another manual repair publisher restored it. Therefore a successful manual branch execution is not proof of reliable recurring delivery on main. No automatic merge is performed. The repair publisher now preserves the diagnostic and fails visibly with PUBLISH_EMPTY when an operationally failed slate is empty.

At the next real PHI/JAX horizons, use bounded recovery on the reviewed repair ref if main remains unmerged: start a 45-minute run around 13:00Z on October10 for T24, and 11:30Z on October11 for T90, then invoke the repair publisher. Capture itself still starts only at 13:26Z  / 11:56Z. No recurring external service was activated. Both actual horizon receipts and public IDs remain pending.
