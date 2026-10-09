# NFL shadow dispatch incident and recovery

This repair starts from main, independently of draft research PR #64. No missed horizon is backfilled. R11 remains unresolved; operational shadow delivery is not scientific promotion.

## Proven incident

- Run [37622405399](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/37622405399), job112795664348, logged `HTTP Error 404` during schedule acquisition. The old release URL is unavailable. The verified replacement is `https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv` (HTTP200 on 2026-10-09; exact retrieval digest in the evidence artifact).
- Run [37839965174](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/37839965174), job113526572023, was green after100ticks; its final three ticks said `another dispatcher holds the lock`.
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

Next PHI/JAX kickoff:2026-10-11T13:30Z. T24:2026-10-10T13:30Z; T90:2026-10-11T12:00Z. The capture begins four minutes before each cutoff; completed retrieval must be within five minutes before cutoff and never later. Generation must finish before kickoff.

For each horizon verify: original prefit existed before cutoff; real source retrieval timestamp and source hashes; meaningful immutable batch; DONE or explicit partial comparator status; durable branch SHA after capture and after forecast; read-only publisher includes exactly those IDs. Check the public raw artifact and New Engine page, not only Actions logs. Only after BOTH real horizons are present may delivery be called operationally verified. Synthetic tests are plumbing evidence, never forward forecast evidence.
