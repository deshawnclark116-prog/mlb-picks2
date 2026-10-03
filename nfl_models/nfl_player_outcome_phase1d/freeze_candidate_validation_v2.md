# Freeze-candidate validation v2 (Phase 1D)

Two sections only. Every PASS below was computed from recorded evidence by `nfl_phase1d_report.py validation`; anything not demonstrated is a BLOCKER. The final `nfl_player_outcome_phase1_freeze.json` has NOT been created.

## PASS

- **R1** Post-hoc audit: every Phase 1C post-hoc decision classified; 2022 wk1 = DATA/WARMUP DEFINITION CORRECTION with a permanent prospective rule and both analyses kept in provenance; calibration / depth universe development-selected; N=25,000 an engineering selection; machine scan of Phase 1C files enumerated  
  evidence: posthoc_audit.json/.md: 19 decisions, 191 scan hits; 2022 wk1 rows have all-zero decayed history in 100% of rows
- **R2** Accuracy-number integrity: slices A-H published separately for rushing / receiving / receptions / passing at T24 and T90, tolerances +/-5..40 (passing wider), every figure names its universe, a guard refuses unlabeled accuracy fields  
  evidence: accuracy_slices.json/.md; nfl_phase1d_accuracy.assert_labeled; tests/test_nfl_phase1d.py::test_no_unlabeled_accuracy_field_can_be_emitted
- **R3** Protocol Amendment G registered in protocol v1.2 before any clean-forward outcome, consistent with the code and the evaluators  
  evidence: nfl_models/nfl_player_outcome_phase1_protocol.json amendments_v1_2[G]; nfl_phase1d_gates.py
- **R4** Complete input snapshot / content-addressed store: every source read by Phase 1 code is stored as an immutable sha256 blob with retrieval ts, cutoff, byte count, blob path, parser version; identical bytes stored once; hash verified on read; all snapshot sets of the dry run re-verified  
  evidence: 82 snapshot sets verified, 0 problems
- **R5** Schedule / kickoff safety: kickoff and cutoff from the schedule snapshot, retrieval <= cutoff enforced, append-only revision ledger, flex / postponement / cancellation / neutral-site rules implemented and tested  
  evidence: nfl_phase1d_schedule.py; tests; schedule ledger rows in the dry run
- **R6** Phase 1A serialized (SHA256 per artifact) and LiveLoader accepted: serialized model + snapshot time-travel reconstruction equals the historical research replay within 1e-9 on every selected week / horizon  
  evidence: equivalence_results.json max |diff| 0.0; freeze bundle 02d2ca55c1e2b2f6
- **R7** Deterministic walk-forward refit: training = completed weeks before the target week, no forward-target rows (audited), hyper-parameters / architecture frozen, every refit hashed and logged; two independent fits of the same window give the same bundle hash  
  evidence: freeze_fit_determinism.json: 02d2ca55c1e2b2f67b4adba55870379a01d384172258782a2de2fb8d89c222a9
- **R8** Concurrent writer safety: inter-process lock, stale-lock recovery after kill -9, concurrent writers, same id + different bytes HARD ERROR, lock timeout explicit, store bound to one host (flock is not distributed)  
  evidence: nfl_phase1_store_lock.py; tests/test_nfl_phase1d.py
- **R9** Multi-week time-travel dry run of the complete runner on the pre-registered burned weeks (early / mid / late 2025 + 2026 wk1-3), T24 and T90, with snapshot retrieval, serialization, model fit/load, forecast, restart, duplicate verification, grading, schedule cutoff, provenance, score append, board  
  evidence: 2025 wk4: 32/32 ok; 2025 wk11: 30/30 ok; 2025 wk16: 32/32 ok; 2026 wk1: 32/32 ok; 2026 wk2: 32/32 ok; 2026 wk3: 32/32 ok
- **R10** Zero-leak perturbation: corrupting target-game outcomes leaves forecast bytes unchanged (source-level) and distribution outputs unchanged even when the corrupted rows are handed to the loader in the snapshot; corrupting a prior completed game changes later forecasts  
  evidence: dry_run/dry_run_*.json perturbation blocks
- **R12** Calibration frozen: exact fitted values and sha256 of every stored map, labelled development-selected; no calibration selection re-run  
  evidence: calibration_freeze.json all_maps_sha256 7eb317b28b183572
- **R13** Freeze candidate v2 rebuilt from a clean commit: recorded git SHA equals HEAD at verification and every listed file hash re-verified from disk  
  evidence: nfl_phase1d_freeze_candidate.py verify
- **R14** Complete test matrix green (Phase 0B, 1A, 1B, 1C and 1D tests)  
  evidence: test_nfl_phase1_invariants.py: 31 PASS / 0 FAIL; test_nfl_phase1b_invariants.py: 1 PASS / 0 FAIL; test_nfl_phase1c_invariants.py: 14 PASS / 0 FAIL; test_nfl_phase1d.py: 24 PASS / 0 FAIL; test_nfl_player_outcome_invariants.py: 110 PASS / 0 FAIL
- **R15** No final freeze file created; production serving untouched (only shadow research paths changed since Phase 1C)  
  evidence: tests/test_nfl_phase1d.py::test_no_final_freeze_file_and_production_untouched

## BLOCKER

- **R11 NOT DEMONSTRATED** N = 25,000 engineering audit against criteria fixed before running (MC SE <= 1% of model RMSE; CRPS within 0.5% of a 100k reference; tail-probability SE <= 0.005; repeat stability)  
  evidence: n_engineering_audit.json verdict {'C1': True, 'C2': True, 'C3': False, 'C4': False, 'N_25000_retained': False}
- **S1** Live end-to-end operation is not demonstrated. Retrieval was exercised against a local HTTP mirror (nfl_phase1d_cas.live_sources) and every other step in time-travel mode; nothing was retrieved from the real providers (outbound network is restricted in this environment), no scheduler / cron invokes the runner at each cutoff, and the real providers' update lags and file layouts are unverified. Until a dry run against the live providers (before the first scored kickoff) succeeds, live readiness is not shown.
- **S2** The v2 incumbent challenge gate needs the production v2 forecasts logged BEFORE kickoff by the production pipeline (an external input that uses bookmaker features). No such forward log is wired to or verified by the Phase 1 system, so the incumbent-replacement gate cannot be evaluated; the research verdict is unaffected.
