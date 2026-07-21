# Trained model backups

This directory holds a git-backed copy of every trained model file that
production actually serves (normally live only at `/data/models` on
Render's persistent disk). It exists for one reason: if that disk were
ever lost, wiped, or the service recreated, the trained artifacts — weeks
of gated, predictions-first-validated work — would otherwise be gone with
no way to recover them. The *code* that produces models was always in
git; the models themselves were not, until now.

## How it works

- **Backup**: run `backup_models_to_repo_a.py` on Render (where the real
  model files live) any time a model gets retrained/re-gated and
  promoted. It copies everything from `/data/models` into this directory
  and writes `manifest.json` (sha256 + size per file), then tells you the
  `git add` / `commit` / `push` commands to run.
- **Restore**: fully automatic. `load_models()` in `api.py` calls
  `_restore_models_from_repo_if_missing()` on every startup — if a model
  file is missing from `/data/models` but present here, it gets copied
  over before loading. A fresh deploy onto an empty `/data` self-heals
  with no manual steps. It will **never** overwrite a file that already
  exists at `/data/models` (so a live model newer than the last backup is
  never silently rolled back).

## Keeping this current

This is a point-in-time snapshot, not a live mirror. After promoting any
new or retrained model, re-run the backup script and push — otherwise a
disaster-recovery restore would bring back an older version than what was
actually live.
