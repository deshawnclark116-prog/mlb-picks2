"""Regression: concurrent valid NHL publishers may never destroy one another's site updates.

Local Git remotes simulate a stale website checkout colliding with a newer
independently published forecast. Nothing here connects to live GitHub.
"""
import json
import subprocess
from pathlib import Path

import nhl_v2_cutoff_window_worker as W


def git(where, *args):
    return subprocess.check_output(["git", *args], cwd=where, text=True,
                                   stderr=subprocess.STDOUT).strip()


def test_site_publication_race_refreshes_and_preserves_another_valid_commit(tmp_path, monkeypatch):
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    writer = tmp_path / "seed"
    subprocess.run(["git", "clone", str(bare), str(writer)], check=True, capture_output=True)
    git(writer, "config", "user.name", "Fixture")
    git(writer, "config", "user.email", "fixture@example.org")
    (writer / "docs").mkdir()
    (writer / "docs/nhl_v2_shadow.json").write_text('{"status":"prior"}\n')
    (writer / "docs/other_sport.txt").write_text("MUST_PRESERVE\n")
    git(writer, "add", ".")
    git(writer, "commit", "-m", "base")
    git(writer, "branch", "-M", "main")
    git(writer, "push", "-u", "origin", "main")
    git(writer, "checkout", "-b", "research")
    git(writer, "push", "-u", "origin", "research")

    web = tmp_path / "web"
    other = tmp_path / "other"
    original = tmp_path / "original"
    for target in [web, other, original]:
        subprocess.run(["git", "clone", "--branch", "main", str(bare), str(target)],
                       check=True, capture_output=True)
    git(other, "config", "user.name", "Other worker")
    git(other, "config", "user.email", "other@example.org")
    git(original, "checkout", "-b", "research", "origin/research")
    rerenders = []

    def fake_verified_shadow(o, site):
        rerenders.append(git(site, "rev-parse", "HEAD"))
        current = json.loads((site / "docs/nhl_v2_shadow.json").read_text())
        # A real source-derived updater regenerates after detecting remote changes.
        current["fresh_nhl_count_forecasts"] = 693
        (site / "docs/nhl_v2_shadow.json").write_text(json.dumps(current) + "\n")
    monkeypatch.setattr(W, "_render_verified_shadow", fake_verified_shadow)
    monkeypatch.setattr(W.time, "sleep", lambda _: None)

    real_run = W.subprocess.run
    raced = [False]

    def concurrent_push(argv, *args, **kwargs):
        if (list(argv) == ["git", "push", "origin", "HEAD:main"]
                and not raced[0]):
            raced[0] = True
            # A different runner publishes the large shadow first.
            (other / "docs/nhl_v2_shadow.json").write_text(
                '{"other_worker_forecast":123}\n')
            git(other, "add", "docs/nhl_v2_shadow.json")
            git(other, "commit", "-m", "simultaneous valid forecast update")
            git(other, "push", "origin", "HEAD:main")
        return real_run(argv, *args, **kwargs)
    monkeypatch.setattr(W.subprocess, "run", concurrent_push)
    W.BRANCH = "research"
    try:
        W.publish(original, web)
    finally:
        W.BRANCH = "feature/nhl-v2-nfl-style-projection-board"
    assert raced[0] and len(rerenders) == 2
    git(web, "fetch", "origin", "main")
    published = json.loads(git(web, "show", "origin/main:docs/nhl_v2_shadow.json"))
    assert published["fresh_nhl_count_forecasts"] == 693
    assert published["other_worker_forecast"] == 123
    assert git(web, "show", "origin/main:docs/other_sport.txt") == "MUST_PRESERVE"
    assert "rebase" not in "\n".join(rerenders)


def test_collision_recovery_does_not_force_push_or_rewrite_any_other_prediction_card():
    import inspect
    source = inspect.getsource(W.publish)
    assert "reset" in source and "--hard" in source
    assert "_render_verified_shadow" in source
    assert "merge" in source and "--ff-only" in source
    assert "rebase" not in [x for x in source.splitlines() if "command(" in x]
    assert "--force" not in source
