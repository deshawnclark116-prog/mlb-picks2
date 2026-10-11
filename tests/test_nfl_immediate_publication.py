"""Immediate delivery cannot wait for NHL and cannot commit other sports early."""
from pathlib import Path
import subprocess
import yaml


def test_immediate_validated_nfl_commit_is_ordered_before_slow_sports():
    source=Path(".github/workflows/all_sports_predictions.yml").read_text()
    job=yaml.safe_load(source)["jobs"]["build"]
    steps=job["steps"]
    indices={s["name"]:i for i,s in enumerate(steps) if "name" in s}
    assert indices["NFL pipeline"] < indices["Publish validated NFL game-day predictions immediately"]
    assert indices["Publish validated NFL game-day predictions immediately"] < indices["CFB pipeline"]
    assert indices["CFB pipeline"] < indices["Tennis pipeline"] < indices["NHL pipeline"] < indices["Commit JSON"]
    immediate=next(s for s in steps if s.get("id")=="nfl_publish")
    assert immediate["if"]=="steps.nfl.outcome == 'success'"
    assert immediate["continue-on-error"] is True
    run=immediate["run"]
    assert "python -u nfl_game_day_contract.py" in run
    assert "git add -A -- docs/nfl_predictions.json" in run
    assert "git push origin HEAD:main" in run
    assert "git rebase --autostash origin/main" in run
    assert "docs/cfb" not in run and "docs/tennis" not in run and "docs/nhl" not in run
    last=steps[indices["Fail the run if any sport's pipeline failed"]]["run"]
    assert "steps.nfl_publish.outcome" in last
    assert "steps.nfl.outcome" in last


def test_real_git_immediate_only_nfl_files_are_publicated_before_other_sports(tmp_path):
    root=tmp_path
    remote=root/"remote.git"
    working=root/"work"
    def run(*args,cwd=None):
        result=subprocess.run(["git",*args],cwd=cwd,capture_output=True,text=True)
        assert result.returncode==0,(args,result.stdout,result.stderr)
        return result.stdout.strip()
    run("init","--bare",str(remote))
    run("clone",str(remote),str(working))
    run("checkout","-b","main",cwd=working)
    run("config","user.name","delivery-tester",cwd=working)
    run("config","user.email","delivery@invalid.test",cwd=working)
    docs=working/"docs";docs.mkdir()
    originals={
      "nfl_predictions.json":"old validated NFL",
      "nfl_predictions_2026_w05.json":"old NFL archive",
      "nfl_picks_log.jsonl":"old immutable prediction log",
      "nfl_record.json":"old NFL grade",
      "mlb_predictions.json":"old MLB",
      "nhl_predictions.json":"old NHL"
    }
    for name,data in originals.items():
        (docs/name).write_text(data)
    run("add","-A","docs",cwd=working)
    run("commit","-m","before",cwd=working)
    run("push","origin","HEAD:main",cwd=working)
    # Production pipeline computed and verified an NFL document and also
    # changed some other sport outputs; only NFL may be staged and pushed now.
    (docs/"nfl_predictions.json").write_text("validated 2026-10-11 NFL")
    (docs/"nfl_predictions_2026_w05.json").write_text("validated Week5 archive")
    (docs/"nfl_picks_log.jsonl").write_text("new first-seen pregame entry")
    (docs/"nfl_record.json").write_text("updated prior grading")
    (docs/"mlb_predictions.json").write_text("pending MLB data")
    (docs/"nhl_predictions.json").write_text("pending NHL data")
    run("add","-A","--","docs/nfl_predictions.json",
        "docs/nfl_predictions_2026_w05.json","docs/nfl_picks_log.jsonl","docs/nfl_record.json",cwd=working)
    run("commit","-m","NFL early publication",cwd=working)
    run("push","origin","HEAD:main",cwd=working)
    head=run("--git-dir",str(remote),"rev-parse","refs/heads/main")
    def snapshot(name):
        return run("--git-dir",str(remote),"show",f"{head}:docs/{name}")
    assert snapshot("nfl_predictions.json")=="validated 2026-10-11 NFL"
    assert snapshot("nfl_picks_log.jsonl")=="new first-seen pregame entry"
    assert snapshot("mlb_predictions.json")=="old MLB"
    assert snapshot("nhl_predictions.json")=="old NHL"
    # Original combined workflow commits the other sports later.
    run("add","-A","docs",cwd=working)
    run("commit","-m","other sports",cwd=working)
    run("push","origin","HEAD:main",cwd=working)
    final=run("--git-dir",str(remote),"rev-parse","refs/heads/main")
    assert run("--git-dir",str(remote),"show",f"{final}:docs/mlb_predictions.json")=="pending MLB data"
    assert run("--git-dir",str(remote),"show",f"{final}:docs/nfl_predictions.json")=="validated 2026-10-11 NFL"
