import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import nfl_phase1e_scheduler as S  # noqa: E402


def make(tmp_path):
    d = S.Dispatcher(tmp_path, 1000, runner=object(), log=lambda m: None, sleep=lambda s: None)
    d.heartbeat = lambda *a, **k: None
    return d


def test_checkpoint_runs_after_each_changing_tick_before_next_wait(tmp_path):
    d = make(tmp_path)
    marks = tmp_path / "marks.txt"
    calls = []

    def tick():
        calls.append("tick")
        if len(calls) in (1, 3):                                               # ticks 1 and 3 capture something
            (tmp_path / "forecasts" / "batches").mkdir(parents=True, exist_ok=True)
            (tmp_path / "forecasts" / "batches" / f"b{len(calls)}.jsonl").write_text("x")
        return {"n": len(calls)}
    d.tick = tick
    ticks = iter(range(100)); t0 = S.utcnow()
    d.clock = lambda: t0 + S.timedelta(seconds=next(ticks) * 60)
    d.sleep = lambda s: calls.append("wait")
    d.run(duration_min=4, every_sec=60, do_prefit=False, checkpoint_cmd=f"echo $(ls forecasts/batches | wc -l) >> {marks}")
    seq = calls
    assert marks.read_text().split() == ["1", "2"]                              # one durable checkpoint per tick that changed state, none for the idle ticks
    assert seq.index("wait") > 0


def test_checkpoint_runs_even_if_tick_raises_and_not_when_unchanged(tmp_path):
    d = make(tmp_path)
    marks = tmp_path / "m.txt"

    def tick():
        (tmp_path / "dispatch_ledger.jsonl").write_text("row\n")
        raise RuntimeError("provider blew up after the ledger write")
    d.tick = tick
    with pytest.raises(RuntimeError):
        d.run(duration_min=0, every_sec=60, do_prefit=False, checkpoint_cmd=f"echo cp >> {marks}")
    assert marks.read_text().strip() == "cp"
    d2 = make(tmp_path / "idle")
    d2.tick = lambda: {}
    d2.run(duration_min=0, every_sec=60, do_prefit=False, checkpoint_cmd="echo cp >> m2.txt")
    assert not (tmp_path / "idle" / "m2.txt").exists()


def test_workflow_passes_checkpoint_and_script_is_state_only():
    w = (REPO / ".github/workflows/nfl_phase1e_shadow.yml").read_text()
    assert "--checkpoint-cmd" in w and "scripts/nfl_state_checkpoint.sh" in w
    sc = (REPO / "scripts/nfl_state_checkpoint.sh").read_text()
    assert "git push" in sc and "--force" not in sc and "reset --hard" not in sc
