"""Sunday international early-game refresh must run after inactive release and pre-kickoff."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

UTC=ZoneInfo("UTC")
EASTERN=ZoneInfo("America/New_York")

def test_early_sunday_2026_madrid_post_inactives_refresh_contract():
    workflow=(Path(__file__).resolve().parents[1]/".github/workflows/all_sports_predictions.yml").read_text()
    assert 'cron: "7 13 * * *"' in workflow
    assert 'cron: "43 13 * * 0"' in workflow
    tip=datetime(2026,11,8,9,30,tzinfo=EASTERN)
    inactive_time=datetime(2026,11,8,8,0,tzinfo=EASTERN)
    early=datetime(2026,11,8,13,7,tzinfo=UTC).astimezone(EASTERN)
    backup=datetime(2026,11,8,13,43,tzinfo=UTC).astimezone(EASTERN)
    assert inactive_time<early<backup<tip
    assert int((tip-backup).total_seconds()/60)==47
    assert 'id: nfl_publish' in workflow
    assert "git add -A -- docs/nfl_predictions.json" in workflow
    assert "python -u nfl_role_snapshot_batch.py" in workflow
    assert "python -u nfl_game_day_contract.py" in workflow
    assert "if: steps.nfl.outcome == 'success'" in workflow
