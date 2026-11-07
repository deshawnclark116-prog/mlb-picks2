"""Sunday international-game NFL inactives refresh remains before kickoff."""
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import yaml

REPO=Path(__file__).resolve().parents[1]
ET=ZoneInfo("America/New_York")

def test_week9_madrid_second_sunday_pregame_refresh_follows_inactive_window():
    doc=yaml.safe_load((REPO/".github/workflows/all_sports_predictions.yml").read_text())
    schedule=doc.get("on",doc.get(True))["schedule"]
    entries=[x["cron"] for x in schedule]
    assert "7 13 * * *" in entries  # first 8:07 EST run
    assert "43 13 * * 0" in entries # independent 8:43 EST backup
    first=datetime(2026,11,8,13,7,tzinfo=timezone.utc)
    backup=datetime(2026,11,8,13,43,tzinfo=timezone.utc)
    inactive=datetime(2026,11,8,13,0,tzinfo=timezone.utc)
    kickoff=datetime(2026,11,8,14,30,tzinfo=timezone.utc)
    assert first.astimezone(ET).strftime("%H:%M")=="08:07"
    assert backup.astimezone(ET).strftime("%H:%M")=="08:43"
    assert inactive<first<backup<kickoff
    assert (kickoff-backup).total_seconds()==47*60
    assert "37 14 * * *" in entries # this run is too late for Madrid

def test_extra_refresh_preserves_original_nfl_role_and_schedule_pipeline():
    text=(REPO/".github/workflows/all_sports_predictions.yml").read_text()
    assert text.count('cron: "43 13 * * 0"')==1
    assert "python -u nfl_serving_builder_a.py" in text
    assert "python -u nfl_role_snapshot_batch.py" in text
    assert "docs/nfl_predictions.json" in text
    assert "docs/nfl_role_snapshot.json" in text
    doc=yaml.safe_load(text)
    nfl_step=next(s for s in doc["jobs"]["build"]["steps"] if s.get("id")=="nfl")
    assert "set -e" in nfl_step["run"]
    assert "nfl_pregame_delivery.py" in nfl_step["run"] or "nfl_serving_builder_a.py" in nfl_step["run"]
