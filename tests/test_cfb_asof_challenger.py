"""Leak-safe CFB shadow experiment invariants; no incumbent retraining here."""
import json
import sqlite3

import pytest

import cfb_asof_challenger as challenger
import cfb_context_v2 as legacy


def test_strict_asof_features_unchanged_by_target_game_roster_and_result():
    db = sqlite3.connect(":memory:")
    db.execute("""CREATE TABLE games (
        season INTEGER, week INTEGER, home_team TEXT, away_team TEXT,
        home_points REAL, away_points REAL, neutral_site INTEGER)""")
    db.execute("""CREATE TABLE player_games (
        season INTEGER, week INTEGER, game_id TEXT, game_date TEXT,
        team TEXT, player_id TEXT, carries INTEGER, receptions INTEGER,
        pass_attempts INTEGER)""")
    for week in range(1, 4):
        db.execute("INSERT INTO games VALUES (2025,?,?,?, ?,?,0)",
                   (week, "A", "B", 22 + week, 10))
        db.executemany(
            "INSERT INTO player_games VALUES (2025,?,?,?,?,?,?,?,?)",
            [(week, str(week), f"2025-09-{week:02d}", "A", "rb", 10, 1, 0),
             (week, str(week), f"2025-09-{week:02d}", "A", "qb", 1, 0, 20),
             (week, str(week), f"2025-09-{week:02d}", "B", "rb2", 10, 1, 0)])
    db.commit()
    before = challenger.StrictAsOf(db).features(
        {}, 2025, 4, "A", "B", "rb")
    assert before["vacated_car"] is None
    assert before["vacated_rec"] is None
    assert before["qb_change"] is None
    assert before["carry_share_last3"] == pytest.approx(10 / 11)
    assert before["team_plays_last3"] == pytest.approx(31)
    # Poison target week's actual absences and entire outcome.
    db.execute("INSERT INTO games VALUES (2025,4,'A','B',0,98,0)")
    db.execute("INSERT INTO games VALUES (2025,5,'A','B',99,0,0)")
    db.execute(
        "INSERT INTO player_games VALUES (2025,4,'4','2025-10-10','A','replacement',150,50,0)")
    db.commit()
    after = challenger.StrictAsOf(db).features(
        {}, 2025, 4, "A", "B", "rb")
    assert before == after


def test_serving_future_week_is_not_unknown_rating_or_hindsight():
    db = sqlite3.connect(":memory:")
    db.execute("""CREATE TABLE games (
       season INTEGER, week INTEGER, home_team TEXT, away_team TEXT,
       home_points REAL, away_points REAL, neutral_site INTEGER)""")
    db.execute("""CREATE TABLE player_games (
        season INTEGER, week INTEGER, game_id TEXT, game_date TEXT,
        team TEXT, player_id TEXT, carries INTEGER, receptions INTEGER,
        pass_attempts INTEGER)""")
    db.execute("INSERT INTO games VALUES (2025,1,'A','B',50,0,0)")
    db.execute("INSERT INTO player_games VALUES (2025,1,'1','2025-08-21','A','rb',12,0,0)")
    db.commit()
    pre = challenger.StrictAsOf(db).features({}, 2025, 2, "A", "B", None)
    assert pre["carry_share_last3"] is None
    assert pre["rec_share_last3"] is None
    assert pre["pass_share_last3"] is None
    assert pre["team_rating"] > pre["opp_rating"]
    assert pre["vacated_car"] is None
    # After Week 2 outcome and Week 3 have appeared in full DB,
    # reconstructing BEFORE week 2 must return exactly the same inputs.
    db.execute("INSERT INTO games VALUES (2025,2,'A','B',0,50,0)")
    db.execute("INSERT INTO games VALUES (2025,3,'A','B',0,50,0)")
    db.commit()
    post = challenger.StrictAsOf(db).features({}, 2025, 2, "A", "B", None)
    assert pre == post


def test_trainer_isolation_restores_production_globals(tmp_path, monkeypatch):
    data = tmp_path / "has_source.db"
    data.touch()
    out = tmp_path / "isolated_challenger"
    calls = []
    originals = (legacy.MODEL_DIR, legacy.build_context, legacy.add_context)

    def fake_run(db, eval_seasons):
        assert legacy.MODEL_DIR == out.resolve()
        assert legacy.build_context is challenger.StrictAsOf
        assert legacy.add_context is challenger._inject_strict_features
        assert eval_seasons == (2025,)
        calls.append("train")

    def fake_finalize(db, calib_season):
        assert calib_season == 2025
        calls.append("calibrate")

    monkeypatch.setattr(legacy, "run", fake_run)
    monkeypatch.setattr(legacy, "finalize", fake_finalize)
    manifest = challenger.train_shadow(data, out)
    assert calls == ["train", "calibrate"]
    assert (legacy.MODEL_DIR, legacy.build_context, legacy.add_context) == originals
    assert manifest["production_models_modified"] is False
    assert manifest["blind_2026_forward_evidence_required"] is True
    assert json.loads((out / "research_manifest.json").read_text()) == manifest


def test_reject_production_and_published_artifact_targets():
    for target in (legacy.MODEL_DIR, legacy.REPO / "docs",
                   legacy.REPO / "cfb_models",
                   legacy.MODEL_DIR / "nested",
                   legacy.REPO / "docs" / "oops"):
        with pytest.raises(ValueError, match="refusing|research artifacts"):
            challenger.guarded_output_dir(target)


def test_adapter_uses_same_exact_feature_entrypoint_for_replay_and_shadow():
    class Dummy:
        def features(self, *args):
            return {"joined": args}
    x = challenger._inject_strict_features(
        {"base": 1}, 2025, "rb", 9, "A", "B", Dummy(), {})
    assert x["joined"] == ({"base": 1}, 2025, 9, "A", "B", "rb")
