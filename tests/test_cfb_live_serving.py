"""CFB live-serving safety + readiness tests (existing system only). Synthetic data except the artifact-hash / config checks and an optional full-board determinism run
(skipped unless the refreshed foundation db with schedule_snapshot is present).   python tests/test_cfb_live_serving.py"""
import ast
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import cfb_live_readiness_a as RD  # noqa: E402
import cfb_serving_builder_a as SB  # noqa: E402

UTC = timezone.utc
NOW = datetime(2026, 10, 3, 12, 0, 0, tzinfo=UTC)


def ts(dt):
    return dt.strftime("%Y-%m-%dT%H:%MZ")


def pick(market="rushing_yards", team="A", opp="B", pid="p1", **kw):
    return {"market": market, "player_id": pid, "player": "Pat Player", "team": team, "opponent": opp, "season": 2026, "week": 5, "model_prob": 0.6, "prob_over": 0.6, "games_played": 4, **kw}


# ------------------------------------------------------------------ started-game rule
def test_parse_kickoff_variants_and_unverifiable_timestamps():
    a = SB.parse_kickoff("2026-10-03T15:00Z")
    assert a == datetime(2026, 10, 3, 15, 0, tzinfo=UTC) == SB.parse_kickoff("2026-10-03T15:00:00.000Z") == SB.parse_kickoff("2026-10-03T11:00:00-04:00") == SB.parse_kickoff("2026-10-03T15:00:00+00:00")
    for bad in (None, "", "garbage", "2026-10-03T15:00", "2026-10-03"):                                             # no timezone -> unverifiable, never guessed
        assert SB.parse_kickoff(bad) is None, bad


def test_future_game_remains_started_final_and_exact_now_are_removed():
    ko = {frozenset(("F1", "F2")): ts(NOW + timedelta(hours=3)), frozenset(("S1", "S2")): ts(NOW - timedelta(minutes=1)), frozenset(("N1", "N2")): ts(NOW),
          frozenset(("L1", "L2")): ts(NOW - timedelta(hours=5)), frozenset(("J1", "J2")): ts(NOW + timedelta(minutes=1))}
    state = {frozenset(("L1", "L2")): "post", frozenset(("S1", "S2")): "in", frozenset(("F1", "F2")): "pre"}
    picks = [pick(team="F1", opp="F2"), pick(team="S1", opp="S2"), pick(team="N1", opp="N2"), pick(team="L1", opp="L2"), pick(team="J1", opp="J2"), pick(team="M1", opp="M2")]
    keep, dropped = SB.split_pregame(picks, ko, state, NOW)
    assert [p["team"] for p in keep] == ["F1", "J1"]                                                                # future remains; one minute before kickoff is still pregame
    assert keep[0]["kickoff_utc"] == ko[frozenset(("F1", "F2"))]
    assert dropped == {"kickoff_reached_or_passed": 3, "no_authoritative_kickoff": 1}, dropped                       # started (S), exactly-now (N), final (L), unknown kickoff (M)
    assert SB.pregame_status(ts(NOW), NOW) == (False, "kickoff_reached_or_passed")                                  # exactly now: removed, no grace period
    assert SB.pregame_status(ts(NOW + timedelta(minutes=1)), NOW) == (True, None)
    assert SB.pregame_status(ts(NOW + timedelta(hours=1)), NOW, "in") == (False, "espn_state_in")                    # kickoff in the future but ESPN says the game is not 'pre'
    assert SB.pregame_status(ts(NOW + timedelta(hours=1)), NOW, "post") == (False, "espn_state_post")
    assert "S1" not in [p["team"] for p in keep] and picks[1].get("kickoff_utc") is None                              # dropped picks are never mutated


def test_ledger_is_append_only_pregame_only_and_survives_board_removal():
    with tempfile.TemporaryDirectory() as t:
        led = Path(t) / "ledger.jsonl"
        ko = {frozenset(("A", "B")): ts(NOW + timedelta(hours=2)), frozenset(("C", "D")): ts(NOW - timedelta(hours=1))}
        p_pre, p_started = pick(team="A", opp="B", pid="p1"), pick(team="C", opp="D", pid="p2")
        # run 1 (before kickoff of A-B): board keeps A-B; ledger logs only the genuinely pregame pick, with the original timestamp
        led_picks, _ = SB.split_pregame([p_pre, p_started], ko, {}, NOW)
        keys = SB.load_logged_pick_keys(led)
        assert SB.append_new_picks_to_log(led, keys, led_picks, logged_at="2026-10-03T12:00:00+00:00") == 1
        first = led.read_text()
        assert json.loads(first)["player_id"] == "p1" and json.loads(first)["logged_at"] == "2026-10-03T12:00:00+00:00" and json.loads(first)["kickoff_utc"] == ts(NOW + timedelta(hours=2))
        # run 2 (after A-B kicked off): the live board drops it; the ledger entry is untouched and nothing new is appended for the started game
        later = NOW + timedelta(hours=3)
        board, dropped = SB.split_pregame([dict(p_pre, model_prob=0.9), p_started], ko, {}, later)
        assert board == [] and dropped == {"kickoff_reached_or_passed": 2}
        led2, _ = SB.split_pregame([dict(p_pre, model_prob=0.9), p_started], ko, {}, later)
        keys = SB.load_logged_pick_keys(led)
        assert SB.append_new_picks_to_log(led, keys, led2, logged_at="2026-10-03T15:00:00+00:00") == 0
        assert led.read_text() == first                                                                            # byte-identical: removing a pick from the board never deletes / rewrites the ledger entry
        # a re-run before kickoff with a different probability does not overwrite the original independent probability either
        keys = SB.load_logged_pick_keys(led)
        assert SB.append_new_picks_to_log(led, keys, [dict(p_pre, model_prob=0.99)], logged_at="2026-10-03T12:30:00+00:00") == 0 and led.read_text() == first


# ------------------------------------------------------------------ markets / roster
def test_active_markets_and_suspended_markets_cannot_leak():
    assert sorted(SB.MARKETS) == ["passing_touchdowns", "rushing_yards"] and sorted(SB.SUSPENDED_MARKETS) == ["passing_yards", "receiving_yards"]
    assert RD.ACTIVE_MARKETS == ["rushing_yards", "passing_touchdowns", "anytime_touchdowns", "moneyline"] and RD.SUSPENDED == ["passing_yards", "receiving_yards"]
    tree = ast.parse((REPO / "cfb_serving_builder_a.py").read_text())
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "SUSPENDED_MARKETS" and isinstance(n.ctx, ast.Load)]
    assert not loads                                                                                             # no code path reads the suspended configs
    assert set(SB.MARKETS) & set(SB.SUSPENDED_MARKETS) == set()
    for m in RD.SUSPENDED:
        board = {"picks": [pick(market=m)]}
        ok, _ = RD.check_suspended_and_odds(board)
        assert not ok["ok"] and ok["suspended_markets_on_board"] == [m]
        ok, _ = RD.check_suspended_and_odds({"picks": [pick(market=m + "_early_season")]})
        assert not ok["ok"]
    ok, odds = RD.check_suspended_and_odds({"picks": [pick(market=m) for m in RD.ACTIVE_MARKETS]})
    assert ok["ok"] and odds["ok"], (ok, odds)
    assert not RD.check_suspended_and_odds({"picks": [pick(market="rushing_touchdowns")]})[0]["ok"]               # an unknown market is also rejected
    assert "passing_yards" in (REPO / "cfb_serving_builder_a.py").read_text() and SB.SUSPENDED_MARKETS["passing_yards"]["calibration_policy"] == "growing"


def test_roster_labels_never_pretend_certainty():
    idx = {"Team A": {SB._norm_name("Pat Player Jr."), SB._norm_name("José Núñez")}}
    assert SB.roster_label(idx, "Team A", "Pat Player", "1") == "ON_CURRENT_ROSTER_SNAPSHOT"
    assert SB.roster_label(idx, "Team A", "Jose Nunez", "2") == "ON_CURRENT_ROSTER_SNAPSHOT"
    assert SB.roster_label(idx, "Team A", "Someone Else", "3") == "NOT_ON_CURRENT_ROSTER_SNAPSHOT"
    assert SB.roster_label(idx, "Team Z", "Pat Player", "4") == "NO_ROSTER_SNAPSHOT_FOR_TEAM"
    assert SB.roster_label({}, "Team A", "Pat Player", "5") == "ROSTER_SNAPSHOT_MISSING"
    assert SB.roster_label(idx, "Team A", "Team A", None) == "NOT_APPLICABLE_TEAM_MARKET"
    with tempfile.TemporaryDirectory() as t:                                                                       # readiness: a candidate absent from the snapshot fails the roster check
        con = sqlite3.connect(Path(t) / "d.sqlite")
        con.execute("CREATE TABLE current_roster (team TEXT, player_name TEXT, position TEXT, season INTEGER)")
        con.execute("INSERT INTO current_roster VALUES ('Team A','Pat Player','RB',2026)")
        ok = RD.check_roster({"picks": [pick(roster_verification="ON_CURRENT_ROSTER_SNAPSHOT")]}, con, 2026)
        bad = RD.check_roster({"picks": [pick(roster_verification="NOT_ON_CURRENT_ROSTER_SNAPSHOT")]}, con, 2026)
        unl = RD.check_roster({"picks": [pick()]}, con, 2026)
        assert ok["ok"] and not bad["ok"] and bad["n_not_on_roster"] == 1 and not unl["ok"] and "NONE" in ok["injury_inactive_knowledge"]
        con.execute("DELETE FROM current_roster")
        assert not RD.check_roster({"picks": [pick(roster_verification="ROSTER_SNAPSHOT_MISSING")]}, con, 2026)["ok"]


# ------------------------------------------------------------------ schedule
def schedule_db(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE games (game_id TEXT PRIMARY KEY, season INTEGER, week INTEGER, game_date TEXT, kickoff_utc TEXT, home_team TEXT, away_team TEXT, home_points INTEGER, away_points INTEGER)")
    con.execute("CREATE TABLE schedule_snapshot (game_id TEXT PRIMARY KEY, season INTEGER, week INTEGER, kickoff_utc TEXT, espn_state TEXT, espn_status TEXT, home_team TEXT, away_team TEXT, retrieved_at_utc TEXT)")
    for gid, h, a, ko, st, hp in rows:
        con.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)", (gid, 2026, 5, ko[:10], ko, h, a, hp, hp))
        con.execute("INSERT INTO schedule_snapshot VALUES (?,?,?,?,?,?,?,?,?)", (gid, 2026, 5, ko, st, "x", h, a, "2026-10-03T11:00:00+00:00"))
    con.commit()
    return con


def test_schedule_uniqueness_future_split_and_timezone_integrity():
    with tempfile.TemporaryDirectory() as t:
        rows = [("1", "A", "B", ts(NOW + timedelta(hours=3)), "pre", None), ("2", "C", "D", ts(NOW - timedelta(hours=1)), "in", None), ("3", "E", "F", ts(NOW - timedelta(hours=9)), "post", 21)]
        con = schedule_db(Path(t) / "a.sqlite", rows)
        res, games = RD.check_schedule(con, 2026, 5, NOW, live_compare=False)
        assert res["ok"] and res["future_games"] == 1 and res["started_or_final_games"] == 2 and res["earliest_future_kickoff_utc"] == ts(NOW + timedelta(hours=3))
        assert {g["game_id"]: g["future"] for g in games} == {"1": True, "2": False, "3": False} and all(g["schedule_retrieved_at_utc"] for g in games) and [g for g in games if g["game_id"] == "3"][0]["final"]
        con.execute("INSERT INTO games VALUES ('9',2026,5,'2026-10-03','2026-10-03T20:00Z','B','A',NULL,NULL)")                  # the same matchup twice (reversed home/away)
        con.commit()
        res, _ = RD.check_schedule(con, 2026, 5, NOW, live_compare=False)
        assert not res["ok"] and res["duplicate_team_pairs"] == [["A", "B"]]
        con.execute("DELETE FROM games WHERE game_id='9'"); con.execute("UPDATE games SET kickoff_utc='2026-10-03T15:00' WHERE game_id='1'"); con.commit()      # a timestamp without timezone is rejected
        res, _ = RD.check_schedule(con, 2026, 5, NOW, live_compare=False)
        assert not res["ok"] and any(i.get("issue") == "unparseable_or_missing_kickoff" for i in res["issues"])
        con2 = schedule_db(Path(t) / "b.sqlite", [("1", "A", "B", ts(NOW - timedelta(hours=1)), "in", None)])
        assert not RD.check_schedule(con2, 2026, 5, NOW, live_compare=False)[0]["ok"]                                              # no future game at all -> not ready


def test_board_check_requires_strictly_future_pregame_picks():
    games = [{"home": "A", "away": "B", "future": True, "kickoff_utc": ts(NOW + timedelta(hours=1))}, {"home": "C", "away": "D", "future": False, "kickoff_utc": ts(NOW - timedelta(hours=1))}]
    good = {"generated_at_utc": NOW.isoformat(), "picks": [pick(team="A", opp="B", kickoff_utc=ts(NOW + timedelta(hours=1)))], "board_filter": {}}
    assert RD.check_board(good, games, NOW)["ok"]
    started = {**good, "picks": good["picks"] + [pick(team="C", opp="D", kickoff_utc=ts(NOW - timedelta(hours=1)), pid="p2")]}
    r = RD.check_board(started, games, NOW)
    assert not r["ok"] and r["n_not_future"] == 1
    at_ko = {**good, "picks": [pick(team="A", opp="B", kickoff_utc=ts(NOW + timedelta(hours=1)))]}
    assert not RD.check_board(at_ko, games, NOW + timedelta(hours=1))["ok"]                                                         # exactly at kickoff
    dup = {**good, "picks": good["picks"] * 2}
    assert not RD.check_board(dup, games, NOW)["ok"] and RD.check_board(dup, games, NOW)["duplicate_player_market_game"]


# ------------------------------------------------------------------ leakage / calibration / ledger
def synthetic_player_db(path, mutate_target=None):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE player_games (player_id TEXT, player_name TEXT, position TEXT, team TEXT, opponent TEXT, season INTEGER, week INTEGER, game_id TEXT, game_date TEXT, is_home INTEGER, carries INTEGER, rushing_yards INTEGER, receptions INTEGER, receiving_yards INTEGER, pass_attempts INTEGER, completions INTEGER, passing_yards INTEGER, passing_touchdowns INTEGER, passing_interceptions INTEGER, rushing_touchdowns INTEGER, receiving_touchdowns INTEGER)")
    con.execute("CREATE TABLE games (game_id TEXT, season INTEGER, week INTEGER, game_date TEXT, home_team TEXT, away_team TEXT, home_points INTEGER, away_points INTEGER)")
    for w in range(1, 5):
        con.execute("INSERT INTO player_games VALUES ('p1','Pat','RB','A','B',2026,?,?,?,1,?,?,0,0,0,0,0,0,0,0,0)", (w, f"g{w}", f"2026-09-{w:02d}", 15 + w, 60 + 10 * w))
        con.execute("INSERT INTO games VALUES (?,2026,?,?,?,?,?,?)", (f"g{w}", w, f"2026-09-{w:02d}", "A", "B", 20 + w, 10))
    if mutate_target is not None:                                                                                    # a target-week (week 5) row for the same player, with a huge outcome
        con.execute("INSERT INTO player_games VALUES ('p1','Pat','RB','A','B',2026,5,'g5','2026-10-03',1,30,?,0,0,0,0,0,0,0,0,0)", (mutate_target,))
    con.commit()
    return con


def test_no_target_week_outcome_leakage_in_served_features():
    with tempfile.TemporaryDirectory() as t:
        feats = []
        for target_rows in (None, 5, 400):                                                                          # the target-week row's outcome varies; served features must not
            con = synthetic_player_db(Path(t) / f"d{target_rows}.sqlite", target_rows)
            eng = SB.SeasonEngine(con, "rushing_yards", 2026)
            cand = eng.asof_future(5, [("A", "B")])
            assert len(cand) == 1
            feats.append(cand[0][5])
        assert feats[0] == feats[1] == feats[2] and feats[0]["games_played"] == 4                                   # identical features whatever the target-week outcome was
        con = synthetic_player_db(Path(t) / "x.sqlite", 400)
        good = {"picks": [pick(games_played=4, season=2026, week=5)]}
        assert RD.check_leakage(con, 2026, 5, good)["ok"]
        leaked = {"picks": [pick(games_played=5, season=2026, week=5)]}                                              # a board that counted the week-5 game would be caught
        r = RD.check_leakage(con, 2026, 5, leaked)
        assert not r["ok"] and r["mismatches"][0]["reason"] == "games_played_mismatch"
        assert not RD.check_leakage(con, 2026, 5, {"picks": [pick(games_played=4, season=2026, week=6)]})["ok"]


def test_calibration_uses_only_completed_prior_data():
    src = (REPO / "cfb_serving_builder_a.py").read_text()
    assert src.count("cur_seen = [row for row in cur if row[4] < target_week]") == 2 and src.count("cur_seen = [row for row in cur if row[2] < target_week]") == 1     # all three pools: current-season rows from weeks BEFORE the target week only
    assert src.count('"SELECT DISTINCT season FROM player_games WHERE season < ? ORDER BY season DESC"') == 2 and src.count('"SELECT DISTINCT season FROM games WHERE season < ? ORDER BY season DESC"') == 1   # warm-up = a completed earlier season
    prod = json.loads((REPO / "cfb_models" / "cfb_context_v2_work" / "production.json").read_text())
    assert all(prod[m]["calibration_season"] == 2025 for m in RD.ACTIVE_MARKETS)
    board = {"markets": {"rushing_yards": {"calibration_pool": {"warmup_season": 2025, "current_season_n": 10}}}}
    assert RD.check_calibration(2026, 5, board)["ok"]
    assert not RD.check_calibration(2025, 5, board)["ok"]                                                            # a calibration season >= the serving season would fail
    assert not RD.check_calibration(2026, 5, {"markets": {"rushing_yards": {"calibration_pool": {"warmup_season": 2026, "current_season_n": 1}}}})["ok"]


def test_artifact_hashes_and_feature_order():
    r = RD.check_artifacts()
    assert r["ok"] and r["manifest_present"] and all(r["files"].values()) and len(r["files"]) == 1 + 4 * 4
    assert all(v["context_columns_match_code"] and v["champion_columns_match_code"] for v in r["feature_order"].values())
    with tempfile.TemporaryDirectory() as t:
        good = json.loads(RD.MANIFEST.read_text())
        key = "context:rushing_yards"
        good["files"][key]["sha256"] = "0" * 64
        tmp = Path(t) / "m.json"; tmp.write_text(json.dumps(good))
        old = RD.MANIFEST; RD.MANIFEST = tmp
        try:
            r2 = RD.check_artifacts()
        finally:
            RD.MANIFEST = old
        assert not r2["ok"] and r2["files"][key] is False                                                            # a changed / unregistered hash fails readiness


def test_ledger_check_requires_every_served_pick_to_be_logged_pregame():
    with tempfile.TemporaryDirectory() as t:
        led = Path(t) / "l.jsonl"
        games = [{"home": "A", "away": "B", "kickoff_utc": ts(NOW + timedelta(hours=3)), "future": True}]
        p = pick(team="A", opp="B")
        led.write_text(json.dumps({**p, "logged_at": (NOW - timedelta(hours=1)).isoformat()}) + "\n")
        assert RD.check_ledger(led, {"picks": [dict(p, model_prob=0.7)]}, games)["ok"]                               # probabilities may differ: the ledger keeps the ORIGINAL
        assert RD.check_ledger(led, {"picks": [dict(p, model_prob=0.7)]}, games)["board_picks_whose_current_prob_differs_from_original_logged_prob"] == 1
        assert not RD.check_ledger(led, {"picks": [dict(p, pid="x", player_id="other")]}, games)["ok"]                # a served pick that was never logged
        led.write_text(json.dumps({**p, "logged_at": (NOW + timedelta(hours=4)).isoformat()}) + "\n")                  # logged after kickoff -> not pregame evidence
        assert not RD.check_ledger(led, {"picks": [p]}, games)["ok"]
        led.write_text((json.dumps({**p, "logged_at": NOW.isoformat()}) + "\n") * 2)
        assert not RD.check_ledger(led, {"picks": [p]}, games)["ok"]                                                  # duplicate ledger key


def test_no_sportsbook_odds_as_predictive_inputs():
    ok, odds = RD.check_suspended_and_odds({"picks": [pick()]})
    assert odds["ok"] and odds["odds_like_feature_columns"] == [] and odds["odds_references_in_serving_code"] == []
    bad = RD.check_suspended_and_odds({"picks": [pick(odds_price=-110)]})[1]
    assert not bad["ok"] and bad["odds_like_pick_fields"] == ["odds_price"]
    base, ctx = RD.expected_columns()
    for cols in list(base.values()) + list(ctx.values()):
        assert not any(w in c.lower() for c in cols for w in RD.ODDS_WORDS)


def test_freshness_is_exposed_and_stale_data_fails():
    with tempfile.TemporaryDirectory() as t:
        con = sqlite3.connect(Path(t) / "f.sqlite")
        con.execute("CREATE TABLE data_freshness (season INTEGER, key TEXT, value TEXT, PRIMARY KEY (season, key))")
        con.execute("CREATE TABLE player_games (season INTEGER, game_date TEXT)"); con.execute("INSERT INTO player_games VALUES (2026,'2026-10-02')")
        fresh = {"schedule_scan_completed_utc": (NOW - timedelta(hours=1)).isoformat(), "roster_snapshot_completed_utc": (NOW - timedelta(hours=1)).isoformat()}
        for k, v in fresh.items():
            con.execute("INSERT INTO data_freshness VALUES (2026,?,?)", (k, v))
        r = RD.check_freshness(con, 2026, {"generated_at_utc": NOW.isoformat()}, NOW)
        assert r["ok"] and r["schedule_age_hours"] == 1.0 and r["latest_2026_player_game_date"] == "2026-10-02" and r["current_player_data_refreshed_utc"]
        stale = RD.check_freshness(con, 2026, {}, NOW + timedelta(hours=RD.MAX_SCHEDULE_AGE_H + 1))
        assert not stale["ok"]
        con.execute("DELETE FROM data_freshness")
        assert not RD.check_freshness(con, 2026, {}, NOW)["ok"]                                                          # no freshness record -> not ready (never assumed fresh)


def test_board_generation_is_deterministic_for_the_same_snapshot_and_clock():
    db = REPO / "cfb_models" / "cfb_model.sqlite"
    if not db.exists():
        return
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    has = con.execute("SELECT name FROM sqlite_master WHERE name='schedule_snapshot'").fetchone()
    wk = con.execute("SELECT week FROM schedule_snapshot WHERE season=2026 AND espn_state='pre' ORDER BY kickoff_utc LIMIT 1").fetchone() if has else None
    ko = con.execute("SELECT MIN(kickoff_utc) FROM schedule_snapshot WHERE season=2026 AND week=? AND espn_state='pre'", (wk[0],)).fetchone()[0] if wk else None
    if not ko:
        return                                                                                                          # needs the refreshed live db (CI / local pipeline run); skipped otherwise
    now = (SB.parse_kickoff(ko) - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    outs = []
    with tempfile.TemporaryDirectory() as t:
        for i in (1, 2):
            out, led = Path(t) / f"b{i}.json", Path(t) / f"l{i}.jsonl"
            r = subprocess.run([sys.executable, str(REPO / "cfb_serving_builder_a.py"), "--db", str(db), "--season", "2026", "--week", str(wk[0]), "--out", str(out), "--ledger", str(led), "--now", now], cwd=REPO, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-400:]
            outs.append((json.loads(out.read_text()), led.read_text()))
        a, b = outs
        assert a[0]["picks"] == b[0]["picks"] and a[0]["generated_at_utc"] == b[0]["generated_at_utc"] == now.replace("Z", "+00:00") and a[1] == b[1]
        assert a[0]["picks"] and all(SB.parse_kickoff(p["kickoff_utc"]) > SB.parse_kickoff(now) for p in a[0]["picks"])
        assert not {p["market"].replace("_early_season", "") for p in a[0]["picks"]} & set(RD.SUSPENDED)


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
