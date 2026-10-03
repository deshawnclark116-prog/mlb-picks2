"""CFB forward-evaluation grading tests: game-level moneyline grading, first-pregame-entry canonicity, canonical keys, probability scoring, forward status.
Synthetic data only (no network).   python tests/test_cfb_grading.py"""
import json
import math
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import cfb_grade_record_a as G  # noqa: E402


def game(gid, week, home, away, hp=None, ap=None, ko="2026-10-03T16:00Z", season=2026):
    return {"game_id": gid, "season": season, "week": week, "kickoff_utc": ko, "home_team": home, "away_team": away, "home_points": hp, "away_points": ap}


def ml(team, opp, week=5, prob=0.7, logged="2026-10-03T10:00:00+00:00", market="moneyline", **kw):
    return {"market": market, "player_id": None, "player": team, "team": team, "opponent": opp, "season": 2026, "week": week, "pick": f"{team} ML", "model_prob": prob, "logged_at": logged, **kw}


def prop(pid="p1", team="A", opp="B", week=5, pick="OVER 69.5", line=69.5, prob=0.6, logged="2026-10-03T10:00:00+00:00", market="rushing_yards", **kw):
    return {"market": market, "player_id": pid, "player": "Pat", "team": team, "opponent": opp, "season": 2026, "week": week, "line": line, "pick": pick, "model_prob": prob, "logged_at": logged, **kw}


def run(ledger, games, player_rows=None, snap=None):
    return G.evaluate(ledger, games, player_rows or {}, snap or {})


def only(results, i=0):
    return results[i]


# ------------------------------------------------------------------ moneyline truth
def test_moneyline_home_winner_away_winner_and_predicted_loser():
    games = [game("1", 5, "H1", "A1", 31, 10), game("2", 5, "H2", "A2", 14, 28), game("3", 5, "H3", "A3", 7, 21)]
    ledger = [ml("H1", "A1"), ml("A2", "H2"), ml("H3", "A3")]
    res, ung, exc, audit = run(ledger, games)
    by = {r["team"]: r for r in res}
    assert by["H1"]["result"] == "hit" and by["H1"]["winner" if False else "actual"] == "H1"                          # predicted home winner
    assert by["A2"]["result"] == "hit" and by["A2"]["actual"] == "A2"                                                  # predicted away winner
    assert by["H3"]["result"] == "miss" and by["H3"]["actual"] == "A3"                                                 # predicted loser
    assert ung == {} and exc == {} and audit["canonical_predictions"] == 3 and all(r["evaluation_probability_source"] == "FIRST_PREGAME_LEDGER_ENTRY" for r in res)


def test_unfinished_cancelled_tie_and_invalid_are_never_wins_or_losses():
    games = [game("1", 5, "H1", "A1"), game("2", 5, "H2", "A2"), game("3", 5, "H3", "A3", 17, 17), game("4", 5, "H4", "A4"), game("5", 5, "H5", "A5", 3, 0)]
    ledger = [ml("H1", "A1"), ml("H2", "A2"), ml("H3", "A3"), ml("H4", "A4"), ml("ZZZ", "A5", game_id="5")]
    snap = {"2": "STATUS_CANCELED", "4": "STATUS_POSTPONED"}
    res, ung, exc, _ = run(ledger, games, snap=snap)
    assert res == []                                                                                                    # nothing was silently called a win or a loss
    assert ung == {"unfinished": 1} and exc == {"cancelled_excluded": 2, "tie_excluded": 1, "invalid": 1}, (ung, exc)
    # a game that is IN PROGRESS (no final points, no cancel status) stays ungraded and becomes gradable only after the final score arrives
    res2, ung2, _, _ = run([ml("H1", "A1")], [game("1", 5, "H1", "A1", 24, 20)])
    assert res2[0]["result"] == "hit" and ung2 == {}
    assert G.grade_moneyline(ml("H5", "A5"), game("5", 5, "H5", "A5", 3, 0), "STATUS_FORFEIT")[0] == "cancelled_excluded"


def test_duplicate_team_names_across_weeks_cannot_cross_match():
    games = [game("10", 3, "H", "A", 10, 40, ko="2026-09-19T16:00Z"), game("20", 9, "H", "A", 35, 3, ko="2026-11-07T17:00Z")]              # the same pair meets twice
    ledger = [ml("H", "A", week=3, logged="2026-09-19T10:00:00+00:00"), ml("H", "A", week=9, logged="2026-11-07T10:00:00+00:00")]
    res, ung, exc, audit = run(ledger, games)
    by = {r["week"]: r for r in res}
    assert by[3]["result"] == "miss" and by[3]["game_id"] == "10" and by[9]["result"] == "hit" and by[9]["game_id"] == "20" and audit["duplicate_key_count"] == 0
    assert by[3]["canonical_key"] != by[9]["canonical_key"]
    # the same pair twice in ONE week number is ambiguous without a game_id -> ungraded + reported; with a game_id it resolves exactly
    games2 = [game("1", 5, "H", "A", 10, 0, ko="2026-10-03T16:00Z"), game("2", 5, "A", "H", 0, 10, ko="2026-10-03T20:00Z")]
    r1, u1, _, a1 = run([ml("H", "A")], games2)
    assert r1 == [] and u1 == {} and a1["unresolved_or_ambiguous_game_rows"] == 1 and a1["canonical_predictions"] == 0      # reported in the audit, never graded to either game
    r2, _, _, _ = run([ml("H", "A", game_id="1")], games2)
    assert r2[0]["game_id"] == "1" and r2[0]["result"] == "hit"


def test_early_season_moneyline_grades_identically_to_normal_moneyline():
    games = [game("1", 3, "H", "A", 28, 14, ko="2026-09-19T16:00Z"), game("2", 3, "H2", "A2", 3, 30, ko="2026-09-19T16:00Z")]
    normal = [ml("H", "A", week=3, logged="2026-09-19T10:00:00+00:00"), ml("H2", "A2", week=3, logged="2026-09-19T10:00:00+00:00")]
    early = [ml("H", "A", week=3, logged="2026-09-19T10:00:00+00:00", market="moneyline_early_season"), ml("H2", "A2", week=3, logged="2026-09-19T10:00:00+00:00", market="moneyline_early_season")]
    rn, _, _, _ = run(normal, games); re_, _, _, _ = run(early, games)
    assert [(r["team"], r["result"], r["actual"]) for r in rn] == [(r["team"], r["result"], r["actual"]) for r in re_]
    assert {r["market"] for r in re_} == {"moneyline_early_season"} and rn[0]["canonical_key"] != re_[0]["canonical_key"]   # distinct models stay distinct predictions
    assert re_[0]["generation"] == "early_season_prior_season_informed"


# ------------------------------------------------------------------ first-prediction canonicity
def test_first_pregame_entry_is_canonical_and_later_rows_never_used():
    games = [game("1", 5, "H", "A", 10, 20, ko="2026-10-03T16:00Z")]
    first = ml("H", "A", prob=0.80, logged="2026-10-02T10:00:00+00:00", model_source="context_v2")
    later = ml("H", "A", prob=0.55, logged="2026-10-03T09:00:00+00:00", model_source="champion_growing_platt")         # same key, refreshed probability
    flipped = ml("A", "H", prob=0.60, logged="2026-10-03T12:00:00+00:00")                                              # a refresh that flipped the predicted team
    for order in ([first, later, flipped], [flipped, later, first]):                                                   # file order must not matter, only logged_at
        res, _, _, audit = run(order, games)
        assert len(res) == 1 and res[0]["model_prob"] == 0.80 == res[0]["original_model_prob"] and res[0]["team"] == "H" and res[0]["result"] == "miss"
        assert res[0]["logged_at"] == "2026-10-02T10:00:00+00:00" and res[0]["model_source"] == "context_v2" and res[0]["evaluation_probability_source"] == "FIRST_PREGAME_LEDGER_ENTRY"
        assert audit["duplicate_key_count"] == 1 and audit["duplicate_extra_rows"] == 2 and audit["keys_with_conflicting_side_or_team"] == 1 and audit["canonical_predictions"] == 1
    # no averaging: the brier of the canonical row uses 0.80 (a miss), not the mean / the latest
    agg = G.score_group(res)
    assert abs(agg["mean_brier"] - 0.64) < 1e-9


def test_post_kickoff_and_unverifiable_rows_are_not_forward_evidence():
    games = [game("1", 5, "H", "A", 10, 20, ko="2026-10-03T16:00Z"), game("2", 5, "H2", "A2", 10, 20, ko=None)]
    late = ml("H", "A", logged="2026-10-03T16:00:00+00:00")                                                            # logged exactly at kickoff
    after = ml("H", "A", logged="2026-10-03T18:00:00+00:00", prob=0.99)
    nok = ml("H2", "A2", logged="2026-10-01T00:00:00+00:00")                                                           # kickoff unverifiable
    res, _, _, audit = run([late, after, nok], games)
    assert res == [] and audit["canonical_predictions"] == 0 and audit["rows_not_pregame_or_unverifiable"] == 3 and audit["keys_with_no_valid_pregame_row"] == 2
    ok = ml("H", "A", logged="2026-10-03T15:59:59+00:00", prob=0.6)                                                    # a genuine earlier pregame row wins even with later junk
    res, _, _, audit = run([after, ok, late], games)
    assert len(res) == 1 and res[0]["model_prob"] == 0.6 and audit["canonical_predictions"] == 1
    assert run([dict(ok, kickoff_utc="2026-10-03T15:00Z")], games)[0] == []                                           # the row's OWN kickoff wins over the schedule's


def test_canonical_key_is_stable_and_prevents_collisions_and_double_grading():
    g1, g2 = game("1", 5, "H", "A"), game("2", 5, "C", "D")
    k = G.canonical_key(prop(pid="p1"), g1)
    assert k == "2026|5|1|rushing_yards|p1|69.5"
    assert G.canonical_key(prop(pid="p1", prob=0.9, pick="UNDER 69.5"), g1) == k                                       # probability / side drift cannot create a new key
    assert G.canonical_key(prop(pid="p1"), g2) != k and G.canonical_key(prop(pid="p2"), g1) != k and G.canonical_key(prop(pid="p1", market="rushing_yards_early_season"), g1) != k
    assert G.canonical_key(ml("H", "A"), g1) == "2026|5|1|moneyline|GAME|ML" == G.canonical_key(ml("A", "H", prob=0.2), g1)  # one moneyline prediction per game, whichever team
    assert G.canonical_key(prop(), None).startswith("2026|5|UNRESOLVED:")


# ------------------------------------------------------------------ props still graded, now by game identity
def test_prop_grading_uses_game_identity_and_original_side():
    games = [game("1", 5, "A", "B", 30, 10), game("2", 6, "A", "C", 30, 10, ko="2026-10-10T16:00Z")]
    rows = {("p1", "1"): {"rushing_yards": 100, "player_id": "p1", "game_id": "1"}, ("p1", "2"): {"rushing_yards": 10, "player_id": "p1", "game_id": "2"}}
    res, ung, _, _ = run([prop(pid="p1", opp="B", week=5), prop(pid="p1", opp="C", week=6, pick="UNDER 69.5", logged="2026-10-09T10:00:00+00:00")], games, rows)
    by = {r["week"]: r for r in res}
    assert by[5]["actual"] == 100 and by[5]["result"] == "hit" and by[6]["actual"] == 10 and by[6]["result"] == "hit"      # each week graded only against ITS game
    res, ung, _, _ = run([prop(pid="p9", opp="C", week=6, logged="2026-10-09T10:00:00+00:00")], [game("2", 6, "A", "C", ko="2026-10-10T16:00Z")], rows)
    assert res == [] and ung == {"unfinished": 1}
    games_f = [game("1", 5, "A", "B", 30, 10)]
    res, ung, _, _ = run([prop(pid="p9")], games_f, rows)
    assert res == [] and ung == {"player_no_stat_line_after_final": 1}                                                 # reported, never counted as a loss


# ------------------------------------------------------------------ probability scoring + generations
def test_probability_scoring_uses_original_probability_and_reports_generations_separately():
    res = [{"result": "hit", "model_prob": 0.8}, {"result": "miss", "model_prob": 0.6}, {"result": "hit", "model_prob": 0.5}]
    s = G.score_group(res)
    assert s["n_graded"] == 3 and s["wins"] == 2 and s["losses"] == 1 and abs(s["hit_rate"] - 0.6667) < 1e-4
    assert abs(s["mean_brier"] - ((0.2 ** 2 + 0.6 ** 2 + 0.5 ** 2) / 3)) < 1e-5
    assert abs(s["mean_log_loss"] - ((-math.log(0.8) - math.log(0.4) - math.log(0.5)) / 3)) < 1e-5
    assert abs(s["mean_predicted_probability"] - 0.6333) < 1e-4 and abs(s["observed_success_rate"] - 0.6667) < 1e-4
    assert G.score_group([]) == {"n_graded": 0}
    # generation labels: explicit model_source wins; legacy rows are inferred from the log time; early-season is its own generation
    assert G.generation({"market": "rushing_yards", "model_source": "context_v2"}) == "in_season_context_v2"
    assert G.generation({"market": "rushing_yards", "model_source": "champion_growing_platt"}) == "in_season_champion_growing_platt"
    assert G.generation({"market": "rushing_yards", "logged_at": "2026-09-27T16:05:38+00:00"}) == "in_season_context_v2_inferred_by_log_time"
    assert G.generation({"market": "rushing_yards", "logged_at": "2026-09-27T16:05:37+00:00"}) == "in_season_champion_growing_platt_inferred_by_log_time"
    assert G.generation({"market": "moneyline_early_season", "logged_at": "2026-10-01T00:00:00+00:00"}) == "early_season_prior_season_informed"
    games = [game("1", 5, "H", "A", 10, 20), game("2", 3, "H2", "A2", 20, 10, ko="2026-09-19T16:00Z")]
    r, _, _, _ = run([ml("H", "A", model_source="context_v2"), ml("H2", "A2", week=3, market="moneyline_early_season", model_source="prior_season_informed", logged="2026-09-19T10:00:00+00:00")], games)
    _, by_gen, by_gen_mkt = G.aggregate(r)
    assert set(by_gen) == {"in_season_context_v2", "early_season_prior_season_informed"} and "in_season_context_v2::moneyline" in by_gen_mkt


# ------------------------------------------------------------------ end to end: files, ledger untouched, forward status
def test_end_to_end_never_rewrites_ledger_and_writes_forward_status():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        db = sqlite3.connect(t / "d.sqlite")
        db.execute("CREATE TABLE games (game_id TEXT, season INTEGER, week INTEGER, game_date TEXT, kickoff_utc TEXT, home_team TEXT, away_team TEXT, home_points INTEGER, away_points INTEGER)")
        db.execute("CREATE TABLE player_games (player_id TEXT, season INTEGER, week INTEGER, game_id TEXT, carries INTEGER, rushing_yards INTEGER, receptions INTEGER, receiving_yards INTEGER, pass_attempts INTEGER, completions INTEGER, passing_yards INTEGER, passing_touchdowns INTEGER, passing_interceptions INTEGER, rushing_touchdowns INTEGER, receiving_touchdowns INTEGER)")
        db.execute("CREATE TABLE schedule_snapshot (game_id TEXT, season INTEGER, week INTEGER, kickoff_utc TEXT, espn_state TEXT, espn_status TEXT, home_team TEXT, away_team TEXT, retrieved_at_utc TEXT)")
        db.execute("INSERT INTO games VALUES ('1',2026,5,'2026-10-03','2026-10-03T16:00Z','H','A',31,10)")
        db.execute("INSERT INTO games VALUES ('2',2026,5,'2026-10-03','2026-10-03T16:00Z','H2','A2',NULL,NULL)")
        db.execute("INSERT INTO games VALUES ('3',2026,5,'2026-10-03','2026-10-03T16:00Z','H3','A3',NULL,NULL)")
        db.execute("INSERT INTO schedule_snapshot VALUES ('3',2026,5,'2026-10-03T16:00Z','post','STATUS_CANCELED','H3','A3','x')")
        db.execute("INSERT INTO player_games VALUES ('p1',2026,5,'1',20,120,0,0,0,0,0,0,0,0,0)")
        db.commit()
        rows = [ml("H", "A", prob=0.75), ml("H2", "A2"), ml("H3", "A3"), prop(pid="p1", team="H", opp="A"), prop(pid="p1", team="H", opp="A", prob=0.99, logged="2026-10-03T11:00:00+00:00")]
        led = t / "ledger.jsonl"; led.write_text("".join(json.dumps(r) + "\n" for r in rows))
        before = led.read_bytes()
        (t / "readiness.json").write_text(json.dumps({"READY": True, "checked_at_utc": "2026-10-03T12:00:00Z", "season": 2026, "week": 5, "failing_checks": [], "future_games": 3}))
        (t / "board.json").write_text(json.dumps({"generated_at_utc": "2026-10-03T12:00:00+00:00", "picks": []}))
        r = subprocess.run([sys.executable, str(REPO / "cfb_grade_record_a.py"), "--db", str(t / "d.sqlite"), "--log", str(led), "--out", str(t / "rec.json"), "--status-out", str(t / "status.json"),
                            "--readiness", str(t / "readiness.json"), "--board", str(t / "board.json")], cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-300:]
        assert led.read_bytes() == before                                                                              # the ledger is never rewritten
        rec, st = json.loads((t / "rec.json").read_text()), json.loads((t / "status.json").read_text())
        assert rec["summary"] == {"total": 2, "hits": 2, "misses": 0, "hit_rate": 100.0} and set(rec["by_market"]) == {"moneyline", "rushing_yards"}
        assert rec["forward_evaluation"]["excluded"] == {"cancelled_excluded": 1} and rec["forward_evaluation"]["ungraded"] == {"unfinished": 1}
        ml_row = [x for x in rec["results"] if x["market"] == "moneyline"][0]
        assert ml_row["original_model_prob"] == 0.75 and ml_row["evaluation_probability_source"] == "FIRST_PREGAME_LEDGER_ENTRY"
        pr_row = [x for x in rec["results"] if x["market"] == "rushing_yards"][0]
        assert pr_row["model_prob"] == 0.6 and pr_row["logged_at"] == "2026-10-03T10:00:00+00:00"                       # first row (0.6), not the later 0.99
        assert st["first_prediction_only"] is True and st["ledger_entries_total"] == 5 and st["canonical_predictions"] == 4 and st["graded_predictions"] == 2 and st["ungraded_predictions"] == 1
        assert st["duplicate_key_count"] == 1 and st["excluded_predictions"] == {"cancelled_excluded": 1} and st["current_week_readiness"]["READY"] is True and st["current_board_generated_at_utc"] == "2026-10-03T12:00:00+00:00"
        assert "NO CFB predictive model" in st["pre_slate_freeze"] and "observational" in st["note"] and st["decision"].startswith("NONE")
        assert set(st["markets"]) == {"moneyline", "rushing_yards"} and st["markets"]["moneyline"]["mean_brier"] is not None


def test_freeze_rule_is_documented_and_no_predictive_file_is_touched_by_the_grader():
    src = (REPO / "cfb_grade_record_a.py").read_text()
    assert "PRE-SLATE FREEZE" in src and "cannot trigger same-week rescue tuning" in src
    for forbidden in ("xgboost", "fit_platt", "Booster", "odds_api", "ODDS_API"):
        assert forbidden not in src, forbidden
    doc = REPO / "cfb_models" / "CFB_FORWARD_EVALUATION_FREEZE.md"
    assert doc.exists() and "NO CFB predictive model" in doc.read_text()


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
