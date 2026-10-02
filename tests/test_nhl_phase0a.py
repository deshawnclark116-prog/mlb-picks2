"""Focused checks of the NHL Phase 0A artifact. python tests/test_nhl_phase0a.py"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
A = json.loads((REPO / "nhl_models/nhl_outcome_engine/phase0a_data_feasibility.json").read_text())


def test_sample_spans_two_seasons_and_is_deterministic():
    ids = A["sample_game_reconstruction"]["sample_game_ids"]
    assert len({str(i)[:4] for i in ids}) >= 2 and len(ids) == 8
    import hashlib, sqlite3
    con = sqlite3.connect(REPO / "nhl_models/nhl_model.sqlite")
    for s in (2018, 2021, 2023, 2025):
        g = [r[0] for r in con.execute("select game_id from games where season=? and game_state in ('OFF','FINAL')", (s,))]
        g.sort(key=lambda x: hashlib.sha256(f"nhl-phase0a-{x}".encode()).hexdigest())
        assert g[:2] == [i for i in ids if str(i).startswith(str(s))]


def test_reconstruction_evidence_is_consistent():
    g = A["sample_game_reconstruction"]
    ag = g["aggregate"]
    assert ag["sog_mismatches_boxscore_vs_pbp"] == 0 and ag["sog_mismatches_boxscore_vs_stats_rest"] == 0 and ag["toi_mismatch_stats_rest_vs_boxscore"] == 0 and ag["shift_toi_mismatch_after_dedup"] == 0
    for x in g["games"]:
        assert x["identity"]["startTimeUTC"].endswith("Z") and x["provenance"]["boxscore"]["sha256"] and x["provenance"]["boxscore"]["retrieved_at_utc"]
        d, home, away, hs, aws = x["local_games_row[date,home,away,hs,as]"]
        assert d == x["identity"]["gameDate"] and home == x["identity"]["home"] and away == x["identity"]["away"]


def test_status_table_complete_and_binary():
    st = A["status_by_data_category"]
    assert all(c["status"] in ("PASS", "BLOCKER") and c["evidence"] for c in st)
    need = {"as-of pregame roster / scratches / injuries", "pregame confirmed starting goalie", "existing repo table as outcome source of truth"}
    assert need <= {c["category"] for c in st if c["status"] == "BLOCKER"}
    assert next(c for c in st if c["category"] == "sportsbook independence")["status"] == "PASS"


def test_no_nhl_production_file_modified():
    import subprocess
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    bad = [l for l in out.splitlines() if ("docs/nhl" in l or "nhl_models/nhl_model.sqlite" in l or l.split()[-1].startswith(("nhl_serving", "nhl_live", "nhl_player_games")))]
    assert not bad, bad


if __name__ == "__main__":
    f = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); f += 1; print("FAIL", n, repr(e))
    sys.exit(1 if f else 0)
