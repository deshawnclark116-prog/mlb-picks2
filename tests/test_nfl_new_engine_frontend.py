"""Frontend/publisher tests: the NFL new-engine page is a read-only presentation of genuine nfl-shadow-state T24/T90 forecasts."""
import ast
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import nfl_phase1_publisher as P                         # noqa: E402
import nfl_phase1_store as ST                            # noqa: E402  (test fixtures only: builds a synthetic state tree)

REF = "2026-10-04T12:05:00.000000Z"
KICK = "2026-10-04T13:30:00.000000Z"


def rec(gid, hz, cutoff, player, outcome, **kw):
    q99 = [float(i) for i in range(-1, 98)]
    r = {"id": ST.make_id(gid, hz, cutoff, player, outcome), "game_id": gid, "horizon": hz, "cutoff": cutoff, "kickoff": KICK, "season": 2026, "week": 4, "player_id": "p-" + player, "player_name": player, "team": "IND",
         "opponent": "WAS", "position": "RB", "outcome": outcome, "p_active": 0.9, "expected_opportunities": 12.3, "mean": 55.0, "median": 52.0, "sd": 20.0, "p_zero": 0.05, "event_probability_ge1": None,
         "quantiles": {"p05": 20.0, "p10": 25.0, "p25": 38.0, "p50": 52.0, "p75": 70.0, "p90": 85.0, "p95": 95.0}, "quantile_grid_99": q99, "role_state": {"propensity_share": 0.4},
         "uncertainty": {"score": 0.3, "reasons": ["spread"]}, "model_version": "p1c-test", "simulation": {"n_draws": 100000}, "input_snapshots": {"time_travel": False}}
    r.update(kw)
    return r


def ledger_row(gid, hz, cutoff, state):
    return {"key": f"{gid}|{hz}|{cutoff}", "game_id": gid, "horizon": hz, "cutoff": cutoff, "kickoff": KICK, "state": state, "at": REF}


def make_state(tmp, spec):
    """spec: list of (gid, hz, cutoff, ledger_states(list), records_kwargs list)."""
    st = tmp / "state"; st.mkdir()
    (st / "status.json").write_text(json.dumps({"last_invocation_utc": REF, "last_successful_provider_retrieval_utc": "2026-10-04T12:00:00Z", "scheduler_version": "t", "readiness": {"READY": True}}))
    led = []
    store = ST.Store(st, "forecasts")
    for gid, hz, cutoff, states, recs, header in spec:
        for s in states:
            led.append(ledger_row(gid, hz, cutoff, s))
        if recs:
            store.append_batch(f"2026_wk04_{hz}_{gid}", header or {}, recs)
    (st / "dispatch_ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in led))
    return st


def tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


G1, G2, G3 = "2026_04_IND_WAS", "2026_04_ARI_NYG", "2026_04_PIT_CLE"
C90, C24 = "2026-10-04T12:00:00.000000Z", "2026-10-03T13:30:00.000000Z"


@pytest.fixture
def state(tmp_path):
    spec = [
        (G1, "T90", C90, ["PLANNED", "DONE"], [rec(G1, "T90", C90, "alpha", "rush_yds"), rec(G1, "T90", C90, "alpha", "atd", event_probability_ge1=0.4)], {}),
        (G1, "T24", C24, ["PLANNED", "DONE"], [rec(G1, "T24", C24, "alpha", "rush_yds", mean=50.0)], {}),
        (G2, "T90", C90, ["PLANNED", "MISSED_REAL_CUTOFF"], [rec(G2, "T90", C90, "ghost", "rush_yds")], {}),                    # a record exists but the ledger says MISSED: never shown
        (G2, "T24", C24, ["PLANNED", "FAILED"], [rec(G2, "T24", C24, "ghost2", "rush_yds")], {}),
        (G3, "T90", C90, ["PLANNED", "DONE"], [rec(G3, "T90", C90, "adhoc", "rush_yds", horizon="AD_HOC_PREGAME")], {"adhoc_label": "AD_HOC_PREGAME"}),   # ad hoc: never shown
        ("2026_04_TEN_BAL", "T90", C90, ["PLANNED", "DONE"], [rec("2026_04_TEN_BAL", "T90", C90, "tt", "rush_yds", snapshots=None, input_snapshots={"time_travel": True})], {}),   # time travel: never shown
    ]
    return make_state(tmp_path, spec)


def test_main_page_reads_new_file_not_legacy():
    s = (REPO / "docs/nfl.html").read_text()
    assert "nfl_phase1_shadow.json" in s and "nfl_predictions.json" not in s
    for lab in ("NEW NFL OUTCOME ENGINE", "SHADOW / RESEARCH", "NO SPORTSBOOK INPUTS", "NOT YET PROMOTED TO PRODUCTION", "R11"):
        assert lab in s
    assert P.build.__module__ == "nfl_phase1_publisher"


def test_legacy_page_still_reads_legacy_and_nav():
    s = (REPO / "docs/nfl_legacy.html").read_text()
    assert 'fetch("nfl_predictions.json"' in s
    for page in ("nfl.html", "nfl_legacy.html", "nfl_phase1_experimental.html"):
        assert f'href="{page}"' in s
    base = subprocess.run(["git", "show", "ce90afb143c2937101768a3cf0b4a6d5ecb869e4^:docs/nfl.html"], cwd=REPO, capture_output=True, text=True)
    if base.returncode == 0 and base.stdout:
        # identical to the previous main page except nav/title lines
        a, b = base.stdout.splitlines(), s.splitlines()
        diff = [x for x in b if x not in a] + [x for x in a if x not in b]
        assert all(("nav" in x.lower() or "<a href" in x or "<title>" in x) for x in diff), diff


def test_legacy_predictions_untouched():
    out = subprocess.run(["git", "diff", "--name-only", "origin/main", "--", "docs/nfl_predictions.json", "docs/nfl_predictions_2026_w01.json", "docs/nfl_predictions_2026_w02.json", "docs/nfl_predictions_2026_w03.json",
                          "docs/nfl_predictions_2026_w04.json", "docs/nfl_record.json", "docs/nfl_picks_log.jsonl", "nfl_serving_builder_a.py"], cwd=REPO, capture_output=True, text=True)
    if out.returncode == 0:
        assert out.stdout.strip() == ""


def test_publisher_imports_nothing_that_can_write_forecasts():
    tree = ast.parse((REPO / "nfl_phase1_publisher.py").read_text())
    mods = {n.module if isinstance(n, ast.ImportFrom) else a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in (n.names if isinstance(n, ast.Import) else [None])}
    assert not {m for m in mods if m and (m.startswith("nfl_phase1") or "scheduler" in m or "store" in m)}
    src = (REPO / "nfl_phase1_publisher.py").read_text()
    assert not re.search(r"open\([^)]*[\"'][wax]", src) and "append_batch" not in src and "shutil" not in src
    writes = re.findall(r"\.write_text|\.write_bytes", src)
    assert len(writes) == 2                                                  # shadow json (publish) and adhoc json (publish_adhoc) only


def test_publisher_does_not_touch_state_and_cannot_create_forecasts(state, tmp_path):
    before = tree_hash(state)
    out = tmp_path / "o.json"
    P.publish(state, out)
    assert tree_hash(state) == before
    for p in state.rglob("*"):                                               # read-only state tree still publishes
        os.chmod(p, 0o555 if p.is_dir() else 0o444)
    try:
        P.publish(state, tmp_path / "o2.json")
    finally:
        for p in state.rglob("*"):
            os.chmod(p, 0o755 if p.is_dir() else 0o644)
    assert (tmp_path / "o2.json").read_text() == out.read_text()


def test_only_genuine_horizons_rows_and_no_missed_failed_adhoc_timetravel(state):
    d = P.build(state)
    names = {r[1] for h in P.HORIZONS for r in d["forecasts"][h]}
    assert names == {"alpha"}
    assert d["status"]["counts"]["T90"]["missed"] == 1 and d["status"]["counts"]["T24"]["failed"] == 1
    assert d["status"]["counts"]["T90"]["valid_games"] == 1 and d["status"]["counts"]["T24"]["valid_games"] == 1
    text = json.dumps(d)
    for bad in ("ghost", "adhoc", "AD_HOC", "tt"):
        assert f'"{bad}"' not in text


def test_t24_t90_distinct_and_default_t90(state):
    d = P.build(state)
    assert d["default_horizon"] == "T90"
    t24, t90 = {r[-1] for r in d["forecasts"]["T24"]}, {r[-1] for r in d["forecasts"]["T90"]}
    assert t24 and t90 and not (t24 & t90)
    assert len(t90) == 2 and len(t24) == 1
    f = d["row_fields"]
    assert [r[f.index("mean")] for r in d["forecasts"]["T24"]] == [50.0]


def test_default_falls_back_to_t24_and_empty_when_nothing(tmp_path):
    st = make_state(tmp_path, [(G1, "T24", C24, ["PLANNED", "DONE"], [rec(G1, "T24", C24, "alpha", "rush_yds")], {}), (G1, "T90", C90, ["PLANNED"], [], {})])
    assert P.build(st)["default_horizon"] == "T24"
    t2 = tmp_path / "e"; t2.mkdir()
    st2 = make_state(t2, [(G1, "T90", C90, ["PLANNED"], [], {})])
    d = P.build(st2)
    assert d["default_horizon"] is None and d["forecasts"] == {"T24": [], "T90": []} and "No valid New Engine forecast" in d["empty_message"]


def test_no_sportsbook_fields_required_and_adapter_is_downstream(state, tmp_path):
    d = P.build(state, odds_path=None)
    assert d["lines"] == [] and all("book" not in k and "line" not in k for k in d["row_fields"])
    odds = tmp_path / "odds.json"
    odds.write_text(json.dumps({"date_et": "2026-10-04", "generated_at_utc": "x", "lines": [{"line": 49.5, "book": "dk", "over_price": -110, "under_price": -110, "player_norm": "alpha", "market": "rushing_yards"}]}))
    d2 = P.build(state, odds_path=odds)
    assert len(d2["lines"]) == 2 and {l["horizon"] for l in d2["lines"]} == {"T24", "T90"}
    assert 0.0 < d2["lines"][0]["p_over"] < 1.0 and abs(d2["lines"][0]["p_over"] + d2["lines"][0]["p_under"] - 1) < 1e-6
    assert d2["forecasts"] == d["forecasts"]                                  # forecast rows byte-identical with or without lines
    assert P.build(state, odds_path=tmp_path / "missing.json")["forecasts"] == d["forecasts"]


def test_deterministic_and_unique_ids(state, tmp_path):
    a, b = P.dumps(P.build(state)), P.dumps(P.build(state))
    assert a == b
    ids = [r[-1] for h in P.HORIZONS for r in json.loads(a)["forecasts"][h]]
    assert len(ids) == len(set(ids))


def test_corrupt_batch_is_ignored(state):
    f = next((state / "forecasts" / "batches").glob("*T90*IND_WAS*.jsonl"))
    os.chmod(f, 0o644); f.write_bytes(f.read_bytes().replace(b"alpha", b"alphb"))
    d = P.build(state)
    assert d["forecasts"]["T90"] == []


def test_heartbeat_only_change_does_not_rewrite(state, tmp_path):
    out = tmp_path / "o.json"
    assert P.publish(state, out) == "written"
    assert P.publish(state, out) == "unchanged"


def test_scientific_branch_not_modified_by_the_frontend_pr():
    """Scope guard pinned to the squash commit of PR #56 (the frontend switch): no scientific file was changed by it."""
    r = subprocess.run(["git", "diff", "--name-only", "ce90afb143c2937101768a3cf0b4a6d5ecb869e4^", "ce90afb143c2937101768a3cf0b4a6d5ecb869e4"], cwd=REPO, capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        changed = set(r.stdout.split())
        forbidden = [c for c in changed if c.startswith("nfl_models/") or (re.match(r"nfl_phase1[a-z_]*\.py$", c) and c != "nfl_phase1_publisher.py") or c == ".github/workflows/nfl_phase1e_shadow.yml"]
        assert forbidden == []


def test_workflow_is_read_only_wrt_state_branch():
    s = (REPO / ".github/workflows/nfl_new_engine_publish.yml").read_text()
    assert "workflow_run" in s and "NFL Phase 1E shadow" in s
    assert "persist-credentials: false" in s                                  # the state checkout carries no push credential
    assert not re.search(r"push[^\n]*nfl-shadow-state|HEAD:nfl-shadow-state", s)
    assert "git add docs/nfl_phase1_shadow.json" in s and "git add -A" not in s
    assert "docs/nfl_phase1_shadow.json" in s


def test_main_page_javascript_parses():
    """Parse the exact committed inline script (regression: a malformed string concatenation shipped in the first version)."""
    import shutil
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    for page in ("nfl.html", "nfl_legacy.html"):
        s = (REPO / "docs" / page).read_text()
        js = re.findall(r"<script>(.*?)</script>", s, re.S)
        assert js
        for i, block in enumerate(js):
            f = Path(os.environ.get("TMPDIR", "/tmp")) / f"_parse_{page}_{i}.js"
            f.write_text(block)
            r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr


def test_adhoc_section_is_separate_and_fails_closed(tmp_path):
    ev = REPO / "nfl_models/nfl_player_outcome_phase1e/adhoc_ind_was_20261004"
    a = P.build_adhoc(ev)
    assert a and a["horizon"] == "AD_HOC_PREGAME" and a["not_t24_t90"] and a["minutes_before_kickoff_at_retrieval"] > 0
    assert json.loads((REPO / "docs/nfl_phase1_adhoc.json").read_text())["horizon"] == "AD_HOC_PREGAME"
    shadow = json.loads((REPO / "docs/nfl_phase1_shadow.json").read_text())
    assert all(r[-1] not in {x[-1] for x in a["rows"]} for h in P.HORIZONS for r in shadow["forecasts"][h])      # never mixed into T24/T90 rows
    lag = json.loads((ev / "provider_lag.json").read_text()); lag["retrieval_ts"] = "2026-10-04T13:45:00.000000Z"
    import shutil
    d = tmp_path / "ev"; shutil.copytree(ev, d); (d / "provider_lag.json").write_text(json.dumps(lag))
    assert P.build_adhoc(d) is None                                                                     # retrieval after kickoff: fail closed
