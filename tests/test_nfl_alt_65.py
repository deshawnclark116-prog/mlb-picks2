"""Distribution-only alt milestone, frontend integration, and strict presentation scope."""
import copy
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import nfl_phase1_publisher as P  # noqa: E402

# PR comparison; becomes HEAD once merged, so later main updates remain valid.
BASE = subprocess.check_output(["git", "merge-base", "HEAD", "origin/main"], cwd=REPO, text=True).strip()
EVIDENCE = REPO / "nfl_models/nfl_player_outcome_phase1e/adhoc_ind_was_20261004"


@pytest.mark.parametrize("record, milestone, probability, next_probability", [
    ({"outcome": "rush_yds", "quantile_grid_99": list(range(1, 100))}, 30, .70, .60),
    ({"outcome": "rec_yds", "quantile_grid_99": list(range(1, 100))}, 30, .70, .60),
    ({"outcome": "pass_yds", "quantile_grid_99": list(range(10, 1000, 10))}, 350, .65, .62),
    ({"outcome": "rec", "cdf_lattice": {"step": 1, "cdf": [.1, .2, .35, .6, .8, 1]}}, 3, .65, .40),
    ({"outcome": "sacks", "cdf_lattice": {"step": .5, "cdf": [.1, .15, .2, .25, .3, .35, .5, .6, .8, 1]}}, 3, .65, .40),
])
def test_highest_milestone_threshold_and_next(record, milestone, probability, next_probability):
    a = P.alt_65(record)
    assert a["milestone"] == milestone
    assert a["probability"] == pytest.approx(probability)
    assert a["probability"] >= .65
    step = 25 if record["outcome"] == "pass_yds" else 10 if "yds" in record["outcome"] else 1
    nxt = milestone + step
    if "yds" in record["outcome"]:
        # Independent first matching (level, quantile) enumeration.
        pnext = next((1 - i / 100 for i, q in enumerate(record["quantile_grid_99"], 1) if q >= nxt), 0)
    else:
        lat = record["cdf_lattice"]
        pnext = 1 - lat["cdf"][int(nxt / lat["step"]) - 1]
    assert pnext == pytest.approx(next_probability) and pnext < .65


def test_counts_use_inclusive_empirical_cdf_and_ignore_quantiles():
    r = {"outcome": "rec", "cdf_lattice": {"step": 1, "cdf": [.1, .2, .35, .9, 1]},
         "quantile_grid_99": [999] * 99, "event_probability_ge1": .01}
    assert P.alt_65(r)["milestone"] == 3  # 1-CDF(2), not 1-CDF(3)
    r["cdf_lattice"]["cdf"][2] = .350001
    assert P.alt_65(r)["milestone"] == 2  # Never round into eligibility.


def test_quantile_ties_keep_mass_at_milestone():
    r = {"outcome": "rush_yds", "quantile_grid_99": [0] * 29 + [30] * 50 + [100] * 20}
    a = P.alt_65(r)
    assert a["milestone"] == 30 and a["probability"] == .70


def test_distribution_not_mean_or_median_and_deterministic_no_mutation():
    r = {"outcome": "rush_yds", "mean": 74, "median": 74, "p_active": .01,
         "quantile_grid_99": list(range(1, 100))}
    before = copy.deepcopy(r)
    a = P.alt_65(r)
    assert r == before
    assert all(P.alt_65(r) == a for _ in range(10))
    r.update(mean=-1000, median=99999, p_active=1)
    assert P.alt_65(r) == a
    r["quantile_grid_99"] = [x * 2 for x in r["quantile_grid_99"]]
    assert P.alt_65(r)["milestone"] == 70


@pytest.mark.parametrize("r", [
    {"outcome": "rush_yds", "mean": 999, "median": 999},
    {"outcome": "rush_yds", "quantile_grid_99": [5] * 99},
    {"outcome": "pass_yds", "quantile_grid_99": [99] * 99},
    {"outcome": "rec", "cdf_lattice": {"step": 1, "cdf": [.36, 1]}},
    {"outcome": "rec", "quantile_grid_99": [10] * 99},
    {"outcome": "rec", "cdf_lattice": {"step": 1, "cdf": [.1, .2]}},  # truncated tail
    {"outcome": "rec", "cdf_lattice": {"step": .5, "cdf": [.1, 1]}},
    {"outcome": "rec", "cdf_lattice": {"step": 1, "cdf": [.5, .4, 1]}},
    {"outcome": "rec", "cdf_lattice": {"step": 1, "cdf": [-.1, 1]}},
    {"outcome": "rec", "cdf_lattice": [0, 1]},
    {"outcome": "rush_yds", "quantile_grid_99": [1] * 98},
    {"outcome": "rush_yds", "quantile_grid_99": [float("nan")] * 99},
    {"outcome": "rush_yds", "quantile_grid_99": list(range(99, 0, -1))},
    {"outcome": "unknown", "quantile_grid_99": [100] * 99},
])
def test_no_useful_or_invalid_distribution_fails_closed(r):
    assert P.alt_65(r) is None


def test_real_ind_was_fixture_and_addition_only():
    a = P.build_adhoc(EVIDENCE, process_start_utc="2026-10-04T13:07:49.952742Z")
    fields = a["row_fields"]
    row = next(r for r in a["rows"] if r[fields.index("player")] == "jonathan taylor" and r[fields.index("outcome")] == "rush_yds")
    alt = a["alt_65"][row[-1]]
    assert row[fields.index("median")] == 74
    assert row[fields.index("mean")] == pytest.approx(78.5469)
    assert alt == {"milestone": 60, "probability": .66, "method": "quantile_grid_99_conservative"}
    assert a["not_t24_t90"] and a["horizon"] == "AD_HOC_PREGAME"
    source = next(r for f in (EVIDENCE / "forecasts").glob("*.jsonl") for r in P.read_batch_verified(f)[1] if r.get("id") == row[-1])
    assert source["quantile_grid_99"][45] == 70  # next standard milestone: .54 < .65
    assert P.dumps(a) == P.dumps(P.build_adhoc(EVIDENCE, a["retrieval_start_utc"]))
    for name in ("nfl_phase1_adhoc.json", "nfl_phase1_shadow.json"):
        current = json.loads((REPO / "docs" / name).read_text())
        old = json.loads(subprocess.check_output(["git", "show", f"{BASE}:docs/{name}"], cwd=REPO))
        assert "alt_65" in current
        current.pop("alt_65")
        old.pop("alt_65", None)
        assert current == old  # Every existing row, status, percentile, ID and metadata unchanged.


def test_t24_t90_publish_alts_without_odds_or_mutating_state(tmp_path):
    from test_nfl_new_engine_frontend import make_state, rec, tree_hash, G1, C90, C24
    state = make_state(tmp_path, [(G1, hz, cutoff, ["DONE"], [rec(G1, hz, cutoff, "player", "rush_yds")], {}) for hz, cutoff in [("T24", C24), ("T90", C90)]])
    before = tree_hash(state)
    d = P.build(state)
    assert len(d["alt_65"]) == 2 and d["lines"] == []
    assert all(a["milestone"] == 30 and a["probability"] >= .65 for a in d["alt_65"].values())
    assert P.publish(state, tmp_path / "published.json") == "written"
    assert tree_hash(state) == before
    assert {r[-1] for rows in d["forecasts"].values() for r in rows} == set(d["alt_65"])


def test_scope_predictive_scientific_legacy_and_cfb_untouched():
    allowed = {"nfl_phase1_publisher.py", "docs/nfl.html", "docs/nfl_phase1_adhoc.json", "docs/nfl_phase1_shadow.json", "tests/test_nfl_alt_65.py"}
    changed = set(subprocess.check_output(["git", "diff", "--name-only", BASE], cwd=REPO, text=True).splitlines())
    untracked = set(subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=REPO, text=True).splitlines())
    assert changed | untracked <= allowed


def test_both_frontend_sections_render_alt_and_keep_existing_details(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node unavailable")
    script = re.search(r"<script>(.*?)</script>", (REPO / "docs/nfl.html").read_text(), re.S)[1]
    stub = r'''
const assert = require('node:assert/strict');
class Element {
  constructor() { this.children=[]; this.value=''; this.checked=false; this.textContent=''; }
  append(...xs) { this.children.push(...xs); }
  addEventListener() {}
  get text() { return this.textContent + this.children.map(x => x.text || '').join(' '); }
}
const elems = {};
const document = {getElementById: id => elems[id] ||= new Element(), createElement: () => new Element(), querySelectorAll: () => []};
const fetch = () => new Promise(() => {});
'''
    checks = r'''
const fields=['game_id','player','team','opp','pos','outcome','mean','median','sd','p10','p25','p75','p90','p_active','id'];
const row=['2026_04_IND_WAS','jonathan taylor','IND','WAS','RB','rush_yds',78.5,74,30,36,53,99,127,.98,'taylor'];
D={row_fields:fields,forecasts:{T90:[row]},games:[],alt_65:{taylor:{milestone:60,probability:.66,method:'quantile_grid_99_conservative'}}};
state.h='T90'; render(true);
assert.match(elems.list.text,/65%\+ ALT LINE.*60\+ rushing yards.*Model hit probability: 66%/);
assert.match(elems.list.text,/Mean.*78.5.*Median.*74.0.*SD.*P\(active\).*P25.*P75/);
assert.match(elems.list.text,/Uncertainty/);
A={row_fields:fields,rows:[row],alt_65:D.alt_65}; renderAdhoc();
assert.match(elems.adhoclist.text,/65%\+ ALT LINE.*60\+ rushing yards/);
assert.match(elems.adhoclist.text,/Mean.*Median.*P10–P90.*P\(active\)/);
assert.match(altCard({alt_65:{taylor:null}},row,k=>fields.indexOf(k)).text,/No 65%\+ alt/);
assert.match(altCard({},row,k=>fields.indexOf(k)).text,/No 65%\+ alt/);
'''
    path = tmp_path / "frontend.js"
    path.write_text(stub + script + checks)
    result = subprocess.run([node, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
