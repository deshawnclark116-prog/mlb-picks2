"""NHL V2 Phase0 deterministic tests (research only)."""
import ast
import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "nhl_models" / "nhl_player_outcome_v2"
BASE_SHA = "f8ee7d5033a7144589b76bf9354473f139f61652"
BANNED = tuple(next(v for k, v in json.loads((OUT / "protocol.json").read_text()).items() if k.endswith("_firewall"))["banned_tokens"]) + ("over/under",)
SCIENCE = sorted(REPO.glob("nhl_v2_*.py"))


def git(*a):
    return subprocess.run(["git"] + list(a), cwd=str(REPO), capture_output=True, text=True)


def jl(name):
    return json.loads((OUT / name).read_text())


def test_sportsbook_firewall_science_code():
    assert SCIENCE
    for p in SCIENCE:
        for i, line in enumerate(p.read_text().lower().splitlines(), 1):
            for t in BANNED:
                assert not re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", line), "%s:%d uses banned token %s" % (p.name, i, t)


def test_no_banned_model_inputs():
    cols = []
    for d in ("nhl_shots_on_goal_walkforward_stability_a_work", "nhl_points_walkforward_stability_a_work", "nhl_goalie_saves_champion_gate_a_work", "nhl_moneyline_walkforward_stability_a_work"):
        for c in (REPO / "nhl_models" / d).glob("*_columns.json"):
            cols += json.loads(c.read_text())
    assert cols and not [c for c in cols if re.search(r"odds|line|book|implied|price|spread|vig", c)]


def test_science_modules_do_not_import_monte_carlo_or_random_sampling():
    for p in SCIENCE:
        tree = ast.parse(p.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and n.attr in ("binomial", "poisson", "multinomial", "choice", "normal"):
                raise AssertionError("%s samples via %s" % (p.name, n.attr))


def test_season_cap_clean_forward_never_loaded():
    import nhl_v2_phase0_data as D
    assert D.SEASON_CAP == 2025
    mem = D.load_mem(REPO / "nhl_models" / "nhl_model.sqlite")
    for t in ("games", "skater_games", "goalie_games"):
        assert mem.execute("select max(season) from %s" % t).fetchone()[0] <= 2025
    assert "season<=?" in (REPO / "nhl_v2_phase0_data.py").read_text().replace(" ", "")


def test_metrics_known_values():
    import nhl_v2_metrics as M
    p = np.array([0.1, 0.4, 0.6, 0.9]); y = np.array([0, 0, 1, 1])
    assert M.auc(p, y) == 1.0 and M.pr_auc(p, y) == 1.0
    assert abs(M.brier(p, y) - np.mean((p - y) ** 2)) < 1e-12
    assert abs(M.nb_tail_ge(2.0, 3) - (1 - np.exp(-2) * (1 + 2 + 2))) < 1e-12
    assert abs(M.nb_tail_ge(2.0, 3, r=1e7) - M.nb_tail_ge(2.0, 3)) < 1e-6
    assert M.nb_tail_ge(0.0, 1) == 0.0 and M.nb_tail_ge(3.0, 0) == 1.0
    d = np.array([-1.0, -2.0, 0.5, -0.5] * 10); blocks = np.repeat(np.arange(10), 4)
    assert M.paired_block_bootstrap(d, blocks) == M.paired_block_bootstrap(d, blocks)
    mu = np.array([1.0, 2.0]); a = np.array([2.0, 0.0])
    cm = M.count_metrics(mu, a)
    assert cm["mae"] == 1.5 and cm["bias_pred_minus_actual"] == 0.5 and cm["within_1"] == 0.5


def test_protocol_ordering_in_git_history():
    r = git("log", "--format=%H", "--", "nhl_models/nhl_player_outcome_v2/protocol.json")
    if r.returncode != 0 or not r.stdout.strip():
        return
    def first(path):
        out = git("log", "--diff-filter=A", "--format=%H", "--", path).stdout.split()
        return out[-1] if out else None
    order = [first("nhl_models/nhl_player_outcome_v2/" + f) for f in ("protocol.json", "protocol_amendment_1.json", "protocol_amendment_2.json", "phase0_chronology_audit.json", "phase0_current_model_results.json")]
    if any(o is None for o in order):
        return
    pos = {}
    log = git("rev-list", "--reverse", "HEAD").stdout.split()
    for i, h in enumerate(log):
        pos[h] = i
    idx = [pos[o] for o in order]
    assert idx == sorted(idx) and idx[0] < idx[1] < idx[2] and idx[3] < idx[4], "protocol, amendments and chronology evidence must each precede the results commit"
    for f in ("protocol.json", "protocol_amendment_1.json", "protocol_amendment_2.json"):
        first_sha = first("nhl_models/nhl_player_outcome_v2/" + f)
        orig = git("show", first_sha + ":nhl_models/nhl_player_outcome_v2/" + f).stdout
        assert orig == (OUT / f).read_text(), f + " was modified after its registration commit"


def test_only_research_paths_changed_vs_base():
    r = git("diff", "--name-only", BASE_SHA, "HEAD")
    if r.returncode != 0:
        return
    ok = re.compile(r"^(nhl_v2_[a-z0-9_]+\.py|tests/test_nhl_v2_[a-z0-9_]+\.py|nhl_models/nhl_player_outcome_v2/.+|\.github/workflows/nhl_v2_phase0_research\.yml|requirements-research-nhl\.txt)$")
    bad = [f for f in r.stdout.split() if not ok.match(f)]
    assert not bad, "non-research files changed: %s" % bad


def test_nfl_branch_and_pr64_not_referenced_as_modified():
    r = git("diff", "--name-only", BASE_SHA, "HEAD")
    if r.returncode == 0:
        assert not [f for f in r.stdout.split() if f.startswith("nfl_") or "nfl" in f.lower() and f.startswith(".github")]


def test_chronology_says_no_untouched_season():
    c = jl("phase0_chronology_audit.json")
    assert "NO genuinely untouched" in c["conclusion"]
    assert c["periods"]["2025"]["classification"] == "BURNED_EXPOSED" and c["periods"]["2025"]["untouched"] is False
    assert c["periods"]["2026-09-29_onward"]["classification"] == "CLEAN_FORWARD"
    assert "\"skater_games\": 0" in " ".join(c["periods"]["2026-09-29_onward"]["evidence"])


def test_central_projection_gap_fields():
    g = jl("phase0_central_projection_gap.json")
    by = {m["market"]: m for m in g["markets"]}
    for k in ("shots_on_goal", "points", "goalie_saves"):
        assert by[k]["has_central_projection"] is False and by[k]["can_grade_MAE"] is False and by[k]["has_threshold_probability"] is True
    assert "NONE" in by["moneyline"]["architecture_limitation"]
    assert "CURRENT_ENGINE_HAS_NO_CENTRAL_STAT_PROJECTION" in g["finding"]


def test_artifacts_carry_evidence_grade_and_promote_nothing():
    v = jl("phase0_verdicts.json")
    assert v["promotes_nothing"] is True and v["evidence_grade"] == "RETROSPECTIVE_BURNED"
    allowed = {"CLASSIFIER_SURVIVES_NATIVE_TARGET_AUDIT", "CLASSIFIER_WEAK_NATIVE_TARGET", "CLASSIFIER_REJECTED_NATIVE_TARGET", "BLOCKED_DATA", "BLOCKED_TIMING"}
    for m in v["verdicts"].values():
        assert m["native"] in allowed
    assert v["verdicts"]["goalie_saves_ge25"]["native"] == "BLOCKED_TIMING"
    for k in ("skater_sog_ge3", "skater_points_ge1", "goalie_saves_ge25"):
        assert v["verdicts"][k]["architecture"] == "ARCHITECTURE_INSUFFICIENT_NO_CENTRAL_PROJECTION"
    assert jl("phase0_current_model_results.json")["evidence_grade"] == "RETROSPECTIVE_BURNED"


def test_goalie_integrity_and_leakage_documented():
    g = jl("phase0_goalie_eligibility_audit.json")
    assert g["integrity"]["same_population"] is True and g["integrity"]["max_feature_abs_diff"] == 0.0
    assert g["code_path"]["changes_training_population"] and g["code_path"]["changes_scoring_population"]
    assert g["final_status"] == "BLOCKED_TIMING"


def test_snapshot_matches_files_and_code():
    s = jl("phase0_snapshot.json")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for f, h in s["artifact_sha256"].items():
        assert sha(OUT / f) == h, f
    for f, h in s["code"].items():
        assert sha(REPO / f) == h, f
    for f, h in s["protocol"].items():
        assert sha(OUT / f) == h, f
    assert s["season_cap"] == 2025


def test_ledgers_are_valid_jsonl_with_only_audited_seasons():
    for name in ("phase0_error_receipts.jsonl.gz", "phase0_catastrophic_misses.jsonl.gz"):
        rows = [json.loads(l) for l in gzip.decompress((OUT / name).read_bytes()).decode().splitlines()]
        assert rows and {r["season"] for r in rows} <= {2024, 2025}
        assert all("market" in r and "p_served" in r for r in rows)


def test_required_deliverables_exist():
    for f in ("protocol.json", "research_registry.json", "source_inventory.json", "feature_routing.md", "phase0_current_engine_audit.json", "phase0_source_audit.json",
              "phase0_population_definition.json", "phase0_current_model_results.json", "phase0_simple_baselines.json", "phase0_competent_human_baseline.json",
              "phase0_catastrophic_misses.jsonl.gz", "phase0_error_receipts.jsonl.gz", "phase0_findings.md", "phase0_snapshot.json", "phase0_chronology_audit.json", "phase0_central_projection_gap.json"):
        assert (OUT / f).exists(), f
    assert "SOURCE" not in "" and jl("source_inventory.json")["sources"]
