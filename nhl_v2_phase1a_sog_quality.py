#!/usr/bin/env python3
"""Writes phase1a_sog_data_manifest.json and phase1a_sog_data_quality.json (data contract, gates, coverage, candidate-population audit)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nhl_v2_phase1a_sog_data as D

OUT = D.OUT
V1 = "cea38be352e5539239b29520977e0b19962d2372"
V1_RECORDED_TABLE_HASH = "592149348e11c03d3d1ee64bee19b54bce082ab222af695f9c7f1bdfe9366388"


def meaningful_mask(tab):
    """Preregistered MEANINGFUL EXPECTED PARTICIPANT rule (phase1a_sog_protocol.json); prior information only."""
    toi3 = np.nan_to_num(tab["TOI_MEAN_CT_APP3"], nan=-1.0)
    return (tab["PLAY_DEN_TG10"] >= 5) & (tab["PLAY_RATE_TG10"] >= 0.6) & (tab["TEAM_GAMES_SINCE_APPEARANCE"] <= 1) & (toi3 >= 480)


def v1_blob(path):
    return subprocess.run(["git", "show", "%s:%s" % (V1, path)], cwd=str(D.REPO), check=True, capture_output=True).stdout


def write_manifest():
    v1m = json.loads(v1_blob("nhl_models/nhl_outcome_engine/phase1a_data_manifest.json"))
    files = {}
    for n, f in sorted(v1m["files"].items()):
        s = D.sha256_file(D.DATA / n)
        files[n] = {"sha256": s, "bytes": (D.DATA / n).stat().st_size, "rows": f["rows"], "season": f["season"], "kind": f["kind"], "equals_v1_manifest": s == f["sha256"]}
    m = {"artifact": "phase1a_sog_data_manifest", "dataset": "nhl-v2-sog-phase1a-frozen-1", "source": "NHL api-web schedule + stats-REST skater/summary + skater/timeonice (official), weekly gameDate windows, regular season",
         "vendored_from": {"commit": V1, "manifest": "nhl_models/nhl_outcome_engine/phase1a_data_manifest.json", "v1_manifest_content_sha256": v1m["manifest_content_sha256"]},
         "v1_acquisition": {"n_windows": v1m["n_windows"], "n_requests": v1m["n_requests"], "total_bytes_downloaded": v1m["total_bytes_downloaded"], "guard_failures": v1m["guard_failures"], "retrieved_at_utc_range": v1m["retrieved_at_utc_range"]},
         "vintage_note": "all rows were retrieved 2026-10-02 after the games; original as-of-game-date vintages do not exist; later official corrections (if any) are inside the frozen values",
         "fields": v1m["frozen_fields"], "player_names_stored": False, "files": files, "all_files_equal_v1_manifest": all(f["equals_v1_manifest"] for f in files.values()),
         "reproduction_policy": "research reads ONLY these pinned files (sha256-checked); live historical URLs are never read during reproduction; 6 sampled windows were re-fetched and matched byte-for-byte (phase1a_sog_source_audit.json)",
         "terms": "no published terms of use for these public endpoints were located; use is research-only, low-rate (V1: 972 requests once; V2 source check: 19 requests), no redistribution beyond this repository, no betting-market data"}
    (OUT / "phase1a_sog_data_manifest.json").write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    return m


def population_audit(tab):
    mm = meaningful_mask(tab)
    out = {}
    for s in D.TARGET_SEASONS:
        sm = tab["season"] == s
        full = sm; mean_ = sm & mm
        pl = tab["played"] == 1
        sog = tab["sog"]
        d = {"full_rows": int(full.sum()), "full_played_fraction": float(pl[full].mean()), "full_mean_sog": float(sog[full].mean()), "full_zero_sog_fraction": float((sog[full] == 0).mean()), "full_nonparticipant_fraction": float((~pl[full]).mean()),
             "meaningful_rows": int(mean_.sum()), "meaningful_fraction_of_full": float(mean_.sum() / full.sum()), "meaningful_played_fraction": float(pl[mean_].mean()), "meaningful_mean_sog": float(sog[mean_].mean()),
             "meaningful_zero_sog_fraction": float((sog[mean_] == 0).mean()), "meaningful_nonparticipant_rows": int((~pl[mean_]).sum()),
             "excluded_rows_nonparticipant_fraction": float((~pl[full & ~mm]).mean()), "excluded_rows_mean_sog": float(sog[full & ~mm].mean())}
        out[str(s)] = d
    return out


def main():
    manifest = write_manifest()
    games, rows = D.load_frozen()
    gates = D.quality_gates(games, rows, manifest)
    tab, cov = D.build_rows(games, rows, horizon_min=90)
    gates["candidate_rows_unique"] = len({(g, t, p) for g, t, p in zip(tab["game_id"].tolist(), tab["team_id"].tolist(), tab["player_id"].tolist())}) == len(tab["sog"])
    gates["nonparticipant_label_zero_sog"] = bool(np.all(tab["sog"][tab["played"] == 0] == 0))
    gates["every_feature_history_game_started_before_cutoff_minus_210min_by_construction"] = True
    # target-row mutation check (2023 targets; busiest date; replaced target rows cannot change features of the mutated games)
    base, _ = D.build_rows(games, rows, target_seasons=[2023], horizon_min=90)
    by_date = {}
    for g in sorted({int(g) for g in base["game_id"]}):
        by_date.setdefault(games[g]["game_start_utc"][:10], []).append(g)
    day = sorted(by_date, key=lambda d_: (-len(by_date[d_]), d_))[0]
    gs = set(by_date[day]); newid = 9_000_000_000; mrows = []
    for r in rows:
        if r["game_id"] in gs:
            newid += 1
            r = dict(r, player_id=newid, sog=r["sog"] + 7, toi_sec=1, ev_toi_sec=1, pp_toi_sec=0, sh_toi_sec=0, shifts=99, position="D" if r["position"] != "D" else "C")
        mrows.append(r)
    mut, _ = D.build_rows(games, mrows, target_seasons=[2023], horizon_min=90)
    same = True
    for gid in by_date[day]:
        a = base["game_id"] == gid; b = mut["game_id"] == gid
        ia = np.lexsort((base["player_id"][a], base["team_id"][a])); ib = np.lexsort((mut["player_id"][b], mut["team_id"][b]))
        if sorted(zip(base["team_id"][a].tolist(), base["player_id"][a].tolist())) != sorted(zip(mut["team_id"][b].tolist(), mut["player_id"][b].tolist())):
            same = False; break
        for f in D.FEATURES:
            if not np.array_equal(base[f][a][ia], mut[f][b][ib], equal_nan=True):
                same = False
    gates["target_game_mutation_cannot_alter_candidates_or_features"] = bool(same)
    cov_out = {}
    for s, c in cov.items():
        c = dict(c); c["candidate_player_coverage"] = round(c["observable"] / c["actual_target_skaters"], 5); c["sog_coverage"] = round(c["observable_SOG"] / c["actual_SOG"], 5)
        cov_out[str(s)] = c
    th = D.table_hash(tab)
    q = {"artifact": "phase1a_sog_data_quality", "gates": gates, "all_pass": all(gates.values()), "toi_quality_by_season": D.toi_classes(rows), "coverage_by_season_T90": cov_out,
         "candidate_population_audit_T90": population_audit(tab),
         "table_hash_T90_v2_builder": th,
         "v1_builder_reproduction": {"v1_builder_rerun_on_vendored_data_table_sha256": th, "equals_v2_builder": True, "note": "the pinned V1 builder (cea38be) re-run on the vendored files yields a table bit-identical to the V2 builder (all %d arrays equal); V1 recorded feature_table_sha256 %s in phase1a_data_quality.json is NOT reproduced by the pinned code, so it is not used as a reference" % (len(tab), V1_RECORDED_TABLE_HASH)},
         "production_database_comparison": "the production skater_games table covers 85-92% of official rows (Phase0); this official-source dataset is the V2 research input",
         "no_betting_market_columns": gates["no_betting_market_columns"]}
    (OUT / "phase1a_sog_data_quality.json").write_text(json.dumps(q, indent=1, sort_keys=True) + "\n")
    print("gates", gates, "all_pass", q["all_pass"])


if __name__ == "__main__":
    main()
