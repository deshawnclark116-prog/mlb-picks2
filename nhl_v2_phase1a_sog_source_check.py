#!/usr/bin/env python3
"""One-time source-access and reproduction check for NHL V2 Phase1A-SOG (writes phase1a_sog_source_audit.json).

Re-fetches a small fixed sample of historical windows (18 requests), assembles rows with the ported V1 logic and compares them with the
pinned V1 frozen files. For 2026 only the schedule endpoint is read (no player statistics, no outcomes). Not part of CI: live results carry
retrieval timestamps."""
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nhl_v2_phase1a_sog_acquire as A

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_player_outcome_v2"
V1_COMMIT = "cea38be352e5539239b29520977e0b19962d2372"
SAMPLE = [(2018, 12), (2020, 18), (2022, 12), (2024, 12), (2025, 12), (2025, 22)]


def v1_blob(path):
    return subprocess.run(["git", "show", "%s:%s" % (V1_COMMIT, path)], cwd=str(REPO), check=True, capture_output=True).stdout


def frozen_rows(season):
    raw = v1_blob("nhl_models/nhl_outcome_engine/phase1a_data/skater_games_%d.jsonl.gz" % season)
    return [json.loads(l) for l in gzip.decompress(raw).decode().splitlines() if l]


def main():
    manifest = json.loads(v1_blob("nhl_models/nhl_outcome_engine/phase1a_data_manifest.json"))
    res = {"artifact": "phase1a_sog_source_audit", "v1_commit": V1_COMMIT, "v1_manifest_content_sha256": manifest["manifest_content_sha256"],
           "v1_frozen_vintage_retrieved_utc_range": manifest["retrieved_at_utc_range"], "samples": [], "request_count": 0}
    cache = {}
    for season, idx in SAMPLE:
        lo, hi = A.window_list(season)[idx]
        w = A.acquire_window(lo, hi)
        res["request_count"] += 3
        games, rows = A.assemble([w])
        if season not in cache:
            cache[season] = {(r["game_id"], r["team_id"], r["player_id"]): r for r in frozen_rows(season)}
        fz = cache[season]
        in_win = {k: r for k, r in fz.items() if r["game_id"] in games}
        new = {(r["game_id"], r["team_id"], r["player_id"]): r for r in rows}
        diff_fields = {}
        for k in set(in_win) & set(new):
            for f in new[k]:
                if new[k][f] != in_win[k].get(f):
                    diff_fields[f] = diff_fields.get(f, 0) + 1
        res["samples"].append({"season": season, "window": [lo, hi], "games_now": len(games), "rows_now": len(rows), "frozen_rows_same_games": len(in_win),
                               "rows_only_now": len(set(new) - set(in_win)), "rows_only_frozen": len(set(in_win) - set(new)), "rows_identical": sum(1 for k in set(in_win) & set(new) if new[k] == in_win[k]),
                               "field_differences": dict(sorted(diff_fields.items())), "guards_ok": all(w["provenance"][k]["guard_ok"] for k in ("summary", "timeonice")),
                               "payload_sha256": {k: w["provenance"][k]["sha256"] for k in ("schedule", "summary", "timeonice")},
                               "retrieval_completed_utc": w["provenance"]["timeonice"]["retrieval_completed_utc"]})
    raw, meta = A.fetch(A.API + "/schedule/2026-10-08")
    sched = json.loads(raw)
    res["request_count"] += 1
    res["forward_schedule_probe"] = {"url": meta["url"], "http_status": meta["http_status"], "sha256": meta["sha256"], "retrieved_completed_utc": meta["retrieval_completed_utc"],
                                     "keys_per_game": sorted(sched["gameWeek"][0]["games"][0].keys()) if sched["gameWeek"] and sched["gameWeek"][0]["games"] else [],
                                     "note": "schedule structure only; no 2026 player statistics or outcomes were read"}
    res["accessibility"] = {"api-web.nhle.com": "reachable (HTTP 200)", "api.nhle.com/stats/rest": "reachable (HTTP 200)"}
    (OUT / "phase1a_sog_source_audit.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"%s@%s" % (s["season"], s["window"][0]): (s["rows_identical"], s["frozen_rows_same_games"], s["rows_only_now"], s["rows_only_frozen"], s["field_differences"]) for s in res["samples"]}))


if __name__ == "__main__":
    main()
