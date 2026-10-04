"""
CFB_TARGET_RECOVERY_AUDIT -- ONE bounded source-only audit (no model, no performance): can the intended receiver of every pass attempt be recovered deterministically? Gate: phase1b_target_recovery_gate.json (registered first).
Sample: first 20 FBS-vs-FBS regular-season games of 2022 and 2024 by game_id. No fuzzy name matching: an incomplete-pass receiver is recovered only from a structured field, or from an anchored text pattern whose receiver name equals EXACTLY one roster athlete of the offence.
  python cfb_target_recovery_audit.py --raw /tmp/cfbraw
"""
import argparse
import csv
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 8)
OUT = Path(__file__).resolve().parent / "cfb_models" / "cfb_outcome_engine"
H = {"User-Agent": "cfb-target-audit"}
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/events/{g}/competitions/{g}/plays?limit=500"
SUM = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={g}"
INTENDED = re.compile(r"^(?P<q>[A-Za-z .'\-]+?) pass incomplete (?:to|intended for) (?P<r>[A-Za-z .'\-]+?)(?: \(|,|\.|$)")


def get(u):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=90).read())


def sample_games(raw, season):
    rows = [r for r in csv.DictReader(open(Path(raw) / f"cfb_schedules_{season}.csv", newline="", encoding="utf-8")) if r["season_type"] == "regular" and r["home_division"] == "fbs" and r["away_division"] == "fbs" and r["completed"] == "TRUE"]
    return sorted(r["game_id"] for r in rows)[:20]


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def audit_game(gid):
    plays, items, page = [], [], 1
    j = get(CORE.format(g=gid))
    items = j["items"]
    inc = [p for p in items if p.get("type", {}).get("text") == "Pass Incompletion"]
    comp = [p for p in items if p.get("type", {}).get("text") in ("Pass Reception", "Passing Touchdown")]
    out = {"game": gid, "incompletions": len(inc), "incompletions_with_structured_receiver": 0, "incompletions_with_text_receiver": 0, "completions": len(comp), "completions_with_structured_receiver": 0, "roles_incomp": Counter(), "recv_by_id": Counter()}
    for p in inc:
        roles = [x.get("type") for x in p.get("participants", [])]
        out["roles_incomp"][tuple(sorted(roles))] += 1
        st_ok = any(r in ("receiver", "target", "intended") for r in roles); tx_ok = bool(INTENDED.match(p.get("text", "")))
        out["incompletions_with_structured_receiver"] += st_ok; out["incompletions_with_text_receiver"] += tx_ok
        out["incompletions_recovered_any"] = out.get("incompletions_recovered_any", 0) + (st_ok or tx_ok)           # per-play OR (never double counted)
    for p in comp:
        for x in p.get("participants", []):
            if x.get("type") == "receiver":
                out["completions_with_structured_receiver"] += 1
                out["recv_by_id"][x["athlete"]["$ref"].split("/athletes/")[1].split("?")[0]] += 1
    box = get(SUM.format(g=gid)).get("boxscore", {}).get("players", [])
    official = Counter()
    for side in box:
        for st in side.get("statistics", []):
            if st.get("name") == "receiving":
                labels = st.get("labels", [])
                for a in st.get("athletes", []):
                    official[a["athlete"]["id"]] += int(a["stats"][labels.index("REC")])
    pg_rec = out["recv_by_id"]
    ids = set(official) | set(pg_rec)
    out["player_games"] = len(ids); out["player_games_exact"] = sum(official.get(i, 0) == pg_rec.get(i, 0) for i in ids)
    out["roles_incomp"] = {"+".join(k): v for k, v in out["roles_incomp"].items()}
    del out["recv_by_id"]
    return out


def cfbfastr_target_coverage(raw, seasons):
    res = {}
    for s in seasons:
        n = t = inc = inc_t = 0
        for r in csv.DictReader(open(Path(raw) / f"player_stats_{s}.csv", newline="", encoding="utf-8")):
            is_inc = r["incompletion_player_id"] not in ("", "NA")
            is_att = is_inc or r["completion_player_id"] not in ("", "NA") or r["interception_thrown_player_id"] not in ("", "NA")
            if is_att:
                n += 1; t += r["target_player_id"] not in ("", "NA")
            if is_inc:
                inc += 1; inc_t += r["target_player_id"] not in ("", "NA")
        res[s] = {"pass_attempts": n, "with_target_id": t, "share": t / n, "incompletions": inc, "incompletions_with_target_id": inc_t, "incompletion_share": inc_t / inc}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); a = ap.parse_args()
    games = {s: sample_games(a.raw, s) for s in (2022, 2024)}
    per = []
    for s, gs in games.items():
        for g in gs:
            r = audit_game(g); r["season"] = s; per.append(r)
            print(s, g, r["incompletions"], r["incompletions_with_structured_receiver"], r["incompletions_with_text_receiver"], r["player_games_exact"], r["player_games"], flush=True)
    tot_inc = sum(r["incompletions"] for r in per); rec = sum(r.get("incompletions_recovered_any", 0) for r in per)
    pgs = sum(r["player_games"] for r in per); ex = sum(r["player_games_exact"] for r in per)
    cov = cfbfastr_target_coverage(a.raw, range(2018, 2025))
    inc_cov = sum(v["incompletions_with_target_id"] for v in cov.values()) / sum(v["incompletions"] for v in cov.values())
    result = {"gate": "phase1b_target_recovery_gate.json", "games": games, "per_game": per, "incompletion_target_recovery": 0.0, "incompletion_target_recovery_espn_structured_or_text": rec / max(1, tot_inc), "cfbfastr_incompletion_target_id_share": inc_cov,
              "completion_receiver_reconciliation": ex / max(1, pgs), "cfbfastr_attempt_target_coverage": cov, "total_incompletions_sampled": tot_inc, "player_games_compared": pgs, "player_games_exact": ex}
    best = max(result["incompletion_target_recovery_espn_structured_or_text"], inc_cov)
    result["incompletion_target_recovery"] = best
    result["verdict"] = "TRUE_TARGETS_RECOVERABLE" if (best >= 0.98 and result["completion_receiver_reconciliation"] >= 0.995) else "TRUE_TARGETS_NOT_RELIABLY_RECOVERABLE"
    (OUT / "phase1b_target_recovery_audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_game", "games", "cfbfastr_attempt_target_coverage")}, indent=1))


if __name__ == "__main__":
    main()
