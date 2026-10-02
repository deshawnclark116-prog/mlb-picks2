"""
NFL_PHASE1_BOARD  (Phase 1C, shadow research)

Research-only board: "what does the system believe this player will actually do?"  Reads forecast records (never writes them).
No sportsbook line, no over/under: only the model's own distribution summaries.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import nfl_phase1_forecast as FC
import nfl_phase1_store as ST

SHOW = ["rush_yds", "rec_yds", "rec", "pass_yds", "pass_td", "int", "rush_td", "rec_td", "atd", "tackles", "sacks", "def_int"]
UNIT = {"rush_yds": "rush yds", "rec_yds": "rec yds", "rec": "receptions", "pass_yds": "pass yds", "pass_td": "pass TD", "int": "INT thrown", "rush_td": "rush TD", "rec_td": "rec TD",
        "atd": "rush+rec TD", "tackles": "tackles", "sacks": "sacks", "def_int": "def INT"}


def build(store, season, week, horizon, per_team=(1, 3, 5, 6)):
    recs = [r for r in FC.read_forecasts(store) if r["season"] == season and r["week"] == week and r["horizon"] == horizon]
    by_game = defaultdict(list)
    for r in recs:
        by_game[(r["game_id"], r["team"])].append(r)
    lines = [f"# Shadow board - {season} week {week} - {horizon}", "", "Model beliefs about what each player will do (mean, median, central 80% interval, P(active), uncertainty). "
             "No sportsbook line is used or shown. Development / shadow only.", ""]
    for (gid, team), rs in sorted(by_game.items()):
        opp = rs[0]["opponent"]
        lines += [f"## {gid}: {team} (vs {opp}) - cutoff {rs[0]['cutoff']}", "", "| player | pos | outcome | mean | median | 80% interval | P(active) | exp. opportunities | role share | uncertainty (reasons) |", "|---|---|---|---|---|---|---|---|---|---|"]
        players = defaultdict(dict)
        for r in rs:
            players[r["player_id"]][r["outcome"]] = r
        def top(outcomes, k):
            ranked = sorted(players.items(), key=lambda kv: -max((kv[1][o]["mean"] for o in outcomes if o in kv[1]), default=-1))
            return [p for p, d in ranked if any(o in d for o in outcomes)][:k]
        chosen = []
        for outs, k in ((("pass_yds",), per_team[0]), (("rush_yds",), per_team[1]), (("rec_yds",), per_team[2]), (("tackles",), per_team[3])):
            chosen += [p for p in top(outs, k) if p not in chosen]
        for pid in chosen:
            for o in SHOW:
                r = players[pid].get(o)
                if r is None:
                    continue
                q = r["quantiles"]
                ev = "" if r.get("event_probability_ge1") is None else f", P(>=1)={r['event_probability_ge1']:.3f}"
                lines.append(f"| {r['player_name'] or pid} | {r['position']} | {UNIT[o]} | {r['mean']:.2f} | {r['median']:.1f} | {q['p10']:.1f} to {q['p90']:.1f}{ev} | "
                             f"{'n/a' if r['p_active'] is None else format(r['p_active'], '.2f')} | {'n/a' if r['expected_opportunities'] is None else format(r['expected_opportunities'], '.1f')} | "
                             f"{'n/a' if (r.get('role_state') or {}).get('propensity_share') is None else format(r['role_state']['propensity_share'], '.2f')} | {r['uncertainty']['score']:.2f} ({', '.join(r['uncertainty']['reasons'])}) |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--season", type=int, required=True); ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--horizon", default="T24"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).write_text(build(ST.Store(a.root, "forecasts"), a.season, a.week, a.horizon))
