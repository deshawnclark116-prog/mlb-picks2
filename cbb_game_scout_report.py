#!/usr/bin/env python3
"""Human-readable source-grounded CBB matchup notes, NOT betting advice."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def render(report, max_games=80):
    games = report.get("game_predictions", [])
    if max_games <= 0:
        raise ValueError("max_games must be positive")
    lines = [
        "# NCAA men's basketball — game outcome intelligence",
        "",
        f"Research generated at {report.get('generated_utc', 'unknown')}.",
        "**Independent point projections, not sportsbook lines or official picks.**",
        "Observed matchups are retrospective descriptive evidence, not proven causal score adjustments.",
        "No game's starting five or injury status is verified by the team/player boxscore sources.",
        "",
        "## Research benchmark",
        "",
    ]
    tests = report.get("report", {}).get("untouched_2026_holdout", {})
    if tests:
        lines += [
            f"- Historical 2025–26 game cohort: {tests.get('games')} games; "
            f"winner accuracy {100 * tests.get('moneyline_accuracy', 0):.2f}%; "
            f"moneyline Brier {tests.get('moneyline_brier')}; "
            f"margin MAE {tests.get('margin_mae')} points; "
            f"total MAE {tests.get('total_mae')} points.",
            "- That prior cohort has been used to evaluate the model. It is not a fresh "
            "or independently audited pregame betting performance record.",
            "",
        ]
    by_day = defaultdict(list)
    for p in games[:max_games]:
        try:
            day = datetime.fromisoformat(p["tip_utc"]).strftime("%Y-%m-%d")
        except (ValueError, KeyError):
            day = "unknown"
        by_day[day].append(p)
    for day, day_games in sorted(by_day.items()):
        lines += [f"## {day}", ""]
        for p in day_games:
            home, away = p["home"], p["away"]
            lines += [
                f"### {away} at {home}",
                "",
                f"Projected score: **{home} {p['projected_home_score']} — "
                f"{away} {p['projected_away_score']}**",
                f"Projected home margin: **{p['projected_home_margin']:+.1f}**; "
                f"combined points: **{p['projected_total']:.1f}**; "
                f"home model P(win): **{100 * p['p_home_win']:.1f}%**.",
                "",
            ]
            s = p.get("game_intelligence")
            if not s:
                lines += ["No source-grounded game intelligence snapshot supplied.", ""]
                continue
            lines += [
                f"- **Tempo:** {home} prior pace {s['home_pace_prior']:.1f}; "
                f"{away} prior pace {s['away_pace_prior']:.1f}; "
                f"shared proxy {s['shared_pace_proxy']:.1f} possessions.",
            ]
            for team, opponent, label in [
                (home, away, "home_offense_vs_away_defense"),
                (away, home, "away_offense_vs_home_defense"),
            ]:
                d = s[label]
                lines += [
                    f"- **{team} offense against {opponent} defense:** "
                    f"attack {d['offensive_points_per_100_possessions_prior']:.1f} points/100 possessions "
                    f"vs opponent's {d['opponent_defensive_points_allowed_per_100_possessions_prior']:.1f} allowed; "
                    f"eFG% historical overlap {d['shooting_efficiency_vs_defensive_allowance_proxy_pct_points']:+.1f} pp; "
                    f"turnover difference {d['turnover_rate_vs_opponent_forced_rate_pct_points']:+.1f} pp; "
                    f"offensive rebound difference {d['offensive_rebound_rate_vs_opponent_allowed_pct_points']:+.1f} pp.",
                ]
                named = d["top_previous_rotation_players"][:4]
                if named:
                    roster = ", ".join(
                        f"{r['name']} ({r['minutes_in_last_3_observed_team_games']:.0f} "
                        f"observed minutes)" for r in named)
                    lines += [f"  - Recent rotation observed: {roster}. "
                              "**Not a confirmed lineup.**"]
                else:
                    lines += ["  - Recent player rotation: unavailable."]
            lines += [
                "- **Unknown:** starters, injuries, coaching game plan, individual defensive assignments, "
                "and sportsbook prices unless separately sourced.",
                "- **No spread or O/U betting edge asserted:** prices require independently timestamped actual lines.",
                "",
            ]
    lines += [
        "---", "",
        f"Displayed {sum(map(len, by_day.values()))} of {len(games)} upcoming research forecasts.",
        "Full machine-readable event and matchup evidence remains in the source JSON.", "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=1000)
    args = ap.parse_args()
    data = json.loads(Path(args.predictions).read_text())
    result = render(data, max_games=args.max_games)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(result)
    print(f"rendered {len(data.get('game_predictions', []))} game records to {args.out}")


if __name__ == "__main__":
    main()
