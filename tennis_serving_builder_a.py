#!/usr/bin/env python3
"""
TENNIS_SERVING_BUILDER_A

Builds docs/tennis_predictions.json for today's live ATP/WTA schedule.
Ships exactly the two markets that cleared a real, honest backtest
this session (see tennis_total_games_champion_gate_a.py and
tennis_set_betting_champion_gate_a.py) -- total_aces, double_faults,
moneyline, and games_spread all FAILED their pre-registered gates and
are deliberately NOT served here (see their own gate scripts/reports
for why).

  total_games   real market line + odds fetched from The Odds API's
                tennis totals market (confirmed live on the currently-
                active tournament-specific sport keys, e.g.
                tennis_atp_us_open) -- graded like every other market
                in this repo: model probability vs a REAL line, HIGH/
                MEDIUM/LOW confidence computed by /tennis/predictions
                in api.py the same way /cfb and /nfl already do.

  set_betting   NO real market exists for this on any odds provider
                checked (The Odds API's tennis coverage is h2h/totals/
                spreads only). Served as an explicitly-labeled
                projection only -- the model's own most likely exact
                score and its probability, with no confidence tier,
                since "confidence" implies a real line to be confident
                against and there isn't one. api.py's /tennis/predictions
                route must not run conf_from_prob() on these.

Feature/model code below (Elo with tennis-standard K decay, surface
blending, recency-weighted games-per-set) is copied from the champion-
gate scripts rather than imported, matching this repo's existing
gate-script/serving-builder pairing convention (e.g. cfb_serving_builder_a.py
alongside cfb_*_champion_gate_a.py) -- keep any future constant changes
in sync with tennis_total_games_champion_gate_a.py and
tennis_set_betting_champion_gate_a.py or re-validate.

Schedule source: ESPN's public tennis scoreboard API (confirmed real,
free, unauthenticated -- see tennis_player_matches_foundation_a.py's
sibling research), queried with today's date and filtered to
status.type.state in ("pre", "in") -- excludes already-final matches
from the board, same principle as CFB's graded-pick removal.

Player matching: ESPN's athlete names are matched against a
normalized-name index built from TML-Database's own historical
winner_name/loser_name columns (no shared player-ID crosswalk exists
between ESPN and TML-Database). Unmatched players are skipped with a
note, not guessed at.

Surface: ESPN's scoreboard doesn't expose it directly. Grand Slam
surfaces are hardcoded (real, static facts); everything else is looked
up from the most common surface TML-Database has recorded for that
tourney_name historically. Falls back to "Hard" (the tour's plurality
surface) with a note if no historical match is found -- a real
approximation, disclosed rather than hidden.

ATP ONLY, but not for a data reason anymore: a real WTA data source
exists (stats.tennismylife.org/data/{year}_wta.csv -- TML-Database's
own GitHub repo has none, but their website does) and is fully wired
into tennis_player_matches_foundation_a.py. The reason WTA still isn't
served is that all 6 markets were backtested separately against real
WTA data with the same bars used for ATP, and NONE of them passed --
see tennis_*_gate_report_wta.json for each market's real result
(closest was moneyline at 64.2% holdout accuracy, just under the 0.65
floor). --tours defaults to atp for that reason: passing wta would
build real WTA ratings and produce real picks, just picks from models
that failed their own validation, which is not something to serve
silently. Re-run the WTA gates if the underlying data source adds more
history/seasons and re-evaluate before ever changing this default.

Confirmed real-world quirk (found while testing this script): ESPN's
per-tour scoreboard endpoints (.../tennis/atp/scoreboard and
.../tennis/wta/scoreboard) both return the FULL shared bracket for
combined events like majors -- Men's Singles, Women's Singles, and
doubles groupings all appear under BOTH endpoints during a Slam. Must
filter by grouping.displayName, not just by which endpoint was hit, or
ATP and WTA end up serving duplicate picks against each other.
"""
import argparse
import datetime as dt
import json
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
import requests

DB_PATH_DEFAULT = Path("tennis_models/tennis_model.sqlite")
OUT_PATH_DEFAULT = Path("docs/tennis_predictions.json")

RECENCY_DECAY = 0.6
MIN_PRIOR_MATCHES = 5
MIN_PRIOR_MATCHES_TOTAL_GAMES = 10
MIN_SURFACE_MATCHES_FOR_BLEND = 4
INITIAL_ELO = 1500.0

GRAND_SLAM_SURFACE = {
    "australian open": "Hard",
    "french open": "Clay",
    "roland garros": "Clay",
    "wimbledon": "Grass",
    "us open": "Hard",
}
GRAND_SLAMS_BO5 = set(GRAND_SLAM_SURFACE)

THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis"

BUILDER_VERSION = "tennis_serving_builder_a/1.0"


def normalize_name(name):
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def k_factor(n):
    return 250.0 / ((n + 5) ** 0.4)


def elo_expected(a, b):
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


def recency_weighted_mean(values, decay=RECENCY_DECAY):
    if not values:
        return None, 0
    num = den = 0.0
    w = 1.0
    for v in reversed(values):
        num += w * v
        den += w
        w *= decay
    return (num / den if den > 0 else None), len(values)


def match_win_prob_from_set_prob(q, races_to):
    total = 0.0
    for k in range(races_to, 2 * races_to):
        total += comb(k - 1, races_to - 1) * (q ** races_to) * ((1 - q) ** (k - races_to))
    return total


def invert_to_set_prob(p, races_to):
    # p == 0.5 exactly (e.g. two players still at INITIAL_ELO facing off)
    # must short-circuit -- otherwise this recurses on 1-p, which is ALSO
    # exactly 0.5, forever. Real bug caught on WTA data; see
    # tennis_set_betting_champion_gate_a.py's identical fix for detail.
    if p == 0.5:
        return 0.5
    if p < 0.5:
        return 1 - invert_to_set_prob(1 - p, races_to)
    lo, hi = 0.5, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if match_win_prob_from_set_prob(mid, races_to) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def outcome_distribution(q, races_to):
    dist = {}
    for k in range(races_to, 2 * races_to):
        loser_sets = k - races_to
        dist[("P1", races_to, loser_sets)] = comb(k - 1, races_to - 1) * (q ** races_to) * ((1 - q) ** loser_sets)
        dist[("P2", races_to, loser_sets)] = comb(k - 1, races_to - 1) * ((1 - q) ** races_to) * (q ** loser_sets)
    return dist


def build_state(con):
    """Replays full match history once to produce: final Elo ratings
    (overall + surface), each player's recency-weighted games-per-set/
    expected-sets history for total_games, a normalized-name -> most
    recent player_id index, and a tourney_name -> most-common-surface
    lookup."""
    rows = con.execute(
        "SELECT match_id, match_date, tourney_name, surface, best_of, "
        "winner_id, winner_name, loser_id, loser_name, score, is_incomplete "
        "FROM matches ORDER BY match_date, match_id").fetchall()

    overall_elo, overall_n = {}, {}
    surface_elo, surface_n = {}, {}
    total_games_hist = defaultdict(list)  # pid -> [(games_per_set, n_sets, surface, best_of)]
    name_index = {}
    tourney_surface_votes = defaultdict(lambda: defaultdict(int))

    SET_RE = re.compile(r"^(\d+)-(\d+)")

    for (match_id, match_date, tourney_name, surface, best_of,
         winner_id, winner_name, loser_id, loser_name, score, is_incomplete) in rows:
        if winner_id and winner_name:
            name_index[normalize_name(winner_name)] = winner_id
        if loser_id and loser_name:
            name_index[normalize_name(loser_name)] = loser_id
        if tourney_name and surface:
            tourney_surface_votes[normalize_name(tourney_name)][surface] += 1

        if not surface or not winner_id or not loser_id:
            continue

        p1, p2 = sorted([winner_id, loser_id])
        p1_is_winner = 1 if p1 == winner_id else 0

        e1o, e2o = overall_elo.get(p1, INITIAL_ELO), overall_elo.get(p2, INITIAL_ELO)
        n1o, n2o = overall_n.get(p1, 0), overall_n.get(p2, 0)
        e1s, e2s = surface_elo.get((p1, surface), INITIAL_ELO), surface_elo.get((p2, surface), INITIAL_ELO)
        n1s, n2s = surface_n.get((p1, surface), 0), surface_n.get((p2, surface), 0)

        k1, k2 = k_factor(n1o), k_factor(n2o)
        overall_elo[p1] = e1o + k1 * (p1_is_winner - elo_expected(e1o, e2o))
        overall_elo[p2] = e2o + k2 * ((1 - p1_is_winner) - elo_expected(e2o, e1o))
        overall_n[p1], overall_n[p2] = n1o + 1, n2o + 1

        ks1, ks2 = k_factor(n1s), k_factor(n2s)
        surface_elo[(p1, surface)] = e1s + ks1 * (p1_is_winner - elo_expected(e1s, e2s))
        surface_elo[(p2, surface)] = e2s + ks2 * ((1 - p1_is_winner) - elo_expected(e2s, e1s))
        surface_n[(p1, surface)], surface_n[(p2, surface)] = n1s + 1, n2s + 1

        if is_incomplete:
            continue
        sets = []
        for token in (score or "").split():
            m = SET_RE.match(token)
            if m:
                sets.append((int(m.group(1)), int(m.group(2))))
        if not sets:
            continue
        n_sets = len(sets)
        # Combined (both players') games per set -- matches the validated
        # gate's feature exactly: total_games_champion_gate_a.py stores
        # this SAME combined-match value under BOTH players' own history
        # (their personal "what do my matches tend to look like" signal),
        # then AVERAGES the two players' combined-history estimates when
        # predicting. An earlier version of this function stored each
        # player's OWN games won (not the match's combined total) and
        # then also divided by 2 when predicting -- a different, wrong
        # feature caught by a live predicted_mean of 15.18 games against
        # a real market line of 40.0 (physically impossible for a Bo5
        # US Open match) during dry-run testing. Fixed to match the gate.
        combined_games_per_set = sum(a + b for a, b in sets) / n_sets
        total_games_hist[winner_id].append((combined_games_per_set, n_sets, surface, best_of))
        total_games_hist[loser_id].append((combined_games_per_set, n_sets, surface, best_of))

    tourney_surface = {t: max(v, key=v.get) for t, v in tourney_surface_votes.items()}

    return {
        "overall_elo": overall_elo, "overall_n": overall_n,
        "surface_elo": surface_elo, "surface_n": surface_n,
        "total_games_hist": total_games_hist,
        "name_index": name_index,
        "tourney_surface": tourney_surface,
    }


def resolve_surface(state, tourney_name):
    norm = normalize_name(tourney_name)
    for slam, surf in GRAND_SLAM_SURFACE.items():
        if slam in norm:
            return surf, True
    if norm in state["tourney_surface"]:
        return state["tourney_surface"][norm], True
    return "Hard", False


def resolve_player(state, espn_name):
    norm = normalize_name(espn_name)
    if norm in state["name_index"]:
        return state["name_index"][norm]
    # fall back: last-token (surname) match if unambiguous
    surname = norm.split()[-1] if norm else ""
    candidates = {pid for n, pid in state["name_index"].items() if n.split()[-1:] == [surname]}
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def combined_elo(state, pid, surface):
    n_overall = state["overall_n"].get(pid, 0)
    e_overall = state["overall_elo"].get(pid, INITIAL_ELO)
    n_surf = state["surface_n"].get((pid, surface), 0)
    e_surf = state["surface_elo"].get((pid, surface), INITIAL_ELO)
    if n_surf >= MIN_SURFACE_MATCHES_FOR_BLEND:
        blend_w = min(n_surf / (n_surf + 15), 0.6)
        return blend_w * e_surf + (1 - blend_w) * e_overall, n_overall
    return e_overall, n_overall


def total_games_prediction(state, pid, surface, best_of):
    hist = state["total_games_hist"].get(pid, [])
    if len(hist) < MIN_PRIOR_MATCHES_TOTAL_GAMES:
        return None
    overall_gps = [g for g, _, _, _ in hist]
    surf_gps = [g for g, _, sf, _ in hist if sf == surface]
    bo_sets = [s for _, s, _, bo in hist if bo == best_of]
    overall_sets = [s for _, s, _, _ in hist]

    own_gps, _ = recency_weighted_mean(overall_gps)
    surf_gps_mean, surf_n = recency_weighted_mean(surf_gps)
    own_sets_mean, _ = recency_weighted_mean(overall_sets)
    bo_sets_mean, bo_n = recency_weighted_mean(bo_sets)

    if surf_n >= MIN_SURFACE_MATCHES_FOR_BLEND and surf_gps_mean is not None:
        blend_w = min(surf_n / (surf_n + 10), 0.7)
        combined_gps = blend_w * surf_gps_mean + (1 - blend_w) * own_gps
    else:
        combined_gps = own_gps
    expected_sets = bo_sets_mean if bo_n >= MIN_SURFACE_MATCHES_FOR_BLEND else own_sets_mean
    if combined_gps is None or expected_sets is None:
        return None
    return combined_gps * expected_sets


def fetch_espn_schedule(tour, date_str):
    url = f"{ESPN_BASE}/{tour}/scoreboard"
    r = requests.get(url, params={"dates": date_str}, timeout=20)
    r.raise_for_status()
    return r.json()


SINGLES_GROUPING = {"atp": "men's singles", "wta": "women's singles"}


def extract_actionable_matches(espn_json, tour):
    """ESPN's atp/wta scoreboard endpoints both return the FULL shared
    bracket for combined events like majors (Men's Singles, Women's
    Singles, doubles, mixed all appear under BOTH the atp and wta URLs)
    -- confirmed empirically: hitting the wta endpoint during the US
    Open returned the men's singles matches too. Must filter by
    grouping, not just by which endpoint was queried, or ATP and WTA
    matches end up duplicated against each other."""
    wanted_grouping = SINGLES_GROUPING[tour]
    out = []
    for event in espn_json.get("events", []):
        tourney_name = event.get("name") or event.get("shortName") or ""
        for grouping in event.get("groupings", []):
            slug = (grouping.get("grouping", {}).get("displayName") or "").strip().lower()
            if slug != wanted_grouping:
                continue
            for comp in grouping.get("competitions", []):
                state = comp.get("status", {}).get("type", {}).get("state")
                if state not in ("pre", "in"):
                    continue
                competitors = comp.get("competitors", [])
                if len(competitors) != 2:
                    continue
                names = [c.get("athlete", {}).get("displayName") for c in competitors]
                if not all(names) or "TBD" in names:
                    continue
                periods = comp.get("format", {}).get("regulation", {}).get("periods")
                out.append({
                    "tour": tour,
                    "tourney_name": tourney_name,
                    "competition_id": comp.get("id"),
                    "start_time": comp.get("date"),
                    "p1_name": names[0], "p2_name": names[1],
                    "best_of": periods or 3,
                    "state": state,
                })
    return out


def fetch_totals_odds(the_odds_api_key):
    """Returns list of {sport_key, home, away, line, over_price, under_price, book}
    for every currently-active tennis tournament's totals market."""
    if not the_odds_api_key:
        return [], "no THE_ODDS_API_KEY configured"
    r = requests.get(f"{THE_ODDS_API_BASE}/sports/", params={"apiKey": the_odds_api_key, "all": "true"}, timeout=20)
    r.raise_for_status()
    sports = r.json()
    tennis_active = [s for s in sports if "tennis" in s.get("key", "").lower() and s.get("active")]
    results = []
    for s in tennis_active:
        key = s["key"]
        try:
            r2 = requests.get(f"{THE_ODDS_API_BASE}/sports/{key}/odds/",
                               params={"apiKey": the_odds_api_key, "regions": "us", "markets": "totals"},
                               timeout=20)
            r2.raise_for_status()
        except Exception as e:
            continue
        for event in r2.json():
            home, away = event.get("home_team"), event.get("away_team")
            for bk in event.get("bookmakers", []):
                for m in bk.get("markets", []):
                    if m.get("key") != "totals":
                        continue
                    outcomes = {o["name"]: o for o in m.get("outcomes", [])}
                    over = outcomes.get("Over")
                    under = outcomes.get("Under")
                    if not over or not under:
                        continue
                    results.append({
                        "sport_key": key, "home": home, "away": away,
                        "line": over.get("point"),
                        "over_price": over.get("price"), "under_price": under.get("price"),
                        "book": bk.get("key"),
                    })
                    break
            if any(x["home"] == home and x["away"] == away for x in results):
                break
    return results, None


def simulate_over_prob(predicted_mean, line, sims=8000, rng=None):
    rng = rng or np.random.default_rng()
    sample = rng.normal(predicted_mean, predicted_mean * 0.16, sims)
    sample = np.clip(sample, 8, 65)
    return float(np.mean(sample > line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_PATH_DEFAULT)
    ap.add_argument("--date", default=None, help="YYYYMMDD, default today (UTC)")
    ap.add_argument("--tours", nargs="+", default=["atp"], choices=["atp", "wta"],
                     help="ATP only by default. A real WTA data source exists and is wired "
                          "into the foundation script, but every one of the 6 markets tested "
                          "(moneyline/total_games/set_betting/games_spread/total_aces/"
                          "double_faults) failed its backtest on real WTA data -- see the "
                          "tennis_*_gate_report_wta.json files. Passing --tours wta will "
                          "build real WTA ratings and produce real picks, just from models "
                          "that failed their own validation -- not recommended.")
    ap.add_argument("--odds-api-key", default=None)
    args = ap.parse_args()

    date_str = args.date or dt.datetime.utcnow().strftime("%Y%m%d")
    import os
    odds_key = args.odds_api_key or os.environ.get("THE_ODDS_API_KEY") or os.environ.get("ODDS_API_KEY") or ""

    if not args.db.exists():
        print(f"no db at {args.db} -- writing empty predictions doc")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
            "builder": BUILDER_VERSION, "markets": {}, "picks": [],
            "note": f"no database at {args.db}",
        }, indent=2))
        return 0

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    print("replaying full match history to build Elo + total_games state...")
    state = build_state(con)
    con.close()
    print(f"state built: {len(state['overall_elo'])} rated players, "
          f"{len(state['name_index'])} names indexed")

    odds, odds_err = fetch_totals_odds(odds_key)
    print(f"fetched {len(odds)} real totals-market lines"
          + (f" ({odds_err})" if odds_err else ""))

    picks = []
    n_matches_seen = n_unmatched_players = n_ineligible = 0
    n_total_games_picks = n_set_betting_picks = n_lines_matched = 0
    rng = np.random.default_rng()

    for tour in args.tours:
        try:
            espn_json = fetch_espn_schedule(tour, date_str)
        except Exception as e:
            print(f"  {tour}: ESPN fetch failed: {e}")
            continue
        matches = extract_actionable_matches(espn_json, tour)
        print(f"  {tour}: {len(matches)} actionable (not-yet-final) matches today")
        for m in matches:
            n_matches_seen += 1
            pid1 = resolve_player(state, m["p1_name"])
            pid2 = resolve_player(state, m["p2_name"])
            if not pid1 or not pid2:
                n_unmatched_players += 1
                continue
            surface, surface_confirmed = resolve_surface(state, m["tourney_name"])
            best_of = m["best_of"]
            races_to = best_of // 2 + 1

            c1, n1 = combined_elo(state, pid1, surface)
            c2, n2 = combined_elo(state, pid2, surface)
            if n1 < MIN_PRIOR_MATCHES or n2 < MIN_PRIOR_MATCHES:
                n_ineligible += 1
                continue
            p1_win_prob = elo_expected(c1, c2)
            q = invert_to_set_prob(p1_win_prob, races_to)

            base_pick = {
                "tour": tour, "tourney": m["tourney_name"], "surface": surface,
                "surface_confirmed": surface_confirmed, "best_of": best_of,
                "start_time_utc": m["start_time"],
                "player1": m["p1_name"], "player1_id": pid1,
                "player2": m["p2_name"], "player2_id": pid2,
            }

            # -- set_betting: unagraded projection, always attempted if eligible --
            dist = outcome_distribution(q, races_to)
            top_cat = max(dist, key=dist.get)
            side, w, l = top_cat
            winner_label = m["p1_name"] if side == "P1" else m["p2_name"]
            picks.append({
                **base_pick,
                "market": "set_betting",
                "pick": f"{winner_label} {w}-{l}",
                "model_prob": round(dist[top_cat], 4),
                "unagraded": True,
                "note": "no real correct-score market found on any odds provider checked -- model projection only, not graded against a market line",
            })
            n_set_betting_picks += 1

            # -- total_games: needs both real odds AND total_games eligibility --
            # Real bug caught in production (2026-09-08, Michelsen vs
            # Tiafoe): once a match goes "in progress", The Odds API's
            # totals market silently becomes a LIVE/in-play line (games
            # already played + the rest), not a pre-match line -- but
            # predicted_mean here is always a pre-match estimate. Feeding
            # a live line into a stale pre-match prediction produced a
            # "100% confidence" pick that then lost. total_games is a
            # graded pick with a real confidence claim (unlike
            # set_betting's disclosed unagraded projection), so it must
            # only ever be generated pre-match.
            if m["state"] != "pre":
                continue
            tg1 = total_games_prediction(state, pid1, surface, best_of)
            tg2 = total_games_prediction(state, pid2, surface, best_of)
            if tg1 is None or tg2 is None:
                n_ineligible += 1
                continue
            predicted_mean = (tg1 + tg2) / 2.0

            line_match = None
            for o in odds:
                names_norm = {normalize_name(o["home"]), normalize_name(o["away"])}
                if normalize_name(m["p1_name"]) in names_norm and normalize_name(m["p2_name"]) in names_norm:
                    line_match = o
                    break
            if not line_match:
                continue
            n_lines_matched += 1
            line = line_match["line"]
            over_prob = simulate_over_prob(predicted_mean, line, rng=rng)
            side_pick = "OVER" if over_prob >= 0.5 else "UNDER"
            model_prob = over_prob if side_pick == "OVER" else 1 - over_prob
            picks.append({
                **base_pick,
                "market": "total_games",
                "line": line,
                "pick": f"{side_pick} {line}",
                "model_prob": round(model_prob, 4),
                "predicted_mean": round(predicted_mean, 2),
                "odds": line_match["over_price"] if side_pick == "OVER" else line_match["under_price"],
                "book": line_match["book"],
            })
            n_total_games_picks += 1

    doc = {
        "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "builder": BUILDER_VERSION,
        "design": (
            "total_games graded against real The-Odds-API totals lines "
            "(champion-gated, 14.4% MAE improvement over naive); "
            "set_betting is an unagraded model projection only (champion-"
            "gated at ~40% vs 30% naive, but no real correct-score market "
            "exists to grade it against). total_aces, double_faults, "
            "moneyline, and games_spread all failed their backtests and "
            "are not served."
        ),
        "markets": {
            "total_games": {"eligible": n_total_games_picks, "real_lines_matched": n_lines_matched},
            "set_betting": {"eligible": n_set_betting_picks, "unagraded": True},
        },
        "note": (
            f"{n_matches_seen} live matches seen, {n_unmatched_players} skipped "
            f"(player name not matched to history), {n_ineligible} skipped "
            f"(insufficient rating history)"
        ),
        "picks": picks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {len(picks)} picks ({n_total_games_picks} total_games, "
          f"{n_set_betting_picks} set_betting) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
