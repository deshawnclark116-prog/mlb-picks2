#!/usr/bin/env python3
"""Tennis match-winner-only production builder (ATP and WTA).

The original Elo/surface model is unchanged. Only original pre-match winner
probabilities are served, first-seen logged and graded from real results.
No total-games, exact-score, aces, doubles, or spread picks are published.
Do not call paid odds endpoints for any discarded tennis market.

Critical: the historic champion gate FAILED on both tours; first-seen
accuracy (ATP 148/206, WTA 190/306 as of 2026-10-18) does not establish
betting profitability, positive edge, or calibrated superiority to bookmakers.
Publish as unvalidated match-winner research forecasts, not official bets.
Historical models and pick ledger remain intact for independent audits.

ESPN pre-match schedule supplies participants; TML-Database supplies past
matches and surface-adjusted Elo. Unmatched names or insufficient history
are skipped. This builder must NEVER publish in-progress match winners.
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

from shared_odds_api_quota_guard import the_odds_api_quota_locked

DB_PATH_DEFAULT = Path("tennis_models/tennis_model.sqlite")
OUT_PATH_DEFAULT = Path("docs/tennis_predictions.json")
DOCS_DEFAULT = Path("docs")
TENNIS_ODDS_CACHE_PATH = DOCS_DEFAULT / "tennis_odds_cache.json"
TENNIS_ODDS_CACHE_TTL_MINUTES = 90


def _load_tennis_odds_cache(now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    try:
        data = json.loads(TENNIS_ODDS_CACHE_PATH.read_text(encoding="utf-8"))
        checked = dt.datetime.fromisoformat(data["captured_at_utc"].replace("Z", "+00:00"))
        if checked.tzinfo is None:
            return None
        age = (now - checked).total_seconds() / 60
        if not 0 <= age < TENNIS_ODDS_CACHE_TTL_MINUTES:
            return None
        if data.get("provider") != "the_odds_api" or not isinstance(data.get("lines"), list):
            return None
        if not all(isinstance(row, dict) and {"sport_key","home","away","line",
                       "over_price","under_price","book"} <= set(row) for row in data["lines"]):
            return None
        return data["lines"]
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def _save_tennis_odds_cache(lines, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    payload = {
        "provider": "the_odds_api",
        "captured_at_utc": now.isoformat(),
        "ttl_minutes": TENNIS_ODDS_CACHE_TTL_MINUTES,
        "lines": lines,
    }
    TENNIS_ODDS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TENNIS_ODDS_CACHE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# Append-only ledger of every pick this builder has ever produced --
# mirrors nfl_serving_builder_a.py's PICKS_LOG_PATH exactly (see that
# file's docstring). tennis_grade_record_a.py grades from this, not from
# docs/tennis_predictions.json, because that doc gets overwritten in
# place on every run (no per-day snapshot the way MLB has) and total_games
# lines are known to shift intraday (the live-line contamination bug
# fixed above) -- keying on first-seen (competition_id, market) freezes
# the pick at whatever it was the first time this match was eligible,
# which for total_games is always still pre-match given the state=="pre"
# guard, so grading never chases a moving line.
PICKS_LOG_PATH = DOCS_DEFAULT / "tennis_picks_log.jsonl"

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


def now_utc_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_logged_pick_keys(path):
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            keys.add((r.get("competition_id"), r.get("market")))
    return keys


def append_new_picks_to_log(path, keys, picks):
    new_lines = []
    for p in picks:
        k = (p["competition_id"], p["market"])
        if k in keys:
            continue
        keys.add(k)
        new_lines.append(json.dumps({**p, "logged_at": now_utc_iso()}))
    if new_lines:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write("\n".join(new_lines) + "\n")
    return len(new_lines)


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
GRAND_SLAM_NAMES = ("australian open", "roland garros", "french open", "wimbledon", "us open")


def infer_best_of(tour, tourney_name):
    """ESPN's format.regulation.periods is 5 for EVERY ATP singles match
    (confirmed live 2026-09-27: Chengdu and Hangzhou, both best-of-3 ATP
    250s, report periods=5) -- it's the max, not the real format. Taking
    it at face value modeled every regular tour match as best-of-5:
    set_betting predicted impossible "3-1"/"3-0" scores (3 of 43 hits,
    all at majors) and total_games projected 5-set totals (~40 games)
    for matches that usually finish in the low 20s. Only men's Grand
    Slam singles are best-of-5."""
    name = (tourney_name or "").lower()
    return 5 if tour == "atp" and any(gs in name for gs in GRAND_SLAM_NAMES) else 3


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
                out.append({
                    "tour": tour,
                    "tourney_name": tourney_name,
                    "competition_id": comp.get("id"),
                    "start_time": comp.get("date"),
                    "p1_name": names[0], "p2_name": names[1],
                    "best_of": infer_best_of(tour, tourney_name),
                    "state": state,
                })
    return out


def fetch_totals_odds(the_odds_api_key):
    """Returns list of {sport_key, home, away, line, over_price, under_price, book}
    for every currently-active tennis tournament's totals market."""
    if not the_odds_api_key:
        return [], "no THE_ODDS_API_KEY configured"
    if the_odds_api_quota_locked():
        return [], "shared The Odds API quota exhausted: no paid requests"
    cached = _load_tennis_odds_cache()
    if cached is not None:
        return cached, "90-minute persisted odds cache (no provider requests)"
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
        except requests.HTTPError as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            # No repeated per-tournament calls after a provider quota or
            # auth block. Never print the exception URL containing the key.
            if code in (401, 403, 429):
                return results, "provider HTTP " + str(code) + ": remaining tournaments skipped"
            continue
        except Exception:
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
    _save_tennis_odds_cache(results)
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
    ap.add_argument("--tours", nargs="+", default=["atp", "wta"], choices=["atp", "wta"],
                     help="Both by default as of 2026-09-20. A real WTA data source exists "
                          "(wired into the foundation script) but 5 of the 6 markets tested "
                          "(total_games/set_betting/games_spread/total_aces/double_faults) "
                          "failed their backtest on real WTA data -- see the tennis_*_gate_"
                          "report_wta.json files -- and stay ATP-only regardless of this flag "
                          "(gated per-market in main(), not by this list). moneyline is the one "
                          "exception: shipped on both tours by explicit product decision despite "
                          "ALSO failing its own gate on both tours (ATP holdout 63.6%, doesn't "
                          "beat naive; WTA holdout 64.2%, just under the 0.65 floor) -- see "
                          "tennis_moneyline_gate_report.json / _wta.json.")
    ap.add_argument("--odds-api-key", default=None)
    args = ap.parse_args()

    date_str = args.date or dt.datetime.utcnow().strftime("%Y%m%d")
    # Sportsbook odds are intentionally NOT requested. H2H odds/market
    # validation are separate from predicting who wins, and the former
    # total-games API calls consumed shared provider credits unnecessarily.
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
    print("replaying prior tennis results to build match-winner Elo state...")
    state = build_state(con)
    con.close()
    print(f"state built: {len(state['overall_elo'])} rated players, "
          f"{len(state['name_index'])} names indexed")

    picks = []
    n_matches_seen = n_unmatched_players = n_ineligible = n_moneyline_picks = 0
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
            # No in-play/live moneylines and no guessing an unavailable
            # player. The model's pre-match probability is frozen once.
            if m["state"] != "pre":
                continue
            p1_win_prob = elo_expected(c1, c2)
            winner_is_p1 = p1_win_prob >= 0.5
            ml_prob = p1_win_prob if winner_is_p1 else 1 - p1_win_prob
            ml_winner = m["p1_name"] if winner_is_p1 else m["p2_name"]

            # Research-only alternative: when the tournament surface was
            # NOT verified, current V1 assumes hard courts. Evaluate an
            # overall-Elo-only challenger on exactly the same pregame game,
            # rather than guessing that hard surface is correct.
            overall_p1_win = elo_expected(
                state["overall_elo"].get(pid1, INITIAL_ELO),
                state["overall_elo"].get(pid2, INITIAL_ELO))
            neutral_winner = m["p1_name"] if overall_p1_win >= .5 else m["p2_name"]
            neutral_probability = max(overall_p1_win, 1 - overall_p1_win)
            neutral_selected_probability = (
                overall_p1_win if winner_is_p1 else 1-overall_p1_win)

            pick = {
                "tour": tour, "tourney": m["tourney_name"], "surface": surface,
                "surface_confirmed": surface_confirmed, "best_of": best_of,
                "start_time_utc": m["start_time"], "player1": m["p1_name"],
                "player1_id": pid1, "player2": m["p2_name"],
                "player2_id": pid2, "competition_id": m["competition_id"],
                "espn_date": date_str, "market": "moneyline",
                "pick": f"{ml_winner} ML", "model_prob": round(ml_prob, 4),
                "surface_evidence": (
                    "HISTORICAL_TOURNAMENT_NAME_MATCH"
                    if surface_confirmed else "UNVERIFIED_SURFACE_HARD_COURT_ASSUMPTION"),
                "overall_elo_research_challenger": {
                    "pick": f"{neutral_winner} ML",
                    "model_prob": round(neutral_probability, 4),
                    "original_selected_side_prob": round(neutral_selected_probability, 4),
                    "tested_for_betting": False,
                    "promotion_status": "NOT_PROMOTED_REQUIRES_FUTURE_MATCH_VALIDATION",
                },
                "prediction_status": "UNVALIDATED_MONEYLINE_RESEARCH_ONLY",
                "model_source": "elo_surface_matchwinner_without_market_price",
                "ungraded": True,
                "note": ("Not a certified wager: no bookmaker moneyline was "
                         "captured. Historic moneyline champion gate failed "
                         "for both tours. Accuracy is not proof of betting edge."),
            }
            picks.append(pick)
            n_moneyline_picks += 1

    assert all(p["market"] == "moneyline" and p["ungraded"] is True
               for p in picks)
    doc = {
        "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "builder": BUILDER_VERSION,
        "design": ("TENNIS MONEYLINE ONLY. Historical first-seen audit: "
                   "ATP and WTA match-winner probability forecasts. "
                   "Both original champion gates failed; probabilities have "
                   "not demonstrated positive expected value vs bookmakers. "
                   "No live h2h book quote or moneyline odds were requested, "
                   "so no wager availability, value, or ROI is asserted."),
        "published_markets": ["moneyline"],
        "moneyline_market_odds_requested": False,
        "discarded_market_odds_requests": 0,
        "markets": {"moneyline": {
            "eligible": n_moneyline_picks,
            "gate_status": "failed_both_tours",
            "bettable": False,
            "source": "Elo/surface pregame only; prices unverified"}},
        "note": (f"{n_matches_seen} live matches seen, {n_unmatched_players} "
                 f"skipped (player match history identity unresolved), "
                 f"{n_ineligible} skipped (insufficient history)"),
        "picks": picks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(f"wrote {len(picks)} model-only moneyline forecasts to {args.out}; "
          "0 sportsbook provider requests")
    logged_keys = load_logged_pick_keys(PICKS_LOG_PATH)
    n_new_logged = append_new_picks_to_log(PICKS_LOG_PATH, logged_keys, picks)
    print(f"logged {n_new_logged} new picks to {PICKS_LOG_PATH} "
          f"({len(logged_keys)} total ever logged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
