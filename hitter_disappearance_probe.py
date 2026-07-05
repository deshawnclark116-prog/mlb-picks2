#!/usr/bin/env python3
"""
hitter_disappearance_probe.py
Standalone diagnostic for Prop Edge MLB API hitter pick disappearance.

Run from the same directory as api.py on Render:
    python hitter_disappearance_probe.py

Optional:
    python hitter_disappearance_probe.py --samples 25
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def safe_len(x):
    try:
        return len(x or [])
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()

    try:
        import api
    except Exception as e:
        print("IMPORT_API_ERROR")
        print(repr(e))
        raise

    print("=" * 80)
    print("HITTER DISAPPEARANCE PROBE")
    print("=" * 80)
    print(f"api_version: {getattr(api, 'VERSION', 'unknown')}")
    try:
        today = api.today_et().isoformat()
        print(f"today_et: {today}")
        print(f"now_et: {api.now_et().isoformat()}")
    except Exception as e:
        today = "unknown"
        print(f"date_error: {e!r}")

    pred_dir = Path(getattr(api, 'PRED_DIR', '/data/predictions'))
    print(f"pred_dir: {pred_dir}")

    print("\nLOADING GAME SLATE")
    try:
        games = api.todays_games()
    except Exception as e:
        print("todays_games_error:", repr(e))
        games = []

    pregame_games = []
    started_games = []
    for g in games:
        try:
            if api._is_pregame_game(g):
                pregame_games.append(g)
            else:
                started_games.append(g)
        except Exception:
            pregame_games.append(g)

    print(f"games_total: {len(games)}")
    print(f"pregame_games: {len(pregame_games)}")
    print(f"started_or_final_games: {len(started_games)}")

    print("\nLOADING PROP TABLES")
    try:
        over_under, thresholds, book_of = api.fetch_propline_props()
        print(f"over_under_keys: {len(over_under)}")
        print(f"threshold_keys: {len(thresholds)}")
        ou_by_market = Counter(k[1] for k in over_under.keys() if isinstance(k, tuple) and len(k) >= 2)
        print("over_under_by_market:", dict(ou_by_market))
    except Exception as e:
        print("fetch_propline_props_error:", repr(e))
        over_under, thresholds, book_of = {}, {}, {}

    print("\nSCANNING HITTER PIPELINE")
    total_home_lineup = 0
    total_away_lineup = 0
    total_lineup_batters = 0
    feature_rows = 0
    feature_missing = 0
    raw_hitter_candidates = []
    raw_by_prop = Counter()
    hr_fires = 0
    game_rows = []
    errors = []

    for game in pregame_games:
        gid = game.get("game_id")
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        try:
            lineup = api.get_confirmed_lineup(gid) or {}
        except Exception as e:
            lineup = {}
            errors.append({"game_id": gid, "stage": "get_confirmed_lineup", "error": repr(e)})

        home_lineup = lineup.get("home", []) or []
        away_lineup = lineup.get("away", []) or []
        total_home_lineup += len(home_lineup)
        total_away_lineup += len(away_lineup)
        total_lineup_batters += len(home_lineup) + len(away_lineup)

        game_raw_count = 0
        game_feature_count = 0
        game_feature_missing = 0

        for tside in ("home", "away"):
            team_name = home_team if tside == "home" else away_team
            opp = away_team if tside == "home" else home_team
            opp_pitcher = game.get("away_pitcher") if tside == "home" else game.get("home_pitcher")
            lineup_ids = home_lineup if tside == "home" else away_lineup

            for spot, pid in enumerate(lineup_ids, start=1):
                try:
                    pdata = api.get(f"{api.MLB}/people/{pid}")
                    name = pdata.get("people", [{}])[0].get("fullName", str(pid))
                except Exception:
                    name = str(pid)

                try:
                    base = api.batter_feature_row(pid)
                except Exception as e:
                    base = None
                    errors.append({"game_id": gid, "player_id": pid, "stage": "batter_feature_row", "error": repr(e)})

                if base:
                    feature_rows += 1
                    game_feature_count += 1
                    try:
                        base["batting_order"] = spot
                        picks = api.build_batter_prop_picks(
                            name, team_name, opp, gid, base,
                            over_under, thresholds, book_of,
                            batter_id=pid, pitcher_id=opp_pitcher,
                            lineup_spot=spot,
                        ) or []
                        raw_hitter_candidates.extend(picks)
                        game_raw_count += len(picks)
                        for p in picks:
                            raw_by_prop[p.get("prop_type", "unknown")] += 1
                    except Exception as e:
                        errors.append({"game_id": gid, "player_id": pid, "stage": "build_batter_prop_picks", "error": repr(e)})
                else:
                    feature_missing += 1
                    game_feature_missing += 1

                try:
                    hr_pick = api.build_hr_pick(
                        name, team_name, opp, gid,
                        pid, opp_pitcher,
                        over_under, book_of,
                        lineup_spot=spot,
                    )
                    if hr_pick:
                        hr_fires += 1
                        try:
                            hr_pick["hr_official_quality_ok"] = api._hr_official_quality_ok(hr_pick)
                        except Exception:
                            pass
                        raw_hitter_candidates.append(hr_pick)
                        raw_by_prop[hr_pick.get("prop_type", "unknown")] += 1
                        game_raw_count += 1
                except Exception as e:
                    errors.append({"game_id": gid, "player_id": pid, "stage": "build_hr_pick", "error": repr(e)})

        game_rows.append({
            "game_id": gid,
            "away": away_team,
            "home": home_team,
            "away_pitcher": game.get("away_pitcher"),
            "home_pitcher": game.get("home_pitcher"),
            "away_lineup_count": len(away_lineup),
            "home_lineup_count": len(home_lineup),
            "feature_rows": game_feature_count,
            "feature_missing": game_feature_missing,
            "raw_hitter_candidates": game_raw_count,
        })

    print(f"home_lineup_batters_total: {total_home_lineup}")
    print(f"away_lineup_batters_total: {total_away_lineup}")
    print(f"lineup_batters_total: {total_lineup_batters}")
    print(f"batter_feature_rows_found: {feature_rows}")
    print(f"batter_feature_rows_missing: {feature_missing}")
    print(f"raw_hitter_candidates_total: {len(raw_hitter_candidates)}")
    print("raw_hitter_candidates_by_prop:", dict(raw_by_prop))
    print(f"hr_fires: {hr_fires}")

    print("\nGOVERNOR CHECK")
    try:
        hitter_official, hitter_debug = api.govern_hitter_board(raw_hitter_candidates)
    except Exception as e:
        print("govern_hitter_board_error:", repr(e))
        hitter_official, hitter_debug = [], []

    print(f"governor_official_or_watchlist_count: {len(hitter_official)}")
    print("governor_board_status:", dict(Counter((p.get("board_status") or p.get("candidate_status") or "unknown") for p in hitter_debug)))
    print("governor_reject_reasons:", dict(Counter((p.get("reject_reason") or "none") for p in hitter_debug)))
    print("governor_official_by_prop:", dict(Counter(p.get("prop_type", "unknown") for p in hitter_official)))

    print("\nGAME LINEUP TABLE")
    for r in game_rows:
        print(json.dumps(r, ensure_ascii=False))

    print("\nCURRENT FILE CHECK")
    for filename in [
        f"predictions_{today}.json",
        f"hitter_candidates_{today}.json",
        f"pitcher_k_candidates_{today}.json",
        f"all_candidates_{today}.json",
    ]:
        path = pred_dir / filename
        if not path.exists():
            print(json.dumps({"file": filename, "exists": False}))
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(json.dumps({"file": filename, "exists": True, "json_error": repr(e)}))
            continue
        rows = data if isinstance(data, list) else data.get("candidates", []) if isinstance(data, dict) else []
        print(json.dumps({
            "file": filename,
            "exists": True,
            "rows": len(rows),
            "by_prop_type": dict(Counter((r.get("prop_type", "unknown") if isinstance(r, dict) else "bad_row") for r in rows)),
            "by_status": dict(Counter(((r.get("board_status") or r.get("candidate_status") or "none") if isinstance(r, dict) else "bad_row") for r in rows)),
        }, ensure_ascii=False))

    print("\nSAMPLE RAW HITTER CANDIDATES")
    for p in raw_hitter_candidates[:args.samples]:
        keep = {
            "player": p.get("player"),
            "team": p.get("team"),
            "prop_type": p.get("prop_type"),
            "pick": p.get("pick"),
            "projected": p.get("projected"),
            "model_prob": p.get("model_prob"),
            "line_source": p.get("line_source"),
            "bvp_flag": p.get("bvp_flag"),
            "lineup_spot": p.get("lineup_spot"),
        }
        print(json.dumps(keep, ensure_ascii=False))

    print("\nERROR SAMPLE")
    for e in errors[:args.samples]:
        print(json.dumps(e, ensure_ascii=False))

    print("\nDIAGNOSIS_HINT")
    if total_lineup_batters == 0:
        print("No confirmed lineup batters were returned. Hitter picks cannot be generated from the current live path until lineups are available or an early/projected-lineup hitter path is added.")
    elif len(raw_hitter_candidates) == 0:
        print("Lineups exist, but raw hitter candidates are zero. Check batter_feature_row/model_predict/probability thresholds/build_batter_prop_picks.")
    elif len(hitter_official) == 0:
        print("Raw hitter candidates exist, but governor rejected all of them. Check govern_hitter_board reject reasons above.")
    else:
        print("Hitter official/watchlist picks are being built by the standalone path. If /run/now still has zero hitters, the merge/write path is the bug.")


if __name__ == "__main__":
    main()
