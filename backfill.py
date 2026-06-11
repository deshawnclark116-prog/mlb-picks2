"""
backfill.py - One-time historical data puller.
Pulls a single MLB season's game logs from the free MLB Stats API
and saves them to the persistent disk at /data.

Run ONE season at a time to stay within memory/time limits:
    python backfill.py 2019
    python backfill.py 2020
    ... etc through 2026

Each run is independent and resumable - if it stops, just run it again
for that season and it picks up where it left off.
"""
import sys, json, time, datetime as dt
from pathlib import Path
import requests

MLB = "https://statsapi.mlb.com/api/v1"
DATA_DIR = Path("/data")          # the persistent disk
DATA_DIR.mkdir(exist_ok=True)

S = requests.Session()
S.headers["User-Agent"] = "prop-edge-backfill/1.0"


def get(url, **params):
    for attempt in range(4):
        try:
            r = S.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  retry {url}: {e}")
            time.sleep(2 * (attempt + 1))
    return {}


def season_dates(year):
    """MLB regular season roughly late March to early October."""
    start = dt.date(year, 3, 20)
    end = dt.date(year, 10, 5)
    today = dt.date.today()
    if end > today:
        end = today
    return start, end


def get_schedule(date_str):
    data = get(f"{MLB}/schedule", sportId=1, date=date_str)
    games = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            state = g.get("status", {}).get("abstractGameState", "")
            if state == "Final":
                games.append(str(g.get("gamePk")))
    return games


def get_boxscore(game_pk):
    return get(f"{MLB}/game/{game_pk}/boxscore")


def extract_player_lines(game_pk, date_str, box):
    """Pull every batter and pitcher stat line from one game."""
    rows = []
    teams = box.get("teams", {})
    for side in ("home", "away"):
        team = teams.get(side, {})
        team_name = team.get("team", {}).get("name", "")
        opp = "away" if side == "home" else "home"
        opp_name = teams.get(opp, {}).get("team", {}).get("name", "")
        players = team.get("players", {})
        for pid_key, pdata in players.items():
            person = pdata.get("person", {})
            pid = person.get("id")
            name = person.get("fullName", "")
            stats = pdata.get("stats", {})
            batting = stats.get("batting", {})
            pitching = stats.get("pitching", {})
            order = pdata.get("battingOrder")

            if batting and batting.get("atBats") is not None:
                rows.append({
                    "type": "batter", "game_pk": game_pk, "date": date_str,
                    "player_id": pid, "name": name, "team": team_name,
                    "opponent": opp_name,
                    "batting_order": int(order) // 100 if order else None,
                    "ab": batting.get("atBats", 0),
                    "pa": batting.get("plateAppearances", 0),
                    "h": batting.get("hits", 0),
                    "2b": batting.get("doubles", 0),
                    "3b": batting.get("triples", 0),
                    "hr": batting.get("homeRuns", 0),
                    "rbi": batting.get("rbi", 0),
                    "bb": batting.get("baseOnBalls", 0),
                    "so": batting.get("strikeOuts", 0),
                    "sb": batting.get("stolenBases", 0),
                    "tb": batting.get("totalBases", 0),
                    "runs": batting.get("runs", 0),
                })

            if pitching and pitching.get("inningsPitched") is not None:
                rows.append({
                    "type": "pitcher", "game_pk": game_pk, "date": date_str,
                    "player_id": pid, "name": name, "team": team_name,
                    "opponent": opp_name,
                    "ip": pitching.get("inningsPitched", "0.0"),
                    "bf": pitching.get("battersFaced", 0),
                    "h_allowed": pitching.get("hits", 0),
                    "er": pitching.get("earnedRuns", 0),
                    "bb_allowed": pitching.get("baseOnBalls", 0),
                    "so": pitching.get("strikeOuts", 0),
                    "hr_allowed": pitching.get("homeRuns", 0),
                    "pitches": pitching.get("numberOfPitches", 0),
                    "outs": pitching.get("outs", 0),
                })
    return rows


def backfill_season(year):
    out_path = DATA_DIR / f"season_{year}.jsonl"
    progress_path = DATA_DIR / f"season_{year}_progress.txt"

    # Resume support: skip dates already done
    done_dates = set()
    if progress_path.exists():
        done_dates = set(progress_path.read_text().splitlines())
        print(f"  Resuming - {len(done_dates)} dates already done")

    start, end = season_dates(year)
    print(f"Backfilling {year}: {start} to {end}")

    # Open in append mode so we never lose prior progress
    fout = open(out_path, "a")
    prog = open(progress_path, "a")

    d = start
    total_rows = 0
    days_done = 0
    while d <= end:
        ds = d.isoformat()
        if ds in done_dates:
            d += dt.timedelta(days=1)
            continue

        games = get_schedule(ds)
        day_rows = 0
        for gpk in games:
            box = get_boxscore(gpk)
            if not box:
                continue
            rows = extract_player_lines(gpk, ds, box)
            for r in rows:
                fout.write(json.dumps(r) + "\n")
            day_rows += len(rows)
            time.sleep(0.3)   # be gentle on the API

        fout.flush()
        prog.write(ds + "\n")
        prog.flush()
        total_rows += day_rows
        days_done += 1
        if games:
            print(f"  {ds}: {len(games)} games, {day_rows} player lines")

        d += dt.timedelta(days=1)

    fout.close()
    prog.close()
    print(f"\nDone {year}: {days_done} new days, {total_rows} new player lines")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backfill.py YEAR")
        print("Example: python backfill.py 2019")
        sys.exit(1)
    year = int(sys.argv[1])
    backfill_season(year)
