"""Research-only reconciliation; never writes the serving database or board."""
import argparse
import json
import re
import sqlite3
from pathlib import Path
from cfb_espn_live_foundation_a import BASE, get

def parse_team_stats(block):
    stats = {s.get("name"): s.get("displayValue") for s in block.get("statistics", [])}
    passing = re.fullmatch(r"(\d+)[-/](\d+)", str(stats.get("completionAttempts", "")))
    rushing = re.fullmatch(r"\d+", str(stats.get("rushingAttempts", "")))
    if not passing or not rushing:
        return None
    completions, attempts = map(int, passing.groups())
    carries = int(rushing.group())
    if completions > attempts or attempts + carries == 0:
        return None
    return {"pass_attempts": attempts, "carries": carries, "completions": completions}

def reconcile(box, player_totals):
    rows = []
    for block in (box.get("boxscore") or {}).get("teams", []):
        team = block.get("team") or {}
        name = team.get("displayName")
        official = parse_team_stats(block)
        observed = player_totals.get(name)
        rows.append({"team": name, "team_id": team.get("id"),
                     "official": official, "player_sum": observed,
                     "status": "missing_team_stats" if official is None else
                     "missing_player_rows" if observed is None else
                     "match" if all(official[k] == observed[k] for k in ("pass_attempts", "carries")) else "mismatch"})
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="cfb_models/cfb_model.sqlite")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    con = sqlite3.connect(f"file:{Path(args.db).resolve()}?mode=ro", uri=True)
    games = con.execute("SELECT game_id FROM games WHERE season=? AND week<? AND home_points IS NOT NULL AND away_points IS NOT NULL ORDER BY week, game_id", (args.season,args.week)).fetchall()
    results = []
    for (gid,) in games:
        sums = {team: {"pass_attempts": a or 0, "carries": c or 0} for team,a,c in con.execute("SELECT team,SUM(pass_attempts),SUM(carries) FROM player_games WHERE season=? AND game_id=? GROUP BY team",(args.season,gid))}
        try:
            box = get(f"{BASE}/summary?event={gid}")
            rows = reconcile(box or {}, sums)
            results.append({"game_id": gid, "rows": rows, "source_missing": len(rows)!=2})
        except Exception as exc:
            results.append({"game_id": gid, "rows": [], "source_missing": True, "error": str(exc)})
    con.close()
    Path(args.out).write_text(json.dumps({"season":args.season,"target_week":args.week,"research_only":True,"games":results},indent=2)+"\n")

if __name__ == "__main__":
    main()
