#!/usr/bin/env python3
"""Immutable pregame CFB personnel and workload evidence receipts.

Captures actual source observations *before kickoff*, by player/game/market.
No edits to champion probabilities, no invented active-starter certification,
no retroactive backfill, and no sportsbook odds or bet recommendations.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA="CFB_FIRST_PREGAME_PERSONNEL_EVIDENCE_V1"
SOURCE_SCHEMA="cfb-analyst-evidence-v1"
MARKETS={"rushing_yards","receiving_yards","passing_yards","passing_touchdowns",
         "anytime_touchdowns","rushing_yards_early_season",
         "passing_touchdowns_early_season","anytime_touchdowns_early_season"}


def utc(s):
    if not isinstance(s,str) or not s:
        raise ValueError("timestamp missing")
    d=datetime.fromisoformat(s.replace("Z","+00:00"))
    if d.tzinfo is None:
        raise ValueError("timestamp without timezone")
    return d.astimezone(timezone.utc)


def immutable_key(row):
    return "|".join(str(row.get(k) or "") for k in
                    ("season","week","game_id","market","team","player_id"))


def digest(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def pregame_rows(snapshot,board,at):
    """Validate exact model generation and authentic as-of sources before appending."""
    if (snapshot.get("schema")!=SOURCE_SCHEMA or
        snapshot.get("status")!="RESEARCH_ONLY_DO_NOT_AUTO_PROMOTE" or
        snapshot.get("model_changed") is not False or
        snapshot.get("markets_book_verified") is not False or
        snapshot.get("is_current_injury_and_starter_feed_complete") is not False or
        snapshot.get("board_generated_at_utc")!=board.get("generated_at_utc") or
        snapshot.get("season")!=board.get("season") or
        snapshot.get("week")!=board.get("week")):
        raise ValueError("snapshot and immutable model generation mismatch")
    captured=utc(snapshot["evidence_captured_at_utc"])
    generated=utc(board["generated_at_utc"])
    if not generated<=captured<=at:
        raise ValueError("evidence was not captured after forecast and before publication")
    predictions={}
    for p in board.get("picks",[]):
        k=immutable_key(p)
        if k in predictions:
            raise ValueError("ambiguous model identity in current prediction board")
        predictions[k]=p
    stats=Counter()
    valid=[]
    for row in snapshot.get("player_evidence",[]):
        stats["source_rows"]+=1
        if row.get("market") not in MARKETS:
            stats["unsupported_market"]+=1
            continue
        key=immutable_key(row)
        pred=predictions.get(key)
        if pred is None:
            stats["no_exact_same_board_player_and_market"]+=1
            continue
        if (pred.get("team")!=row.get("team") or
            pred.get("opponent")!=row.get("opponent") or
            pred.get("player")!=row.get("player") or
            pred.get("pick")!=row.get("published_model_side") or
            pred.get("model_prob")!=row.get("published_model_prob")):
            stats["source_model_value_mismatch"]+=1
            continue
        try:
            kickoff=utc(row.get("kickoff_utc"))
            current_kickoff=utc(pred.get("kickoff_utc"))
        except ValueError:
            stats["unverifiable_kickoff"]+=1
            continue
        if kickoff!=current_kickoff or not captured<kickoff or not generated<kickoff or not at<kickoff:
            stats["not_genuinely_pregame"]+=1
            continue
        if row.get("calibrated_market_edge_established") is not False or (
            row.get("fully_verified_active_starter") is not False) or (
            row.get("published_line_is_sportsbook_verified") is not False):
            stats["false_status_or_market_certification"]+=1
            continue
        roster=row.get("roster") or {}
        usage=row.get("recent_usage")
        raw_checked=roster.get("roster_checked_utc")
        if raw_checked:
            try:
                checked=utc(raw_checked)
                if checked>captured or checked>=kickoff:
                    stats["post_capture_or_game_roster_source"]+=1
                    continue
            except ValueError:
                stats["bad_roster_timestamp"]+=1
                continue
        if usage is not None and usage.get("snapshot_is_not_starting_lineup_certification") is not True:
            stats["incorrect_role_certification"]+=1
            continue
        evidence={
            "schema":SCHEMA,
            "first_source_capture_utc":snapshot["evidence_captured_at_utc"],
            "model_generated_at_utc":board["generated_at_utc"],
            "season":row["season"],"week":row["week"],"game_id":row["game_id"],
            "market":row["market"],"team":row["team"],"opponent":row["opponent"],
            "player":row["player"],"player_id":row["player_id"],
            "kickoff_utc":row["kickoff_utc"],
            "original_model_probability":pred["model_prob"],
            "original_model_pick":pred["pick"],
            "original_research_line_NOT_book":pred.get("line"),
            "original_model_source":pred.get("model_source"),
            "roster_source_evidence":roster,
            "prior_team_game_usage_evidence":usage,
            "source_quality_flags":list(row.get("evidence_warnings") or []),
            "verified_active_starter":False,
            "sportsbook_line_verified":False,
            "betting_edge_validated":False,
        }
        evidence["source_receipt_sha256"]=digest(row)
        valid.append(evidence)
        stats["valid_pregame"]+=1
    return valid,dict(stats)


def append_first_seen(path,rows):
    """Never rewrite an existing snapshot; preserve very first captured evidence."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Local runner lock prevents simultaneous writers from generating duplicate
    # rows. GitHub remote reconciliation still uses its normal merge discipline.
    with path.open("a+",encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
        handle.seek(0)
        seen=set()
        for i,line in enumerate(handle,1):
            if not line.strip():continue
            try:row=json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupted immutable receipt at line {i}") from exc
            if row.get("schema")!=SCHEMA:raise ValueError("unsupported receipt in frozen ledger")
            key=immutable_key(row)
            if key in seen:raise ValueError("duplicate frozen first-seen identity")
            seen.add(key)
        new=[]
        for row in rows:
            key=immutable_key(row)
            if key not in seen:
                new.append(row)
                seen.add(key)
        for row in new:
            handle.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
    return len(new),len(seen)


def capture(snapshot,board,ledger,*,clock=None):
    now=clock or datetime.now(timezone.utc)
    if now.tzinfo is None:raise ValueError("local clock without timezone")
    rows,checks=pregame_rows(snapshot,board,now.astimezone(timezone.utc))
    added,total=append_first_seen(ledger,rows)
    return {"schema":SCHEMA,"status":"APPEND_ONLY_SOURCE_PROOF",
            "source_rows":checks.get("source_rows",0),
            "eligible_pregame":len(rows),"newly_frozen":added,
            "ledger_unique_records":total,
            "rejection_reasons":{k:v for k,v in checks.items() if k not in
                                 ("source_rows","valid_pregame")},
            "never_validates_a_betting_edge":True}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snapshot",default="docs/cfb_pregame_evidence.json")
    ap.add_argument("--board",default="docs/cfb_predictions.json")
    ap.add_argument("--ledger",default="docs/cfb_personnel_first_seen.jsonl")
    ap.add_argument("--report",default="docs/cfb_personnel_ledger_status.json")
    args=ap.parse_args()
    board=json.loads(Path(args.board).read_text())
    doc=json.loads(Path(args.snapshot).read_text())
    r=capture(doc,board,args.ledger)
    out=Path(args.report)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(r,sort_keys=True))


if __name__=="__main__":
    main()
