#!/usr/bin/env python3
'''
PATCH_API_FOR_HITS_B1_BLIND_LOGGER_A

Safely patches the current api.py to:
1. import hits_b1_blind_live_logger_a
2. call the blind logger at the exact pregame batter-feature hook
3. expose an outcome-blind operational-status endpoint

The patch:
- makes a backup
- refuses ambiguous anchors
- refuses partial duplicate patches
- never changes existing production pick logic
'''

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


IMPORT_LINE = (
    "import hits_b1_blind_live_logger_a as hits_b1_holdout"
)

IMPORT_ANCHOR = "import lineupk\n"

HOOK_OLD = '''                if base:
                    base["batting_order"] = spot
                    pks = build_batter_prop_picks(
'''

HOOK_NEW = '''                if base:
                    base["batting_order"] = spot

                    # Frozen B1 prospective holdout logger.
                    # Outcome-blind and isolated from the production board.
                    _holdout_log_result = hits_b1_holdout.log_blind_hits_prediction(
                        game_date=today,
                        game_id=gid,
                        player_id=pid,
                        player_name=name,
                        team=team_name,
                        opponent=opp,
                        opposing_pitcher_id=opp_pitcher,
                        lineup_spot=spot,
                        base_features=base,
                    )
                    if _holdout_log_result.get("status") == "operational_error":
                        print(
                            f"  B1 HOLDOUT LOGGER ERROR: {name} "
                            f"{_holdout_log_result.get('error_type')}: "
                            f"{_holdout_log_result.get('error_message')}"
                        )

                    pks = build_batter_prop_picks(
'''

ENDPOINT_ANCHOR = '''@app.post("/run/daily")
def trigger_daily():
'''

ENDPOINT_BLOCK = '''@app.get("/debug/hits-b1-holdout-status")
def debug_hits_b1_holdout_status():
    """
    Outcome-blind operational status only.
    Never exposes hit rate, Brier, logloss, AUC, top buckets,
    official-tier performance, or any outcome-conditioned metric.
    """
    return hits_b1_holdout.blind_hits_logger_status()


'''


def count_exact(text, needle):
    return text.count(needle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api",
        default="api.py",
        help="Path to api.py (default: ./api.py)",
    )
    args = parser.parse_args()

    api_path = Path(args.api)

    if not api_path.exists():
        raise SystemExit(f"Missing api.py: {api_path}")

    text = api_path.read_text(
        encoding="utf-8",
        errors="strict",
    )

    already_import = IMPORT_LINE in text
    already_hook = (
        "hits_b1_holdout.log_blind_hits_prediction(" in text
    )
    already_endpoint = (
        '@app.get("/debug/hits-b1-holdout-status")' in text
    )

    states = [already_import, already_hook, already_endpoint]

    if all(states):
        print("api.py is already fully patched.")
        print("verdict=NO_CHANGE_ALREADY_PATCHED")
        return 0

    if any(states):
        raise SystemExit(
            "Partial prior patch detected. Refusing automatic modification.\n"
            f"import={already_import} hook={already_hook} "
            f"endpoint={already_endpoint}"
        )

    checks = {
        "import_anchor_count": count_exact(text, IMPORT_ANCHOR),
        "hook_anchor_count": count_exact(text, HOOK_OLD),
        "endpoint_anchor_count": count_exact(text, ENDPOINT_ANCHOR),
    }

    for name, count in checks.items():
        if count != 1:
            raise SystemExit(
                f"{name} expected exactly 1 match, found {count}. "
                "Refusing ambiguous patch."
            )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = api_path.with_name(
        f"{api_path.name}.pre_hits_b1_logger_{timestamp}.bak"
    )

    shutil.copy2(api_path, backup)

    text = text.replace(
        IMPORT_ANCHOR,
        IMPORT_ANCHOR + IMPORT_LINE + "\n",
        1,
    )

    text = text.replace(
        HOOK_OLD,
        HOOK_NEW,
        1,
    )

    text = text.replace(
        ENDPOINT_ANCHOR,
        ENDPOINT_BLOCK + ENDPOINT_ANCHOR,
        1,
    )

    api_path.write_text(
        text,
        encoding="utf-8",
    )

    print("PATCH_API_FOR_HITS_B1_BLIND_LOGGER_A")
    print("=====================================")
    print(f"api_path={api_path}")
    print(f"backup={backup}")
    print("import_inserted=True")
    print("live_hook_inserted=True")
    print("status_endpoint_inserted=True")
    print("production_pick_logic_changed=False")
    print("status_endpoint=/debug/hits-b1-holdout-status")
    print("verdict=API_PATCH_COMPLETE")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
