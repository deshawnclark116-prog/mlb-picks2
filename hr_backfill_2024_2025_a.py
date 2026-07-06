#!/usr/bin/env python3
"""
HR_BACKFILL_2024_2025_A

Orchestrates existing scripts for historical backfill.

Requires these files beside this script:
- hr_dataset_builder_a.py
- hr_sqlite_foundation_a.py
- hr_statcast_bbe_loader_a.py
- hr_sqlite_feature_builder_b.py

Dry run batter-games only:
  python hr_backfill_2024_2025_a.py --start-date 2025-04-01 --end-date 2025-04-07 --skip-bbe --skip-features

Small BBE test:
  python hr_backfill_2024_2025_a.py --start-date 2025-04-01 --end-date 2025-04-03

Full backfill:
  python hr_backfill_2024_2025_a.py --start-date 2024-03-01 --end-date 2025-10-15 --max-games 6000 --max-batter-rows 120000
"""
import argparse, subprocess, sys

def run(cmd):
    print("\nRUN:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(f"FAILED {p.returncode}: {' '.join(cmd)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--max-games", type=int, default=9999)
    ap.add_argument("--max-batter-rows", type=int, default=200000)
    ap.add_argument("--csv", default="/data/hr_model/hr_batter_game_dataset.csv")
    ap.add_argument("--windows", default="7,15,30,60")
    ap.add_argument("--bbe-start-date", default=None, help="Optional earlier BBE lookback start. Defaults to start-date.")
    ap.add_argument("--skip-batter-games", action="store_true")
    ap.add_argument("--skip-bbe", action="store_true")
    ap.add_argument("--skip-features", action="store_true")
    ap.add_argument("--chunk-days", type=int, default=1)
    ap.add_argument("--max-lines", type=int, default=50000)
    args = ap.parse_args()

    py = sys.executable
    print("HR_BACKFILL_2024_2025_A")
    print("=======================")
    print(f"range: {args.start_date} to {args.end_date}")
    print(f"csv: {args.csv}")
    print(f"windows: {args.windows}")

    if not args.skip_batter_games:
        run([
            py, "hr_dataset_builder_a.py",
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--max-games", str(args.max_games),
            "--max-batter-rows", str(args.max_batter_rows),
            "--no-statcast",
            "--output", args.csv,
        ])
        run([py, "hr_sqlite_foundation_a.py", "--import-csv", args.csv])

    if not args.skip_bbe:
        bbe_start = args.bbe_start_date or args.start_date
        run([
            py, "hr_statcast_bbe_loader_a.py",
            "--start-date", bbe_start,
            "--end-date", args.end_date,
            "--chunk-days", str(args.chunk_days),
            "--max-lines", str(args.max_lines),
        ])

    if not args.skip_features:
        run([py, "hr_sqlite_feature_builder_b.py", "--windows", args.windows, "--export-csv"])

    print("\nDONE HR_BACKFILL_2024_2025_A")

if __name__ == "__main__":
    main()
