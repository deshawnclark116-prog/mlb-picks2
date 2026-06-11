"""
weekly_update.py - Automated weekly learning cycle.
1. Refreshes the current season's game data (pulls newest completed games)
2. Retrains both models on full history including the fresh data
3. Saves updated models to /data/models

Run by a Render Cron Job weekly. Also runnable manually:
    python weekly_update.py
"""
import sys, datetime as dt

# Reuse the exact backfill + train logic we already wrote and tested
import backfill
import train


def main():
    print("=" * 55)
    print(f"WEEKLY UPDATE  {dt.datetime.now()}")
    print("=" * 55)

    # 1. Refresh current season data (resumable - only pulls new days)
    year = dt.date.today().year
    print(f"\n[1/2] Refreshing {year} season data...")
    backfill.backfill_season(year)

    # 2. Retrain both models on full history (incremental, low-memory)
    print(f"\n[2/2] Retraining models on full history...")
    train.main()

    print("\n" + "=" * 55)
    print("WEEKLY UPDATE COMPLETE - models refreshed with latest games")
    print("=" * 55)


if __name__ == "__main__":
    main()
