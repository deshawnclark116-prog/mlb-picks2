"""Live-only, audited schedule acquisition. No fitted model or frozen CAS code changes."""
import csv
import hashlib
import io
import urllib.error
from datetime import datetime, timedelta, timezone

URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
FIELDS = ('game_id', 'season', 'game_type', 'week', 'gameday', 'gametime', 'away_team', 'home_team',
          'home_rest', 'away_rest', 'home_coach', 'away_coach')


def provider_url(name):
    import nfl_phase1d_cas as CAS
    return URL if name == 'games.csv' else CAS.provider_url(name)


def sanitize(raw, now=None):
    """Only consumed schedule metadata; retain historical completion BOOLEAN for loader compatibility.

    The frozen loader tests nonempty `result`, never its numeric value. Future/unfinished
    games always have blank result; no score, QB starter, odds, weather or result value survives.
    """
    import nfl_phase1_data as P1
    now = now or datetime.now(timezone.utc)
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    if not rows or not set(FIELDS[:8]).issubset(rows[0]):
        raise RuntimeError('DATA_UNAVAILABLE: schedule metadata schema missing required columns')
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=[*FIELDS, 'result'], lineterminator='\n'); w.writeheader()
    for r in rows:
        completed = (r.get('result') not in ('', 'NA', None) and
                     P1.kickoff_utc(r['gameday'], r['gametime']) + timedelta(hours=24) <= now)
        w.writerow({**{k: r.get(k, "") for k in FIELDS}, 'result': 'COMPLETED' if completed else ''})
    return out.getvalue().encode(), hashlib.sha256(raw).hexdigest(), 'live_schedule_metadata_v1+historical_completion_boolean'


def fetch():
    import nfl_phase1e_live as LIVE
    try:
        raw, lm, final = LIVE.http_get(URL)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'SCHEDULE_FETCH_{e.code}: {URL}') from e
    if final != URL:
        raise RuntimeError(f'DATA_UNAVAILABLE: unexpected schedule origin {final}')
    # Validate before any usable probe is logged.
    sanitize(raw)
    return raw, lm
