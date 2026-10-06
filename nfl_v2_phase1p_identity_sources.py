#!/usr/bin/env python3
"""Phase1P-DATA identity sources: the pinned nflverse player-identity extract and the pinned weekly-roster crosswalk.

Identity fields only. No outcome column of any file is ever read here. The upstream `players.csv` release asset is a mutable rolling file, so the
exact bytes were frozen once into a committed identity-only extract (gsis_id, pfr_id, display_name, position); reproduction reads only that
committed, digest-verified extract. Names are carried for forensics and are never used to match.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import gzip
import hashlib
import io
from pathlib import Path
import subprocess

import nfl_v2_phase1o_role_sources as O

ROOT = O.ROOT
ART = O.ART
PROTOCOL = 'phase1p_identity_protocol.json'
EXTRACT = 'phase1p_identity_extract.csv.gz'
EXTRACT_COLUMNS = ('gsis_id', 'pfr_id', 'display_name', 'position')
PLAYERS_URL = 'https://github.com/nflverse/nflverse-data/releases/download/players/players.csv'
YEARS = (2023, 2024)
SNAP_COLUMNS = ('season', 'week', 'game_type', 'team', 'player', 'pfr_player_id', 'position', 'offense_pct')   # identity / deployment metadata only
ROSTER_COLUMNS = ('season', 'week', 'team', 'gsis_id', 'pfr_id', 'game_type')
PRIMARY_POSITIONS = ('WR', 'TE', 'RB', 'FB', 'HB')
DIAGNOSTIC_POSITIONS = ('QB',)
BLANKS = (None, '', 'NA')
sha = O.sha


def extract_bytes(players_csv_text):
    """Deterministic identity-only extract: rows with both ids, sorted, gzip with mtime 0. Pure function of the input bytes."""
    rows = []
    for r in csv.DictReader(io.StringIO(players_csv_text)):
        if r['gsis_id'] not in BLANKS and r['pfr_id'] not in BLANKS:
            rows.append((r['gsis_id'], r['pfr_id'], r['display_name'], r['position']))
    rows.sort()
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(EXTRACT_COLUMNS)
    w.writerows(rows)
    raw = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=9) as g:
        g.write(buf.getvalue().encode('utf-8'))
    return raw.getvalue()


def build_extract(players_csv_path, out_path=None):
    data = extract_bytes(Path(players_csv_path).read_text(encoding='utf-8'))
    Path(out_path or ART / EXTRACT).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def download_players(dest):
    """Provenance check only: downloads the CURRENT upstream file. Never used for reproduction."""
    subprocess.run(['curl', '-fsSL', '--retry', '5', '--retry-all-errors', '--retry-delay', '5', '--max-time', '280', PLAYERS_URL, '-o', str(dest)], check=True)
    return sha(dest)


def protocol():
    return O.read_json(ART / PROTOCOL)


def verify_extract():
    expected = protocol()['identity_source']['extract_sha256']
    if sha(ART / EXTRACT) != expected:
        raise ValueError('Frozen identity extract changed or missing')


def canonical_map():
    """{pfr_id: sorted tuple of gsis ids} from the frozen canonical extract; {gsis_id: sorted tuple of pfr ids}; {pfr_id: display_name}."""
    verify_extract()
    p2g, g2p, names = defaultdict(set), defaultdict(set), {}
    with gzip.open(ART / EXTRACT, 'rt', encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            p2g[r['pfr_id']].add(r['gsis_id'])
            g2p[r['gsis_id']].add(r['pfr_id'])
            names[r['pfr_id']] = r['display_name']
    return ({k: tuple(sorted(v)) for k, v in p2g.items()}, {k: tuple(sorted(v)) for k, v in g2p.items()}, names)


def roster_crosswalk(directory, years=YEARS):
    """{pfr_id: sorted tuple of gsis ids} and the set of pfr ids that appear on a roster row with a blank gsis id (pinned rosters, identity columns only)."""
    O.verify(directory, years)
    p2g, blank_gsis = defaultdict(set), set()
    for year in years:
        with open(Path(directory) / O.manifest()[year]['roster']['local_name'], newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('game_type', 'REG') not in ('REG', ''):
                    continue
                pfr, gsis = row.get('pfr_id'), row.get('gsis_id')
                if pfr in BLANKS:
                    continue
                if gsis in BLANKS:
                    blank_gsis.add(pfr)
                else:
                    p2g[pfr].add(gsis)
    return {k: tuple(sorted(v)) for k, v in p2g.items()}, blank_gsis


def snap_rows(directory, years=YEARS):
    """Regular-season snap rows with offensive snaps > 0: identity / deployment metadata columns only."""
    for year in years:
        with open(Path(directory) / O.manifest()[year]['snaps']['local_name'], newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['game_type'] != 'REG' or int(O.number(row['season'])) != year:
                    continue
                if O.number(row['offense_pct']) <= 0:
                    continue
                yield {c: row[c] for c in SNAP_COLUMNS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-extract', metavar='PLAYERS_CSV', help='maintainer use: freeze the identity extract from a downloaded players.csv')
    ap.add_argument('--check-upstream', metavar='DEST', help='provenance only: download current players.csv and report whether it still has the pinned digest')
    a = ap.parse_args()
    if a.build_extract:
        print(build_extract(a.build_extract))
    if a.check_upstream:
        print(download_players(a.check_upstream))


if __name__ == '__main__':
    main()
