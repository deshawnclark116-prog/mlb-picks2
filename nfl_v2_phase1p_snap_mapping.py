#!/usr/bin/env python3
"""Phase1P-DATA snap-count PFR -> GSIS mapping audit. Exact id rules only; no names, no outcomes, no model.

Writes the identity source audit, the mapping results with the Phase1Q gate decision, and the deterministic unmapped-row forensics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import nfl_v2_phase1o_role_sources as O
import nfl_v2_phase1p_identity_sources as I

ART = I.ART
THRESHOLD = 0.95
CONFLICT_BLOCK = 0.005
REASONS = ('PFR_ID_MISSING', 'NO_CANONICAL_ID_RECORD', 'AMBIGUOUS_PFR_TO_GSIS', 'ROSTER_CROSSWALK_MISSING', 'SOURCE_CONFLICT', 'OTHER')
POSITIONS = I.PRIMARY_POSITIONS + I.DIAGNOSTIC_POSITIONS
CODE_FILES = ('nfl_v2_phase1p_identity_sources.py', 'nfl_v2_phase1p_snap_mapping.py')
DECISIONS = ('PHASE1Q_SNAP_CAN_OPEN', 'PHASE1Q_SNAP_REMAINS_BLOCKED_ID_MAPPING', 'PHASE1Q_SNAP_BLOCKED_SOURCE_INTEGRITY')


def rnd(v, nd=6):
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, dict):
        return {k: rnd(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rnd(x, nd) for x in v]
    return v


def resolve(pfr, canon, roster, roster_blank):
    """Exact, name-free resolution of one snap-source PFR id. Returns (gsis|None, rule, reason|None, candidates)."""
    if pfr in I.BLANKS:
        return None, None, 'PFR_ID_MISSING', ()
    c, r = canon.get(pfr, ()), roster.get(pfr, ())
    cands = tuple(sorted(set(c) | set(r)))
    if len(c) > 1 or len(r) > 1:
        return None, None, 'AMBIGUOUS_PFR_TO_GSIS', cands
    if c and r:
        return (c[0], 'AGREE_CANONICAL_ROSTER', None, cands) if c[0] == r[0] else (None, None, 'SOURCE_CONFLICT', cands)
    if c:
        return c[0], 'CANONICAL_ONLY', None, cands
    if r:
        return r[0], 'ROSTER_ONLY', None, cands
    return None, None, ('ROSTER_CROSSWALK_MISSING' if pfr in roster_blank else 'NO_CANONICAL_ID_RECORD'), cands


def classify(rows, canon, roster, roster_blank):
    out = []
    for r in rows:
        gsis, rule, reason, cands = resolve(r['pfr_player_id'], canon, roster, roster_blank)
        out.append({'row': r, 'gsis': gsis, 'rule': rule, 'reason': reason, 'candidates': cands})
    seen = defaultdict(set)                                    # many-to-one inside one team-week
    for x in out:
        if x['gsis']:
            seen[(x['row']['season'], x['row']['week'], x['row']['team'], x['gsis'])].add(x['row']['pfr_player_id'])
    for x in out:
        r = x['row']
        if x['gsis'] and len(seen[(r['season'], r['week'], r['team'], x['gsis'])]) > 1:
            x.update(gsis=None, rule=None, reason='SOURCE_CONFLICT')
    return out


def summarize(items, roster_gsis=None):
    n = len(items)
    mapped = [x for x in items if x['gsis']]
    unmapped = [x for x in items if not x['gsis']]
    pfr_ids = {x['row']['pfr_player_id'] for x in items if x['row']['pfr_player_id'] not in I.BLANKS}
    g2p = defaultdict(set)
    for x in mapped:
        g2p[x['gsis']].add(x['row']['pfr_player_id'])
    out = {'rows': n, 'mapped': len(mapped), 'unmapped': len(unmapped), 'ambiguous': sum(1 for x in unmapped if x['reason'] == 'AMBIGUOUS_PFR_TO_GSIS'),
           'mapping_share': len(mapped) / n if n else None, 'unique_pfr_ids': len(pfr_ids), 'unique_gsis_ids': len(g2p),
           'many_to_one_gsis_with_multiple_pfr_ids': sum(1 for v in g2p.values() if len(v) > 1),
           'rule_counts': dict(sorted(Counter(x['rule'] for x in mapped).items())), 'reason_counts': {k: sum(1 for x in unmapped if x['reason'] == k) for k in REASONS}}
    if roster_gsis is not None:
        out['mapped_rows_with_gsis_on_a_pinned_roster'] = sum(1 for x in mapped if x['gsis'] in roster_gsis[int(x['row']['season'])])
    return out


def decide(share, conflict_rate, other):
    ok = conflict_rate <= CONFLICT_BLOCK and other == 0
    if not ok:
        return 'PHASE1Q_SNAP_BLOCKED_SOURCE_INTEGRITY', ok
    return ('PHASE1Q_SNAP_CAN_OPEN' if share >= THRESHOLD else 'PHASE1Q_SNAP_REMAINS_BLOCKED_ID_MAPPING'), ok


def run(data_dir):
    I.verify_extract()
    O.verify(data_dir, I.YEARS)
    canon, canon_g2p, _ = I.canonical_map()
    roster, roster_blank = I.roster_crosswalk(data_dir)
    roster_gsis = I.roster_gsis_by_season(data_dir)
    rows = list(I.snap_rows(data_dir))
    items = classify([r for r in rows if r['position'] in POSITIONS], canon, roster, roster_blank)
    primary = [x for x in items if x['row']['position'] in I.PRIMARY_POSITIONS]
    by_season = {str(y): summarize([x for x in primary if int(x['row']['season']) == y], roster_gsis) for y in I.YEARS}
    by_pos = {pos: {str(y): summarize([x for x in items if x['row']['position'] == pos and int(x['row']['season']) == y]) for y in I.YEARS} for pos in POSITIONS}
    for pos in POSITIONS:
        by_pos[pos]['combined'] = summarize([x for x in items if x['row']['position'] == pos])
    combined = summarize(primary, roster_gsis)
    # the superseded Phase1O rule (roster crosswalk only, QB included) and the same rule without QB, for the before/after comparison
    old_map, _ = O.pfr_to_gsis(data_dir, I.YEARS)
    def old_share(sel):
        sel = [x for x in items if sel(x)]
        return sum(1 for x in sel if old_map.get(x['row']['pfr_player_id'])) / len(sel)
    old = {'rule': 'Phase1O: exact roster pfr_id -> gsis only', 'share_including_qb': old_share(lambda x: True), 'share_primary_positions': old_share(lambda x: x['row']['position'] in I.PRIMARY_POSITIONS)}
    conflict_rate = combined['reason_counts']['SOURCE_CONFLICT'] / combined['rows']
    decision, integrity_ok = decide(combined['mapping_share'], conflict_rate, combined['reason_counts']['OTHER'])
    proto = I.protocol()
    audit = {'schema': 'nfl-v2-phase1p-identity-source-audit-v1', 'base_head': proto['base_head'], 'opened_seasons': list(I.YEARS), 'later_seasons_opened': False,
             'outcome_columns_read': [], 'fuzzy_or_name_matching_used': False, 'models_fitted': False,
             'identity_source': {'extract_sha256': I.sha(ART / I.EXTRACT), 'source_sha256': proto['identity_source']['source_sha256'], 'extract_rows': len(canon_g2p),
                                 'pfr_ids': len(canon), 'pfr_ids_with_multiple_gsis': sum(1 for v in canon.values() if len(v) > 1), 'gsis_ids_with_multiple_pfr': sum(1 for v in canon_g2p.values() if len(v) > 1)},
             'roster_crosswalk': {'pfr_ids': len(roster), 'pfr_ids_with_multiple_gsis': sum(1 for v in roster.values() if len(v) > 1), 'pfr_ids_on_blank_gsis_rows': len(roster_blank),
                                  'sha256': proto['crosswalk_source']['sha256']},
             'snap_source_sha256': proto['snap_source']['sha256'], 'columns_read': {'snap': list(I.SNAP_COLUMNS), 'roster': list(I.ROSTER_COLUMNS), 'extract': list(I.EXTRACT_COLUMNS)},
             'canonical_vs_roster_disagreements': sum(1 for k, v in roster.items() if k in canon and v != canon[k]),
             'canonical_vs_roster_agreements': sum(1 for k, v in roster.items() if k in canon and v == canon[k]),
             'one_to_many_pfr_to_gsis_among_snap_ids': sum(1 for p in {x['row']['pfr_player_id'] for x in items} if len(canon.get(p, ())) > 1 or len(roster.get(p, ())) > 1),
             'old_mapping': old}
    results = {'schema': 'nfl-v2-phase1p-snap-mapping-results-v1', 'population': 'regular-season snap rows with offense_pct > 0; primary positions WR,TE,RB,FB,HB (QB diagnostic only)',
               'threshold': THRESHOLD, 'old_mapping': old, 'combined_primary': combined, 'primary_by_season': by_season, 'by_position': by_pos,
               'source_conflict_rate': conflict_rate, 'integrity_ok': integrity_ok, 'decision': decision}
    unmapped = [{'season': int(x['row']['season']), 'week': int(x['row']['week']), 'team': x['row']['team'], 'pfr_id': x['row']['pfr_player_id'], 'player': x['row']['player'],
                 'position': x['row']['position'], 'offense_pct': float(x['row']['offense_pct']), 'primary_position': x['row']['position'] in I.PRIMARY_POSITIONS,
                 'candidate_gsis': list(x['candidates']), 'reason': x['reason']} for x in items if not x['gsis']]
    unmapped.sort(key=lambda r: (r['season'], r['week'], r['team'], r['pfr_id'], r['player']))
    return rnd(audit), rnd(results), unmapped


def write_unmapped(path, rows):
    with gzip.GzipFile(filename='', mode='wb', fileobj=open(path, 'wb'), mtime=0) as g:
        for r in rows:
            g.write((json.dumps(rnd(r), sort_keys=True) + '\n').encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    audit, results, unmapped = run(a.data_dir)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'phase1p_identity_source_audit.json').write_text(O.dump(audit))
    (out / 'phase1p_snap_mapping_results.json').write_text(O.dump(results))
    write_unmapped(out / 'phase1p_unmapped_snap_rows.jsonl.gz', unmapped)
    print(results['decision'], results['combined_primary']['mapping_share'])


if __name__ == '__main__':
    main()
