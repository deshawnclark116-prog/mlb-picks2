"""Phase2B-DATA deterministic checks: as-of semantics, append-only storage, capture windows, source acceptance, QB routing, Phase2A untouched."""
import ast
import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import subprocess

import pytest

import nfl_v2_phase2a_forward_forecast as F
import nfl_v2_phase2b_availability as B
import nfl_v2_qb_state_ingestion as Q
from nfl_v2_phase2a_ledger import append_record, read_ledger, verify_chain

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'nfl_models/nfl_player_outcome_v2'
UTC = timezone.utc
FIRST_FORECAST_COMMIT = 'b3d9142'
GAME = {'game_id': '2026_07_T01_T00', 'season': '2026', 'game_type': 'REG', 'week': '7', 'gameday': '2026-10-18', 'gametime': '13:00', 'away_team': 'T01', 'home_team': 'T00'}
KO = F.kickoff_utc('2026-10-18', '13:00')
GAMES = {(2026, 7, 'T00'): GAME, (2026, 7, 'T01'): GAME}
PID = '00-0000100'


def acceptance(tmp_path, **flags):
    srcs = [{'provider': 'nflverse_roster_weekly', 'classification': 'ACCEPTED_FORWARD_CAPTURE', 'authorized': True, 'scope': 'roster_status_only'},
            {'provider': 'nflverse_injuries', 'classification': flags.get('injuries', 'ACCEPTED_FORWARD_CAPTURE'), 'authorized': flags.get('injuries_authorized', True), 'scope': 'fixture'},
            {'provider': 'manually_supplied_authorized_records', 'classification': 'ACCEPTED_FORWARD_CAPTURE', 'authorized': True, 'scope': 'x'}]
    p = tmp_path / 'acc.json'
    p.write_text(json.dumps({'label': B.FIXTURE_LABEL, 'sources': srcs}))
    return p


def injury_row(report='Questionable', practice='Limited Participation in Practice', pid=PID, team='T00', pos='WR'):
    return {'season': '2026', 'week': '7', 'team': team, 'gsis_id': pid, 'position': pos, 'full_name': 'Fixture Player', 'report_status': report, 'practice_status': practice, 'report_primary_injury': 'Knee', 'report_secondary_injury': ''}


def ingest(tmp_path, rows, at, acc, provider='nflverse_injuries', sub='s'):
    raw, norm = tmp_path / sub / 'raw', tmp_path / sub / 'norm'
    return B.ingest_availability({'rows': rows, 'source_file_sha256': 'fixture'}, F.iso(at), provider, GAMES, raw, norm, acc), raw, norm


def stores(tmp_path, sub='s'):
    return tmp_path / sub / 'raw', tmp_path / sub / 'norm'


# ---------------------------------------------------------------- as-of semantics
def test_later_status_never_leaks_into_an_earlier_cutoff(tmp_path):
    acc = acceptance(tmp_path)
    c24, c90 = F.window(KO, 'T24')[1], F.window(KO, 'T90')[1]
    ingest(tmp_path, [injury_row('Questionable')], c24 - timedelta(hours=3), acc)
    ingest(tmp_path, [injury_row('Out', 'Did Not Participate In Practice')], c90 - timedelta(minutes=20), acc)
    ingest(tmp_path, [injury_row('Doubtful')], c90 + timedelta(minutes=5), acc)                          # retrieved AFTER the T90 cutoff
    _, norm = stores(tmp_path)
    a24, a90 = B.availability_at(PID, GAME['game_id'], c24, 'T24', norm), B.availability_at(PID, GAME, c90, 'T90', norm)
    assert a24['state'] == 'QUESTIONABLE' and a90['state'] == 'OUT'
    assert [r['revision_sequence'] for r in a90['records']] == [2]                                      # the revision after the cutoff (3) is invisible
    assert B.availability_at(PID, GAME['game_id'], c24 - timedelta(hours=10), 'T24', norm)['state'] == 'UNKNOWN'          # before anything was captured: no invented state


def test_t24_and_t90_state_separation_by_label_and_eligibility(tmp_path):
    acc = acceptance(tmp_path)
    c24, c90 = F.window(KO, 'T24')[1], F.window(KO, 'T90')[1]
    ingest(tmp_path, [injury_row('Out')], c24 + timedelta(hours=2), acc)                                    # first seen between the cutoffs
    _, norm = stores(tmp_path)
    rec = read_ledger(norm / B.NORM_FILE)[0]['record']
    assert rec['cutoff_eligibility'] == 'T90'
    assert B.availability_at(PID, GAME['game_id'], c24, 'T24', norm)['state'] == 'UNKNOWN'
    assert B.availability_at(PID, GAME['game_id'], c90, 'T90', norm)['state'] == 'OUT'
    assert B.availability_at(PID, GAME['game_id'], c90, 'T24', norm)['state'] == 'UNKNOWN'                  # label T24 requires T24 eligibility


def test_published_at_after_the_cutoff_is_invisible_even_if_retrieved_earlier(tmp_path):
    rec = {'published_at': F.iso(KO - timedelta(hours=10)), 'retrieved_at': F.iso(KO - timedelta(hours=30)), 'valid_from': F.iso(KO - timedelta(hours=10)), 'valid_to': None, 'kickoff_time': F.iso(KO), 'backfilled': False, 'original_vintage': True, 'cutoff_eligibility': 'BOTH'}
    assert not B.visible(rec, F.window(KO, 'T24')[1])
    assert B.visible(rec, F.window(KO, 'T90')[1])
    assert not B.visible({**rec, 'backfilled': True, 'original_vintage': False}, F.window(KO, 'T90')[1])
    assert not B.visible({**rec, 'retrieved_at': F.iso(KO + timedelta(minutes=1)), 'published_at': None}, KO + timedelta(hours=1))      # postgame capture is never visible


def test_conflicting_providers_return_the_most_severe_state_flagged_not_averaged(tmp_path):
    acc = acceptance(tmp_path)
    at = F.window(KO, 'T24')[1] - timedelta(hours=2)
    ingest(tmp_path, [injury_row('Questionable')], at, acc)
    raw, norm = stores(tmp_path)
    B.ingest_availability({'rows': [{'season': '2026', 'week': '7', 'team': 'T00', 'player_id': PID, 'position': 'WR', 'state': 'OUT', 'authorization_ref': 'fixture-auth', 'published_at': F.iso(at - timedelta(hours=1))}], 'source_file_sha256': 'm'},
                          F.iso(at), 'manually_supplied_authorized_records', GAMES, raw, norm, acc)
    a = B.availability_at(PID, GAME['game_id'], F.window(KO, 'T24')[1], 'T24', norm)
    assert a['state'] == 'OUT' and a['conflict'] is True and len(a['records']) == 2


def test_exact_player_and_game_identity_only(tmp_path):
    acc = acceptance(tmp_path)
    at = F.window(KO, 'T24')[1] - timedelta(hours=2)
    ingest(tmp_path, [injury_row(pid='00-0000100'), injury_row(pid='00-0000101', report='Out')], at, acc)
    _, norm = stores(tmp_path)
    c = F.window(KO, 'T24')[1]
    assert B.availability_at('00-0000100', GAME['game_id'], c, 'T24', norm)['state'] == 'QUESTIONABLE'
    assert B.availability_at('00-0000101', GAME['game_id'], c, 'T24', norm)['state'] == 'OUT'
    assert B.availability_at('00-0000100', '2026_07_OTHER', c, 'T24', norm)['reason'] == 'NO_VISIBLE_RECORD'
    with pytest.raises(ValueError, match='player_id_not_gsis'):
        ingest(tmp_path, [injury_row(pid='Fixture Player')], at, acc, sub='t')                           # a name-like id is rejected, never matched


# ---------------------------------------------------------------- append-only, revisions, determinism
def test_append_only_revision_preservation_and_tamper_evidence(tmp_path):
    acc = acceptance(tmp_path)
    at = F.window(KO, 'T24')[1] - timedelta(hours=5)
    s1, raw, norm = ingest(tmp_path, [injury_row('Questionable')], at, acc)
    s2, *_ = ingest(tmp_path, [injury_row('Questionable')], at + timedelta(hours=1), acc)                # same content: not a revision
    s3, *_ = ingest(tmp_path, [injury_row('Out')], at + timedelta(hours=2), acc)
    assert (s1['new'], s2['unchanged'], s3['revisions']) == (1, 1, 1)
    lines = read_ledger(norm / B.NORM_FILE)
    assert [x['record']['revision_sequence'] for x in lines] == [1, 2] and [x['record']['state'] for x in lines] == ['QUESTIONABLE', 'OUT']
    assert lines[0]['record']['normalized_record_sha256'] == B.record_hash(lines[0]['record'])
    verify_chain(raw / B.RAW_FILE)
    verify_chain(norm / B.NORM_FILE)
    first = (norm / B.NORM_FILE).read_text().splitlines()[0]
    p = norm / B.NORM_FILE
    p.write_text(p.read_text().replace('"state":"QUESTIONABLE"', '"state":"ACTIVE"'))
    with pytest.raises(ValueError, match='altered'):
        verify_chain(p)
    assert first                                                                                          # prior revisions are rows, never overwritten


def test_output_is_deterministic(tmp_path):
    acc = acceptance(tmp_path)
    at = F.window(KO, 'T24')[1] - timedelta(hours=5)
    for sub in ('a', 'b'):
        ingest(tmp_path, [injury_row('Questionable'), injury_row('Out', pid='00-0000101')], at, acc, sub=sub)
    for name in (B.RAW_FILE,):
        assert (tmp_path / 'a/raw' / name).read_bytes() == (tmp_path / 'b/raw' / name).read_bytes()
    assert (tmp_path / 'a/norm' / B.NORM_FILE).read_bytes() == (tmp_path / 'b/norm' / B.NORM_FILE).read_bytes()


# ---------------------------------------------------------------- capture windows
def roster_rows(status='ACT'):
    rows = []
    for team in ('T00', 'T01'):
        rows.append({'season': '2026', 'week': '7', 'team': team, 'gsis_id': f'00-{team}-QB'.replace('T', '0'), 'full_name': 'QB', 'position': 'QB', 'status': 'ACT', 'game_type': 'REG'})
        rows.append({'season': '2026', 'week': '7', 'team': team, 'gsis_id': PID if team == 'T00' else '00-0000200', 'full_name': 'WR', 'position': 'WR', 'status': status, 'game_type': 'REG'})
        rows.append({'season': '2026', 'week': '7', 'team': team, 'gsis_id': f'00-9{team[-1]}', 'full_name': 'OL', 'position': 'OL', 'status': 'ACT', 'game_type': 'REG'})
    return rows


def capture(tmp_path, now, retrieved, rows=None, sub='c'):
    acc = acceptance(tmp_path)
    raw, norm = stores(tmp_path, sub)
    res = B.capture_roster(rows or roster_rows(), 'filesha', F.iso(retrieved), [GAME], clock=lambda: now, raw_dir=raw, norm_dir=norm, acceptance_path=acc)
    return res, raw, norm


def test_capture_inside_the_window_records_roster_status_and_never_ol(tmp_path):
    now = F.window(KO, 'T24')[1] - timedelta(hours=3)
    res, raw, norm = capture(tmp_path, now, now - timedelta(minutes=1))
    assert [(g, l) for g, l, _ in res['captured']] == [(GAME['game_id'], 'T24')] and res['missed'] == []
    rows = [x['record'] for x in read_ledger(norm / B.NORM_FILE)]
    assert {r['position'] for r in rows} == {'QB', 'WR'} and all(r['state'] == 'EXPECTED_ACTIVE' and r['confidence'] == 'LOW' and r['capture_window_status'] == 'IN_T24_WINDOW' and r['original_vintage'] for r in rows)
    assert all(r['qb_route'] == 'ROSTER_STATUS_ONLY_NOT_A_QB_STATE' for r in rows if r['position'] == 'QB')
    assert B.availability_at(PID, GAME['game_id'], F.window(KO, 'T24')[1], 'T24', norm)['state'] == 'EXPECTED_ACTIVE'
    ev = [e['record'] for e in read_ledger(norm / B.EVENT_FILE)]
    assert ev[0]['event'] == 'CAPTURED' and ev[0]['new'] == 4


def test_roster_status_change_between_windows_is_a_preserved_revision(tmp_path):
    now24 = F.window(KO, 'T24')[1] - timedelta(hours=3)
    capture(tmp_path, now24, now24 - timedelta(minutes=1))
    now90 = F.window(KO, 'T90')[1] - timedelta(minutes=30)
    res, raw, norm = capture(tmp_path, now90, now90 - timedelta(minutes=1), rows=roster_rows('INA'))
    assert [(g, l) for g, l, _ in res['captured']] == [(GAME['game_id'], 'T90')]
    c24, c90 = F.window(KO, 'T24')[1], F.window(KO, 'T90')[1]
    assert B.availability_at(PID, GAME['game_id'], c24, 'T24', norm)['state'] == 'EXPECTED_ACTIVE'
    assert B.availability_at(PID, GAME['game_id'], c90, 'T90', norm)['state'] == 'INACTIVE'
    unchanged = B.availability_at('00-0000200', GAME['game_id'], c90, 'T90', norm.parent / 'norm')
    assert unchanged['captures_confirming'] == 2                                                         # both captures confirm the unchanged player


def test_late_capture_is_marked_missed_and_never_backfilled(tmp_path):
    cut24 = F.window(KO, 'T24')[1]
    res, raw, norm = capture(tmp_path, cut24 + timedelta(minutes=1), cut24 + timedelta(minutes=1))
    assert res['captured'] == [] and res['missed'] == [(GAME['game_id'], 'T24')]
    assert not (norm / B.NORM_FILE).exists()
    again, *_ = capture(tmp_path, cut24 + timedelta(minutes=2), cut24 + timedelta(minutes=2))
    assert again['missed'] == []                                                                          # logged once
    late_fetch, *_ = capture(tmp_path, cut24 - timedelta(minutes=1), cut24 + timedelta(minutes=1), sub='d')       # clock inside the window but bytes retrieved after the cutoff
    assert late_fetch['captured'] == [] and late_fetch['missed'] == [(GAME['game_id'], 'T24')]
    nothing, raw, norm = capture(tmp_path, F.window(KO, 'T24')[0] - timedelta(hours=1), F.window(KO, 'T24')[0] - timedelta(hours=1), sub='e')
    assert nothing['captured'] == [] and nothing['missed'] == [] and not (norm / B.EVENT_FILE).exists()      # before the window: nothing


# ---------------------------------------------------------------- source acceptance
def test_unaccepted_or_unauthorized_sources_can_never_ingest(tmp_path):
    for bad in (acceptance(tmp_path, injuries='PROMISING_NEEDS_TERMS'), acceptance(tmp_path, injuries_authorized=False)):
        with pytest.raises(B.SourceNotAccepted):
            ingest(tmp_path, [injury_row()], F.window(KO, 'T24')[1] - timedelta(hours=2), bad, sub='x')
    with pytest.raises(B.SourceNotAccepted):
        B.ingest_availability({'rows': []}, F.iso(KO), 'sportradar', GAMES, tmp_path / 'r', tmp_path / 'n', acceptance(tmp_path))
    committed = json.loads((ART / B.ACCEPTANCE).read_text())
    classes = {s['provider']: s['classification'] for s in committed['sources']}
    assert set(classes.values()) <= set(B.CLASSIFICATIONS)
    assert classes['nflverse_injuries'] == 'PROMISING_NEEDS_TERMS' and classes['nfl_com_injury_report'] == 'BLOCKED_LICENSE' and classes['nflverse_roster_weekly'] == 'ACCEPTED_FORWARD_CAPTURE'
    assert not B.accepted('nflverse_injuries') and not B.accepted('nfl_com_injury_report') and not B.accepted('sportradar') and B.accepted('nflverse_roster_weekly')
    assert committed['accepted_injury_or_practice_source'] is None
    with pytest.raises(B.SourceNotAccepted):
        B.ingest_availability({'rows': [{'season': '2026', 'week': '7', 'team': 'T00', 'player_id': PID, 'state': 'OUT'}]}, F.iso(KO - timedelta(hours=30)), 'manually_supplied_authorized_records', GAMES, tmp_path / 'r', tmp_path / 'n', acceptance(tmp_path))


def test_no_scraper_exists_and_no_network_except_the_roster_release_download():
    tree = ast.parse((ROOT / 'nfl_v2_phase2b_availability.py').read_text())
    imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {'urllib', 'urllib.request', 'requests', 'http', 'bs4', 'selenium', 'playwright'} & imports
    text = (ROOT / 'nfl_v2_phase2b_availability.py').read_text().lower()
    for bad in ('nfl.com', 'espn.com', 'spread_line', 'moneyline', 'over_odds', 'sportsbook'):
        assert bad not in text.replace('nfl_com_injury_report', ''), bad
    assert text.count('curl') == 1                                                                         # only the nflverse roster release download


# ---------------------------------------------------------------- QB routing
def test_qb_state_is_routed_to_the_existing_ingestion_layer_not_duplicated(tmp_path):
    with pytest.raises(Q.ProviderNotAuthorized):
        B.route_qb_payload({'players': []}, F.iso(KO - timedelta(hours=30)), 'sportradar', store=tmp_path / 'qbstore', acceptance_path=tmp_path / 'none.json')
    gate = B.qb_state_link('T00', GAME['game_id'], F.window(KO, 'T24')[1], 'T24', acceptance_path=tmp_path / 'none.json')
    assert gate == {'status': 'BLOCKED', 'reason': 'NO_ACCEPTED_QB_STATE_SOURCE'}
    assert 'sportradar' not in B.ADAPTERS and 'sportsdataio' not in B.ADAPTERS
    qb_acc = tmp_path / 'qbacc.json'
    qb_acc.write_text(json.dumps({'providers': {'sportradar': {'classification': 'ACCEPTED_FORWARD_ONLY'}}}))
    payload = {'season': 2026, 'week': 7, 'game_id': GAME['game_id'], 'team': 'T00', 'opponent': 'T01', 'scheduled': F.iso(KO), 'players': [{'id': 'sr-1', 'name': 'F QB', 'position': 'QB', 'depth_status': 'STARTER', 'updated': F.iso(KO - timedelta(hours=40)), 'record_id': 'r', 'revision': '1', 'original_version': True, 'depth': 1}]}
    rows = B.route_qb_payload(payload, F.iso(KO - timedelta(hours=39)), 'sportradar', crosswalk={'sr-1': '00-0000555'}, store=tmp_path / 'qbstore', acceptance_path=qb_acc)
    assert len(rows) == 1 and (tmp_path / 'qbstore' / Q.NORMALIZED_FILE).exists() and not (tmp_path / 'qbstore' / B.NORM_FILE).exists()


# ---------------------------------------------------------------- Phase2A untouched, observational link
def committed_bytes(path):
    r = subprocess.run(['git', 'show', f'{FIRST_FORECAST_COMMIT}:{path.relative_to(ROOT)}'], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        pytest.skip('first-forecast commit unavailable')
    return r.stdout


def test_existing_phase2a_t24_rows_comparators_and_lock_are_byte_identical_to_the_first_forecast_commit():
    for name in ('forecasts_T24.jsonl', 'comparators.jsonl'):
        original = committed_bytes(F.LEDGER / name)
        assert (F.LEDGER / name).read_bytes().startswith(original), name                                    # later lines may append; the original bytes never change
    lock = json.loads((ART / 'phase2a_engine_lock.json').read_text())
    assert lock['code_sha256']['nfl_v2_phase2a_forward_forecast.py'] == F.sha_file(ROOT / 'nfl_v2_phase2a_forward_forecast.py')
    assert committed_bytes(ART / 'phase2a_engine_lock.json') == (ART / 'phase2a_engine_lock.json').read_bytes()
    tb = [x['record'] for x in read_ledger(F.LEDGER / 'forecasts_T24.jsonl') if x['record']['game_id'] == '2026_05_TB_DAL']
    assert len(tb) == 11 and all(r['git_sha'] == '21fdcb7f4b4724cd10409f082b79747c251a95e3' for r in tb)


def test_phase2b_never_writes_to_phase2a_and_linking_is_read_only(tmp_path):
    tree = ast.parse((ROOT / 'nfl_v2_phase2b_availability.py').read_text())
    text = (ROOT / 'nfl_v2_phase2b_availability.py').read_text()
    assert 'F.LEDGER' in text and not any('append_record(ledger_dir' in line or 'append_record(F.LEDGER' in line for line in text.splitlines())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'append_record':
            assert 'ledger' not in ast.dump(node.args[0]).lower().replace('raw_dir', '') or 'dir' in ast.dump(node.args[0]).lower()
    before = {n: (F.LEDGER / n).read_bytes() for n in F.LEDGER_FILES if (F.LEDGER / n).exists()}
    links = B.link_phase2a(F.LEDGER, tmp_path / 'emptynorm')
    assert links and {x['availability_state'] for x in links} == {'UNKNOWN'} and {x['reason'] for x in links} == {'NO_VISIBLE_RECORD'}
    assert before == {n: (F.LEDGER / n).read_bytes() for n in before}


def test_link_reports_the_state_visible_at_each_forecast_cutoff(tmp_path):
    led = tmp_path / 'led'
    cut = F.window(KO, 'T24')[1]
    for cutoff_type in ('T24',):
        rec = {'forecast_id': 'f1', 'game_id': GAME['game_id'], 'player_id': PID, 'cutoff_type': cutoff_type, 'cutoff_time': F.iso(cut), 'availability_state': 'ACT'}
        append_record(led / 'forecasts_T24.jsonl', rec, 'forecast_id')
    now = cut - timedelta(hours=3)
    _res, raw, norm = capture(tmp_path, now, now - timedelta(minutes=1))
    links = B.link_phase2a(led, norm)
    assert links[0]['availability_state'] == 'EXPECTED_ACTIVE' and links[0]['phase2a_availability_state'] == 'ACT' and links[0]['record_sha256']


def graded(fid, cut, game, pid, proj, actual, head='receiving_yards'):
    return {'forecast_id': fid, 'cutoff_type': cut, 'game_id': game, 'player_id': pid, 'heads': {head: {'projection': proj, 'actual': actual}}}


def test_availability_report_counts_changes_and_catastrophic_association_without_causal_claims():
    links = [{'forecast_id': f, 'availability_state': s} for f, s in (('a24', 'QUESTIONABLE'), ('a90', 'OUT'), ('b24', 'EXPECTED_ACTIVE'), ('b90', 'EXPECTED_ACTIVE'), ('c24', 'UNKNOWN'))]
    rows = [graded('a24', 'T24', 'g', 'a', 50, 5), graded('a90', 'T90', 'g', 'a', 50, 0), graded('b24', 'T24', 'g', 'b', 30, 28), graded('b90', 'T90', 'g', 'b', 31, 28), graded('c24', 'T24', 'g', 'c', 20, 90), graded('zz', 'T24', 'g', 'z', 1, 1)]
    rep = B.availability_report(rows, links)
    assert rep['unlinked_graded'] == 1 and rep['counts_by_state']['T24|QUESTIONABLE'] == 1 and rep['counts_by_state']['T90|OUT_OR_INACTIVE'] == 1
    assert rep['status_changes_T24_to_T90'] == {'DOWNGRADE:QUESTIONABLE->OUT_OR_INACTIVE': 1, 'UNCHANGED:ACTIVE->ACTIVE': 1}
    assert rep['catastrophic_association']['T24|receiving_yards'] == {'QUESTIONABLE': 1, 'UNKNOWN': 1} and rep['catastrophic_association']['T90|receiving_yards'] == {'OUT_OR_INACTIVE': 1}
    assert rep['status'] == 'DESCRIPTIVE_ONLY_NO_CAUSALITY' and 'caus' not in json.dumps(rep).lower().replace('no_causality', '')


# ---------------------------------------------------------------- operator aid
def test_next_due_is_read_only_and_matches_the_real_tb_dal_t90_window(tmp_path):
    sched = [{'game_id': '2026_05_TB_DAL', 'season': '2026', 'game_type': 'REG', 'week': '5', 'gameday': '2026-10-08', 'gametime': '20:15', 'away_team': 'TB', 'home_team': 'DAL'}]
    before = {n: (F.LEDGER / n).read_bytes() for n in F.LEDGER_FILES if (F.LEDGER / n).exists()}
    r = B.next_due(sched, F.LEDGER, now=datetime(2026, 10, 7, 12, tzinfo=UTC))
    assert r['NEXT_DUE_FORECAST_WINDOW'] == 'T90' and r['opens_at'] == '2026-10-08T21:15:00+00:00' and r['cutoff_at'] == '2026-10-08T22:45:00+00:00' and r['games_due'] == ['2026_05_TB_DAL'] and r['state'] == 'NOT_YET_OPEN'
    inside = B.next_due(sched, F.LEDGER, now=datetime(2026, 10, 8, 21, 30, tzinfo=UTC))
    assert inside['state'] == 'OPEN_NOW' and inside['games_due'] == ['2026_05_TB_DAL']
    after = B.next_due(sched, F.LEDGER, now=datetime(2026, 10, 8, 22, 46, tzinfo=UTC))
    assert after['NEXT_DUE_FORECAST_WINDOW'] is None and after['state'] == 'NO_FUTURE_WINDOW'
    assert before == {n: (F.LEDGER / n).read_bytes() for n in before}
    main_src = ast.parse((ROOT / 'nfl_v2_phase2b_availability.py').read_text())
    flags = [c.args[0].value for n in ast.walk(main_src) if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'add_argument' for c in [n] if c.args and isinstance(c.args[0], ast.Constant)]
    assert not any('now' in f or 'clock' in f for f in flags)


def test_next_due_ignores_windows_whose_forecast_already_exists(tmp_path):
    led = tmp_path / 'led'
    sched = [{'game_id': 'G1', 'season': '2026', 'game_type': 'REG', 'week': '9', 'gameday': '2026-10-25', 'gametime': '13:00', 'away_team': 'A', 'home_team': 'B'}]
    now = F.F if False else datetime(2026, 10, 24, 0, tzinfo=UTC)
    assert B.next_due(sched, led, now=now)['NEXT_DUE_FORECAST_WINDOW'] == 'T24'
    append_record(led / 'forecasts_T24.jsonl', {'forecast_id': 'x', 'game_id': 'G1'}, 'forecast_id')
    assert B.next_due(sched, led, now=now)['NEXT_DUE_FORECAST_WINDOW'] == 'T90'


# ---------------------------------------------------------------- committed artifacts and scope
def test_committed_schema_manifest_and_stores_are_consistent_and_contain_no_fixtures():
    assert json.loads((ART / 'phase2b_availability_schema.json').read_text()) == B.schema_document()
    man = json.loads((ART / 'phase2b_capture_manifest.json').read_text())
    assert man['phase2a_engine_modified'] is False and man['production_promotion'] is False
    for rel, info in man['ledgers'].items():
        d, n = rel.split('/')
        cnt, last = verify_chain(ART / d / n)
        assert info['lines'] <= cnt
    for d in (B.RAW_DIR, B.NORM_DIR):
        for name in (B.RAW_FILE, B.NORM_FILE, B.EVENT_FILE):
            p = d / name
            if p.exists():
                assert B.FIXTURE_LABEL not in p.read_text() and 'FIXTURE' not in p.read_text().upper().replace('FIXTURES', '')
    proto = json.loads((ART / B.PROTOCOL).read_text())
    assert set(proto['states']) == set(B.STATES) and proto['severity_order'] == list(B.SEVERITY) and proto['no_go'][0] == 'no Phase2A v1.1'
    assert B.FIXTURE_LABEL == 'FIXTURE_ONLY_NOT_REAL_AVAILABILITY_EVIDENCE'


def test_scope_no_production_or_other_sports_or_phase2a_changes():
    r = subprocess.run(['git', 'diff', '--name-status', 'e67c066c58b88b30b69c0be1f35b57ec12aa0d91', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('base commit unavailable')
    for line in r.stdout.splitlines():
        status, _, path = line.partition('\t')
        if status == 'A':
            assert any(k in path for k in ('phase2b', 'phase0_frozen')), path
        else:
            assert path in {'nfl_models/nfl_player_outcome_v2/research_registry.json', 'nfl_v2_phase0_audit.py', '.github/workflows/nfl_v2_phase0_audit.yml', 'tests/test_nfl_v2_phase1l_qb_opportunity.py', 'tests/test_nfl_v2_phase1m_qb_state.py',
                            'tests/test_nfl_v2_phase1n_team_opportunity.py', 'tests/test_nfl_v2_phase1r_error_budget.py', 'nfl_models/nfl_player_outcome_v2/phase2a_forecast_manifest.json'} or path.startswith('nfl_models/nfl_player_outcome_v2/phase2a_ledger/') or path.startswith('nfl_models/nfl_player_outcome_v2/phase2a_inputs/'), (status, path)
