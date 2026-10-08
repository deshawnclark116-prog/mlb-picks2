"""Manual/authorized snapshot pipeline. No scraper, credentials, timer or default fetcher.

Caller supplies an authorized transport and explicit license basis. Importer stores
actual retrieval completion, immutable bytes and observations without inventing
lineup/PP certainty. Failed/late capture is recorded, never used for forecasting.
"""
import hashlib
import json

from nhl_v2_phase1a_sog_forward import cutoff_of
from nhl_v2_phase1b_snapshots import SnapshotStore,snapshot,timestamp


def roster_observations(payload, game):
    """NHL rosterSpots are roster observations only; goalie rows excluded from SOG."""
    teams={game['away_team_id'],game['home_team_id']}
    result=[]
    for r in payload.get('rosterSpots',[]):
        if r.get('positionCode')=='G':
            continue
        if r.get('teamId') not in teams or not isinstance(r.get('playerId'),int):
            raise ValueError('unmapped roster observation')
        result.append({'player_id':r['playerId'],'team_id':r['teamId'],'observed_state':'ROSTER_OBSERVED',
                       'membership_verified':False,'line':None,'pp_unit':None,'pk_role':None,
                       'uncertainty':'Published roster is not confirmed dressing or deployment'})
    return result


def capture_game(game,horizon,fetcher,rights_basis,root):
    if not rights_basis or fetcher is None:
        raise ValueError('documented authorized transport required; no default collection')
    store=SnapshotStore(root)
    cutoff=cutoff_of(game['game_start_utc'],horizon).isoformat()
    results=[]
    for endpoint in ('landing','right-rail','boxscore'):
        url=f"https://api-web.nhle.com/v1/gamecenter/{game['game_id']}/{endpoint}"
        raw,meta=fetcher(url)
        if meta.get('url')!=url or meta.get('http_status')!=200 or not meta.get('retrieval_completed_utc') or meta.get('sha256')!=hashlib.sha256(raw).hexdigest():
            raise ValueError('invalid transport provenance')
        payload=json.loads(raw)
        record=snapshot(raw,game=game,horizon=horizon,cutoff=cutoff,
                        retrieved_at=meta['retrieval_completed_utc'],source_kind='game_roster',rights_basis=rights_basis,
                        observations=roster_observations(payload,game))
        # Distinguish endpoints when selecting their latest revisions.
        record['endpoint']=endpoint
        from nhl_v2_phase1a_sog_forward import canon,sha_text
        record['record_sha256']=sha_text(canon({k:v for k,v in record.items() if k!='record_sha256'}))
        store.append(record,raw)
        results.append(record)
    return results


def compare_final_dressed(snapshot_records,truth):
    """Postgame labels only. No truth enters candidate creation or forecast features."""
    if truth.get('state') not in ('OFF','FINAL') or not truth.get('both_teams_complete') or not truth.get('source_hash'):
        raise ValueError('complete official-final dressed truth required')
    game=truth['game']
    if timestamp(truth['retrieved_at'])<=timestamp(game['game_start_utc']):
        raise ValueError('postgame truth cannot precede kickoff')
    rows=[]
    for h in ('T24H','T90','T30'):
        captures=[r for r in snapshot_records if r['game_id']==game['game_id'] and r['scheduled_start']==game['game_start_utc'] and r['horizon']==h and r['timing_eligible']]
        # Exactly all three independent endpoint captures at this horizon, not
        # a pool of partial/late captures from a different scheduled start.
        latest={}
        for r in captures:
            key=r.get('endpoint')
            if key not in latest or timestamp(r['retrieved_at'])>timestamp(latest[key]['retrieved_at']):
                latest[key]=r
        captures=list(latest.values())
        endpoints={r.get('endpoint') for r in captures}
        ids={(o['team_id'],o['player_id']) for r in captures for o in r['observations'] if o['observed_state'] in ('ROSTER_OBSERVED','EXPECTED_DRESSED','CONFIRMED_DRESSED')}
        rows.append({'game_id':game['game_id'],'horizon':h,'date':game['game_start_utc'][:10],
                     'capture_eligible':endpoints=={'landing','right-rail','boxscore'},'both_teams_complete':truth['both_teams_complete'],
                     'final_truth_complete':True,'observed_ids':sorted(ids),'dressed_ids':truth['dressed_ids'],
                     'source_hashes':[r['record_sha256'] for r in captures],'truth_hash':truth['source_hash']})
    return rows
