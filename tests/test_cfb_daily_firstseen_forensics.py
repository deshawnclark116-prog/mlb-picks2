import json
from pathlib import Path
from datetime import datetime,timezone,timedelta
import pytest
import cfb_daily_firstseen_forensics as f

def row(kickoff,logged='2026-10-10T10:00:00Z',hit=True,p=.85,market='moneyline'):
 return {'kickoff_utc':kickoff,'logged_at':logged,
   'evaluation_probability_source':'FIRST_PREGAME_LEDGER_ENTRY',
   'canonical_key':'test|match','market':market,
   'generation':'in_season_context_v2','team':'Test Team','opponent':'Away Team',
   'player':'Test Team','game_id':'1',
   'pick':'Test Team ML' if market=='moneyline' else 'UNDER 69.5',
   'model_prob':p,'result':'hit' if hit else 'miss',
   'actual':'Other Team' if not hit else 'Test Team'}

def test_timezone_sat_et_not_utc_day():
 # 01:30Z Sunday is Saturday night Eastern
 record={'results':[row('2026-10-11T01:30:00Z')]}
 assert f.evaluate(record,{},'2026-10-10')['all']['n']==1
 assert f.evaluate(record,{},'2026-10-11')['all']['n']==0

def test_lookahead_or_non_first_snapshot_is_fatal():
 x=row('2026-10-10T15:00:00Z',logged='2026-10-10T15:01:00Z')
 with pytest.raises(ValueError,match='unverifiable'):f.evaluate({'results':[x]}, {},'2026-10-10')
 x['logged_at']='2026-10-10T14:00:00Z'
 x['evaluation_probability_source']='LAST_BOARD_REFRESH'
 with pytest.raises(ValueError,match='unverifiable'):f.evaluate({'results':[x]}, {},'2026-10-10')

def test_moneyline_95_percent_loss_not_hidden_and_no_fictitious_roi():
 rows=[row('2026-10-10T18:00:00Z',hit=False,p=.95),
       row('2026-10-10T18:00:00Z',hit=True,p=.70,market='rushing_yards')]
 res=f.evaluate({'results':rows},{'ungraded_reasons':{'player_no_stat_line_after_final':91}},'2026-10-10')
 assert res['moneyline']['high80']['n']==1 and res['moneyline']['high80']['wins']==0
 assert res['moneyline']['highest_confidence_losses'][0]['stated_p']==.95
 assert res['global_ungraded_NOT_date_scoped']['player_no_stat_line_after_final']==91
 assert 'ROI is unknown' in ' '.join(res['limits'])
 assert res['decision']==f.FREEZE

def test_real_saturday_record_first_seen_regression():
 # Repo's published record is immutable FIRST_PREGAME_LEDGER material.
 path=Path('docs/cfb_record.json')
 if not path.exists():pytest.skip('not checked out with repo record')
 result=f.evaluate(json.loads(path.read_text()),
                   json.loads(Path('docs/cfb_forward_status.json').read_text()),'2026-10-10')
 assert result['all']['n']==197
 assert result['all']['wins']==131 and result['all']['losses']==66
 assert result['moneyline']['overall']['n']==42
 assert result['moneyline']['overall']['wins']==25
 assert result['moneyline']['high80']['n']==16
 assert result['moneyline']['high80']['wins']==10
 assert result['model_threshold_side_counts']['rushing_yards']=={'OVER':13,'UNDER':41}
 assert result['first_pregame_only'] is True
