"""Deterministic mechanism, scoring, source firewall and adversarial replay tests."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tennis_v2_data as D
import tennis_v2_phase1h_inputs as I
import tennis_v2_phase1h_research as R
from tennis_v2_phase1h_components import point_projection, fit_stack, stack_predict
from tennis_v2_phase1h_distributions import match_distribution
from tennis_v2_phase1_scoring import match_win_probability
from tennis_v2_phase1_guards import EvidenceBlocked, FormatEvidence


def row(day='2019-01-01',event='2019-event',num=1,winner='a',loser='b',tour='atp'):
    return dict(tour=tour,match_date=day,tourney_id=event,tourney_name='Example',tourney_level='A',surface='Hard',best_of=3,round='R32',match_id=f'{tour}_{event}_{num}',winner_id=winner,loser_id=loser,winner_name=winner,loser_name=loser,winner_rank_points=1000,loser_rank_points=500,score='6-4 6-4',is_incomplete=0)


def stats(first=40,second=12):
    return dict(serve_points=80,first_serve_in=50,first_serve_won=first,second_serve_won=second,serve_games=10,break_points_faced=3,break_points_saved=2)


def target(day='2019-02-01',event='target',p1='a',p2='b',tour='atp',surface='Hard'):
    return I.Target(tour,event,day,surface,p1,p2,FormatEvidence(3,'TB7_AT_6_ALL_SETS'))


class Mechanisms(unittest.TestCase):
    def state(self):
        s=I.State('atp')
        for i in range(6):
            s.release(row(num=i),{'a':stats(44,20),'b':stats(30,7)},None)
        return s
    def test_input_target_contains_no_outcome_fields(self):
        self.assertEqual(set(I.Target.__dataclass_fields__),{'tour','event','day','surface','p1','p2','fmt'})
    def test_first_second_count_identities(self):
        w,n=I.stats_channels(stats())
        self.assertEqual((w,n),([52,50,40,12],[80,80,50,30]))
    def test_invalid_not_zero(self):
        for x in ({},stats(60),dict(stats(),serve_points=None),dict(stats(),break_points_saved=4)):
            self.assertIsNone(I.stats_channels(x))
    def test_hold_label_checked(self):
        self.assertEqual(I.observed_hold(stats()),.9)
        self.assertIsNone(I.observed_hold(dict(stats(),serve_games=0)))
    def test_embargo_boundary(self):
        s=self.state()
        s.features(target(day='2019-01-29'))
        with self.assertRaises(EvidenceBlocked): s.features(target(day='2019-01-28'))
    def test_same_event_blocked_even_if_old(self):
        with self.assertRaises(EvidenceBlocked): self.state().features(target(event='2019-event'))
    def test_tour_isolation(self):
        with self.assertRaises(EvidenceBlocked): self.state().features(target(tour='wta'))
    def test_unknown_surface_blocked(self):
        with self.assertRaises(EvidenceBlocked): self.state().features(target(surface='Unknown'))
    def test_identity_no_surname_fallback(self):
        x=self.state().features(target(p1='unknown'))['players'][0]
        self.assertEqual(x['prior_matches'],0)
    def test_repeat_features_deterministic(self):
        s=self.state(); self.assertEqual(s.features(target()),s.features(target()))
    def test_opponent_explicitly_changes_projection(self):
        s=self.state(); a=s.features(target())['players'][0]; b=s.features(target(p2='unknown'))['players'][0]
        self.assertNotEqual(point_projection(a,'C')['serve_point_win'],point_projection(b,'C')['serve_point_win'])
        self.assertEqual(point_projection(a,'U')['serve_point_win'],point_projection(b,'U')['serve_point_win'])
    def test_channel_chain(self):
        x=point_projection(self.state().features(target())['players'][0],'D')
        self.assertAlmostEqual(x['serve_point_win'],x['first_in']*x['first_serve_win']+(1-x['first_in'])*x['second_serve_win'])
        self.assertAlmostEqual(x['receiver_return_point_win']+x['serve_point_win'],1)
    def test_no_physical_penalty(self):
        p=point_projection(self.state().features(target())['players'][0],'C')
        self.assertNotIn('fatigue',p)
    def test_serve_missing_keeps_match_rating(self):
        s=I.State('atp');s.release(row(),{},None)
        self.assertEqual(s.elo.on['a'],1);self.assertEqual(len(s.serve['a']),0)
    def test_retirement_not_serve_training(self):
        s=I.State('atp');s.release(dict(row(),is_incomplete=1,score='RET'),{'a':stats(),'b':stats()},None)
        self.assertEqual(len(s.serve['a']),0);self.assertEqual(s.elo.on['a'],1)
    def test_walkover_not_rating_training(self):
        s=I.State('atp');s.release(dict(row(),score='W/O'),{},None)
        self.assertNotIn('a',s.elo.on)
    def test_source_year_guard(self):
        with self.assertRaises(EvidenceBlocked): I.rule_for(row(day='2025-01-01'))
    def test_special_format_blocks(self):
        for x in [dict(row(),tourney_name='Next Gen Finals',best_of=5),dict(row(),tourney_level='D'),dict(row(),tourney_level='I'),dict(row(),best_of=1)]:
            with self.assertRaises(EvidenceBlocked): I.rule_for(x)
    def test_slam_rules_by_year(self):
        for name,year,rule in [('Wimbledon',2018,'ADVANTAGE_FINAL_SET'),('Wimbledon',2019,'TB7_AT_12_FINAL'),('Australian Open',2019,'TB10_AT_6_FINAL'),('Roland Garros',2021,'ADVANTAGE_FINAL_SET'),('US Open',2021,'TB7_AT_6_ALL_SETS'),('Roland Garros',2022,'TB10_AT_6_FINAL')]:
            self.assertEqual(I.rule_for(dict(row(day=f'{year}-06-01'),tourney_level='G',tourney_name=name,best_of=5)).final_set_rule,rule)
    def test_surface_is_target_season_not_event_name(self):
        s=self.state();self.assertNotEqual(s.features(target(surface='Hard'))['players'][0]['own']['rate'],s.features(target(surface='Grass'))['players'][0]['own']['rate'])
    def test_distributions_match_validated_solver(self):
        for bo in (3,5):
            for rule in ('TB7_AT_6_ALL_SETS','TB10_AT_6_FINAL','ADVANTAGE_FINAL_SET','TB7_AT_12_FINAL'):
                f=FormatEvidence(bo,rule);d=match_distribution(.67,.59,f)
                self.assertAlmostEqual(sum(d['set_score'].values()),1,12)
                self.assertAlmostEqual(sum(d['tiebreak_count'].values()),1,12)
                self.assertAlmostEqual(d['p1_win'],match_win_probability(.67,.59,fmt=f),12)
                self.assertAlmostEqual(d['p1_win']+match_distribution(.59,.67,f)['p1_win'],1,12)
    def test_cluster_boot_resamples_entire_event(self):
        ci=R.cluster_boot([1,1,0],[0,0,0],['eventA','eventA','eventB'])
        self.assertEqual(ci['clusters'],2);self.assertEqual(ci['hi'],1);self.assertEqual(ci['lo'],0)
        self.assertEqual(ci,R.cluster_boot([1,1,0],[0,0,0],['eventA','eventA','eventB']))
    def test_stack_complement(self):
        x=[[.1,.2,.3],[-.1,-.2,-.3]];b=fit_stack(x,[1,0])
        self.assertAlmostEqual(stack_predict([.6,.7,.8],b)+stack_predict([.4,.3,.2],b),1)
    def test_legacy_solver_arithmetic_unchanged(self):
        import subprocess
        text=subprocess.check_output(['git','show','1680e1d533cf6711bac358e00d50406ad573bcce:tennis_v2_phase1_scoring.py'],text=True)
        self.assertEqual(text.split('from __future__')[1],(D.REPO/'tennis_v2_phase1_scoring.py').read_text().split('from __future__')[1])
    def test_source_manifest_stays_pinned(self):
        p=json.loads((D.OUT/'phase1h_source_feasibility.json').read_text())
        self.assertEqual(p['source_manifest_sha256'],hashlib.sha256((D.OUT/'source_manifest.json').read_bytes()).hexdigest())
        self.assertFalse(p['new_timing_source_accepted'])
    def test_unsupported_human_dimensions_explicit(self):
        spec=json.loads((D.OUT/'phase1h_human_research_spec.json').read_text())
        blocked={f['feature'] for f in spec['features'] if f['status']=='BLOCKED_DATA'}
        self.assertIn('rest/fatigue/match duration',blocked);self.assertIn('injury/withdrawal/recovery',blocked)
    def test_diagnostics_requires_hash_lock(self):
        with patch.object(R,'lock_hashes',return_value={}):
            if (D.OUT/'phase1h_development_lock.json').exists():
                with self.assertRaises(EvidenceBlocked): R.run('diagnostics',Path('/tmp/should-not-write-phase1h'))
    def fixture(self,mutation=None,reverse=False):
        rows=[row(day='2018-01-01',event='old',num=i) for i in range(6)]+[row(event='target',num=99)]
        if mutation: mutation(rows[-1])
        rows=sorted(rows,key=D.canonical_key)
        ps={(r['match_id'],pid):stats() for r in rows for pid in ('a','b')}
        if reverse: rows=sorted(reversed(rows),key=D.canonical_key)
        with patch.object(R,'load_tour',return_value=(rows,ps,{})): return R.replay('atp')[0]
    def test_future_target_outcome_cannot_change_inputs(self):
        a=self.fixture()[0]['features'];b=self.fixture(lambda r:r.update(winner_id='b',loser_id='a'))[0]['features']
        self.assertEqual(a,b)
    def test_same_event_shuffle_does_not_change_features(self):
        self.assertEqual(self.fixture()[0]['features'],self.fixture(reverse=True)[0]['features'])
    def test_mutated_target_statistics_cannot_change_features(self):
        rows=[row(day='2018-01-01',event='old',num=i) for i in range(6)]+[row(event='target',num=99)]
        ps={(r['match_id'],pid):stats() for r in rows for pid in ('a','b')}
        with patch.object(R,'load_tour',return_value=(rows,ps,{})): a=R.replay('atp')[0][0]['features']
        ps[(rows[-1]['match_id'],'a')]=stats(5,1)
        with patch.object(R,'load_tour',return_value=(rows,ps,{})): b=R.replay('atp')[0][0]['features']
        self.assertEqual(a,b)
    def test_future_appended_rows_cannot_change_past(self):
        s=self.state();before=s.features(target());future=row(day='2024-12-01',event='future')
        with self.assertRaises(EvidenceBlocked):
            s.release(future,{'a':stats(),'b':stats()},None);s.features(target())
        self.assertEqual(before,self.state().features(target()))


if __name__=='__main__': unittest.main()
