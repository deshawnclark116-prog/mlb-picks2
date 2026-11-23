#!/usr/bin/env python3
"""Time-safe heterogeneous ensemble: forecast weights use completed past weeks.

No participant/future-week labels can enter weights. Comparison is the same
2025 W6–2026 W11 development set and exactly the same player-game population.
"""
import challenger_tenure_v5 as ten
base=ten.base
base.CFG={
 'xgb_role':ten.xcfg(ten.ROLE),
 'xgb_full':ten.xcfg(ten.FULL),
 'xgb_ol':ten.xcfg(ten.OL),
 'lgb_role':ten.lcfg(ten.ROLE),
 'lgb_full':ten.lcfg(ten.FULL),
 'lgb_ol':ten.lcfg(ten.OL),
 'cat_role':ten.ccfg(ten.ROLE),
 'cat_full':ten.ccfg(ten.FULL),
}
base.ENSEMBLES={
 'ens_role_equal':dict(members=['xgb_role','lgb_role','cat_role'],style='mean'),
 'ens_role_median':dict(members=['xgb_role','lgb_role','cat_role'],style='median'),
 'ens_role_adapt_mae':dict(members=['xgb_role','lgb_role','cat_role'],style='adaptive_mae'),
 'ens_role_adapt_cov':dict(members=['xgb_role','lgb_role','cat_role'],style='adaptive_cov'),
 'ens_full_equal':dict(members=['xgb_full','lgb_full','cat_full'],style='mean'),
 'ens_full_median':dict(members=['xgb_full','lgb_full','cat_full'],style='median'),
 'ens_full_adapt_cov':dict(members=['xgb_full','lgb_full','cat_full'],style='adaptive_cov'),
 'ens_ol_equal':dict(members=['xgb_ol','lgb_ol','cat_role'],style='mean'),
 'ens_all_mean':dict(members=list(base.CFG),style='mean'),
 'ens_all_adapt_cov':dict(members=list(base.CFG),style='adaptive_cov'),
}
assert 'live_v2' not in str(base.ENSEMBLES)
if __name__=='__main__':base.main()