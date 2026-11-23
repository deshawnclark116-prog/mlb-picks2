#!/usr/bin/env python3
"""Model-family challenger: XGB vs LightGBM vs CatBoost, no bookmaker inputs.

The legacy v2 comparator is trained by unmodified XGBoost with its historical
feature set/seed; no candidate contains spread, total, odds, or prices.
Every evaluation week is predicted using only earlier completed weeks.
"""
import numpy as np
import xgboost as real_xgb
import challenger_ftn_v2 as base
import nfl_yardage_v2 as v2

F={
 'base':base.BASE,
 'role':base.BASE+base.ROLE+base.TEAM+base.DEPTH,
 'role_inj':base.BASE+base.ROLE+base.TEAM+base.DEPTH+base.INJ,
 'ftn':base.BASE+base.ROLE+base.TEAM+base.DEPTH+base.PROPS+base.FTN,
 'all':base.BASE+base.ROLE+base.TEAM+base.DEPTH+base.PROPS+base.FTN+base.INJ,
}

def cfg(feats,backend,params=None,rounds=200,half_life=None):
    p={'engine':backend,**(params or {})}
    z={'cols':F[feats], 'params':p, 'rounds':rounds}
    if half_life:z['half_life']=half_life
    return z

base.CFG={
 'lgb_core':cfg('base','lgb',{'leaves':15,'min_leaf':24},185),
 'lgb_role':cfg('role','lgb',{'leaves':15,'min_leaf':24},185),
 'lgb_role_inj':cfg('role_inj','lgb',{'leaves':15,'min_leaf':24},185),
 'lgb_ftn':cfg('ftn','lgb',{'leaves':15,'min_leaf':24},185),
 'lgb_full':cfg('all','lgb',{'leaves':15,'min_leaf':24},185),
 'lgb_role_shallow':cfg('role','lgb',{'leaves':7,'min_leaf':35},200),
 'lgb_full_shallow':cfg('all','lgb',{'leaves':7,'min_leaf':35},200),
 'lgb_role_recent':cfg('role','lgb',{'leaves':15,'min_leaf':24},185,half_life=38),
 'lgb_full_recent':cfg('all','lgb',{'leaves':15,'min_leaf':24},185,half_life=38),
 'cb_role':cfg('role','cat',{'depth':5},240),
 'cb_full':cfg('all','cat',{'depth':5},240),
 'cb_full_shallow':cfg('all','cat',{'depth':4},275),
 'xgb_role_depth2':{'cols':F['role'],'params':dict(v2.PARAMS,nthread=2,max_depth=2,min_child_weight=12),'rounds':220},
}
assert all(not any(s in feature for s in ('spread','implied_team','total_line','odds','price'))
       for model in base.CFG.values() for feature in model['cols'])

class Matrix:
    def __init__(self,data,label=None,feature_names=None):
        self.data=np.asarray(data,dtype=np.float32)
        self.label=None if label is None else np.asarray(label,dtype=np.float32)
        self.feature_names=feature_names
        self.weight=None
    def set_weight(self,weight):self.weight=np.asarray(weight,dtype=np.float32)

class Model:
    def __init__(self,fit,kind):self.fit=fit;self.kind=kind
    def predict(self,d):return np.asarray(self.fit.predict(d.data),dtype=np.float32)

class GBDTShim:
    DMatrix=Matrix
    @staticmethod
    def train(params,dtrain,num_boost_round):
        engine=params.get('engine')
        if engine=='lgb':
            import lightgbm as lgb
            ds=lgb.Dataset(dtrain.data,label=dtrain.label,weight=dtrain.weight,feature_name=dtrain.feature_names,free_raw_data=False)
            h={
                'objective':'regression_l1','metric':'l1','verbosity':-1,
                'num_leaves':params.get('leaves',15),'min_data_in_leaf':params.get('min_leaf',25),
                'learning_rate':0.045,'feature_fraction':0.82,'bagging_fraction':0.82,
                'bagging_freq':1,'lambda_l2':7,'max_bin':127,'num_threads':2,
                'seed':v2.PARAMS['seed'],'force_col_wise':True,
            }
            fit=lgb.train(h,ds,num_boost_round=num_boost_round)
            return Model(fit,'lgb')
        if engine=='cat':
            import catboost as cb
            fit=cb.CatBoostRegressor(
                iterations=num_boost_round,loss_function='MAE',learning_rate=0.055,
                depth=params.get('depth',5),l2_leaf_reg=6,random_seed=v2.PARAMS['seed'],
                thread_count=2,verbose=False,allow_writing_files=False)
            fit.fit(dtrain.data,dtrain.label,sample_weight=dtrain.weight)
            return Model(fit,'cat')
        matrix=real_xgb.DMatrix(dtrain.data,label=dtrain.label,feature_names=dtrain.feature_names)
        if dtrain.weight is not None:matrix.set_weight(dtrain.weight)
        fit=real_xgb.train(params,matrix,num_boost_round=num_boost_round)
        return Model(fit,'xgb')

base.xgb=GBDTShim

if __name__=='__main__':base.main()