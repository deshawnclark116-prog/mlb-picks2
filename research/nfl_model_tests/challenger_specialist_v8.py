#!/usr/bin/env python3
"""Position-specialist no-market receiving-yardage estimators.

Learn WR, TE, RB conditional response functions as well as one global model;
predict each position with a fixed, training-size-shrunk blend. Player
position derives from prior games only. No future statistics or bookmaker data.
"""
import numpy as np
import challenger_tenure_v5 as ten

base=ten.base
oldshim=base.xgb

class Specialist:
    def __init__(self,models,features,kind):
        self.models=models;self.features=features;self.kind=kind
    def predict(self,matrix):
        out=np.asarray(self.models['global'].predict(matrix),dtype=np.float32)
        for pos in ('is_wr','is_te','is_rb'):
            if pos not in self.models:continue
            j=self.features.index(pos)
            ix=np.where(matrix.data[:,j]==1.)[0]
            if not len(ix):continue
            sm=oldshim.DMatrix(matrix.data[ix],feature_names=matrix.feature_names)
            custom=np.asarray(self.models[pos][0].predict(sm))
            alpha=self.models[pos][1]
            out[ix]=alpha*custom+(1-alpha)*out[ix]
        return out

class SpecialistShim:
    DMatrix=oldshim.DMatrix
    @staticmethod
    def train(params,dtrain,num_boost_round):
        if params.get('engine')!='specialist':
            return oldshim.train(params,dtrain,num_boost_round)
        engine=params.get('base_engine','lgb')
        child={'engine':engine,'leaves':params.get('leaves',15),
               'min_leaf':params.get('min_leaf',30)}
        if engine=='xgb':
            import nfl_yardage_v2 as v2
            child=dict(v2.PARAMS,nthread=2,max_depth=params.get('depth',3),min_child_weight=15)
        global_fit=oldshim.train(child,dtrain,num_boost_round)
        fits={'global':global_fit}
        for key in ('is_wr','is_te','is_rb'):
            if key not in dtrain.feature_names:continue
            ci=dtrain.feature_names.index(key)
            ix=np.where(dtrain.data[:,ci]==1.)[0]
            if len(ix)<400:continue
            sm=oldshim.DMatrix(dtrain.data[ix],label=dtrain.label[ix],feature_names=dtrain.feature_names)
            if dtrain.weight is not None:sm.set_weight(dtrain.weight[ix])
            sub_fit=oldshim.train(child,sm,num_boost_round)
            # Shrink specialists trained on limited player-games toward pooled.
            alpha=0.70 if len(ix)>=1000 else 0.45
            fits[key]=(sub_fit,alpha)
        return Specialist(fits,dtrain.feature_names,engine)

base.xgb=SpecialistShim
ROLE=ten.ROLE
FTN=ten.FTN
OL=ten.OL
FULL=ten.FULL
base.CFG={
 'split_lgb_role':{'cols':ROLE,'params':{'engine':'specialist','base_engine':'lgb','leaves':15,'min_leaf':30},'rounds':185},
 'split_lgb_full':{'cols':FULL,'params':{'engine':'specialist','base_engine':'lgb','leaves':15,'min_leaf':30},'rounds':185},
 'split_lgb_oline':{'cols':OL,'params':{'engine':'specialist','base_engine':'lgb','leaves':15,'min_leaf':30},'rounds':185},
 'split_xgb_role':{'cols':ROLE,'params':{'engine':'specialist','base_engine':'xgb','depth':3},'rounds':165},
 'split_xgb_full':{'cols':FULL,'params':{'engine':'specialist','base_engine':'xgb','depth':3},'rounds':165},
}
assert all('is_wr' in spec['cols'] for spec in base.CFG.values())
if __name__=='__main__':base.main()