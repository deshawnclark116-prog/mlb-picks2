#!/usr/bin/env python3
"""
HR_OPPONENT_GATE_LADDER_C

Patch for: "Missing batter_games.pitcher_id"

This version does NOT require batter_games.pitcher_id.
It builds batter_game_pitcher_map from statcast_pitches by taking the first pitcher
each starting batter faced in that game, then uses that pitcher as the opposing SP
identity proxy for the blunt opponent gate.

Train: 2025 only
Test:  2026 only

A0 = ENVIRONMENT_C_LOCKED_A
B1 = A0 + one crude pitcher HR suppression index
B2 = B1 + K/GB/barrel/hard-hit/HR indices
B3 = B2 + pitch mix / whiff / CSW indices

Run:
  python hr_opponent_gate_ladder_c.py --rebuild-features
"""

import argparse, math, os, sqlite3
from pathlib import Path

RATE_TO_SAMPLE = {
    "batter_hr_per_air_60d": "batter_bbe_60d",
    "batter_fly_ball_rate_30d": "batter_bbe_60d",
    "batter_barrel_rate_30d": "batter_bbe_60d",
    "pitcher_pull_air_allowed_rate_60d": "pitcher_bbe_allowed_60d",
    "pitcher_hard_hit_allowed_rate_7d": "pitcher_bbe_allowed_60d",
    "pitcher_hrfb_rate_30d": "pitcher_bbe_allowed_60d",
}

BASE = ["expected_pa_v1","temp_f","batter_max_ev_60d","log_batter_bbe_60d","log_pitcher_bbe_allowed_60d"] + [f"{c}_shrunk" for c in RATE_TO_SAMPLE]
ENV = ["league_hr_per_bbe_10d_lag","league_hr_per_air_10d_lag","league_barrel_rate_10d_lag","league_avg_ev_10d_lag","log_league_bbe_10d"]
WX = ["temp_over_75","temp_over_85","wind_out_component","wind_out_and_hot"]
A0 = BASE + ENV + WX

OPP_SAMPLE = {
    "opp_hr_per_bbe_index_365d": "opp_bbe_365d",
    "opp_hr_per_pa_index_365d": "opp_pa_365d",
    "opp_k_per_pa_index_365d": "opp_pa_365d",
    "opp_barrel_per_bbe_index_365d": "opp_bbe_365d",
    "opp_hardhit_per_bbe_index_365d": "opp_bbe_365d",
    "opp_gb_per_bbe_index_365d": "opp_bbe_365d",
    "opp_air_per_bbe_index_365d": "opp_bbe_365d",
    "opp_fb_usage_index_365d": "opp_pitches_365d",
    "opp_br_usage_index_365d": "opp_pitches_365d",
    "opp_os_usage_index_365d": "opp_pitches_365d",
    "opp_whiff_per_pitch_index_365d": "opp_pitches_365d",
    "opp_csw_per_pitch_index_365d": "opp_pitches_365d",
}

B1 = A0 + ["opp_hr_per_bbe_index_365d_shrunk"]
B2 = B1 + [
    "opp_hr_per_pa_index_365d_shrunk","opp_k_per_pa_index_365d_shrunk",
    "opp_barrel_per_bbe_index_365d_shrunk","opp_hardhit_per_bbe_index_365d_shrunk",
    "opp_gb_per_bbe_index_365d_shrunk","opp_air_per_bbe_index_365d_shrunk",
    "log_opp_bbe_365d","log_opp_pa_365d",
]
B3 = B2 + [
    "opp_fb_usage_index_365d_shrunk","opp_br_usage_index_365d_shrunk",
    "opp_os_usage_index_365d_shrunk","opp_whiff_per_pitch_index_365d_shrunk",
    "opp_csw_per_pitch_index_365d_shrunk","log_opp_pitches_365d",
]

CONFIGS = {
    "A0_ENVIRONMENT_C_LOCKED": A0,
    "B1_ONE_HR_SUPPRESSION": B1,
    "B2_HR_K_GB_CONTACT": B2,
    "B3_PLUS_MIX_WHIFF_CSW": B3,
}

RAW = [
    "expected_pa_v1","temp_f","wind_toward_pull_field","weather_wind_mph",
    "batter_hr_per_air_60d","batter_fly_ball_rate_30d","batter_barrel_rate_30d","batter_max_ev_60d",
    "pitcher_pull_air_allowed_rate_60d","pitcher_hard_hit_allowed_rate_7d","pitcher_hrfb_rate_30d",
    "batter_bbe_60d","pitcher_bbe_allowed_60d",
    "league_hr_per_bbe_10d_lag","league_hr_per_air_10d_lag","league_barrel_rate_10d_lag","league_avg_ev_10d_lag",
    "league_bbe_10d","league_air_10d","opp_pitches_365d","opp_pa_365d","opp_bbe_365d"
] + list(OPP_SAMPLE)

def default_db():
    p = Path(os.environ.get("HR_MODEL_DATA_DIR","/data/hr_model"))
    return p/"hr_model.sqlite" if p.parent.exists() else Path("./hr_model/hr_model.sqlite")

def imports():
    try:
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return pd,ColumnTransformer,SimpleImputer,LogisticRegression,brier_score_loss,log_loss,roc_auc_score,Pipeline,StandardScaler
    except Exception as e:
        raise SystemExit("Install deps: pip install pandas numpy scikit-learn\n" + repr(e))

def cols(conn,t):
    try: return [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
    except Exception: return []

def require(conn):
    need = ["batter_games","statcast_pitches","batter_game_features","pitcher_game_features","league_env_lag_features"]
    miss = [t for t in need if not cols(conn,t)]
    if miss: raise SystemExit("Missing tables: " + ", ".join(miss))

def build_gate(conn,rebuild=False):
    require(conn)
    if rebuild:
        with conn:
            conn.execute("DROP TABLE IF EXISTS batter_game_pitcher_map")
            conn.execute("DROP TABLE IF EXISTS pitcher_daily_gate_stats")
            conn.execute("DROP TABLE IF EXISTS league_daily_gate_stats")
            conn.execute("DROP TABLE IF EXISTS opponent_gate_features")

    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS batter_game_pitcher_map AS
        WITH ranked AS (
            SELECT
                CAST(game_pk AS TEXT) AS game_id,
                batter_id,
                pitcher_id,
                game_date,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(game_pk AS TEXT), batter_id
                    ORDER BY COALESCE(at_bat_number, 9999), COALESCE(pitch_number, 9999)
                ) AS rn
            FROM statcast_pitches
            WHERE game_pk IS NOT NULL
              AND batter_id IS NOT NULL
              AND pitcher_id IS NOT NULL
              AND game_date IS NOT NULL
        )
        SELECT game_id, batter_id, pitcher_id, game_date
        FROM ranked
        WHERE rn=1
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bgpm_game_batter ON batter_game_pitcher_map(game_id,batter_id)")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS pitcher_daily_gate_stats AS
        SELECT game_date,pitcher_id,
               COUNT(*) pitches,
               SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END) pa,
               SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) k,
               SUM(CASE WHEN pitch_group='FB' THEN 1 ELSE 0 END) fb,
               SUM(CASE WHEN pitch_group='BR' THEN 1 ELSE 0 END) br,
               SUM(CASE WHEN pitch_group='OS' THEN 1 ELSE 0 END) os,
               SUM(CASE WHEN is_whiff=1 THEN 1 ELSE 0 END) whiff,
               SUM(CASE WHEN is_called_strike=1 THEN 1 ELSE 0 END) called,
               SUM(CASE WHEN is_bbe=1 THEN 1 ELSE 0 END) bbe,
               SUM(CASE WHEN is_hr=1 THEN 1 ELSE 0 END) hr,
               SUM(CASE WHEN is_barrel=1 THEN 1 ELSE 0 END) barrel,
               SUM(CASE WHEN launch_speed>=95 THEN 1 ELSE 0 END) hardhit,
               SUM(CASE WHEN bb_type='ground_ball' THEN 1 ELSE 0 END) gb,
               SUM(CASE WHEN bb_type IN ('fly_ball','line_drive','popup') THEN 1 ELSE 0 END) air
        FROM statcast_pitches
        WHERE pitcher_id IS NOT NULL AND game_date IS NOT NULL
        GROUP BY game_date,pitcher_id
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS league_daily_gate_stats AS
        SELECT game_date,
               COUNT(*) pitches,
               SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END) pa,
               SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) k,
               SUM(CASE WHEN pitch_group='FB' THEN 1 ELSE 0 END) fb,
               SUM(CASE WHEN pitch_group='BR' THEN 1 ELSE 0 END) br,
               SUM(CASE WHEN pitch_group='OS' THEN 1 ELSE 0 END) os,
               SUM(CASE WHEN is_whiff=1 THEN 1 ELSE 0 END) whiff,
               SUM(CASE WHEN is_called_strike=1 THEN 1 ELSE 0 END) called,
               SUM(CASE WHEN is_bbe=1 THEN 1 ELSE 0 END) bbe,
               SUM(CASE WHEN is_hr=1 THEN 1 ELSE 0 END) hr,
               SUM(CASE WHEN is_barrel=1 THEN 1 ELSE 0 END) barrel,
               SUM(CASE WHEN launch_speed>=95 THEN 1 ELSE 0 END) hardhit,
               SUM(CASE WHEN bb_type='ground_ball' THEN 1 ELSE 0 END) gb,
               SUM(CASE WHEN bb_type IN ('fly_ball','line_drive','popup') THEN 1 ELSE 0 END) air
        FROM statcast_pitches
        WHERE game_date IS NOT NULL
        GROUP BY game_date
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pdg_pd ON pitcher_daily_gate_stats(pitcher_id,game_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lgd_d ON league_daily_gate_stats(game_date)")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS opponent_gate_features(
            game_id TEXT,batter_id INTEGER,game_date TEXT,pitcher_id INTEGER,
            opp_pitches_365d REAL,opp_pa_365d REAL,opp_bbe_365d REAL,
            opp_hr_per_bbe_index_365d REAL,opp_hr_per_pa_index_365d REAL,opp_k_per_pa_index_365d REAL,
            opp_barrel_per_bbe_index_365d REAL,opp_hardhit_per_bbe_index_365d REAL,
            opp_gb_per_bbe_index_365d REAL,opp_air_per_bbe_index_365d REAL,
            opp_fb_usage_index_365d REAL,opp_br_usage_index_365d REAL,opp_os_usage_index_365d REAL,
            opp_whiff_per_pitch_index_365d REAL,opp_csw_per_pitch_index_365d REAL,
            PRIMARY KEY(game_id,batter_id))
        """)
        conn.execute("DELETE FROM opponent_gate_features")
        conn.execute("""
        INSERT OR REPLACE INTO opponent_gate_features
        WITH bgx AS (
          SELECT
            CAST(bg.game_id AS TEXT) AS game_id,
            bg.batter_id,
            bg.game_date,
            bpm.pitcher_id
          FROM batter_games bg
          LEFT JOIN batter_game_pitcher_map bpm
            ON CAST(bg.game_id AS TEXT)=bpm.game_id
           AND bg.batter_id=bpm.batter_id
          WHERE bg.actual_hr IS NOT NULL
        ),
        pr AS (
          SELECT bgx.game_id,bgx.batter_id,
                 COALESCE(SUM(p.pitches),0) pp,COALESCE(SUM(p.pa),0) pa,COALESCE(SUM(p.bbe),0) bbe,
                 COALESCE(SUM(p.hr),0) hr,COALESCE(SUM(p.k),0) k,COALESCE(SUM(p.barrel),0) barrel,
                 COALESCE(SUM(p.hardhit),0) hardhit,COALESCE(SUM(p.gb),0) gb,COALESCE(SUM(p.air),0) air,
                 COALESCE(SUM(p.fb),0) fb,COALESCE(SUM(p.br),0) br,COALESCE(SUM(p.os),0) os,
                 COALESCE(SUM(p.whiff),0) whiff,COALESCE(SUM(p.called),0) called
          FROM bgx
          LEFT JOIN pitcher_daily_gate_stats p ON p.pitcher_id=bgx.pitcher_id
             AND p.game_date<bgx.game_date AND p.game_date>=DATE(bgx.game_date,'-365 day')
          GROUP BY bgx.game_id,bgx.batter_id
        ),
        lr AS (
          SELECT bgx.game_id,bgx.batter_id,
                 COALESCE(SUM(l.pitches),0) pp,COALESCE(SUM(l.pa),0) pa,COALESCE(SUM(l.bbe),0) bbe,
                 COALESCE(SUM(l.hr),0) hr,COALESCE(SUM(l.k),0) k,COALESCE(SUM(l.barrel),0) barrel,
                 COALESCE(SUM(l.hardhit),0) hardhit,COALESCE(SUM(l.gb),0) gb,COALESCE(SUM(l.air),0) air,
                 COALESCE(SUM(l.fb),0) fb,COALESCE(SUM(l.br),0) br,COALESCE(SUM(l.os),0) os,
                 COALESCE(SUM(l.whiff),0) whiff,COALESCE(SUM(l.called),0) called
          FROM bgx
          LEFT JOIN league_daily_gate_stats l ON l.game_date<bgx.game_date
             AND l.game_date>=DATE(bgx.game_date,'-365 day')
          GROUP BY bgx.game_id,bgx.batter_id
        )
        SELECT bgx.game_id,bgx.batter_id,bgx.game_date,bgx.pitcher_id,
               pr.pp,pr.pa,pr.bbe,
               CASE WHEN pr.bbe>0 AND lr.bbe>0 AND (1.0*lr.hr/lr.bbe)>0 THEN (1.0*pr.hr/pr.bbe)/(1.0*lr.hr/lr.bbe) END,
               CASE WHEN pr.pa>0 AND lr.pa>0 AND (1.0*lr.hr/lr.pa)>0 THEN (1.0*pr.hr/pr.pa)/(1.0*lr.hr/lr.pa) END,
               CASE WHEN pr.pa>0 AND lr.pa>0 AND (1.0*lr.k/lr.pa)>0 THEN (1.0*pr.k/pr.pa)/(1.0*lr.k/lr.pa) END,
               CASE WHEN pr.bbe>0 AND lr.bbe>0 AND (1.0*lr.barrel/lr.bbe)>0 THEN (1.0*pr.barrel/pr.bbe)/(1.0*lr.barrel/lr.bbe) END,
               CASE WHEN pr.bbe>0 AND lr.bbe>0 AND (1.0*lr.hardhit/lr.bbe)>0 THEN (1.0*pr.hardhit/pr.bbe)/(1.0*lr.hardhit/lr.bbe) END,
               CASE WHEN pr.bbe>0 AND lr.bbe>0 AND (1.0*lr.gb/lr.bbe)>0 THEN (1.0*pr.gb/pr.bbe)/(1.0*lr.gb/lr.bbe) END,
               CASE WHEN pr.bbe>0 AND lr.bbe>0 AND (1.0*lr.air/lr.bbe)>0 THEN (1.0*pr.air/pr.bbe)/(1.0*lr.air/lr.bbe) END,
               CASE WHEN pr.pp>0 AND lr.pp>0 AND (1.0*lr.fb/lr.pp)>0 THEN (1.0*pr.fb/pr.pp)/(1.0*lr.fb/lr.pp) END,
               CASE WHEN pr.pp>0 AND lr.pp>0 AND (1.0*lr.br/lr.pp)>0 THEN (1.0*pr.br/pr.pp)/(1.0*lr.br/lr.pp) END,
               CASE WHEN pr.pp>0 AND lr.pp>0 AND (1.0*lr.os/lr.pp)>0 THEN (1.0*pr.os/pr.pp)/(1.0*lr.os/lr.pp) END,
               CASE WHEN pr.pp>0 AND lr.pp>0 AND (1.0*lr.whiff/lr.pp)>0 THEN (1.0*pr.whiff/pr.pp)/(1.0*lr.whiff/lr.pp) END,
               CASE WHEN pr.pp>0 AND lr.pp>0 AND (1.0*(lr.whiff+lr.called)/lr.pp)>0 THEN (1.0*(pr.whiff+pr.called)/pr.pp)/(1.0*(lr.whiff+lr.called)/lr.pp) END
        FROM bgx
        LEFT JOIN pr ON pr.game_id=bgx.game_id AND pr.batter_id=bgx.batter_id
        LEFT JOIN lr ON lr.game_id=bgx.game_id AND lr.batter_id=bgx.batter_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ogf_game ON opponent_gate_features(game_id,batter_id)")

def expr(conn,alias,cands):
    tables=[("bg",set(cols(conn,"batter_games"))),("bf",set(cols(conn,"batter_game_features"))),("pf",set(cols(conn,"pitcher_game_features"))),("le",set(cols(conn,"league_env_lag_features"))),("og",set(cols(conn,"opponent_gate_features")))]
    for c in cands:
        for t,cs in tables:
            if c in cs: return f"{t}.{c} AS {alias}"
    return f"NULL AS {alias}"

def load(conn,pd):
    cmap={
        "expected_pa_v1":["expected_pa_v1"],"temp_f":["temp_f","weather_temp_f"],"wind_toward_pull_field":["wind_toward_pull_field"],"weather_wind_mph":["weather_wind_mph","wind_mph"],
        "batter_hr_per_air_60d":["batter_hr_per_air_60d"],"batter_fly_ball_rate_30d":["batter_fly_ball_rate_30d"],"batter_barrel_rate_30d":["batter_barrel_rate_30d"],"batter_max_ev_60d":["batter_max_ev_60d"],
        "pitcher_pull_air_allowed_rate_60d":["pitcher_pull_air_allowed_rate_60d"],"pitcher_hard_hit_allowed_rate_7d":["pitcher_hard_hit_allowed_rate_7d"],"pitcher_hrfb_rate_30d":["pitcher_hrfb_rate_30d"],
        "batter_bbe_60d":["batter_bbe_60d"],"pitcher_bbe_allowed_60d":["pitcher_bbe_allowed_60d"],
        "league_hr_per_bbe_10d_lag":["league_hr_per_bbe_10d_lag"],"league_hr_per_air_10d_lag":["league_hr_per_air_10d_lag"],"league_barrel_rate_10d_lag":["league_barrel_rate_10d_lag"],"league_avg_ev_10d_lag":["league_avg_ev_10d_lag"],"league_bbe_10d":["league_bbe_10d"],"league_air_10d":["league_air_10d"],
        "opp_pitches_365d":["opp_pitches_365d"],"opp_pa_365d":["opp_pa_365d"],"opp_bbe_365d":["opp_bbe_365d"],
    }
    for c in OPP_SAMPLE: cmap[c]=[c]
    bg=set(cols(conn,"batter_games"))
    def opt(c): return f"bg.{c} AS {c}" if c in bg else f"NULL AS {c}"
    sel=["bg.game_date AS game_date","bg.game_id AS game_id","bg.batter_id AS batter_id","og.pitcher_id AS pitcher_id","bg.batter_name AS batter_name",opt("team"),opt("opponent"),opt("venue"),"bg.lineup_spot AS lineup_spot","bg.actual_hr AS actual_hr"]
    for a,c in cmap.items(): sel.append(expr(conn,a,c))
    sql=f"""
    SELECT {", ".join(sel)}
    FROM batter_games bg
    LEFT JOIN batter_game_features bf ON bg.game_id=bf.game_id AND bg.batter_id=bf.batter_id
    LEFT JOIN pitcher_game_features pf ON bg.game_id=pf.game_id AND bg.batter_id=pf.batter_id
    LEFT JOIN league_env_lag_features le ON bg.game_date=le.game_date
    LEFT JOIN opponent_gate_features og ON CAST(bg.game_id AS TEXT)=og.game_id AND bg.batter_id=og.batter_id
    WHERE bg.actual_hr IS NOT NULL
    ORDER BY bg.game_date,bg.game_id,bg.lineup_spot
    """
    return pd.read_sql_query(sql,conn)

def boolish(v):
    if v is None: return 0.0
    if isinstance(v,(int,float)):
        try: return 1.0 if float(v)>0 else 0.0
        except Exception: return 0.0
    return 1.0 if str(v).lower().strip() in ("1","true","t","yes","y","out","toward","pull","wind_out","wind toward pull") else 0.0

def shrink(train,test,rate,sample,out,k,hi):
    prior=float(train[rate].dropna().median()) if train[rate].notna().sum() else (1.0 if "index" in rate else 0.0)
    for df in (train,test):
        raw=df[rate].fillna(prior).clip(lower=0,upper=hi)
        n=df[sample].fillna(0).clip(lower=0)
        w=n/(n+k)
        df[out]=prior+(raw-prior)*w

def engineer(train,test,bk,ok):
    for df in (train,test):
        df["log_batter_bbe_60d"]=df["batter_bbe_60d"].fillna(0).clip(lower=0).map(math.log1p)
        df["log_pitcher_bbe_allowed_60d"]=df["pitcher_bbe_allowed_60d"].fillna(0).clip(lower=0).map(math.log1p)
        df["log_league_bbe_10d"]=df["league_bbe_10d"].fillna(0).clip(lower=0).map(math.log1p)
        df["temp_over_75"]=(df["temp_f"].fillna(70)-75).clip(lower=0)
        df["temp_over_85"]=(df["temp_f"].fillna(70)-85).clip(lower=0)
        wind=df["wind_toward_pull_field"].map(boolish)
        mph=df["weather_wind_mph"].fillna(0).clip(lower=0,upper=60)
        df["wind_out_component"]=wind*mph
        df["wind_out_and_hot"]=df["wind_out_component"]*df["temp_over_75"]
        df["log_opp_pitches_365d"]=df["opp_pitches_365d"].fillna(0).clip(lower=0).map(math.log1p)
        df["log_opp_pa_365d"]=df["opp_pa_365d"].fillna(0).clip(lower=0).map(math.log1p)
        df["log_opp_bbe_365d"]=df["opp_bbe_365d"].fillna(0).clip(lower=0).map(math.log1p)
    for r,s in RATE_TO_SAMPLE.items(): shrink(train,test,r,s,f"{r}_shrunk",bk,1.0)
    for r,s in OPP_SAMPLE.items(): shrink(train,test,r,s,f"{r}_shrunk",ok,3.0)

def mkmodel(C,features,CT,Imp,LR,Pipe,Scale):
    pre=CT([("num",Pipe([("imp",Imp(strategy="median")),("sc",Scale())]),features)],remainder="drop")
    return Pipe([("pre",pre),("clf",LR(max_iter=2000,solver="lbfgs",C=C))])

def safe(fn,y,p):
    try: return float(fn(y,p))
    except Exception: return None
def fmt(x): return "None" if x is None else f"{float(x):.8f}"

def run_one(name,features,train,test,args,CT,Imp,LR,brier,logloss,auc,Pipe,Scale):
    m=mkmodel(args.c,features,CT,Imp,LR,Pipe,Scale)
    m.fit(train[features],train["actual_hr"].astype(int))
    p=m.predict_proba(test[features])[:,1]
    out=test.copy(); out["model_prob"]=p
    y=out["actual_hr"].astype(int).tolist()
    br=sum(y)/len(y); bp=[br]*len(y)
    cut=out["model_prob"].quantile(.95); top=out[out["model_prob"]>=cut]
    return {"name":name,"features":features,"model":m,"out":out,
            "brier":safe(brier,y,p.tolist()),"base_brier":safe(brier,y,bp),
            "logloss":safe(logloss,y,p.tolist()),"base_logloss":safe(logloss,y,bp),
            "auc":safe(auc,y,p.tolist()),"top5_cutoff":float(cut),"top5_rows":int(len(top)),
            "top5_pred":float(top["model_prob"].mean()),"top5_actual":float(top["actual_hr"].mean()),"top5_hits":int(top["actual_hr"].sum())}

def monthly(out):
    d=out.copy(); d["month"]=d["game_date"].dt.strftime("%Y-%m")
    for m,g in d.groupby("month"):
        print(f"{m} rows={len(g):5d} actual={g['actual_hr'].mean():.4f} pred={g['model_prob'].mean():.4f} miss={g['model_prob'].mean()-g['actual_hr'].mean():+.4f}")

def coefficients(r):
    try:
        pairs=sorted(zip(r["features"],r["model"].named_steps["clf"].coef_[0]),key=lambda x:abs(x[1]),reverse=True)
        for n,c in pairs[:35]: print(f"{n}: {c:+.5f}")
    except Exception as e: print("coef error:",e)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",default=str(default_db()))
    ap.add_argument("--c",type=float,default=.03)
    ap.add_argument("--batter-shrink-k",type=float,default=75.0)
    ap.add_argument("--opp-shrink-k",type=float,default=200.0)
    ap.add_argument("--rebuild-features",action="store_true")
    args=ap.parse_args()

    pd,CT,Imp,LR,brier,logloss,auc,Pipe,Scale=imports()
    conn=sqlite3.connect(args.db); conn.execute("PRAGMA journal_mode=WAL")
    print("HR_OPPONENT_GATE_LADDER_C")
    print("=========================")
    print("db:",args.db)
    print("building/checking opponent gate features from statcast first-pitcher map...")
    build_gate(conn,args.rebuild_features)

    print("\nJOIN / COVERAGE")
    print("---------------")
    print("batter_game_pitcher_map:",conn.execute("SELECT COUNT(*),MIN(game_date),MAX(game_date) FROM batter_game_pitcher_map").fetchone())
    print("opponent_gate_features:",conn.execute("SELECT COUNT(*),MIN(game_date),MAX(game_date) FROM opponent_gate_features").fetchone())
    print("mapped pitcher rows:",conn.execute("SELECT COUNT(*) FROM opponent_gate_features WHERE pitcher_id IS NOT NULL").fetchone()[0])
    print("zero opponent pitch history rows:",conn.execute("SELECT COUNT(*) FROM opponent_gate_features WHERE COALESCE(opp_pitches_365d,0)<=0").fetchone()[0])
    print("statcast_pitches:",conn.execute("SELECT COUNT(*),MIN(game_date),MAX(game_date) FROM statcast_pitches").fetchone())
    print("2025 pitch rows:",conn.execute("SELECT COUNT(*) FROM statcast_pitches WHERE game_date BETWEEN '2025-03-01' AND '2025-10-01'").fetchone()[0])
    print("2026 pitch rows:",conn.execute("SELECT COUNT(*) FROM statcast_pitches WHERE game_date BETWEEN '2026-03-01' AND '2026-12-31'").fetchone()[0])

    df=load(conn,pd); conn.close()
    df["game_date"]=pd.to_datetime(df["game_date"]); df["actual_hr"]=df["actual_hr"].astype(int)
    for c in RAW:
        if c!="wind_toward_pull_field": df[c]=pd.to_numeric(df[c],errors="coerce")

    train=df[df["game_date"].dt.year==2025].copy().sort_values("game_date")
    test=df[df["game_date"].dt.year==2026].copy().sort_values("game_date")
    engineer(train,test,args.batter_shrink_k,args.opp_shrink_k)

    print("train_rows_2025:",len(train))
    print("test_rows_2026:",len(test))
    print("test_actual_hr_rate:",f"{test['actual_hr'].mean():.5f}")
    print("2026 opponent zero-pitch rows:",int((test["opp_pitches_365d"].fillna(0)<=0).sum()))
    print("2026 opp_pitches median/p75/p95:",f"{test['opp_pitches_365d'].median():.1f}",f"{test['opp_pitches_365d'].quantile(.75):.1f}",f"{test['opp_pitches_365d'].quantile(.95):.1f}")

    results=[]
    for name,features in CONFIGS.items():
        print(f"\nRUNNING {name}")
        print("-"*(8+len(name)))
        r=run_one(name,features,train,test,args,CT,Imp,LR,brier,logloss,auc,Pipe,Scale)
        results.append(r)
        print("features_n:",len(features))
        print("brier:",fmt(r["brier"]),"baseline:",fmt(r["base_brier"]))
        print("logloss:",fmt(r["logloss"]),"baseline:",fmt(r["base_logloss"]))
        print("auc:",fmt(r["auc"]))
        print("top5:",f"cutoff={r['top5_cutoff']:.5f}",f"rows={r['top5_rows']}",f"mean_pred={r['top5_pred']:.5f}",f"actual={r['top5_actual']:.5f}",f"hits={r['top5_hits']}")

    print("\nSUMMARY COMPARISON")
    print("==================")
    print("name | brier | logloss | auc | top5_actual | top5_hits")
    for r in results:
        print(f"{r['name']} | {fmt(r['brier'])} | {fmt(r['logloss'])} | {fmt(r['auc'])} | {r['top5_actual']:.5f} | {r['top5_hits']}")

    a=results[0]
    print("\nDELTAS VS A0")
    print("------------")
    for r in results[1:]:
        print(f"{r['name']} brier_delta={r['brier']-a['brier']:+.8f} logloss_delta={r['logloss']-a['logloss']:+.8f} auc_delta={r['auc']-a['auc']:+.8f} top5_delta={r['top5_actual']-a['top5_actual']:+.5f}")

    for r in results:
        print(f"\nMONTHLY {r['name']}")
        print("----------------" + "-"*len(r["name"]))
        monthly(r["out"])

    print("\nTOP COEFFICIENTS: B3_PLUS_MIX_WHIFF_CSW")
    print("---------------------------------------")
    coefficients(results[-1])

    print("\nREAD")
    print("----")
    print("This run used a statcast-derived first-pitcher map because batter_games.pitcher_id was missing.")
    print("B1 is the one-column pitcher HR suppression sanity gate.")
    print("B2 adds K/GB/barrel/hard-hit/HR pitcher scalars.")
    print("B3 adds arsenal mix, whiff, and CSW.")
    print("If B1/B2/B3 improve holdout, pitcher signal is validated and pitch-type/zone overlap earns the next slot.")
    print("If they do not improve holdout, debug map coverage, rolling window, or load 2024 pitch data before judging the idea dead.")

if __name__=="__main__":
    main()
