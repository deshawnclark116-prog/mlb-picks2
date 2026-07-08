#!/usr/bin/env python3
import argparse, csv, hashlib, io, os, sqlite3, time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SAVANT_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
FASTBALLS={"FF","SI","FC","FA"}; BREAKING={"SL","ST","CU","KC","SV","CS"}; OFFSPEED={"CH","FS","FO","SC","EP"}

def db_path():
    p=Path(os.environ.get("HR_MODEL_DATA_DIR","/data/hr_model"))
    return p/"hr_model.sqlite" if p.parent.exists() else Path("./hr_model/hr_model.sqlite")

def dt(s): return datetime.strptime(s,"%Y-%m-%d").date()
def chunks(a,b,n):
    c=a
    while c<=b:
        e=min(b,c+timedelta(days=n-1)); yield c,e; c=e+timedelta(days=1)
def ti(x):
    try: return None if x is None or str(x).strip()=="" else int(float(x))
    except: return None
def tf(x):
    try: return None if x is None or str(x).strip()=="" else float(x)
    except: return None
def tx(x):
    s="" if x is None else str(x).strip()
    return s or None
def clean_key(k):
    return (k or "").replace("\ufeff","").strip()
def clean_row(r):
    return {clean_key(k): v for k,v in r.items()}
def pg(pt):
    pt=tx(pt)
    if not pt: return "UNK"
    if pt in FASTBALLS: return "FB"
    if pt in BREAKING: return "BR"
    if pt in OFFSPEED: return "OS"
    return "OT"
def zb(px,pz):
    x=tf(px); z=tf(pz)
    if x is None or z is None: return None
    if -0.83<=x<=0.83 and 1.5<=z<=3.5:
        col=0 if x < -0.27 else (1 if x <= 0.27 else 2)
        row=0 if z < 2.16 else (1 if z <= 2.83 else 2)
        return row*3+col+1
    return 10
def key(r):
    # Exclude pitch_type from identity so corrected pitch_type parsing does not duplicate old smoke rows.
    parts=[r.get(k,"") for k in ("game_pk","game_date","at_bat_number","pitch_number","batter","pitcher","balls","strikes","plate_x","plate_z")]
    return hashlib.sha1("|".join(map(str,parts)).encode()).hexdigest()

def params(s,e):
    return {"all":"true","type":"details","player_type":"batter","hfGT":"R|","game_date_gt":s.isoformat(),"game_date_lt":e.isoformat(),
            "hfPT":"","hfAB":"","hfPR":"","hfZ":"","stadium":"","hfBBL":"","hfNewZones":"","hfPull":"","hfC":"","hfSea":"","hfSit":"",
            "hfOuts":"","opponent":"","pitcher_throws":"","batter_stands":"","hfSA":"","team":"","position":"","hfRO":"","home_road":"",
            "hfFlag":"","hfInn":"","min_pitches":"0","min_results":"0","group_by":"name","sort_col":"pitches","player_event_sort":"h_launch_speed","sort_order":"desc","min_pas":"0"}

def fetch(s,e):
    url=SAVANT_CSV_URL+"?"+urlencode(params(s,e))
    req=Request(url,headers={"User-Agent":"Mozilla/5.0 HR pitch loader","Accept":"text/csv,*/*"})
    with urlopen(req,timeout=90) as resp:
        body=resp.read().decode("utf-8-sig",errors="replace"); status=getattr(resp,"status",None)
    reader=csv.DictReader(io.StringIO(body))
    rows=[clean_row(r) for r in reader]
    return status, rows, [clean_key(x) for x in (reader.fieldnames or [])]

def schema(conn):
    with conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS statcast_pitches(
            pitch_key TEXT PRIMARY KEY, game_pk TEXT, game_date TEXT, at_bat_number INTEGER, pitch_number INTEGER,
            batter_id INTEGER, pitcher_id INTEGER, stand TEXT, p_throws TEXT,
            pitch_type TEXT, pitch_name TEXT, pitch_group TEXT, release_speed REAL, release_spin_rate REAL,
            plate_x REAL, plate_z REAL, zone INTEGER, zone_bucket INTEGER, balls INTEGER, strikes INTEGER,
            description TEXT, events TEXT, bb_type TEXT, launch_speed REAL, launch_angle REAL, launch_speed_angle INTEGER,
            estimated_woba_using_speedangle REAL, woba_value REAL,
            is_bbe INTEGER, is_hr INTEGER, is_barrel INTEGER, is_whiff INTEGER, is_called_strike INTEGER, raw_inserted_at TEXT)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pitches_date ON statcast_pitches(game_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pitches_batter_date ON statcast_pitches(batter_id,game_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pitches_pitcher_date ON statcast_pitches(pitcher_id,game_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pitches_group_zone_date ON statcast_pitches(pitch_group,zone_bucket,game_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pitches_game ON statcast_pitches(game_pk)")
INS="""INSERT OR REPLACE INTO statcast_pitches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""

def tup(r):
    ev=tx(r.get("events")); desc=tx(r.get("description")); bb=tx(r.get("bb_type")); ls=tf(r.get("launch_speed")); lsa=ti(r.get("launch_speed_angle"))
    is_bbe=1 if ls is not None or bb is not None or ev in {"home_run","single","double","triple","field_out","force_out","grounded_into_double_play"} else 0
    return (key(r),tx(r.get("game_pk")),tx(r.get("game_date")),ti(r.get("at_bat_number")),ti(r.get("pitch_number")),
            ti(r.get("batter")),ti(r.get("pitcher")),tx(r.get("stand")),tx(r.get("p_throws")),
            tx(r.get("pitch_type")),tx(r.get("pitch_name")),pg(r.get("pitch_type")),tf(r.get("release_speed")),tf(r.get("release_spin_rate")),
            tf(r.get("plate_x")),tf(r.get("plate_z")),ti(r.get("zone")),zb(r.get("plate_x"),r.get("plate_z")),ti(r.get("balls")),ti(r.get("strikes")),
            desc,ev,bb,ls,tf(r.get("launch_angle")),lsa,tf(r.get("estimated_woba_using_speedangle")),tf(r.get("woba_value")),
            is_bbe,1 if ev=="home_run" else 0,1 if lsa==6 else 0,1 if desc in {"swinging_strike","swinging_strike_blocked","foul_tip"} else 0,1 if desc=="called_strike" else 0,
            datetime.utcnow().isoformat(timespec="seconds")+"Z")
def summary(conn):
    print("SUMMARY")
    for label,sql in [("rows","SELECT COUNT(*) FROM statcast_pitches"),("dates","SELECT COUNT(DISTINCT game_date) FROM statcast_pitches"),
        ("range","SELECT MIN(game_date),MAX(game_date) FROM statcast_pitches"),("bbe","SELECT SUM(is_bbe) FROM statcast_pitches"),
        ("hr","SELECT SUM(is_hr) FROM statcast_pitches"),("barrels","SELECT SUM(is_barrel) FROM statcast_pitches"),
        ("pitch_type_nonnull","SELECT COUNT(*) FROM statcast_pitches WHERE pitch_type IS NOT NULL")]:
        print(label+":", conn.execute(sql).fetchone())
    print("PITCH GROUPS")
    for r in conn.execute("SELECT pitch_group,COUNT(*) FROM statcast_pitches GROUP BY pitch_group ORDER BY COUNT(*) DESC"): print(r)
    print("PITCH TYPES")
    for r in conn.execute("SELECT pitch_type,COUNT(*) FROM statcast_pitches GROUP BY pitch_type ORDER BY COUNT(*) DESC LIMIT 20"): print(r)
    print("LATEST DATES")
    for r in conn.execute("SELECT game_date,COUNT(*),SUM(is_bbe),SUM(is_hr) FROM statcast_pitches GROUP BY game_date ORDER BY game_date DESC LIMIT 10"): print(r)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",default=str(db_path())); ap.add_argument("--start-date"); ap.add_argument("--end-date")
    ap.add_argument("--chunk-days",type=int,default=1); ap.add_argument("--sleep",type=float,default=.35)
    ap.add_argument("--max-rows",type=int,default=0); ap.add_argument("--summary-only",action="store_true")
    ap.add_argument("--clear-range",action="store_true",help="Delete existing statcast_pitches rows between start and end before loading.")
    ap.add_argument("--show-columns",action="store_true")
    a=ap.parse_args()
    conn=sqlite3.connect(a.db); conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA synchronous=NORMAL"); schema(conn)
    print("HR_PITCH_STATCAST_LOADER_B"); print("=========================="); print("db:",a.db)
    if a.summary_only: summary(conn); conn.close(); return
    if not a.start_date or not a.end_date: raise SystemExit("Use --start-date/--end-date or --summary-only")
    start,end=dt(a.start_date),dt(a.end_date)
    if a.clear_range:
        with conn:
            n=conn.execute("DELETE FROM statcast_pitches WHERE game_date>=? AND game_date<=?",(start.isoformat(),end.isoformat())).rowcount
        print("cleared_existing_rows:",n)
    seen=inserted=bbe=hr=0; shown=False
    for s,e in chunks(start,end,a.chunk_days):
        print(f"\nFETCH {s} to {e}")
        try: status,rows,fields=fetch(s,e)
        except Exception as ex: print("ERROR",type(ex).__name__,ex); continue
        print("status=",status,"rows_seen=",len(rows))
        if a.show_columns and not shown:
            print("columns:", fields[:80])
            if rows: print("sample_pitch_type:", rows[0].get("pitch_type"), "sample_keys:", list(rows[0].keys())[:20])
            shown=True
        vals=[]
        for r in rows:
            t=tup(r); vals.append(t); seen+=1; bbe+=t[28]; hr+=t[29]
            if a.max_rows and seen>=a.max_rows: break
        if vals:
            with conn: conn.executemany(INS,vals)
        inserted+=len(vals)
        print("inserted_this_chunk=",len(vals),"total_seen=",seen,"insert_attempts=",inserted)
        if a.max_rows and seen>=a.max_rows: print("max_rows reached"); break
        time.sleep(a.sleep)
    print("\nLOAD SUMMARY"); print("rows_seen:",seen); print("insert_attempts:",inserted); print("bbe_seen:",bbe); print("hr_seen:",hr)
    summary(conn); conn.close(); print("\nDONE")
if __name__=="__main__": main()
