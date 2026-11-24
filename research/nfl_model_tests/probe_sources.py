#!/usr/bin/env python3
import csv,io,os,subprocess
from pathlib import Path
P=Path("/tmp/nflfeatures");P.mkdir(exist_ok=True)
files={
 "ftn_charting_2025.csv":"ftn_charting/ftn_charting_2025.csv",
 "ftn_charting_2026.csv":"ftn_charting/ftn_charting_2026.csv",
 "snap_counts_2025.csv":"snap_counts/snap_counts_2025.csv",
 "injuries_2025.csv":"injuries/injuries_2025.csv",
 "depth_charts_2025.csv":"depth_charts/depth_charts_2025.csv",
 "depth_charts_2026.csv":"depth_charts/depth_charts_2026.csv",
}
for name,link in files.items():
 url="https://github.com/nflverse/nflverse-data/releases/download/"+link
 out=P/name
 p=subprocess.run(["curl","-fsSL","--max-time","60","--retry","2",url,"-o",str(out)],capture_output=True,text=True)
 print(name, "download exit",p.returncode, "bytes",out.stat().st_size if out.exists() else None, flush=True)
 if p.returncode:print(p.stderr[:300],flush=True);continue
 with out.open(newline="") as f:
  reader=csv.DictReader(f)
  try:
   row=next(reader); print("COLUMNS",list(row),flush=True)
   print("SAMPLE",{k:row[k] for k in list(row)[:25]},flush=True)
  except Exception as e:print("cannot read",repr(e),flush=True)
