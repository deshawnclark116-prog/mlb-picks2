"""Prints a fingerprint of the team-environment and role-row construction (used by the cross-process determinism test)."""
import hashlib, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfl_phase1_data as P
import nfl_phase1_team_environment as TE

D = P.Data(sys.argv[1])
U = P.build(D)
h = hashlib.sha256()
for u in U:
    h.update(repr((u["key"], [(r["gid"], r["pos"], r["n_hist"]) for r in u["players"]])).encode())
out, fitted, rows = TE.evaluate(U)
print(h.hexdigest()[:16], out["targets"]["rushes"]["candidates"]["C2_poisson_glm"]["combined"]["mae"],
      out["targets"]["plays"]["candidates"]["C2_poisson_glm"]["combined"]["crps"])
