"""Refresh the code hashes (and body hash) of v2_comparator_manifest.json after the v2 logger code changed. Artifacts / residuals are NOT retrained; this script refuses if their hashes differ."""
import hashlib, json
from pathlib import Path
import nfl_phase1e_v2 as V2
P = V2.P1E / "v2_comparator_manifest.json"
m = json.loads(P.read_text())
for oc, a in m["artifacts"].items():
    assert V2.sha_file(V2.REPO / a["file"]) == a["sha256"] and V2.sha_file(V2.REPO / a["residual_file"]) == a["residual_sha256"], "frozen artifact changed"
m["code"] = {f: V2.sha_file(V2.REPO / f) for f in V2.CODE_FILES}
m["code_hash"] = V2.code_hash()
m.pop("manifest_body_sha256", None)
m["manifest_body_sha256"] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
P.write_text(json.dumps(m, indent=1))
print(m["code_hash"], m["manifest_body_sha256"])
