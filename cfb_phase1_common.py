"""
CFB_PHASE1_COMMON -- research cutoff enforcement, constants and distribution metrics for the CFB Outcome Engine v1 (RESEARCH / SHADOW ONLY; independent of the NFL / NHL engines).

RESEARCH CUTOFF: only 2026 Week <= 4 completed games may ever exist in this rebuild, and 2026 Weeks 1-4 are BURNED_DIAGNOSTIC_ONLY (never fit / selection / calibration). 2026 Week >= 5 is FORBIDDEN_FROM_MODEL_RESEARCH:
`assert_research_allowed` raises for it, and the frozen research tables contain NO 2026 row at all.
"""
import hashlib
import math

import numpy as np

RESEARCH_MAX = (2026, 4)                      # last (season, week) that may exist in the research universe
FORBIDDEN_FROM = (2026, 5)
WARMUP_SEASONS = [2018]                       # history only
TARGET_SEASONS = list(range(2019, 2026))
DEV_FOLDS = [("D1", [2019, 2020], 2021), ("D2", [2019, 2020, 2021], 2022), ("D3", [2019, 2020, 2021, 2022], 2023), ("D4", [2019, 2020, 2021, 2022, 2023], 2024)]
LATE_CONFIRMATION_SEASON = 2025               # LATE_PERIOD_CONFIRMATION_PREVIOUSLY_EXPOSED
BURNED_DIAGNOSTIC = {"season": 2026, "weeks": [1, 2, 3, 4]}
SPORTSBOOK_WORDS = ("odds", "spread", "moneyline_odds", "over_under", "sportsbook", "bookmaker", "vegas", "implied_prob", "prop_line", "market_line", "closing_line", "pickcenter", "againstthespread", "fanduel", "draftkings")
MIN_VALID_PLAYS, MAX_VALID_PLAYS = 30, 140     # play-coverage validity (protocol amendment 1): a team-game outside this range has incomplete attributed-play coverage
BOOT_SEED = 20261003
BOOT_REPS = 10000


class ResearchCutoffError(RuntimeError):
    pass


class Research2025Error(RuntimeError):
    pass


FREEZE_FILE = "cfb_phase1_freeze_candidate.json"


def freeze_committed(repo=None):
    """True iff cfb_models/cfb_outcome_engine/cfb_phase1_freeze_candidate.json exists AND is committed with a clean working copy."""
    import subprocess
    from pathlib import Path
    repo = Path(repo) if repo else Path(__file__).resolve().parent
    rel = f"cfb_models/cfb_outcome_engine/{FREEZE_FILE}"
    if not (repo / rel).exists():
        return False
    log = subprocess.run(["git", "log", "--format=%H", "-1", "--", rel], cwd=repo, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=repo, capture_output=True, text=True).stdout.strip()
    return bool(log) and not dirty


def guard_2025(seasons, repo=None):
    """No 2025 access before the freeze artifact is committed (development code requests seasons <= 2024 only)."""
    if any(int(s) >= 2025 for s in seasons) and not freeze_committed(repo):
        raise Research2025Error("2025 is reserved for the ONE integrated confirmation: cfb_phase1_freeze_candidate.json must be committed first")
    return True


def assert_research_allowed(season, week, purpose="fit"):
    """Raise for any row beyond the research cutoff; 2026 Weeks 1-4 may only be used for purpose='diagnostic'."""
    s, w = int(season), int(week)
    if (s, w) >= FORBIDDEN_FROM:
        raise ResearchCutoffError(f"season {s} week {w} is FORBIDDEN_FROM_MODEL_RESEARCH (cutoff: 2026 Week {RESEARCH_MAX[1]})")
    if s == 2026 and purpose != "diagnostic":
        raise ResearchCutoffError(f"2026 Week {w} is BURNED_DIAGNOSTIC_ONLY: not usable for {purpose}")
    return True


def filter_research_rows(rows, purpose="fit", strict=True):
    """Keep rows allowed for `purpose`. strict=True raises on a forbidden row (loader behaviour); strict=False drops them."""
    out = []
    for r in rows:
        try:
            assert_research_allowed(r["season"], r["week"], purpose)
            out.append(r)
        except ResearchCutoffError:
            if strict:
                raise
    return out


def no_sportsbook_columns(names):
    bad = [n for n in names if any(w in str(n).lower() for w in SPORTSBOOK_WORDS)]
    if bad:
        raise ValueError(f"sportsbook-like field(s) forbidden in predictive inputs: {bad}")
    return True


def coverage_valid(r):
    """Team-game play coverage is valid iff play rows exist and the attributed-play total (rush + pass attempt + sack plays) lies in [30, 140]. Invalid games are MISSING DATA (not zeros): excluded from targets, accumulators and role windows."""
    return bool(r.get("has_play_rows")) and r.get("plays") is not None and MIN_VALID_PLAYS <= r["plays"] <= MAX_VALID_PLAYS


def week_index(season, week):
    """Monotone calendar-block index: seasons are far apart so blocks never straddle seasons."""
    return int(season) * 100 + int(week)


# ------------------------------------------------------------------ distribution metrics (PMF matrices over 0..K)
NLL_FLOOR = 1e-15


def nb2_pmf_matrix(mu, alpha, K):
    """PMF[n, K+1] of NB2(mu, alpha) with the unreported tail returned as sf[n] (survival beyond K)."""
    from scipy import stats
    mu = np.maximum(np.asarray(mu, float), 1e-9)
    r = 1.0 / alpha
    p = r / (r + mu)
    ks = np.arange(K + 1)[None, :]
    return stats.nbinom.pmf(ks, r, p[:, None]), stats.nbinom.sf(K, r, p)


def crps_rows(pmf, y):
    cdf = np.cumsum(pmf, axis=1)
    K = pmf.shape[1] - 1
    ind = (np.asarray(y)[:, None] <= np.arange(K + 1)[None, :]).astype(float)
    return ((cdf - ind) ** 2).sum(axis=1)


def nll_rows(pmf, y):
    py = pmf[np.arange(len(y)), np.minimum(np.asarray(y), pmf.shape[1] - 1)]
    return -np.log(np.maximum(py, NLL_FLOOR))


def pit_v(model_id, keys):
    """Deterministic randomisation v in [0,1): first 8 bytes of SHA256(model_id|key) / 2^64 (no global RNG)."""
    out = np.empty(len(keys))
    for i, k in enumerate(keys):
        out[i] = int.from_bytes(hashlib.sha256(f"{model_id}|{k}".encode()).digest()[:8], "big") / 2.0 ** 64
    return out


def randomized_pit(pmf, y, v):
    cdf = np.cumsum(pmf, axis=1)
    ar = np.arange(len(y)); yy = np.minimum(np.asarray(y), pmf.shape[1] - 1)
    below = np.where(yy > 0, cdf[ar, np.maximum(yy - 1, 0)], 0.0)
    return below + v * pmf[ar, yy]


def ks_uniform(u):
    u = np.sort(u); n = len(u); i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def pmf_mean(pmf):
    return pmf @ np.arange(pmf.shape[1])


def pmf_quantile(pmf, q):
    cdf = np.cumsum(pmf, axis=1)
    return (cdf >= q - 1e-12).argmax(axis=1)


def block_bootstrap(delta, block_ids, reps=BOOT_REPS, seed=BOOT_SEED, block_len=2):
    """Moving-block bootstrap over ordered calendar blocks (weeks): `delta` per unit, `block_ids` = week index. Blocks are `block_len` consecutive observed weeks within the same season. Returns bootstrap means."""
    delta = np.asarray(delta, float); bid = np.asarray(block_ids)
    weeks = np.unique(bid)
    pos = {int(w): i for i, w in enumerate(weeks)}
    S = np.zeros(len(weeks)); N = np.zeros(len(weeks))
    for d, b in zip(delta, bid):
        S[pos[int(b)]] += d; N[pos[int(b)]] += 1
    starts = [i for i in range(len(weeks) - block_len + 1) if all(weeks[i + j] // 100 == weeks[i] // 100 and weeks[i + j] == weeks[i] + j for j in range(block_len))]
    if not starts:
        raise ValueError("no valid blocks")
    nb = int(math.ceil(len(weeks) / block_len))
    rng = np.random.default_rng(seed)
    pick = np.array(starts)[rng.integers(0, len(starts), size=(reps, nb))]
    idx = np.stack([pick + j for j in range(block_len)], axis=2).reshape(reps, nb * block_len)[:, :len(weeks)]
    return S[idx].sum(axis=1) / N[idx].sum(axis=1)


def bootstrap_report(delta, block_ids, ref_mean, reps=BOOT_REPS, seed=BOOT_SEED):
    boot = block_bootstrap(delta, block_ids, reps, seed)
    obs = float(np.mean(delta))
    return {"observed_mean_delta": obs, "relative_improvement": float(-obs / ref_mean), "one_sided_95_upper_bound": float(np.percentile(boot, 95)), "ci_2_5": float(np.percentile(boot, 2.5)), "ci_97_5": float(np.percentile(boot, 97.5)),
            "replicates": reps, "seed": seed, "n_units": int(len(delta)), "n_weeks": int(len(np.unique(block_ids)))}


def slice_gate(ch, ref, masks, min_rows, tol):
    detail, ok = {}, True
    for name, m in masks.items():
        n = int(m.sum())
        if n < min_rows:
            detail[name] = {"rows": n, "eligible": False}; continue
        c, r = float(ch[m].mean()), float(ref[m].mean())
        rel = (c - r) / r
        detail[name] = {"rows": n, "eligible": True, "challenger": c, "reference": r, "relative_worse": rel, "pass": rel <= tol}
        ok = ok and rel <= tol
    return ok, detail
