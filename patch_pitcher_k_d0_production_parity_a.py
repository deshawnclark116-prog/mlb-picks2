#!/usr/bin/env python3
"""
PITCHER_K_D0_PRODUCTION_PARITY_PATCH_A

Surgical source patch for the two true blockers from corrected parity audit B:

    D0-02  Preserve general K-data availability for D0 activation.
    TIME-01 Enforce strict D-1 / same-day exclusion in the live pitcher-K path.

Derived blocker:
    D0-05 should clear automatically once D0-02 is fixed.

This patch does NOT:
- change validated weights
- change workload, volatility, TTO, simulations, or thresholds
- change hitter model logic
- change the batter-hits prospective holdout
- modify ksim.py
- authorize production promotion

Dry run:
    python -u patch_pitcher_k_d0_production_parity_a.py

Apply:
    python -u patch_pitcher_k_d0_production_parity_a.py --apply 2>&1 | tee /data/hr_model/patch_pitcher_k_d0_production_parity_a.log

Then rerun:
    python -u pitcher_k_d0_implementation_parity_b.py 2>&1 | tee /data/hr_model/pitcher_k_d0_implementation_parity_b_postpatch.log
"""

import argparse
import ast
import datetime as dt
import hashlib
import json
import py_compile
import shutil
from pathlib import Path

HR_DIR = Path('/data/hr_model')
FORMAL_RESULT = HR_DIR / 'pitcher_k_d0_2026_formal_gate_a_results.json'
PARITY_RESULT = HR_DIR / 'pitcher_k_d0_implementation_parity_b_results.json'
PATCH_RESULT = HR_DIR / 'patch_pitcher_k_d0_production_parity_a_results.json'
PATCH_REPORT = HR_DIR / 'patch_pitcher_k_d0_production_parity_a_report.txt'

PROTECTED_FUNCTIONS = {
    'api.py': [
        'batter_feature_row',
        '_batter_feat_for',
        'build_batter_prop_picks',
        'build_hr_pick',
        'govern_hitter_board',
    ],
    'lineupk.py': [
        'general_batting_avg',
        'head_to_head_avg',
        'batter_hit_sum_score',
        'batter_hr_score',
    ],
    'bvp.py': [
        'classify_batter',
        'power_flag',
    ],
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def discover(name):
    for path in (
        Path.cwd() / name,
        Path('/opt/render/project/src') / name,
        Path('/opt/render/project/src/src') / name,
    ):
        if path.exists():
            return path.resolve()
    raise RuntimeError(f'Could not find {name}')


def exact_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f'{label}: expected 1 exact match, found {n}')
    return text.replace(old, new, 1)


def replace_function(text, func_name, new_source):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    target = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            target = node
            break
    if target is None:
        raise RuntimeError(f'Function not found: {func_name}')
    start = target.lineno - 1
    end = target.end_lineno
    replacement = new_source.rstrip() + '\n\n'
    return ''.join(lines[:start]) + replacement + ''.join(lines[end:])


def function_hashes(path, names):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    tree = ast.parse(text)
    lines = text.splitlines()
    funcs = {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    out = {}
    for name in names:
        node = funcs.get(name)
        if node is None:
            raise RuntimeError(f'Protected function missing: {path}::{name}')
        src = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        out[name] = hashlib.sha256(src.encode('utf-8')).hexdigest()
    return out


def verify_formal():
    if not FORMAL_RESULT.exists():
        raise RuntimeError(f'Missing formal result: {FORMAL_RESULT}')
    payload = load_json(FORMAL_RESULT)
    verdict = str(payload.get('final_verdict') or '')
    gate = payload.get('formal_gate') or {}
    if (
        'D0_FIXED_LINEUP_ACTIVATION_PASSES_2026_FORMAL_GATE' not in verdict
        or gate.get('overall_formal_gate_pass') is not True
    ):
        raise RuntimeError('Formal D0 2026 pass could not be verified.')
    return {
        'formal_verdict': verdict,
        'overall_formal_gate_pass': True,
        'formal_result_sha256': sha256_file(FORMAL_RESULT),
        'production_promotion_authorized': False,
    }


def verify_parity(source_paths):
    if not PARITY_RESULT.exists():
        raise RuntimeError(f'Missing parity-B result: {PARITY_RESULT}')
    payload = load_json(PARITY_RESULT)
    summary = payload.get('summary') or {}
    blockers = {r.get('check_id') for r in summary.get('blockers', [])}
    derived = {r.get('check_id') for r in summary.get('derived_blockers', [])}
    if blockers != {'D0-02', 'TIME-01'}:
        raise RuntimeError(f'Unexpected blocker set: {sorted(blockers)}')
    if derived != {'D0-05'}:
        raise RuntimeError(f'Unexpected derived-blocker set: {sorted(derived)}')

    audited = payload.get('preflight', {}).get('source_hashes', {})
    current = {name: sha256_file(path) for name, path in source_paths.items()}
    mismatches = {
        name: {'audited': audited.get(name), 'current': current[name]}
        for name in current
        if audited.get(name) != current[name]
    }
    if mismatches:
        raise RuntimeError(
            'Source changed after parity audit B; refusing stale patch:\n'
            + json.dumps(mismatches, indent=2)
        )
    return {
        'blockers': sorted(blockers),
        'derived_blockers': sorted(derived),
        'source_hashes_verified': True,
        'parity_result_sha256': sha256_file(PARITY_RESULT),
    }


LINEUPK_GENERAL = '''
def general_k_rate_vs_hand(batter_id, throws, season, as_of_date=None):
    """
    General batter K-rate with optional strict D-1 date bounding.

    D0 usable-data sources:
      - handedness split with >= 15 prior PA
      - current-season overall K rate with >= 15 prior PA

    When as_of_date is supplied, requests are bounded through the prior
    calendar date. No unbounded same-day fallback is used.
    """
    code = "vl" if throws == "L" else "vr"
    prior_date = _prior_date_str(as_of_date)

    hand_params = {
        "stats": "statSplits",
        "sitCodes": code,
        "group": "hitting",
        "season": season,
    }
    if prior_date:
        hand_params["endDate"] = prior_date

    d = _get(f"{MLB}/people/{batter_id}/stats", **hand_params)
    try:
        st = d["stats"][0]["splits"][0]["stat"]
        pa = int(st.get("plateAppearances", 0) or 0)
        so = int(st.get("strikeOuts", 0) or 0)
        if pa >= 15:
            return so / pa, True
    except Exception:
        pass

    season_params = {
        "stats": "season",
        "group": "hitting",
        "season": season,
    }
    if prior_date:
        season_params["endDate"] = prior_date

    d2 = _get(f"{MLB}/people/{batter_id}/stats", **season_params)
    try:
        st = d2["stats"][0]["splits"][0]["stat"]
        pa = int(st.get("plateAppearances", 0) or 0)
        so = int(st.get("strikeOuts", 0) or 0)
        if pa >= 15:
            return so / pa, True
    except Exception:
        pass

    return LEAGUE_AVG_K, False
'''

LINEUPK_H2H = '''
def head_to_head_k_rate(batter_id, pitcher_id, as_of_date=None):
    params = {
        "stats": "vsPlayer",
        "opposingPlayerId": pitcher_id,
        "group": "hitting",
    }

    prior_date = _prior_date_str(as_of_date)
    if prior_date:
        params["endDate"] = prior_date

    d = _get(f"{MLB}/people/{batter_id}/stats", **params)

    pa_tot = 0
    k_tot = 0

    try:
        for s in d["stats"][0]["splits"]:
            if s.get("stat"):
                st = s["stat"]
                pa_tot += int(st.get("plateAppearances", 0) or 0)
                k_tot += int(st.get("strikeOuts", 0) or 0)
    except Exception:
        pass

    if pa_tot >= MIN_H2H_PA:
        return k_tot / pa_tot, pa_tot

    return None, 0
'''

LINEUPK_BLENDED = '''
def blended_batter_k_rate(
    batter_id,
    pitcher_id,
    throws,
    season,
    as_of_date=None,
):
    """
    D0_FIXED_LINEUP_ACTIVATION parity.

    Availability is True when ANY validated K-data source is usable:
      - handedness split >= 15 prior PA
      - season K rate >= 15 prior PA
      - H2H K history >= 2 prior PA

    The validated 40% H2H / 60% general rate blend is unchanged.
    """
    gen_kr, gen_used = general_k_rate_vs_hand(
        batter_id,
        throws,
        season,
        as_of_date=as_of_date,
    )

    h2h_kr, h2h_pa = head_to_head_k_rate(
        batter_id,
        pitcher_id,
        as_of_date=as_of_date,
    )

    if h2h_kr is not None:
        return H2H_WEIGHT * h2h_kr + GEN_WEIGHT * gen_kr, True

    return gen_kr, gen_used
'''

LINEUPK_EXPECTATION = '''
def lineup_k_expectation(
    opp_batter_ids,
    pitcher_throws,
    season,
    expected_bf,
    pitcher_id=None,
    as_of_date=None,
):
    if not opp_batter_ids:
        return None, None, 0

    rates = []
    n_data = 0

    for bid in opp_batter_ids[:9]:
        if pitcher_id is not None:
            kr, used = blended_batter_k_rate(
                bid,
                pitcher_id,
                pitcher_throws,
                season,
                as_of_date=as_of_date,
            )
        else:
            kr, used = general_k_rate_vs_hand(
                bid,
                pitcher_throws,
                season,
                as_of_date=as_of_date,
            )

        rates.append(kr)

        if used:
            n_data += 1

        time.sleep(0.06)

    if not rates:
        return None, None, 0

    avg_k = sum(rates) / len(rates)
    return avg_k * expected_bf, avg_k, n_data
'''

BVP_BATTER_VS = '''
def batter_vs_pitcher(
    batter_id,
    pitcher_id,
    as_of_date=None,
):
    """
    Full BvP line with optional strict D-1 upper date bound.

    Existing hitter callers that omit as_of_date stay on the legacy path.
    The pitcher-K lineup nudge passes as_of_date explicitly.
    """
    params = {
        "stats": "vsPlayer",
        "group": "hitting",
        "opposingPlayerId": pitcher_id,
        "sportId": 1,
    }

    prior_date = _prior_date_str(as_of_date)
    if prior_date:
        params["endDate"] = prior_date

    d = _get(f"{MLB}/people/{batter_id}/stats", **params)

    try:
        for s in d.get("stats", []):
            for sp in s.get("splits", []):
                st = sp.get("stat", {})
                ab = int(st.get("atBats", 0) or 0)
                h = int(st.get("hits", 0) or 0)
                dbl = int(st.get("doubles", 0) or 0)
                trp = int(st.get("triples", 0) or 0)
                hr = int(st.get("homeRuns", 0) or 0)
                rbi = int(st.get("rbi", 0) or 0)
                tb = int(st.get("totalBases", 0) or 0)
                sb = int(st.get("stolenBases", 0) or 0)
                so = int(st.get("strikeOuts", 0) or 0)
                bb = int(st.get("baseOnBalls", 0) or 0)
                pa = int(st.get("plateAppearances", 0) or 0)

                if pa > 0:
                    return {
                        "ab": ab,
                        "h": h,
                        "doubles": dbl,
                        "triples": trp,
                        "hr": hr,
                        "rbi": rbi,
                        "tb": tb,
                        "sb": sb,
                        "so": so,
                        "bb": bb,
                        "pa": pa,
                        "avg": (h / ab) if ab else 0.0,
                        "k_rate": (so / pa) if pa else 0.0,
                        "tb_per_ab": (tb / ab) if ab else 0.0,
                        "iso": ((tb - h) / ab) if ab else 0.0,
                    }
    except Exception:
        pass

    return None
'''

BVP_LINEUP = '''
def lineup_vs_pitcher(
    batter_ids,
    pitcher_id,
    as_of_date=None,
):
    """Aggregate lineup BvP for the pitcher-K nudge."""
    tot_ab = tot_h = tot_so = tot_pa = tot_tb = 0
    per_batter = {}

    for bid in batter_ids:
        batter_bvp = batter_vs_pitcher(
            bid,
            pitcher_id,
            as_of_date=as_of_date,
        )

        per_batter[bid] = batter_bvp

        if batter_bvp:
            tot_ab += batter_bvp["ab"]
            tot_h += batter_bvp["h"]
            tot_so += batter_bvp["so"]
            tot_pa += batter_bvp["pa"]
            tot_tb += batter_bvp["tb"]

        time.sleep(0.08)

    if tot_pa == 0:
        return {
            "lineup_avg": None,
            "lineup_k_rate": None,
            "lineup_slg": None,
            "k_nudge": 1.0,
            "per_batter": per_batter,
            "sample_pa": 0,
        }

    lineup_avg = tot_h / tot_ab if tot_ab else 0
    lineup_k_rate = tot_so / tot_pa
    lineup_slg = tot_tb / tot_ab if tot_ab else 0

    # Validated nudge semantics unchanged.
    raw = lineup_k_rate / 0.22
    k_nudge = max(0.85, min(1.15, raw))

    return {
        "lineup_avg": round(lineup_avg, 3),
        "lineup_k_rate": round(lineup_k_rate, 3),
        "lineup_slg": round(lineup_slg, 3),
        "k_nudge": round(k_nudge, 3),
        "per_batter": per_batter,
        "sample_pa": tot_pa,
    }
'''

API_PITCHER = '''
def pitcher_feature_row(pid, as_of_date=None):
    """
    Pregame pitcher profile with optional strict D-1 exclusion.

    When as_of_date is supplied, only rows with:
        game_date < as_of_date
    are eligible.

    Validated pitcher-profile semantics remain unchanged.
    """
    g = get(
        f"{MLB}/people/{pid}/stats",
        stats="gameLog",
        group="pitching",
        season=SEASON,
    )

    try:
        splits = g["stats"][0]["splits"]
    except Exception:
        return None

    sos = []
    bfs = []
    per_start_krate = []
    cum_bf = cum_so = cum_outs = cum_bb = 0
    n_starts = 0

    for sp in splits:
        if as_of_date:
            game_date = str(sp.get("date") or "")
            if not game_date or not (game_date < str(as_of_date)[:10]):
                continue

        st = sp["stat"]
        bf = int(st.get("battersFaced", 0) or 0)
        so = int(st.get("strikeOuts", 0) or 0)
        outs = int(st.get("outs", 0) or 0) or ip_to_outs(
            st.get("inningsPitched", "0.0")
        )

        if bf >= 12:
            sos.append(so)
            bfs.append(bf)
            per_start_krate.append(so / bf if bf else 0)
            cum_bf += bf
            cum_so += so
            cum_outs += outs
            cum_bb += int(st.get("baseOnBalls", 0) or 0)
            n_starts += 1

    if n_starts < 3:
        return None

    season_kbf = cum_so / cum_bf if cum_bf else 0
    n = len(sos)
    w = [math.exp(-RECENCY_DECAY * (n - 1 - i)) for i in range(n)]

    rec_kbf = (
        sum(wi * s for wi, s in zip(w, sos))
        / sum(wi * b for wi, b in zip(w, bfs))
    ) if sum(w) else season_kbf

    k_per_bf = (
        (1 - SEASON_ANCHOR) * rec_kbf
        + SEASON_ANCHOR * season_kbf
    )

    return {
        "k_per_bf": k_per_bf,
        "season_k_per_bf": season_kbf,
        "avg_bf": sum(bfs[-5:]) / len(bfs[-5:]),
        "recent_k_avg": sum(sos[-5:]) / len(sos[-5:]),
        "bb_rate": cum_bb / cum_bf if cum_bf else 0,
        "outs_per_start": cum_outs / n_starts if n_starts else 0,
        "starts": n_starts,
        "per_start_krate": per_start_krate[-12:],
    }
'''


def patch_lineupk(text):
    text = exact_once(
        text,
        'import time\nimport requests\n',
        'import time\nimport datetime as dt\nimport requests\n',
        'lineupk import',
    )
    text = exact_once(
        text,
        'S.headers["User-Agent"] = "prop-edge-lineupk/4.1"\n\n\n',
        '''S.headers["User-Agent"] = "prop-edge-lineupk/4.1"


def _prior_date_str(as_of_date):
    if not as_of_date:
        return None

    try:
        game_date = dt.date.fromisoformat(str(as_of_date)[:10])
    except Exception:
        return None

    return (game_date - dt.timedelta(days=1)).isoformat()


''',
        'lineupk D-1 helper',
    )
    text = replace_function(text, 'general_k_rate_vs_hand', LINEUPK_GENERAL)
    text = replace_function(text, 'head_to_head_k_rate', LINEUPK_H2H)
    text = replace_function(text, 'blended_batter_k_rate', LINEUPK_BLENDED)
    text = replace_function(text, 'lineup_k_expectation', LINEUPK_EXPECTATION)
    return text


def patch_bvp(text):
    text = exact_once(
        text,
        'import time\nimport requests\n',
        'import time\nimport datetime as dt\nimport requests\n',
        'bvp import',
    )
    text = exact_once(
        text,
        'S.headers["User-Agent"] = "prop-edge-bvp/1.1"\n\n\n',
        '''S.headers["User-Agent"] = "prop-edge-bvp/1.1"


def _prior_date_str(as_of_date):
    if not as_of_date:
        return None

    try:
        game_date = dt.date.fromisoformat(str(as_of_date)[:10])
    except Exception:
        return None

    return (game_date - dt.timedelta(days=1)).isoformat()


''',
        'bvp D-1 helper',
    )
    text = replace_function(text, 'batter_vs_pitcher', BVP_BATTER_VS)
    text = replace_function(text, 'lineup_vs_pitcher', BVP_LINEUP)
    return text


def patch_api(text):
    text = replace_function(text, 'pitcher_feature_row', API_PITCHER)

    text = exact_once(
        text,
        '''    for game in pregame_games:
        gid = game["game_id"]
        lineup = get_confirmed_lineup(gid)

        for side in ("home_pitcher", "away_pitcher"):
''',
        '''    for game in pregame_games:
        gid = game["game_id"]
        lineup = get_confirmed_lineup(gid)

        # Strict D-1 boundary for the pitcher-K path.
        k_as_of_date = game.get("date") or today

        for side in ("home_pitcher", "away_pitcher"):
''',
        'api K-loop boundary',
    )

    text = exact_once(
        text,
        '            feat = pitcher_feature_row(pid)\n',
        '''            feat = pitcher_feature_row(
                pid,
                as_of_date=k_as_of_date,
            )
''',
        'api pitcher_feature_row call',
    )

    text = exact_once(
        text,
        '''                ek, avg_kr, n = lineupk.lineup_k_expectation(
                    opp_batters, throws, SEASON, feat["avg_bf"], pitcher_id=pid)
''',
        '''                ek, avg_kr, n = lineupk.lineup_k_expectation(
                    opp_batters,
                    throws,
                    SEASON,
                    feat["avg_bf"],
                    pitcher_id=pid,
                    as_of_date=k_as_of_date,
                )
''',
        'api lineup_k_expectation call',
    )

    text = exact_once(
        text,
        '                agg = bvp.lineup_vs_pitcher(opp_batters, pid)\n',
        '''                agg = bvp.lineup_vs_pitcher(
                    opp_batters,
                    pid,
                    as_of_date=k_as_of_date,
                )
''',
        'api bvp K-nudge call',
    )
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='Write changes; default is dry run.')
    args = parser.parse_args()

    print('PITCHER_K_D0_PRODUCTION_PARITY_PATCH_A', flush=True)
    print('======================================', flush=True)

    paths = {
        'api.py': discover('api.py'),
        'lineupk.py': discover('lineupk.py'),
        'bvp.py': discover('bvp.py'),
        'ksim.py': discover('ksim.py'),
    }

    formal = verify_formal()
    parity = verify_parity(paths)

    before_hashes = {name: sha256_file(path) for name, path in paths.items()}
    protected_before = {
        name: function_hashes(paths[name], funcs)
        for name, funcs in PROTECTED_FUNCTIONS.items()
    }

    originals = {
        name: path.read_text(encoding='utf-8', errors='replace')
        for name, path in paths.items()
    }

    patched = {
        'api.py': patch_api(originals['api.py']),
        'lineupk.py': patch_lineupk(originals['lineupk.py']),
        'bvp.py': patch_bvp(originals['bvp.py']),
        'ksim.py': originals['ksim.py'],
    }

    temp_paths = []
    try:
        for name in ('api.py', 'lineupk.py', 'bvp.py'):
            temp = paths[name].parent / f'.{name}.d0_patch_tmp.py'
            temp.write_text(patched[name], encoding='utf-8')
            temp_paths.append(temp)
            py_compile.compile(str(temp), doraise=True)
    finally:
        for temp in temp_paths:
            try:
                temp.unlink()
            except Exception:
                pass

    print('\nPATCH PREFLIGHT', flush=True)
    print('---------------', flush=True)
    print(json.dumps({
        'mode': 'APPLY' if args.apply else 'DRY_RUN',
        'formal': formal,
        'parity': parity,
        'true_blockers': ['D0-02', 'TIME-01'],
        'derived_blocker': ['D0-05'],
        'production_promotion_authorized': False,
        'hits_prospective_holdout_touched': False,
        'ksim_modified': False,
    }, indent=2), flush=True)

    print('\nPATCH PLAN', flush=True)
    print('----------', flush=True)
    print('D0-02: preserve gen_used and count valid general/H2H K data.', flush=True)
    print('TIME-01: strict D-1 pitcher game-log filter + prior-date bounds on K batter/H2H/BVP requests.', flush=True)
    print('UNCHANGED: 55/45, 40/60, BF windows, volatility, TTO, 10k sims, thresholds, hitter logic, ksim.py.', flush=True)

    if not args.apply:
        print('\nDRY RUN COMPLETE — NO FILES CHANGED', flush=True)
        print('Run again with --apply to write the patch.', flush=True)
        return 0

    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backups = {}

    for name in ('api.py', 'lineupk.py', 'bvp.py'):
        backup = paths[name].with_name(f'{paths[name].name}.pre_d0_parity_patch_{stamp}.bak')
        shutil.copy2(paths[name], backup)
        backups[name] = str(backup)

    try:
        for name in ('api.py', 'lineupk.py', 'bvp.py'):
            paths[name].write_text(patched[name], encoding='utf-8')

        for name in ('api.py', 'lineupk.py', 'bvp.py'):
            py_compile.compile(str(paths[name]), doraise=True)

        protected_after = {
            name: function_hashes(paths[name], funcs)
            for name, funcs in PROTECTED_FUNCTIONS.items()
        }

        drift = {}
        for file_name, before_map in protected_before.items():
            for func_name, before_hash in before_map.items():
                after_hash = protected_after[file_name][func_name]
                if before_hash != after_hash:
                    drift[f'{file_name}::{func_name}'] = {
                        'before': before_hash,
                        'after': after_hash,
                    }

        if drift:
            raise RuntimeError('Protected hitter-function drift detected:\n' + json.dumps(drift, indent=2))

        if sha256_file(paths['ksim.py']) != before_hashes['ksim.py']:
            raise RuntimeError('ksim.py changed unexpectedly.')

    except Exception:
        for name, backup in backups.items():
            shutil.copy2(backup, paths[name])
        raise

    after_hashes = {name: sha256_file(path) for name, path in paths.items()}

    result = {
        'script': 'PITCHER_K_D0_PRODUCTION_PARITY_PATCH_A',
        'status': 'PATCH_APPLIED_COMPILE_CHECKED',
        'applied_at_utc': dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        'formal_validation_verified': True,
        'parity_b_blocker_set_verified': True,
        'patched_true_blockers': ['D0-02', 'TIME-01'],
        'expected_derived_clear': ['D0-05'],
        'files_changed': ['api.py', 'lineupk.py', 'bvp.py'],
        'files_not_changed': ['ksim.py'],
        'protected_hitter_functions_unchanged': True,
        'hits_prospective_holdout_touched': False,
        'production_promotion_authorized': False,
        'backups': backups,
        'before_file_hashes': before_hashes,
        'after_file_hashes': after_hashes,
        'next_step': 'RERUN_PITCHER_K_D0_IMPLEMENTATION_PARITY_AUDIT_B',
    }

    PATCH_RESULT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    PATCH_REPORT.write_text(
        '\n'.join([
            'PITCHER_K_D0_PRODUCTION_PARITY_PATCH_A',
            '=' * 38,
            '',
            f"STATUS: {result['status']}",
            'PATCHED: D0-02, TIME-01',
            'EXPECTED DERIVED CLEAR: D0-05',
            'PROTECTED HITTER FUNCTIONS UNCHANGED: True',
            'KSIM MODIFIED: False',
            'PRODUCTION PROMOTION AUTHORIZED: False',
            '',
            'NEXT:',
            result['next_step'],
        ]),
        encoding='utf-8',
    )

    print('\nPATCH APPLIED', flush=True)
    print('-------------', flush=True)
    print(json.dumps(result, indent=2), flush=True)

    print('\nNEXT COMMAND', flush=True)
    print('------------', flush=True)
    print(
        'python -u pitcher_k_d0_implementation_parity_b.py '
        '2>&1 | tee '
        '/data/hr_model/pitcher_k_d0_implementation_parity_b_postpatch.log',
        flush=True,
    )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
