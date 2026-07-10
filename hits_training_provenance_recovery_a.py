#!/usr/bin/env python3
import json, re, subprocess, sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

ROOT = Path('.').resolve()
OUT_DIR = Path('/data/hr_model')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT = OUT_DIR / 'hits_training_provenance_recovery_a_report.txt'
OUT_JSON = OUT_DIR / 'hits_training_provenance_recovery_a_report.json'

EXCLUDE = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', '.pytest_cache', 'dist', 'build'}
TEXT_SUFFIXES = {'.py', '.json', '.txt', '.md', '.yaml', '.yml', '.toml', '.csv'}
MODEL_SUFFIXES = {'.json', '.ubj', '.bin', '.model', '.pkl', '.pickle', '.joblib'}

HIT_TERMS = ['batter_hits', 'batter hits', 'batter_hit', 'hits_model', 'hit_model']
TRAIN_TERMS = ['xgbregressor', 'xgb.train', 'booster', 'fit(', 'save_model', 'train_test_split', 'training', 'target', 'label', 'actual_hits', 'hits_target', 'game_date', 'season']

def excluded(p):
    return bool(set(p.parts) & EXCLUDE)

def safe_read(p):
    try:
        if p.stat().st_size > 5_000_000:
            return ''
        return p.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return ''

def inspect_api():
    out = {}
    try:
        import api
        name = api.PROP_MODEL['batter_hits']
        out['model_name'] = name
        out['official_threshold'] = float(api.HITTER_MC_HIT_OFFICIAL_MIN_PROB)
        out['lean_threshold'] = float(api.HITTER_MC_HIT_LEAN_MIN_PROB)
        out['mc_sim_n'] = int(api.HITTER_MC_SIM_N)
        out['mc_model_version'] = str(api.HITTER_MC_MODEL_VERSION)
        if name in api._models:
            booster, cols = api._models[name]
            out['loaded'] = True
            out['feature_columns'] = list(cols)
            try:
                out['booster_num_boosted_rounds'] = int(booster.num_boosted_rounds())
            except Exception:
                pass
            try:
                out['booster_config'] = json.loads(booster.save_config())
            except Exception as e:
                out['booster_config_error'] = repr(e)
        else:
            out['loaded'] = False
    except Exception as e:
        out['error'] = repr(e)
    return out

def find_artifacts():
    rows = []
    for p in ROOT.rglob('*'):
        if not p.is_file() or excluded(p) or p.suffix.lower() not in MODEL_SUFFIXES:
            continue
        lname = p.name.lower()
        if ('batter' in lname and 'hit' in lname) or 'batter_hits' in lname:
            try:
                rows.append({'path': str(p.relative_to(ROOT)), 'size_bytes': p.stat().st_size})
            except Exception:
                pass
    return sorted(rows, key=lambda x: x['path'])

def find_training_files():
    results = []
    for p in ROOT.rglob('*'):
        if not p.is_file() or excluded(p) or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = safe_read(p)
        if not text:
            continue
        low = text.lower()
        if not any(t in low for t in HIT_TERMS):
            continue
        lines = text.splitlines()
        score = sum(1 for t in TRAIN_TERMS if t in low)
        matches = []
        for i, line in enumerate(lines, 1):
            ll = line.lower()
            if any(t in ll for t in HIT_TERMS) or any(t in ll for t in TRAIN_TERMS):
                matches.append({'line': i, 'text': line.strip()})
        results.append({'path': str(p.relative_to(ROOT)), 'training_score': score, 'matches': matches[:120]})
    results.sort(key=lambda x: (-x['training_score'], x['path']))
    return results

def extract_refs(files):
    date_rx = re.compile(r'\b(20\d{2}-\d{2}-\d{2}|20\d{2})\b')
    sample_rx = re.compile(r'\b(?:rows|samples|sample_size|n_train|train_rows|training_rows)\b\s*[:=]\s*([0-9][0-9_,]*)', re.I)
    dates, samples, targets, hyperparams, artifact_refs = [], [], [], [], []
    hp_terms = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','reg_alpha','reg_lambda','min_child_weight','gamma','objective','eval_metric','random_state']
    for item in files[:50]:
        p = ROOT / item['path']
        text = safe_read(p)
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            low = line.lower()
            if any(t in low for t in ['batter_hits','target','label','actual_hits','hits_target','y_train']):
                targets.append({'path': item['path'], 'line': i, 'text': line.strip()})
            if any(t in low for t in hp_terms):
                hyperparams.append({'path': item['path'], 'line': i, 'text': line.strip()})
            sm = sample_rx.search(line)
            if sm:
                samples.append({'path': item['path'], 'line': i, 'value': sm.group(1), 'text': line.strip()})
            ds = date_rx.findall(line)
            if ds and any(t in low for t in ['train','season','date','batter','hit']):
                dates.append({'path': item['path'], 'line': i, 'values': ds, 'text': line.strip()})
            if any(sfx in low for sfx in ['.json','.ubj','.model','.pkl','.joblib']):
                artifact_refs.append({'path': item['path'], 'line': i, 'text': line.strip()})
    return {
        'dates': dates[:150],
        'samples': samples[:80],
        'targets': targets[:180],
        'hyperparams': hyperparams[:180],
        'artifact_refs': artifact_refs[:120],
    }

def git_history(paths):
    out = []
    for p in paths[:20]:
        try:
            cp = subprocess.run(
                ['git','log','--follow','--date=iso','--pretty=format:%H|%ad|%an|%s','--',p],
                cwd=str(ROOT), capture_output=True, text=True, timeout=30
            )
            if cp.stdout.strip():
                out.append({'path': p, 'history': cp.stdout.strip().splitlines()[:30]})
        except Exception:
            pass
    return out

def main():
    print('HITS_TRAINING_PROVENANCE_RECOVERY_A')
    print('===================================')
    print('mode: READ ONLY')

    print('\n1) Inspecting loaded production model...')
    api_model = inspect_api()

    print('2) Finding model artifacts...')
    artifacts = find_artifacts()

    print('3) Scanning training/model files...')
    files = find_training_files()

    likely = [x['path'] for x in files if x['training_score'] >= 2 or any(k in x['path'].lower() for k in ['train','model','build','fit','backtest','bt'])]

    print('4) Extracting dates, samples, targets, hyperparameters...')
    refs = extract_refs(files)

    print('5) Reading git history...')
    hist = git_history([x['path'] for x in artifacts] + likely)

    report = {
        'script': 'HITS_TRAINING_PROVENANCE_RECOVERY_A',
        'mode': 'READ_ONLY',
        'loaded_api_model': api_model,
        'model_artifacts': artifacts,
        'likely_training_files': likely[:40],
        'training_files': files,
        'refs': refs,
        'git_history': hist,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding='utf-8')

    lines = []
    lines += ['HITS_TRAINING_PROVENANCE_RECOVERY_A','='*36,'READ-ONLY PROVENANCE REPORT','']
    lines += ['CURRENT PRODUCTION MODEL','------------------------']
    for k in ['model_name','loaded','official_threshold','lean_threshold','mc_sim_n','mc_model_version','booster_num_boosted_rounds']:
        if k in api_model:
            lines.append(f'{k}: {api_model[k]}')
    lines.append('feature_columns:')
    for c in api_model.get('feature_columns', []):
        lines.append(f'  - {c}')

    lines += ['','MODEL ARTIFACTS','---------------']
    for x in artifacts:
        lines.append(f"{x['path']} | size={x['size_bytes']:,} bytes")

    lines += ['','LIKELY TRAINING / MODEL FILES','-----------------------------']
    lines += likely[:40]

    for title, key in [
        ('TRAINING DATE / SEASON REFERENCES','dates'),
        ('TRAINING SAMPLE-SIZE REFERENCES','samples'),
        ('TARGET-CONSTRUCTION REFERENCES','targets'),
        ('HYPERPARAMETER REFERENCES','hyperparams'),
        ('MODEL ARTIFACT REFERENCES IN CODE','artifact_refs'),
    ]:
        lines += ['', title, '-'*len(title)]
        for row in refs[key]:
            lines.append(f"{row['path']}:{row['line']} | {row['text']}")

    lines += ['','GIT HISTORY','-----------']
    for item in hist:
        lines.append(f"FILE: {item['path']}")
        for h in item['history']:
            lines.append('  ' + h)

    OUT_TXT.write_text('\n'.join(lines), encoding='utf-8')

    print('\nRECOVERY SUMMARY')
    print('----------------')
    print('model_name=', api_model.get('model_name'))
    print('model_loaded=', api_model.get('loaded'))
    print('features=', api_model.get('feature_columns'))
    print('model_artifacts_found=', len(artifacts))
    print('likely_training_files=', len(likely))
    print('date_refs=', len(refs['dates']))
    print('sample_refs=', len(refs['samples']))
    print('target_refs=', len(refs['targets']))
    print('hyperparam_refs=', len(refs['hyperparams']))

    print('\nOUTPUTS')
    print('-------')
    print(OUT_TXT)
    print(OUT_JSON)

    print('\nNEXT STEP')
    print('---------')
    print('Paste the RECOVERY SUMMARY first. If needed, print only the targeted report sections next.')

if __name__ == '__main__':
    main()
