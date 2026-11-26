"""Curated BioModels through roadrunner_nitro and libroadrunner, compared check by check: load, time course and
the full Jacobian at the initial state. No steady state is computed: many curated models have none (oscillators,
events, unbounded growth), so steady-state checks would measure solver convergence rather than correctness.

    python benchmarks/biomodels_download.py     # once: models into tests/models/biomodels
    python benchmarks/biomodels_compare.py      # all models; results in benchmarks/biomodels_compare.json
    python benchmarks/biomodels_compare.py --models BIOMD0000000001 BIOMD0000000012

Each engine runs in worker processes, a batch of models each; a model that crashes or hangs a worker is retried
alone and reported as 'crash'. Matrices are matched by row and column names, so an ordering difference alone is
not a disagreement. Each check gets a verdict per model: agree, disagree, nitro-only (libroadrunner failed),
lr-only (roadrunner_nitro failed) or both-fail. Disagreements need triage: which engine is right is not decided
here.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, '..', 'tests', 'models', 'biomodels')
RESULTS = os.path.join(HERE, 'biomodels_compare.json')
BATCH = 20
T_END, POINTS = 100.0, 101
RTOL, ATOL = 1e-8, 1e-12
CMP_RTOL, CMP_ATOL = 1e-5, 1e-8  # relative; absolute as a fraction of the column's largest value


def as_list(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel()]


def run_model(engine, path):
    """One model on one engine: what each check produced (values, or the error)."""
    if engine == 'nitro':
        import roadrunner_nitro as rr
    else:
        import roadrunner as rr
    out = {}
    try:
        r = rr.RoadRunner(path)
    except Exception as e:
        return {'load': 'error: %s' % str(e)[:200]}
    out['load'] = 'ok'
    try:
        r.integrator.relative_tolerance = RTOL
        r.integrator.absolute_tolerance = ATOL
        sel = ['time'] + ['[%s]' % s for s in r.model.getFloatingSpeciesIds()]
        y = np.asarray(r.simulate(0, T_END, POINTS, sel), dtype=float)
        out['tc'] = dict(cols=sel, data=y.tolist())
    except Exception as e:
        out['tc'] = 'error: %s' % str(e)[:200]
    try:
        r.resetAll()  # the Jacobian at the initial state: exact rate-law derivatives, no steady state needed
        m = r.getFullJacobian()
        out['jac'] = dict(rows=list(m.rownames), cols=list(m.colnames), data=np.asarray(m, dtype=float).tolist())
    except Exception as e:
        out['jac'] = 'error: %s' % str(e)[:200]
    return out


def worker(engine, paths):
    import warnings
    warnings.filterwarnings('ignore')
    for p in paths:
        print(json.dumps({'model': os.path.basename(p), 'out': run_model(engine, p)}), flush=True)


def run_batch(engine, paths, timeout):
    cmd = [sys.executable, os.path.abspath(__file__), '--worker', engine, *paths]
    try:
        # a batch gets a short budget; a model that stalls it is then found by rerunning the batch one by one
        budget = timeout if len(paths) == 1 else 60 + 10 * len(paths)
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=budget).stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or '')
    got = {}
    for line in out.splitlines():
        if line.startswith('{"model"'):
            d = json.loads(line)
            got[d['model']] = d['out']
    missing = [p for p in paths if os.path.basename(p) not in got]
    if len(paths) == 1 and missing:
        return {os.path.basename(paths[0]): {'load': 'crash: worker died or timed out'}}
    for p in missing:
        got.update(run_batch(engine, [p], timeout))
    return got


def close(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False, 'shape %s vs %s' % (a.shape, b.shape)
    scale = np.nanmax(np.abs(b), axis=0, initial=0.0) if b.ndim > 1 else np.nanmax(np.abs(b), initial=0.0)
    ok = (np.abs(a - b) <= CMP_RTOL * np.abs(b) + CMP_ATOL * np.maximum(scale, 1.0)) | (np.isnan(a) & np.isnan(b))
    if ok.all():
        return True, ''
    i = np.unravel_index(np.argmax(~ok), ok.shape)
    return False, 'at %s: nitro %r, libroadrunner %r (%d of %d values differ)' % (
        tuple(int(x) for x in i), float(a[i]), float(b[i]), int((~ok).sum()), ok.size)


def by_name(m, rows, cols):
    """Matrix m reordered to rows x cols by name (None if the names differ)."""
    if sorted(m['rows']) != sorted(rows) or sorted(m['cols']) != sorted(cols):
        return None
    d = np.asarray(m['data'], dtype=float).reshape(len(m['rows']), len(m['cols']))
    return d[np.ix_([m['rows'].index(r) for r in rows], [m['cols'].index(c) for c in cols])]


def compare(n, l):
    """Verdicts per check for one model: n = nitro's run_model output, l = libroadrunner's."""
    v = {}
    nl, ll = n.get('load') == 'ok', l.get('load') == 'ok'
    v['load'] = ('agree' if nl else 'both-fail') if nl == ll else ('nitro-only' if nl else 'lr-only')
    for key in ('tc', 'jac'):
        a, b = n.get(key), l.get(key)
        okn, okl = isinstance(a, dict), isinstance(b, dict)
        if a is None and b is None:
            continue
        if not (okn and okl):
            v[key] = ('nitro-only' if okn else 'lr-only') if (okn or okl) else 'both-fail'
            continue
        if key == 'tc':
            if a['cols'] != b['cols']:
                bi = [b['cols'].index(c) for c in a['cols']] if sorted(a['cols']) == sorted(b['cols']) else None
                if bi is None:
                    v[key] = 'disagree: columns %s vs %s' % (a['cols'][:5], b['cols'][:5])
                    continue
                b = dict(cols=a['cols'], data=np.asarray(b['data'])[:, bi].tolist())
            ok, why = close(a['data'], b['data'])
        else:
            bm = by_name(b, a['rows'], a['cols'])
            if bm is None:
                v[key] = 'disagree: row or column names differ'
                continue
            ok, why = close(np.asarray(a['data']).reshape(len(a['rows']), len(a['cols'])), bm)
        v[key] = 'agree' if ok else 'disagree: ' + why
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*')
    ap.add_argument('--timeout', type=float, default=120.0, help='seconds per model')
    ap.add_argument('--worker', nargs='+')
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker[0], a.worker[1:])
    names = a.models or sorted(f[:-4] for f in os.listdir(MODELS) if f.endswith('.xml'))
    paths = [os.path.join(MODELS, m + '.xml') for m in names]
    raw = {}
    for engine in ('nitro', 'libroadrunner'):
        raw[engine] = {}
        for k in range(0, len(paths), BATCH):
            raw[engine].update(run_batch(engine, paths[k:k + BATCH], a.timeout))
            print('  %s: %d / %d' % (engine, len(raw[engine]), len(paths)), end='\r', flush=True)
        print()
    verdicts = {m: compare(raw['nitro'][m + '.xml'], raw['libroadrunner'][m + '.xml']) for m in names}
    with open(RESULTS, 'w', encoding='utf-8') as f:
        json.dump({'verdicts': verdicts, 'errors': {m: {e: {k: v for k, v in raw[e][m + '.xml'].items() if isinstance(v, str)}
                                                          for e in raw} for m in names}}, f, indent=1)
    print('\n%-6s %8s %9s %11s %8s %10s' % ('check', 'agree', 'disagree', 'nitro-only', 'lr-only', 'both-fail'))
    for key in ('load', 'tc', 'jac'):
        c = Counter((v.get(key) or 'n/a').split(':')[0] for v in verdicts.values())
        print('%-6s %8d %9d %11d %8d %10d' % (key, c['agree'], c['disagree'], c['nitro-only'], c['lr-only'], c['both-fail']))
    print('\ndisagreements (first per model):')
    for m, v in verdicts.items():
        bad = [(k, s) for k, s in v.items() if s.startswith('disagree')]
        if bad:
            print('  %s %s %s' % (m, bad[0][0], bad[0][1][:150]))
    print('\nper-model results: %s' % RESULTS)


if __name__ == '__main__':
    main()
