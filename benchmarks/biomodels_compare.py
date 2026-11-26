"""Curated BioModels through roadrunner_nitro and libroadrunner, compared check by check: load, time course and
the full Jacobian at the initial state. No steady state is computed: many curated models have none (oscillators,
events, unbounded growth), so steady-state checks would measure solver convergence rather than correctness.

    python benchmarks/biomodels_download.py     # once: models into tests/models/biomodels
    python benchmarks/biomodels_compare.py      # all models; results in benchmarks/biomodels_compare.json
    python benchmarks/biomodels_compare.py --models BIOMD0000000001 BIOMD0000000012

Load, time course and Jacobian are timed separately for every model (after an untimed warm-up in each worker
process, so one-off start-up costs are not charged to a model); timeouts and crashes are counted, not timed.

Each engine runs in worker processes, a batch of models each; a model that crashes or hangs a worker is retried
alone and reported as 'crash'. Matrices are matched by row and column names, so an ordering difference alone is
not a disagreement. Each check gets a verdict per model: agree, disagree, nitro-only (libroadrunner failed),
lr-only (roadrunner_nitro failed) or both-fail. Disagreements need triage: which engine is right is not decided
here.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, '..', 'tests', 'models', 'biomodels')
RESULTS = os.path.join(HERE, 'biomodels_compare.json')
RAW = os.path.join(HERE, 'biomodels_raw_%s.json')
WARMUP = os.path.join(HERE, '..', 'tests', 'models', 'sosbench', 'repressilator', 'repressilator-sbml-l2v1.xml')  # each engine's raw output, saved before comparing
BATCH = 20
JOBS = 8  # worker processes at once (libroadrunner uses one core per process)
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
    out, times = {}, {}
    try:
        t0 = time.perf_counter()
        r = rr.RoadRunner(path)
        times['load'] = time.perf_counter() - t0
    except Exception as e:
        return {'load': 'error: %s' % str(e)[:200]}
    out['load'], out['time'] = 'ok', times
    try:
        r.integrator.relative_tolerance = RTOL
        r.integrator.absolute_tolerance = ATOL
        sel = ['time'] + ['[%s]' % s for s in r.model.getFloatingSpeciesIds()]
        t0 = time.perf_counter()
        y = r.simulate(0, T_END, POINTS, sel)
        times['tc'] = time.perf_counter() - t0
        y = np.asarray(y, dtype=float)
        out['tc'] = dict(cols=sel, data=y.tolist())
    except Exception as e:
        out['tc'] = 'error: %s' % str(e)[:200]
    try:
        r.resetAll()  # the Jacobian at the initial state: exact rate-law derivatives, no steady state needed
        t0 = time.perf_counter()
        m = r.getFullJacobian()
        times['jac'] = time.perf_counter() - t0
        out['jac'] = dict(rows=list(m.rownames), cols=list(m.colnames), data=np.asarray(m, dtype=float).tolist())
    except Exception as e:
        out['jac'] = 'error: %s' % str(e)[:200]
    return out


def worker(engine, paths):
    import warnings
    warnings.filterwarnings('ignore')
    if os.path.exists(WARMUP):  # one-off start-up costs (LLVM / Cranelift initialization) outside the timings
        try:
            run_model(engine, WARMUP)
        except Exception:
            pass
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


def shape_ok(m):
    """Whether a matrix's data matches its row and column names."""
    return np.asarray(m['data'], dtype=float).size == len(m['rows']) * len(m['cols'])


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
            bad = [e for e, m in (('nitro', a), ('libroadrunner', b)) if not shape_ok(m)]
            if bad:
                v[key] = 'disagree: %s returned a matrix whose data does not match its row and column names' % bad[0]
                continue
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
    ap.add_argument('--reuse', action='store_true', help='compare the saved raw results instead of simulating')
    ap.add_argument('--jobs', type=int, default=JOBS, help='worker processes at once')
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker[0], a.worker[1:])
    names = a.models or sorted(f[:-4] for f in os.listdir(MODELS) if f.endswith('.xml'))
    paths = [os.path.join(MODELS, m + '.xml') for m in names]
    raw = {}
    for engine in ('nitro', 'libroadrunner'):
        if a.reuse and os.path.exists(RAW % engine):
            with open(RAW % engine, encoding='utf-8') as f:
                raw[engine] = json.load(f)
            print('  %s: reusing %s' % (engine, RAW % engine))
            continue
        raw[engine] = {}
        batches = [paths[k:k + BATCH] for k in range(0, len(paths), BATCH)]
        with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
            for got in pool.map(lambda b: run_batch(engine, b, a.timeout), batches):
                raw[engine].update(got)
                print('  %s: %d / %d' % (engine, len(raw[engine]), len(paths)), flush=True)
        with open(RAW % engine, 'w', encoding='utf-8') as f:  # saved before any comparison
            json.dump(raw[engine], f)
    verdicts = {}
    for m in names:
        try:
            verdicts[m] = compare(raw['nitro'][m + '.xml'], raw['libroadrunner'][m + '.xml'])
        except Exception as e:  # a comparison bug costs one model's verdict, never the run
            verdicts[m] = {'load': 'compare-error: %s: %s' % (type(e).__name__, str(e)[:150])}
    with open(RESULTS, 'w', encoding='utf-8') as f:
        json.dump({'verdicts': verdicts,
                   'times': {m: {e: raw[e][m + '.xml'].get('time') for e in raw} for m in names}, 'errors': {m: {e: {k: v for k, v in raw[e][m + '.xml'].items() if isinstance(v, str)}
                                                          for e in raw} for m in names}}, f, indent=1)
    print('\n%-6s %8s %9s %11s %8s %10s' % ('check', 'agree', 'disagree', 'nitro-only', 'lr-only', 'both-fail'))
    for key in ('load', 'tc', 'jac'):
        c = Counter((v.get(key) or 'n/a').split(':')[0] for v in verdicts.values())
        print('%-6s %8d %9d %11d %8d %10d' % (key, c['agree'], c['disagree'], c['nitro-only'], c['lr-only'], c['both-fail']))
    print('\ntiming per model, milliseconds (successful steps only; %d worker processes at once on both engines):' % a.jobs)
    print('%-6s %-14s %6s %10s %10s %12s' % ('step', 'engine', 'n', 'median', 'p90', 'total (s)'))
    for step in ('load', 'tc', 'jac'):
        for e in ('nitro', 'libroadrunner'):
            t = [raw[e][m + '.xml'].get('time', {}).get(step) for m in names]
            t = np.array([x for x in t if x is not None])
            if len(t):
                print('%-6s %-14s %6d %10.3f %10.3f %12.1f' % (step, e, len(t), 1e3 * np.median(t), 1e3 * np.percentile(t, 90), t.sum()))
    print('\nper-model ratio libroadrunner / nitro (models both engines completed):')
    for step in ('load', 'tc', 'jac'):
        q = [raw['libroadrunner'][m + '.xml'].get('time', {}).get(step) / raw['nitro'][m + '.xml']['time'][step]
             for m in names if raw['nitro'][m + '.xml'].get('time', {}).get(step)
             and raw['libroadrunner'][m + '.xml'].get('time', {}).get(step) is not None]
        if q:
            print('  %-5s n=%4d  median %6.1fx  quartiles %.1fx - %.1fx  nitro faster in %d of %d' % (
                step, len(q), np.median(q), np.percentile(q, 25), np.percentile(q, 75), sum(x > 1 for x in q), len(q)))
    for e in ('nitro', 'libroadrunner'):
        crashed = [m for m in names if raw[e][m + '.xml'].get('load', '').startswith('crash')]
        print('%s: %d models crashed or timed out%s' % (e, len(crashed), (': ' + ', '.join(crashed[:10])) if crashed else ''))
    print('\ndisagreements (first per model):')
    for m, v in verdicts.items():
        bad = [(k, s) for k, s in v.items() if s.startswith('disagree')]
        if bad:
            print('  %s %s %s' % (m, bad[0][0], bad[0][1][:150]))
    print('\nper-model results: %s' % RESULTS)


if __name__ == '__main__':
    main()
