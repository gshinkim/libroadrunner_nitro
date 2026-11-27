"""Triage time-course disagreements: rerun both engines at rtol 1e-11 and compare with the normal (rtol 1e-8) runs.

    python tc_triage.py <repo root>            (driver; workers are this script with --worker)
"""
import concurrent.futures
import json
import subprocess
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else None


def worker(engine, path):
    mod = __import__('roadrunner_nitro' if engine == 'nitro' else 'roadrunner')
    r = mod.RoadRunner(path)
    r.integrator.relative_tolerance = 1e-11
    r.integrator.absolute_tolerance = 1e-16
    sel = ['time'] + ['[%s]' % s for s in r.model.getFloatingSpeciesIds()]
    y = np.asarray(r.simulate(0, 100, 101, sel), float)
    print(json.dumps({'cols': sel, 'data': y.tolist()}))


def tight(engine, path):
    try:
        p = subprocess.run([sys.executable, __file__, '--worker', engine, path], capture_output=True, text=True, timeout=300)
        line = [l for l in p.stdout.splitlines() if l.startswith('{')]
        return json.loads(line[-1]) if line else 'error: ' + (p.stderr.strip().splitlines() or ['?'])[-1][:100]
    except subprocess.TimeoutExpired:
        return 'timeout'


def dist(a, b):
    """max over points of |a - b| / (|b| + 1e-8 * column scale)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = np.maximum(np.nanmax(np.abs(b), axis=0), 1.0)
    return float(np.nanmax(np.abs(a - b) / (np.abs(b) + 1e-8 * scale)))


def aligned(x, cols):
    return np.asarray(x['data'], float)[:, [x['cols'].index(c) for c in cols]]


if __name__ == '__main__':
    if sys.argv[1] == '--worker':
        worker(sys.argv[2], sys.argv[3])
        sys.exit()
    v = json.load(open(ROOT + '/benchmarks/biomodels_compare.json'))['verdicts']
    raw = {e: json.load(open(ROOT + '/benchmarks/biomodels_raw_%s.json' % e)) for e in ('nitro', 'libroadrunner')}
    models = [m for m, x in v.items() if x.get('tc', '').startswith('disagree')]
    jobs = [(m, e) for m in models for e in ('nitro', 'libroadrunner')]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        res = dict(zip(jobs, pool.map(lambda j: tight(j[1], ROOT + '/tests/models/biomodels/%s.xml' % j[0]), jobs)))
    counts, lines = {}, []
    for m in models:
        tn, tl = res[(m, 'nitro')], res[(m, 'libroadrunner')]
        if not (isinstance(tn, dict) and isinstance(tl, dict)):
            k = 'tight run failed'
            lines.append('%s %s: nitro %s, libroadrunner %s' % (m, k, tn if isinstance(tn, str) else 'ok', tl if isinstance(tl, str) else 'ok'))
        else:
            cols = tn['cols']
            if sorted(cols) != sorted(tl['cols']):
                k = 'columns differ'
            else:
                tight_gap = dist(aligned(tn, cols), aligned(tl, cols))
                ref = (aligned(tn, cols) + aligned(tl, cols)) / 2
                en = dist(aligned(raw['nitro'][m + '.xml']['tc'], cols), ref)
                el = dist(aligned(raw['libroadrunner'][m + '.xml']['tc'], cols), ref)
                if tight_gap < 1e-6:
                    k = 'tolerance: tight runs agree; nitro closer' if en < el else 'tolerance: tight runs agree; libroadrunner closer'
                else:
                    k = 'still differ at rtol 1e-11'
                lines.append('%s %-48s tight gap %.1e  normal-run error: nitro %.1e, libroadrunner %.1e' % (m, k, tight_gap, en, el))
        counts[k] = counts.get(k, 0) + 1
    print('time-course disagreements (%d):' % len(models))
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print('  %4d  %s' % (n, k))
    print()
    for line in lines:
        if 'tolerance' not in line:
            print('  ' + line)
