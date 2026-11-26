"""Triage Jacobian disagreements: both engines' initial-state Jacobians against an independent reference,
Richardson-extrapolated central differences of the concentration rates of change (computed by evaluating the
rate laws; the time courses validate that evaluation). Prints which engine matches the reference, per model."""
import json
import sys
import numpy as np
import roadrunner_nitro as rn

ROOT = sys.argv[1]  # repository root
v = json.load(open(ROOT + '/benchmarks/biomodels_compare.json'))['verdicts']
raw = {e: json.load(open(ROOT + '/benchmarks/biomodels_raw_%s.json' % e)) for e in ('nitro', 'libroadrunner')}
models = [m for m, x in v.items() if x.get('jac', '').startswith('disagree: at')]


def mat(m, e, rows, cols):
    j = raw[e][m + '.xml']['jac']
    d = np.asarray(j['data'], float).reshape(len(j['rows']), len(j['cols']))
    return d[np.ix_([j['rows'].index(r) for r in rows], [j['cols'].index(c) for c in cols])]


def reference(path, rows, cols):
    r = rn.RoadRunner(path)
    x0 = {c: r.getValue('[%s]' % c) for c in cols}

    def rates(c, val):
        r.setValue('[%s]' % c, val)
        out = np.array([r.getValue("[%s]'" % s) for s in rows])
        r.setValue('[%s]' % c, x0[c])
        return out
    J = np.zeros((len(rows), len(cols)))
    for k, c in enumerate(cols):
        h = 1e-4 * max(abs(x0[c]), 1e-4)
        d1 = (rates(c, x0[c] + h) - rates(c, x0[c] - h)) / (2 * h)
        d2 = (rates(c, x0[c] + h / 2) - rates(c, x0[c] - h / 2)) / h
        J[:, k] = (4 * d2 - d1) / 3
    return J


def err(a, ref):
    scale = np.abs(ref) + 1e-6 * max(np.abs(ref).max(), 1.0)
    return float((np.abs(a - ref) / scale).max())


counts = {'nitro': 0, 'libroadrunner': 0, 'both': 0, 'neither': 0, 'skipped': 0}
rows_out = []
for m in models:
    j = raw['nitro'][m + '.xml']['jac']
    rows, cols = j['rows'], j['cols']
    try:
        ref = reference(ROOT + '/tests/models/biomodels/%s.xml' % m, rows, cols)
        en, el = err(mat(m, 'nitro', rows, cols), ref), err(mat(m, 'libroadrunner', rows, cols), ref)
    except Exception as e:
        counts['skipped'] += 1
        rows_out.append('%s skipped: %s' % (m, str(e)[:80]))
        continue
    ok_n, ok_l = en < 1e-3, el < 1e-3
    k = 'both' if ok_n and ok_l else 'nitro' if ok_n else 'libroadrunner' if ok_l else 'neither'
    counts[k] += 1
    rows_out.append('%s %-14s nitro err %.1e  libroadrunner err %.1e' % (m, k, en, el))
print('Jacobian disagreements: which engine matches the independent reference (max relative error < 1e-3)')
print(counts)
for line in rows_out:
    if not line.split()[1] in ('nitro',):
        print('  ' + line)
