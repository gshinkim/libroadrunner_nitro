"""SBML Test Suite semantic cases: how one case is run and judged, shared by tests/test_sbml_semantic_suite.py and
benchmarks/sbml_test_suite.py. Adapted from Tellurium Nitro's benchmarks/sts/run_core.py.

A case passes when every point of its time course satisfies |got - expected| <= absolute + relative * |expected|,
with the case's own tolerances from its -settings.txt, as the suite's README defines. Flux-balance cases are not
supported (they are linear programs, not simulations) and are reported as skipped.
"""
import os

import numpy as np

LEVELS = ['l3v2', 'l3v1', 'l2v5', 'l2v4', 'l2v3', 'l2v2', 'l2v1', 'l1v2']  # newest SBML level first
RTOL, ATOL = 1e-8, 1e-10  # integrator tolerances; the pass tolerances come from each case's settings
HERE = os.path.dirname(os.path.abspath(__file__))
SEMANTIC = os.environ.get('SBML_SEMANTIC_DIR', os.path.join(HERE, 'models', 'sbml-test-suite', 'cases', 'semantic'))


def case_ids(root=SEMANTIC):
    return sorted(c for c in os.listdir(root) if c.isdigit()) if os.path.isdir(root) else []


def settings(path):
    s = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                s[k.strip()] = v.strip()
    lst = lambda k: [x.strip() for x in s.get(k, '').split(',') if x.strip()]  # noqa: E731
    return dict(start=float(s['start']), duration=float(s['duration']), steps=int(s['steps']),
                variables=lst('variables'), amount=set(lst('amount')), concentration=set(lst('concentration')),
                atol=float(s['absolute']), rtol=float(s['relative']))


def tags(path):
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            for k in ('componentTags', 'testTags', 'testType'):
                if line.startswith(k + ':'):
                    out[k] = [x.strip() for x in line.split(':', 1)[1].split(',') if x.strip()]
    return out


def run_case(RoadRunner, cid, root=SEMANTIC):
    """Run case ``cid`` with the RoadRunner class given (roadrunner_nitro's or libroadrunner's).
    Returns dict(id, status: pass | fail | error | skip, reason, ...)."""
    d = os.path.join(root, cid)
    rec = dict(id=cid, **tags(os.path.join(d, cid + '-model.m')))
    types = rec.get('testType', [])
    if 'FluxBalanceSteadyState' in types:
        return dict(rec, status='skip', reason='flux balance: not supported')
    if 'TimeCourse' not in types:
        return dict(rec, status='skip', reason='not a time course: %s' % types)
    lv = next((l for l in LEVELS if os.path.exists(os.path.join(d, '%s-sbml-%s.xml' % (cid, l)))), None)
    if lv is None:
        return dict(rec, status='skip', reason='no SBML file')
    rec['level'] = lv
    st = settings(os.path.join(d, cid + '-settings.txt'))
    exp = np.genfromtxt(os.path.join(d, cid + '-results.csv'), delimiter=',', names=True, deletechars='')
    try:
        r = RoadRunner(os.path.join(d, '%s-sbml-%s.xml' % (cid, lv)))
    except Exception as e:
        return dict(rec, status='error', stage='load', reason='%s: %s' % (type(e).__name__, str(e)[:300]))
    sel = ['[%s]' % v if v in st['concentration'] else v for v in st['variables']]
    n = st['steps'] + 1
    try:
        r.integrator.relative_tolerance = RTOL
        r.integrator.absolute_tolerance = ATOL
        res = np.asarray(r.simulate(st['start'], st['start'] + st['duration'], n, sel), dtype=float)
        # the time column is the requested grid (a model id may itself be named 'time')
        res = np.column_stack([np.linspace(st['start'], st['start'] + st['duration'], n), res.reshape(n, -1)])
    except Exception as e:
        return dict(rec, status='error', stage='simulate', reason='%s: %s' % (type(e).__name__, str(e)[:300]))
    want = np.column_stack([exp[c] for c in exp.dtype.names])
    if want.shape != res.shape:
        return dict(rec, status='fail', reason='shape %s, expected %s' % (res.shape, want.shape))
    ok = np.isclose(res, want, rtol=0, atol=0, equal_nan=True) | (np.abs(res - want) <= st['atol'] + st['rtol'] * np.abs(want))
    if ok.all():
        return dict(rec, status='pass')
    i, j = np.argwhere(~ok)[0]
    return dict(rec, status='fail', n_bad=int((~ok).sum()),
                reason='%s at t=%g: got %r, expected %r' % ((['time'] + st['variables'])[j], want[i, 0], res[i, j], want[i, j]))
