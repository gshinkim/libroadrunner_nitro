"""Stage gate of parameter estimation on the PEtab Benchmark Collection (spec: "Validation and performance").

For every problem in the collection (v1 format):
  1. load it with roadrunner_nitro.fit, or record why it is refused (and by which later stage);
  2. simulate at the nominal parameters and compare every measurement's simulation with AMICI's
     (simulations.tsv in the collection);
  3. compare the negative log-likelihood at the nominal parameters with AMICI's reference log-likelihood
     (AMICI's tests/benchmark_models/benchmark_models.yaml), when it has one;
  4. check the exact gradient at the nominal parameters against central finite differences (tolerances
     tightened to rtol 1e-10, atol 1e-14 for this check);
  5. time the objective and the objective with gradient;
  6. least-squares problems: multistart fit, compared with the objective at the nominal (published) parameters.

usage: python benchmarks/petab_gate.py <collection>/problems [--amici-ref benchmark_models.yaml]
                                       [--starts 20] [--only NAME ...] [--out results.json]
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

import numpy as np

from roadrunner_nitro import fit


def amici_refs(path):
    """{problem: llh} from AMICI's benchmark_models.yaml (a flat 'Name:\\n  llh: value' file)."""
    out, cur = {}, None
    if not path:
        return out
    for line in open(path, encoding='utf-8'):
        m = re.match(r'^(\w+):\s*$', line)
        if m:
            cur = m.group(1)
        m = re.match(r'^\s+llh:\s*(\S+)', line)
        if m and cur:
            out[cur] = float(m.group(1))
    return out


def stage_of(msg):
    for key, stage in [('placeholders', 'stage 2: placeholders'), ('pre-equilibration', 'stage 2: pre-equilibration'),
                       ('steady-state', 'stage 2: steady-state measurements'), ('priors', 'stage 3: priors'),
                       ('events', 'stage 4: events'), ('compartment', 'stage 1: compartment sizes'),
                       ('stage 2', 'stage 2: other')]:
        if key in msg:
            return stage
    return 'unsupported (other)'


def read_tsv(path):
    lines = [l.rstrip('\n').split('\t') for l in open(path, encoding='utf-8-sig') if l.strip()]
    head = [h.strip() for h in lines[0]]
    return [dict(zip(head, [c.strip() for c in row] + [''] * (len(head) - len(row)))) for row in lines[1:]]


def compare_simulations(p, theta, sim_path):
    """Max relative difference |ours - AMICI| / max(|AMICI|, 1e-8 * scale), matching rows by (condition,
    observable, time) in order of appearance."""
    ours = p.simulate(theta)
    meas = p._p.measurements
    ref = {}
    for row in read_tsv(sim_path):
        key = (row['simulationConditionId'], row['observableId'], float(row['time']))
        ref.setdefault(key, []).append(float(row['simulation']))
    used = {}
    errs = []
    scale = max(1e-300, max(abs(v) for vs in ref.values() for v in vs))
    for (c, o, t, _), y in zip(meas, ours):
        k = (c, o, t)
        i = used.get(k, 0)
        used[k] = i + 1
        r = ref[k][i]
        errs.append(abs(y - r) / max(abs(r), 1e-8 * scale))
    return float(max(errs)), float(np.median(errs))


STEPS = (1e-3, 1e-4, 1e-5, 1e-6)
MAX_COMPONENTWISE = 20


def gradient_check(p, theta):
    """Up to MAX_COMPONENTWISE parameters: every component (gradient_components); above: directional derivatives
    g . v along 4 random unit directions against central differences along v (gradient_directions), a check of the
    whole gradient at a fixed cost."""
    if len(theta) <= MAX_COMPONENTWISE:
        return gradient_components(p, theta)
    rt = (p.rtol, p.atol)
    p.rtol, p.atol = 1e-10, 1e-14
    try:
        _, g = p.gradient(theta)
        rng = np.random.default_rng(1)
        errs = []
        for _ in range(4):
            v = rng.standard_normal(len(theta))
            v /= np.linalg.norm(v)
            exact = float(g @ v)
            fd = [(p.objective(theta + h * v) - p.objective(theta - h * v)) / (2 * h) for h in STEPS]
            errs.append(min(abs(f - exact) for f in fd) / max(abs(exact), 1e-3 * np.linalg.norm(g), 1e-12))
    finally:
        p.rtol, p.atol = rt
    return {'max_rel_err': float(max(errs)), 'directional': True, 'grad_norm': float(np.linalg.norm(g))}


def gradient_components(p, theta):
    """Exact gradient against central differences at several step sizes. Finite differences of an adaptive
    integration are noisy at small steps and biased at large ones, so each component takes the step that agrees
    best (as AMICI's benchmark tests do). The error of a component is relative to max(|g_k|, 1e-3 * |g| / sqrt(n))
    so near-zero components of a large gradient do not dominate."""
    rt = (p.rtol, p.atol)
    p.rtol, p.atol = 1e-10, 1e-14
    try:
        _, g = p.gradient(theta)
        fds = []
        for h in STEPS:
            fds.append([(p.objective(theta + h * e) - p.objective(theta - h * e)) / (2 * h) for e in np.eye(len(theta))])
    finally:
        p.rtol, p.atol = rt
    fds = np.array(fds)
    gn = np.linalg.norm(g)
    floor = 1e-3 * gn / np.sqrt(len(g)) + 1e-12
    err = np.abs(fds - g[None, :]).min(axis=0) / np.maximum(np.abs(g), floor)
    best = fds[np.abs(fds - g[None, :]).argmin(axis=0), np.arange(len(g))]
    return {'max_rel_err': float(err.max()), 'worst': p.free_ids[int(err.argmax())], 'grad_norm': float(gn),
            'norm_rel_err': float(np.linalg.norm(g - best) / max(gn, 1e-300))}


def off_optimum(p, theta, seed=0):
    """A point away from the optimum (each parameter moved 0.1 of its scaled range toward the middle, a random
    sign elsewhere), where gradients are large and the comparison sharp."""
    lo, hi = p.bounds
    rng = np.random.default_rng(seed)
    step = 0.1 * (hi - lo) * rng.choice([-1, 1], size=len(theta))
    x = theta + step
    out = (x < lo) | (x > hi)
    x[out] = theta[out] - step[out]
    return np.clip(x, lo, hi)


def timed(f, n=5):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def run_problem(name, root, refs, starts):
    y = os.path.join(root, name, 'v1', 'problem.yaml')
    r = {'problem': name}
    t = time.perf_counter()
    try:
        p = fit.Problem.from_petab(y)
    except Exception as e:
        msg = str(e).splitlines()[0]
        return {**r, 'loaded': False, 'refused': msg, 'blocked_by': stage_of(msg)}
    r.update(loaded=True, load_s=time.perf_counter() - t, n_free=len(p.free_ids), n_meas=len(p._p.measurements),
             least_squares=p.least_squares)
    th = p.nominal
    try:
        r['nllh_nominal'] = p.objective(th)
    except Exception as e:
        r['error'] = f'objective at nominal: {e}'
        return r
    sim = os.path.join(root, name, 'v1', 'simulations.tsv')
    if os.path.exists(sim):
        r['sim_max_rel_err'], r['sim_median_rel_err'] = compare_simulations(p, th, sim)
    if name in refs:
        r['amici_nllh'] = -refs[name]
        r['nllh_diff'] = r['nllh_nominal'] - r['amici_nllh']
    r['gradient'] = gradient_check(p, th)
    try:
        r['gradient_off_optimum'] = gradient_check(p, off_optimum(p, th))
    except Exception as e:  # the model may fail to simulate there
        r['gradient_off_optimum'] = {'error': str(e)}
    r['t_objective_s'] = timed(lambda: p.objective(th))
    r['t_gradient_s'] = timed(lambda: p.gradient(th))
    if p.least_squares and starts:
        t = time.perf_counter()
        f = p.fit(n_starts=starts, seed=0)
        r['fit'] = {'s': time.perf_counter() - t, 'best': f.best_value, 'n_at_best': f.n_at_best(),
                    'starts': starts, 'reaches_nominal': f.best_value <= r['nllh_nominal'] + 1e-3}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('problems')
    ap.add_argument('--amici-ref')
    ap.add_argument('--starts', type=int, default=20)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--out')
    ap.add_argument('--timeout', type=float, default=900, help='seconds per problem')
    ap.add_argument('--one', help=argparse.SUPPRESS)  # internal: run one problem, print its JSON
    a = ap.parse_args()
    refs = amici_refs(a.amici_ref)
    if a.one:
        print(json.dumps(run_problem(a.one, a.problems, refs, a.starts)), flush=True)
        return
    import subprocess
    names = sorted(n for n in os.listdir(a.problems) if os.path.isdir(os.path.join(a.problems, n, 'v1')))
    if a.only:
        names = [n for n in names if n in a.only]
    out = []
    for n in names:
        cmd = [sys.executable, os.path.abspath(__file__), a.problems, '--one', n, '--starts', str(a.starts)]
        if a.amici_ref:
            cmd += ['--amici-ref', a.amici_ref]
        t0 = time.perf_counter()
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            lines = [l for l in cp.stdout.splitlines() if l.startswith('{')]
            r = json.loads(lines[-1]) if lines else {'problem': n, 'loaded': True, 'error': (cp.stderr.strip().splitlines() or ['no output'])[-1]}
        except subprocess.TimeoutExpired:
            r = {'problem': n, 'loaded': True, 'error': f'over the {a.timeout:.0f} s limit'}
        r['wall_s'] = time.perf_counter() - t0
        out.append(r)
        if r.get('error'):
            print(f'{n:40s} ERROR: {r["error"]}', flush=True)
            continue
        if not r['loaded']:
            print(f'{n:40s} refused ({r["blocked_by"]})', flush=True)
            continue
        g, g2 = r.get('gradient', {}), r.get('gradient_off_optimum', {})
        print(f'{n:40s} {r["n_free"]:4d} free  sim err {r.get("sim_max_rel_err", float("nan")):.1e}  '
              f'nllh {r.get("nllh_nominal", float("nan")):.6f} (AMICI {r.get("amici_nllh", float("nan")):.6f})  '
              f'grad err {g.get("max_rel_err", float("nan")):.1e} / off-opt {g2.get("max_rel_err", float("nan")):.1e}  '
              f't_obj {r.get("t_objective_s", 0)*1e3:.1f} ms  '
              f't_grad {r.get("t_gradient_s", 0)*1e3:.1f} ms', flush=True)
    loaded = sum(r['loaded'] for r in out)
    blocked = {}
    for r in out:
        if not r['loaded']:
            blocked[r['blocked_by']] = blocked.get(r['blocked_by'], 0) + 1
    print(f'\n{loaded} of {len(out)} problems load; refused: ' + ', '.join(f'{v} {k}' for k, v in sorted(blocked.items())))
    if a.out:
        meta = {'roadrunner_nitro': fit.core.__name__, 'python': sys.version.split()[0], 'date': time.strftime('%Y-%m-%d')}
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump({'meta': meta, 'problems': out}, f, indent=1)


if __name__ == '__main__':
    main()
