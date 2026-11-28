"""Global methods on the PEtab Benchmark Collection (spec: "Optimizers", stage 2, item 7): which particle swarm
and differential evolution variants to use, chosen on the collection.

For every problem that loads (or those given with --only), and each seed:
  - multistart: n_starts Latin hypercube starts, each polished by the local method (the stage 1 method);
  - pso, de, shade: the global method with its default population and budget, then its n_polish best distinct
    points polished by the same local method.
Recorded per run: best objective, wall time, objective evaluations of the global stage. A run "reaches the best"
when its best objective is within --tol of the best any run found on that problem, and "reaches the nominal" when
it is within --tol of (or below) the objective at the published parameters.

usage: python benchmarks/global_compare.py <collection>/problems [--only NAME ...] [--seeds 3] [--starts 20]
                                           [--polish 5] [--out results.json]
"""
import argparse
import json
import os
import subprocess
import sys
import time

from roadrunner_nitro import fit

METHODS = ['multistart', 'pso', 'de', 'shade']


def run_problem(name, root, seeds, starts, polish):
    p = fit.Problem.from_petab(os.path.join(root, name, 'v1', 'problem.yaml'))
    out = {'problem': name, 'n_free': len(p.free_ids), 'least_squares': p.least_squares}
    try:
        out['nominal'] = p.objective(p.nominal)
    except Exception as e:  # noqa: BLE001
        out['nominal'] = None
        out['nominal_error'] = str(e)
    runs = []
    for m in METHODS:
        for s in range(seeds):
            t = time.perf_counter()
            try:
                if m == 'multistart':
                    r = p.fit(n_starts=starts, seed=s)
                else:
                    r = p.fit(n_starts=polish, seed=s, global_method=m)
                g = r.global_stage or {}
                runs.append({'method': m, 'seed': s, 'best': r.best_value, 's': time.perf_counter() - t,
                             'global_best': g.get('best'), 'global_nfev': g.get('nfev'),
                             'global_generations': g.get('generations'), 'global_message': g.get('message'),
                             'local_nfev': int(sum(r.nfev)), 'local_njev': int(sum(r.njev))})
            except Exception as e:  # noqa: BLE001
                runs.append({'method': m, 'seed': s, 'error': str(e), 's': time.perf_counter() - t})
    out['runs'] = runs
    return out


def summarize(r, tol):
    ok = [x['best'] for x in r['runs'] if 'best' in x and x['best'] == x['best']]
    best = min(ok) if ok else float('inf')
    nom = r.get('nominal')
    r['best_known'] = best
    for x in r['runs']:
        if 'best' in x:
            x['reaches_best'] = x['best'] <= best + tol
            x['reaches_nominal'] = nom is not None and x['best'] <= nom + tol
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('problems')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--skip', nargs='*', default=['Chen_MSB2009'])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--starts', type=int, default=20)
    ap.add_argument('--polish', type=int, default=5)
    ap.add_argument('--tol', type=float, default=1e-3)
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--out')
    ap.add_argument('--one', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        print(json.dumps(run_problem(a.one, a.problems, a.seeds, a.starts, a.polish)), flush=True)
        return
    names = sorted(n for n in os.listdir(a.problems) if os.path.isdir(os.path.join(a.problems, n, 'v1')))
    if a.only:
        names = [n for n in names if n in a.only]
    names = [n for n in names if n not in (a.skip or [])]
    results = []
    for n in names:
        try:
            fit.Problem.from_petab(os.path.join(a.problems, n, 'v1', 'problem.yaml'))
        except Exception:  # noqa: BLE001  (refused: not part of the comparison)
            continue
        cmd = [sys.executable, os.path.abspath(__file__), a.problems, '--one', n, '--seeds', str(a.seeds),
               '--starts', str(a.starts), '--polish', str(a.polish)]
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            lines = [l for l in cp.stdout.splitlines() if l.startswith('{')]
            r = json.loads(lines[-1]) if lines else {'problem': n, 'error': (cp.stderr.strip().splitlines() or ['no output'])[-1], 'runs': []}
        except subprocess.TimeoutExpired:
            r = {'problem': n, 'error': f'over the {a.timeout:.0f} s limit', 'runs': []}
        r = summarize(r, a.tol)
        results.append(r)
        cells = []
        for m in METHODS:
            rs = [x for x in r['runs'] if x['method'] == m]
            hit = sum(1 for x in rs if x.get('reaches_best'))
            t = sum(x['s'] for x in rs) / max(1, len(rs))
            cells.append(f'{m} {hit}/{len(rs)} {t:6.1f}s')
        print(f"{n:36s} best {r.get('best_known', float('nan')):14.6f} nominal {r.get('nominal') or float('nan'):14.6f}  " + '  '.join(cells), flush=True)
    print()
    for m in METHODS:
        rs = [x for r in results for x in r['runs'] if x['method'] == m]
        hit = sum(1 for x in rs if x.get('reaches_best'))
        nom = sum(1 for x in rs if x.get('reaches_nominal'))
        t = sum(x['s'] for x in rs)
        print(f'{m:10s} reaches best {hit}/{len(rs)}, reaches nominal {nom}/{len(rs)}, total {t:.0f} s')
    if a.out:
        meta = {'seeds': a.seeds, 'starts': a.starts, 'polish': a.polish, 'tol': a.tol,
                'threads': os.environ.get('RAYON_NUM_THREADS', 'all'), 'python': sys.version.split()[0]}
        json.dump({'meta': meta, 'problems': results}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
