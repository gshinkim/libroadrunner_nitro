"""How load and simulation time grow with model size: roadrunner_nitro against libroadrunner on random networks.

Random mass-action networks (reactions A -> B, A + B -> C, A -> B + C, A + B -> C + D, rate constants and initial
concentrations drawn at random), written in Antimony and converted to SBML once; both engines load the same SBML.
A network whose time course does not stay finite and bounded is replaced by the next seed, so every size is a
model both engines can simulate. Two sweeps:

    reactions   20 species, more and more reactions
    species     more and more species, three reactions per species

Needs both engines in one Python: libroadrunner (pip install libroadrunner) and roadrunner_nitro. Run:
    python benchmarks/bench_scaling.py                     # both sweeps
    python benchmarks/bench_scaling.py --sweep reactions   # one
    python benchmarks/bench_scaling.py --plot              # also bench_scaling.png (needs matplotlib)
Results: bench_scaling.json beside this script.

Times: load = RoadRunner(sbml); simulate = simulate(0, 50, 501) at each engine's default tolerances (first run, then
the median of repeats after resetAll; the first run includes one-off costs). Medians over the load repeats too.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'bench_scaling.json')
T_END, POINTS = 50.0, 501
SWEEPS = {
    'reactions': [(20, r) for r in (20, 50, 100, 200, 300, 500, 800)],
    'species': [(n, 3 * n) for n in (10, 20, 50, 100, 200, 400)],
}


def random_network(n_species, n_reactions, seed):
    """Antimony for a random mass-action network; reactions never have the same species on both sides."""
    rnd = random.Random(seed)
    species = ['S%d' % i for i in range(n_species)]
    lines = []
    for j in range(n_reactions):
        while True:
            n_in, n_out = rnd.choice([(1, 1), (2, 1), (1, 2), (2, 2)])
            ins = rnd.sample(species, n_in) if n_in <= n_species else [rnd.choice(species)]
            outs = rnd.sample(species, n_out) if n_out <= n_species else [rnd.choice(species)]
            if not set(ins) & set(outs):
                break
        rate = '*'.join(['k%d' % j] + ins)
        lines.append('J%d: %s -> %s; %s' % (j, ' + '.join(ins), ' + '.join(outs), rate))
        lines.append('k%d = %.4g' % (j, 10 ** rnd.uniform(-1, 0)))
    lines += ['%s = %.4g' % (s, rnd.uniform(1, 10)) for s in species]
    return '\n'.join(lines)


def well_behaved(sbml):
    """Both engines run it, and the concentrations stay finite, non-negative and below 1e6."""
    import roadrunner
    import roadrunner_nitro as rn
    for mod in (rn, roadrunner):
        try:
            y = np.asarray(mod.RoadRunner(sbml).simulate(0, T_END, POINTS), float)[:, 1:]
        except Exception:
            return False
        if not (np.isfinite(y).all() and y.min() > -1e-6 and y.max() < 1e6):
            return False
    return True


def model(n_species, n_reactions, seed0=1):
    import roadrunner_nitro as rn
    for seed in range(seed0, seed0 + 50):
        sbml = rn.antimony_to_sbml(random_network(n_species, n_reactions, seed))
        if well_behaved(sbml):
            return sbml, seed
    raise RuntimeError('no well-behaved network with %d species, %d reactions in 50 seeds' % (n_species, n_reactions))


def timed(f):
    t = time.perf_counter()
    out = f()
    return time.perf_counter() - t, out


def measure(mod, sbml, load_reps, sim_reps):
    loads = []
    for _ in range(load_reps):
        dt, r = timed(lambda: mod.RoadRunner(sbml))
        loads.append(dt)
    first, _ = timed(lambda: r.simulate(0, T_END, POINTS))
    sims = []
    for _ in range(sim_reps):
        r.resetAll()
        dt, _ = timed(lambda: r.simulate(0, T_END, POINTS))
        sims.append(dt)
    return {'load': statistics.median(loads), 'sim_first': first, 'sim': statistics.median(sims)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', choices=sorted(SWEEPS), action='append', help='default: both')
    ap.add_argument('--max-load', type=float, default=60.0,
                    help='seconds: stop timing an engine in a sweep once one load takes longer')
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()
    warnings.filterwarnings('ignore')
    import roadrunner
    import roadrunner_nitro as rn
    engines = {'nitro': rn, 'libroadrunner': roadrunner}
    warm = rn.antimony_to_sbml(random_network(3, 3, 0))  # one-off start-up costs, outside the timings
    for mod in engines.values():
        mod.RoadRunner(warm).simulate(0, 1, 11)
    results = {'settings': {'t_end': T_END, 'points': POINTS, 'tolerances': 'each engine default',
                            'libroadrunner': roadrunner.__version__, 'nitro': rn.__version__, 'python': sys.version},
               'sweeps': {}}
    for sweep in a.sweep or sorted(SWEEPS):
        print('\nsweep: %s' % sweep)
        print('%8s %9s %6s | %12s %12s %8s | %12s %12s %8s' % ('species', 'reactions', 'seed', 'load lr ms', 'load nitro',
                                                             'ratio', 'sim lr ms', 'sim nitro', 'ratio'))
        rows, stopped = [], set()
        for n_species, n_reactions in SWEEPS[sweep]:
            sbml, seed = model(n_species, n_reactions)
            row = {'species': n_species, 'reactions': n_reactions, 'seed': seed}
            for name, mod in engines.items():
                if name in stopped:
                    continue
                probe, _ = timed(lambda: mod.RoadRunner(sbml))
                reps = 5 if probe < 1 else 3 if probe < 10 else 1
                row[name] = measure(mod, sbml, reps, 10)
                if row[name]['load'] > a.max_load:
                    stopped.add(name)
            rows.append(row)
            lr, nt = row.get('libroadrunner'), row['nitro']
            f = lambda x: '%12.3f' % (1e3 * x)
            print('%8d %9d %6d | %s %s %7.1fx | %s %s %7.1fx' % (
                n_species, n_reactions, seed,
                f(lr['load']) if lr else '%12s' % '-', f(nt['load']), lr['load'] / nt['load'] if lr else float('nan'),
                f(lr['sim']) if lr else '%12s' % '-', f(nt['sim']), lr['sim'] / nt['sim'] if lr else float('nan')),
                flush=True)
        results['sweeps'][sweep] = rows
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=1)
    print('\nresults: %s' % OUT)
    if a.plot:
        plot(results)


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sweeps = results['sweeps']
    fig, axes = plt.subplots(len(sweeps), 2, figsize=(10, 4 * len(sweeps)), squeeze=False)
    for row_axes, (sweep, rows) in zip(axes, sweeps.items()):
        x = [r[sweep] for r in rows]
        for ax, key, title in zip(row_axes, ('load', 'sim'), ('load', 'simulate (median)')):
            for name, style in (('libroadrunner', 'o-'), ('nitro', 's-')):
                pts = [(xi, 1e3 * r[name][key]) for xi, r in zip(x, rows) if name in r]
                ax.plot(*zip(*pts), style, label=name)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('%s (%s)' % (sweep, '20 species' if sweep == 'reactions' else '3 reactions per species'))
            ax.set_ylabel('ms')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(HERE, 'bench_scaling.png')
    fig.savefig(path, dpi=120)
    print('plot: %s' % path)


if __name__ == '__main__':
    main()
