"""How load, simulation and Jacobian time grow with model size: roadrunner_nitro against libroadrunner.

Two kinds of model, written in Antimony and converted to SBML once; both engines load the same SBML:
- random mass-action networks: reactions A -> B, A + B -> C, A -> B + C, A + B -> C + D, rate constants and
  initial concentrations drawn at random; a network whose time course does not stay finite and bounded is
  replaced by the next seed, so every size is a model both engines can simulate;
- linear chains $Xo -> S1 -> ... -> $X1 with reversible Michaelis-Menten rate laws, as teUtils'
  getLinearChain(n, 'Michaelis'): Vm/Km0*(S - P/Keq)/(1 + S/Km0 + P/Km1), parameters drawn from 0.1 to 10.
  Their derivatives are rational functions, so they exercise the Jacobian code harder than mass action.

Three sweeps:

    reactions   random networks, 20 species, more and more reactions
    species     random networks, more and more species, three reactions per species
    chain       reversible Michaelis-Menten chains, more and more reactions (species = reactions - 1)

Needs both engines in one Python: libroadrunner (pip install libroadrunner) and roadrunner_nitro. Run:
    python benchmarks/bench_scaling.py                     # both sweeps
    python benchmarks/bench_scaling.py --sweep reactions   # one
    python benchmarks/bench_scaling.py --plot              # also bench_scaling.png (needs matplotlib)
Results: bench_scaling.json beside this script.

Times: load = RoadRunner(sbml), each load with no other instance of the model alive (libroadrunner reuses the
compiled code of a live instance, so a repeat load timed while one exists measures almost nothing); simulate = simulate(0, 50, 501) at each engine's default tolerances (first run, then
the median of repeats after resetAll; the first run includes one-off costs); with --jacobian, jac = getFullJacobian() at the
initial state (libroadrunner: finite differences; roadrunner_nitro: exact derivatives). Medians over repeats.
"""
import argparse
import gc
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
    'chain': [(n - 1, n) for n in (10, 20, 50, 100, 200, 500, 1000)],
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


def mm_chain(n_reactions, seed):
    """Antimony for $Xo -> S1 -> ... -> S(n-1) -> $X1 with reversible Michaelis-Menten rate laws (teUtils'
    getLinearChain(n, 'Michaelis'), with parameters from 0.1 rather than 0 so that no Km or Keq is zero)."""
    rnd = random.Random(seed)
    names = ['Xo'] + ['S%d' % i for i in range(1, n_reactions)] + ['X1']
    lines = []
    for k in range(1, n_reactions + 1):
        s, p = names[k - 1], names[k]
        lhs = '$' + s if s == 'Xo' else s
        rhs = '$' + p if p == 'X1' else p
        lines.append('J%d: %s -> %s; Vm%d/Km%d0*(%s - %s/Keq%d)/(1 + %s/Km%d0 + %s/Km%d1)' % (
            k, lhs, rhs, k, k, s, p, k, s, k, p, k))
    for k in range(1, n_reactions + 1):
        for par in ('Vm%d', 'Km%d0', 'Km%d1', 'Keq%d'):
            lines.append('%s = %.2f' % (par % k, rnd.uniform(0.1, 10)))
    lines += ['Xo = %d' % rnd.randint(1, 10), 'X1 = 0'] + ['S%d = 0' % i for i in range(1, n_reactions)]
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


def model(sweep, n_species, n_reactions, seed0=1):
    import roadrunner_nitro as rn
    for seed in range(seed0, seed0 + 50):
        if sweep == 'chain':  # a chain always runs: no check (it would cost libroadrunner a full compile)
            return rn.antimony_to_sbml(mm_chain(n_reactions, seed)), seed
        sbml = rn.antimony_to_sbml(random_network(n_species, n_reactions, seed))
        if well_behaved(sbml):
            return sbml, seed
    raise RuntimeError('no well-behaved network with %d species, %d reactions in 50 seeds' % (n_species, n_reactions))


def timed(f):
    t = time.perf_counter()
    out = f()
    return time.perf_counter() - t, out


def cold_load(mod, sbml):
    """Time one load with no other instance of the model alive: libroadrunner reuses the compiled code of a
    live instance of the same SBML, so a load timed while an earlier one is alive measures almost nothing."""
    gc.collect()
    return timed(lambda: mod.RoadRunner(sbml))


def measure(mod, sbml, first_load, load_reps, sim_reps, jacobian=True):
    """first_load: an already timed cold load (seconds); load_reps - 1 more are timed, keeping the last instance
    for the simulations."""
    loads, r = [first_load], None
    for i in range(load_reps - 1):
        r = None
        dt, r = cold_load(mod, sbml)
        loads.append(dt)
    if r is None:
        r = mod.RoadRunner(sbml)
    first, _ = timed(lambda: r.simulate(0, T_END, POINTS))
    sims = []
    for _ in range(sim_reps):
        r.resetAll()
        dt, _ = timed(lambda: r.simulate(0, T_END, POINTS))
        sims.append(dt)
    out = {'load': statistics.median(loads), 'sim_first': first, 'sim': statistics.median(sims), 'jac': None}
    if jacobian:
        r.resetAll()
        jacs = [timed(r.getFullJacobian)[0]]
        if jacs[0] < 1.0:  # repeat only cheap calls (libroadrunner's takes 34 s at 100 chain reactions)
            jacs += [timed(r.getFullJacobian)[0] for _ in range(sim_reps - 1)]
        out['jac'] = statistics.median(jacs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', choices=sorted(SWEEPS), action='append', help='default: all')
    ap.add_argument('--max-time', type=float, default=60.0,
                    help='seconds: in a sweep, stop timing an engine once one load takes longer')
    ap.add_argument('--max-jac', type=float, default=10.0,
                    help='seconds: in a sweep, stop timing an engine\'s Jacobian once one call takes longer (it can '
                         'grow ~30x per step: libroadrunner 1.2 s at 50 chain reactions, 34 s at 100)')
    ap.add_argument('--jacobian', action='store_true',
                    help='also time getFullJacobian (rarely used in practice; libroadrunner 2.11.0 is ~30x slower than '
                         '2.9.1 at it, reported upstream)')
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
        head = ['load lr ms', 'load nitro', 'ratio', 'sim lr ms', 'sim nitro', 'ratio']
        head += ['jac lr ms', 'jac nitro', 'ratio'] if a.jacobian else []
        print('%7s %9s %4s | ' % ('species', 'reactions', 'seed') + ' | '.join(
            '%11s %11s %7s' % tuple(head[i:i + 3]) for i in range(0, len(head), 3)))
        rows, stopped, no_jac = [], set(), set()
        for n_species, n_reactions in SWEEPS[sweep]:
            sbml, seed = model(sweep, n_species, n_reactions)
            row = {'species': n_species, 'reactions': n_reactions, 'seed': seed}
            for name, mod in engines.items():
                if name in stopped:
                    continue
                probe, _ = cold_load(mod, sbml)
                _ = None
                reps = 5 if probe < 1 else 3 if probe < 10 else 1
                row[name] = measure(mod, sbml, probe, reps, 10, jacobian=a.jacobian and name not in no_jac)
                if row[name]['load'] > a.max_time:
                    stopped.add(name)
                if (row[name]['jac'] or 0) > a.max_jac:
                    no_jac.add(name)
            rows.append(row)
            lr, nt = row.get('libroadrunner'), row['nitro']
            cells = []
            for key in ('load', 'sim', 'jac') if a.jacobian else ('load', 'sim'):
                x = lr.get(key) if lr else None
                cells.append('%11s %11.3f %6sx' % ('%.3f' % (1e3 * x) if x is not None else '-', 1e3 * nt[key],
                                                   '%.1f' % (x / nt[key]) if x is not None else '-'))
            print('%7d %9d %4d | %s' % (n_species, n_reactions, seed, ' | '.join(cells)), flush=True)
        results['sweeps'][sweep] = rows
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=1)
    print('\nresults: %s' % OUT)
    if a.plot:
        plot(results, a.jacobian)


def plot(results, jacobian=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sweeps = results['sweeps']
    keys = [('load', 'load'), ('sim', 'simulate (median)')] + ([('jac', 'Jacobian')] if jacobian else [])
    fig, axes = plt.subplots(len(sweeps), len(keys), figsize=(5 * len(keys), 4 * len(sweeps)), squeeze=False)
    xlabel = {'reactions': 'reactions (random, 20 species)', 'species': 'species (random, 3 reactions per species)',
              'chain': 'reactions (reversible Michaelis-Menten chain)'}
    for row_axes, (sweep, rows) in zip(axes, sweeps.items()):
        x = [r['species' if sweep == 'species' else 'reactions'] for r in rows]
        for ax, (key, title) in zip(row_axes, keys):
            for name, style in (('libroadrunner', 'o-'), ('nitro', 's-')):
                pts = [(xi, 1e3 * r[name][key]) for xi, r in zip(x, rows) if r.get(name, {}).get(key) is not None]
                if pts:
                    ax.plot(*zip(*pts), style, label=name)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(xlabel[sweep])
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
