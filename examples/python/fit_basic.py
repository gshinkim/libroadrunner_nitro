"""Fit a model to data with roadrunner_nitro.fit: the whole workflow on a small example.

    python examples/python/fit_basic.py

The model: X0 -> S1 -> S2 ->, with X0 held fixed. Two experiments: the model as it is, and one with X0 halved.
We measure S1 and S2, S2 only up to an unknown scale factor (common with relative measurements such as western
blots). The "lab data" are made by simulating the model at known parameter values and adding noise, so we can
see whether the fit recovers them.
"""
import numpy as np

import roadrunner_nitro as rn
from roadrunner_nitro import fit

MODEL = """
model pathway
    compartment cell = 1
    species $X0 in cell, S1 in cell, S2 in cell
    J1: X0 -> S1; k1*X0
    J2: S1 -> S2; k2*S1
    J3: S2 -> ;   k3*S2
    X0 = 10; S1 = 0; S2 = 0
    k1 = 1; k2 = 1; k3 = 1
end
"""
sbml = rn.antimony_to_sbml(MODEL)

# ------------------------------------------------------------------ 0. "lab data"

truth = {'k1': 0.5, 'k2': 0.3, 'k3': 0.1, 'scale_S2': 2.0}
times = np.array([1, 2, 4, 6, 8, 10, 15, 20, 25, 30])
rng = np.random.default_rng(1)
data = {}
for condition, x0 in [('control', 10), ('low_X0', 5)]:
    r = rn.RoadRunner(sbml)
    r['k1'], r['k2'], r['k3'], r['X0'] = truth['k1'], truth['k2'], truth['k3'], x0
    sim = r.simulate(0, 30, 31, ['time', 'S1', 'S2'])          # one row per time unit
    s1, s2 = sim[times, 1], sim[times, 2]
    data[condition, 'S1_obs'] = s1 + 0.2 * rng.standard_normal(times.size)                       # sd 0.2
    data[condition, 'S2_obs'] = truth['scale_S2'] * s2 * 10 ** (0.1 * rng.standard_normal(times.size))  # 0.1 in log10

# ------------------------------------------------------------------ 1. the problem

p = fit.Problem(sbml)

# Parameters to estimate: scale (log10 is usual for rate constants), bounds and a nominal value.
for k in ['k1', 'k2', 'k3']:
    p.add_parameter(k, scale='log10', lower=1e-3, upper=1e3, nominal=1)
p.add_parameter('scale_S2', scale='log10', lower=1e-2, upper=1e2, nominal=1)   # not in the model: used below

# Experiments: what each changes from the model.
p.add_condition('control')
p.add_condition('low_X0', {'X0': 5})

# What is measured, and its noise (standard deviation; on the log10 scale for the log10-transformed one).
p.add_observable('S1_obs', 'S1', noise_formula='0.2')
p.add_observable('S2_obs', 'scale_S2 * S2', noise_formula='0.1', transformation='log10')

for (condition, observable), values in data.items():
    p.add_measurements(condition, observable, times, values)
print(p)

# ------------------------------------------------------------------ 2. objective and its exact gradient

theta = p.nominal                  # the free parameters in p.free_ids order, each on its own scale (here log10)
value, gradient = p.gradient(theta)
print(f'\nat the nominal values: objective {value:.2f} (a negative log-likelihood), gradient {np.round(gradient, 2)}')

# ------------------------------------------------------------------ 3. fit from many starting points

def progress(done, total, best):
    print(f'  start {done:2d}/{total}: best objective so far {best:.4f}')
    return False                   # True would stop the fit early

result = p.fit(n_starts=10, seed=0, callback=progress)
print()
print(result)                      # best objective, how many starts found it, the best parameters
print('true values:', truth)

# ------------------------------------------------------------------ 4. 95% confidence intervals

ci = p.fisher(result.best, level=0.95)
print()
print(ci)

# ------------------------------------------------------------------ 5. tables (needs pandas), plot (needs matplotlib)

try:
    print('\nevery start, best first (did they agree on the optimum?):')
    print(result.to_dataframe()[['value', 'converged', 'nit', 'k1', 'k2', 'k3', 'scale_S2']].round(4).to_string())
except ImportError as e:
    print(e)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = p.measurement_table(result.best)       # the data with the fitted simulation beside it
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, obs in zip(axes, ['S1_obs', 'S2_obs']):
        for condition, colour in [('control', 'C0'), ('low_X0', 'C1')]:
            t = table[(table['observableId'] == obs) & (table['simulationConditionId'] == condition)]
            ax.plot(t['time'], t['measurement'], 'o', color=colour, label=f'{condition} data')
            ax.plot(t['time'], t['simulation'], '-', color=colour, label=f'{condition} fit')
        ax.set_title(obs)
        ax.set_xlabel('time')
    axes[1].set_yscale('log')
    axes[0].legend()
    fig.tight_layout()
    fig.savefig('fit_basic.png', dpi=120)
    print('\nplot of the fit: fit_basic.png')
except ImportError:
    pass
