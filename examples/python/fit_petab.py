"""Fit a PEtab problem with roadrunner_nitro.fit.

    python examples/python/fit_petab.py path/to/problem.yaml [n_starts]

Least-squares problems (normal noise with known standard deviations) use roadrunner_nitro's own optimizer.
Others, such as problems that also estimate their noise, are fitted here with scipy's L-BFGS-B driven by
roadrunner_nitro's objective and exact gradient: any optimizer can be used this way. (roadrunner_nitro's own
optimizer for these comes in stage 2.)

Problems from the PEtab Benchmark Collection (https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab)
work if they use only what stage 1 supports, for example problems/Bertozzi_PNAS2020/v1/problem.yaml.
"""
import sys
import time

import numpy as np

from roadrunner_nitro import fit

path = sys.argv[1]
n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 10

try:
    p = fit.Problem.from_petab(path)
except ValueError as e:
    sys.exit(f'cannot load this problem yet: {e}')
print(p)
lower, upper = p.bounds
print('free parameters:', ', '.join(f'{i} ({s})' for i, s in zip(p.free_ids, p.scales)))

nominal = p.nominal
f_nominal = p.objective(nominal)
print(f'objective at the nominal values in the parameter table: {f_nominal:.6f}')

t0 = time.perf_counter()
if p.least_squares:
    result = p.fit(n_starts=n_starts, seed=0)
    best, f_best = result.best, result.best_value
    print(result)
else:
    from scipy.optimize import minimize
    rng = np.random.default_rng(0)
    starts = lower + rng.random((n_starts, len(lower))) * (upper - lower)

    def objective_and_gradient(theta):
        try:
            return p.gradient(theta)             # (value, exact gradient)
        except RuntimeError:                     # the model could not be simulated there
            return np.inf, np.zeros_like(theta)

    fits = []
    for k, x0 in enumerate(starts):
        # tolerances tighter than scipy's defaults, which stop early on these problems
        r = minimize(objective_and_gradient, x0, jac=True, method='L-BFGS-B', bounds=list(zip(lower, upper)),
                     options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 2000})
        fits.append((r.fun, r.x))
        print(f'  start {k + 1:2d}/{n_starts}: objective {r.fun:.6f} ({r.nit} iterations)')
    f_best, best = min(fits, key=lambda t: t[0])
print(f'\nbest objective {f_best:.6f} in {time.perf_counter() - t0:.1f} s '
      f'({"better than" if f_best < f_nominal - 1e-6 else "same as" if abs(f_best - f_nominal) <= 1e-6 else "worse than"} '
      f'the nominal values: {f_nominal:.6f})')
for i, v in p.to_linear(best).items():
    print(f'  {i} = {v:.6g}')
