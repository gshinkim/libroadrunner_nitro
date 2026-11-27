"""The simplest fit: two rate constants from one time course.

    python examples/python/fit_simple.py

A species S1 is made at rate k1*X0 and broken down at rate k2*S1. We measured S1 at ten times and want k1 and k2.
(The data below were made with k1 = 0.5 and k2 = 0.2 plus a little noise, so the fit comes out close to those
values but not exactly on them: k1 = 0.47, k2 = 0.19.)
"""
from roadrunner_nitro import fit

# 1. The model, written in Antimony. The values of k1 and k2 here are only starting guesses.
p = fit.Problem.from_antimony("""
    J1: $X0 -> S1; k1*X0
    J2: S1 -> ;    k2*S1
    X0 = 10; S1 = 0
    k1 = 1; k2 = 1
""")

# 2. What to estimate, and the range to search in. scale='log10' searches the range evenly in orders of magnitude
#    (0.001-0.01, 0.01-0.1, ... get the same attention); without it most starting points would be above 10.
p.add_parameter('k1', scale='log10', lower=0.001, upper=100)
p.add_parameter('k2', scale='log10', lower=0.001, upper=100)

# 3. The experiment (here just one: the model as it is) ...
p.add_condition('experiment')

# ... what was measured (S1, with a measurement error of about 0.5) ...
p.add_observable('S1_measured', 'S1', noise_formula='0.5')

# ... and the data.
times = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30]
values = [0.0, 7.0, 14.0, 17.2, 19.7, 21.5, 22.7, 24.4, 24.4, 26.6]
p.add_measurements('experiment', 'S1_measured', times, values)

# 4. Fit, starting from 10 different points in the search range.
result = p.fit(n_starts=10)
print(result)

# 5. How well are k1 and k2 determined? 95% confidence intervals.
print(p.fisher(result.best))
