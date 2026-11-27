"""Fit three rate constants from two measured species.

    python examples/python/fit_two_species.py

The pathway X0 -> S1 -> S2 -> with X0 held fixed. Both S1 and S2 were measured; each is a column of the data,
and both are fitted at once. (The data were made with k1 = 0.5, k2 = 0.3, k3 = 0.1 plus noise of standard
deviation 0.3.)
"""
from roadrunner_nitro import fit

model = """
    J1: $X0 -> S1; k1*X0
    J2: S1 -> S2;  k2*S1
    J3: S2 -> ;    k3*S2
    X0 = 10; S1 = 0; S2 = 0
    k1 = 1; k2 = 1; k3 = 1
"""
nan = float('nan')
data = {'time': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
        'S1':   [0.04, 9.85, 14.1, 15.58, 16.05, 16.59, 16.98, 16.92, 16.44, 16.28, 16.48],
        'S2':   [0.01, 3.91, 12.91, nan, 27.87, 33.38, 37.62, 40.99, 43.53, 44.93, 46.68]}   # nan: not measured

result = fit.fit_timecourse(model, data, ['k1', 'k2', 'k3'])
print(result)

# Fit of 3 parameters to 21 data points (10 of 10 starts reached the best fit)
#   k1 = 0.49284    (95% interval 0.4859 to 0.4998)
#   k2 = 0.29536    (95% interval 0.2906 to 0.3002)
#   k3 = 0.097288   (95% interval 0.09498 to 0.09965)
#   measurement error estimated from the scatter of the data: 0.2431
#
# The measurement error is estimated from the data: one value shared by both species. From only 21 points it came
# out at 0.24 against the true 0.3, so these intervals are a little too narrow (k1's and k3's just miss the true
# values). If you know the error, give it, especially when the species differ in size or were measured
# differently:
#
#   result = fit.fit_timecourse(model, data, ['k1', 'k2', 'k3'], sigma={'S1': 0.3, 'S2': 0.3})
#
# which gives the same values with intervals k1 0.4843 to 0.5015, k2 0.2895 to 0.3013, k3 0.09445 to 0.1002.

try:
    import matplotlib.pyplot as plt
    result.plot()                  # both species: data as points, the fit as lines
    plt.show()
except ImportError:
    pass
