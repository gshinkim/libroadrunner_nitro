"""The simplest fit: two rate constants from one time course.

    python examples/python/fit_simple.py

A species S1 is made at rate k1*X0 and broken down at rate k2*S1. We measured S1 at ten times and want k1 and k2.
(The data were made with k1 = 0.5 and k2 = 0.2 plus noise, so the fit comes out close to those values.)
"""
from roadrunner_nitro import fit

model = """
    J1: $X0 -> S1; k1*X0
    J2: S1 -> ;    k2*S1
    X0 = 10; S1 = 0; k1 = 1; k2 = 1
"""
data = {'time': [0, 2, 4, 6, 8, 10, 15, 20, 25, 30],
        'S1':   [0.0, 7.0, 14.0, 17.2, 19.7, 21.5, 22.7, 24.4, 24.4, 26.6]}

result = fit.fit_timecourse(model, data, ['k1', 'k2'])
print(result)

# Fit of 2 parameters to 10 data points (9 of 10 starts reached the best fit)
#   k1 = 0.47004    (95% interval 0.4275 to 0.5168)
#   k2 = 0.18585    (95% interval 0.1652 to 0.2091)
#   measurement error estimated from the scatter of the data: 0.7813

try:
    import matplotlib.pyplot as plt
    result.plot()                  # the data and the fitted curve
    plt.show()
except ImportError:
    pass
