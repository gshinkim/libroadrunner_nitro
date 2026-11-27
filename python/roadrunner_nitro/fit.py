"""Parameter estimation (stage 1): fit SBML models to data, with exact gradients, multistart least squares and
Fisher-information intervals. The work happens in Rust (roadrunner_nitro.core.FitProblem), the same core the C
API (rrc_fit.h) exposes; this module adds a Python-friendly surface.

    from roadrunner_nitro import fit

    p = fit.Problem.from_petab('problem.yaml')          # PEtab v1
    r = p.fit(n_starts=20, seed=1)
    print(r)                                            # best objective, parameters, convergence
    r.to_dataframe()                                    # one row per start (pandas)
    p.fisher(r.x).to_dataframe()                        # 95% Wald intervals

or built by hand:

    p = fit.Problem.from_antimony(model_text)           # also Problem(sbml), Problem.from_roadrunner(rr)
    p.add_parameter('k1', scale='log10', lower=1e-3, upper=1e3, nominal=1)
    p.add_condition('high', {'X0': 20})                 # numbers, parameter ids, or None (model value)
    p.add_observable('obs', 'S1', noise_formula='0.1')
    p.add_measurements('high', 'obs', times, values)

Parameter vectors ("theta") hold the free parameters in `free_ids` order, each on its own scale (lin, log or
log10). Objective values are negative log-likelihoods (PEtab's convention, constants included). A failed
simulation raises RuntimeError; bad input raises ValueError.

Stage 1 covers normal and Laplace noise, lin/log/log10 observable transformations and parameter scales, and
condition overrides of parameters, initial values and compartment sizes (fixed values; sizes are not estimated). Placeholders, pre-equilibration, priors and events are
refused with an error until their stages.
"""
from dataclasses import dataclass, field
import os

import numpy as np

from . import core

__all__ = ['Problem', 'FitResult', 'FisherResult']


def _unscale(scale, x):
    x = np.asarray(x, dtype=float)
    return {'lin': x, 'log': np.exp(x), 'log10': 10.0 ** x}[scale]


def _scale(scale, v):
    v = np.asarray(v, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return {'lin': v, 'log': np.log(v), 'log10': np.log10(v)}[scale]


def _pandas():
    try:
        import pandas
    except ImportError as e:
        raise ImportError('to_dataframe() needs pandas (pip install pandas)') from e
    return pandas


class Problem:
    """A fitting problem: an SBML model, a parameter table, conditions, observables and measurements.

    Problem(sbml) takes SBML text or a file path; see also from_petab, from_antimony, from_roadrunner.
    """

    def __init__(self, sbml):
        if isinstance(sbml, core.FitProblem):
            self._p = sbml
            return
        if not sbml.lstrip().startswith('<') and os.path.exists(sbml):
            with open(sbml, encoding='utf-8') as f:
                sbml = f.read()
        self._p = core.FitProblem(sbml)

    @classmethod
    def from_petab(cls, yaml_path):
        """Load a PEtab v1 problem from its YAML file."""
        return cls(core.FitProblem.from_petab(os.fspath(yaml_path)))

    @classmethod
    def from_antimony(cls, text):
        return cls(core.antimony_to_sbml(text))

    @classmethod
    def from_roadrunner(cls, rr):
        """A problem on the model loaded in a RoadRunner instance (its SBML; later changes to rr do not carry over)."""
        return cls(rr.getSBML())

    # ------------------------------------------------------------ building

    def add_parameter(self, id, scale='lin', lower=np.nan, upper=np.nan, nominal=np.nan, estimate=True):
        """A parameter-table row. Bounds and nominal value are linear values. An id that is a model parameter,
        species or compartment overrides the model's value; other ids can appear in observable and noise
        formulas and in conditions."""
        self._p.add_parameter(id, scale, float(lower), float(upper), float(nominal), bool(estimate))
        return self

    def add_condition(self, id, overrides=None):
        """An experimental condition: {target: value} with targets model parameters, species (initial values,
        in the species' units) or compartments (sizes) and values numbers, parameter-table ids, or None (the model's value)."""
        items = list((overrides or {}).items())
        self._p.add_condition(id, [(str(k), None if v is None else (v if isinstance(v, str) else float(v))) for k, v in items])
        return self

    def add_observable(self, id, formula, noise_formula, transformation='lin', noise='normal'):
        """An observable: formula and noise formula over model symbols, parameter-table ids and time.
        transformation: 'lin', 'log', 'log10'; noise: 'normal', 'laplace'."""
        self._p.add_observable(id, formula, str(noise_formula), transformation, noise)
        return self

    def add_measurements(self, condition, observable, times, values):
        t = np.asarray(times, dtype=float).ravel()
        v = np.asarray(values, dtype=float).ravel()
        self._p.add_measurements(condition, observable, t.tolist(), v.tolist())
        return self

    # ------------------------------------------------------------ description

    @property
    def free_ids(self):
        return list(self._p.free_ids)

    @property
    def scales(self):
        """Scale of each free parameter ('lin', 'log', 'log10')."""
        return list(self._p.free_scales)

    @property
    def bounds(self):
        """(lower, upper) of the free parameters, on their scales."""
        return np.asarray(self._p.lower_bounds), np.asarray(self._p.upper_bounds)

    @property
    def nominal(self):
        """Nominal values of the free parameters, on their scales."""
        return np.asarray(self._p.nominal_values)

    @property
    def least_squares(self):
        """True when the objective is a sum of squares plus a constant (normal noise, noise independent of the
        state and of the free parameters): fit() and fisher() need this in stage 1."""
        return self._p.least_squares

    @property
    def sbml(self):
        return self._p.sbml

    @property
    def rtol(self):
        return self._p.rtol

    @rtol.setter
    def rtol(self, v):
        self._p.rtol = v

    @property
    def atol(self):
        return self._p.atol

    @atol.setter
    def atol(self, v):
        self._p.atol = v

    def parameter_table(self):
        """The parameter table (pandas DataFrame; bounds and nominal values linear)."""
        return _pandas().DataFrame(self._p.parameters, columns=['parameterId', 'parameterScale', 'lowerBound',
                                                                'upperBound', 'nominalValue', 'estimate'])

    def measurement_table(self, theta=None):
        """The measurements (pandas DataFrame), with the simulation at theta in a 'simulation' column if given."""
        df = _pandas().DataFrame(self._p.measurements, columns=['simulationConditionId', 'observableId', 'time', 'measurement'])
        if theta is not None:
            df['simulation'] = self.simulate(theta)
        return df

    def to_linear(self, theta):
        """{id: linear value} for theta on the parameter scales."""
        return {i: float(_unscale(s, x)) for i, s, x in zip(self.free_ids, self.scales, np.asarray(theta, dtype=float))}

    def to_scale(self, values):
        """theta (on the parameter scales) from {id: linear value} or linear values in free_ids order."""
        if isinstance(values, dict):
            values = [values[i] for i in self.free_ids]
        return np.array([float(_scale(s, v)) for s, v in zip(self.scales, values)])

    def __repr__(self):
        try:
            return (f'<fit.Problem: {len(self.free_ids)} free parameters, {len(self._p.measurements)} measurements, '
                    f'{len(self._p.condition_ids)} conditions, {len(self._p.observables)} observables>')
        except Exception as e:  # does not compile yet
            return f'<fit.Problem (not compiled: {e})>'

    # ------------------------------------------------------------ objective

    def _theta(self, theta):
        return np.ascontiguousarray(theta, dtype=float).ravel()

    def objective(self, theta):
        """Negative log-likelihood at theta."""
        return self._p.objective(self._theta(theta))

    def gradient(self, theta):
        """(negative log-likelihood, exact gradient) at theta."""
        return self._p.gradient(self._theta(theta))

    def residuals(self, theta):
        """(h(y) - h(measurement)) / sigma per measurement (least-squares problems)."""
        return self._p.residuals(self._theta(theta))

    def residual_jacobian(self, theta):
        """(residuals, Jacobian [measurements x free parameters])."""
        return self._p.residual_jacobian(self._theta(theta))

    def simulate(self, theta):
        """The simulated observable of every measurement, in measurement order."""
        return self._p.simulate_measurements(self._theta(theta))

    # ------------------------------------------------------------ fitting and uncertainty

    def fit(self, n_starts=10, seed=0, threads=0, x0=None, ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=0, callback=None):
        """Multistart trust-region reflective least squares from Latin hypercube starts within the bounds.

        x0: a first start (theta); threads: 0 for all cores; max_nfev: evaluations per start (0: 100 x free
        parameters). callback(finished, total, best) runs after each start; returning True cancels (running
        starts stop after their current iteration and keep their best point). The same seed gives the same
        result, to the last bit, whatever the number of threads.
        """
        x0 = None if x0 is None else [float(v) for v in np.asarray(x0, dtype=float).ravel()]
        d = self._p.fit(int(n_starts), int(seed), int(threads), x0, ftol, xtol, gtol, int(max_nfev), callback)
        return FitResult(problem=self, **d)

    def fisher(self, theta, level=0.95):
        """Fisher-information (Wald) intervals at theta, from the Gauss-Newton Hessian J^T J of the residuals."""
        d = self._p.fisher(self._theta(theta), float(level))
        return FisherResult(scales=self.scales, **d)


@dataclass(repr=False)
class FitResult:
    """Every start of a multistart fit, best (lowest objective) first.

    x, x0: [starts, free parameters] final and starting points on the parameter scales; value: objective per
    start (inf: the start failed); status / message / converged per start; nfev, njev, nit: residual and
    Jacobian evaluations and iterations; start: each row's index in the original sample.
    """
    problem: Problem
    ids: list
    start: list
    x: np.ndarray
    x0: np.ndarray
    value: np.ndarray
    status: list
    message: list
    converged: list
    nfev: list
    njev: list
    nit: list

    @property
    def best(self):
        """theta of the best start."""
        return self.x[0]

    @property
    def best_value(self):
        return float(self.value[0])

    @property
    def best_linear(self):
        """{id: linear value} of the best start."""
        return self.problem.to_linear(self.best)

    def n_at_best(self, tol=1e-3):
        """Starts that ended within tol of the best objective (how often the optimum was found)."""
        return int(np.sum(self.value - self.value[0] <= tol))

    def waterfall(self):
        """Objective per start, sorted (for a waterfall plot)."""
        return np.asarray(self.value)

    def to_dataframe(self):
        """One row per start: start, value, converged, message, nfev, njev, nit, then theta per parameter."""
        pd = _pandas()
        df = pd.DataFrame({'start': self.start, 'value': self.value, 'converged': self.converged, 'message': self.message,
                           'nfev': self.nfev, 'njev': self.njev, 'nit': self.nit})
        for k, i in enumerate(self.ids):
            df[i] = self.x[:, k]
        return df

    def __repr__(self):
        lines = [f'<fit.FitResult: best objective {self.best_value:.10g}, {self.n_at_best()} of {len(self.value)} starts '
                 f'within 1e-3 of it, {sum(self.converged)} converged>']
        for i, v in self.best_linear.items():
            lines.append(f'  {i} = {v:.8g}')
        return '\n'.join(lines)


@dataclass(repr=False)
class FisherResult:
    """Fisher-information (Wald) intervals: x, sd, lower, upper on the parameter scales; identifiable per
    parameter (False: along a direction of zero curvature, sd and bounds infinite); cov and corr matrices."""
    ids: list
    x: np.ndarray
    sd: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    identifiable: list
    cov: np.ndarray
    corr: np.ndarray
    level: float
    scales: list = field(default_factory=list)

    @property
    def lower_linear(self):
        return np.array([float(_unscale(s, v)) for s, v in zip(self.scales, self.lower)])

    @property
    def upper_linear(self):
        return np.array([float(_unscale(s, v)) for s, v in zip(self.scales, self.upper)])

    def to_dataframe(self):
        """One row per parameter: estimate, sd, lower, upper (scale) and estimate, lower, upper (linear)."""
        pd = _pandas()
        return pd.DataFrame({
            'scale': self.scales, 'estimate': self.x, 'sd': self.sd, 'lower': self.lower, 'upper': self.upper,
            'identifiable': self.identifiable,
            'estimate_linear': [float(_unscale(s, v)) for s, v in zip(self.scales, self.x)],
            'lower_linear': self.lower_linear, 'upper_linear': self.upper_linear,
        }, index=pd.Index(self.ids, name='parameterId'))

    def __repr__(self):
        lines = [f'<fit.FisherResult: {self.level:.0%} intervals>']
        for k, i in enumerate(self.ids):
            lo, hi = self.lower_linear[k], self.upper_linear[k]
            tag = '' if self.identifiable[k] else '  (not identifiable)'
            lines.append(f'  {i}: [{lo:.6g}, {hi:.6g}]{tag}')
        return '\n'.join(lines)
