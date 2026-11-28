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

__all__ = ['fit_timecourse', 'TimecourseFit', 'Problem', 'FitResult', 'FisherResult']


def _unscale(scale, x):
    x = np.asarray(x, dtype=float)
    return {'lin': x, 'log': np.exp(x), 'log10': 10.0 ** x}[scale]


def _scale(scale, v):
    v = np.asarray(v, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return {'lin': v, 'log': np.log(v), 'log10': np.log10(v)}[scale]


def _auto_scale(lower, upper):
    """log10 for a positive range spanning a factor of 100 or more, lin otherwise."""
    lo, hi = float(lower), float(upper)
    return 'log10' if lo > 0 and np.isfinite(hi) and hi / lo >= 100 else 'lin'


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

    def add_parameter(self, id, scale=None, lower=np.nan, upper=np.nan, nominal=np.nan, estimate=True):
        """A parameter-table row. Bounds and nominal value are linear values. An id that is a model parameter,
        species or compartment overrides the model's value; other ids can appear in observable and noise
        formulas and in conditions.

        scale: 'lin', 'log' or 'log10', the scale the optimizer searches on. None picks log10 for a positive range
        spanning a factor of 100 or more (rate constants, usually), so the search covers every order of magnitude
        evenly; lin otherwise."""
        if scale is None:
            scale = _auto_scale(lower, upper)
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

    def add_measurements(self, condition, observable, times, values, preequilibration=None):
        """Measurements of an observable in a condition. A time of inf is a measurement at steady state (reached from
        the last finite time). preequilibration: a condition whose steady state the simulation starts from (states
        the simulation condition sets excepted)."""
        t = np.asarray(times, dtype=float).ravel()
        v = np.asarray(values, dtype=float).ravel()
        self._p.add_measurements(condition, observable, t.tolist(), v.tolist(), preequilibration)
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
        pre = self._p.preequilibration_conditions
        if any(pre):
            df.insert(0, 'preequilibrationConditionId', pre)
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

    def fit(self, n_starts=10, seed=0, threads=0, x0=None, ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=0, callback=None,
            global_method=None, population=0, max_evals=0, polish=True):
        """Multistart local fitting from Latin hypercube starts within the bounds (trust-region least squares, or
        L-BFGS-B for problems that are not least squares).

        global_method: 'pso' (particle swarm, SPSO-2011), 'de' (differential evolution, DE/rand/1/bin) or 'shade'
        (adaptive differential evolution) searches the whole box first, then its n_starts best distinct points
        are polished by the local method (polish=False: returned as the search left them). population: members
        (0: the method's default); max_evals: objective evaluations of the search (0: 1000 per parameter, at
        least 10 000).

        x0: a first start (theta); threads: 0 for all cores; max_nfev: evaluations per local start (0: 100 x free
        parameters). callback(finished, total, best) runs after each start, and during a search after each
        generation (evaluations, budget, best); returning True cancels (running starts stop after their current
        iteration and keep their best point). The same seed gives the same result, to the last bit, whatever the
        number of threads.
        """
        x0 = None if x0 is None else [float(v) for v in np.asarray(x0, dtype=float).ravel()]
        d = self._p.fit(int(n_starts), int(seed), int(threads), x0, ftol, xtol, gtol, int(max_nfev), callback,
                        global_method, int(population), int(max_evals), bool(polish))
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
    Jacobian evaluations and iterations; start: each row's index in the original sample. global_stage: with a
    global method, a dict of method, best (its best objective), nfev, generations and message.
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
    global_stage: dict = None

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


# ---------------------------------------------------------------- the short form: one time course, a few parameters

def _model_sbml(model):
    """SBML text from a RoadRunner instance, SBML text, a file path, or Antimony text."""
    if hasattr(model, 'getSBML'):
        return model.getSBML()
    s = os.fspath(model)
    if not s.lstrip().startswith('<') and len(s) < 4096 and os.path.exists(s):
        with open(s, encoding='utf-8') as f:
            s = f.read()
    return s if s.lstrip().startswith('<') else core.antimony_to_sbml(s)


def _data_columns(data):
    """(times, {column name: values}) from a dict, a pandas DataFrame, a roadrunner simulation result (NamedArray)
    or a numpy structured array; the time column is called 'time'."""
    if hasattr(data, 'colnames') and np.ndim(data) == 2 and len(data.colnames) == np.shape(data)[1]:
        a = np.asarray(data, dtype=float)
        cols = {n: a[:, i] for i, n in enumerate(data.colnames)}
    elif hasattr(data, 'columns') and hasattr(data, 'to_numpy'):      # pandas
        cols = {str(c): data[c].to_numpy(dtype=float) for c in data.columns}
    elif isinstance(data, dict):
        cols = {str(k): np.asarray(v, dtype=float).ravel() for k, v in data.items()}
    elif getattr(getattr(data, 'dtype', None), 'names', None):
        cols = {n: np.asarray(data[n], dtype=float) for n in data.dtype.names}
    else:
        raise TypeError('data: a dict {"time": [...], "S1": [...]}, a pandas DataFrame, or a simulation result')
    tkey = next((k for k in cols if k.lower() in ('time', '[time]', 't')), None)
    if tkey is None:
        raise ValueError(f'data: no time column (columns: {", ".join(cols)})')
    times = cols.pop(tkey)
    if not cols:
        raise ValueError('data: no measured columns besides time')
    for k, v in cols.items():
        if len(v) != len(times):
            raise ValueError(f'data: column {k!r} has {len(v)} values, time has {len(times)}')
    return times, cols


def _symbol(column):
    """Model symbol of a data column: '[S1]' (roadrunner's concentration selection) is S1."""
    c = column.strip()
    return c[1:-1] if c.startswith('[') and c.endswith(']') else c


def fit_timecourse(model, data, params, sigma=None, n_starts=10, seed=0, level=0.95, global_method=None):
    """Fit model parameters to one time course: the short form of Problem for the common case.

        r = fit.fit_timecourse(model, {'time': t, 'S1': s1}, ['k1', 'k2'])
        print(r)            # values with confidence intervals
        r.plot()            # data and fitted curves

    model:  Antimony text, SBML text, a model file, or a RoadRunner instance.
    data:   a dict, pandas DataFrame or roadrunner simulation result with a 'time' column and one column per
            measured quantity, named by its model symbol (a species, '[S1]' for its concentration, a parameter, an
            assignment-rule variable). Missing values (NaN) are skipped. The simulation starts at time 0 from the
            model's initial state.
    params: the parameters to estimate: a list of ids, searched from 1/1000 to 1000 times their value in the
            model (a log scale), or a dict {id: (lower, upper)} or {id: (lower, upper, initial)}. The model's
            values are also used as the first starting point.
    sigma:  the measurement error (standard deviation): one number, a dict per column, or None (the default) to
            estimate one common error from the scatter of the data around the fit, as lmfit and scipy's
            curve_fit do. The confidence intervals use it.
    n_starts, seed: starting points of the multistart fit and the random seed choosing them; with a global method,
            the number of its best points refined by the local fit.
    level:  confidence level of the intervals.
    global_method: None (multistart), or 'pso' (particle swarm), 'de' (differential evolution) or 'shade'
            (adaptive differential evolution) to search the whole range first, then refine its best points.
    """
    from .roadrunner import RoadRunner
    sbml = _model_sbml(model)
    times, cols = _data_columns(data)
    if isinstance(params, str):
        params = [params]
    if not isinstance(params, dict):
        params = {k: None for k in params}
    rr = RoadRunner(sbml)
    p = Problem(sbml)
    x0 = []
    for pid, spec in params.items():
        try:
            v = float(rr[pid])
        except Exception:
            raise ValueError(f'parameter {pid!r} is not in the model') from None
        if spec is None:
            if not (v > 0 and np.isfinite(v)):
                raise ValueError(f'{pid!r} is {v} in the model, so no default range can be taken from it: give one, '
                                 f'params={{{pid!r}: (lower, upper)}}')
            lo, hi, init = v / 1000, v * 1000, v
        else:
            lo, hi = float(spec[0]), float(spec[1])
            init = float(spec[2]) if len(spec) > 2 else (v if lo <= v <= hi else np.sqrt(lo * hi) if lo > 0 else 0.5 * (lo + hi))
        p.add_parameter(pid, None, lo, hi, init)
        x0.append(init)
    p.add_condition('data')
    for name, values in cols.items():
        sym = _symbol(name)
        s = sigma.get(name, sigma.get(sym)) if isinstance(sigma, dict) else sigma
        p.add_observable(name, sym, '1' if s is None else repr(float(s)))
        ok = ~np.isnan(values)
        p.add_measurements('data', name, times[ok], values[ok])
    estimate = sigma is None or (isinstance(sigma, dict) and any(sigma.get(n, sigma.get(_symbol(n))) is None for n in cols))
    if estimate and isinstance(sigma, dict):
        raise ValueError('sigma: give every column an error, or none (to estimate one common error)')
    start = p.to_scale(x0)
    lower, upper = p.bounds
    start = np.clip(start, lower, upper)
    result = p.fit(n_starts=n_starts, seed=seed, x0=start, global_method=global_method)
    best = result.best
    res = p.residuals(best)
    n, k = len(res), len(best)
    ssr = float(np.sum(res ** 2))
    fi = p.fisher(best, level)
    if estimate:
        s_hat = np.sqrt(ssr / (n - k)) if n > k else np.nan
        # the residuals were computed with sigma = 1: scale the covariance by the estimated error
        z = (fi.upper - fi.x) / np.where(fi.sd > 0, fi.sd, 1)
        sd = fi.sd * s_hat
        fi = FisherResult(ids=fi.ids, x=fi.x, sd=sd, lower=fi.x - z * sd, upper=fi.x + z * sd, identifiable=fi.identifiable,
                          cov=fi.cov * s_hat ** 2, corr=fi.corr, level=fi.level, scales=fi.scales)
        sig, ssr_out = float(s_hat), ssr * s_hat ** 2
    else:
        sig, ssr_out = sigma, None
    values = p.to_linear(best)
    intervals = {i: (float(lo), float(hi)) for i, lo, hi in zip(fi.ids, fi.lower_linear, fi.upper_linear)}
    return TimecourseFit(values=values, intervals=intervals, identifiable=dict(zip(fi.ids, fi.identifiable)),
                         sigma=sig, sigma_estimated=estimate, n_data=n, level=level, problem=p, result=result, fisher=fi,
                         sbml=sbml, times=times, data=cols)


@dataclass(repr=False)
class TimecourseFit:
    """The result of fit_timecourse.

    values: {id: fitted value}; intervals: {id: (lower, upper)} at `level`; identifiable: {id: bool};
    sigma: the measurement error used (estimated from the residuals when sigma_estimated); problem, result,
    fisher: the full Problem, FitResult and FisherResult behind it.
    """
    values: dict
    intervals: dict
    identifiable: dict
    sigma: object
    sigma_estimated: bool
    n_data: int
    level: float
    problem: Problem
    result: FitResult
    fisher: FisherResult
    sbml: str
    times: np.ndarray
    data: dict

    def roadrunner(self):
        """A RoadRunner instance of the model with the fitted values set."""
        from .roadrunner import RoadRunner
        rr = RoadRunner(self.sbml)
        floating = set(rr.model.getFloatingSpeciesIds())
        for k, v in self.values.items():
            if k in floating:
                rr.setValue(f'init([{k}])', v)
            else:
                rr[k] = v
        rr.reset()
        return rr

    def simulate(self, end=None, points=201):
        """The fitted model simulated from 0 to `end` (default: the last data time): columns time and every
        measured quantity."""
        end = float(np.max(self.times)) if end is None else end
        rr = self.roadrunner()
        floating = set(rr.model.getFloatingSpeciesIds())
        sels = ['time'] + [f'[{s}]' if s in floating else s for s in (_symbol(c) for c in self.data)]
        return rr.simulate(0, end, points, sels)

    def plot(self, ax=None):
        """Data (points) and the fitted model (lines), one colour per measured quantity. Needs matplotlib."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        sim = np.asarray(self.simulate())
        for q, (name, vals) in enumerate(self.data.items()):
            c = f'C{q % 10}'
            ax.plot(self.times, vals, 'o', color=c, label=f'{name} data')
            ax.plot(sim[:, 0], sim[:, q + 1], '-', color=c, label=f'{name} fit')
        ax.set_xlabel('time')
        ax.legend()
        return ax

    def __repr__(self):
        agree = self.result.n_at_best()
        lines = [f'Fit of {len(self.values)} parameter{"s" * (len(self.values) != 1)} to {self.n_data} data points '
                 f'({agree} of {len(self.result.value)} starts reached the best fit)']
        w = max(len(k) for k in self.values)
        for k, v in self.values.items():
            lo, hi = self.intervals[k]
            ci = (f'{self.level:.0%} interval {lo:.4g} to {hi:.4g}' if self.identifiable[k]
                  else 'not identifiable from these data')
            lines.append(f'  {k:{w}s} = {v:<10.5g} ({ci})')
        if self.sigma_estimated:
            lines.append(f'  measurement error estimated from the scatter of the data: {self.sigma:.4g}')
        return '\n'.join(lines)
