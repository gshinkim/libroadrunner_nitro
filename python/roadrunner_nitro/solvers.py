"""Integrator, steady-state and sensitivity solver objects (r.integrator, r.steadyStateSolver, ...)."""
# Moved from Tellurium Nitro (tellurium_nitro/roadrunner/extended_roadrunner.py, Apache 2.0).
import numpy as np

INTEGRATORS = ['cvode', 'gillespie', 'rk4', 'rk45', 'euler']
STEADY_STATE_SOLVERS = ['nleq1', 'nleq2', 'newton', 'newton_linesearch']


class Solver(object):
    """An integrator or steady-state solver: settings as attributes, getValue/setValue/getSettings.
    Without ``name`` it is the current one; with ``name``, that solver (current or not)."""

    def __init__(self, rr, ss, name=None):
        object.__setattr__(self, '_rr', rr)
        object.__setattr__(self, '_ss', ss)
        object.__setattr__(self, '_name', name)

    def _call(self, f, *args):
        rr, ss = self._rr, self._ss
        cur = rr.getSteadyStateSolverName() if ss else rr.getIntegratorName()
        use = rr.setSteadyStateSolver if ss else rr.setIntegrator
        if self._name is None or self._name == cur:
            return f(*args)
        use(self._name)  # settings of the other solvers are kept: switch, act, switch back
        try:
            return f(*args)
        finally:
            use(cur)

    def getName(self):
        if self._name is not None:
            return self._name
        return self._rr.getSteadyStateSolverName() if self._ss else self._rr.getIntegratorName()

    def getSettings(self):
        return list(self._call(self._rr.getSteadyStateSolverSettings if self._ss else self._rr.getIntegratorSettings))

    def getValue(self, key):
        return self._call(self._rr.getSteadyStateSolverSetting if self._ss else self._rr.getIntegratorSetting, key)

    def setValue(self, key, value):
        if isinstance(value, np.generic):
            value = value.item()
        self._call(self._rr.setSteadyStateSolverSetting if self._ss else self._rr.setIntegratorSetting, key, value)

    def getSettingsMap(self):
        return {k: self.getValue(k) for k in self.getSettings()}

    def resetSettings(self):
        """Every setting of this solver back to its default."""
        from . import core
        fresh = core.RoadRunner()
        name = self.getName()
        (fresh.setSteadyStateSolver if self._ss else fresh.setIntegrator)(name)
        get = fresh.getSteadyStateSolverSetting if self._ss else fresh.getIntegratorSetting
        for k in self.getSettings():
            self.setValue(k, get(k))

    def syncWithModel(self, model=None):
        """Kept for compatibility: the solver always works on its RoadRunner's current model."""

    def getSetting(self, key):
        return self.getValue(key)

    def setSetting(self, key, value):
        self.setValue(key, value)

    def __getattr__(self, key):
        if key.startswith('_'):
            raise AttributeError(key)
        try:
            return self.getValue(key)
        except ValueError:
            raise AttributeError("%s has no setting '%s'" % (self.getName(), key))

    def __setattr__(self, key, value):
        self.setValue(key, value)

    def __dir__(self):
        return sorted(set(dir(type(self))) | set(self.getSettings()))

    def __repr__(self):
        s = self.getSettingsMap()
        w = max([len(k) for k in s] or [0])
        kind = 'SteadyStateSolver' if self._ss else 'Integrator'
        return '< roadrunner.%s() >\n  name: %s\n  settings:\n%s' % (kind, self.getName(), ''.join(
            '  %s: %s\n' % (k.rjust(w + 4), str(v).lower() if isinstance(v, bool) else '%g' % v if isinstance(v, float) else v) for k, v in s.items()))

    __str__ = __repr__


# libroadrunner 2.11's sensitivity solver settings and defaults. The engine reads the tolerances; the
# others are kept and reported (its derivatives are exact, so sensitivity_method and DQ_method do not apply).
_SENS_DEFAULTS = {'relative_tolerance': 1e-06, 'absolute_tolerance': 1e-12, 'stiff': True, 'maximum_bdf_order': 5,
                  'maximum_adams_order': 12, 'maximum_num_steps': 20000, 'maximum_time_step': 0.0,
                  'minimum_time_step': 0.0, 'initial_time_step': 0.0, 'multiple_steps': False,
                  'variable_step_size': False, 'max_output_rows': 100000, 'nonlinear_solver': 'newton',
                  'sensitivity_method': 'simultaneous', 'DQ_method': 'centered'}


class SensitivitySolver(Solver):
    """r.getSensitivitySolver(): the settings timeSeriesSensitivities integrates with ('forward': exact
    symbolic derivatives, one stiff integration per parameter)."""

    def __init__(self):
        object.__setattr__(self, '_s', dict(_SENS_DEFAULTS))

    def getName(self):
        return 'forward'

    def getSettings(self):
        return list(self._s)

    def getValue(self, key):
        if key not in self._s:
            raise ValueError("forward: no setting '%s'" % key)
        return self._s[key]

    def setValue(self, key, value):
        old = self.getValue(key)
        if isinstance(value, np.generic):
            value = value.item()
        self._s[key] = type(old)(value) if isinstance(old, (bool, int, float)) and not isinstance(value, str) else value

    def resetSettings(self):
        self._s.clear()
        self._s.update(_SENS_DEFAULTS)

    def syncWithModel(self, model=None):
        """Kept for compatibility: the settings apply to whatever model the solver is used on."""

    def __repr__(self):
        s = self.getSettingsMap()
        w = max([len(k) for k in s] or [0])
        return '< roadrunner.SensitivitySolver() >\n  name: %s\n  settings:\n%s' % (self.getName(), ''.join(
            '  %s: %s\n' % (k.rjust(w + 4), str(v).lower() if isinstance(v, bool) else v) for k, v in s.items()))

    __str__ = __repr__


class TimeSeriesSensitivitySolver(SensitivitySolver):
    """libroadrunner's base class of time-series sensitivity solvers."""


class SteadyStateSensitivitySolver(SensitivitySolver):
    """libroadrunner's base class of steady-state sensitivity solvers (none is registered)."""


class ForwardSensitivitySolver(TimeSeriesSensitivitySolver):
    """Forward sensitivities on a model, step by step, as libroadrunner's:

        s = ForwardSensitivitySolver(r.getModel(), ['kf'])
        t = s.integrate(0, 1); s.getSensitivityMatrix()   # rows: parameters, columns: variables

    The sensitivities run from the model's state when the solver was made (d variable / d parameter, in
    amounts, exact symbolic derivatives). ``integrate`` also advances the model."""

    def __init__(self, model=None, whichParameters=None):
        SensitivitySolver.__init__(self)
        r = model._r if model is not None else None
        object.__setattr__(self, '_m', model)
        object.__setattr__(self, '_r', r)
        object.__setattr__(self, 'whichParameters', list(whichParameters or []))
        object.__setattr__(self, 'plist', ())
        object.__setattr__(self, '_mat', {})
        if r is not None:
            object.__setattr__(self, '_state0', r.saveStateS())
            object.__setattr__(self, '_t0', r.getCurrentTime())
            object.__setattr__(self, '_t', r.getCurrentTime())

    def __getattr__(self, key):
        if key.startswith('_'):
            raise AttributeError(key)
        return SensitivitySolver.__getattr__(self, key)

    def __setattr__(self, key, value):
        if key in ('whichParameters', 'plist'):
            object.__setattr__(self, key, value)
        else:
            self.setValue(key, value)

    def getModel(self):
        return self._m

    def getGlobalParameterNames(self):
        return tuple(self._m.getGlobalParameterIds())

    def _selected(self):
        names = self.getGlobalParameterNames()
        return [p for p in names if p in self.whichParameters] if self.whichParameters else list(names)

    def getVariableNames(self):
        """The model variables the sensitivities are taken of (rate-rule states, then floating species)."""
        r = self._r
        fids = list(self._m.getFloatingSpeciesIds())
        return tuple([v for v in r.getRateRuleIds() if v not in fids] + fids)

    @property
    def Np(self):
        return len(self.getGlobalParameterNames())

    @property
    def Ns(self):
        return len(self._selected())

    @property
    def numModelVariables(self):
        return len(self.getVariableNames())

    def getModelParametersAsMap(self):
        return dict(zip(self.getGlobalParameterNames(), self.getModelParametersAsVector()))

    def getModelParametersAsVector(self):
        return np.asarray(self._m.getGlobalParameterValues(), dtype=float)

    def deducePlist(self):
        names = self.getGlobalParameterNames()
        object.__setattr__(self, 'plist', tuple(names.index(p) for p in self._selected()))

    def _solve(self, t1, k):
        """Sensitivities from the solver's start to t1 (a copy of the model at the start, so the model
        itself is not moved)."""
        from .roadrunner import RoadRunner
        c = RoadRunner()
        c.loadStateS(self._state0)
        c.getSensitivitySolver()._s.update(self._s)
        if t1 == self._t0:
            return np.zeros((self.Ns, self.numModelVariables))
        _, S, _, _ = c.timeSeriesSensitivities(self._t0, t1, 2, self._selected(), list(self.getVariableNames()), k)
        return S[-1]

    def integrate(self, t, h):
        """Advance the model from ``t`` by ``h``; returns the new time. getSensitivityMatrix() is then at it."""
        # step with this solver's own integration settings (as libroadrunner's CVODES-based solver)
        ig = self._r.getIntegratorByName('cvode')
        keys = [k for k in ('relative_tolerance', 'absolute_tolerance', 'stiff', 'maximum_bdf_order',
                            'maximum_num_steps', 'maximum_time_step', 'minimum_time_step', 'initial_time_step')
                if k in self._s and k in ig.getSettings()]
        old = {k: ig.getValue(k) for k in keys}
        cur = self._r.getIntegrator().getName()
        try:
            for k in keys:
                ig.setValue(k, self._s[k])
            if cur != 'cvode':
                self._r.setIntegrator('cvode')
            t1 = self._r.oneStep(float(t), float(h))
        finally:
            for k, v in old.items():
                ig.setValue(k, v)
            if cur != 'cvode':
                self._r.setIntegrator(cur)
        object.__setattr__(self, '_t', t1)
        self._mat.clear()
        return t1

    def getSensitivityMatrix(self, k=0):
        """d variable / d parameter at the current time: rows the selected parameters, columns
        getVariableNames() (``k``: 0 or 1, as timeSeriesSensitivities)."""
        from .named_array import NamedArray
        if k not in self._mat:
            self._mat[k] = self._solve(self._t, k)
        return NamedArray(self._mat[k])

    def solveSensitivities(self, start, stop, num, params=(), k=0):
        """As r.timeSeriesSensitivities on this solver's model: (time, sensitivities, parameters, variables)."""
        return self._r.timeSeriesSensitivities(start, stop, num, list(params) or self._selected(),
                                               list(self.getVariableNames()), k)


class Integrator(Solver):
    """An integrator. libroadrunner's named classes (CVODEIntegrator, RK4Integrator, ...) are its subclasses
    and can be built on a model, as libroadrunner allows: ``CVODEIntegrator(r.getModel())``."""
    _SS, _NAME = False, None

    def __init__(self, *args):
        if len(args) == 1 and hasattr(args[0], '_r'):  # an ExecutableModel
            Solver.__init__(self, args[0]._r._rr, self._SS, self._NAME)
        else:
            Solver.__init__(self, *args)

    def integrate(self, t0, h):
        """Advance the model from time ``t0`` by ``h`` with this integrator; returns the new time."""
        return self._call(self._rr.oneStep, float(t0), float(h))

    def restart(self, t0):
        self._rr.setValue('time', float(t0))

    def getIntegrationMethod(self):
        return 'Stochastic' if self.getName() == 'gillespie' else 'Deterministic'


class SteadyStateSolver(Integrator):
    """A steady-state solver; the named classes (NLEQ2Solver, NewtonIteration, ...) are its subclasses."""
    _SS = True

    def solve(self):
        """Solve for the steady state of the model; returns the norm of the rates of change there."""
        return self._call(self._rr.steadyState)


def _named(base, cls, name):
    return type(cls, (base,), {'_NAME': name, '__doc__': "libroadrunner's %s ('%s')." % (cls, name)})


CVODEIntegrator = _named(Integrator, 'CVODEIntegrator', 'cvode')
GillespieIntegrator = _named(Integrator, 'GillespieIntegrator', 'gillespie')
RK4Integrator = _named(Integrator, 'RK4Integrator', 'rk4')
RK45Integrator = _named(Integrator, 'RK45Integrator', 'rk45')
EulerIntegrator = _named(Integrator, 'EulerIntegrator', 'euler')
NLEQSolver = _named(SteadyStateSolver, 'NLEQSolver', None)
NLEQ1Solver = _named(NLEQSolver, 'NLEQ1Solver', 'nleq1')
NLEQ2Solver = _named(NLEQSolver, 'NLEQ2Solver', 'nleq2')
KinsolSteadyStateSolver = _named(SteadyStateSolver, 'KinsolSteadyStateSolver', None)
NewtonIteration = _named(KinsolSteadyStateSolver, 'NewtonIteration', None)
BasicNewtonIteration = _named(NewtonIteration, 'BasicNewtonIteration', 'newton')
LinesearchNewtonIteration = _named(NewtonIteration, 'LinesearchNewtonIteration', 'newton_linesearch')

SOLVER_CLASSES = {c._NAME: c for c in (CVODEIntegrator, GillespieIntegrator, RK4Integrator, RK45Integrator,
                                       EulerIntegrator, NLEQ1Solver, NLEQ2Solver, BasicNewtonIteration,
                                       LinesearchNewtonIteration)}


def solver_for(rr, ss, name=None):
    """The named solver object for ``name`` (default: the current integrator or steady-state solver)."""
    cur = name or (rr.getSteadyStateSolverName() if ss else rr.getIntegratorName())
    return SOLVER_CLASSES.get(cur, SteadyStateSolver if ss else Integrator)(rr, ss, name)
