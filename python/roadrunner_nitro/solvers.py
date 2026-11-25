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


class SensitivitySolver(Solver):
    """r.getSensitivitySolver(): the tolerances timeSeriesSensitivities integrates with ('forward': exact
    symbolic derivatives, one stiff integration per parameter)."""

    def __init__(self):
        object.__setattr__(self, '_s', {'relative_tolerance': 1e-6, 'absolute_tolerance': 1e-12})

    def getName(self):
        return 'forward'

    def getSettings(self):
        return list(self._s)

    def getValue(self, key):
        if key not in self._s:
            raise ValueError("forward: no setting '%s'" % key)
        return self._s[key]

    def setValue(self, key, value):
        self.getValue(key)
        self._s[key] = float(value)


# libroadrunner's module exposes these solver base classes by name.
Integrator = SteadyStateSolver = Solver
