"""RoadRunner: libroadrunner's Python API on the Rust engine (roadrunner_nitro.core)."""
# Moved from Tellurium Nitro (tellurium_nitro/roadrunner/extended_roadrunner.py, Apache 2.0).
import os
import re
import tempfile
from xml.sax.saxutils import quoteattr

import numpy as np

from . import core
from .config import SelectionRecord
from .model import ExecutableModel
from .named_array import NamedArray, _named
from .solvers import INTEGRATORS, STEADY_STATE_SOLVERS, Solver, SensitivitySolver

# Model getters libroadrunner's RoadRunner has itself; the rest live on r.model (Tellurium adds them to r).
_MODEL_FUNCTIONS = ['getReactionRates', 'getStoichiometryIds']

# ---------------------------------------------------------------------
# Model editing: libroadrunner's method -> (core.sbml_edit op, argument defaults after the ids)
# ---------------------------------------------------------------------
_EDITS = {
    'addSpeciesAmount': ('add_species_amount', 2, (0.0, False, False, '')),
    'addSpeciesConcentration': ('add_species_concentration', 2, (0.0, False, False, '')),
    'removeSpecies': ('remove_species', 1, ()),
    'setBoundary': ('set_boundary', 2, ()),
    'setHasOnlySubstanceUnits': ('set_has_only_substance_units', 2, ()),
    'setConstant': ('set_constant', 2, ()),
    'removeReaction': ('remove_reaction', 1, (False,)),
    'setReversible': ('set_reversible', 2, ()),
    'setKineticLaw': ('set_kinetic_law', 2, ()),
    'addParameter': ('add_parameter', 2, ()),
    'removeParameter': ('remove_parameter', 1, ()),
    'addCompartment': ('add_compartment', 2, ()),
    'removeCompartment': ('remove_compartment', 1, ()),
    'addInitialAssignment': ('add_initial_assignment', 2, ()),
    'addAssignmentRule': ('add_assignment_rule', 2, ()),
    'addRateRule': ('add_rate_rule', 2, ()),
    'addEvent': ('add_event', 3, ()),
    'addTrigger': ('add_trigger', 2, ()),
    'setPersistent': ('set_persistent', 2, ()),
    'setTriggerInitialValue': ('set_trigger_initial_value', 2, ()),
    'addPriority': ('add_priority', 2, ()),
    'addDelay': ('add_delay', 2, ()),
    'addEventAssignment': ('add_event_assignment', 3, ()),
    'removeEventAssignments': ('remove_event_assignments', 2, ()),
    'removeEvent': ('remove_event', 1, ()),
}


def _edit_method(name, op, n, defaults):
    def f(self, *args, **kwargs):
        regen = kwargs.pop('forceRegenerate', True)
        args = list(args) + [None] * (n + len(defaults) - len(args))
        if len(args) > n + len(defaults):  # forceRegenerate given positionally
            regen = args.pop()
        for i, d in enumerate(defaults):
            if args[n + i] is None:
                args[n + i] = d
        names = f._kw
        for k, v in kwargs.items():
            args[names.index(k)] = v
        self._edit(op, args, regen)
    f.__name__ = name
    f._kw = {'addSpeciesAmount': ['sid', 'compartment', 'initAmount', 'hasOnlySubstanceUnits', 'boundaryCondition', 'substanceUnits'],
             'addSpeciesConcentration': ['sid', 'compartment', 'initConcentration', 'hasOnlySubstanceUnits',
                                         'boundaryCondition', 'substanceUnits'],
             'removeReaction': ['rid', 'deleteUnusedParameters']}.get(name, [])
    f.__doc__ = "libroadrunner's %s (SBML edited in Rust; forceRegenerate=True recompiles keeping the current values)." % name
    return f


def _set_attr(sbml, tag, id_, attr, value, id_attr='id'):
    """SBML text with attribute ``attr`` of the <tag id_attr="id_"> start tag (id_ None: the first <tag>) set to ``value``."""
    sel = r'[^>]*?' if id_ is None else r'[^>]*?\b%s\s*=\s*"%s"[^>]*?' % (id_attr, re.escape(id_))
    pat = re.compile(r'<((?:\w+:)?%s)\b(%s)(/?)>' % (tag, sel))
    m = pat.search(sbml)
    if m is None:
        raise ValueError("no %s '%s' in the model" % (tag, id_))
    body = re.sub(r'\s%s\s*=\s*"[^"]*"' % attr, '', m.group(2)) + ' %s=%s' % (attr, quoteattr(str(value)))
    return sbml[:m.start()] + '<%s%s%s>' % (m.group(1), body, m.group(3)) + sbml[m.end():]



# ---------------------------------------------------------------------
# RoadRunner
# ---------------------------------------------------------------------
class RoadRunner(object):
    """libroadrunner's RoadRunner API on the Rust engine.

    RoadRunner(sbml=None): SBML text or a file path."""

    def __init__(self, sbml=None, options=None):
        # options (libroadrunner's LoadSBMLOptions): the Rust engine has no load options
        object.__setattr__(self, '_rr', core.RoadRunner())
        object.__setattr__(self, '_data', None)
        object.__setattr__(self, '_last_run', None)
        object.__setattr__(self, '_attrs', None)
        object.__setattr__(self, 'model', ExecutableModel(self))
        object.__setattr__(self, '_sens', SensitivitySolver())
        if sbml is not None:
            self.load(sbml)

    # ------------------------------------------------------------ loading, SBML
    def load(self, sbml):
        """Load an SBML model from text or a file path."""
        if isinstance(sbml, bytes):
            sbml = sbml.decode('utf-8')
        if sbml.startswith('file://'):
            sbml = sbml[len('file://'):]
        self._rr.load(sbml)
        self._changed()

    def _changed(self):  # the model's ids may have changed
        object.__setattr__(self, '_attrs', None)

    def clearModel(self):
        self._rr.clearModel()
        self._changed()

    def isModelLoaded(self):
        return self._rr.isModelLoaded()

    def getModel(self):
        return self.model

    def getSBML(self, level=0, version=0):
        """The loaded SBML, optionally converted to another level and version."""
        return self._rr.getSBML(level, version)

    def getCurrentSBML(self, level=0, version=0):
        """SBML with current values as initial values, optionally converted."""
        return self._rr.getCurrentSBML(level, version)

    def getParamPromotedSBML(self, sbml=None):
        """SBML with local parameters promoted to global ones (of ``sbml``, default the current model)."""
        return core.sbml_edit('get_param_promoted_sbml', self.getSBML() if sbml is None else sbml)

    def validateCurrentSBML(self):
        return core.sbml_edit('validate_current_sbml', self.getSBML())

    def getModelName(self):
        m = re.search(r'<(?:\w+:)?model\b[^>]*?\bname\s*=\s*"([^"]*)"', self._rr.getSBML())
        return m.group(1) if m else self.getModelId()

    def getModelId(self):
        m = re.search(r'<(?:\w+:)?model\b[^>]*?\bid\s*=\s*"([^"]*)"', self._rr.getSBML())
        return m.group(1) if m else ''

    def getInfo(self):
        rr = self._rr
        return '\n'.join(['<roadrunner.RoadRunner() >', 'engine: Rust (roadrunner_nitro)',
                          'model loaded: %s' % rr.isModelLoaded(), 'modelName: %s' % self.getModelName(),
                          'integrator: %s' % rr.getIntegratorName(), 'conservedMoietyAnalysis: %s' % rr.conservedMoietyAnalysis,
                          'floating species: %d' % len(rr.getFloatingSpeciesIds()),
                          'reactions: %d' % len(rr.getReactionIds())])

    def getExtendedVersionInfo(self):
        return 'roadrunner_nitro (Rust engine)'

    # ------------------------------------------------------------ values
    def getValue(self, sel):
        return self._rr.getValue(sel)

    def setValue(self, sel, value):
        self._rr.setValue(sel, float(value))

    def setValues(self, keysOrDict, values=None):
        items = keysOrDict.items() if values is None else zip(keysOrDict, values)
        for k, v in items:
            self.setValue(k, v)

    def __getitem__(self, sel):
        if isinstance(sel, (list, tuple)):
            return np.array(self._rr.getValues(list(sel)))
        return self._rr.getValue(sel)

    def __setitem__(self, sel, value):
        self.setValue(sel, value)

    def keys(self):
        return self.model.keys()

    def values(self):
        return self.model.values()

    def items(self):
        return self.model.items()

    iterkeys, itervalues, iteritems = keys, values, items

    def resetSelectionLists(self):
        """Selections back to the defaults: time, floating species concentrations, rate-rule values."""
        fs = self._rr.getFloatingSpeciesIds()
        self.timeCourseSelections = ['time'] + ['[%s]' % s for s in fs] + [v for v in self._rr.getRateRuleIds() if v not in fs]
        self.steadyStateSelections = ['[%s]' % s for s in fs]

    def getHasOnlySubstanceUnits(self, sid):
        m = re.search(r'<(?:\w+:)?species\b[^>]*?\bid\s*=\s*"%s"[^>]*' % re.escape(sid), self._rr.getSBML())
        if m is None:
            raise ValueError("no species '%s'" % sid)
        return re.search(r'\bhasOnlySubstanceUnits\s*=\s*"(true|1)"', m.group(0)) is not None

    def getSelectedValues(self):
        return np.array(self._rr.getValues(self.timeCourseSelections))

    def _attr_map(self):
        """Model ids as attributes: r.S1 (concentration), r.S1_amt, r.S1_conc, parameters, compartments, rates."""
        a = self._attrs
        if a is None:
            rr, a = self._rr, {}
            for s in rr.getFloatingSpeciesIds():
                a.update({s: '[%s]' % s, s + '_conc': '[%s]' % s, s + '_amt': s})
            a.update({s: '[%s]' % s for s in rr.getBoundarySpeciesIds()})
            a.update({k: k for k in rr.getGlobalParameterIds() + rr.getCompartmentIds() + rr.getReactionIds()})
            object.__setattr__(self, '_attrs', a)
        return a

    def __getattr__(self, name):
        if not name.startswith('_'):
            sel = self._attr_map().get(name)
            if sel is not None:
                return self._rr.getValue(sel)
        raise AttributeError("'RoadRunner' object has no attribute '%s'" % name)

    def __setattr__(self, name, value):
        if not name.startswith('_') and not hasattr(type(self), name):
            sel = self._attr_map().get(name)
            if sel is not None:
                return self.setValue(sel, value)
        object.__setattr__(self, name, value)

    def __dir__(self):
        return sorted(set(dir(type(self))) | set(self._attr_map()))

    def getCurrentTime(self):
        return self._rr.getCurrentTime()

    def _ran(self, method):
        object.__setattr__(self, '_last_run', dict(method=method, engine='rust', reason=None))

    @property
    def lastRun(self):
        """The last simulate/steadyState/MCA-type call: dict(method, engine='rust', reason=None);
        None before any such call. (Kept from the two-engine era: the engine is always Rust.)"""
        r = self._last_run
        return None if r is None else dict(r)

    # ------------------------------------------------------------ selections
    timeCourseSelections = selections = property(
        lambda self: list(self._rr.timeCourseSelections),
        lambda self, s: setattr(self._rr, 'timeCourseSelections', list(s)))
    steadyStateSelections = property(
        lambda self: list(self._rr.steadyStateSelections),
        lambda self, s: setattr(self._rr, 'steadyStateSelections', list(s)))

    def getSteadyStateSelectionStrings(self):
        return self.steadyStateSelections

    @property
    def conservedMoietyAnalysis(self):
        return self._rr.conservedMoietyAnalysis

    @conservedMoietyAnalysis.setter
    def conservedMoietyAnalysis(self, on):
        self._rr.conservedMoietyAnalysis = bool(on)
        self._changed()

    # ------------------------------------------------------------ solvers
    def getIntegrator(self):
        return Solver(self._rr, False)

    def setIntegrator(self, name):
        self._rr.setIntegrator(name if isinstance(name, str) else name.getName())

    integrator = property(getIntegrator, setIntegrator)

    def getIntegratorByName(self, name):
        if name not in INTEGRATORS:
            raise ValueError("no integrator '%s' (one of %s)" % (name, INTEGRATORS))
        return Solver(self._rr, False, name)

    makeIntegrator = getIntegratorByName

    def setIntegratorSetting(self, integratorName, settingName, value):
        self.getIntegratorByName(integratorName).setValue(settingName, value)

    def integratorExists(self, name):
        return name in INTEGRATORS

    def getExistingIntegratorNames(self):
        return list(INTEGRATORS)

    getAvailableIntegrators = getRegisteredIntegratorNames = getExistingIntegratorNames

    def getSteadyStateSolver(self):
        return Solver(self._rr, True)

    def setSteadyStateSolver(self, name):
        self._rr.setSteadyStateSolver(name if isinstance(name, str) else name.getName())

    steadyStateSolver = property(getSteadyStateSolver, setSteadyStateSolver)

    def getSteadyStateSolverByName(self, name):
        if name not in STEADY_STATE_SOLVERS:
            raise ValueError("no steady-state solver '%s' (one of %s)" % (name, STEADY_STATE_SOLVERS))
        return Solver(self._rr, True, name)

    makeSteadyStateSolver = getSteadyStateSolverByName

    def steadyStateSolverExists(self, name):
        return name in STEADY_STATE_SOLVERS

    def getExistingSteadyStateSolverNames(self):
        return list(STEADY_STATE_SOLVERS)

    getRegisteredSteadyStateSolverNames = getExistingSteadyStateSolverNames

    def getSensitivitySolver(self):
        return self._sens

    def setSensitivitySolver(self, name):
        self.getSensitivitySolverByName(name if isinstance(name, str) else name.getName())

    def getSensitivitySolverByName(self, name):
        if name != 'forward':
            raise ValueError("no sensitivity solver '%s' (one of ['forward'])" % name)
        return self._sens

    makeSensitivitySolver = getSensitivitySolverByName

    def sensitivitySolverExists(self, name):
        return name == 'forward'

    def getExistingSensitivitySolverNames(self):
        return ['forward']

    getRegisteredSensitivitySolverNames = getExistingSensitivitySolverNames

    def getDiffStepSize(self):
        return self._rr.diffStepSize

    def setDiffStepSize(self, value):
        self._rr.diffStepSize = float(value)

    diffstep = property(getDiffStepSize, setDiffStepSize)

    def getFluxThreshold(self):
        return self._rr.getFluxThreshold()

    def setFluxThreshold(self, value):
        self._rr.setFluxThreshold(float(value))

    def getSteadyStateThreshold(self):
        return self._rr.getSteadyStateThreshold()

    def setSteadyStateThreshold(self, value):
        self._rr.setSteadyStateThreshold(float(value))

    steadyStateThresh = property(getSteadyStateThreshold, setSteadyStateThreshold)

    def resetSeed(self):
        self._rr.resetSeed()

    def getSeed(self):
        return self._rr.getSeed()

    def setSeed(self, seed, resetModel=True):
        self._rr.setSeed(int(seed))
        if resetModel:
            self.reset()

    # ------------------------------------------------------------ time courses
    def simulate(self, start=None, end=None, points=None, selections=None, output_file=None, steps=None, times=None):
        """Time course from ``start`` to ``end`` at ``points`` points (or ``steps`` + 1, or at ``times``)
        with the current integrator; returns a NamedArray (columns: the selections, which become
        timeCourseSelections). The model is left at the end. ``output_file``: also written as CSV."""
        if isinstance(points, (list, tuple)):  # simulate(start, end, selections)
            points, selections = None, points
        if steps is not None:
            points = int(steps) + 1
        rr = self._rr
        sel = None if selections is None else list(selections)
        # Explicit output points request a grid even when the integrator uses variable steps.
        vss = rr.getIntegratorName() in ('cvode', 'gillespie', 'rk45') and (points is not None or times is not None) and \
            rr.getIntegratorSetting('variable_step_size')
        if vss:
            rr.setIntegratorSetting('variable_step_size', False)
        try:
            if times is not None:
                y = rr.simulateTimes([float(t) for t in times], sel)
            else:
                # as libroadrunner: start 0, end 5, 51 points unless given (each call)
                y = rr.simulate(0.0 if start is None else float(start), 5.0 if end is None else float(end),
                                51 if points is None else int(points), sel)
        finally:
            if vss:
                rr.setIntegratorSetting('variable_step_size', True)
        out = NamedArray(y, colnames=rr.timeCourseSelections)
        object.__setattr__(self, '_data', out)
        self._ran('simulate')
        if output_file:
            np.savetxt(output_file, out, delimiter=',', header=','.join(out.colnames), comments='')
        return out

    def getSimulationData(self):
        return self._data

    def internalOneStep(self, currentTime, stepSize, reset=True):
        self._ran('internalOneStep')
        return self._rr.internalOneStep(currentTime, stepSize, reset)

    def oneStep(self, currentTime, stepSize, reset=True):
        self._ran('oneStep')
        return self._rr.oneStep(float(currentTime), float(stepSize))

    def simulateBatch(self, params, start, end, points, selections=None, threads=None):
        """Many time courses at once on all CPU cores, one per parameter set.

        :param params: dict ``{'k1': [...], 'k2': [...]}`` of equal-length sequences (zipped, not a
            grid), or a pandas DataFrame. Ids can be parameters, compartments, boundary or floating
            species (concentrations), as in ``r[id] = value``.
        :param start, end, points: as in ``simulate``.
        :param selections: output columns (default: timeCourseSelections).
        :param threads: CPU threads (default: all cores).
        :returns: array (n sets, points, len(selections)). Every run starts from the current state;
            the model itself is not changed.
        """
        from ._sbml import from_sbml
        if hasattr(params, 'to_dict'):
            params = params.to_dict('list')
        names = list(params)
        vals = np.array([np.atleast_1d(np.asarray(params[k], dtype=float)) for k in names]).T.reshape(-1, len(names))
        sel = list(self.timeCourseSelections if selections is None else selections)
        times = np.linspace(start, end, points)
        sbml = self.getCurrentSBML()
        info = {}
        m = from_sbml(sbml, info)
        comp_of, amt = info['comp_of'], set(info['amount_params'])
        outs, cols = [], []
        for s in sel:  # selection -> (output column, column of the volume to multiply / divide by)
            br = s[:1] == '[' and s[-1:] == ']'
            v = s[1:-1] if br else s
            if v in comp_of:
                f = 0 if br != (v in amt) else (1 if not br else -1)  # amount wanted from conc: x V
                cols.append((v, comp_of[v] if f else None, f))
                outs += [v] + ([comp_of[v]] if f else [])
            else:
                cols.append((s, None, 0))
                outs.append(s)
        outs = list(dict.fromkeys(outs))
        ok = set(outs) <= set(m.output_ids) and all(k.strip('[]') in set(m.state_ids) | set(m.output_ids) for k in names)
        out = np.full((len(vals), points, len(sel)), np.nan)
        if ok and len(vals):
            ig = self.integrator
            ic = dict(zip(self.getFloatingSpeciesIds(), self.model.getFloatingSpeciesInitConcentrations()))
            asc = [abs(ic[s]) if abs(ic.get(s, 0.0)) > 1e-30 else 1.0 for s in m.state_ids]
            Y = np.asarray(m.simulate_batch([k.strip('[]') for k in names], vals, times, threads=threads,
                                            rtol=float(ig.relative_tolerance), atol=float(ig.absolute_tolerance),
                                            outputs=outs, atol_scale=asc))
            o = {n: i for i, n in enumerate(outs)}
            for j, (c, vol, f) in enumerate(cols):
                y = Y[:, :, o[c]]
                out[:, :, j] = y * Y[:, :, o[vol]] if f == 1 else y / Y[:, :, o[vol]] if f == -1 else y
        # selections the batch engine has no column for, and failed runs: one simulate per set
        for i in (range(len(vals)) if not ok else np.flatnonzero(~np.isfinite(out).all(axis=(1, 2)))):
            out[i] = self._one_run(sbml, names, vals[i], start, end, points, sel)
        self._ran('simulateBatch')
        return out

    def _one_run(self, sbml, names, v, start, end, points, sel):
        """One simulateBatch run on a copy of the current model (this one unchanged)."""
        r = RoadRunner(sbml)
        r.setIntegrator(self.integrator.getName())
        for k, x in self.integrator.getSettingsMap().items():
            r.integrator.setValue(k, x)
        sp = set(r.getFloatingSpeciesIds()) | set(r.getBoundarySpeciesIds())
        for k, x in zip(names, v):
            r.setValue('[%s]' % k if k in sp else k, float(x))  # species ids are concentrations
        return np.asarray(r.simulate(start, end, points, sel))

    def timeSeriesSensitivities(self, start, stop, num, params=(), species=(), k=0):
        """Forward sensitivities d species / d parameter along a time course (exact symbolic derivatives).

        :returns: (times, sensitivities[time, parameter, species] in amounts, parameter ids, species ids);
            params default: the global parameters; species: rate-rule states, then the (independent)
            floating species. The model is left at ``stop``."""
        from ._sbml import from_sbml
        if k not in (0, 1):
            raise ValueError('timeSeriesSensitivities: k must be 0 or 1')
        info = {}
        m = from_sbml(self.getCurrentSBML(), info)
        states, fids = list(m.state_ids), self.getFloatingSpeciesIds()
        cma = self.conservedMoietyAnalysis
        cs = self.getConservedMoietyIds() if cma else []
        rr_ids = [v for v in self.getRateRuleIds() if v in states and v not in fids]
        base = rr_ids + (self.getIndependentFloatingSpeciesIds() if cma else fids)
        params = list(params) or self.getGlobalParameterIds()
        species = [s for s in base if not species or s in species]
        zero = set(rr_ids)
        rp = [p for p in params if p not in cs and p not in zero]
        dep = self.getDependentFloatingSpeciesIds() if any(p in cs for p in params) else []
        times = np.linspace(start, stop, int(num))
        X, S = core.time_sensitivities(m, [], [], times, rp + dep, order=k, rtol=self._sens.relative_tolerance,
                                                   atol=self._sens.absolute_tolerance)
        vol = {c: self.getValue(c) for c in self.getCompartmentIds()}
        V = np.array([vol.get(info['comp_of'].get(s), 1.0) for s in fids])
        rows = []
        if dep:  # moiety total T moves the dependent species: d conc_dep / d T = inv(Gamma[:, dep]) / V_dep
            G = self.getConservationMatrix()
            gcols = list(G.colnames)
            C = np.linalg.inv(np.asarray(G)[:, [gcols.index(d) for d in dep]]) / V[[fids.index(d) for d in dep]][:, None]
        for p in params:
            if p in zero:
                rows.append(np.zeros((len(times), S.shape[2])))
            elif p in cs:
                rows.append(np.einsum('tdx,d->tx', S[:, len(rp):], C[:, cs.index(p)]))
            else:
                rows.append(S[:, rp.index(p)])
        S = np.stack(rows, axis=1) if rows else S[:, :0]
        if k:
            S[0] = 0.0  # as libroadrunner: a zero first row for k = 1 too
        for s, x in zip(states, X[-1]):  # the model is left at the end
            if s in fids or s in rr_ids:
                self.setValue('[%s]' % s if s in fids else s, x)
        self.setValue('time', times[-1])
        idx = [states.index(s) for s in species]
        Vs = np.array([V[fids.index(s)] if s in fids else 1.0 for s in species])
        self._ran('timeSeriesSensitivities')
        return times, S[:, :, idx] * Vs, params, species  # amount sensitivities, as libroadrunner

    # ------------------------------------------------------------ reset, state
    def reset(self, options=None):
        """reset(): floating species, rate-rule values and time to their initial values;
        reset(SelectionRecord flags): also parameters (_GLOBAL_PARAMETER) or everything (ALL)."""
        if options is None:
            return self._rr.reset()
        if options == SelectionRecord.ALL:
            return self._rr.resetAll()
        if options & SelectionRecord._GLOBAL_PARAMETER:
            self._rr.resetParameter()
        if options & (SelectionRecord.TIME | SelectionRecord.RATE | SelectionRecord.FLOATING):
            self._rr.reset()

    def resetAll(self):
        self._rr.resetAll()

    def resetToOrigin(self):
        self._rr.resetToOrigin()
        self._changed()

    def resetParameter(self):
        self._rr.resetParameter()

    def saveState(self, filename, opt='b'):
        """Save the model, its values, time and event state to a file (``opt`` is ignored: one format)."""
        self._rr.saveState(filename)

    def loadState(self, filename):
        self._rr.loadState(filename)
        self._changed()

    def saveStateS(self, opt='b'):
        """saveState to a string."""
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            self._rr.saveState(path)
            with open(path, encoding='utf-8', newline='') as f:
                return f.read()
        finally:
            os.remove(path)

    def loadStateS(self, state):
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            f.write(state)
        try:
            self.loadState(path)
        finally:
            os.remove(path)

    def __getstate__(self):  # pickle / copy.deepcopy: model state and settings
        ig, ss = self.integrator, self.steadyStateSolver
        return dict(state=self.saveStateS() if self.isModelLoaded() else None,
                    integrators={n: Solver(self._rr, False, n).getSettingsMap() for n in INTEGRATORS},
                    integrator=ig.getName(), ss=ss.getName(), ss_settings=ss.getSettingsMap(),
                    sel=self.timeCourseSelections if self.isModelLoaded() else None,
                    ssel=self.steadyStateSelections if self.isModelLoaded() else None,
                    diff=self.getDiffStepSize(), flux=self.getFluxThreshold(), sens=dict(self._sens._s))

    def __setstate__(self, d):
        self.__init__()
        if d['state'] is not None:
            self.loadStateS(d['state'])
            self.timeCourseSelections, self.steadyStateSelections = d['sel'], d['ssel']
        for n, s in d['integrators'].items():
            for k, v in s.items():
                Solver(self._rr, False, n).setValue(k, v)
        self.setIntegrator(d['integrator'])
        self.setSteadyStateSolver(d['ss'])
        for k, v in d['ss_settings'].items():
            self.steadyStateSolver.setValue(k, v)
        self.setDiffStepSize(d['diff'])
        self.setFluxThreshold(d['flux'])
        self._sens._s.update(d['sens'])

    # ------------------------------------------------------------ steady state, MCA
    def steadyState(self, dict=None):
        """Solve for the steady state (``dict``: steady-state solver settings for this call);
        returns the norm of the rates of change there."""
        ss = self.steadyStateSolver
        old = {k: ss.getValue(k) for k in (dict or {})}
        for k, v in (dict or {}).items():
            ss.setValue(k, v)
        try:
            v = self._rr.steadyState()
        finally:
            for k, x in old.items():
                ss.setValue(k, x)
        self._ran('steadyState')
        return v

    def mcaSteadyState(self, *args):
        return self.steadyState()

    def getSteadyStateValues(self):
        v = np.array(self._rr.getSteadyStateValues())
        self._ran('steadyState')
        return v

    def getSteadyStateValuesNamedArray(self):
        return NamedArray(self.getSteadyStateValues()[None], colnames=self.steadyStateSelections)

    def steadyStateNamedArray(self, dict=None):
        self.steadyState(dict)
        return NamedArray(np.array(self._rr.getValues(self.steadyStateSelections))[None], colnames=self.steadyStateSelections)

    def _mat(self, method, *args):
        out = _named(getattr(self._rr, method)(*args))
        self._ran(method)
        return out

    def getFullStoichiometryMatrix(self):
        return self._mat('getFullStoichiometryMatrix')

    def getKMatrix(self):
        return self._mat('getKMatrix')

    def getExtendedStoichiometryMatrix(self):
        return self._mat('getExtendedStoichiometryMatrix')

    def getReactantsStoichiometryMatrix(self, boundary=False):
        return self._mat('getReactantsStoichiometryMatrix', boundary)

    def getProductsStoichiometryMatrix(self, boundary=False):
        return self._mat('getProductsStoichiometryMatrix', boundary)

    def getFrequencyResponse(self, startFrequency, numberOfDecades, numberOfPoints,
                             parameterName, variableName, useDB, useHz):
        self._ran('getFrequencyResponse')
        return self._rr.getFrequencyResponse(startFrequency, numberOfDecades, numberOfPoints,
                                             parameterName, variableName, useDB, useHz)

    def getReducedStoichiometryMatrix(self):
        return self._mat('getNrMatrix')

    getNrMatrix = getReducedStoichiometryMatrix

    def getLinkMatrix(self):
        return self._mat('getLinkMatrix')

    def getL0Matrix(self):
        return self._mat('getL0Matrix')

    def getConservationMatrix(self):
        return self._mat('getConservationMatrix')

    def getFullJacobian(self):
        return self._mat('getFullJacobian')

    def getReducedJacobian(self, h=None):
        return self._mat('getReducedJacobian')

    def getFullEigenValues(self):
        self._ran('getFullEigenValues')
        return np.asarray(self._rr.getFullEigenValues())

    def getReducedEigenValues(self):
        self._ran('getReducedEigenValues')
        return np.asarray(self._rr.getReducedEigenValues())

    def getFullEigenValuesNamedArray(self):
        e = self.getFullEigenValues()
        return NamedArray(np.column_stack([e.real, e.imag]), self._rr.getFullJacobian()[1], ['real', 'imaginary'])

    def getReducedEigenValuesNamedArray(self):
        e = self.getReducedEigenValues()
        return NamedArray(np.column_stack([e.real, e.imag]), self._rr.getReducedJacobian()[1], ['real', 'imaginary'])

    def getUnscaledElasticityMatrix(self):
        return self._mat('getElasticityMatrix', False)

    def getScaledElasticityMatrix(self):
        return self._mat('getElasticityMatrix', True)

    def getUnscaledConcentrationControlCoefficientMatrix(self):
        return self._mat('getConcentrationControlCoefficientMatrix', False)

    def getScaledConcentrationControlCoefficientMatrix(self):
        return self._mat('getConcentrationControlCoefficientMatrix', True)

    def getUnscaledFluxControlCoefficientMatrix(self):
        return self._mat('getFluxControlCoefficientMatrix', False)

    def getScaledFluxControlCoefficientMatrix(self):
        return self._mat('getFluxControlCoefficientMatrix', True)

    def getCC(self, variable, parameter):
        self._ran('getCC')
        return self._rr.getCC(variable, parameter)

    def getuCC(self, variable, parameter):
        self._ran('getuCC')
        return self._rr.getuCC(variable, parameter)

    def getEE(self, reactionName, parameterName, computeSteadyState=True):
        return self._rr.getEE(reactionName, parameterName, bool(computeSteadyState))

    def getuEE(self, reactionName, parameterName, computeSteadyState=True):
        return self._rr.getuEE(reactionName, parameterName, bool(computeSteadyState))

    def getScaledFloatingSpeciesElasticity(self, reactionId, speciesId):
        return self._rr.getEE(reactionId, speciesId, False)

    def getUnscaledSpeciesElasticity(self, reactionIndex, speciesIndex):
        return self._rr.getuEE(self._rr.getReactionIds()[reactionIndex], self._rr.getFloatingSpeciesIds()[speciesIndex], False)

    def getUnscaledParameterElasticity(self, reactionName, parameterName):
        return self._rr.getuEE(reactionName, parameterName, False)

    # ------------------------------------------------------------ ids and value vectors
    def getIds(self, types=SelectionRecord.ALL):
        return self._rr.getIds(types)

    def getSupportedIdTypes(self):
        return self._rr.getSupportedIdTypes()

    def getEventRoots(self, time=None, y=None, gdot=None):
        return self.model.getEventRoots(time, y, gdot)

    def getPiecewiseTriggerRoots(self, time=None, y=None, gdot=None):
        return self.model.getPiecewiseTriggerRoots(time, y, gdot)

    def getStoichiometry(self, *args):
        return self.model.getStoichiometry(*args)

    def getInitStoichiometry(self, *args):
        return self.model.getInitStoichiometry(*args)

    def getStoichiometryId(self, index):
        return self.model.getStoichiometryId(index)

    def getStoichiometryIndex(self, *args):
        return self.model.getStoichiometryIndex(*args)

    def setStoichiometry(self, *args):
        result = self.model.setStoichiometry(*args)
        self._changed()
        return result

    def getIndependentFloatingSpeciesIds(self):
        return list(self._rr.getIndependentFloatingSpeciesIds())

    def getDependentFloatingSpeciesIds(self):
        return list(self._rr.getDependentFloatingSpeciesIds())

    def getNumIndFloatingSpecies(self):
        return len(self._rr.getIndependentFloatingSpeciesIds())

    def getNumDepFloatingSpecies(self):
        return len(self._rr.getDependentFloatingSpeciesIds())

    getNumberOfIndependentSpecies, getNumberOfDependentSpecies = getNumIndFloatingSpecies, getNumDepFloatingSpecies

    def getConservedMoietyIds(self):
        return list(self._rr.getConservedMoietyIds())

    def getConservedMoietyValues(self):
        return self.model.getConservedMoietyValues()

    def getEigenValueIds(self):
        return list(self._rr.getEigenValueIds())

    def getRatesOfChangeIds(self):
        return list(self._rr.getRatesOfChangeIds())

    def getRateRuleIds(self):
        return list(self._rr.getRateRuleIds())

    def getAssignmentRuleIds(self):
        return self.model.getAssignmentRuleIds()

    def getInitialAssignmentIds(self):
        return self.model.getInitialAssignmentIds()

    def getFloatingSpeciesInitialConcentrationIds(self):
        return self.model.getFloatingSpeciesInitConcentrationIds()

    def getAvailableTimeCourseSymbols(self):
        return self._rr.getAvailableTimeCourseSymbols()

    def getRatesOfChange(self):
        return np.array(self._rr.getRatesOfChange())

    def getRatesOfChangeNamedArray(self):
        return NamedArray(self.getRatesOfChange()[None], colnames=self.getRatesOfChangeIds())

    def getKineticLaw(self, rid):
        return core.sbml_edit('get_kinetic_law', self.getSBML(), rid)

    # ------------------------------------------------------------ model editing
    def _edit(self, op, args, regen=True):
        self._set_sbml(core.sbml_edit(op, self._rr.getSBML(), *args), regen)

    def _set_sbml(self, sbml, regen=True):
        self._rr.setSBML(sbml, bool(regen))
        self._changed()

    def regenerateModel(self, forceRegenerate=True, reset=False):
        """Recompile the (edited) model, keeping the current values (``reset``: then resetAll)."""
        self._rr.regenerate()
        self._changed()
        if reset:
            self.resetAll()

    def addSpecies(self, sid, compartment, initAmount=0.0, initConcentration=0.0, hasOnlySubstanceUnits=False,
                   boundaryCondition=False, substanceUnits='', forceRegenerate=True):
        if initConcentration and not initAmount:
            self.addSpeciesConcentration(sid, compartment, initConcentration, hasOnlySubstanceUnits, boundaryCondition,
                                         substanceUnits, forceRegenerate)
        else:
            self.addSpeciesAmount(sid, compartment, initAmount, hasOnlySubstanceUnits, boundaryCondition,
                                  substanceUnits, forceRegenerate)

    def addReaction(self, *args, **kwargs):
        """addReaction(sbmlRep[, forceRegenerate]) or addReaction(rid, reactants, products, kineticLaw[, forceRegenerate])."""
        regen = kwargs.pop('forceRegenerate', True)
        if len(args) in (1, 2) and not kwargs:
            if len(args) == 2:
                regen = args[1]
            return self._edit('add_reaction_from_sbml', [args[0]], regen)
        a = dict(zip(['rid', 'reactants', 'products', 'kineticLaw', 'forceRegenerate'], args))
        a.update(kwargs)
        self._edit('add_reaction', [a['rid'], list(a['reactants']), list(a['products']), a['kineticLaw']],
                   a.get('forceRegenerate', regen))

    def removeRules(self, vid, useInitialValue=False, forceRegenerate=True):
        """Remove the rules of ``vid`` (``useInitialValue``: its value becomes the rule's value at time 0)."""
        v = self.getValue('init(%s)' % vid) if useInitialValue else None
        self._edit('remove_rules', [vid], forceRegenerate and v is None)
        if v is not None:
            self.setInitValue(vid, v, forceRegenerate)

    def setInitAmount(self, sid, initAmount, forceRegenerate=True):
        self._edit('set_init_amount', [sid, float(initAmount)], forceRegenerate)

    def setInitConcentration(self, sid, initConcentration, forceRegenerate=True):
        self._edit('set_init_concentration', [sid, float(initConcentration)], forceRegenerate)

    def setInitValue(self, sid, initValue, forceRegenerate=True):
        """Initial value of any species (amount), compartment or parameter."""
        sbml = self._rr.getSBML()
        if sid in self.getFloatingSpeciesIds() + self.getBoundarySpeciesIds():
            return self.setInitAmount(sid, initValue, forceRegenerate)
        tag, attr = ('compartment', 'size') if sid in self.getCompartmentIds() else ('parameter', 'value')
        self._set_sbml(_set_attr(sbml, tag, sid, attr, repr(float(initValue))), forceRegenerate)

    def removeInitialAssignment(self, vid, forceRegenerate=True, errIfNotExist=True, replaceInitVal=True):
        """Remove the initial assignment of ``vid`` (``replaceInitVal``: its value becomes the initial value).
        Returns whether the model was regenerated."""
        sbml = self._rr.getSBML()
        pat = re.compile(r'<(?:\w+:)?initialAssignment\b[^>]*?\bsymbol\s*=\s*"%s".*?</(?:\w+:)?initialAssignment>\s*'
                         % re.escape(vid), re.S)
        if not pat.search(sbml):
            if errIfNotExist:
                raise ValueError("no initial assignment for '%s'" % vid)
            return False
        v = self.getValue('init(%s)' % vid) if replaceInitVal else None
        self._set_sbml(pat.sub('', sbml, count=1), forceRegenerate and v is None)
        if v is not None:
            self.setInitValue(vid, v, forceRegenerate)
        return bool(forceRegenerate)

    def setInitStoichiometryValue(self, speciesId, reactionId, initValue):
        """Stoichiometry of the (unnamed) species reference of ``speciesId`` in reaction ``reactionId``."""
        sbml = self._rr.getSBML()
        m = re.search(r'<(?:\w+:)?reaction\b[^>]*?\bid\s*=\s*"%s".*?</(?:\w+:)?reaction>' % re.escape(reactionId), sbml, re.S)
        if m is None:
            raise ValueError("no reaction '%s'" % reactionId)
        block = _set_attr(m.group(0), 'speciesReference', speciesId, 'stoichiometry', repr(float(initValue)), 'species')
        self._set_sbml(sbml[:m.start()] + block + sbml[m.end():])

    def setModelId(self, id):
        self._set_sbml(_set_attr(self._rr.getSBML(), 'model', None, 'id', id), False)

    def setModelName(self, name):
        self._set_sbml(_set_attr(self._rr.getSBML(), 'model', None, 'name', name), False)

    def setGlobalParameterByName(self, param, value):
        self.setValue(param, value)

    def getGlobalParameterByName(self, param):
        return self.getValue(param)


    # ------------------------------------------------------------ plotting (libroadrunner's basic plot)
    def plot(self, result=None, loc='upper left', show=True):
        """Plot a time course with matplotlib: ``result`` (default: the last simulate), first column on x,
        one line per other column, labelled by its column name. ``show``: call matplotlib's show()."""
        p = _pyplot()
        result = self.getSimulationData() if result is None else result
        if result is None:
            raise ValueError("no simulation result: call simulate() first")
        names = list(getattr(result, 'colnames', None) or self.timeCourseSelections)
        a = np.asarray(result)
        if a.ndim != 2 or a.shape[1] != len(names):
            raise ValueError("simulation result columns do not match the selections")
        lines = [p.plot(a[:, 0], a[:, i], label=str(names[i]), linewidth=2) for i in range(1, a.shape[1])]
        if show:
            p.show()
        return lines

    def showPlot(self):
        _pyplot().show()

    def plotLegend(self):
        _pyplot().legend()


def _pyplot():
    try:
        import matplotlib.pyplot as p
    except ImportError:
        raise ImportError("plotting needs matplotlib: pip install matplotlib")
    return p


for _n, (_op, _k, _d) in _EDITS.items():
    setattr(RoadRunner, _n, _edit_method(_n, _op, _k, _d))



def _model_function(key):
    def f(self):
        return getattr(self.model, key)()
    f.__name__ = key
    return f


for _n in _MODEL_FUNCTIONS:
    setattr(RoadRunner, _n, _model_function(_n))


def _vector(ids, fmt, named):  # r.get<Independent|Dependent|>FloatingSpecies<Amounts|Concentrations><V|NamedArray>()
    def f(self):
        names = getattr(self._rr, ids)()
        v = np.array(self._rr.getValues([fmt % s for s in names]))
        return NamedArray(v[None], colnames=names) if named else v
    return f


for _g, _ids in (('', 'getFloatingSpeciesIds'), ('Independent', 'getIndependentFloatingSpeciesIds'),
                 ('Dependent', 'getDependentFloatingSpeciesIds'), ('Boundary', 'getBoundarySpeciesIds')):
    _sp = 'BoundarySpecies' if _g == 'Boundary' else _g + 'FloatingSpecies'
    for _q, _fmt in (('Amounts', '%s'), ('Concentrations', '[%s]')):
        for _sfx, _named_ in (('V', False), ('NamedArray', True)):
            if _g in ('', 'Boundary') and _sfx == 'V':
                continue
            _f = _vector(_ids, _fmt, _named_)
            _f.__name__ = 'get%s%s%s' % (_sp, _q, _sfx)
            setattr(RoadRunner, _f.__name__, _f)
    if _g in ('Independent', 'Dependent'):
        _f = _vector(_ids, "%s'", False)
        _f.__name__ = 'get%sRatesOfChange' % _g
        setattr(RoadRunner, _f.__name__, _f)
        _f = _vector(_ids, "%s'", True)
        _f.__name__ = 'get%sRatesOfChangeNamedArray' % _g
        setattr(RoadRunner, _f.__name__, _f)

