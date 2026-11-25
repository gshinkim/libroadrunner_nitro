"""r.model: libroadrunner's ExecutableModel API."""
# Moved from Tellurium Nitro (tellurium_nitro/roadrunner/extended_roadrunner.py, Apache 2.0).
import re

import numpy as np

from .config import SelectionRecord


def _sbml_ids(sbml, tag, attr):
    """Ids from the SBML text: <tag ... attr="x"> (rule variables, initial assignment symbols, event ids)."""
    return re.findall(r'<(?:\w+:)?%s\b[^>]*?\b%s\s*=\s*"([^"]*)"' % (tag, attr), sbml)


class ExecutableModel(object):
    """``r.model``: ids and values of the loaded model, as libroadrunner's ExecutableModel."""

    def __init__(self, r):
        self._r = r

    def _get(self, ids, fmt, idx=None):
        ids = ids if idx is None else [ids[i] for i in np.atleast_1d(idx)]
        return np.array(self._r._rr.getValues([fmt % i for i in ids]), dtype=float)

    def _set(self, ids, fmt, args):
        idx, vals = (range(len(ids)), args[0]) if len(args) == 1 else args
        for i, v in zip(np.atleast_1d(idx), np.atleast_1d(vals)):
            self._r._rr.setValue(fmt % ids[i], float(v))

    def __getitem__(self, key):
        return self._r.getValue(key)

    def __setitem__(self, key, value):
        self._r.setValue(key, value)

    def keys(self):
        return [s for _, l in self._r._rr.getAvailableTimeCourseSymbols() for s in l]

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return list(zip(self.keys(), self.values()))

    def getTime(self):
        return self._r._rr.getCurrentTime()

    def setTime(self, t):
        self._r._rr.setValue('time', float(t))

    def reset(self):
        self._r.reset()

    def getModelName(self):
        return self._r.getModelName()

    def getInfo(self):
        return self._r.getInfo()

    def getNumEvents(self):
        return self._r._rr.getNumEvents()

    def getEventIds(self):
        return self._r._rr.getEventIds()

    def getIds(self, types=SelectionRecord.ALL):
        return self._r._rr.getIds(types, False)

    def getSupportedIdTypes(self):
        return self._r._rr.getSupportedIdTypes(False)

    def getNumPiecewiseTriggers(self):
        return self._r._rr.getNumPiecewiseTriggers()

    def getEventRoots(self, time=None, y=None, gdot=None):
        return self._roots(self._r._rr.getEventRoots, time, y, gdot)

    def getPiecewiseTriggerRoots(self, time=None, y=None, gdot=None):
        return self._roots(self._r._rr.getPiecewiseTriggerRoots, time, y, gdot)

    @staticmethod
    def _roots(get, time, y, gdot):
        roots = get(time, y)
        if gdot is None:
            return np.asarray(roots, dtype=float)
        if len(gdot) != len(roots):
            raise ValueError("expected %d root outputs" % len(roots))
        gdot[:] = roots

    def getEventIndex(self, eid):
        return next((i for i, name in enumerate(self.getEventIds()) if name == eid), -1)

    def getRandom(self):
        return self._r._rr.getRandom()

    def getAssignmentRuleIds(self):
        return _sbml_ids(self._r._rr.getSBML(), 'assignmentRule', 'variable')

    def getInitialAssignmentIds(self):
        return _sbml_ids(self._r._rr.getSBML(), 'initialAssignment', 'symbol')

    def getRateRuleIds(self):
        return list(self._r._rr.getRateRuleIds())

    def getNumRateRules(self):
        return len(self.getRateRuleIds())

    def getRateRuleRates(self):
        return self._get(self.getRateRuleIds(), "%s'")

    def getConservedMoietyIds(self):
        return list(self._r._rr.getConservedMoietyIds())

    def getNumConservedMoieties(self):
        return len(self.getConservedMoietyIds())

    def getConservedMoietyValues(self, idx=None):
        return self._get(self.getConservedMoietyIds(), '%s', idx)

    def setConservedMoietyValues(self, *args):
        self._set(self.getConservedMoietyIds(), '%s', args)

    def getFloatingSpeciesConcentrationIds(self):
        return ['[%s]' % s for s in self._r._rr.getFloatingSpeciesIds()]

    def getBoundarySpeciesConcentrationIds(self):
        return ['[%s]' % s for s in self._r._rr.getBoundarySpeciesIds()]

    def getFloatingSpeciesInitConcentrationIds(self):
        return ['init([%s])' % s for s in self._r._rr.getFloatingSpeciesIds()]

    def getFloatingSpeciesInitAmountIds(self):
        return ['init(%s)' % s for s in self._r._rr.getFloatingSpeciesIds()]

    def getConservedMoietyId(self, i):
        return self.getConservedMoietyIds()[i]

    def getEventId(self, i):
        return self.getEventIds()[i]

    def getFloatingSpeciesAmountRate(self, i, reactionRates=None):
        """Amount rate of floating species i from the reactions: N[i] . reactionRates (default: the current rates)."""
        n, rows, cols = self._r._rr.getFullStoichiometryMatrix()
        v = self.getReactionRates() if reactionRates is None else np.asarray(reactionRates)
        ids = self._r._rr.getReactionIds()
        return float(np.asarray(n)[rows.index(self._r._rr.getFloatingSpeciesIds()[i]), [cols.index(r) for r in ids]] @ v)

    def _compartment_index(self, sid):
        m = re.search(r'<(?:\w+:)?species\b[^>]*?\bid\s*=\s*"%s"[^>]*' % re.escape(sid), self._r._rr.getSBML())
        c = re.search(r'\bcompartment\s*=\s*"([^"]*)"', m.group(0)).group(1)
        return self._r._rr.getCompartmentIds().index(c)

    def getCompartmentIndexForFloatingSpecies(self, i):
        return self._compartment_index(self._r._rr.getFloatingSpeciesIds()[i])

    def getCompartmentIndexForBoundarySpecies(self, i):
        return self._compartment_index(self._r._rr.getBoundarySpeciesIds()[i])

    def getStoichiometry(self, *args):
        return self._r._rr.getStoichiometry(*args)

    def getInitStoichiometry(self, *args):
        return self._r._rr.getInitStoichiometry(*args)

    def getStoichiometryIds(self):
        return self._r._rr.getStoichiometryIds()

    def getStoichiometryId(self, index):
        return self._r._rr.getStoichiometryId(index)

    def getStoichiometryIndex(self, *args):
        return self._r._rr.getStoichiometryIndex(*args)

    def setStoichiometry(self, *args):
        return self._r._rr.setStoichiometry(*args)

    def getStateVectorId(self, i):
        return self._r._rr.getStateVectorIds()[i]


def _model_group(ids, getters, setters):
    """ExecutableModel ids / count / value getters (name -> selection format) / setters for one id list."""
    def ids_(self):
        return list(getattr(self._r._rr, ids)())

    def num(self):
        return len(getattr(self._r._rr, ids)())

    def getter(fmt):
        return lambda self, idx=None: self._get(getattr(self._r._rr, ids)(), fmt, idx)

    def setter(fmt):
        return lambda self, *args: self._set(getattr(self._r._rr, ids)(), fmt, args)
    group = ids[3:-3]  # FloatingSpecies, GlobalParameter, ...
    out = {ids: ids_, 'getNum' + group + ('' if group.endswith('Species') else 's'): num,
           'get%sId' % group: lambda self, i: getattr(self._r._rr, ids)()[i]}
    out.update({n: getter(f) for n, f in getters.items()})
    out.update({n: setter(f) for n, f in setters.items()})
    return out


for _ids, _get, _put in (
        ('getFloatingSpeciesIds', {'getFloatingSpeciesConcentrations': '[%s]', 'getFloatingSpeciesAmounts': '%s',
                                   'getFloatingSpeciesInitConcentrations': 'init([%s])',
                                   'getFloatingSpeciesInitAmounts': 'init(%s)',
                                   'getFloatingSpeciesConcentrationRates': "[%s]'",
                                   'getFloatingSpeciesAmountRates': "%s'"},
         {'setFloatingSpeciesConcentrations': '[%s]', 'setFloatingSpeciesAmounts': '%s',
          'setFloatingSpeciesInitConcentrations': 'init([%s])', 'setFloatingSpeciesInitAmounts': 'init(%s)'}),
        ('getBoundarySpeciesIds', {'getBoundarySpeciesConcentrations': '[%s]', 'getBoundarySpeciesAmounts': '%s',
                                   'getBoundarySpeciesInitConcentrations': 'init([%s])',
                                   'getBoundarySpeciesInitAmounts': 'init(%s)'},
         {'setBoundarySpeciesConcentrations': '[%s]', 'setBoundarySpeciesAmounts': '%s',
          'setBoundarySpeciesInitConcentrations': 'init([%s])', 'setBoundarySpeciesInitAmounts': 'init(%s)'}),
        ('getGlobalParameterIds', {'getGlobalParameterValues': '%s', 'getGlobalParameterInitValues': 'init(%s)'},
         {'setGlobalParameterValues': '%s', 'setGlobalParameterInitValues': 'init(%s)'}),
        ('getCompartmentIds', {'getCompartmentVolumes': '%s', 'getCompartmentInitVolumes': 'init(%s)'},
         {'setCompartmentVolumes': '%s', 'setCompartmentInitVolumes': 'init(%s)'}),
        ('getReactionIds', {'getReactionRates': '%s'}, {})):
    for _n, _f in _model_group(_ids, _get, _put).items():
        _f.__name__ = _n
        setattr(ExecutableModel, _n, _f)
ExecutableModel.getStateVectorRate = ExecutableModel.getFloatingSpeciesAmountRates
ExecutableModel.getRateRuleSymbols = ExecutableModel.getRateRuleIds
ExecutableModel.getAllTimeCourseComponentIds = ExecutableModel.keys
ExecutableModel.getValue = ExecutableModel.__getitem__
ExecutableModel.setValue = ExecutableModel.__setitem__
ExecutableModel.iterkeys, ExecutableModel.itervalues, ExecutableModel.iteritems = \
    ExecutableModel.keys, ExecutableModel.values, ExecutableModel.items
