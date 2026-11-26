"""libroadrunner's option objects and small helper types: LoadSBMLOptions, SimulateOptions, RoadRunnerOptions,
Dictionary, IntegratorException, RoadRunnerMap and plotTimeSeriesSens."""
import concurrent.futures
import os


class IntegratorException(Exception):
    """libroadrunner's integrator failure. Defined so scripts that name it import cleanly; the engine
    reports integration failures as ValueError."""


class Dictionary(object):
    """libroadrunner's string-keyed settings dictionary (keys(), values(), items(), r[key])."""

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

    def keys(self):
        return list(self._d.keys())

    def values(self):
        return list(self._d.values())

    def items(self):
        return list(self._d.items())

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        self._d[key] = value

    def __contains__(self, key):
        return key in self._d

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(self._d)

    def __repr__(self):
        return '<roadrunner.%s() %r>' % (type(self).__name__, self._d)

    def helloPython(self):
        return 'hello python'


class BasicDictionary(Dictionary):
    pass


class LoadSBMLOptions(Dictionary):
    """libroadrunner's model-loading options. The engine compiles every model the same way, so the flags are
    kept and reported but change nothing; ``conservedMoieties`` is applied by RoadRunner(sbml, options)."""
    CONSERVED_MOIETIES, RECOMPILE, READ_ONLY, MUTABLE_INITIAL_CONDITIONS = 1, 2, 4, 8
    OPTIMIZE_GVN, OPTIMIZE_CFG_SIMPLIFICATION, OPTIMIZE_INSTRUCTION_COMBINING = 16, 32, 64
    OPTIMIZE_DEAD_INST_ELIMINATION, OPTIMIZE_DEAD_CODE_ELIMINATION, OPTIMIZE_INSTRUCTION_SIMPLIFIER = 128, 256, 512
    OPTIMIZE, USE_MCJIT, LLVM_SYMBOL_CACHE, TURN_ON_VALIDATION = 1008, 1024, 2048, 4096
    MCJIT, LLJIT, NONE, LESS, DEFAULT, AGGRESSIVE = 16384, 32768, 65536, 131072, 262144, 524288
    NO_DEFAULT_SELECTIONS, NO_DEFAULT_STEADY_STATE_SELECTIONS = 1, 2

    def __init__(self, *args):
        super(LoadSBMLOptions, self).__init__()
        self.version, self.size, self.loadFlags = 0, 88, 0
        self.modelGeneratorOpt = 83976
        if args and isinstance(args[0], LoadSBMLOptions):
            self.__dict__.update({k: v for k, v in args[0].__dict__.items() if k != '_d'})

    def _flag(bit):
        return property(lambda self: bool(self.modelGeneratorOpt & bit),
                        lambda self, on: setattr(self, 'modelGeneratorOpt',
                                                 self.modelGeneratorOpt | bit if on else self.modelGeneratorOpt & ~bit))
    conservedMoieties = _flag(CONSERVED_MOIETIES)
    recompile = _flag(RECOMPILE)
    readOnly = _flag(READ_ONLY)
    mutableInitialConditions = _flag(MUTABLE_INITIAL_CONDITIONS)
    del _flag

    @property
    def noDefaultSelections(self):
        return bool(self.loadFlags & self.NO_DEFAULT_SELECTIONS)

    @noDefaultSelections.setter
    def noDefaultSelections(self, on):
        self.loadFlags = self.loadFlags | 1 if on else self.loadFlags & ~1


class SimulateOptions(Dictionary):
    """libroadrunner's simulation options: start, duration (= end - start), steps, times, reset_model, ...
    ``r.simulate(options)`` uses them."""

    def __init__(self):
        super(SimulateOptions, self).__init__()
        self._start, self._duration, self._steps = 0.0, 5.0, 50
        self.times = []
        self.reset_model, self.structured_result, self.copy_result = False, False, True
        self.output_file = ''
        self.variables, self.amounts, self.concentrations = (), (), ()

    # Setting start, duration, end or steps puts the options back on an even grid (times cleared),
    # as libroadrunner's SimulateOptions.
    def _grid(name):
        def get(self):
            return getattr(self, name)

        def put(self, v):
            setattr(self, name, int(v) if name == '_steps' else float(v))
            self.times = []
        return property(get, put)
    start = _grid('_start')
    duration = _grid('_duration')
    steps = _grid('_steps')
    del _grid
    end = property(lambda self: self.start + self.duration,
                   lambda self, v: setattr(self, 'duration', float(v) - self.start))
    resetModel = property(lambda self: self.reset_model, lambda self, v: setattr(self, 'reset_model', bool(v)))
    structuredResult = property(lambda self: self.structured_result,
                                lambda self, v: setattr(self, 'structured_result', bool(v)))
    copyResult = property(lambda self: self.copy_result, lambda self, v: setattr(self, 'copy_result', bool(v)))

    def getNext(self, i):
        """Time of output point i on this grid."""
        if self.times:
            return self.times[i]
        return self.start + i * self.duration / self.steps


class RoadRunnerOptions(object):
    """libroadrunner's per-instance options (r.options)."""
    DISABLE_PYTHON_DYNAMIC_PROPERTIES = 1

    def __init__(self):
        self.flags = 0
        self.jacobianStepSize = 1e-05
        self.diffStepSize, self.steadyStateThreshold, self.fluxThreshold = 0.02, 0.01, 1e-09

    @property
    def disablePythonDynamicProperties(self):
        return bool(self.flags & self.DISABLE_PYTHON_DYNAMIC_PROPERTIES)

    @disablePythonDynamicProperties.setter
    def disablePythonDynamicProperties(self, on):
        self.flags = self.flags | 1 if on else self.flags & ~1


class RoadRunnerMap(object):
    """libroadrunner's RoadRunnerMap: RoadRunner instances keyed by model id, loaded in parallel.

        m = RoadRunnerMap(['a.xml', 'b.xml'], numThreads=4); m['modelA'].simulate(0, 10, 11)
    """

    def __init__(self, sbmlOrUrls=(), numThreads=1):
        self._map = {}
        self._threads = max(1, int(numThreads))
        if sbmlOrUrls:
            self.insert(list(sbmlOrUrls))

    def _load(self, sbml):
        from .roadrunner import RoadRunner
        r = RoadRunner(sbml)
        return r.getModelName(), r  # the model's name, else its id (as libroadrunner)

    def insert(self, *args):
        """insert(sbmlOrPaths) or insert(key, sbmlOrPath or RoadRunner)."""
        if len(args) == 2:
            key, item = args
            from .roadrunner import RoadRunner
            self._map[key] = item if isinstance(item, RoadRunner) else self._load(item)[1]
            return
        items = args[0] if isinstance(args[0], (list, tuple)) else [args[0]]
        with concurrent.futures.ThreadPoolExecutor(self._threads) as pool:
            for key, r in pool.map(self._load, items):
                self._map[key] = r

    def remove(self, key):
        del self._map[key]

    def clear(self):
        self._map.clear()

    def keys(self):
        return list(self._map.keys())

    def values(self):
        return list(self._map.values())

    def items(self):
        return list(self._map.items())

    def count(self, key):
        return int(key in self._map)

    def size(self):
        return len(self._map)

    __len__ = size

    def empty(self):
        return not self._map

    def getNumThreads(self):
        return self._threads

    def setNumThreads(self, n):
        self._threads = max(1, int(n))

    def __getitem__(self, key):
        return self._map[key]

    def __setitem__(self, key, sbml):
        """m[key] = SBML text or path. A RoadRunner is refused (TypeError), as in libroadrunner: pass a list
        of models to the constructor or insert() to load them in parallel."""
        from .roadrunner import RoadRunner
        if isinstance(sbml, RoadRunner):
            raise TypeError("RoadRunnerMap: assign SBML (text or path), not a RoadRunner")
        self._map[key] = self._load(sbml)[1]

    def __delitem__(self, key):
        del self._map[key]

    def __contains__(self, key):
        return key in self._map

    def __iter__(self):
        return iter(self._map)


def plotTimeSeriesSens(time, sens, param_ids, species_ids, ncols=4, show=True, savefig=None):
    """Plot timeSeriesSensitivities output: one panel per parameter, one line per species."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("plotting needs matplotlib: pip install matplotlib")
    n = len(param_ids)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, min(ncols, max(n, 1)), squeeze=False, figsize=(4 * min(ncols, max(n, 1)), 3 * nrows))
    for k, p in enumerate(param_ids):
        ax = axes[k // ncols][k % ncols]
        for j, s in enumerate(species_ids):
            ax.plot(time, sens[:, k, j], label=s)
        ax.set_title(p)
    axes[0][0].legend()
    if savefig:
        fig.savefig(os.fspath(savefig))
    if show:
        plt.show()
    return fig
