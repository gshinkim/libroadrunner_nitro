"""libroadrunner's Logger, Config and SelectionRecord."""
# Moved from Tellurium Nitro (tellurium_nitro/roadrunner/extended_roadrunner.py, Apache 2.0).


class Logger(object):
    """libroadrunner's Logger API (levels as libroadrunner numbers them), on Python's logging."""
    LOG_CURRENT, LOG_FATAL, LOG_CRITICAL, LOG_ERROR, LOG_WARNING, LOG_NOTICE, LOG_INFORMATION, LOG_DEBUG, LOG_TRACE = range(9)
    _level = LOG_NOTICE

    @classmethod
    def setLevel(cls, level=LOG_CURRENT):
        if level != cls.LOG_CURRENT:
            cls._level = level

    @classmethod
    def getLevel(cls):
        return cls._level

    @classmethod
    def log(cls, level, msg):
        if level <= cls._level:
            import logging
            logging.getLogger('roadrunner_nitro').log(max(10, 60 - 10 * level), msg)


# libroadrunner 2.11's Config keys, numbered as libroadrunner numbers them, with its default values.
_KEYS = [
    ('LOADSBMLOPTIONS_CONSERVED_MOIETIES', False), ('LOADSBMLOPTIONS_RECOMPILE', False),
    ('LOADSBMLOPTIONS_READ_ONLY', False), ('LOADSBMLOPTIONS_MUTABLE_INITIAL_CONDITIONS', True),
    ('LOADSBMLOPTIONS_OPTIMIZE_GVN', False), ('LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION', False),
    ('LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING', False), ('LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION', False),
    ('LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION', False), ('LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER', False),
    ('LOADSBMLOPTIONS_USE_MCJIT', False), ('SIMULATEOPTIONS_STEPS', 50), ('SIMULATEOPTIONS_DURATION', 5.0),
    ('SIMULATEOPTIONS_ABSOLUTE', 1e-10), ('SIMULATEOPTIONS_RELATIVE', 1e-05), ('SIMULATEOPTIONS_STRUCTURED_RESULT', False),
    ('SIMULATEOPTIONS_STIFF', True), ('SIMULATEOPTIONS_MULTI_STEP', False),
    ('SIMULATEOPTIONS_DETERMINISTIC_VARIABLE_STEP', False), ('SIMULATEOPTIONS_STOCHASTIC_VARIABLE_STEP', True),
    ('SIMULATEOPTIONS_INTEGRATOR', 'CVODE'), ('SIMULATEOPTIONS_INITIAL_TIMESTEP', -1),
    ('SIMULATEOPTIONS_MINIMUM_TIMESTEP', -1), ('SIMULATEOPTIONS_MAXIMUM_TIMESTEP', -1),
    ('SIMULATEOPTIONS_MAXIMUM_NUM_STEPS', -1), ('ROADRUNNER_DISABLE_WARNINGS', False),
    ('ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES', False), ('SBML_APPLICABLEVALIDATORS', 111),
    ('ROADRUNNER_JACOBIAN_STEP_SIZE', 1e-05), ('MODEL_RESET', 41), ('CVODE_MIN_ABSOLUTE', 1e-12),
    ('CVODE_MIN_RELATIVE', 1e-06), ('SIMULATEOPTIONS_COPY_RESULT', True), ('STEADYSTATE_PRESIMULATION', False),
    ('STEADYSTATE_PRESIMULATION_MAX_STEPS', 100), ('STEADYSTATE_PRESIMULATION_TIME', 100.0),
    ('STEADYSTATE_APPROX', False), ('STEADYSTATE_APPROX_TOL', 1e-06), ('STEADYSTATE_APPROX_MAX_STEPS', 10000),
    ('STEADYSTATE_APPROX_TIME', 10000.0), ('STEADYSTATE_RELATIVE', 1e-12), ('STEADYSTATE_MAXIMUM_NUM_STEPS', 100),
    ('STEADYSTATE_MINIMUM_DAMPING', 1e-20), ('STEADYSTATE_BROYDEN', 0), ('STEADYSTATE_LINEARITY', 3),
    ('ROADRUNNER_JACOBIAN_MODE', 1), ('TEMP_DIR_PATH', '.'), ('LOGGER_LOG_FILE_PATH', ''), ('RANDOM_SEED', -1),
    ('PYTHON_ENABLE_NAMED_MATRIX', True), ('LLVM_SYMBOL_CACHE', True), ('OPTIMIZE_REACTION_RATE_SELECTION', True),
    ('LOADSBMLOPTIONS_PERMISSIVE', True), ('MAX_OUTPUT_ROWS', 100000), ('ALLOW_EVENTS_IN_STEADY_STATE_CALCULATIONS', False),
    ('MAX_EVENT_CASCADE', 10000), ('VALIDATION_IN_REGENERATION', True), ('K_ROWS_PER_WRITE', 1000),
    ('LLVM_BACKEND', 0), ('LLJIT_OPTIMIZATION_LEVEL', 0), ('LLJIT_NUM_THREADS', 1),
    ('METABOLIC_CONTROL_ANALYSIS_FLUX_THRESHOLD', 1e-09), ('METABOLIC_CONTROL_ANALYSIS_DIFFERENTIAL_STEP_SIZE', 0.02),
    ('METABOLIC_CONTROL_ANALYSIS_STEADY_STATE_THRESHOLD', 0.01),
]


class Config(object):
    """libroadrunner's Config: settings by key (``Config.setValue(Config.MAX_OUTPUT_ROWS, 10**6)``).

    All of libroadrunner's keys exist with its defaults. The engine has no LLVM, so the LLVM and
    LOADSBMLOPTIONS_OPTIMIZE_* keys are kept and reported back but have no effect."""
    CONFIG_END = len(_KEYS)
    # values of libroadrunner's enumerations that live on Config
    ROADRUNNER_JACOBIAN_MODE_AMOUNTS, ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS = 0, 1
    ROADRUNNER_DISABLE_WARNINGS_STEADYSTATE, ROADRUNNER_DISABLE_WARNINGS_CONSERVED_MOIETY = 1, 2
    MCJIT, LLJIT = 0, 1
    NONE, LESS, DEFAULT, AGGRESSIVE = 0, 1, 2, 3
    _values = {i: v for i, (_, v) in enumerate(_KEYS)}
    _path = ''

    @classmethod
    def _key(cls, key):
        if isinstance(key, str):
            key = cls.stringToKey(key)
        if not isinstance(key, int) or not 0 <= key < cls.CONFIG_END:
            raise ValueError('invalid Config key: %r' % (key,))
        return key

    @classmethod
    def setValue(cls, key, value):
        cls._values[cls._key(key)] = value

    @classmethod
    def getValue(cls, key):
        return cls._values[cls._key(key)]

    @classmethod
    def getKeyList(cls):
        return [n for n, _ in _KEYS]

    @classmethod
    def stringToKey(cls, name):
        try:
            return [n for n, _ in _KEYS].index(name)
        except ValueError:
            raise ValueError('no Config key named %r' % (name,))

    @classmethod
    def getConfigFilePath(cls):
        return cls._path

    @classmethod
    def readConfigFile(cls, path):
        """Read ``KEY: value`` lines (libroadrunner's format) into the settings."""
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.split('#', 1)[0].strip()
                if ':' not in line:
                    continue
                name, value = (s.strip() for s in line.split(':', 1))
                old = cls.getValue(name)
                cls.setValue(name, value.lower() in ('true', '1') if isinstance(old, bool) else type(old)(value))
        cls._path = path

    @classmethod
    def writeConfigFile(cls, path):
        with open(path, 'w', encoding='utf-8') as f:
            for i, (name, _) in enumerate(_KEYS):
                f.write('%s: %s\n' % (name, cls._values[i]))


for _i, (_name, _) in enumerate(_KEYS):
    setattr(Config, _name, _i)


class SelectionRecord(object):
    """libroadrunner's selection bit flags for getIds and reset."""
    TIME, CONCENTRATION, AMOUNT, RATE, BOUNDARY, FLOATING, _COMPARTMENT, _GLOBAL_PARAMETER = 1, 2, 4, 8, 16, 32, 64, 128
    REACTION, INITIAL, CURRENT, UNSCALED, ELASTICITY, CONTROL = 256, 512, 1024, 2048, 4096, 8192
    EIGENVALUE_REAL, ELEMENT, STOICHIOMETRY, UNKNOWN = 0x4000, 0x8000, 0x10000, 0x20000
    DEPENDENT, INDEPENDENT, CONSERVED_MOIETY = 0x40000, 0x80000, 0x100000
    STATE_VECTOR = SBML_INITIALIZE = 0x200000
    EVENT, EIGENVALUE_IMAG, MULTI_SPECIES_REFERENCE = 0x400000, 0x800000, 0x1000000
    EIGENVALUE_COMPLEX = EIGENVALUE_REAL | EIGENVALUE_IMAG
    UNKNOWN_CONCENTRATION, UNKNOWN_ELEMENT = UNKNOWN | CONCENTRATION, UNKNOWN | ELEMENT

    COMPARTMENT = 0xc0440
    GLOBAL_PARAMETER = 0xc0480
    FLOATING_AMOUNT, FLOATING_CONCENTRATION = 0xc0424, 0xc0422
    INDEPENDENT_FLOATING_AMOUNT, DEPENDENT_FLOATING_AMOUNT = 0x80424, 0x40424
    FLOATING_AMOUNT_RATE, FLOATING_CONCENTRATION_RATE = 0x40028, 0x4002a
    BOUNDARY_AMOUNT, BOUNDARY_CONCENTRATION = 0xc0414, 0xc0412
    INDEPENDENT_BOUNDARY_AMOUNT, DEPENDENT_BOUNDARY_AMOUNT = 0x80414, 0x40414
    BOUNDARY_AMOUNT_RATE, BOUNDARY_CONCENTRATION_RATE = 0x40018, 0x4001a
    GLOBAL_PARAMETER_RATE, REACTION_RATE, COMPARTMENT_RATE = 0x88, 0x40108, 0xc0448

    INITIAL_AMOUNT, INITIAL_CONCENTRATION = 0xc0204, 0xc0202
    INITIAL_FLOATING_AMOUNT, INITIAL_FLOATING_CONCENTRATION = 0xc0224, 0xc0222
    INITIAL_BOUNDARY_AMOUNT, INITIAL_BOUNDARY_CONCENTRATION = 0xc0214, 0xc0212
    INITIAL_COMPARTMENT, INITIAL_GLOBAL_PARAMETER = 0xc0240, 0xc0280
    DEPENDENT_INITIAL_GLOBAL_PARAMETER = 0x40280
    INITIAL_STOICHIOMETRY = INITIAL | STOICHIOMETRY | DEPENDENT | INDEPENDENT
    INITIAL_MULTI_SPECIES_REFERENCE = INITIAL | MULTI_SPECIES_REFERENCE | DEPENDENT | INDEPENDENT
    UNSCALED_ELASTICITY, UNSCALED_CONTROL = UNSCALED | ELASTICITY, UNSCALED | CONTROL

    ALL = -1
    ALL_INDEPENDENT, ALL_DEPENDENT = ~DEPENDENT, ~INDEPENDENT
    ALL_INDEPENDENT_AMOUNT, ALL_DEPENDENT_AMOUNT = ALL_INDEPENDENT & ~CONCENTRATION, ALL_DEPENDENT & ~CONCENTRATION
    ALL_INDEPENDENT_CONCENTRATION, ALL_DEPENDENT_CONCENTRATION = ALL_INDEPENDENT & ~AMOUNT, ALL_DEPENDENT & ~AMOUNT
    MODEL_STATE = ALL_INDEPENDENT_AMOUNT & ~INITIAL

