"""roadrunner_nitro: libroadrunner's engine rewritten in Rust, with libroadrunner's Python API.

    import roadrunner_nitro as roadrunner
    r = roadrunner.RoadRunner('model.xml')
    r.simulate(0, 10, 101)

The raw engine binding (plain arrays, no wrappers) is ``roadrunner_nitro.core``.
"""
from . import core
from .config import Config, Logger, SelectionRecord
from .model import ExecutableModel
from .named_array import NamedArray
from .roadrunner import RoadRunner
from .solvers import INTEGRATORS, STEADY_STATE_SOLVERS, Integrator, SensitivitySolver, Solver, SteadyStateSolver

# The engine's helpers that do not clash with libroadrunner's names stay at the top level.
from .core import (Model, RateDerivatives, Session, antimony_to_sbml, eigenvalues, flatten_comp,  # noqa: F401
                   from_sbml, gillespie_steps, sbml_edit, simulate_fixed, time_sensitivities)

try:
    from importlib.metadata import version as _version
    __version__ = _version('libroadrunner_nitro')
except Exception:  # running from a source tree
    __version__ = '0.0.0'

# libroadrunner's module-level names
integrators = list(INTEGRATORS)
steadyStateSolvers = list(STEADY_STATE_SOLVERS)
solvers = integrators + steadyStateSolvers

VERSIONSTR_BASIC, VERSIONSTR_COMPILER, VERSIONSTR_DATE = 1, 2, 4
VERSIONSTR_LIBSBML, VERSIONSTR_JITCOMPILER, VERSIONSTR_GIT_BRANCH, VERSIONSTR_GIT_COMMIT = 8, 16, 32, 64
VALIDATE_GENERAL, VALIDATE_UNITS, VALIDATE_IDENTIFIER = 1, 2, 4
VALIDATE_MATHML, VALIDATE_SBO, VALIDATE_OVERDETERMINED, VALIDATE_MODELING_PRACTICE = 8, 16, 32, 64


def getVersionStr(info=VERSIONSTR_BASIC):
    """The package version (``info`` flags are accepted for compatibility; the engine has no compiler or
    libSBML to report)."""
    return __version__


def getCopyrightStr():
    return 'roadrunner_nitro: Apache License 2.0. Based on libroadrunner, (c) University of Washington.'


def validateSBML(sbml, options=VALIDATE_GENERAL):
    """Problems found in an SBML document (text or file path), one per line; empty when it reads cleanly."""
    import os
    if not sbml.lstrip().startswith('<') and os.path.exists(sbml):
        with open(sbml, encoding='utf-8') as f:
            sbml = f.read()
    return core.sbml_edit('validate_current_sbml', sbml)
