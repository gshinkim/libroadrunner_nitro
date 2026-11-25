"""API parity: every public name of libroadrunner's Python module, its RoadRunner class and its Config class
exists in roadrunner_nitro, apart from the written allowlists below (Tier 3 of the spec: SWIG and C++
plumbing with no meaning for the Rust engine). Skipped when libroadrunner is not installed.

    python -m pytest tests/test_api_parity.py -rA     # prints what is missing
"""
import pytest

roadrunner = pytest.importorskip('roadrunner')
import roadrunner_nitro  # noqa: E402

# libroadrunner names that are SWIG or C++ plumbing, not part of the API a script uses.
MODULE_ALLOW = {
    'SwigPyIterator', 'SHARED_PTR_DISOWN', 'DictionaryVector', 'IntVector', 'StringVector', 'DoubleVector',
    'roadrunner', 'np', 'weakref', 'sigtrap', 'runTests', 'Optional', 'List', 'Compiler',
    'ExecutableModelFactory', 'RegistrationFactory', 'Registrable', 'PyConservedMoietyConverter',
    'PyEventListener', 'PyIntegratorListener', 'EventListener', 'fixMissingStoichAndMath', 'SBMLReader',
}
CLASS_ALLOW = {'thisown', 'this'}
CONFIG_ALLOW = {'thisown'}


def public(obj):
    return {n for n in dir(obj) if not n.startswith('_')}


def missing(ref, ours, allow):
    return sorted(public(ref) - public(ours) - allow)


def test_module_names():
    gone = missing(roadrunner, roadrunner_nitro, MODULE_ALLOW)
    assert not gone, 'module names missing from roadrunner_nitro: %s' % gone


def test_roadrunner_class_names():
    gone = missing(roadrunner.RoadRunner, roadrunner_nitro.RoadRunner, CLASS_ALLOW)
    assert not gone, 'RoadRunner names missing from roadrunner_nitro: %s' % gone


def test_config_names():
    gone = missing(roadrunner.Config, roadrunner_nitro.Config, CONFIG_ALLOW)
    assert not gone, 'Config names missing from roadrunner_nitro: %s' % gone
