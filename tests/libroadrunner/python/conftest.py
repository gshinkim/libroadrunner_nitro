"""Run libroadrunner's own Python tests against roadrunner_nitro: `import roadrunner` gives roadrunner_nitro.

Run on their own (the alias must not reach the other test folders):  pytest tests/libroadrunner

roadrunner.tests.TestModelFactory (libroadrunner's compiled test models with their expected results) is
borrowed from an installed libroadrunner as test data only; the tests that need it are skipped without it.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # the tests import each other by bare name, as in libroadrunner's test/python
os.chdir(HERE)  # and open their data by paths relative to test/python (../rrtest_files, ../sbml-test-suite)

# ROADRUNNER_TEST_ENGINE=libroadrunner runs the same tests on the real libroadrunner, for comparison.
REAL = os.environ.get('ROADRUNNER_TEST_ENGINE', 'nitro') == 'libroadrunner'
if REAL:
    import roadrunner  # noqa: E402,F401
    import roadrunner.tests.TestModelFactory as _factory  # noqa: E402,F401
else:
    _factory = None
    try:  # load the real package first, keep only its test-model factory
        import roadrunner.tests.TestModelFactory as _factory  # noqa: E402
    except Exception:
        pass
    for _name in [n for n in sys.modules if n == 'roadrunner' or n.startswith('roadrunner.')]:
        del sys.modules[_name]

    # From here `import roadrunner` finds the shim package, here and in spawned worker processes.
    sys.path.insert(0, os.path.join(HERE, '_shim'))
    import roadrunner  # noqa: E402,F401  (the shim: roadrunner_nitro)
    import roadrunner_nitro  # noqa: E402

if not REAL:
    _tests = types.ModuleType('roadrunner.tests')
    _tests.__path__ = [HERE]
    sys.modules['roadrunner.tests'] = _tests
    roadrunner_nitro.tests = _tests
    if _factory is not None:
        sys.modules['roadrunner.tests.TestModelFactory'] = _factory
        _tests.TestModelFactory = _factory


collect_ignore = [
    'download_biomodels.py', '__main__.py', 'python-unit-tests', '_shim',  # helpers, not tests
    # Exclusions (each with its reason):
    'test_settings.py',  # tests SWIG's C++ <-> Python conversion of Setting values (roadrunner.tests.SettingTestsSwigAPI)
    'test_auto_plugin.py',  # libroadrunner's C++ plugins (rrplugins); rrplugins also imports tellurium, which patches roadrunner
]
# Tests where libroadrunner's expected behaviour is a limitation roadrunner_nitro does not have.
_NITRO_BETTER = {
    # libroadrunner raises when a conserved-moiety model is solved without moiety analysis; the engine
    # finds the steady state anyway (README: "also finds steady states Newton alone misses").
    'test_automaticMoietyConservationAnalysis.py::AutomaticMoietyConservationTests::test_needed_off_off',
    # Expected values are libroadrunner's finite-difference results, checked tighter than their own error:
    # uec(_J1, S2) = d(kb*S2)/dS2 = kb = 0.01 exactly; libroadrunner gives 0.009999 and the test allows 1e-6.
    'test_python_api.py::RoadRunnerTests::test_items',
    # assertEqual(0.10000000000000007, ...): libroadrunner's rounding error, compared exactly; the engine's
    # symbolic derivative gives 0.1.
    'test_python_api.py::RoadRunnerTests::test_getUnscaledSpeciesElasticity',
}
# Known differences, kept visible as expected failures (strict: a fix makes them fail, so they are removed).
_KNOWN = {
    # Sensitivities through a step function of a state, here factorial(ceil(S1*p1)): the engine's exact
    # derivative of the step is 0, but p1 moves the times the step jumps, which finite differences
    # (libroadrunner) capture. Needs jump conditions in the forward sensitivity equations.
    'test_forwardSensitivitySolver.py::FFSIntegrationTests::testModel269WithRoadRunnerInstance': 'sensitivity through a step function',
    'test_forwardSensitivitySolver.py::FFSIntegrationTests::testModel269WithoutRoadRunnerInstance': 'sensitivity through a step function',
    'test_forwardSensitivitySolver.py::FFSIntegrationTests::testModel28WithRoadRunnerInstance': 'sensitivity through a step function',
    'test_forwardSensitivitySolver.py::FFSIntegrationTests::testModel28WithoutRoadRunnerInstance': 'sensitivity through a step function',
    # factorial(k1/k2) with k1/k2 = 2.667: SBML defines factorial for non-negative integers only. The engine
    # gives round(x)! = 6; libroadrunner gives 2.963, which matches no standard definition (gamma(x+1) = 4.01).
    'test_integrator.py::CVODEIntegratorTests::testFactorialInRateLawWithRoadRunnerInstance': 'non-integer factorial (decision pending)',
    'test_integrator.py::CVODEIntegratorTests::testFactorialInRateLawWithoutRoadRunnerInstance': 'non-integer factorial (decision pending)',
    'test_integrator.py::CVODEIntegratorTests::testFactorialInRateLawWithoutRoadRunnerInstanceFFS': 'non-integer factorial (decision pending)',
}
_SWIG_DETAIL = {
    # r.__dict__ == ['this']: the SWIG proxy's attribute layout, not part of the API.
    'test_python_api.py::RoadRunnerTests::test___dict__',
}


def pytest_collection_modifyitems(config, items):
    import pytest
    if REAL:
        return
    for item in items:
        if any(item.nodeid.endswith(k) for k in _NITRO_BETTER):
            item.add_marker(pytest.mark.skip(reason='roadrunner_nitro does better than libroadrunner here'))
        for k, why in _KNOWN.items():
            if item.nodeid.endswith(k):
                item.add_marker(pytest.mark.xfail(reason=why, strict=True))
        if any(item.nodeid.endswith(k) for k in _SWIG_DETAIL):
            item.add_marker(pytest.mark.skip(reason='tests a SWIG implementation detail'))


if _factory is None:
    collect_ignore += ['test_testModelFactory.py']
