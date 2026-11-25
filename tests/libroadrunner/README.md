# libroadrunner's own tests, run against roadrunner_nitro

Copied from [libroadrunner](https://github.com/sys-bio/roadrunner) at tag `v2.11.0` (Apache License 2.0,
University of Washington), keeping its `test/` layout:

- `python/`: libroadrunner's Python tests (`test/python`), unchanged except `__init__.py` (emptied: the
  tests are run by pytest, not as a package) and the added `conftest.py`.
- `rrtest_files/`: the `.rrtest` cases `python/test_rrtests.py` reads (`test/rrtest_files`).
- `sbml-test-suite/stochastic/`: the stochastic SBML Test Suite cases `python/test_runStochasticTestSuite.py`
  reads (`test/sbml-test-suite/stochastic`).

`python/conftest.py` makes `import roadrunner` load `roadrunner_nitro`, so every test runs on the Rust
engine. Run them on their own:

    pytest tests/libroadrunner

`roadrunner.tests.TestModelFactory` (libroadrunner's compiled test models with their expected results)
comes from an installed libroadrunner, as test data only; without libroadrunner those tests are skipped.

Excluded tests, each with its reason, are listed in `python/conftest.py`.
