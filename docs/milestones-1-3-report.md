# libroadrunner compatibility layer: milestones 1-3 report

Date: 2026-11-25. Platform: Windows 10, Rust 1.99, Python 3.12. Reference: libroadrunner 2.11.0.

This report covers the first three milestones of the spec "libroadrunner compatibility layer in
libroadrunner_nitro". Milestones 4 (switch Tellurium Nitro) and 5 (release) are not started.

## Summary

- `import roadrunner_nitro as roadrunner` now gives libroadrunner's Python API, with no Tellurium Nitro
  installed. Every public name of libroadrunner's module, of its `RoadRunner` class and of its `Config` class
  exists in roadrunner_nitro, apart from SWIG plumbing (`tests/test_api_parity.py`).
- **libroadrunner's own Python test suite (2.11.0) runs unchanged on roadrunner_nitro: 425 pass, 0 fail.**
  7 known differences are tracked as expected failures, each with its reason. Real libroadrunner passes the
  same suite.
- **SBML Test Suite, 1789 semantic time courses, judged against the suite's expected results:
  roadrunner_nitro passes all 1789; libroadrunner 2.11 passes 1576.** roadrunner_nitro passes every case
  libroadrunner passes, plus 213 more. Flux-balance cases (34) are skipped: not supported.
- The test suite found **8 bugs in the engine or the Python layer**, now fixed. One of them made every
  process draw the same "random" numbers whatever seed was set.
- Two decisions are needed (see "Open decisions"). One earlier decision is implemented: elasticities ignore
  `computeSteadyState`.

Commits (local, `main`, not pushed):

| Commit | Milestone |
| --- | --- |
| `07fab4c` | 1: package layout, tests run standalone |
| `3ef1b93` | 2: libroadrunner API moved into roadrunner_nitro |
| `763085b` | 3: libroadrunner's tests pass; engine fixes; SBML Test Suite |

## Test results

| Suite | Engine | Pass | Fail | Notes |
| --- | --- | --- | --- | --- |
| libroadrunner 2.11 Python tests (`tests/libroadrunner`) | roadrunner_nitro | 425 | 0 | 7 xfail (known differences), 45 skipped |
| same | libroadrunner 2.11 | 415 | 0 | 41 skipped |
| SBML Test Suite, semantic time courses | roadrunner_nitro | 1789 | 0 | 34 flux-balance cases skipped |
| same | libroadrunner 2.11 | 1576 | 213 | 5 wrong results, 208 refused (below) |
| SBML conversion round trips (`tests/test_sbml_conversion.py`) | roadrunner_nitro | 1830 | 0 | every level/version |
| API parity (`tests/test_api_parity.py`) | roadrunner_nitro | 3 | 0 | module, RoadRunner, Config |
| Other Python tests (`tests/`) | roadrunner_nitro | all but 7 | 7 | the 7 failed before milestone 1 (below) |
| C API (`capi/tests`) | roadrunner_nitro | 25 | 3 | the same 3 failed before milestone 1 |
| Rust (`cargo test --workspace`) | roadrunner_nitro | 601 | 1 | failed before milestone 1 (below) |

The two engines differ on libroadrunner's suite only in what is skipped: tests excluded for roadrunner_nitro
(SWIG internals, tests where roadrunner_nitro is more accurate) and the 7 known differences.

Why libroadrunner fails 213 SBML Test Suite cases:

| Reason | Cases |
| --- | --- |
| Refuses algebraic rules ("Unable to support algebraic rules") | 124 |
| Refuses delay differential equations | 50 |
| Refuses fast reactions | 34 |
| Wrong numbers | 5 |

## What was built

### Milestone 1: package layout

- maturin mixed Rust/Python project: the PyO3 extension is `roadrunner_nitro.core`, the public package is
  `python/roadrunner_nitro`. Version 0.2.0; matplotlib is an optional extra (`pip install .[plot]`).
- The Python tests no longer need a Tellurium Nitro checkout beside the repository: the models they use are in
  `tests/models`. The SBML Test Suite is not committed; the tests look for it in
  `tests/models/sbml-test-suite/cases/semantic` (or `SBML_SEMANTIC_DIR`) and skip without it
  (`tests/models/README.md`).

### Milestone 2: the libroadrunner API in roadrunner_nitro

Moved from Tellurium Nitro's `extended_roadrunner.py`, split by topic:

| File | Contents |
| --- | --- |
| `roadrunner.py` | `RoadRunner` |
| `named_array.py` | `NamedArray` |
| `solvers.py` | `Solver`, `Integrator`, `SteadyStateSolver`, the named solver classes, sensitivity solvers |
| `model.py` | `ExecutableModel` (`r.model`) |
| `config.py` | `Config`, `Logger`, `SelectionRecord` |
| `options.py` | `LoadSBMLOptions`, `SimulateOptions`, `RoadRunnerOptions`, `Dictionary`, `RoadRunnerMap`, `plotTimeSeriesSens` |
| `_sbml.py` | `from_sbml` (was Tellurium Nitro's `analysis/rrs_bridge.py`) |

Tellurium-only members are not in roadrunner_nitro; they go to Tellurium Nitro's subclass in milestone 4:
`getAntimony`, `export*`, `getCellML`, `getMatlab`, `draw`, `show`, `gillespie`, `seed`, the short aliases
(`fs`, `ps`, `sm`, ...), and `r.getFloatingSpeciesIds()`-style getters, which plain libroadrunner 2.11 has
only on `r.model`. `plot`, `showPlot` and `plotLegend` are libroadrunner's basic matplotlib versions.

### Milestone 3: libroadrunner's own tests

- `tests/libroadrunner/` mirrors libroadrunner's `test/` folder at tag v2.11.0: `python/` (its Python tests,
  unchanged), `rrtest_files/` and `sbml-test-suite/stochastic/`. A small `roadrunner` shim package makes
  `import roadrunner` load roadrunner_nitro, in the test process and in the worker processes the stochastic
  test starts. Run with `pytest tests/libroadrunner`; `ROADRUNNER_TEST_ENGINE=libroadrunner` runs the same
  tests on real libroadrunner.
- To pass them, the Python layer gained: all `Config` keys with libroadrunner's defaults; the option classes;
  named solver classes that can be built on a model (`CVODEIntegrator(r.getModel())`);
  `ForwardSensitivitySolver`; `RoadRunnerMap`; `NamedArray(shape)` and libroadrunner's pickle format; tuple
  return types where libroadrunner returns tuples; `keys()`, `values()`, `items()`, `len()` in libroadrunner's
  order; errors that are both `ValueError` and `RuntimeError`; named stoichiometries as attributes (`r.n`);
  solver registries that list solvers made so far, as libroadrunner does; elasticities in amount mode
  (`Config.ROADRUNNER_JACOBIAN_MODE`).
- `tests/test_sbml_semantic_suite.py` runs the SBML Test Suite on roadrunner_nitro;
  `benchmarks/sbml_test_suite.py` runs it on both engines and prints the comparison.

## Bugs found and fixed

Found by libroadrunner's tests and the SBML Test Suite, and confirmed against real libroadrunner:

1. **Random numbers ignored the seed.** SBML distrib draws (`normal(...)` and others) and the random order of
   simultaneous events used a generator created with a fixed constant. `setSeed` never reached it, and
   neither did libroadrunner's clock-based default. Every process therefore drew the same numbers, so
   parallel replicates were copies of each other: 61 of the 94 stochastic tests failed. Now seeded from the
   seed, so different seeds differ and one seed reproduces.
2. **Steady-state solver settings were lost on switching.** Setting a value on a steady-state solver that was
   not current was discarded. Integrators kept theirs; steady-state solvers now do too.
3. **Boundary species and compartment volume.** When a compartment's volume changed, floating species kept
   their amount (as libroadrunner) but boundary species kept their concentration. They now keep their amount.
4. **A negative steady state was refused.** For a model whose only steady state has a negative
   concentration, NLEQ failed; libroadrunner returns it. A non-negative steady state is still preferred,
   and found first where both exist; the negative one is now the last resort.
5. **Variable-step output repeated the event time.** The rows before and after an event had the same time;
   times now strictly increase, as libroadrunner's do.
6. **`removeRules(vid, useInitialValue=True)`** changed the variable's initial value to the rule's value. As
   documented, it now sets the current value and leaves the initial value alone.
7. **Pickling and `saveStateS` failed on Windows.** The state text went through a file in text mode, which
   rewrote its line endings.
8. **`ForwardSensitivitySolver.integrate`** stepped with the RoadRunner's integrator settings instead of the
   sensitivity solver's own.

Decided and implemented: **elasticities ignore `computeSteadyState`.** `getEE` and `getuEE` (Python and C API)
evaluate at the current state, as every other elasticity path does and as libroadrunner does.

## Known differences (tracked as expected failures)

| Tests | Difference | Status |
| --- | --- | --- |
| 4 forward sensitivity tests (Model269, Model28) | The rate law contains `factorial(ceil(S1*p1))`. The exact derivative of a step function is 0, but `p1` moves the times the step jumps, which libroadrunner's finite differences capture. | Limitation: exact forward sensitivities need jump conditions at discontinuities. |
| 3 integrator tests (FactorialInRateLaw) | `factorial(k1/k2)` with k1/k2 = 2.667. roadrunner_nitro gives round(x)! = 6. libroadrunner gives 2.963, which matches no standard definition (gamma(x+1) = 4.01). SBML defines factorial only for non-negative integers. | Decision needed. |

## Excluded from libroadrunner's suite

| What | Reason |
| --- | --- |
| `test_settings.py` | Tests SWIG's conversion of C++ setting types (`SettingTestsSwigAPI`). |
| `test_auto_plugin.py` | libroadrunner's C++ plugins (`rrplugins`); also imports Tellurium, which patches `roadrunner`. |
| `test___dict__` | Checks the SWIG proxy's attribute layout. |
| `test_needed_off_off` | Expects an error that roadrunner_nitro does not give: it finds this steady state. |
| `test_items`, `test_getUnscaledSpeciesElasticity` | Expected values are libroadrunner's finite-difference results compared more tightly than their own error. roadrunner_nitro gives the exact values (0.01, not 0.009999). |

## Failing before this work, not yet fixed

These failed on Windows before milestone 1. Milestones 1-3 did not change the engine code involved. The
original author only tested on macOS.

- **7 Python tests and 3 C API tests:** the order of independent and dependent species, `getKMatrix`, and
  the order of one set of eigenvalues differ from the stored libroadrunner results. A platform difference in
  the conservation analysis is likely (pivot choice or ties), but not yet investigated.
- **1 Rust test** (`jit_matches_interpreter`): compiled and interpreted `y^3` differ in the last bit at
  y = 1.3. The test requires bit-identical results; the build uses `target-cpu=native`.

## Open decisions

1. **Non-integer `factorial`:** keep round(x)!, match libroadrunner's 2.963 (once someone finds what it
   computes), or use gamma(x+1)?
2. **The libroadrunner reference for correctness:** the spec's criterion "within 1e-6 of libroadrunner"
   should become "judged against the test suites' expected results; roadrunner_nitro must pass everything
   libroadrunner passes". libroadrunner is wrong or refuses in 213 SBML Test Suite cases.

## Deviations from the spec

- The engine's own tests (`tests/test_core_vs_roadrunner.py` and others) still drive the raw engine,
  `roadrunner_nitro.core`, rather than being rerouted to the public API. They test the engine; the public API
  is tested by libroadrunner's suite and the API parity test.
- libroadrunner's test-model factory (`roadrunner.tests.TestModelFactory`) is a compiled module. It is
  borrowed from an installed libroadrunner as test data, so those tests skip without libroadrunner. Exporting
  its models and expected results to data files in the repository would remove that dependency.
- The API parity test is `tests/test_api_parity.py`, not `tests/python/test_api_parity.py`.

## Notes for running the tests on this machine

- The `PYTHONPATH` environment variable points at `site-packages\tellurium\roadrunner`, which puts regular
  Tellurium's RoadRunner on the import path of every Python. Run the tests with it unset.
- Git for Windows is set to `core.autocrlf=true`. The SBML Test Suite files must keep their LF line endings
  (the conversion test checks their checksums): clone it with `git -c core.autocrlf=false`.
- Simulations of models with random events now differ run to run unless a seed is set (as in
  libroadrunner). `tests/test_sbml_conversion.py` sets one.
- `tests/libroadrunner/sbml-test-suite/stochastic` holds SBML Test Suite files, which have their own license
  (LGPL), as in libroadrunner's repository. The rest of `tests/libroadrunner` is libroadrunner's (Apache 2.0).

## How to run

```
pip install .                                   # builds roadrunner_nitro
pytest tests                                    # engine, conversion, SBML Test Suite, API parity
pytest tests/libroadrunner                      # libroadrunner's own tests on roadrunner_nitro
ROADRUNNER_TEST_ENGINE=libroadrunner pytest tests/libroadrunner     # the same on libroadrunner
python benchmarks/sbml_test_suite.py            # SBML Test Suite on both engines, side by side
```

## Next

- Milestone 4: Tellurium Nitro subclasses `roadrunner_nitro.RoadRunner`, its `rrs/` engine copy is removed,
  and its test suite and regular Tellurium's run against it.
- Milestone 5: release 0.2.0; later, wheels on PyPI.
- Investigate the 7 + 3 Windows mismatches and the JIT last-bit difference.
