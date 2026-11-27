# Test models

Models the Python tests need, so that `pytest` runs from this repository alone.

- `sosbench/`: copied from Tellurium Nitro's `benchmarks/sosbench`
  (https://github.com/gshinkim/tellurium-nitro). The numbered cases are SBML Test Suite semantic cases
  (https://github.com/sbmlteam/sbml-test-suite); `repressilator` is the repressilator model.
- `bigmodel/synthetic_1000_species.ant`: copied from Tellurium Nitro's `benchmarks/bigmodel`.
- `sbml-test-suite/` (not committed): the SBML Test Suite semantic cases at commit `bbcbc80`, used by
  `tests/test_sbml_conversion.py`. Put the suite's `cases/semantic` folder at
  `sbml-test-suite/cases/semantic`, or set `SBML_SEMANTIC_DIR`. The files must keep the suite's LF line
  endings (the tests check their checksums): on Windows, clone with `git -c core.autocrlf=false`.
  Without the suite those cases are skipped.
- `biomodels_regression/`: 20 curated models from BioModels (https://www.ebi.ac.uk/biomodels/, released under
  CC0), unchanged, for `tests/test_biomodels_regression.py`; `tests/make_biomodels_regression.py` lists what each
  guards and makes their reference time courses (`tests/reference/biomodels_regression.npz`).
- `biomodels/` (not committed): all 1075 curated BioModels, for `benchmarks/biomodels_compare.py`.
