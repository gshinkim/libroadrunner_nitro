"""SBML Test Suite, semantic time courses, through roadrunner_nitro's libroadrunner-compatible API, judged against
the suite's expected results (see tests/sts_cases.py). Skipped when the suite is absent (tests/models/README.md).
Flux-balance cases are not supported and are skipped.

    pytest tests/test_sbml_semantic_suite.py
    python benchmarks/sbml_test_suite.py      # the same cases on roadrunner_nitro and on libroadrunner, side by side
"""
import pytest

import roadrunner_nitro
from sts_cases import case_ids, run_case

CASES = case_ids()


@pytest.mark.skipif(not CASES, reason='SBML Test Suite not found: see tests/models/README.md')
@pytest.mark.parametrize('cid', CASES or ['none'])
def test_semantic_case(cid):
    out = run_case(roadrunner_nitro.RoadRunner, cid)
    if out['status'] == 'skip':
        pytest.skip(out['reason'])
    assert out['status'] == 'pass', '%s %s: %s' % (cid, out['status'], out.get('reason'))
