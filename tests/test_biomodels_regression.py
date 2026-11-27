"""Curated BioModels that exposed integrator and event bugs, against reference time courses from COPASI or
libroadrunner (tests/make_biomodels_regression.py says which, and what each model guards). The bugs were errors
of 5% to 100%; the tolerance (1e-3 relative, 1e-6 of the column's largest value absolute) leaves room for the
engines' ordinary tolerance differences. tests/test_regressions.py has each bug in its smallest form.
"""
import os
import sys
import warnings

import numpy as np
import pytest

import roadrunner_nitro as rn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_biomodels_regression import MODELS, OUT, POINTS, RUN_ONLY, SOURCE, T_END  # noqa: E402

REF = np.load(OUT, allow_pickle=False)


def simulate(model, cols=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # equality-trigger warnings
        r = rn.RoadRunner(os.path.join(MODELS, model + '.xml'))
    r.integrator.relative_tolerance = 1e-8  # the BioModels comparison's settings
    r.integrator.absolute_tolerance = 1e-12
    cols = cols or ['time'] + ['[%s]' % s for s in r.model.getFloatingSpeciesIds()]
    return np.asarray(r.simulate(0, T_END, POINTS, cols), float)


@pytest.mark.parametrize('model', sorted(SOURCE))
def test_against_reference(model):
    cols = [str(c) for c in REF[model + '/cols']]
    want = REF[model + '/data']
    got = simulate(model, cols)
    scale = np.abs(want).max(axis=0)
    bad = np.abs(got - want) > 1e-3 * np.abs(want) + 1e-6 * scale
    if bad.any():
        i, j = np.argwhere(bad)[0]
        pytest.fail('%s (%s): %d values differ from %s; first at t = %g, %s: nitro %.8g, reference %.8g' % (
            model, SOURCE[model][1], bad.sum(), str(REF[model + '/source']), want[i, 0], cols[j], got[i, j], want[i, j]))


@pytest.mark.parametrize('model', sorted(RUN_ONLY))
def test_runs_to_the_end(model):
    y = simulate(model)
    assert y.shape[0] == POINTS and np.isfinite(y).all(), RUN_ONLY[model]
