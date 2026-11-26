"""Regression tests for integrator and event bugs found with the curated BioModels comparison
(benchmarks/biomodels_compare.py, COPASI as referee). Each test is the bug in its smallest form, checked against
the exact answer where there is one; tests/test_biomodels_regression.py checks the BioModels themselves.

All run at rtol 1e-8, atol 1e-12, the settings of the BioModels comparison.
"""
import warnings

import antimony
import numpy as np
import pytest

import roadrunner_nitro as rn


def sbml_of(src):
    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString(src) >= 0, antimony.getLastError()
    return antimony.getSBMLString(antimony.getMainModuleName())


def run(src, end, points, sel, rtol=1e-8, atol=1e-12):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = rn.RoadRunner(sbml_of(src))
    r.integrator.relative_tolerance = rtol
    r.integrator.absolute_tolerance = atol
    return np.asarray(r.simulate(0, end, points, ['time'] + sel))


def test_fractional_power_at_zero():
    """d/dV V^0.2 is infinite at V = 0: once a NaN/inf in the Newton matrix made every step fail; a large finite
    stand-in then made BDF accept unconverged steps (BIOMD0000000246, VIP^0.2 from VIP = 0)."""
    y = run('species V = 0; J1: -> V; 0.3; J2: V -> ; 0.5*V^0.2', 1, 11, ['[V]'])
    # libroadrunner 2.11 (CVODE) and COPASI-tight agree on these to 8 digits
    want = [0.0, 0.0121858, 0.02024689, 0.02663421, 0.03194861, 0.03648556,
            0.04042233, 0.04387724, 0.04693477, 0.04965819, 0.0520965]
    np.testing.assert_allclose(y[:, 1], want, rtol=1e-6, atol=1e-9)


def test_tanh_past_overflow():
    """tanh(x) was built as (e^x - e^-x) / (e^x + e^-x): NaN past |x| = 709.78 (BIOMD0000000749, ld / tanh(R / ld))."""
    y = run('species x = 212; J: -> x; 0.3/tanh(x/0.3)', 10, 11, ['[x]'])
    np.testing.assert_allclose(y[:, 1], 212 + 0.3 * y[:, 0], rtol=1e-12)


def test_output_at_event_time_is_post_event():
    """An event exactly at an output time: the output reports the state after the event, as libroadrunner and
    COPASI do (root finding had placed the event 1e-11 after the output)."""
    y = run('species S = 1; J: S -> ; 0.1*S; E: at time >= 15: S = 5', 20, 21, ['[S]'])
    np.testing.assert_allclose(y[14, 1], np.exp(-1.4), rtol=1e-7)
    assert y[15, 1] == pytest.approx(5.0, rel=1e-12)
    np.testing.assert_allclose(y[20, 1], 5 * np.exp(-0.5), rtol=1e-7)


def test_equality_trigger_fires_at_output_time():
    """`time == 20` is true at one instant: it fires when an output time lands on it, as in libroadrunner (steps
    never landed on t = 20 before), and loading such a model warns to use >= instead."""
    src = 'species S = 1; J: S -> ; 0; E: at time == 20: S = S + 1'
    y = run(src, 30, 31, ['[S]'])
    assert (y[19, 1], y[20, 1], y[30, 1]) == (1.0, 2.0, 2.0)
    with pytest.warns(UserWarning, match='tests equality'):
        rn.RoadRunner(sbml_of(src))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        rn.RoadRunner(sbml_of('species S = 1; J: S -> ; 0; E: at time >= 20: S = S + 1'))


def test_short_pulse_in_rate_law():
    """A 5 ms window in a rate law was stepped over, the condition false at both ends of a step (BIOMD0000000162)."""
    y = run('species S = 0; J: -> S; piecewise(100, time > 1 && time < 1.005, 0)', 3, 4, ['[S]'])
    assert y[-1, 1] == pytest.approx(0.5, rel=1e-9)


def test_infusion_window_at_tight_atol():
    """`time >= 21` switching on an input to a state near 0: a step ending at the switch, or starting just
    before it, never passed its error test at atol 1e-12 (BIOMD0000000879)."""
    y = run('species Q = 0; J1: -> Q; piecewise(8, time >= 21 && time <= 21.125, 0); J2: Q -> ; 4*Q',
            30, 31, ['[Q]'])
    q_end = 2 * (1 - np.exp(-0.5))  # Q' = 8 - 4 Q for 0.125
    np.testing.assert_allclose(y[22, 1], q_end * np.exp(-4 * 0.875), rtol=1e-6)
    np.testing.assert_allclose(y[30, 1], q_end * np.exp(-4 * 8.875), atol=1e-12)  # ~3e-16: below atol
    assert np.all(y[:21, 1] == 0)


def test_reset_on_boundary_does_not_refire():
    """`at V < 0: V = 0` with V' < 0 at V = 0 fired again at every, ever shorter, step (BIOMD0000000711). As CVODE
    (which deactivates a root function that is exactly 0 at a restart), the trigger left on its boundary counts
    as still true: V drifts below 0 without re-firing, exactly as in libroadrunner 2.11."""
    y = run('species V = 0.001; J: V -> ; 0.001; E: at V < 0: V = 0', 5, 6, ['[V]'])
    np.testing.assert_allclose(y[:, 1], [0.001, 0.0, -0.001, -0.002, -0.003, -0.004], atol=1e-12)


def test_state_switch_in_rate_law_at_tight_atol():
    """A rate-law switch feeding a state at 0 (BIOMD0000000722, piecewise on LECs + Capillaries < L): every trial
    step across it was rejected at atol 1e-12 and steps shrank toward the switch until the run failed."""
    y = run('species L = 0, C = 0; J1: -> L; 1; J2: -> C; piecewise(0, L < 10, 1000)', 12, 13, ['[C]'])
    np.testing.assert_allclose(y[10:, 1], [0, 1000, 2000], rtol=1e-9, atol=1e-6)
    assert np.all(y[:10, 1] == 0)


def test_time_modulo_switch_at_tight_atol():
    """Light off at time % 24 = 8 with a fast-decaying state (BIOMD0000000859): failed at atol 1e-12."""
    y = run('species T = 1; k := piecewise(10, (time % 24) - 8 < 0, 0); J1: -> T; k; J2: T -> ; 50*T',
            12, 13, ['[T]'])
    np.testing.assert_allclose(y[1:9, 1], 0.2, rtol=1e-6)
    np.testing.assert_allclose(y[10:, 1], 0, atol=1e-12)
