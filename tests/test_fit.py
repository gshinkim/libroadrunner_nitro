"""roadrunner_nitro.fit (parameter estimation, stage 1) through Python: problem building, PEtab import, exact
gradients against finite differences, multistart recovery of known parameters, determinism across thread counts,
cancelling, Fisher intervals, and agreement with scipy's least_squares driven by our residuals and Jacobian.

The model X0 -> S1 (k1 X0), S1 -> (k2 S1) has S1(t) = k1 X0 / k2 (1 - exp(-k2 t)) from S1(0) = 0.
"""
import numpy as np
import pytest

import roadrunner_nitro as rn
from roadrunner_nitro import fit

MODEL = """
model m
    compartment c = 1
    species $X0 in c, S1 in c
    J1: X0 -> S1; k1*X0
    J2: S1 -> ; k2*S1
    X0 = 10; S1 = 0
    k1 = 0.5; k2 = 0.2
end
"""
K1, K2 = 0.8, 0.3
TIMES = np.arange(1.0, 17.0, 2.0)


def s1(t, x0, k1=K1, k2=K2):
    return k1 * x0 / k2 * (1 - np.exp(-k2 * t))


def problem(noise=0.0):
    p = fit.Problem.from_antimony(MODEL)
    p.add_parameter('k1', 'log10', 1e-3, 1e3, 1.0)
    p.add_parameter('k2', 'log10', 1e-3, 1e3, 1.0)
    p.add_parameter('scale', 'log', 0.1, 10, 1.0)
    p.add_condition('full')
    p.add_condition('half', {'X0': 5})
    p.add_observable('S1_obs', 'S1', '0.1')
    p.add_observable('S1_scaled', 'scale*S1', '0.05', transformation='log10')
    rng = np.random.default_rng(1)
    for cond, x0 in [('full', 10.0), ('half', 5.0)]:
        p.add_measurements(cond, 'S1_obs', TIMES, s1(TIMES, x0) + noise * rng.standard_normal(TIMES.size))
        p.add_measurements(cond, 'S1_scaled', TIMES, 2.0 * s1(TIMES, x0))
    return p


TRUTH = np.array([np.log10(K1), np.log10(K2), np.log(2.0)])


def test_describe():
    p = problem()
    assert p.free_ids == ['k1', 'k2', 'scale']
    assert p.scales == ['log10', 'log10', 'log']
    lo, hi = p.bounds
    np.testing.assert_allclose(lo, [-3, -3, np.log(0.1)])
    assert p.least_squares
    assert p.to_linear(TRUTH) == pytest.approx({'k1': K1, 'k2': K2, 'scale': 2.0})
    np.testing.assert_allclose(p.to_scale({'k1': K1, 'k2': K2, 'scale': 2.0}), TRUTH)
    assert 'free parameters' in repr(p)


def test_objective_is_exact_at_the_truth():
    p = problem()
    p.rtol, p.atol = 1e-11, 1e-14
    np.testing.assert_allclose(p.simulate(TRUTH)[:8], s1(TIMES, 10.0), rtol=1e-8)
    r = p.residuals(TRUTH)
    assert np.max(np.abs(r)) < 1e-6
    # nllh = sum of the normalization terms when every residual is 0
    m = np.array([v for (_, o, _, v) in p._p.measurements])
    obs = np.array([o for (_, o, _, _) in p._p.measurements])
    const = np.where(obs == 'S1_obs', 0.5 * np.log(2 * np.pi * 0.01),
                     0.5 * np.log(2 * np.pi * 0.0025) + np.log(m * np.log(10)))
    assert p.objective(TRUTH) == pytest.approx(const.sum(), abs=1e-9)


def test_gradient_and_jacobian_match_finite_differences():
    p = problem(noise=0.05)
    p.rtol, p.atol = 1e-11, 1e-14
    th = np.array([0.0, -0.5, 0.3])
    f, g = p.gradient(th)
    assert f == p.objective(th)
    r, J = p.residual_jacobian(th)
    assert J.shape == (32, 3)
    for k in range(3):
        h = 1e-6
        e = np.eye(3)[k] * h
        fd = (p.objective(th + e) - p.objective(th - e)) / (2 * h)
        assert g[k] == pytest.approx(fd, rel=1e-5)
        fdr = (p.residuals(th + e) - p.residuals(th - e)) / (2 * h)
        np.testing.assert_allclose(J[:, k], fdr, rtol=1e-5, atol=1e-7)


def test_preequilibration_and_steady_state():
    # pre-equilibrated with X0 = 5 (S1 = 5 k1 / k2), then X0 = 10; time inf: the steady state 10 k1 / k2
    p = fit.Problem.from_antimony(MODEL)
    p.add_parameter('k1', 'log10', 1e-3, 1e3, 1.0)
    p.add_parameter('k2', 'log10', 1e-3, 1e3, 1.0)
    p.add_condition('full')
    p.add_condition('half', {'X0': 5})
    p.add_observable('y', 'S1', '0.1')
    times = np.array([0.0, 1.0, 4.0, np.inf])
    p.add_measurements('full', 'y', times, np.ones(4), preequilibration='half')
    p.rtol, p.atol = 1e-11, 1e-14
    th = np.array([np.log10(K1), np.log10(K2)])
    lo, hi = 5 * K1 / K2, 10 * K1 / K2
    want = hi + (lo - hi) * np.exp(-K2 * times[:3])
    np.testing.assert_allclose(p.simulate(th), list(want) + [hi], rtol=1e-8)
    f, g = p.gradient(th)
    for k in range(2):
        e = np.eye(2)[k] * 1e-6
        assert g[k] == pytest.approx((p.objective(th + e) - p.objective(th - e)) / 2e-6, rel=1e-5)
    pytest.importorskip('pandas')
    assert list(p.measurement_table()['preequilibrationConditionId']) == ['half'] * 4


@pytest.mark.parametrize('method', ['pso', 'de', 'shade'])
def test_global_then_local(method):
    p = problem()
    r = p.fit(n_starts=2, seed=1, global_method=method, max_evals=1500)
    np.testing.assert_allclose(r.best, TRUTH, atol=1e-5)
    g = r.global_stage
    assert g['method'] == method and g['nfev'] >= 1 and r.best_value <= g['best']
    r1 = p.fit(n_starts=2, seed=1, global_method=method, max_evals=1500, threads=1)
    assert r1.best_value == r.best_value and (r1.best == r.best).all()
    raw = p.fit(n_starts=2, seed=1, global_method=method, max_evals=1500, polish=False)
    assert raw.best_value == g['best'] and all(n == 0 for n in raw.nit)
    with pytest.raises(ValueError):
        p.fit(global_method='nope')


def test_multistart_recovers_and_is_deterministic():
    p = problem()
    calls = []
    r = p.fit(n_starts=8, seed=3, callback=lambda done, total, best: calls.append((done, total)) and False)
    assert len(calls) == 8 and calls[-1] == (8, 8)
    np.testing.assert_allclose(r.best, TRUTH, atol=1e-6)
    assert r.best_linear == pytest.approx({'k1': K1, 'k2': K2, 'scale': 2.0}, rel=1e-5)
    assert r.n_at_best() >= 6 and all(r.converged[:r.n_at_best()])
    assert list(r.waterfall()) == sorted(r.waterfall())
    r1 = p.fit(n_starts=8, seed=3, threads=1)
    assert r1.best.tobytes() == r.best.tobytes() and r1.best_value == r.best_value
    assert 'best objective' in repr(r)


def test_dataframes():
    pd = pytest.importorskip('pandas')
    p = problem(noise=0.05)
    r = p.fit(n_starts=4, seed=0)
    df = r.to_dataframe()
    assert list(df.columns[-3:]) == ['k1', 'k2', 'scale'] and len(df) == 4
    fi = p.fisher(r.best).to_dataframe()
    assert list(fi.index) == ['k1', 'k2', 'scale'] and fi['identifiable'].all()
    assert (fi['lower_linear'] < fi['estimate_linear']).all() and (fi['estimate_linear'] < fi['upper_linear']).all()
    mt = p.measurement_table(r.best)
    assert isinstance(mt, pd.DataFrame) and 'simulation' in mt and len(mt) == 32
    assert list(p.parameter_table()['parameterId']) == ['k1', 'k2', 'scale']


def test_fisher_intervals_and_identifiability():
    p = problem(noise=0.05)
    r = p.fit(n_starts=4, seed=0)
    f = p.fisher(r.best, level=0.95)
    assert all(f.identifiable)
    np.testing.assert_allclose(f.upper - f.x, 1.959963984540054 * f.sd)
    assert np.allclose(np.diag(f.corr), 1.0)
    # y = a * b * S1 with S1 proportional to k1: only the product a * b * k1 is determined (and k2, from the shape)
    q = fit.Problem.from_antimony(MODEL)
    for i in ['k1', 'k2', 'a', 'b']:
        q.add_parameter(i, 'log10', 1e-3, 1e3, 1)
    q.add_condition('full')
    q.add_observable('y', 'a*b*S1', '0.1')
    q.add_measurements('full', 'y', TIMES, s1(TIMES, 10.0))
    f = q.fisher(np.r_[TRUTH[:2], 0.2, -0.2])
    assert f.identifiable == [False, True, False, False]
    assert 'not identifiable' in repr(f)


def test_scipy_least_squares_agrees():
    """Hosts with their own optimizer: scipy driven by our residuals and exact Jacobian reaches the same optimum."""
    so = pytest.importorskip('scipy.optimize')
    p = problem(noise=0.05)
    ours = p.fit(n_starts=6, seed=2)
    lo, hi = p.bounds
    s = so.least_squares(p.residuals, p.nominal, jac=lambda x: p.residual_jacobian(x)[1], bounds=(lo, hi),
                         method='trf', x_scale='jac', xtol=1e-12, ftol=1e-12, gtol=1e-12)
    np.testing.assert_allclose(s.x, ours.best, atol=1e-5)
    assert p.objective(s.x) == pytest.approx(ours.best_value, abs=1e-8)


def test_cancel_and_callback_errors():
    p = problem()
    r = p.fit(n_starts=6, seed=0, threads=1, callback=lambda d, t, b: True)
    assert r.message.count('cancelled') == 5

    def boom(d, t, b):
        raise KeyError('from the callback')
    with pytest.raises(KeyError, match='from the callback'):
        p.fit(n_starts=3, threads=1, callback=boom)


def test_from_roadrunner_and_errors():
    rr = rn.RoadRunner(rn.core.antimony_to_sbml(MODEL))
    p = fit.Problem.from_roadrunner(rr)
    p.add_parameter('k1', 'log10', 1e-3, 1e3, 1.0)
    p.add_condition('full')
    p.add_observable('o', 'S1', '0.1')
    p.add_measurements('full', 'o', TIMES, s1(TIMES, 10.0, k2=0.2))
    assert p.free_ids == ['k1']
    with pytest.raises(ValueError, match='expected 1 parameter values'):
        p.objective([0.0, 1.0])
    with pytest.raises(ValueError, match='already added'):
        p.add_parameter('k1')
    p.add_observable('bad', 'nope*S1', '1')
    with pytest.raises(ValueError, match="unknown symbol 'nope'"):
        p.free_ids
    with pytest.raises(ValueError):
        fit.Problem('<not sbml')


def test_petab(tmp_path):
    (tmp_path / 'model.xml').write_text(rn.core.antimony_to_sbml(MODEL))
    (tmp_path / 'problem.yaml').write_text(
        'format_version: 1\nparameter_file: parameters.tsv\nproblems:\n- condition_files: [conditions.tsv]\n'
        '  measurement_files: [measurements.tsv]\n  observable_files: [observables.tsv]\n  sbml_files: [model.xml]\n')
    (tmp_path / 'parameters.tsv').write_text(
        'parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n'
        'k1\tlog10\t0.001\t1000\t1\t1\nk2\tlog10\t0.001\t1000\t1\t1\nsigma\tlin\t\t\t0.1\t0\n')
    (tmp_path / 'conditions.tsv').write_text('conditionId\tX0\nfull\t\nhalf\t5\n')
    (tmp_path / 'observables.tsv').write_text('observableId\tobservableFormula\tnoiseFormula\nS1_obs\tS1\tsigma\n')
    rows = ['observableId\tsimulationConditionId\ttime\tmeasurement']
    for cond, x0 in [('full', 10.0), ('half', 5.0)]:
        rows += [f'S1_obs\t{cond}\t{t}\t{float(v)!r}' for t, v in zip(TIMES, s1(TIMES, x0))]
    (tmp_path / 'measurements.tsv').write_text('\n'.join(rows) + '\n')
    p = fit.Problem.from_petab(tmp_path / 'problem.yaml')
    assert p.free_ids == ['k1', 'k2']
    r = p.fit(n_starts=5, seed=1)
    np.testing.assert_allclose(r.best, TRUTH[:2], atol=1e-6)
    (tmp_path / 'conditions.tsv').write_text('conditionId\tX0\nfull\t\nhalf\tnot_a_parameter\n')
    with pytest.raises(ValueError, match="'not_a_parameter' is not in the parameter table"):
        fit.Problem.from_petab(tmp_path / 'problem.yaml')


# ---------------------------------------------------------------- the short form: fit_timecourse

SIMPLE = 'J1: $X0 -> S1; k1*X0; J2: S1 -> ; k2*S1; X0 = 10; S1 = 0; k1 = 1; k2 = 1'
T10 = np.array([0, 2, 4, 6, 8, 10, 15, 20, 25, 30.])
Y10 = np.array([0.0, 7.0, 14.0, 17.2, 19.7, 21.5, 22.7, 24.4, 24.4, 26.6])


def test_fit_timecourse_matches_curve_fit():
    so = pytest.importorskip('scipy.optimize')
    r = fit.fit_timecourse(SIMPLE, {'time': T10, 'S1': Y10}, ['k1', 'k2'])
    f = lambda t, k1, k2: 10 * k1 / k2 * (1 - np.exp(-k2 * t))
    popt, pcov = so.curve_fit(f, T10, Y10, p0=[1, 1])
    k = np.array([r.values['k1'], r.values['k2']])
    np.testing.assert_allclose(k, popt, rtol=1e-5)
    # intervals from the error estimated from the residuals, as curve_fit's (absolute_sigma=False)
    np.testing.assert_allclose(r.fisher.sd * k * np.log(10), np.sqrt(np.diag(pcov)), rtol=1e-4)
    assert r.sigma_estimated and r.sigma == pytest.approx(np.sqrt(np.sum((f(T10, *popt) - Y10) ** 2) / 8), rel=1e-6)
    assert r.problem.scales == ['log10', 'log10']
    lo, hi = r.intervals['k1']
    assert lo < 0.5 < hi and 'interval' in repr(r)
    sim = np.asarray(r.simulate(end=30, points=31))
    np.testing.assert_allclose(sim[:, 1], f(sim[:, 0], *k), rtol=1e-5, atol=1e-6)
    # a global method first, then its 3 best points refined: the same fit
    for m in ['pso', 'de', 'shade']:
        rg = fit.fit_timecourse(SIMPLE, {'time': T10, 'S1': Y10}, ['k1', 'k2'], n_starts=3, global_method=m)
        np.testing.assert_allclose([rg.values['k1'], rg.values['k2']], popt, rtol=1e-5)
        assert rg.result.global_stage['method'] == m and len(rg.result.value) <= 3


def test_fit_timecourse_from_a_simulation_result():
    # data straight from roadrunner: '[S1]', '[S2]' columns, two species measured
    model = 'J1: $X0 -> S1; k1*X0; J2: S1 -> S2; k2*S1; J3: S2 -> ; k3*S2; X0 = 10; k1 = 0.5; k2 = 0.3; k3 = 0.1'
    r0 = rn.RoadRunner(rn.core.antimony_to_sbml(model))
    data = r0.simulate(0, 20, 11, ['time', '[S1]', '[S2]'])
    r = fit.fit_timecourse(model.replace('k1 = 0.5', 'k1 = 1').replace('k2 = 0.3', 'k2 = 1'), data, ['k1', 'k2'], sigma=0.1)
    assert r.values == pytest.approx({'k1': 0.5, 'k2': 0.3}, rel=1e-5)
    assert not r.sigma_estimated and r.sigma == 0.1
    # bounds given; a parameter that is 0 in the model needs them
    r = fit.fit_timecourse(model, data, {'k1': (0.01, 10), 'k3': (0.001, 1)})
    assert r.values == pytest.approx({'k1': 0.5, 'k3': 0.1}, rel=1e-5)
    with pytest.raises(ValueError, match='give one'):
        fit.fit_timecourse(model.replace('k3 = 0.1', 'k3 = 0'), data, ['k3'])
    with pytest.raises(ValueError, match='not in the model'):
        fit.fit_timecourse(model, data, ['nope'])
    with pytest.raises(ValueError, match='no time column'):
        fit.fit_timecourse(model, {'S1': [1, 2]}, ['k1'])


def test_add_parameter_picks_the_scale():
    p = fit.Problem.from_antimony(SIMPLE)
    p.add_parameter('k1', lower=0.001, upper=100)    # a wide positive range: log10
    p.add_parameter('k2', lower=0.1, upper=0.5)      # narrow: lin
    p.add_condition('c')
    p.add_observable('o', 'S1', '1')
    p.add_measurements('c', 'o', T10, Y10)
    assert p.scales == ['log10', 'lin']
