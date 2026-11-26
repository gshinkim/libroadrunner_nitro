"""Reference time courses for tests/test_biomodels_regression.py -> tests/reference/biomodels_regression.npz.

Each model's reference comes from the engine listed in SOURCE: COPASI (rtol 1e-10, atol 1e-14) where it
simulates the model correctly, libroadrunner 2.11 where COPASI does not (it steps over the infusion windows of
BIOMD0000000879 and stops early on 722 and 859), at rtol 1e-8, atol 1e-12: at 1e-10 / 1e-14 CVODE chatters on
the rate-law switch of 722 and runs out of steps. Run once with each engine; every run
updates the entries of the engine it has:

    python tests/make_biomodels_regression.py copasi          (needs copasi-basico; PyPI `basico` is unrelated)
    python tests/make_biomodels_regression.py libroadrunner   (needs libroadrunner)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, 'models', 'biomodels_regression')
OUT = os.path.join(HERE, 'reference', 'biomodels_regression.npz')
T_END, POINTS = 100.0, 101

# models with no reference: neither COPASI ("invalid state" at t = 40) nor libroadrunner (CVODE convergence)
# simulates them; the test checks that roadrunner_nitro runs them to the end
RUN_ONLY = {
    'BIOMD0000000659': 'NaN Jacobian entry from an invalid trial state (x^1.4 at x < 0) must stay NaN',
}

# model -> (reference engine, what it guards)
SOURCE = {
    'BIOMD0000000053': ('copasi', 'infinite Jacobian entry (fractional power at 0)'),
    'BIOMD0000000246': ('copasi', 'BDF with a huge stale Jacobian entry: VIP^0.2 from VIP = 0'),
    'BIOMD0000000451': ('copasi', 'infinite Jacobian; libroadrunner fails (CVODE convergence)'),
    'BIOMD0000000466': ('copasi', 'infinite Jacobian entry'),
    'BIOMD0000000719': ('copasi', 'infinite Jacobian entry (0 * inf = NaN at 0)'),
    'BIOMD0000000873': ('copasi', 'infinite Jacobian entry'),
    'BIOMD0000000952': ('copasi', 'infinite Jacobian entry'),
    'BIOMD0000000749': ('copasi', 'tanh past |x| = 709.78'),
    'BIOMD0000000337': ('copasi', 'output at an event time reports the post-event state'),
    'BIOMD0000000439': ('copasi', 'output at an event time reports the post-event state'),
    'BIOMD0000001019': ('copasi', 'output at an event time (persistent=false, fromTrigger=false)'),
    'BIOMD0000000816': ('copasi', 'time == t triggers'),
    'BIOMD0000000828': ('copasi', '(time % 12) == 0 and time == t triggers'),
    'BIOMD0000000928': ('copasi', 'time == 20 trigger'),
    'BIOMD0000000162': ('copasi', '5 ms pulse windows in rate laws'),
    'BIOMD0000000711': ('copasi', 'event reset leaving its trigger on the boundary (V < 0: V = 0)'),
    'BIOMD0000000879': ('libroadrunner', 'infusion windows (time >= 21) at atol 1e-12; COPASI misses them'),
    'BIOMD0000000722': ('libroadrunner', 'state switch in a rate law at atol 1e-12; COPASI stops early'),
    'BIOMD0000000859': ('libroadrunner', 'time % 24 switch at atol 1e-12; COPASI stops early'),
}


def copasi(path):
    import basico
    dm = basico.load_model(path)
    df = basico.run_time_course(duration=T_END, intervals=POINTS - 1, r_tol=1e-10, a_tol=1e-14, max_steps=10 ** 7,
                                use_sbml_id=True, model=dm)
    sp = basico.get_species(model=dm)
    ids = [i for i, kind in zip(sp['sbml_id'], sp['type']) if kind in ('reactions', 'ode') and i in df.columns]
    data = np.array([[t] + [df.loc[t, i] for i in ids] for t in df.index], float)
    basico.remove_datamodel(dm)
    return ['time'] + ['[%s]' % i for i in ids], data


def libroadrunner(path):
    import roadrunner
    r = roadrunner.RoadRunner(path)
    r.integrator.relative_tolerance = 1e-8
    r.integrator.absolute_tolerance = 1e-12
    cols = ['time'] + ['[%s]' % s for s in r.model.getFloatingSpeciesIds()]
    return cols, np.asarray(r.simulate(0, T_END, POINTS, cols), float)


def main(engine):
    ref = dict(np.load(OUT, allow_pickle=False)) if os.path.exists(OUT) else {}
    for m, (src, _) in SOURCE.items():
        if src != engine:
            continue
        cols, data = (copasi if engine == 'copasi' else libroadrunner)(os.path.join(MODELS, m + '.xml'))
        assert data.shape == (POINTS, len(cols)), (m, data.shape)
        ref[m + '/cols'] = np.array(cols)
        ref[m + '/data'] = data
        ref[m + '/source'] = np.array(engine)
        print(m, engine, data.shape)
    np.savez_compressed(OUT, **ref)


if __name__ == '__main__':
    main(sys.argv[1])
