"""COPASI time courses (0..100, 101 points, concentrations by SBML id) -> JSON {model: {cols, data} | 'error'}."""
import json, sys, warnings
warnings.filterwarnings('ignore')
import basico
out = {}
for m in sys.argv[2:]:
    try:
        dm = basico.load_model('D:/Documents/Rust/libroadrunner_nitro/tests/models/biomodels/%s.xml' % m)
        df = basico.run_time_course(duration=100, intervals=100, r_tol=1e-10, a_tol=1e-14, max_steps=10**7,
                                    use_sbml_id=True, model=dm)
        sp = basico.get_species(model=dm)
        cols = ['time'] + ['[%s]' % i for i in sp['sbml_id'] if i in df.columns and sp.loc[sp['sbml_id'] == i, 'type'].iloc[0] in ('reactions', 'ode')]
        ids = [c[1:-1] for c in cols[1:]]
        data = [[float(t)] + [float(df.loc[t, i]) for i in ids] for t in df.index]
        out[m] = {'cols': cols, 'data': data}
        basico.remove_datamodel(dm)
    except Exception as e:
        out[m] = 'error: %s' % str(e)[:200]
    print(m, 'ok' if isinstance(out[m], dict) else out[m], flush=True)
json.dump(out, open(sys.argv[1], 'w'))
