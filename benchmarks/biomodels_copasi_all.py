"""Driver: COPASI time courses for many models, one subprocess per model (timeout), 6 at once.
    python copasi_all.py out.json model..."""
import concurrent.futures, json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
def one(m):
    f = os.path.join(tempfile.gettempdir(), 'cop_%s.json' % m)
    try:
        subprocess.run([sys.executable, os.path.join(HERE, 'biomodels_copasi_tc.py'), f, m], capture_output=True, timeout=300)
        return json.load(open(f))[m]
    except subprocess.TimeoutExpired:
        return 'error: timeout'
    except Exception as e:
        return 'error: %s' % e
ms = sys.argv[2:]
with concurrent.futures.ThreadPoolExecutor(6) as p:
    res = dict(zip(ms, p.map(one, ms)))
json.dump(res, open(sys.argv[1], 'w'))
print('done', sum(isinstance(v, dict) for v in res.values()), 'of', len(ms))
