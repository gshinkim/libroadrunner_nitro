"""SBML Test Suite (semantic time courses) on roadrunner_nitro and on libroadrunner, side by side, judged against the
suite's expected results. Each engine runs in worker processes, batches of cases each, so a crash in one case
(libroadrunner can segfault) costs only its batch, which is then rerun case by case.

    python benchmarks/sbml_test_suite.py                  # both engines; needs libroadrunner installed
    python benchmarks/sbml_test_suite.py --engine nitro   # one engine
    python benchmarks/sbml_test_suite.py --cases 00001 00002

Suite location: tests/models/sbml-test-suite/cases/semantic, or SBML_SEMANTIC_DIR (tests/models/README.md).
Per-case results go to benchmarks/sbml_test_suite_<engine>.json.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, '..', 'tests')
sys.path.insert(0, TESTS)
from sts_cases import case_ids  # noqa: E402

BATCH = 50


def worker(engine, ids):
    """Run cases ids in this process; one JSON line per case on stdout."""
    if engine == 'nitro':
        from roadrunner_nitro import RoadRunner
    else:
        from roadrunner import RoadRunner
    from sts_cases import run_case
    for cid in ids:
        print(json.dumps(run_case(RoadRunner, cid)), flush=True)


def run_batch(engine, ids, timeout):
    """Run ids in a worker process; cases it did not report (crash or hang) are rerun one by one."""
    cmd = [sys.executable, os.path.abspath(__file__), '--worker', engine, *ids]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout * len(ids))
        out = p.stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or '')
    got = {}
    for line in out.splitlines():
        if line.startswith('{'):
            rec = json.loads(line)
            got[rec['id']] = rec
    missing = [c for c in ids if c not in got]
    if len(ids) == 1 and missing:
        return [dict(id=ids[0], status='crash', reason='worker died or timed out')]
    for c in missing:
        got.update({r['id']: r for r in run_batch(engine, [c], timeout)})
    return [got[c] for c in ids]


def run_engine(engine, ids, timeout):
    results = []
    for k in range(0, len(ids), BATCH):
        results += run_batch(engine, ids[k:k + BATCH], timeout)
        print('  %s: %d / %d' % (engine, len(results), len(ids)), end='\r', flush=True)
    print()
    with open(os.path.join(HERE, 'sbml_test_suite_%s.json' % engine), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=0)
    return {r['id']: r for r in results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=['nitro', 'libroadrunner', 'both'], default='both')
    ap.add_argument('--cases', nargs='*')
    ap.add_argument('--timeout', type=float, default=60.0, help='seconds per case')
    ap.add_argument('--worker', nargs='+')
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker[0], a.worker[1:])
    ids = a.cases or case_ids()
    if not ids:
        sys.exit('SBML Test Suite not found: see tests/models/README.md')
    engines = ['nitro', 'libroadrunner'] if a.engine == 'both' else [a.engine]
    res = {e: run_engine(e, ids, a.timeout) for e in engines}
    print('\n%-14s %6s %6s %6s %6s %6s' % ('engine', 'pass', 'fail', 'error', 'crash', 'skip'))
    for e in engines:
        c = Counter(r['status'] for r in res[e].values())
        print('%-14s %6d %6d %6d %6d %6d' % (e, c['pass'], c['fail'], c['error'], c['crash'], c['skip']))
    if len(engines) == 2:
        n, l = res['nitro'], res['libroadrunner']
        only_l = sorted(c for c in ids if l[c]['status'] == 'pass' and n[c]['status'] != 'pass')
        only_n = sorted(c for c in ids if n[c]['status'] == 'pass' and l[c]['status'] != 'pass')
        print('\npassed by libroadrunner only: %d' % len(only_l))
        for c in only_l:
            print('  %s  nitro %s: %s' % (c, n[c]['status'], n[c].get('reason', '')[:110]))
        print('passed by roadrunner_nitro only: %d' % len(only_n))


if __name__ == '__main__':
    main()
