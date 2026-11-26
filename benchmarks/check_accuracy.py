"""Which engine is right? Default-tolerance runs vs tight-tolerance references from both engines.

Needs both engines in one Python: libroadrunner (pip install libroadrunner) and this package
(pip install . from the repository root). Run from this folder:
    python bench_stiff.py       # speed table
    python check_accuracy.py    # accuracy against rtol 1e-10 references from both engines
"""
import numpy as np, roadrunner, roadrunner_nitro as rn
from bench_stiff import MODELS

def run(mod, sbml, t0, t1, n, rtol, atol):
    r = mod.RoadRunner(sbml)
    if rtol is not None:
        if mod is rn:
            r.setIntegratorSetting('relative_tolerance', rtol); r.setIntegratorSetting('absolute_tolerance', atol)
        else:
            r.integrator.relative_tolerance = rtol; r.integrator.absolute_tolerance = atol
    y = r.simulate(t0, t1, n)
    y = y[0] if isinstance(y, tuple) else y
    return np.asarray(y), list(r.timeCourseSelections)

for name, ant, t0, t1, n in MODELS[1:]:
    sbml = rn.antimony_to_sbml(ant)
    ref_rr, cols = run(roadrunner, sbml, t0, t1, n, 1e-10, 1e-16)
    ref_n, _ = run(rn, sbml, t0, t1, n, 1e-10, 1e-16)
    d_rr, _ = run(roadrunner, sbml, t0, t1, n, None, None)
    d_n, _ = run(rn, sbml, t0, t1, n, None, None)
    sc = np.abs(ref_rr).max(axis=0) + 1e-300
    err = lambda y: np.abs(y - ref_rr) / sc
    print(f'\n{name}')
    print(f'  tight rr vs tight nitro : {err(ref_n).max():.2e}')
    for lab, y in (('default libroadrunner', d_rr), ('default nitro', d_n)):
        e = err(y); j = e.max(axis=0).argmax(); i = e[:, j].argmax()
        print(f'  {lab:22s}: max rel err {e.max():.2e} (median over columns {np.median(e.max(axis=0)):.1e}); '
              f'worst {cols[j]} at t={ref_rr[i,0]:.3g}: ref {ref_rr[i,j]:.4g}, got {y[i,j]:.4g}, column max {sc[j]:.3g}')
    # Species that stay far below the absolute tolerance (e.g. ~1e-123 at the end of the chain) are
    # noise in every solver: score again with each column's scale floored at 1e-6.
    sc6 = np.maximum(sc, 1e-6)
    for lab, y in (('tight nitro', ref_n), ('default libroadrunner', d_rr), ('default nitro', d_n)):
        print(f'  {lab:22s}: max rel err, scale floored at 1e-6: {(np.abs(y - ref_rr) / sc6).max():.2e}')
