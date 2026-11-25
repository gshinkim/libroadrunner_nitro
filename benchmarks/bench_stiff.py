"""Stiff-model speed: roadrunner_nitro vs libroadrunner, same SBML, default tolerances.

Needs both engines in one Python: libroadrunner (pip install libroadrunner) and this package
(pip install . from the repository root). Run from this folder:
    python bench_stiff.py       # speed table
    python check_accuracy.py    # accuracy against rtol 1e-10 references from both engines
"""
import time, statistics, random
import numpy as np
import roadrunner
import roadrunner_nitro as rn

ROBERTSON = '''
J1: A -> B; k1*A
J2: B + B -> B + C; k2*B*B
J3: B + C -> A + C; k3*B*C
A = 1; B = 0; C = 0
k1 = 0.04; k2 = 3e7; k3 = 1e4
'''

OREGONATOR = '''
// Field-Noyes Oregonator (Hairer & Wanner scaling), stiff relaxation oscillator
J1: -> X; s*(Y - X*Y + X - q*X*X)
J2: -> Y; (-Y - X*Y + Z)/s
J3: -> Z; w*(X - Z)
X = 1; Y = 2; Z = 3
s = 77.27; q = 8.375e-6; w = 0.161
'''

def chain(n, seed=1):
    rnd = random.Random(seed)
    lines = []
    for i in range(n):
        kf = 10 ** rnd.uniform(-2, 4); kr = 10 ** rnd.uniform(-2, 4)
        lines.append(f'J{i}: S{i} -> S{i+1}; kf{i}*S{i} - kr{i}*S{i+1}')
        lines.append(f'kf{i} = {kf:.6g}; kr{i} = {kr:.6g}')
    lines.append(f'J{n}: S{n} -> ; kd*S{n}/(Km + S{n}); kd = 5; Km = 0.1')
    lines.append(f'J{n+1}: -> S0; 1')
    lines += [f'S{i} = 0' for i in range(1, n + 1)] + ['S0 = 10']
    return '\n'.join(lines)

MODELS = [
    ('Robertson (3 sp)', ROBERTSON, 0, 1e5, 1001),
    ('Oregonator (3 sp)', OREGONATOR, 0, 360, 2001),
    ('Chain (201 sp)', chain(200), 0, 100, 1001),
]

def bench(make, t0, t1, n, reps):
    t = time.perf_counter(); r = make(); load = time.perf_counter() - t
    r.simulate(t0, t1, n)  # warm-up
    times = []
    for _ in range(reps):
        r.reset()
        t = time.perf_counter(); y = r.simulate(t0, t1, n); times.append(time.perf_counter() - t)
    return load, statistics.median(times), np.asarray(y)

if __name__ == '__main__':
  print(f'libroadrunner {roadrunner.__version__}; both at defaults (cvode, rtol 1e-6, atol 1e-12)\n')
  print(f'{"model":20s} {"load rr":>9s} {"load nitro":>10s} {"sim rr":>10s} {"sim nitro":>10s} {"speedup":>8s} {"max rel diff":>13s}')
  for name, ant, t0, t1, n in MODELS:
      sbml = rn.antimony_to_sbml(ant)
      reps = 20
      lr, sr, yr = bench(lambda: roadrunner.RoadRunner(sbml), t0, t1, n, reps)
      ln, sn, yn = bench(lambda: rn.RoadRunner(sbml), t0, t1, n, reps)
      yn = yn[0] if isinstance(yn, tuple) else yn
      scale = np.abs(yr).max(axis=0) + 1e-300
      diff = (np.abs(yr - yn) / scale).max()
      print(f'{name:20s} {lr*1e3:8.1f}ms {ln*1e3:9.1f}ms {sr*1e3:9.2f}ms {sn*1e3:9.2f}ms {sr/sn:7.1f}x {diff:13.2e}')
