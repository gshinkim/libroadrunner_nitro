//! Global, derivative-free optimizers (spec: "Optimizers", stage 2), from their published descriptions:
//! - particle swarm: SPSO-2011 (M. Clerc, "Standard Particle Swarm Optimisation", 2011/2012; M. Zambrano-Bigiarini,
//!   M. Clerc, R. Rojas, IEEE CEC 2013): adaptive random topology (each particle informs K = 3 random others, redrawn
//!   after an iteration without improvement), w = 1 / (2 ln 2), c = 1/2 + ln 2, the new velocity towards a random
//!   point in the hypersphere around the centre of gravity of the particle, its best and its informants' best
//!   (rotation invariant; uniform direction and uniform radius, as Clerc's reference code, by way of hydroPSO's
//!   port of it), positions confined to the box with the velocity halved and reversed;
//! - differential evolution: DE/rand/1/bin (R. Storn, K. Price, J. Global Optim. 11, 1997), F = 0.5, CR = 0.9;
//! - SHADE (R. Tanabe, A. Fukunaga, IEEE CEC 2013): current-to-pbest/1/bin with an archive of replaced parents and
//!   a history of H successful (CR, F) means, CR ~ N(M_CR, 0.1), F ~ Cauchy(M_F, 0.1), p ~ U[2/NP, 0.2].
//!
//! All three work in the box of the bounds scaled to the unit cube, need the objective only, and evaluate every
//! member of a generation in parallel. Updates are synchronous (a generation is made from the previous one; SPSO's
//! reference code is asynchronous), every member draws from its own seeded stream and the shared steps (topology,
//! archive) from one more, so a given seed gives the same result, to the last bit, whatever the number of threads.
use super::optimize::latin_hypercube;
use super::trf::Status;
use crate::model::Rng;
use rayon::prelude::*;

/// What a global method minimizes.
pub trait Cost: Sync {
    /// The objective at x (free parameters on their scales); None where it cannot be evaluated (worse than any
    /// value).
    fn f(&self, x: &[f64]) -> Option<f64>;
    /// After each generation: (evaluations so far, best value); true stops.
    fn cancel(&self, _nfev: usize, _best: f64) -> bool { false }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Global { Pso, De, Shade }

impl Global {
    pub fn parse(s: &str) -> Result<Global, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "pso" | "particle_swarm" => Ok(Global::Pso),
            "de" | "differential_evolution" => Ok(Global::De),
            "shade" => Ok(Global::Shade),
            _ => Err(format!("unknown global method '{s}' (pso, de, shade)")),
        }
    }
    pub fn name(self) -> &'static str { match self { Global::Pso => "pso", Global::De => "de", Global::Shade => "shade" } }
    /// Default number of members for d parameters.
    pub fn default_population(self, d: usize) -> usize {
        match self { Global::Pso => 40, Global::De => (10 * d).clamp(20, 200), Global::Shade => 100 }
    }
}

#[derive(Clone, Debug)]
pub struct GlobalOptions {
    /// members (0: the method's default: PSO 40, DE 10 per parameter within [20, 200], SHADE 100)
    pub population: usize,
    /// objective evaluations (0: 1000 per parameter, at least 10 000); the generation that reaches it completes
    pub max_evals: usize,
    /// converged when the members' values (PSO: their best values) have a standard deviation <= tol (1 + |mean|)
    pub tol: f64,
    /// also stop after this many generations in which the best value improved by less than 1e-9 (1 + |best|)
    pub stall: usize,
}

impl Default for GlobalOptions {
    fn default() -> Self { GlobalOptions { population: 0, max_evals: 0, tol: 1e-3, stall: 200 } }
}

#[derive(Clone, Debug)]
pub struct GlobalResult {
    pub x: Vec<f64>,
    pub f: f64,
    /// final members (PSO: their best points), best first, values with them (infinity: not evaluable)
    pub members: Vec<(Vec<f64>, f64)>,
    pub nfev: usize,
    pub ngen: usize,
    /// MaxEval (budget), Ftol (converged or stalled), Cancelled, or Failed (no member could be evaluated)
    pub status: Status,
}

/// Stream of member i (i = usize::MAX: the shared one).
fn stream(seed: u64, i: usize) -> Rng {
    Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i as u64).wrapping_add(1).wrapping_mul(0xD1B5_4A32_D192_ED03))
}

fn normal(r: &mut Rng) -> f64 { (-2.0 * r.unif().ln()).sqrt() * (2.0 * std::f64::consts::PI * r.unif()).cos() }
fn below(r: &mut Rng, n: usize) -> usize { (r.next_u64() % n as u64) as usize }

struct Ctx<'a> { cost: &'a dyn Cost, lb: &'a [f64], ub: &'a [f64] }

impl Ctx<'_> {
    fn x_of(&self, u: &[f64]) -> Vec<f64> { u.iter().enumerate().map(|(k, &v)| self.lb[k] + v * (self.ub[k] - self.lb[k])).collect() }
    /// Values of unit-cube points, in parallel.
    fn eval(&self, us: &[Vec<f64>]) -> Vec<f64> {
        us.par_iter().map(|u| self.cost.f(&self.x_of(u)).filter(|v| !v.is_nan()).unwrap_or(f64::INFINITY)).collect()
    }
}

/// Index of the best value (ties: the lowest index).
fn argmin(f: &[f64]) -> usize { (0..f.len()).fold(0, |b, i| if f[i] < f[b] { i } else { b }) }

/// Stopping rule after a generation: Some(status) to stop.
struct Stop { max_evals: usize, tol: f64, stall: usize, last: f64, quiet: usize }

impl Stop {
    fn check(&mut self, cost: &dyn Cost, nfev: usize, vals: &[f64]) -> Option<Status> {
        let best = vals.iter().copied().fold(f64::INFINITY, f64::min);
        if best < self.last - 1e-9 * (1.0 + best.abs()) { self.quiet = 0; } else { self.quiet += 1; }
        self.last = self.last.min(best);
        if cost.cancel(nfev, best) { return Some(Status::Cancelled); }
        if nfev >= self.max_evals { return Some(Status::MaxEval); }
        if vals.iter().all(|v| v.is_finite()) {
            let n = vals.len() as f64;
            let mean = vals.iter().sum::<f64>() / n;
            let sd = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
            if sd <= self.tol * (1.0 + mean.abs()) { return Some(Status::Ftol); }
        }
        if self.quiet >= self.stall { return Some(Status::Ftol); }
        None
    }
}

/// Minimize `cost` over the box [lb, ub] (finite) with method `g`. `init`: points (on the parameter scales) to
/// place among the first members; the rest is a Latin hypercube sample.
pub fn minimize(cost: &dyn Cost, g: Global, lb: &[f64], ub: &[f64], init: &[Vec<f64>], seed: u64, o: &GlobalOptions)
                -> Result<GlobalResult, String> {
    let d = lb.len();
    if d == 0 { return Err("no parameter to estimate".into()); }
    if lb.iter().zip(ub).any(|(a, b)| !a.is_finite() || !b.is_finite() || !(a <= b)) {
        return Err("global methods need finite bounds (lower <= upper) for every estimated parameter".into());
    }
    let np = if o.population == 0 { g.default_population(d) } else { o.population };
    let min_np = match g { Global::Pso => 2, Global::De => 4, Global::Shade => 4 };
    if np < min_np { return Err(format!("{} needs a population of at least {min_np}", g.name())); }
    let max_evals = if o.max_evals == 0 { (1000 * d).max(10_000) } else { o.max_evals };
    let unit = vec![0.0; d];
    let one = vec![1.0; d];
    let mut pop = latin_hypercube(np, &unit, &one, seed);
    for (k, x) in init.iter().take(np).enumerate() {
        if x.len() != d { return Err(format!("a start point has {} values, the problem {d} free parameters", x.len())); }
        pop[k] = x.iter().enumerate().map(|(j, &v)| if ub[j] > lb[j] { ((v - lb[j]) / (ub[j] - lb[j])).clamp(0.0, 1.0) } else { 0.0 }).collect();
    }
    let ctx = Ctx { cost, lb, ub };
    let mut stop = Stop { max_evals, tol: o.tol, stall: o.stall.max(1), last: f64::INFINITY, quiet: 0 };
    let (members, nfev, ngen, status) = match g {
        Global::Pso => pso(&ctx, pop, seed, &mut stop),
        Global::De => de(&ctx, pop, seed, &mut stop),
        Global::Shade => shade(&ctx, pop, seed, &mut stop),
    };
    let mut order: Vec<usize> = (0..members.len()).collect();
    order.sort_by(|&a, &b| members[a].1.total_cmp(&members[b].1).then(a.cmp(&b)));
    let members: Vec<(Vec<f64>, f64)> = order.into_iter().map(|i| (ctx.x_of(&members[i].0), members[i].1)).collect();
    let status = if members[0].1.is_finite() { status } else { Status::Failed };
    Ok(GlobalResult { x: members[0].0.clone(), f: members[0].1, members, nfev, ngen, status })
}

type Run = (Vec<(Vec<f64>, f64)>, usize, usize, Status);

/// SPSO-2011, synchronous.
fn pso(ctx: &Ctx, mut x: Vec<Vec<f64>>, seed: u64, stop: &mut Stop) -> Run {
    const K: usize = 3;
    let (s, d) = (x.len(), x[0].len());
    let w = 1.0 / (2.0 * std::f64::consts::LN_2);
    let c = 0.5 + std::f64::consts::LN_2;
    let mut rngs: Vec<Rng> = (0..s).map(|i| stream(seed, i)).collect();
    let mut shared = stream(seed, usize::MAX);
    let mut v: Vec<Vec<f64>> = x.iter().zip(&mut rngs).map(|(xi, r)| xi.iter().map(|&xd| -xd + r.unif() * 1.0).collect()).collect();
    // v_d = U(lo - x_d, hi - x_d) with lo = 0, hi = 1: -x_d + U(0, 1)
    let mut fx = ctx.eval(&x);
    let mut nfev = s;
    let mut p = x.clone();
    let mut fp = fx.clone();
    // informants of each particle: itself and those that chose it
    let topology = |r: &mut Rng| -> Vec<Vec<usize>> {
        let mut inf: Vec<Vec<usize>> = (0..s).map(|j| vec![j]).collect();
        for i in 0..s { for _ in 0..K { let j = below(r, s); if !inf[j].contains(&i) { inf[j].push(i); } } }
        inf
    };
    let mut links = topology(&mut shared);
    let mut gbest = fp[argmin(&fp)];
    let mut ngen = 0;
    if let Some(st) = stop.check(ctx.cost, nfev, &fp) { return (p.into_iter().zip(fp).collect(), nfev, ngen, st); }
    loop {
        ngen += 1;
        // best informant of each particle (from the previous generation)
        let l: Vec<usize> = links.iter().map(|inf| inf.iter().copied().fold(inf[0], |b, j| if fp[j] < fp[b] || (fp[j] == fp[b] && j < b) { j } else { b })).collect();
        x.par_iter_mut().zip(v.par_iter_mut()).zip(rngs.par_iter_mut()).enumerate().for_each(|(i, ((xi, vi), r))| {
            let (pi, li) = (&p[i], &p[l[i]]);
            let gc: Vec<f64> = (0..d).map(|k| if l[i] != i { xi[k] + c * (pi[k] + li[k] - 2.0 * xi[k]) / 3.0 } else { xi[k] + c * (pi[k] - xi[k]) / 2.0 }).collect();
            let rad = (0..d).map(|k| (gc[k] - xi[k]).powi(2)).sum::<f64>().sqrt();
            // Clerc's alea_sphere: a uniform direction (normalised normal draws) and a uniform radius rad U, so points
            // gather near G (uniform in volume, rad U^(1/d), puts them near the surface in high dimension, and the
            // swarm does not contract: on a 10-dimensional sphere it stalls near 1)
            let dir: Vec<f64> = (0..d).map(|_| normal(r)).collect();
            let norm = dir.iter().map(|z| z * z).sum::<f64>().sqrt();
            let rr = rad * r.unif();
            for k in 0..d {
                let xp = gc[k] + if norm > 0.0 { rr * dir[k] / norm } else { 0.0 };
                vi[k] = w * vi[k] + xp - xi[k];
                xi[k] += vi[k];
                if xi[k] < 0.0 { xi[k] = 0.0; vi[k] *= -0.5; } else if xi[k] > 1.0 { xi[k] = 1.0; vi[k] *= -0.5; }
            }
        });
        fx = ctx.eval(&x);
        nfev += s;
        for i in 0..s { if fx[i] < fp[i] { fp[i] = fx[i]; p[i].clone_from(&x[i]); } }
        let b = fp[argmin(&fp)];
        if !(b < gbest) { links = topology(&mut shared); }
        gbest = gbest.min(b);
        if let Some(st) = stop.check(ctx.cost, nfev, &fp) { return (p.into_iter().zip(fp).collect(), nfev, ngen, st); }
    }
}

/// Binomial crossover of target t with mutant v (at least one coordinate, jr, from v); a mutant coordinate outside
/// [0, 1] goes halfway between the target's and the bound.
fn crossover(t: &[f64], v: &[f64], cr: f64, r: &mut Rng) -> Vec<f64> {
    let d = t.len();
    let jr = below(r, d);
    (0..d).map(|j| {
        if j == jr || r.unif() <= cr {
            let y = v[j];
            if y < 0.0 { t[j] / 2.0 } else if y > 1.0 { (t[j] + 1.0) / 2.0 } else { y }
        } else { t[j] }
    }).collect()
}

/// Three distinct indices below n, none equal to i.
fn three(r: &mut Rng, n: usize, i: usize) -> (usize, usize, usize) {
    let mut pick = |not: &[usize]| loop { let k = below(r, n); if !not.contains(&k) { break k; } };
    let a = pick(&[i]);
    let b = pick(&[i, a]);
    let c = pick(&[i, a, b]);
    (a, b, c)
}

/// DE/rand/1/bin, F = 0.5, CR = 0.9.
fn de(ctx: &Ctx, mut x: Vec<Vec<f64>>, seed: u64, stop: &mut Stop) -> Run {
    const F: f64 = 0.5;
    const CR: f64 = 0.9;
    let n = x.len();
    let d = x[0].len();
    let mut rngs: Vec<Rng> = (0..n).map(|i| stream(seed, i)).collect();
    let mut fx = ctx.eval(&x);
    let mut nfev = n;
    let mut ngen = 0;
    if let Some(st) = stop.check(ctx.cost, nfev, &fx) { return (x.into_iter().zip(fx).collect(), nfev, ngen, st); }
    loop {
        ngen += 1;
        let trial: Vec<Vec<f64>> = rngs.par_iter_mut().enumerate().map(|(i, r)| {
            let (a, b, c) = three(r, n, i);
            let v: Vec<f64> = (0..d).map(|k| x[a][k] + F * (x[b][k] - x[c][k])).collect();
            crossover(&x[i], &v, CR, r)
        }).collect();
        let ft = ctx.eval(&trial);
        nfev += n;
        for (i, (t, f)) in trial.into_iter().zip(ft).enumerate() { if f <= fx[i] { x[i] = t; fx[i] = f; } }
        if let Some(st) = stop.check(ctx.cost, nfev, &fx) { return (x.into_iter().zip(fx).collect(), nfev, ngen, st); }
    }
}

/// SHADE (history size H = NP, archive size NP).
fn shade(ctx: &Ctx, mut x: Vec<Vec<f64>>, seed: u64, stop: &mut Stop) -> Run {
    let n = x.len();
    let d = x[0].len();
    let h = n;
    let mut mcr = vec![0.5; h];
    let mut mf = vec![0.5; h];
    let mut kmem = 0;
    let mut archive: Vec<Vec<f64>> = vec![];
    let mut rngs: Vec<Rng> = (0..n).map(|i| stream(seed, i)).collect();
    let mut shared = stream(seed, usize::MAX);
    let mut fx = ctx.eval(&x);
    let mut nfev = n;
    let mut ngen = 0;
    if let Some(st) = stop.check(ctx.cost, nfev, &fx) { return (x.into_iter().zip(fx).collect(), nfev, ngen, st); }
    loop {
        ngen += 1;
        let mut rank: Vec<usize> = (0..n).collect();
        rank.sort_by(|&a, &b| fx[a].total_cmp(&fx[b]).then(a.cmp(&b)));
        let pmin = 2.0 / n as f64;
        // per member: (trial, CR, F)
        let gen: Vec<(Vec<f64>, f64, f64)> = rngs.par_iter_mut().enumerate().map(|(i, r)| {
            let ri = below(r, h);
            let cr = (mcr[ri] + 0.1 * normal(r)).clamp(0.0, 1.0);
            let f = loop {
                let f = mf[ri] + 0.1 * (std::f64::consts::PI * (r.unif() - 0.5)).tan();
                if f > 0.0 { break f.min(1.0); }
            };
            let pi = pmin + r.unif() * (0.2 - pmin).max(0.0);
            let top = ((pi * n as f64).round() as usize).clamp(1, n);
            let pbest = rank[below(r, top)];
            let r1 = loop { let k = below(r, n); if k != i { break k; } };
            let r2 = loop {
                let k = below(r, n + archive.len());
                if k != i && k != r1 { break k; }
            };
            let x2 = if r2 < n { &x[r2] } else { &archive[r2 - n] };
            let v: Vec<f64> = (0..d).map(|k| x[i][k] + f * (x[pbest][k] - x[i][k]) + f * (x[r1][k] - x2[k])).collect();
            (crossover(&x[i], &v, cr, r), cr, f)
        }).collect();
        let trial: Vec<Vec<f64>> = gen.iter().map(|g| g.0.clone()).collect();
        let ft = ctx.eval(&trial);
        nfev += n;
        let (mut scr, mut sf, mut wd) = (vec![], vec![], vec![]);
        for (i, ((t, cr, f), v)) in gen.into_iter().zip(ft).enumerate() {
            if v <= fx[i] {
                if v < fx[i] {
                    archive.push(std::mem::replace(&mut x[i], t));
                    scr.push(cr);
                    sf.push(f);
                    wd.push(if fx[i].is_finite() { fx[i] - v } else { 1.0 });
                } else {
                    x[i] = t;
                }
                fx[i] = v;
            }
        }
        while archive.len() > n { let k = below(&mut shared, archive.len()); archive.swap_remove(k); }
        if !scr.is_empty() {
            let sw: f64 = wd.iter().sum();
            let w: Vec<f64> = wd.iter().map(|x| if sw > 0.0 { x / sw } else { 1.0 / wd.len() as f64 }).collect();
            mcr[kmem] = w.iter().zip(&scr).map(|(a, b)| a * b).sum();
            let num: f64 = w.iter().zip(&sf).map(|(a, b)| a * b * b).sum();
            let den: f64 = w.iter().zip(&sf).map(|(a, b)| a * b).sum();
            if den > 0.0 { mf[kmem] = num / den; }
            kmem = (kmem + 1) % h;
        }
        if let Some(st) = stop.check(ctx.cost, nfev, &fx) { return (x.into_iter().zip(fx).collect(), nfev, ngen, st); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct F<G: Fn(&[f64]) -> f64 + Sync>(G);
    impl<G: Fn(&[f64]) -> f64 + Sync> Cost for F<G> { fn f(&self, x: &[f64]) -> Option<f64> { Some((self.0)(x)) } }

    fn rastrigin(x: &[f64]) -> f64 { 10.0 * x.len() as f64 + x.iter().map(|v| v * v - 10.0 * (2.0 * std::f64::consts::PI * v).cos()).sum::<f64>() }
    fn rosenbrock(x: &[f64]) -> f64 { x.windows(2).map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2)).sum() }

    #[test]
    fn global_minima() {
        // Rosenbrock (a curved valley) in 4 dimensions for every method; Rastrigin (many local minima) for the two DE
        // variants. SPSO-2011 often ends in one of Rastrigin's local minima (as published), so it is not tested there.
        let o = GlobalOptions { max_evals: 60_000, tol: 1e-10, ..Default::default() };
        for g in [Global::Pso, Global::De, Global::Shade] {
            let r = minimize(&F(rosenbrock), g, &[-5.0; 4], &[5.0; 4], &[], 1, &o).unwrap();
            assert!(r.f < 1e-2, "{}: rosenbrock {} at {:?}", g.name(), r.f, r.x);
        }
        for g in [Global::De, Global::Shade] {
            let r = minimize(&F(rastrigin), g, &[-5.12; 4], &[5.12; 4], &[], 1, &o).unwrap();
            assert!(r.f < 1e-3, "{}: rastrigin {} at {:?}", g.name(), r.f, r.x);
        }
    }

    #[test]
    #[ignore]
    fn survey() {
        fn sphere(x: &[f64]) -> f64 { x.iter().map(|v| v * v).sum() }
        let fs: [(&str, fn(&[f64]) -> f64, f64); 3] = [("sphere", sphere, 5.0), ("rastrigin", rastrigin, 5.12), ("rosenbrock", rosenbrock, 5.0)];
        for g in [Global::Pso, Global::De, Global::Shade] {
            for (name, f, b) in fs {
                for d in [4, 10] {
                    let o = GlobalOptions { max_evals: 2000 * d * 10, tol: 1e-12, stall: 1000, ..Default::default() };
                    let r: Vec<String> = (0..5).map(|s| { let r = minimize(&F(f), g, &vec![-b; d], &vec![b; d], &[], s, &o).unwrap(); format!("{:.1e}/{}/{:?}", r.f, r.nfev, r.status) }).collect();
                    println!("{:5} {name:10} d={d:2}: {}", g.name(), r.join("  "));
                }
            }
        }
    }

    #[test]
    fn reproducible_across_threads() {
        for g in [Global::Pso, Global::De, Global::Shade] {
            let o = GlobalOptions { max_evals: 3000, ..Default::default() };
            let run = |t: usize| rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap()
                .install(|| minimize(&F(rastrigin), g, &[-5.12; 3], &[5.12; 3], &[], 7, &o).unwrap());
            let (a, b) = (run(1), run(4));
            assert_eq!(a.f.to_bits(), b.f.to_bits(), "{}", g.name());
            assert_eq!(a.x, b.x);
            assert_eq!((a.nfev, a.ngen), (b.nfev, b.ngen));
        }
    }

    #[test]
    fn failures_and_bounds() {
        // a region that cannot be evaluated, and a minimum on the boundary
        struct Half;
        impl Cost for Half { fn f(&self, x: &[f64]) -> Option<f64> { (x[0] > -1.0).then(|| (x[0] - 2.0).powi(2) + x[1] * x[1]) } }
        for g in [Global::Pso, Global::De, Global::Shade] {
            let o = GlobalOptions { tol: 1e-12, ..Default::default() };
            let r = minimize(&Half, g, &[-3.0, -1.0], &[2.0, 1.0], &[], 3, &o).unwrap();
            assert!((r.x[0] - 2.0).abs() < 1e-3 && r.x[1].abs() < 1e-3 && r.f < 1e-6, "{}: {:?} {}", g.name(), r.x, r.f);
            assert!(r.x[0] <= 2.0);
        }
    }
}
