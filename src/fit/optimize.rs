//! Multistart local fitting, global then local, and Fisher-information intervals (spec: "Optimizers").
use super::global::{minimize, Cost, Global, GlobalOptions};
use super::problem::Problem;
use super::lbfgsb::{lbfgsb, LbfgsbOptions, LbfgsbStatus, Objective};
use super::trf::{trf, Residuals, Status, TrfOptions};
use crate::model::Rng;
use nalgebra::{DMatrix, SymmetricEigen};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug)]
pub struct FitOptions {
    pub n_starts: usize,
    pub seed: u64,
    pub method: Method,
    pub trf: TrfOptions,
    pub lbfgsb: LbfgsbOptions,
    /// a first start given on the parameter scale (the others are Latin hypercube samples)
    pub x0: Option<Vec<f64>>,
    /// worker threads (0: all cores)
    pub threads: usize,
    /// a global method run first; its best distinct members (n_starts of them) are the local starts
    pub global: Option<Global>,
    pub global_opts: GlobalOptions,
    /// with a global method: polish its best members with the local method (false: report them as they are)
    pub polish: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions { n_starts: 10, seed: 0, method: Method::Auto, trf: TrfOptions::default(), lbfgsb: LbfgsbOptions::default(), x0: None, threads: 0,
                     global: None, global_opts: GlobalOptions::default(), polish: true }
    }
}

/// The local optimizer of each start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// trust-region least squares for least-squares problems, L-BFGS-B otherwise
    Auto,
    /// trust-region reflective least squares (least-squares problems only)
    LeastSquares,
    /// L-BFGS-B on the negative log-likelihood and its gradient (any problem)
    Lbfgsb,
}

impl Method {
    pub fn parse(s: &str) -> Result<Method, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Method::Auto),
            "least_squares" | "trf" => Ok(Method::LeastSquares),
            "lbfgsb" | "l-bfgs-b" => Ok(Method::Lbfgsb),
            _ => Err(format!("unknown method '{s}' (auto, least_squares, lbfgsb)")),
        }
    }
}

struct Nllh<'a> { p: &'a Problem, stop: &'a AtomicBool }

impl Objective for Nllh<'_> {
    fn eval(&self, x: &[f64]) -> Option<(f64, Vec<f64>)> { self.p.evaluate(x, true, false).ok().map(|e| (e.nllh, e.grad)) }
    fn cancel(&self, _nit: usize, _f: f64) -> bool { self.stop.load(Ordering::Relaxed) }
}

#[derive(Clone, Debug)]
pub struct StartResult {
    /// index of the start (its place in the sample)
    pub start: usize,
    pub x0: Vec<f64>,
    /// final point, free parameters on their scales
    pub x: Vec<f64>,
    /// negative log-likelihood at x (infinity if the start failed)
    pub nllh: f64,
    pub nfev: usize,
    pub njev: usize,
    pub nit: usize,
    pub status: Status,
}

#[derive(Clone, Debug)]
pub struct FitResult {
    /// every start, best (lowest nllh) first; ties keep start order
    pub starts: Vec<StartResult>,
    pub ids: Vec<String>,
    /// the global stage, if one ran
    pub global: Option<GlobalSummary>,
}

/// The global stage of a fit: its best value, objective evaluations, generations and why it stopped.
#[derive(Clone, Debug)]
pub struct GlobalSummary { pub method: Global, pub best: f64, pub nfev: usize, pub ngen: usize, pub status: Status }

struct NllhOnly<'a> { p: &'a Problem, stop: &'a AtomicBool, progress: Option<Progress<'a>>, budget: usize }

impl Cost for NllhOnly<'_> {
    fn f(&self, x: &[f64]) -> Option<f64> { self.p.evaluate(x, false, false).ok().map(|e| e.nllh) }
    fn cancel(&self, nfev: usize, best: f64) -> bool {
        // the generation that reaches the budget completes: report at most the budget
        if let Some(cb) = self.progress { if cb(nfev.min(self.budget), self.budget, best) { self.stop.store(true, Ordering::Relaxed); } }
        self.stop.load(Ordering::Relaxed)
    }
}

/// Up to n members, best first, each at least `sep` (a fraction of the box, in one coordinate) from those taken.
fn distinct(members: &[(Vec<f64>, f64)], lb: &[f64], ub: &[f64], n: usize, sep: f64) -> Vec<(Vec<f64>, f64)> {
    let mut out: Vec<(Vec<f64>, f64)> = vec![];
    for (x, f) in members {
        if out.len() == n { break; }
        if !f.is_finite() { continue; }
        let far = out.iter().all(|(y, _)| (0..x.len()).any(|k| (x[k] - y[k]).abs() > sep * (ub[k] - lb[k])));
        if far { out.push((x.clone(), *f)); }
    }
    out
}

fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> Result<T, String> {
    if threads == 0 { Ok(f()) } else { Ok(rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| e.to_string())?.install(f)) }
}

impl FitResult {
    pub fn best(&self) -> &StartResult { &self.starts[0] }
}

/// Progress callback: (finished starts, total starts, best nllh so far) -> true cancels. Starts already
/// running finish their current iteration and keep their best point.
pub type Progress<'a> = &'a (dyn Fn(usize, usize, f64) -> bool + Sync);

struct Lsq<'a> { p: &'a Problem, stop: &'a AtomicBool }

impl Residuals for Lsq<'_> {
    fn res(&self, x: &[f64]) -> Option<Vec<f64>> { self.p.evaluate(x, false, true).ok().map(|e| e.res) }
    fn res_jac(&self, x: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> { self.p.evaluate(x, false, true).ok().map(|e| (e.res, e.jac)) }
    fn cancel(&self, _nit: usize, _cost: f64) -> bool { self.stop.load(Ordering::Relaxed) }
}

/// Latin hypercube sample of n points in the box [lb, ub] (finite bounds), from one seeded stream.
pub fn latin_hypercube(n: usize, lb: &[f64], ub: &[f64], seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed);
    let d = lb.len();
    let mut pts = vec![vec![0.0; d]; n];
    for k in 0..d {
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() { let j = (rng.next_u64() % (i as u64 + 1)) as usize; perm.swap(i, j); }
        for (i, pt) in pts.iter_mut().enumerate() {
            let u = (perm[i] as f64 + rng.unif()) / n as f64;
            pt[k] = lb[k] + u * (ub[k] - lb[k]);
        }
    }
    pts
}

/// Multistart trust-region reflective least squares. Each start is independent and deterministic, so the
/// result for a given seed does not depend on the number of threads.
pub fn fit(p: &Problem, o: &FitOptions, progress: Option<Progress>) -> Result<FitResult, String> {
    let least_squares = match o.method {
        Method::Auto => p.least_squares,
        Method::LeastSquares if !p.least_squares => {
            return Err("this problem is not a least-squares problem (Laplace noise, or noise depending on a free parameter or the state): \
                        use method lbfgsb (or auto)".into());
        }
        Method::LeastSquares => true,
        Method::Lbfgsb => false,
    };
    let n = p.n_free();
    if n == 0 { return Err("no parameter to estimate".into()); }
    let (lb, ub) = p.bounds();
    if lb.iter().chain(&ub).any(|v| !v.is_finite()) { return Err("multistart needs finite bounds for every estimated parameter".into()); }
    if let Some(x0) = &o.x0 {
        if x0.len() != n { return Err(format!("x0 has {} values, the problem {n} free parameters", x0.len())); }
    }
    let stop = AtomicBool::new(false);
    // global stage: its best distinct members are the local starts
    let mut global = None;
    let starts = match o.global {
        None => {
            let mut s = latin_hypercube(o.n_starts.max(1), &lb, &ub, o.seed);
            if let Some(x0) = &o.x0 { s[0] = x0.clone(); }
            s
        }
        Some(g) => {
            let budget = if o.global_opts.max_evals == 0 { (1000 * n).max(10_000) } else { o.global_opts.max_evals };
            let cost = NllhOnly { p, stop: &stop, progress, budget };
            let init: Vec<Vec<f64>> = o.x0.iter().cloned().collect();
            let gr = in_pool(o.threads, || minimize(&cost, g, &lb, &ub, &init, o.seed, &o.global_opts))??;
            global = Some(GlobalSummary { method: g, best: gr.f, nfev: gr.nfev, ngen: gr.ngen, status: gr.status });
            let s = distinct(&gr.members, &lb, &ub, o.n_starts.max(1), 1e-3);
            if s.is_empty() { return Err(format!("{}: no point of the search could be evaluated", g.name())); }
            if !o.polish || stop.load(Ordering::Relaxed) {
                let starts = s.into_iter().enumerate().map(|(i, (x, nllh))|
                    StartResult { start: i, x0: x.clone(), x, nllh, nfev: 0, njev: 0, nit: 0, status: gr.status }).collect();
                return Ok(FitResult { starts, ids: p.free_ids(), global });
            }
            s.into_iter().map(|m| m.0).collect()
        }
    };
    let ns = starts.len();
    let done = std::sync::Mutex::new((0usize, f64::INFINITY));
    let run = || -> Vec<StartResult> {
        starts.par_iter().enumerate().map(|(i, x0)| {
            let s = if stop.load(Ordering::Relaxed) {
                StartResult { start: i, x0: x0.clone(), x: x0.clone(), nllh: f64::INFINITY, nfev: 0, njev: 0, nit: 0, status: Status::Cancelled }
            } else if least_squares {
                let r = trf(&Lsq { p, stop: &stop }, x0, &lb, &ub, &o.trf);
                // nllh from the objective itself (cost + the constant terms)
                let nllh = if r.status == Status::Failed { f64::INFINITY } else { p.evaluate(&r.x, false, false).map_or(f64::INFINITY, |e| e.nllh) };
                StartResult { start: i, x0: x0.clone(), x: r.x, nllh, nfev: r.nfev, njev: r.njev, nit: r.nit, status: r.status }
            } else {
                let r = lbfgsb(&Nllh { p, stop: &stop }, x0, &lb, &ub, &o.lbfgsb);
                let status = match r.status {
                    LbfgsbStatus::Failed => Status::Failed, LbfgsbStatus::MaxIter | LbfgsbStatus::MaxEval => Status::MaxEval,
                    LbfgsbStatus::Gtol => Status::Gtol, LbfgsbStatus::Ftol => Status::Ftol,
                    LbfgsbStatus::LineSearch => Status::LineSearch, LbfgsbStatus::Cancelled => Status::Cancelled,
                };
                // every evaluation gives the gradient too
                StartResult { start: i, x0: x0.clone(), x: r.x, nllh: r.f, nfev: r.nfev, njev: r.nfev, nit: r.nit, status }
            };
            if let Some(cb) = progress {
                let mut d = done.lock().unwrap();
                d.0 += 1;
                d.1 = d.1.min(s.nllh);
                if cb(d.0, ns, d.1) { stop.store(true, Ordering::Relaxed); }
            }
            s
        }).collect()
    };
    let mut res = in_pool(o.threads, run)?;
    res.sort_by(|a, b| a.nllh.total_cmp(&b.nllh).then(a.start.cmp(&b.start)));
    Ok(FitResult { starts: res, ids: p.free_ids(), global })
}

/// Fisher-information (Wald) intervals at theta: from the Gauss-Newton Hessian J^T J of the residuals.
#[derive(Clone, Debug)]
pub struct Fisher {
    pub ids: Vec<String>,
    /// theta, on the parameter scales
    pub x: Vec<f64>,
    /// covariance on the parameter scales (pseudo-inverse of J^T J), row-major
    pub cov: Vec<f64>,
    /// standard deviations (infinity: not identifiable from this information)
    pub sd: Vec<f64>,
    /// interval bounds on the parameter scales
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    /// correlation matrix, row-major (NaN where a standard deviation is infinite)
    pub corr: Vec<f64>,
    /// false where J^T J is (numerically) singular along that parameter
    pub identifiable: Vec<bool>,
}

/// Wald intervals at confidence `level` (e.g. 0.95). A parameter with a sizeable component along a direction
/// of near-zero curvature (eigenvalue below 1e-10 of the largest) is reported as not identifiable.
pub fn fisher(p: &Problem, theta: &[f64], level: f64) -> Result<Fisher, String> {
    if !p.least_squares { return Err("Fisher intervals need a least-squares problem in stage 1".into()); }
    if !(level > 0.0 && level < 1.0) { return Err("level must be in (0, 1)".into()); }
    let e = p.evaluate(theta, false, true)?;
    let n = theta.len();
    let m = e.res.len();
    let j = DMatrix::from_row_slice(m, n, &e.jac);
    let h = j.transpose() * &j;
    let eig = SymmetricEigen::new(h);
    let lmax = eig.eigenvalues.iter().copied().fold(0.0, f64::max);
    let tol = 1e-10 * lmax;
    let mut cov = DMatrix::<f64>::zeros(n, n);
    let mut identifiable = vec![true; n];
    for k in 0..n {
        let l = eig.eigenvalues[k];
        let v = eig.eigenvectors.column(k);
        if l > tol {
            cov += (v * v.transpose()) / l;
        } else {
            for i in 0..n { if v[i].abs() > 1e-3 { identifiable[i] = false; } }
        }
    }
    let z = normal_quantile(0.5 + level / 2.0);
    let sd: Vec<f64> = (0..n).map(|i| if identifiable[i] { cov[(i, i)].max(0.0).sqrt() } else { f64::INFINITY }).collect();
    let corr = (0..n * n).map(|q| {
        let (a, b) = (q / n, q % n);
        if sd[a].is_finite() && sd[b].is_finite() && sd[a] > 0.0 && sd[b] > 0.0 { cov[(a, b)] / (sd[a] * sd[b]) } else { f64::NAN }
    }).collect();
    Ok(Fisher {
        ids: p.free_ids(), x: theta.to_vec(), cov: (0..n * n).map(|q| cov[(q / n, q % n)]).collect(),
        lower: (0..n).map(|i| theta[i] - z * sd[i]).collect(), upper: (0..n).map(|i| theta[i] + z * sd[i]).collect(),
        sd, corr, identifiable,
    })
}

/// Standard normal quantile (Acklam's rational approximation refined by one Halley step; ~1e-15).
pub fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
    const B: [f64; 5] = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const C: [f64; 6] = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const D: [f64; 4] = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    if !(p > 0.0 && p < 1.0) { return f64::NAN; }
    let pl = 0.02425;
    let x = if p < pl {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - pl {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    // Halley step with erfc
    let e = 0.5 * erfc(-x / std::f64::consts::SQRT_2) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (x * x / 2.0).exp();
    x - u / (1.0 + x * u / 2.0)
}

/// Complementary error function: the erf series below 2, a continued fraction (Lentz) above; ~1e-15 relative.
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let r = if z < 2.0 {
        // series for erf
        let mut sum = z;
        let mut term = z;
        let z2 = z * z;
        let mut k = 0.0;
        loop {
            k += 1.0;
            term *= -z2 / k;
            let t = term / (2.0 * k + 1.0);
            sum += t;
            if t.abs() <= 1e-17 * sum.abs() { break; }
        }
        1.0 - 2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        // continued fraction (Lentz)
        let tiny = 1e-300;
        let mut f = z;
        let mut c = z;
        let mut d = 0.0;
        for k in 1..200 {
            let a = k as f64 / 2.0;
            d = z + a * d; if d.abs() < tiny { d = tiny; }
            c = z + a / c; if c.abs() < tiny { c = tiny; }
            d = 1.0 / d;
            let delta = c * d;
            f *= delta;
            if (delta - 1.0).abs() < 1e-16 { break; }
        }
        (-z * z).exp() / (f * std::f64::consts::PI.sqrt())
    };
    if x < 0.0 { 2.0 - r } else { r }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles() {
        assert!((normal_quantile(0.975) - 1.959963984540054).abs() < 1e-12);
        assert!((normal_quantile(0.5)).abs() < 1e-15);
        assert!((normal_quantile(0.001) + 3.090232306167813).abs() < 1e-11);
    }

    #[test]
    fn lhs_strata() {
        let pts = latin_hypercube(8, &[0.0, -1.0], &[1.0, 1.0], 3);
        for k in 0..2 {
            let mut bins: Vec<usize> = pts.iter().map(|p| (((p[k] - [0.0, -1.0][k]) / [1.0, 2.0][k]) * 8.0) as usize).collect();
            bins.sort();
            assert_eq!(bins, (0..8).collect::<Vec<_>>());
        }
    }
}
