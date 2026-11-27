//! Trust-region reflective least squares with bounds (Branch, Coleman and Li, SIAM J. Sci. Comput. 21, 1999),
//! the bounded Levenberg-Marquardt method of the spec. Minimizes 0.5 * |r(x)|^2 subject to lb <= x <= ub.
//!
//! Each iteration scales the variables by the Coleman-Li vector (distance to the bound the gradient points
//! at), solves the trust-region subproblem exactly through an SVD of the scaled, augmented Jacobian, and
//! keeps iterates strictly inside the bounds: a step that would leave them is cut at the bound, and the
//! better of the cut step, its reflection off the bound and the scaled gradient direction is taken. The
//! structure and defaults follow the "exact" variant of scipy.optimize.least_squares(method="trf").
use nalgebra::DMatrix;

#[derive(Clone, Debug)]
pub struct TrfOptions {
    /// stop when the cost falls by less than ftol * cost in a step (with a good model fit)
    pub ftol: f64,
    /// stop when the step is below xtol * (xtol + |x|)
    pub xtol: f64,
    /// stop when the scaled gradient's largest entry is below gtol
    pub gtol: f64,
    /// residual evaluations allowed (0: 100 * number of parameters)
    pub max_nfev: usize,
}

impl Default for TrfOptions {
    fn default() -> Self { TrfOptions { ftol: 1e-8, xtol: 1e-8, gtol: 1e-8, max_nfev: 0 } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// the starting point could not be evaluated
    Failed,
    /// residual evaluation budget used up
    MaxEval,
    Gtol,
    Ftol,
    Xtol,
    FtolXtol,
    /// cancelled by the caller
    Cancelled,
}

impl Status {
    pub fn message(self) -> &'static str {
        match self {
            Status::Failed => "the starting point could not be evaluated",
            Status::MaxEval => "maximum number of function evaluations reached",
            Status::Gtol => "gradient small enough (gtol)",
            Status::Ftol => "objective change small enough (ftol)",
            Status::Xtol => "step small enough (xtol)",
            Status::FtolXtol => "objective change and step small enough (ftol, xtol)",
            Status::Cancelled => "cancelled",
        }
    }
    pub fn converged(self) -> bool { matches!(self, Status::Gtol | Status::Ftol | Status::Xtol | Status::FtolXtol) }
}

#[derive(Clone, Debug)]
pub struct TrfResult {
    pub x: Vec<f64>,
    /// 0.5 * |r|^2 at x
    pub cost: f64,
    pub res: Vec<f64>,
    /// row-major, len(res) x len(x)
    pub jac: Vec<f64>,
    pub nfev: usize,
    pub njev: usize,
    pub nit: usize,
    pub status: Status,
}

/// What the optimizer needs: residuals, and residuals with their Jacobian (row-major m x n). None: the point
/// cannot be evaluated (a failed simulation); it is rejected and the trust region shrinks.
pub trait Residuals {
    fn res(&self, x: &[f64]) -> Option<Vec<f64>>;
    fn res_jac(&self, x: &[f64]) -> Option<(Vec<f64>, Vec<f64>)>;
    /// Called after every accepted step with (iteration, cost); true cancels.
    fn cancel(&self, _nit: usize, _cost: f64) -> bool { false }
}

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
fn norm(a: &[f64]) -> f64 { dot(a, a).sqrt() }

/// J s for J row-major m x n.
fn jmul(j: &[f64], n: usize, s: &[f64]) -> Vec<f64> { j.chunks(n).map(|row| dot(row, s)).collect() }

/// J^T f.
fn jtmul(j: &[f64], n: usize, f: &[f64]) -> Vec<f64> {
    let mut g = vec![0.0; n];
    for (row, &fi) in j.chunks(n).zip(f) { for (gk, &jk) in g.iter_mut().zip(row) { *gk += jk * fi; } }
    g
}

/// 0.5 s^T (J^T J + diag) s + g^T s.
fn quadratic(jh: &[f64], n: usize, g: &[f64], s: &[f64], diag: &[f64]) -> f64 {
    let js = jmul(jh, n, s);
    0.5 * (dot(&js, &js) + s.iter().zip(diag).map(|(x, d)| x * x * d).sum::<f64>()) + dot(g, s)
}

/// The quadratic along s0 + t s as a t^2 + b t + c.
fn quadratic_1d(jh: &[f64], n: usize, g: &[f64], s: &[f64], diag: &[f64], s0: Option<&[f64]>) -> (f64, f64, f64) {
    let v = jmul(jh, n, s);
    let a = 0.5 * (dot(&v, &v) + s.iter().zip(diag).map(|(x, d)| x * x * d).sum::<f64>());
    let mut b = dot(g, s);
    let mut c = 0.0;
    if let Some(s0) = s0 {
        let u = jmul(jh, n, s0);
        b += dot(&u, &v) + s0.iter().zip(s).zip(diag).map(|((x, y), d)| x * y * d).sum::<f64>();
        c = 0.5 * (dot(&u, &u) + s0.iter().zip(diag).map(|(x, d)| x * x * d).sum::<f64>()) + dot(g, s0);
    }
    (a, b, c)
}

/// Minimum of a t^2 + b t + c on [lo, hi]: (t, value).
fn minimize_1d(a: f64, b: f64, lo: f64, hi: f64, c: f64) -> (f64, f64) {
    let mut ts = vec![lo, hi];
    if a != 0.0 {
        let e = -0.5 * b / a;
        if lo < e && e < hi { ts.push(e); }
    }
    ts.into_iter().map(|t| (t, t * (a * t + b) + c)).min_by(|p, q| p.1.total_cmp(&q.1)).unwrap()
}

/// Largest t with x + t s inside the bounds, and per component: -1 / 1 if it hits its lower / upper bound there.
fn step_to_bound(x: &[f64], s: &[f64], lb: &[f64], ub: &[f64]) -> (f64, Vec<i8>) {
    let steps: Vec<f64> = (0..x.len()).map(|i| {
        if s[i] == 0.0 { f64::INFINITY } else { ((lb[i] - x[i]) / s[i]).max((ub[i] - x[i]) / s[i]) }
    }).collect();
    let t = steps.iter().copied().fold(f64::INFINITY, f64::min);
    let hits = (0..x.len()).map(|i| if steps[i] == t && s[i] != 0.0 { s[i].signum() as i8 } else { 0 }).collect();
    (t, hits)
}

/// The two t with |x + t s| = delta, ascending.
fn intersect_trust_region(x: &[f64], s: &[f64], delta: f64) -> (f64, f64) {
    let (a, b, c) = (dot(s, s), dot(x, s), dot(x, x) - delta * delta);
    let d = (b * b - a * c).max(0.0).sqrt();
    let q = -(b + d.copysign(b));
    let (t1, t2) = (q / a, c / q);
    if t1 < t2 { (t1, t2) } else { (t2, t1) }
}

fn in_bounds(x: &[f64], lb: &[f64], ub: &[f64]) -> bool { (0..x.len()).all(|i| x[i] >= lb[i] && x[i] <= ub[i]) }

fn next_toward(a: f64, b: f64) -> f64 {
    if a == b || a.is_nan() || b.is_nan() { return a; }
    if a == 0.0 { return if b > 0.0 { f64::from_bits(1) } else { -f64::from_bits(1) }; }
    let bits = a.to_bits();
    f64::from_bits(if (b > a) == (a > 0.0) { bits + 1 } else { bits - 1 })
}

/// x moved strictly inside the bounds: by rstep (relative) off an active bound, or one ulp if rstep is 0.
fn strictly_feasible(x: &[f64], lb: &[f64], ub: &[f64], rstep: f64) -> Vec<f64> {
    (0..x.len()).map(|i| {
        let (l, u) = (lb[i], ub[i]);
        let mut v = x[i];
        let (lower, upper) = if rstep == 0.0 {
            (v <= l, v >= u)
        } else {
            let (ld, ud) = (v - l, u - v);
            let (lt, ut) = (rstep * l.abs().max(1.0), rstep * u.abs().max(1.0));
            (l.is_finite() && ld <= lt.min(ud), u.is_finite() && ud <= ut.min(ld))
        };
        if lower { v = if rstep == 0.0 { next_toward(l, u) } else { l + rstep * l.abs().max(1.0) }; }
        if upper { v = if rstep == 0.0 { next_toward(u, l) } else { u - rstep * u.abs().max(1.0) }; }
        if v < l || v > u { v = 0.5 * (l + u); }
        v
    }).collect()
}

/// Coleman-Li scaling vector v and its derivative dv.
fn cl_scaling(x: &[f64], g: &[f64], lb: &[f64], ub: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let (mut v, mut dv) = (vec![1.0; n], vec![0.0; n]);
    for i in 0..n {
        if g[i] < 0.0 && ub[i].is_finite() { v[i] = ub[i] - x[i]; dv[i] = -1.0; }
        if g[i] > 0.0 && lb[i].is_finite() { v[i] = x[i] - lb[i]; dv[i] = 1.0; }
    }
    (v, dv)
}

/// Column norms of J as (scale = 1 / norm, scale_inv = norm), growing from `old`.
fn jac_scale(j: &[f64], n: usize, old: Option<&[f64]>) -> (Vec<f64>, Vec<f64>) {
    let mut si = vec![0.0; n];
    for row in j.chunks(n) { for k in 0..n { si[k] += row[k] * row[k]; } }
    for k in 0..n {
        si[k] = si[k].sqrt();
        match old { None => if si[k] == 0.0 { si[k] = 1.0 }, Some(o) => si[k] = si[k].max(o[k]) }
    }
    (si.iter().map(|x| 1.0 / x).collect(), si)
}

/// Trust-region subproblem min |J p + f| s.t. |p| <= delta given the SVD of J (uf = U^T f, singular values s,
/// right singular vectors as columns of v): (p, Levenberg-Marquardt parameter alpha).
fn solve_trust_region(m: usize, uf: &[f64], s: &[f64], v: &DMatrix<f64>, delta: f64, alpha0: f64) -> (Vec<f64>, f64) {
    let n = s.len();
    let suf: Vec<f64> = s.iter().zip(uf).map(|(a, b)| a * b).collect();
    let smax = s.iter().copied().fold(0.0, f64::max);
    let smin = s.iter().copied().fold(f64::INFINITY, f64::min);
    let full_rank = m >= n && smin > f64::EPSILON * m as f64 * smax;
    let vmul = |c: &[f64]| -> Vec<f64> { (0..n).map(|i| -(0..n).map(|k| v[(i, k)] * c[k]).sum::<f64>()).collect() };
    if full_rank {
        let p = vmul(&uf.iter().zip(s).map(|(a, b)| a / b).collect::<Vec<_>>());
        if norm(&p) <= delta { return (p, 0.0); }
    }
    let phi = |alpha: f64| {
        let den: Vec<f64> = s.iter().map(|x| x * x + alpha).collect();
        let pn = suf.iter().zip(&den).map(|(a, d)| (a / d).powi(2)).sum::<f64>().sqrt();
        let dphi = -suf.iter().zip(&den).map(|(a, d)| a * a / d.powi(3)).sum::<f64>() / pn;
        (pn - delta, dphi)
    };
    let mut hi = norm(&suf) / delta;
    let mut lo = if full_rank { let (p, d) = phi(0.0); -p / d } else { 0.0 };
    let mut alpha = if alpha0 == 0.0 { (0.001 * hi).max((lo * hi).sqrt()) } else { alpha0 };
    for _ in 0..10 {
        if alpha < lo || alpha > hi { alpha = (0.001 * hi).max((lo * hi).sqrt()); }
        let (p, d) = phi(alpha);
        if p < 0.0 { hi = alpha; }
        let ratio = p / d;
        lo = lo.max(alpha - ratio);
        alpha -= (p + delta) * ratio / delta;
        if p.abs() < 0.01 * delta { break; }
    }
    let mut p = vmul(&suf.iter().zip(s).map(|(a, b)| a / (b * b + alpha)).collect::<Vec<_>>());
    let pn = norm(&p);
    if pn > 0.0 { p.iter_mut().for_each(|x| *x *= delta / pn); }
    (p, alpha)
}

/// The step taken from x: the trust-region step p if it stays inside the bounds; otherwise the best of p cut
/// at the bound, its reflection, and the scaled gradient step. Returns (step, scaled step, predicted reduction).
#[allow(clippy::too_many_arguments)]
fn select_step(x: &[f64], jh: &[f64], n: usize, diag_h: &[f64], g_h: &[f64], mut p: Vec<f64>, mut p_h: Vec<f64>,
               d: &[f64], delta: f64, lb: &[f64], ub: &[f64], theta: f64) -> (Vec<f64>, Vec<f64>, f64) {
    let xp: Vec<f64> = x.iter().zip(&p).map(|(a, b)| a + b).collect();
    if in_bounds(&xp, lb, ub) {
        let v = quadratic(jh, n, g_h, &p_h, diag_h);
        return (p, p_h, -v);
    }
    let (p_stride, hits) = step_to_bound(x, &p, lb, ub);
    // reflection
    let mut r_h = p_h.clone();
    for i in 0..n { if hits[i] != 0 { r_h[i] = -r_h[i]; } }
    let mut r: Vec<f64> = d.iter().zip(&r_h).map(|(a, b)| a * b).collect();
    p.iter_mut().for_each(|v| *v *= p_stride);
    p_h.iter_mut().for_each(|v| *v *= p_stride);
    let x_on: Vec<f64> = x.iter().zip(&p).map(|(a, b)| a + b).collect();
    let (_, to_tr) = intersect_trust_region(&p_h, &r_h, delta);
    let (to_bound, _) = step_to_bound(&x_on, &r, lb, ub);
    let r_stride = to_bound.min(to_tr);
    let (r_lo, r_hi) = if r_stride > 0.0 {
        ((1.0 - theta) * p_stride / r_stride, if r_stride == to_bound { theta * to_bound } else { to_tr })
    } else { (0.0, -1.0) };
    let r_value = if r_lo <= r_hi {
        let (a, b, c) = quadratic_1d(jh, n, g_h, &r_h, diag_h, Some(&p_h));
        let (t, val) = minimize_1d(a, b, r_lo, r_hi, c);
        for i in 0..n { r_h[i] = r_h[i] * t + p_h[i]; r[i] = r_h[i] * d[i]; }
        val
    } else { f64::INFINITY };
    // the cut step, pulled back off the bound
    p.iter_mut().for_each(|v| *v *= theta);
    p_h.iter_mut().for_each(|v| *v *= theta);
    let p_value = quadratic(jh, n, g_h, &p_h, diag_h);
    // the scaled gradient
    let mut ag_h: Vec<f64> = g_h.iter().map(|v| -v).collect();
    let mut ag: Vec<f64> = d.iter().zip(&ag_h).map(|(a, b)| a * b).collect();
    let to_tr = delta / norm(&ag_h);
    let (to_bound, _) = step_to_bound(x, &ag, lb, ub);
    let ag_max = if to_bound < to_tr { theta * to_bound } else { to_tr };
    let (a, b, _) = quadratic_1d(jh, n, g_h, &ag_h, diag_h, None);
    let (t, ag_value) = minimize_1d(a, b, 0.0, ag_max, 0.0);
    ag_h.iter_mut().for_each(|v| *v *= t);
    ag.iter_mut().for_each(|v| *v *= t);
    if p_value < r_value && p_value < ag_value { (p, p_h, -p_value) }
    else if r_value < p_value && r_value < ag_value { (r, r_h, -r_value) }
    else { (ag, ag_h, -ag_value) }
}

/// Minimize 0.5 |r(x)|^2 on [lb, ub] from x0.
pub fn trf(f: &dyn Residuals, x0: &[f64], lb: &[f64], ub: &[f64], o: &TrfOptions) -> TrfResult {
    let n = x0.len();
    let max_nfev = if o.max_nfev == 0 { 100 * n.max(1) } else { o.max_nfev };
    let mut x = strictly_feasible(x0, lb, ub, 1e-10);
    let failed = |x: Vec<f64>| TrfResult { x, cost: f64::INFINITY, res: vec![], jac: vec![], nfev: 1, njev: 1, nit: 0, status: Status::Failed };
    let Some((mut fv, mut j)) = f.res_jac(&x) else { return failed(x) };
    let m = fv.len();
    if fv.iter().chain(&j).any(|v| !v.is_finite()) { return failed(x); }
    let (mut nfev, mut njev, mut nit) = (1, 1, 0);
    let mut cost = 0.5 * dot(&fv, &fv);
    let mut g = jtmul(&j, n, &fv);
    let (mut scale, mut scale_inv) = jac_scale(&j, n, None);
    let (mut v, dv) = cl_scaling(&x, &g, lb, ub);
    for i in 0..n { if dv[i] != 0.0 { v[i] *= scale_inv[i]; } }
    let mut delta = norm(&(0..n).map(|i| x[i] * scale_inv[i] / v[i].sqrt()).collect::<Vec<_>>());
    if delta == 0.0 || !delta.is_finite() { delta = 1.0; }
    let mut alpha = 0.0;
    let status = loop {
        let (mut v, dv) = cl_scaling(&x, &g, lb, ub);
        let g_norm = (0..n).map(|i| (g[i] * v[i]).abs()).fold(0.0, f64::max);
        if g_norm < o.gtol { break Status::Gtol; }
        if nfev >= max_nfev { break Status::MaxEval; }
        for i in 0..n { if dv[i] != 0.0 { v[i] *= scale_inv[i]; } }
        let d: Vec<f64> = (0..n).map(|i| v[i].sqrt() * scale[i]).collect();
        let diag_h: Vec<f64> = (0..n).map(|i| g[i] * dv[i] * scale[i]).collect();
        let g_h: Vec<f64> = (0..n).map(|i| d[i] * g[i]).collect();
        let jh: Vec<f64> = j.chunks(n).flat_map(|row| row.iter().zip(&d).map(|(a, b)| a * b).collect::<Vec<_>>()).collect();
        let mut ja = DMatrix::<f64>::zeros(m + n, n);
        for r in 0..m { for k in 0..n { ja[(r, k)] = jh[r * n + k]; } }
        for k in 0..n { ja[(m + k, k)] = diag_h[k].max(0.0).sqrt(); }
        let svd = ja.svd(true, true);
        let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
        let s: Vec<f64> = svd.singular_values.iter().copied().collect();
        let uf: Vec<f64> = (0..s.len()).map(|k| (0..m).map(|r| u[(r, k)] * fv[r]).sum()).collect();
        let vmat = vt.transpose();
        let theta = (1.0 - g_norm).max(0.995);
        let mut reduction = -1.0;
        let mut step_norm = 0.0;
        let mut accepted: Option<(Vec<f64>, Vec<f64>, f64)> = None;
        let mut term = None;
        while reduction <= 0.0 && nfev < max_nfev {
            let (p_h, a) = solve_trust_region(m, &uf, &s, &vmat, delta, alpha);
            alpha = a;
            let p: Vec<f64> = (0..n).map(|i| d[i] * p_h[i]).collect();
            let (step, step_h, predicted) = select_step(&x, &jh, n, &diag_h, &g_h, p, p_h, &d, delta, lb, ub, theta);
            let xt: Vec<f64> = x.iter().zip(&step).map(|(a, b)| a + b).collect();
            let x_new = strictly_feasible(&xt, lb, ub, 0.0);
            nfev += 1;
            let step_h_norm = norm(&step_h);
            let f_new = f.res(&x_new).filter(|r| r.len() == m && r.iter().all(|v| v.is_finite()));
            let Some(f_new) = f_new else { delta = 0.25 * step_h_norm; continue };
            let cost_new = 0.5 * dot(&f_new, &f_new);
            reduction = cost - cost_new;
            let ratio = if predicted > 0.0 { reduction / predicted } else if predicted == 0.0 && reduction == 0.0 { 1.0 } else { 0.0 };
            let delta_new = if ratio < 0.25 { 0.25 * step_h_norm } else if ratio > 0.75 && step_h_norm > 0.95 * delta { 2.0 * delta } else { delta };
            step_norm = norm(&step);
            let ftol_ok = reduction < o.ftol * cost && ratio > 0.25;
            let xtol_ok = step_norm < o.xtol * (o.xtol + norm(&x));
            term = match (ftol_ok, xtol_ok) { (true, true) => Some(Status::FtolXtol), (true, false) => Some(Status::Ftol), (false, true) => Some(Status::Xtol), _ => None };
            if reduction > 0.0 { accepted = Some((x_new, f_new, cost_new)); }
            if term.is_some() { break; }
            if delta_new > 0.0 { alpha *= delta / delta_new; }
            delta = delta_new;
        }
        let _ = step_norm;
        if let Some((x_new, _, cost_new)) = accepted {
            match f.res_jac(&x_new) {
                Some((f2, j2)) if f2.iter().chain(&j2).all(|v| v.is_finite()) => {
                    x = x_new; fv = f2; j = j2; cost = cost_new; njev += 1;
                    g = jtmul(&j, n, &fv);
                    let (sc, si) = jac_scale(&j, n, Some(&scale_inv));
                    scale = sc; scale_inv = si;
                }
                _ => { delta *= 0.25; }
            }
        }
        nit += 1;
        if let Some(t) = term { break t; }
        if f.cancel(nit, cost) { break Status::Cancelled; }
    };
    TrfResult { x, cost, res: fv, jac: j, nfev, njev, nit, status }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rosen;
    impl Residuals for Rosen {
        fn res(&self, x: &[f64]) -> Option<Vec<f64>> { Some(vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]) }
        fn res_jac(&self, x: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
            Some((self.res(x)?, vec![-20.0 * x[0], 10.0, -1.0, 0.0]))
        }
    }

    #[test]
    fn rosenbrock_unbounded_and_bounded() {
        let inf = f64::INFINITY;
        let r = trf(&Rosen, &[-1.2, 1.0], &[-inf, -inf], &[inf, inf], &TrfOptions::default());
        assert!(r.status.converged(), "{:?}", r.status);
        assert!((r.x[0] - 1.0).abs() < 1e-6 && (r.x[1] - 1.0).abs() < 1e-6, "{:?}", r.x);
        // scipy's documented example: bounds x1 <= 1.5 on the Rosenbrock problem with x0 = (2, 2)
        let r = trf(&Rosen, &[2.0, 2.0], &[-inf, 1.5], &[inf, inf], &TrfOptions::default());
        assert!(r.status.converged(), "{:?}", r.status);
        assert!((r.x[0] - 1.22437075).abs() < 1e-5 && (r.x[1] - 1.5).abs() < 1e-8, "{:?}", r.x);
    }

    // exponential decay fit: y = a exp(-b t), data from a = 2, b = 0.7
    struct Exp { t: Vec<f64>, y: Vec<f64> }
    impl Residuals for Exp {
        fn res(&self, x: &[f64]) -> Option<Vec<f64>> { Some(self.t.iter().zip(&self.y).map(|(t, y)| x[0] * (-x[1] * t).exp() - y).collect()) }
        fn res_jac(&self, x: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
            let j = self.t.iter().flat_map(|t| { let e = (-x[1] * t).exp(); [e, -x[0] * t * e] }).collect();
            Some((self.res(x)?, j))
        }
    }

    #[test]
    fn exponential_with_active_bound() {
        let t: Vec<f64> = (0..20).map(|i| i as f64 * 0.25).collect();
        let y = t.iter().map(|t| 2.0 * (-0.7 * t).exp()).collect();
        let p = Exp { t, y };
        let r = trf(&p, &[1.0, 0.1], &[0.0, 0.0], &[10.0, 10.0], &TrfOptions::default());
        assert!(r.status.converged() && (r.x[0] - 2.0).abs() < 1e-7 && (r.x[1] - 0.7).abs() < 1e-7, "{r:?}");
        // b held below its true value: the optimum sits on the bound
        let r = trf(&p, &[1.0, 0.1], &[0.0, 0.0], &[10.0, 0.5], &TrfOptions::default());
        assert!(r.status.converged() && (r.x[1] - 0.5).abs() < 1e-6, "{r:?}");
    }
}
