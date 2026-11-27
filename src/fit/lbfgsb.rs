//! L-BFGS-B: limited-memory quasi-Newton minimization with bounds (Byrd, Lu, Nocedal and Zhu, SIAM J. Sci.
//! Comput. 16, 1995; Zhu, Byrd, Lu and Nocedal, ACM TOMS 23, 1997). For objectives that are not sums of squares:
//! estimated noise parameters, Laplace noise, priors.
//!
//! Each iteration finds the generalized Cauchy point (the first local minimizer of the quadratic model along the
//! projected steepest-descent path), minimizes the model over the variables still free there (the direct
//! primal method through the compact representation B = theta I - W M W^T), and searches along the resulting
//! direction with the Moré-Thuente line search (MINPACK-2 dcsrch / dcstep), so the curvature pairs satisfy the
//! strong Wolfe conditions.
use nalgebra::{DMatrix, DVector};

#[derive(Clone, Debug)]
pub struct LbfgsbOptions {
    /// stored correction pairs
    pub m: usize,
    /// stop when (f_old - f) / max(|f_old|, |f|, 1) <= ftol
    pub ftol: f64,
    /// stop when the largest entry of the projected gradient is <= gtol
    pub gtol: f64,
    pub max_iter: usize,
    /// objective evaluations allowed (0: 50 x iterations allowed)
    pub max_fev: usize,
}

impl Default for LbfgsbOptions {
    fn default() -> Self { LbfgsbOptions { m: 10, ftol: 1e-12, gtol: 1e-8, max_iter: 1000, max_fev: 0 } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LbfgsbStatus {
    /// the starting point could not be evaluated
    Failed,
    MaxIter,
    MaxEval,
    /// the projected gradient is small (gtol)
    Gtol,
    /// the objective no longer decreases (ftol)
    Ftol,
    /// no step along the search direction decreases the objective, even after a restart
    LineSearch,
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct LbfgsbResult {
    pub x: Vec<f64>,
    pub f: f64,
    pub g: Vec<f64>,
    pub nit: usize,
    pub nfev: usize,
    pub status: LbfgsbStatus,
}

/// The objective and its gradient; None: the point cannot be evaluated (a failed simulation). `cancel` is
/// asked after every iteration.
pub trait Objective {
    fn eval(&self, x: &[f64]) -> Option<(f64, Vec<f64>)>;
    fn cancel(&self, _nit: usize, _f: f64) -> bool { false }
}

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

/// Largest entry of the projected gradient (the gradient with components that point out of the box at an active
/// bound removed).
fn proj_grad_norm(x: &[f64], g: &[f64], lb: &[f64], ub: &[f64]) -> f64 {
    (0..x.len()).map(|i| {
        let gi = g[i];
        let pg = if gi < 0.0 { (x[i] - ub[i]).max(gi) } else { (x[i] - lb[i]).min(gi) };
        pg.abs()
    }).fold(0.0, f64::max)
}

/// The limited-memory matrix B = theta I - W M W^T from the stored pairs (oldest first).
struct Memory { s: Vec<Vec<f64>>, y: Vec<Vec<f64>>, theta: f64, w: DMatrix<f64>, mm: DMatrix<f64> }

impl Memory {
    fn new(n: usize) -> Memory { Memory { s: vec![], y: vec![], theta: 1.0, w: DMatrix::zeros(n, 0), mm: DMatrix::zeros(0, 0) } }
    fn k(&self) -> usize { self.s.len() }
    fn clear(&mut self) { let n = self.w.nrows(); *self = Memory::new(n); }

    fn push(&mut self, s: Vec<f64>, y: Vec<f64>, m: usize) {
        self.theta = dot(&y, &y) / dot(&s, &y);
        self.s.push(s);
        self.y.push(y);
        if self.s.len() > m { self.s.remove(0); self.y.remove(0); }
        self.rebuild();
    }

    /// W = [Y, theta S]; M = [[-D, L^T], [L, theta S^T S]]^-1 (D = diag(s_i^T y_i), L_ij = s_i^T y_j for i > j).
    fn rebuild(&mut self) {
        let (n, k) = (self.w.nrows(), self.k());
        let mut w = DMatrix::zeros(n, 2 * k);
        for j in 0..k {
            for i in 0..n { w[(i, j)] = self.y[j][i]; w[(i, k + j)] = self.theta * self.s[j][i]; }
        }
        let mut a = DMatrix::zeros(2 * k, 2 * k);
        for i in 0..k {
            for j in 0..k {
                let sy = dot(&self.s[i], &self.y[j]);
                if i == j { a[(i, i)] = -sy; }
                if i > j { a[(k + i, j)] = sy; a[(j, k + i)] = sy; } // L and L^T
                a[(k + i, k + j)] = self.theta * dot(&self.s[i], &self.s[j]);
            }
        }
        self.mm = a.try_inverse().unwrap_or_else(|| DMatrix::zeros(2 * k, 2 * k));
        self.w = w;
    }

    fn wrow(&self, i: usize) -> DVector<f64> { self.w.row(i).transpose() }
}

/// Generalized Cauchy point from x along -g (Algorithm CP). Returns (x_cp, c = W^T (x_cp - x)).
fn cauchy(x: &[f64], g: &[f64], lb: &[f64], ub: &[f64], mem: &Memory) -> (Vec<f64>, DVector<f64>) {
    let n = x.len();
    let k2 = mem.w.ncols();
    let mut t = vec![f64::INFINITY; n];
    let mut d = vec![0.0; n];
    for i in 0..n {
        if g[i] < 0.0 { t[i] = (x[i] - ub[i]) / g[i]; } else if g[i] > 0.0 { t[i] = (x[i] - lb[i]) / g[i]; }
        if t[i] > 0.0 { d[i] = -g[i]; } else { t[i] = 0.0; }
    }
    let theta = mem.theta;
    let mut xcp = x.to_vec();
    let mut p = DVector::zeros(k2);
    for i in 0..n { if d[i] != 0.0 { p += mem.wrow(i) * d[i]; } }
    let mut c = DVector::zeros(k2);
    let mut fp = -dot(&d, &d);
    let mut fpp = -theta * fp - if k2 > 0 { p.dot(&(&mem.mm * &p)) } else { 0.0 };
    if fpp <= 0.0 { fpp = f64::EPSILON * fp.abs().max(1.0); }
    let mut dt_min = -fp / fpp;
    let mut order: Vec<usize> = (0..n).filter(|&i| t[i] > 0.0 && t[i].is_finite()).collect();
    order.sort_by(|&a, &b| t[a].total_cmp(&t[b]));
    let mut t_old = 0.0;
    let mut q = 0;
    let mut tb = order.first().map_or(f64::INFINITY, |&b| t[b]);
    while q < order.len() && dt_min >= tb - t_old {
        let b = order[q];
        let dt = tb - t_old;
        xcp[b] = if d[b] > 0.0 { ub[b] } else { lb[b] };
        let zb = xcp[b] - x[b];
        c += &p * dt;
        let gb = g[b];
        let (wmc, wmp, wmw) = if k2 > 0 {
            let wb = mem.wrow(b);
            let mwb = &mem.mm * &wb;
            (mwb.dot(&c), mwb.dot(&p), mwb.dot(&wb))
        } else { (0.0, 0.0, 0.0) };
        fp += dt * fpp + gb * gb + theta * gb * zb - gb * wmc;
        fpp -= theta * gb * gb + 2.0 * gb * wmp + gb * gb * wmw;
        if fpp <= 0.0 { fpp = f64::EPSILON * fp.abs().max(1.0); }
        if k2 > 0 { p += mem.wrow(b) * gb; }
        d[b] = 0.0;
        dt_min = -fp / fpp;
        t_old = tb;
        q += 1;
        tb = order.get(q).map_or(f64::INFINITY, |&b| t[b]);
    }
    let dt_min = dt_min.max(0.0);
    let t_new = t_old + dt_min;
    for i in 0..n { if d[i] != 0.0 { xcp[i] = (x[i] + t_new * d[i]).clamp(lb[i], ub[i]); } }
    c += &p * dt_min;
    (xcp, c)
}

/// Direct primal subspace minimization over the variables free at x_cp; the result is cut back to the box.
fn subspace(x: &[f64], g: &[f64], lb: &[f64], ub: &[f64], mem: &Memory, xcp: &[f64], c: &DVector<f64>) -> Vec<f64> {
    let n = x.len();
    let free: Vec<usize> = (0..n).filter(|&i| xcp[i] > lb[i] && xcp[i] < ub[i]).collect();
    if free.is_empty() || mem.k() == 0 { return xcp.to_vec(); }
    let theta = mem.theta;
    let k2 = mem.w.ncols();
    let mc = &mem.mm * c;
    // reduced gradient r = Z^T (g + theta (x_cp - x) - W M c)
    let r: Vec<f64> = free.iter().map(|&i| g[i] + theta * (xcp[i] - x[i]) - mem.wrow(i).dot(&mc)).collect();
    // d = -(1/theta) r - (1/theta^2) Z^T W (I - (1/theta) M W^T Z Z^T W)^-1 M W^T Z r
    let mut wtzr = DVector::zeros(k2);
    let mut wzzw = DMatrix::zeros(k2, k2);
    for (q, &i) in free.iter().enumerate() {
        let wi = mem.wrow(i);
        wtzr += &wi * r[q];
        wzzw += &wi * wi.transpose();
    }
    let v = &mem.mm * wtzr;
    let nmat = DMatrix::identity(k2, k2) - (&mem.mm * wzzw) / theta;
    let v = nmat.lu().solve(&v).unwrap_or_else(|| DVector::zeros(k2));
    let du: Vec<f64> = free.iter().enumerate().map(|(q, &i)| -r[q] / theta - mem.wrow(i).dot(&v) / (theta * theta)).collect();
    // largest step <= 1 keeping the free variables in the box
    let mut alpha: f64 = 1.0;
    for (q, &i) in free.iter().enumerate() {
        if du[q] > 0.0 { alpha = alpha.min((ub[i] - xcp[i]) / du[q]); } else if du[q] < 0.0 { alpha = alpha.min((lb[i] - xcp[i]) / du[q]); }
    }
    let mut xbar = xcp.to_vec();
    for (q, &i) in free.iter().enumerate() { xbar[i] = (xcp[i] + alpha.max(0.0) * du[q]).clamp(lb[i], ub[i]); }
    xbar
}

/// One Moré-Thuente step update (MINPACK-2 dcstep): the safeguarded cubic / quadratic step and the interval.
#[allow(clippy::too_many_arguments)]
fn dcstep(stx: &mut f64, fx: &mut f64, dx: &mut f64, sty: &mut f64, fy: &mut f64, dy: &mut f64, stp: &mut f64, fp: f64, dp: f64,
          brackt: &mut bool, stpmin: f64, stpmax: f64) {
    let sgnd = dp * (*dx / dx.abs());
    let stpf;
    if fp > *fx {
        let theta = 3.0 * (*fx - fp) / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dx / s) * (dp / s)).max(0.0).sqrt();
        if *stp < *stx { gamma = -gamma; }
        let p = (gamma - *dx) + theta;
        let q = ((gamma - *dx) + gamma) + dp;
        let r = p / q;
        let stpc = *stx + r * (*stp - *stx);
        let stpq = *stx + ((*dx / ((*fx - fp) / (*stp - *stx) + *dx)) / 2.0) * (*stp - *stx);
        stpf = if (stpc - *stx).abs() < (stpq - *stx).abs() { stpc } else { stpc + (stpq - stpc) / 2.0 };
        *brackt = true;
    } else if sgnd < 0.0 {
        let theta = 3.0 * (*fx - fp) / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dx / s) * (dp / s)).max(0.0).sqrt();
        if *stp > *stx { gamma = -gamma; }
        let p = (gamma - dp) + theta;
        let q = ((gamma - dp) + gamma) + *dx;
        let r = p / q;
        let stpc = *stp + r * (*stx - *stp);
        let stpq = *stp + (dp / (dp - *dx)) * (*stx - *stp);
        stpf = if (stpc - *stp).abs() > (stpq - *stp).abs() { stpc } else { stpq };
        *brackt = true;
    } else if dp.abs() < dx.abs() {
        let theta = 3.0 * (*fx - fp) / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dx / s) * (dp / s)).max(0.0).sqrt();
        if *stp > *stx { gamma = -gamma; }
        let p = (gamma - dp) + theta;
        let q = (gamma + (*dx - dp)) + gamma;
        let r = p / q;
        let stpc = if r < 0.0 && gamma != 0.0 { *stp + r * (*stx - *stp) } else if *stp > *stx { stpmax } else { stpmin };
        let stpq = *stp + (dp / (dp - *dx)) * (*stx - *stp);
        if *brackt {
            let mut f = if (stpc - *stp).abs() < (stpq - *stp).abs() { stpc } else { stpq };
            f = if *stp > *stx { f.min(*stp + 0.66 * (*sty - *stp)) } else { f.max(*stp + 0.66 * (*sty - *stp)) };
            stpf = f;
        } else {
            let f = if (stpc - *stp).abs() > (stpq - *stp).abs() { stpc } else { stpq };
            stpf = f.min(stpmax).max(stpmin);
        }
    } else if *brackt {
        let theta = 3.0 * (fp - *fy) / (*sty - *stp) + *dy + dp;
        let s = theta.abs().max(dy.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dy / s) * (dp / s)).max(0.0).sqrt();
        if *stp > *sty { gamma = -gamma; }
        let p = (gamma - dp) + theta;
        let q = ((gamma - dp) + gamma) + *dy;
        let r = p / q;
        stpf = *stp + r * (*sty - *stp);
    } else {
        stpf = if *stp > *stx { stpmax } else { stpmin };
    }
    if fp > *fx {
        *sty = *stp; *fy = fp; *dy = dp;
    } else {
        if sgnd < 0.0 { *sty = *stx; *fy = *fx; *dy = *dx; }
        *stx = *stp; *fx = fp; *dx = dp;
    }
    *stp = stpf;
}

/// Moré-Thuente line search (MINPACK-2 dcsrch) along d from x: a step meeting the strong Wolfe conditions with
/// ftol 1e-3, gtol 0.9 (as L-BFGS-B), or the best point found within `max_eval` evaluations. A point that cannot be
/// evaluated pulls the step back toward the last good one. Returns (step, f, g) at the accepted point, None if no
/// decrease was found.
#[allow(clippy::too_many_arguments)]
fn line_search(obj: &dyn Objective, x: &[f64], d: &[f64], f0: f64, g0: &[f64], mut stp: f64, stpmax: f64, lb: &[f64], ub: &[f64],
               nfev: &mut usize, max_eval: usize) -> Option<(f64, f64, Vec<f64>)> {
    let (ftol, gtol, xtol, stpmin) = (1e-3, 0.9, 0.1, 0.0);
    let ginit = dot(g0, d);
    if !(ginit < 0.0) { return None; }
    let gtest = ftol * ginit;
    let mut brackt = false;
    let mut stage1 = true;
    let (mut width, mut width1) = (stpmax - stpmin, 2.0 * (stpmax - stpmin));
    let (mut stx, mut fx, mut gx) = (0.0, f0, ginit);
    let (mut sty, mut fy, mut gy) = (0.0, f0, ginit);
    let (mut stmin, mut stmax) = (0.0, stp + 4.0 * stp);
    let mut best: Option<(f64, f64, Vec<f64>)> = None;
    let point = |a: f64| -> Vec<f64> { (0..x.len()).map(|i| (x[i] + a * d[i]).clamp(lb[i], ub[i])).collect() };
    let mut fails = 0;
    for _ in 0..max_eval {
        *nfev += 1;
        let Some((f, g)) = obj.eval(&point(stp)).filter(|(f, g)| f.is_finite() && g.iter().all(|v| v.is_finite())) else {
            // cannot evaluate here: back toward the last good step
            fails += 1;
            if fails > 30 { break; }
            sty = stp; fy = f64::INFINITY; brackt = true;
            stp = stx + 0.25 * (stp - stx);
            stmin = stx.min(sty); stmax = stx.max(sty);
            continue;
        };
        let gd = dot(&g, d);
        if f < best.as_ref().map_or(f0, |b| b.1) { best = Some((stp, f, g.clone())); }
        let ftest = f0 + stp * gtest;
        if stage1 && f <= ftest && gd >= 0.0 { stage1 = false; }
        // converged, or at a limit the conditions cannot improve on
        if f <= ftest && gd.abs() <= gtol * (-ginit) { return Some((stp, f, g)); }
        if (brackt && (stp <= stmin || stp >= stmax)) || (brackt && stmax - stmin <= xtol * stmax) { break; }
        if stp == stpmax && f <= ftest && gd <= gtest { return Some((stp, f, g)); }
        if stp == stpmin && (f > ftest || gd >= gtest) { break; }
        if stage1 && f <= fx && f > ftest {
            // the modified function psi(stp) = f - f0 - stp * ftol * ginit while no step has met both conditions
            let (fm, gm) = (f - stp * gtest, gd - gtest);
            let (mut fxm, mut gxm, mut fym, mut gym) = (fx - stx * gtest, gx - gtest, fy - sty * gtest, gy - gtest);
            dcstep(&mut stx, &mut fxm, &mut gxm, &mut sty, &mut fym, &mut gym, &mut stp, fm, gm, &mut brackt, stmin, stmax);
            fx = fxm + stx * gtest; fy = fym + sty * gtest; gx = gxm + gtest; gy = gym + gtest;
        } else {
            dcstep(&mut stx, &mut fx, &mut gx, &mut sty, &mut fy, &mut gy, &mut stp, f, gd, &mut brackt, stmin, stmax);
        }
        if brackt {
            if (sty - stx).abs() >= 0.66 * width1 { stp = stx + 0.5 * (sty - stx); }
            width1 = width;
            width = (sty - stx).abs();
            stmin = stx.min(sty);
            stmax = stx.max(sty);
        } else {
            stmin = stp + 1.1 * (stp - stx);
            stmax = stp + 4.0 * (stp - stx);
        }
        stp = stp.max(stpmin).min(stpmax);
        if (brackt && (stp <= stmin || stp >= stmax)) || (brackt && stmax - stmin <= xtol * stmax) { stp = stx; }
    }
    // no Wolfe point within the budget: the best decrease found, if any
    best
}

/// Minimize f on [lb, ub] from x0.
pub fn lbfgsb(obj: &dyn Objective, x0: &[f64], lb: &[f64], ub: &[f64], o: &LbfgsbOptions) -> LbfgsbResult {
    let n = x0.len();
    let max_fev = if o.max_fev == 0 { 50 * o.max_iter } else { o.max_fev };
    let mut x: Vec<f64> = (0..n).map(|i| x0[i].clamp(lb[i], ub[i])).collect();
    let mut nfev = 1;
    let Some((mut f, mut g)) = obj.eval(&x).filter(|(f, g)| f.is_finite() && g.iter().all(|v| v.is_finite())) else {
        return LbfgsbResult { x, f: f64::INFINITY, g: vec![], nit: 0, nfev, status: LbfgsbStatus::Failed };
    };
    let mut mem = Memory::new(n);
    let mut nit = 0;
    let mut restarted = false;
    let status = loop {
        if proj_grad_norm(&x, &g, lb, ub) <= o.gtol { break LbfgsbStatus::Gtol; }
        if nit >= o.max_iter { break LbfgsbStatus::MaxIter; }
        if nfev >= max_fev { break LbfgsbStatus::MaxEval; }
        let (xcp, c) = cauchy(&x, &g, lb, ub, &mem);
        let xbar = subspace(&x, &g, lb, ub, &mem, &xcp, &c);
        let d: Vec<f64> = (0..n).map(|i| xbar[i] - x[i]).collect();
        let dn = dot(&d, &d).sqrt();
        if dn == 0.0 { break LbfgsbStatus::Gtol; }
        // largest feasible step along d (>= 1, since x + d is in the box)
        let mut stpmax = f64::INFINITY;
        for i in 0..n {
            if d[i] > 0.0 && ub[i].is_finite() { stpmax = stpmax.min((ub[i] - x[i]) / d[i]); }
            if d[i] < 0.0 && lb[i].is_finite() { stpmax = stpmax.min((lb[i] - x[i]) / d[i]); }
        }
        if !stpmax.is_finite() { stpmax = 1e10; }
        let stp0 = if mem.k() == 0 && nit == 0 { (1.0 / dn).min(stpmax).min(1.0) } else { 1.0f64.min(stpmax) };
        let budget = (max_fev - nfev).clamp(1, 20);
        match line_search(obj, &x, &d, f, &g, stp0, stpmax, lb, ub, &mut nfev, budget) {
            Some((stp, fnew, gnew)) => {
                let xnew: Vec<f64> = (0..n).map(|i| (x[i] + stp * d[i]).clamp(lb[i], ub[i])).collect();
                let s: Vec<f64> = (0..n).map(|i| xnew[i] - x[i]).collect();
                let y: Vec<f64> = (0..n).map(|i| gnew[i] - g[i]).collect();
                let fold = f;
                x = xnew; f = fnew; g = gnew;
                nit += 1;
                restarted = false;
                let sy = dot(&s, &y);
                if sy > f64::EPSILON * dot(&y, &y) { mem.push(s, y, o.m); }
                if (fold - f) / fold.abs().max(f.abs()).max(1.0) <= o.ftol { break LbfgsbStatus::Ftol; }
                if obj.cancel(nit, f) { break LbfgsbStatus::Cancelled; }
            }
            None => {
                // no decrease: forget the curvature pairs and retry once from steepest descent
                if restarted || mem.k() == 0 { break LbfgsbStatus::LineSearch; }
                mem.clear();
                restarted = true;
            }
        }
    };
    LbfgsbResult { x, f, g, nit, nfev, status }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct F<T: Fn(&[f64]) -> (f64, Vec<f64>)>(T);
    impl<T: Fn(&[f64]) -> (f64, Vec<f64>)> Objective for F<T> {
        fn eval(&self, x: &[f64]) -> Option<(f64, Vec<f64>)> { Some((self.0)(x)) }
    }

    fn rosen(x: &[f64]) -> (f64, Vec<f64>) {
        let n = x.len();
        let mut f = 0.0;
        let mut g = vec![0.0; n];
        for i in 0..n - 1 {
            let (a, b) = (x[i + 1] - x[i] * x[i], 1.0 - x[i]);
            f += 100.0 * a * a + b * b;
            g[i] += -400.0 * x[i] * a - 2.0 * b;
            g[i + 1] += 200.0 * a;
        }
        (f, g)
    }

    #[test]
    fn rosenbrock() {
        let inf = f64::INFINITY;
        let n = 10;
        let x0: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect();
        let r = lbfgsb(&F(rosen), &x0, &vec![-inf; n], &vec![inf; n], &LbfgsbOptions::default());
        assert!(matches!(r.status, LbfgsbStatus::Gtol | LbfgsbStatus::Ftol), "{r:?}");
        for v in &r.x { assert!((v - 1.0).abs() < 1e-5, "{:?}", r.x); }
        assert!(r.nit < 200, "{} iterations", r.nit);
    }

    #[test]
    fn bounded_quadratic() {
        // f = sum (x_i - c_i)^2 / (i + 1) with c outside the box for some i: the minimum is c clipped to the box
        let c = [3.0, -2.0, 0.5, 10.0, -0.3];
        let (lb, ub) = ([-1.0; 5], [1.0; 5]);
        let f = |x: &[f64]| {
            let f = (0..5).map(|i| (x[i] - c[i]).powi(2) / (i + 1) as f64).sum();
            (f, (0..5).map(|i| 2.0 * (x[i] - c[i]) / (i + 1) as f64).collect())
        };
        let r = lbfgsb(&F(f), &[0.0; 5], &lb, &ub, &LbfgsbOptions::default());
        for i in 0..5 { assert!((r.x[i] - c[i].clamp(-1.0, 1.0)).abs() < 1e-8, "{:?}", r.x); }
    }

    #[test]
    fn bounded_rosenbrock() {
        // scipy's example (minimize(rosen, [2, 2], bounds=[(-inf, inf), (1.5, inf)])): x = (1.2243707, 1.5)
        let inf = f64::INFINITY;
        let r = lbfgsb(&F(rosen), &[2.0, 2.0], &[-inf, 1.5], &[inf, inf], &LbfgsbOptions::default());
        assert!((r.x[0] - 1.22437075).abs() < 1e-6 && r.x[1] == 1.5, "{r:?}");
    }

    #[test]
    fn failed_points_are_avoided() {
        // f undefined for x0 > 2: the line search must back off
        struct G;
        impl Objective for G {
            fn eval(&self, x: &[f64]) -> Option<(f64, Vec<f64>)> {
                if x[0] > 2.0 { return None; }
                Some(((x[0] - 1.9).powi(2) + (x[1] + 1.0).powi(2), vec![2.0 * (x[0] - 1.9), 2.0 * (x[1] + 1.0)]))
            }
        }
        let inf = f64::INFINITY;
        let r = lbfgsb(&G, &[-5.0, 3.0], &[-inf, -inf], &[inf, inf], &LbfgsbOptions::default());
        assert!((r.x[0] - 1.9).abs() < 1e-6 && (r.x[1] + 1.0).abs() < 1e-6, "{r:?}");
    }
}
