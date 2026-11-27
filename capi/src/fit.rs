//! rrc_fit.h: parameter estimation (roadrunner_nitro::fit) through the C API. Handles are boxed Rust values;
//! errors go to getLastError and no panic crosses the FFI boundary (api::run).
use crate::api::{inst, run, st};
use crate::types::*;
use libc::{c_char, c_int, c_void};
use roadrunner_nitro::fit::trf::Status;
use roadrunner_nitro::fit::{fisher, fit, petab, FitOptions, FitResult, Measurement, Noise, ObservableDef, ParamDef, Problem, ProblemDef, Scale, Value};
use std::ptr::null_mut;

type R<T> = Result<T, String>;

pub type FitProblemHandle = *mut c_void;
pub type FitOptionsHandle = *mut c_void;
pub type FitResultHandle = *mut c_void;
pub type FitProgressCallback = Option<unsafe extern "C" fn(c_int, c_int, f64, *mut c_void) -> c_int>;

/// A problem: its definition, compiled on first use and again after a change.
struct Prob { def: ProblemDef, p: Option<Problem>, tol: (f64, f64) }

impl Prob {
    fn new(def: ProblemDef) -> Prob { Prob { def, p: None, tol: (1e-8, 1e-12) } }
    fn get(&mut self) -> R<&Problem> {
        if self.p.is_none() {
            let mut p = Problem::new(self.def.clone())?;
            (p.rtol, p.atol) = self.tol;
            self.p = Some(p);
        }
        Ok(self.p.as_ref().unwrap())
    }
    fn changed(&mut self) -> &mut ProblemDef { self.p = None; &mut self.def }
}

struct Res { r: FitResult }

unsafe fn prob<'a>(h: FitProblemHandle) -> R<&'a mut Prob> { (h as *mut Prob).as_mut().ok_or_else(|| "NULL fitting problem handle".to_string()) }
unsafe fn opts<'a>(h: FitOptionsHandle) -> R<&'a mut FitOptions> { (h as *mut FitOptions).as_mut().ok_or_else(|| "NULL options handle".to_string()) }
unsafe fn res<'a>(h: FitResultHandle) -> R<&'a Res> { (h as *const Res).as_ref().ok_or_else(|| "NULL result handle".to_string()) }

/// An optional C string (NULL: "").
unsafe fn opt_st(p: *const c_char) -> String { cstr(p).unwrap_or_default() }

/// n doubles from a C array.
unsafe fn doubles(p: *const f64, n: c_int) -> R<Vec<f64>> {
    if n < 0 { return Err("negative length".into()); }
    if n == 0 { return Ok(vec![]); }
    if p.is_null() { return Err("NULL array argument".into()); }
    Ok(std::slice::from_raw_parts(p, n as usize).to_vec())
}

/// theta of the problem's length.
unsafe fn theta(pr: &Problem, p: *const f64, n: c_int) -> R<Vec<f64>> {
    let t = doubles(p, n)?;
    if t.len() != pr.n_free() { return Err(format!("expected {} parameter values, got {}", pr.n_free(), t.len())); }
    Ok(t)
}

fn status_code(s: Status) -> c_int {
    match s { Status::Failed => 0, Status::MaxEval => 1, Status::Gtol => 2, Status::Ftol => 3, Status::Xtol => 4, Status::FtolXtol => 5, Status::Cancelled => 6, Status::LineSearch => 7 }
}

fn boxed<T>(v: T) -> *mut c_void { Box::into_raw(Box::new(v)) as *mut c_void }

// ---------------------------------------------------------------- problem

#[no_mangle]
pub unsafe extern "C" fn fitLoadPEtab(yamlPath: *const c_char) -> FitProblemHandle {
    run(null_mut(), || {
        let def = petab::load(std::path::Path::new(&st(yamlPath)?))?;
        let mut p = Prob::new(def);
        p.get()?; // report problems with the files now
        Ok(boxed(p))
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitCreateProblem(rr: RRHandle) -> FitProblemHandle {
    run(null_mut(), || {
        let i = inst(rr)?;
        if !i.rr.is_model_loaded() { return Err("A model needs to be loaded before one can use this method".into()); }
        Ok(boxed(Prob::new(ProblemDef { sbml: i.rr.get_sbml()?, ..Default::default() })))
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitCreateProblemFromSBML(sbml: *const c_char) -> FitProblemHandle {
    run(null_mut(), || {
        let sbml = st(sbml)?;
        roadrunner_nitro::sbml::from_sbml(&sbml).map_err(|e| e.to_string())?;
        Ok(boxed(Prob::new(ProblemDef { sbml, ..Default::default() })))
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitAddParameter(p: FitProblemHandle, id: *const c_char, scale: *const c_char, lowerBound: f64,
                                         upperBound: f64, nominalValue: f64, estimate: bool) -> bool {
    run(false, || {
        let pr = prob(p)?;
        let id = st(id)?;
        if pr.def.params.iter().any(|q| q.id == id) { return Err(format!("parameter '{id}' already added")); }
        let scale = Scale::parse(&opt_st(scale))?;
        pr.changed().params.push(ParamDef { id, scale, lb: lowerBound, ub: upperBound, nominal: nominalValue, estimate });
        Ok(true)
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitAddCondition(p: FitProblemHandle, id: *const c_char, n: c_int, targets: *const *const c_char,
                                         values: *const *const c_char) -> bool {
    run(false, || {
        let pr = prob(p)?;
        let id = st(id)?;
        if pr.def.conditions.iter().any(|c| c.0 == id) { return Err(format!("condition '{id}' already added")); }
        if n < 0 || (n > 0 && (targets.is_null() || values.is_null())) { return Err("bad condition arrays".into()); }
        let mut e = vec![];
        for k in 0..n as usize {
            let t = st(*targets.add(k))?;
            let v = opt_st(*values.add(k));
            let v = v.trim();
            e.push((t, match v.parse::<f64>() {
                Ok(x) => Value::Num(x),
                Err(_) if v.is_empty() || v.eq_ignore_ascii_case("nan") => Value::Num(f64::NAN),
                Err(_) => Value::Par(v.to_string()),
            }));
        }
        pr.changed().conditions.push((id, e));
        Ok(true)
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitAddObservable(p: FitProblemHandle, id: *const c_char, formula: *const c_char, noiseFormula: *const c_char,
                                          transformation: *const c_char, noiseDistribution: *const c_char) -> bool {
    run(false, || {
        let pr = prob(p)?;
        let id = st(id)?;
        if pr.def.observables.iter().any(|o| o.id == id) { return Err(format!("observable '{id}' already added")); }
        let o = ObservableDef { id, formula: st(formula)?, noise_formula: st(noiseFormula)?,
                                transform: Scale::parse(&opt_st(transformation))?, noise: Noise::parse(&opt_st(noiseDistribution))? };
        pr.changed().observables.push(o);
        Ok(true)
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitAddMeasurements(p: FitProblemHandle, conditionId: *const c_char, observableId: *const c_char, n: c_int,
                                            times: *const f64, values: *const f64) -> bool {
    run(false, || {
        let pr = prob(p)?;
        let (c, o) = (st(conditionId)?, st(observableId)?);
        let (t, v) = (doubles(times, n)?, doubles(values, n)?);
        pr.changed().measurements.extend(t.into_iter().zip(v).map(|(time, value)| Measurement { condition: c.clone(), observable: o.clone(), time, value, ..Default::default() }));
        Ok(true)
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitAddPreequilibratedMeasurements(p: FitProblemHandle, preequilibrationId: *const c_char, conditionId: *const c_char,
                                                           observableId: *const c_char, n: c_int, times: *const f64, values: *const f64) -> bool {
    run(false, || {
        let pr = prob(p)?;
        let (pc, c, o) = (st(preequilibrationId)?, st(conditionId)?, st(observableId)?);
        let (t, v) = (doubles(times, n)?, doubles(values, n)?);
        pr.changed().measurements.extend(t.into_iter().zip(v).map(|(time, value)|
            Measurement { condition: c.clone(), preeq_condition: pc.clone(), observable: o.clone(), time, value, ..Default::default() }));
        Ok(true)
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitSetTolerances(p: FitProblemHandle, rtol: f64, atol: f64) -> bool {
    run(false, || {
        let pr = prob(p)?;
        if !(rtol > 0.0 && atol > 0.0) { return Err("tolerances must be > 0".into()); }
        pr.tol = (rtol, atol);
        if let Some(q) = pr.p.as_mut() { (q.rtol, q.atol) = (rtol, atol); }
        Ok(true)
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitFreeProblem(p: FitProblemHandle) -> bool {
    run(false, || { if !p.is_null() { drop(Box::from_raw(p as *mut Prob)); } Ok(true) })
}

#[no_mangle]
pub unsafe extern "C" fn fitGetNumFreeParameters(p: FitProblemHandle) -> c_int {
    run(-1, || Ok(prob(p)?.get()?.n_free() as c_int))
}

#[no_mangle]
pub unsafe extern "C" fn fitGetFreeParameterIds(p: FitProblemHandle) -> *mut RRStringArray {
    run(null_mut(), || Ok(string_array(&prob(p)?.get()?.free_ids())))
}

#[no_mangle]
pub unsafe extern "C" fn fitGetFreeParameterScales(p: FitProblemHandle) -> *mut RRStringArray {
    run(null_mut(), || {
        let q = prob(p)?.get()?;
        Ok(string_array(&q.free.iter().map(|&j| q.params[j].scale.name()).collect::<Vec<_>>()))
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitGetLowerBounds(p: FitProblemHandle) -> *mut RRVector { run(null_mut(), || Ok(vector(&prob(p)?.get()?.bounds().0))) }

#[no_mangle]
pub unsafe extern "C" fn fitGetUpperBounds(p: FitProblemHandle) -> *mut RRVector { run(null_mut(), || Ok(vector(&prob(p)?.get()?.bounds().1))) }

#[no_mangle]
pub unsafe extern "C" fn fitGetNominalValues(p: FitProblemHandle) -> *mut RRVector { run(null_mut(), || Ok(vector(&prob(p)?.get()?.nominal()))) }

#[no_mangle]
pub unsafe extern "C" fn fitIsLeastSquares(p: FitProblemHandle) -> c_int { run(-1, || Ok(prob(p)?.get()?.least_squares as c_int)) }

#[no_mangle]
pub unsafe extern "C" fn fitGetNumMeasurements(p: FitProblemHandle) -> c_int { run(-1, || Ok(prob(p)?.get()?.measurements.len() as c_int)) }

// ---------------------------------------------------------------- objective

#[no_mangle]
pub unsafe extern "C" fn fitObjective(p: FitProblemHandle, theta_: *const f64, n: c_int, value: *mut f64) -> bool {
    run(false, || {
        let q = prob(p)?.get()?;
        if value.is_null() { return Err("NULL output argument".into()); }
        *value = q.evaluate(&theta(q, theta_, n)?, false, false)?.nllh;
        Ok(true)
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitGradient(p: FitProblemHandle, theta_: *const f64, n: c_int, value: *mut f64) -> *mut RRVector {
    run(null_mut(), || {
        let q = prob(p)?.get()?;
        let e = q.evaluate(&theta(q, theta_, n)?, true, false)?;
        if !value.is_null() { *value = e.nllh; }
        Ok(vector(&e.grad))
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitResiduals(p: FitProblemHandle, theta_: *const f64, n: c_int) -> *mut RRVector {
    run(null_mut(), || {
        let q = prob(p)?.get()?;
        Ok(vector(&q.evaluate(&theta(q, theta_, n)?, false, true)?.res))
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitResidualJacobian(p: FitProblemHandle, theta_: *const f64, n: c_int) -> *mut RRDoubleMatrix {
    run(null_mut(), || {
        let q = prob(p)?.get()?;
        let e = q.evaluate(&theta(q, theta_, n)?, false, true)?;
        Ok(matrix(e.res.len(), q.n_free(), &e.jac))
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitSimulateMeasurements(p: FitProblemHandle, theta_: *const f64, n: c_int) -> *mut RRVector {
    run(null_mut(), || {
        let q = prob(p)?.get()?;
        Ok(vector(&q.simulate_measurements(&theta(q, theta_, n)?)?))
    })
}

// ---------------------------------------------------------------- optimization

#[no_mangle]
pub extern "C" fn fitCreateOptions() -> FitOptionsHandle { run(null_mut(), || Ok(boxed(FitOptions::default()))) }

#[no_mangle]
pub unsafe extern "C" fn fitSetOption(o: FitOptionsHandle, name: *const c_char, value: f64) -> bool {
    run(false, || {
        let o = opts(o)?;
        let name = st(name)?;
        let count = |what: &str| -> R<usize> {
            if value >= 0.0 && value.fract() == 0.0 && value < 1e15 { Ok(value as usize) } else { Err(format!("{what} must be a whole number >= 0")) }
        };
        let tol = || -> R<f64> { if value >= 0.0 { Ok(value) } else { Err(format!("{name} must be >= 0")) } };
        match name.as_str() {
            "n_starts" => { let v = count("n_starts")?; if v == 0 { return Err("n_starts must be >= 1".into()); } o.n_starts = v }
            "seed" => o.seed = count("seed")? as u64,
            "threads" => o.threads = count("threads")?,
            "ftol" => o.trf.ftol = tol()?,
            "xtol" => o.trf.xtol = tol()?,
            "gtol" => o.trf.gtol = tol()?,
            "max_nfev" => o.trf.max_nfev = count("max_nfev")?,
            _ => return Err(format!("unknown option '{name}' (n_starts, seed, threads, ftol, xtol, gtol, max_nfev)")),
        }
        Ok(true)
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitSetStartPoint(o: FitOptionsHandle, theta_: *const f64, n: c_int) -> bool {
    run(false, || {
        let o = opts(o)?;
        let t = doubles(theta_, n)?;
        o.x0 = (!t.is_empty()).then_some(t);
        Ok(true)
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitFreeOptions(o: FitOptionsHandle) -> bool {
    run(false, || { if !o.is_null() { drop(Box::from_raw(o as *mut FitOptions)); } Ok(true) })
}

/// The callback and its user pointer, shared with the worker threads (calls are serialized by fit()).
struct Cb { f: unsafe extern "C" fn(c_int, c_int, f64, *mut c_void) -> c_int, user: *mut c_void }
unsafe impl Sync for Cb {}
unsafe impl Send for Cb {}

#[no_mangle]
pub unsafe extern "C" fn fitRun(p: FitProblemHandle, o: FitOptionsHandle, callback: FitProgressCallback, user: *mut c_void) -> FitResultHandle {
    run(null_mut(), || {
        let q = prob(p)?.get()?;
        let def = FitOptions::default();
        let o = if o.is_null() { &def } else { &*opts(o)? };
        let cb = callback.map(|f| Cb { f, user });
        let progress = |done: usize, total: usize, best: f64| -> bool {
            let c = cb.as_ref().unwrap();
            unsafe { (c.f)(done as c_int, total as c_int, best, c.user) != 0 }
        };
        let r = fit(q, o, cb.is_some().then_some(&progress as &(dyn Fn(usize, usize, f64) -> bool + Sync)))?;
        Ok(boxed(Res { r }))
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitGetNumStarts(r: FitResultHandle) -> c_int { run(-1, || Ok(res(r)?.r.starts.len() as c_int)) }

#[no_mangle]
pub unsafe extern "C" fn fitGetBest(r: FitResultHandle, value: *mut f64) -> *mut RRVector {
    run(null_mut(), || {
        let b = res(r)?.r.best();
        if !value.is_null() { *value = b.nllh; }
        Ok(vector(&b.x))
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitGetStarts(r: FitResultHandle) -> *mut RRDoubleMatrix {
    run(null_mut(), || {
        let r = &res(r)?.r;
        let n = r.ids.len();
        let data: Vec<f64> = r.starts.iter().flat_map(|s| {
            [s.start as f64, s.nllh, status_code(s.status) as f64, s.nfev as f64, s.njev as f64, s.nit as f64].into_iter().chain(s.x.iter().copied())
        }).collect();
        Ok(matrix(r.starts.len(), 6 + n, &data))
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitGetStartPoints(r: FitResultHandle) -> *mut RRDoubleMatrix {
    run(null_mut(), || {
        let r = &res(r)?.r;
        Ok(matrix(r.starts.len(), r.ids.len(), &r.starts.iter().flat_map(|s| s.x0.iter().copied()).collect::<Vec<_>>()))
    })
}

#[no_mangle]
pub extern "C" fn fitGetStatusMessage(status: c_int) -> *const c_char {
    let s: &'static std::ffi::CStr = match status {
        0 => c"the starting point could not be evaluated",
        1 => c"maximum number of function evaluations reached",
        2 => c"gradient small enough (gtol)",
        3 => c"objective change small enough (ftol)",
        4 => c"step small enough (xtol)",
        5 => c"objective change and step small enough (ftol, xtol)",
        6 => c"cancelled",
        7 => c"no further decrease along the search direction (L-BFGS-B)",
        _ => c"unknown status",
    };
    s.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn fitFreeResult(r: FitResultHandle) -> bool {
    run(false, || { if !r.is_null() { drop(Box::from_raw(r as *mut Res)); } Ok(true) })
}

// ---------------------------------------------------------------- uncertainty

#[no_mangle]
pub unsafe extern "C" fn fitFisherIntervals(p: FitProblemHandle, theta_: *const f64, n: c_int, level: f64) -> *mut RRDoubleMatrix {
    run(null_mut(), || {
        let q = prob(p)?.get()?;
        let f = fisher(q, &theta(q, theta_, n)?, level)?;
        let k = f.x.len();
        let data: Vec<f64> = (0..k).flat_map(|i| [f.x[i], f.sd[i], f.lower[i], f.upper[i], f.identifiable[i] as u8 as f64]).collect();
        Ok(matrix(k, 5, &data))
    })
}

#[no_mangle]
pub unsafe extern "C" fn fitFisherCovariance(p: FitProblemHandle, theta_: *const f64, n: c_int) -> *mut RRDoubleMatrix {
    run(null_mut(), || {
        let q = prob(p)?.get()?;
        let f = fisher(q, &theta(q, theta_, n)?, 0.95)?;
        let k = f.x.len();
        Ok(matrix(k, k, &f.cov))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{createRRInstance, freeRRInstance, getLastError, loadSBML};
    use crate::utilities::{freeMatrix, freeStringArray, freeVector};
    use std::ffi::{CStr, CString};

    const ANT: &str = "model decay
        compartment C = 1
        species S1 in C, S2 in C
        J1: S1 -> S2; k1*S1
        J2: S2 -> ; k2*S2
        S1 = 10; S2 = 0
        k1 = 0.5; k2 = 0.2
    end";

    fn c(s: &str) -> CString { CString::new(s).unwrap() }
    unsafe fn err() -> String { CStr::from_ptr(getLastError()).to_string_lossy().into_owned() }
    unsafe fn vec_of(v: *mut RRVector) -> Vec<f64> {
        assert!(!v.is_null(), "{}", err());
        let r = std::slice::from_raw_parts((*v).Data, (*v).Count as usize).to_vec();
        freeVector(v);
        r
    }
    unsafe fn mat_of(m: *mut RRDoubleMatrix) -> (usize, usize, Vec<f64>) {
        assert!(!m.is_null(), "{}", err());
        let (r, k) = ((*m).RSize as usize, (*m).CSize as usize);
        let d = std::slice::from_raw_parts((*m).Data, r * k).to_vec();
        freeMatrix(m);
        (r, k, d)
    }

    /// The decay problem built through the C API on a RoadRunner instance: two conditions (S1(0) estimated in the
    /// second), observables S1 and scale * S2, data from k1 = 0.4, k2 = 0.25, scale = 3, S1(0) = 20.
    unsafe fn build() -> FitProblemHandle {
        let rr = createRRInstance();
        let sbml = c(&antimony::sbml::write_sbml(&antimony::parse(ANT).document));
        assert!(loadSBML(rr, sbml.as_ptr()), "{}", err());
        let p = fitCreateProblem(rr);
        assert!(!p.is_null(), "{}", err());
        freeRRInstance(rr); // the problem keeps its own copy of the model
        let lin = c("lin");
        for (id, sc, lb, ub, nom, est) in [("k1", "log10", 1e-3, 1e2, 0.5, true), ("k2", "log10", 1e-3, 1e2, 0.2, true),
                                           ("scale", "log", 0.1, 10.0, 2.0, true), ("S1_init", "lin", 1.0, 50.0, 15.0, true)] {
            assert!(fitAddParameter(p, c(id).as_ptr(), c(sc).as_ptr(), lb, ub, nom, est), "{}", err());
        }
        assert!(fitAddCondition(p, c("c1").as_ptr(), 0, std::ptr::null(), std::ptr::null()));
        let (t, v) = (c("S1"), c("S1_init"));
        assert!(fitAddCondition(p, c("c2").as_ptr(), 1, [t.as_ptr()].as_ptr(), [v.as_ptr()].as_ptr()));
        assert!(fitAddObservable(p, c("A").as_ptr(), c("S1").as_ptr(), c("0.1").as_ptr(), lin.as_ptr(), std::ptr::null()));
        assert!(fitAddObservable(p, c("B").as_ptr(), c("scale*S2").as_ptr(), c("0.05").as_ptr(), c("log10").as_ptr(), c("normal").as_ptr()));
        let times = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
        for (cond, s10) in [("c1", 10.0), ("c2", 20.0)] {
            let (k1, k2) = (0.4f64, 0.25f64);
            let a: Vec<f64> = times.iter().map(|t| s10 * (-k1 * t).exp()).collect();
            let b: Vec<f64> = times.iter().map(|t| 3.0 * s10 * k1 / (k2 - k1) * ((-k1 * t).exp() - (-k2 * t).exp())).collect();
            assert!(fitAddMeasurements(p, c(cond).as_ptr(), c("A").as_ptr(), 7, times.as_ptr(), a.as_ptr()));
            assert!(fitAddMeasurements(p, c(cond).as_ptr(), c("B").as_ptr(), 7, times.as_ptr(), b.as_ptr()));
        }
        p
    }

    #[test]
    fn problem_objective_and_fit() {
        unsafe {
            let p = build();
            assert_eq!(fitGetNumFreeParameters(p), 4);
            assert_eq!(fitGetNumMeasurements(p), 28);
            assert_eq!(fitIsLeastSquares(p), 1);
            let ids = fitGetFreeParameterIds(p);
            assert_eq!((*ids).Count, 4);
            assert_eq!(CStr::from_ptr(*(*ids).String.add(2)).to_str().unwrap(), "scale");
            freeStringArray(ids);
            assert_eq!(vec_of(fitGetLowerBounds(p)), [-3.0, -3.0, 0.1f64.ln(), 1.0]);
            // objective and gradient against the Rust core, through a finite difference
            let th = [0.6f64.log10(), 0.15f64.log10(), 2.5f64.ln(), 12.0];
            let mut f = 0.0;
            assert!(fitObjective(p, th.as_ptr(), 4, &mut f));
            let mut f2 = 0.0;
            let g = vec_of(fitGradient(p, th.as_ptr(), 4, &mut f2));
            assert_eq!(f.to_bits(), f2.to_bits());
            let h = 1e-6;
            let mut tp = th;
            tp[0] += h;
            let mut fp = 0.0;
            assert!(fitObjective(p, tp.as_ptr(), 4, &mut fp));
            assert!(((fp - f) / h - g[0]).abs() < 1e-3 * g[0].abs(), "{} vs {}", (fp - f) / h, g[0]);
            let r = vec_of(fitResiduals(p, th.as_ptr(), 4));
            assert!((0.5 * r.iter().map(|x| x * x).sum::<f64>() - f).abs() < 30.0); // f adds constant terms
            let (m, n, _) = mat_of(fitResidualJacobian(p, th.as_ptr(), 4));
            assert_eq!((m, n), (28, 4));
            // wrong length: an error, not a crash
            assert!(!fitObjective(p, th.as_ptr(), 3, &mut f));
            assert!(err().contains("expected 4 parameter values"), "{}", err());

            // fit with progress and options
            let o = fitCreateOptions();
            assert!(fitSetOption(o, c("n_starts").as_ptr(), 8.0) && fitSetOption(o, c("seed").as_ptr(), 3.0));
            assert!(!fitSetOption(o, c("bogus").as_ptr(), 1.0));
            unsafe extern "C" fn cb(done: c_int, total: c_int, _best: f64, user: *mut c_void) -> c_int {
                assert!(done >= 1 && done <= total);
                *(user as *mut c_int) += 1;
                0
            }
            let mut calls: c_int = 0;
            let r = fitRun(p, o, Some(cb), &mut calls as *mut c_int as *mut c_void);
            assert!(!r.is_null(), "{}", err());
            assert_eq!(calls, 8);
            assert_eq!(fitGetNumStarts(r), 8);
            let mut best = 0.0;
            let x = vec_of(fitGetBest(r, &mut best));
            let truth = [0.4f64.log10(), 0.25f64.log10(), 3f64.ln(), 20.0];
            for k in 0..4 { assert!((x[k] - truth[k]).abs() < 1e-4 * truth[k].abs().max(1.0), "{x:?}"); }
            let (ns, nc, starts) = mat_of(fitGetStarts(r));
            assert_eq!((ns, nc), (8, 10));
            assert_eq!(starts[1].to_bits(), best.to_bits());
            assert!((2..=5).contains(&(starts[2] as c_int)), "status {}", starts[2]);
            assert_eq!(CStr::from_ptr(fitGetStatusMessage(6)).to_str().unwrap(), "cancelled");
            let (fr, fc, fi) = mat_of(fitFisherIntervals(p, x.as_ptr(), 4, 0.95));
            assert_eq!((fr, fc), (4, 5));
            for k in 0..4 { assert!(fi[k * 5 + 2] < x[k] && x[k] < fi[k * 5 + 3] && fi[k * 5 + 4] == 1.0); }
            assert!(fitFreeResult(r));

            // cancel after the first start
            unsafe extern "C" fn stop(_d: c_int, _t: c_int, _b: f64, _u: *mut c_void) -> c_int { 1 }
            assert!(fitSetOption(o, c("threads").as_ptr(), 1.0));
            let r = fitRun(p, o, Some(stop), std::ptr::null_mut());
            let (_, _, s) = mat_of(fitGetStarts(r));
            assert_eq!(s.chunks(10).filter(|row| row[2] == 6.0).count(), 7);
            assert!(fitFreeResult(r) && fitFreeOptions(o) && fitFreeProblem(p));
        }
    }

    #[test]
    fn errors_are_reported() {
        unsafe {
            assert!(fitLoadPEtab(c("no/such/problem.yaml").as_ptr()).is_null());
            assert!(err().contains("PEtab YAML"), "{}", err());
            assert!(fitCreateProblemFromSBML(c("<not sbml").as_ptr()).is_null());
            let p = build();
            assert!(!fitAddParameter(p, c("k1").as_ptr(), std::ptr::null(), 0.0, 1.0, 0.5, true));
            assert!(err().contains("already added"));
            assert!(fitAddObservable(p, c("bad").as_ptr(), c("nope*S1").as_ptr(), c("1").as_ptr(), std::ptr::null(), std::ptr::null()));
            assert_eq!(fitGetNumFreeParameters(p), -1); // compiling fails now
            assert!(err().contains("unknown symbol 'nope'"), "{}", err());
            assert!(fitFreeProblem(p));
            assert_eq!(fitGetNumFreeParameters(std::ptr::null_mut()), -1);
        }
    }
}
