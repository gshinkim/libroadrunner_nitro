//! Python binding of the fitting core: `roadrunner_nitro.core.FitProblem`. The public, documented layer is
//! python/roadrunner_nitro/fit.py; this class takes and returns plain values and numpy arrays.
use super::optimize::{fisher, fit, FitOptions};
use super::problem::{Measurement, Noise, ObservableDef, ParamDef, Problem, ProblemDef, Scale, Value};
use super::trf::Status;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Mutex;

fn verr(e: String) -> PyErr { PyValueError::new_err(e) }
fn rerr(e: String) -> PyErr { PyRuntimeError::new_err(e) }

fn arr<'py>(py: Python<'py>, v: Vec<f64>, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyArray1::from_vec(py, v).reshape(shape)?.into_any())
}

fn status_code(s: Status) -> i32 {
    match s { Status::Failed => 0, Status::MaxEval => 1, Status::Gtol => 2, Status::Ftol => 3, Status::Xtol => 4, Status::FtolXtol => 5, Status::Cancelled => 6, Status::LineSearch => 7 }
}

/// FitProblem(sbml): a fitting problem on an SBML model (text). FitProblem.from_petab(yaml_path) loads PEtab v1.
/// Compiled on first use and again after a change. Parameter vectors (theta) hold the free parameters in
/// `free_ids` order, each on its scale. Evaluation errors (a failed simulation) raise RuntimeError; bad input
/// raises ValueError.
#[pyclass(name = "FitProblem", module = "roadrunner_nitro.core")]
pub struct PyFitProblem { def: ProblemDef, p: Option<Problem>, tol: (f64, f64) }

impl PyFitProblem {
    fn get(&mut self) -> PyResult<&Problem> {
        if self.p.is_none() {
            let mut p = Problem::new(self.def.clone()).map_err(verr)?;
            (p.rtol, p.atol) = self.tol;
            self.p = Some(p);
        }
        Ok(self.p.as_ref().unwrap())
    }
    fn changed(&mut self) -> &mut ProblemDef { self.p = None; &mut self.def }
}

fn theta(p: &Problem, t: PyReadonlyArray1<f64>) -> PyResult<Vec<f64>> {
    let t: Vec<f64> = t.as_array().iter().copied().collect();
    if t.len() != p.n_free() { return Err(verr(format!("expected {} parameter values, got {}", p.n_free(), t.len()))); }
    Ok(t)
}

#[pymethods]
impl PyFitProblem {
    #[new]
    fn new(sbml: String) -> PyResult<Self> {
        crate::sbml::from_sbml(&sbml).map_err(|e| verr(e.to_string()))?;
        Ok(PyFitProblem { def: ProblemDef { sbml, ..Default::default() }, p: None, tol: (1e-8, 1e-12) })
    }

    #[staticmethod]
    fn from_petab(py: Python<'_>, path: String) -> PyResult<Self> {
        let def = py.detach(|| super::petab::load(std::path::Path::new(&path))).map_err(verr)?;
        let mut p = PyFitProblem { def, p: None, tol: (1e-8, 1e-12) };
        p.get()?;
        Ok(p)
    }

    #[pyo3(signature = (id, scale="lin", lower=f64::NAN, upper=f64::NAN, nominal=f64::NAN, estimate=true))]
    fn add_parameter(&mut self, id: String, scale: &str, lower: f64, upper: f64, nominal: f64, estimate: bool) -> PyResult<()> {
        if self.def.params.iter().any(|q| q.id == id) { return Err(verr(format!("parameter '{id}' already added"))); }
        let scale = Scale::parse(scale).map_err(verr)?;
        self.changed().params.push(ParamDef { id, scale, lb: lower, ub: upper, nominal, estimate });
        Ok(())
    }

    /// overrides: [(target, value)], value a number, a parameter id, or None / NaN (keep the model's value).
    #[pyo3(signature = (id, overrides=vec![]))]
    fn add_condition(&mut self, id: String, overrides: Vec<(String, Option<Bound<'_, PyAny>>)>) -> PyResult<()> {
        if self.def.conditions.iter().any(|c| c.0 == id) { return Err(verr(format!("condition '{id}' already added"))); }
        let mut e = vec![];
        for (t, v) in overrides {
            let v = match v {
                None => Value::Num(f64::NAN),
                Some(v) => match v.extract::<f64>() { Ok(x) => Value::Num(x), Err(_) => Value::Par(v.extract::<String>()?) },
            };
            e.push((t, v));
        }
        self.changed().conditions.push((id, e));
        Ok(())
    }

    #[pyo3(signature = (id, formula, noise_formula, transformation="lin", noise="normal"))]
    fn add_observable(&mut self, id: String, formula: String, noise_formula: String, transformation: &str, noise: &str) -> PyResult<()> {
        if self.def.observables.iter().any(|o| o.id == id) { return Err(verr(format!("observable '{id}' already added"))); }
        let o = ObservableDef { id, formula, noise_formula, transform: Scale::parse(transformation).map_err(verr)?, noise: Noise::parse(noise).map_err(verr)? };
        self.changed().observables.push(o);
        Ok(())
    }

    fn add_measurements(&mut self, condition: String, observable: String, times: Vec<f64>, values: Vec<f64>) -> PyResult<()> {
        if times.len() != values.len() { return Err(verr("times and values differ in length".into())); }
        self.changed().measurements.extend(times.into_iter().zip(values).map(|(time, value)|
            Measurement { condition: condition.clone(), observable: observable.clone(), time, value, ..Default::default() }));
        Ok(())
    }

    #[getter] fn rtol(&self) -> f64 { self.tol.0 }
    #[getter] fn atol(&self) -> f64 { self.tol.1 }
    #[setter] fn set_rtol(&mut self, v: f64) -> PyResult<()> { self.set_tol(v, self.tol.1) }
    #[setter] fn set_atol(&mut self, v: f64) -> PyResult<()> { self.set_tol(self.tol.0, v) }

    fn set_tol(&mut self, rtol: f64, atol: f64) -> PyResult<()> {
        if !(rtol > 0.0 && atol > 0.0) { return Err(verr("tolerances must be > 0".into())); }
        self.tol = (rtol, atol);
        if let Some(p) = self.p.as_mut() { (p.rtol, p.atol) = self.tol; }
        Ok(())
    }

    /// The model (SBML text).
    #[getter] fn sbml(&self) -> String { self.def.sbml.clone() }
    #[getter] fn free_ids(&mut self) -> PyResult<Vec<String>> { Ok(self.get()?.free_ids()) }
    #[getter] fn free_scales(&mut self) -> PyResult<Vec<String>> {
        let p = self.get()?;
        Ok(p.free.iter().map(|&j| p.params[j].scale.name().to_string()).collect())
    }
    #[getter] fn lower_bounds<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { let b = self.get()?.bounds().0; arr(py, b.clone(), &[b.len()]) }
    #[getter] fn upper_bounds<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { let b = self.get()?.bounds().1; arr(py, b.clone(), &[b.len()]) }
    #[getter] fn nominal_values<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { let b = self.get()?.nominal(); arr(py, b.clone(), &[b.len()]) }
    #[getter] fn least_squares(&mut self) -> PyResult<bool> { Ok(self.get()?.least_squares) }
    /// The parameter table: [(id, scale, lower, upper, nominal, estimate)], bounds and nominal linear.
    #[getter] fn parameters(&self) -> Vec<(String, String, f64, f64, f64, bool)> {
        self.def.params.iter().map(|p| (p.id.clone(), p.scale.name().into(), p.lb, p.ub, p.nominal, p.estimate)).collect()
    }
    /// The measurements: [(condition, observable, time, value)].
    #[getter] fn measurements(&self) -> Vec<(String, String, f64, f64)> {
        self.def.measurements.iter().map(|m| (m.condition.clone(), m.observable.clone(), m.time, m.value)).collect()
    }
    /// The observables: [(id, formula, noise formula, transformation, noise distribution)].
    #[getter] fn observables(&self) -> Vec<(String, String, String, String, String)> {
        self.def.observables.iter().map(|o| (o.id.clone(), o.formula.clone(), o.noise_formula.clone(), o.transform.name().into(),
            match o.noise { Noise::Normal => "normal".into(), Noise::Laplace => "laplace".into() })).collect()
    }
    #[getter] fn condition_ids(&self) -> Vec<String> { self.def.conditions.iter().map(|c| c.0.clone()).collect() }

    /// Negative log-likelihood at theta.
    fn objective(&mut self, py: Python<'_>, theta_: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let p = self.get()?;
        let t = theta(p, theta_)?;
        py.detach(|| p.evaluate(&t, false, false)).map(|e| e.nllh).map_err(rerr)
    }

    /// (negative log-likelihood, its gradient) at theta.
    fn gradient<'py>(&mut self, py: Python<'py>, theta_: PyReadonlyArray1<f64>) -> PyResult<(f64, Bound<'py, PyAny>)> {
        let p = self.get()?;
        let t = theta(p, theta_)?;
        let e = py.detach(|| p.evaluate(&t, true, false)).map_err(rerr)?;
        let n = e.grad.len();
        Ok((e.nllh, arr(py, e.grad, &[n])?))
    }

    fn residuals<'py>(&mut self, py: Python<'py>, theta_: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyAny>> {
        let p = self.get()?;
        let t = theta(p, theta_)?;
        let e = py.detach(|| p.evaluate(&t, false, true)).map_err(rerr)?;
        let m = e.res.len();
        arr(py, e.res, &[m])
    }

    /// (residuals, Jacobian [measurements, free parameters]).
    fn residual_jacobian<'py>(&mut self, py: Python<'py>, theta_: PyReadonlyArray1<f64>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let p = self.get()?;
        let (t, n) = (theta(p, theta_)?, p.n_free());
        let e = py.detach(|| p.evaluate(&t, false, true)).map_err(rerr)?;
        let m = e.res.len();
        Ok((arr(py, e.res, &[m])?, arr(py, e.jac, &[m, n])?))
    }

    fn simulate_measurements<'py>(&mut self, py: Python<'py>, theta_: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyAny>> {
        let p = self.get()?;
        let t = theta(p, theta_)?;
        let y = py.detach(|| p.simulate_measurements(&t)).map_err(rerr)?;
        let m = y.len();
        arr(py, y, &[m])
    }

    /// Multistart least squares. callback(finished, total, best) is called after each start (from a worker
    /// thread, holding the GIL); a true return value cancels. Returns a dict of per-start results, best first.
    #[pyo3(signature = (n_starts=10, seed=0, threads=0, x0=None, ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=0, callback=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit<'py>(&mut self, py: Python<'py>, n_starts: usize, seed: u64, threads: usize, x0: Option<Vec<f64>>, ftol: f64, xtol: f64,
                gtol: f64, max_nfev: usize, callback: Option<Py<PyAny>>) -> PyResult<Bound<'py, PyDict>> {
        if n_starts == 0 { return Err(verr("n_starts must be >= 1".into())); }
        let p = self.get()?;
        let mut o = FitOptions { n_starts, seed, threads, x0, ..Default::default() };
        (o.trf.ftol, o.trf.xtol, o.trf.gtol, o.trf.max_nfev) = (ftol, xtol, gtol, max_nfev);
        let cb_err: Mutex<Option<PyErr>> = Mutex::new(None);
        let progress = |done: usize, total: usize, best: f64| -> bool {
            let Some(cb) = &callback else { return false };
            Python::attach(|py| match cb.call1(py, (done, total, best)).and_then(|r| r.is_truthy(py)) {
                Ok(stop) => stop,
                Err(e) => { cb_err.lock().unwrap().get_or_insert(e); true }
            })
        };
        let r = py.detach(|| fit(p, &o, callback.is_some().then_some(&progress as &(dyn Fn(usize, usize, f64) -> bool + Sync))))
            .map_err(verr)?;
        if let Some(e) = cb_err.into_inner().unwrap() { return Err(e); }
        let (ns, n) = (r.starts.len(), r.ids.len());
        let d = PyDict::new(py);
        d.set_item("ids", r.ids.clone())?;
        d.set_item("start", r.starts.iter().map(|s| s.start).collect::<Vec<_>>())?;
        d.set_item("x", arr(py, r.starts.iter().flat_map(|s| s.x.clone()).collect(), &[ns, n])?)?;
        d.set_item("x0", arr(py, r.starts.iter().flat_map(|s| s.x0.clone()).collect(), &[ns, n])?)?;
        d.set_item("value", arr(py, r.starts.iter().map(|s| s.nllh).collect(), &[ns])?)?;
        d.set_item("status", r.starts.iter().map(|s| status_code(s.status)).collect::<Vec<_>>())?;
        d.set_item("message", r.starts.iter().map(|s| s.status.message()).collect::<Vec<_>>())?;
        d.set_item("converged", r.starts.iter().map(|s| s.status.converged()).collect::<Vec<_>>())?;
        d.set_item("nfev", r.starts.iter().map(|s| s.nfev).collect::<Vec<_>>())?;
        d.set_item("njev", r.starts.iter().map(|s| s.njev).collect::<Vec<_>>())?;
        d.set_item("nit", r.starts.iter().map(|s| s.nit).collect::<Vec<_>>())?;
        Ok(d)
    }

    /// Fisher-information (Wald) intervals at theta: a dict of x, sd, lower, upper, identifiable, cov, corr.
    #[pyo3(signature = (theta_, level=0.95))]
    fn fisher<'py>(&mut self, py: Python<'py>, theta_: PyReadonlyArray1<f64>, level: f64) -> PyResult<Bound<'py, PyDict>> {
        let p = self.get()?;
        let t = theta(p, theta_)?;
        let f = py.detach(|| fisher(p, &t, level)).map_err(rerr)?;
        let n = f.x.len();
        let d = PyDict::new(py);
        d.set_item("ids", f.ids)?;
        d.set_item("x", arr(py, f.x, &[n])?)?;
        d.set_item("sd", arr(py, f.sd, &[n])?)?;
        d.set_item("lower", arr(py, f.lower, &[n])?)?;
        d.set_item("upper", arr(py, f.upper, &[n])?)?;
        d.set_item("identifiable", f.identifiable)?;
        d.set_item("cov", arr(py, f.cov, &[n, n])?)?;
        d.set_item("corr", arr(py, f.corr, &[n, n])?)?;
        d.set_item("level", level)?;
        Ok(d)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_class::<PyFitProblem>() }
