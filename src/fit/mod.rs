//! Parameter estimation (spec: "Spec: parameter estimation in roadrunner_nitro", stage 1).
//!
//! - `problem`: problem definition (the native API; PEtab import produces the same `ProblemDef`), compiled once;
//!   negative log-likelihood, gradient, residuals and residual Jacobian with exact forward sensitivities.
//! - `trf`: trust-region reflective least squares with bounds.
//! - `optimize`: Latin hypercube multistart and Fisher-information intervals.
pub mod lbfgsb;
pub mod optimize;
pub mod petab;
pub mod problem;
#[cfg(feature = "python")]
pub mod py;
pub mod trf;

pub use optimize::{fisher, fit, FitOptions, FitResult, Fisher, Method, StartResult};
pub use problem::{Eval, Measurement, Noise, ObservableDef, ParamDef, Problem, ProblemDef, Scale, Value};

#[cfg(test)]
mod tests {
    use super::*;

    fn sbml(ant: &str) -> String { antimony::sbml::write_sbml(&antimony::parse(ant).document) }

    fn par(id: &str, scale: Scale, lb: f64, ub: f64, nominal: f64, estimate: bool) -> ParamDef {
        ParamDef { id: id.into(), scale, lb, ub, nominal, estimate }
    }
    fn obs(id: &str, f: &str, noise: &str, dist: Noise, tr: Scale) -> ObservableDef {
        ObservableDef { id: id.into(), formula: f.into(), noise_formula: noise.into(), noise: dist, transform: tr }
    }

    const DECAY: &str = "model decay
        compartment C = 1
        species S1 in C, S2 in C
        J1: S1 -> S2; k1*S1
        J2: S2 -> ; k2*S2
        S1 = 10; S2 = 0
        k1 = 0.5; k2 = 0.2
    end";

    /// Analytic S1, S2 of DECAY.
    fn decay(t: f64, s10: f64, k1: f64, k2: f64) -> (f64, f64) {
        let a = s10 * (-k1 * t).exp();
        (a, s10 * k1 / (k2 - k1) * ((-k1 * t).exp() - (-k2 * t).exp()))
    }

    const TIMES: [f64; 8] = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0];

    /// Three conditions: the model as is, S1(0) estimated, k2 fixed to 0.3. Observables: S1 (lin), scale * S2
    /// (log10). Data from k1 = 0.4, k2 = 0.25, scale = 3, S1_0 (condition 2) = 20, with a small deterministic
    /// perturbation.
    fn decay_problem(perturb: f64) -> ProblemDef {
        let (k1, k2, sc, s2) = (0.4, 0.25, 3.0, 20.0);
        let mut meas = vec![];
        for (c, s10, kk2) in [("c1", 10.0, k2), ("c2", s2, k2), ("c3", 10.0, 0.3)] {
            for (q, &t) in TIMES.iter().enumerate() {
                let (a, b) = decay(t, s10, k1, kk2);
                let e = perturb * ((q as f64 * 1.7 + c.len() as f64).sin());
                meas.push(Measurement { condition: c.into(), observable: "obsA".into(), time: t, value: a + e, ..Default::default() });
                if t > 0.0 { meas.push(Measurement { condition: c.into(), observable: "obsB".into(), time: t, value: sc * b * (1.0 + 0.1 * e), ..Default::default() }); }
            }
        }
        ProblemDef {
            sbml: sbml(DECAY),
            params: vec![
                par("k1", Scale::Log10, 1e-3, 1e2, 0.5, true),
                par("k2", Scale::Log10, 1e-3, 1e2, 0.2, true),
                par("scale", Scale::Log, 0.1, 10.0, 2.0, true),
                par("S1_init", Scale::Lin, 1.0, 50.0, 15.0, true),
                par("sigma", Scale::Lin, 0.0, 1.0, 0.1, false),
            ],
            conditions: vec![
                ("c1".into(), vec![]),
                ("c2".into(), vec![("S1".into(), Value::Par("S1_init".into()))]),
                ("c3".into(), vec![("k2".into(), Value::Num(0.3))]),
            ],
            observables: vec![obs("obsA", "S1", "sigma", Noise::Normal, Scale::Lin), obs("obsB", "scale*S2", "0.05", Noise::Normal, Scale::Log10)],
            measurements: meas,
        }
    }

    fn fd_check(p: &Problem, theta: &[f64], tol: f64) {
        let e = p.evaluate(theta, true, p.least_squares).unwrap();
        let n = theta.len();
        for f in 0..n {
            let h = 1e-5 * theta[f].abs().max(1.0);
            let mut tp = theta.to_vec();
            tp[f] += h;
            let ep = p.evaluate(&tp, false, p.least_squares).unwrap();
            tp[f] -= 2.0 * h;
            let em = p.evaluate(&tp, false, p.least_squares).unwrap();
            let fd = (ep.nllh - em.nllh) / (2.0 * h);
            assert!((e.grad[f] - fd).abs() < tol * (1.0 + fd.abs()), "gradient {f}: {} vs {fd}", e.grad[f]);
            if p.least_squares {
                for i in 0..e.res.len() {
                    let fd = (ep.res[i] - em.res[i]) / (2.0 * h);
                    let a = e.jac[i * n + f];
                    assert!((a - fd).abs() < tol * (1.0 + fd.abs()), "jacobian ({i}, {f}): {a} vs {fd}");
                }
            }
        }
    }

    #[test]
    fn objective_matches_analytic_solution() {
        let mut p = Problem::new(decay_problem(0.02)).unwrap();
        p.rtol = 1e-11;
        p.atol = 1e-14;
        assert!(p.least_squares);
        let theta = [0.6f64.log10(), 0.15f64.log10(), 2.5f64.ln(), 12.0];
        let e = p.evaluate(&theta, false, true).unwrap();
        let mut nllh = 0.0;
        for (q, ms) in p.measurements.iter().enumerate() {
            let (s10, k2) = match ms.condition.as_str() { "c2" => (12.0, 0.15), "c3" => (10.0, 0.3), _ => (10.0, 0.15) };
            let (a, b) = decay(ms.time, s10, 0.6, k2);
            let (r, sig, lj) = if ms.observable == "obsA" { (a - ms.value, 0.1, 0.0) } else {
                ((2.5 * b).log10() - ms.value.log10(), 0.05, (ms.value * std::f64::consts::LN_10).ln())
            };
            nllh += 0.5 * (2.0 * std::f64::consts::PI * sig * sig).ln() + lj + 0.5 * (r / sig).powi(2);
            assert!((e.res[q] - r / sig).abs() < 1e-6, "residual {q}: {} vs {}", e.res[q], r / sig);
        }
        assert!((e.nllh - nllh).abs() < 1e-6 * nllh.abs(), "{} vs {nllh}", e.nllh);
    }

    #[test]
    fn gradients_match_finite_differences() {
        let mut p = Problem::new(decay_problem(0.02)).unwrap();
        p.rtol = 1e-11;
        p.atol = 1e-14;
        fd_check(&p, &[0.6f64.log10(), 0.15f64.log10(), 2.5f64.ln(), 12.0], 1e-5);
    }

    #[test]
    fn initial_assignment_and_volume() {
        // S1(0) = tot * frac by an initial assignment, compartment volume 2, a kinetic law in amount/time
        let ant = "model ia
            compartment C = 2
            species S1 in C, S2 in C
            J1: S1 -> S2; C*k1*S1/(Km + S1)
            J2: S2 -> ; k2*S2
            S1 = tot*frac; S2 = 0
            tot = 8; frac = 0.75; k1 = 1; Km = 2; k2 = 0.3
        end";
        let mut meas = vec![];
        for (q, &t) in TIMES.iter().enumerate() {
            meas.push(Measurement { condition: "a".into(), observable: "y".into(), time: t, value: 3.0 + 0.1 * q as f64, ..Default::default() });
            meas.push(Measurement { condition: "b".into(), observable: "y".into(), time: t, value: 2.0 + 0.2 * q as f64, ..Default::default() });
        }
        let def = ProblemDef {
            sbml: sbml(ant),
            params: vec![par("tot", Scale::Log10, 1.0, 100.0, 8.0, true), par("k1", Scale::Log, 0.01, 10.0, 1.0, true),
                         par("Km", Scale::Lin, 0.1, 10.0, 2.0, true), par("frac_b", Scale::Lin, 0.1, 1.0, 0.5, true)],
            conditions: vec![("a".into(), vec![]), ("b".into(), vec![("frac".into(), Value::Par("frac_b".into()))])],
            observables: vec![obs("y", "S1 + 2*S2", "0.2", Noise::Normal, Scale::Lin)],
            measurements: meas,
        };
        let mut p = Problem::new(def).unwrap();
        p.rtol = 1e-11;
        p.atol = 1e-14;
        let theta = [12f64.log10(), 0.8f64.ln(), 1.5, 0.4];
        fd_check(&p, &theta, 1e-5);
        // at t = 0 the observable is S1(0) = tot * frac: 12 * 0.75 in condition a, 12 * 0.4 in b
        let y = p.simulate_measurements(&theta).unwrap();
        assert!((y[0] - 9.0).abs() < 1e-12 && (y[1] - 4.8).abs() < 1e-12, "{y:?}");
    }

    #[test]
    fn laplace_and_estimated_noise() {
        let mut def = decay_problem(0.02);
        def.params[4].estimate = true; // sigma
        def.params[4].lb = 0.01;
        def.observables[1].noise = Noise::Laplace;
        let mut p = Problem::new(def).unwrap();
        p.rtol = 1e-11;
        p.atol = 1e-14;
        assert!(!p.least_squares);
        fd_check(&p, &[0.6f64.log10(), 0.15f64.log10(), 2.5f64.ln(), 12.0, 0.2], 1e-5);
        assert!(fit(&p, &FitOptions { method: Method::LeastSquares, ..Default::default() }, None).is_err());
    }

    #[test]
    fn multistart_recovers_parameters() {
        let p = Problem::new(decay_problem(0.0)).unwrap();
        let o = FitOptions { n_starts: 12, seed: 7, ..Default::default() };
        let r = fit(&p, &o, None).unwrap();
        let b = r.best();
        let truth = [0.4f64.log10(), 0.25f64.log10(), 3f64.ln(), 20.0];
        for k in 0..4 { assert!((b.x[k] - truth[k]).abs() < 1e-4 * truth[k].abs().max(1.0), "{k}: {:?} vs {truth:?}", b.x); }
        // most starts find the optimum
        let hits = r.starts.iter().filter(|s| s.nllh - b.nllh < 1e-3).count();
        assert!(hits >= 6, "{hits} of 12 starts reached the optimum");
        // same seed, one thread: the same result to the last bit
        let r1 = fit(&p, &FitOptions { threads: 1, ..o.clone() }, None).unwrap();
        assert_eq!(r1.best().x, b.x);
        assert_eq!(r1.best().nllh.to_bits(), b.nllh.to_bits());
        let fi = fisher(&p, &b.x, 0.95).unwrap();
        assert!(fi.identifiable.iter().all(|&x| x));
        for k in 0..4 { assert!(fi.sd[k] > 0.0 && fi.sd[k] < 0.1 * truth[k].abs().max(1.0), "{:?}", fi.sd); }
    }

    #[test]
    fn lbfgsb_fits_estimated_noise() {
        // the noise of obsA estimated (not least squares): L-BFGS-B recovers the parameters, and on the least-squares
        // version of the same problem it reaches the trust-region method's optimum
        let mut def = decay_problem(0.0);
        def.params[4] = par("sigma", Scale::Log10, 1e-3, 1.0, 0.1, true);
        let p = Problem::new(def).unwrap();
        assert!(!p.least_squares);
        let r = fit(&p, &FitOptions { n_starts: 12, seed: 5, ..Default::default() }, None).unwrap();
        let b = r.best();
        let truth = [0.4f64.log10(), 0.25f64.log10(), 3f64.ln(), 20.0];
        for k in 0..4 { assert!((b.x[k] - truth[k]).abs() < 1e-3 * truth[k].abs().max(1.0), "{k}: {:?} ({:?})", b.x, b.status); }
        let q = Problem::new(decay_problem(0.02)).unwrap();
        let o = FitOptions { n_starts: 8, seed: 2, ..Default::default() };
        let (a, l) = (fit(&q, &o, None).unwrap(), fit(&q, &FitOptions { method: Method::Lbfgsb, ..o }, None).unwrap());
        assert!((a.best().nllh - l.best().nllh).abs() < 1e-6 * a.best().nllh.abs(), "{} vs {}", a.best().nllh, l.best().nllh);
        for k in 0..4 { assert!((a.best().x[k] - l.best().x[k]).abs() < 1e-4 * a.best().x[k].abs().max(1.0)); }
    }

    #[test]
    fn unidentifiable_parameter_is_flagged() {
        // y = a * b * S1: only the product a * b is determined
        let mut def = decay_problem(0.0);
        def.params.push(par("a", Scale::Log10, 0.1, 10.0, 1.0, true));
        def.params.push(par("b", Scale::Log10, 0.1, 10.0, 1.0, true));
        def.observables[0].formula = "a*b*S1".into();
        let p = Problem::new(def).unwrap();
        let x = [0.4f64.log10(), 0.25f64.log10(), 3f64.ln(), 20.0, 0.2, -0.2];
        let fi = fisher(&p, &x, 0.95).unwrap();
        assert!(!fi.identifiable[4] && !fi.identifiable[5] && fi.identifiable[3], "{:?}", fi.identifiable);
    }

    #[test]
    fn compartment_sizes() {
        // inflow v0 in amount/time: d[S1]/dt = v0 / C - k [S1]; S1(t) = v0/(k C) + (S1_0 - v0/(k C)) exp(-k t)
        let ant = "model comp
            compartment C = 1
            species S1 in C
            J0: -> S1; v0
            J1: S1 -> ; k*S1*C
            S1 = 2; v0 = 3; k = 0.5
        end";
        let s1 = |t: f64, c: f64, v0: f64, k: f64| v0 / (k * c) + (2.0 - v0 / (k * c)) * (-k * t).exp();
        let def = |table_c: Option<f64>| {
            let mut params = vec![par("v0", Scale::Log10, 0.01, 100.0, 3.0, true), par("k", Scale::Log10, 0.01, 10.0, 0.5, true),
                                  par("C_big", Scale::Lin, 0.1, 10.0, 4.0, false)];
            if let Some(c) = table_c { params.push(par("C", Scale::Lin, 0.1, 10.0, c, false)); }
            let mut meas = vec![];
            for c in ["one", "two", "four"] {
                for &t in &TIMES[1..] { meas.push(Measurement { condition: c.into(), observable: "y".into(), time: t, value: 1.0, ..Default::default() }); }
            }
            ProblemDef {
                sbml: sbml(ant), params,
                conditions: vec![("one".into(), vec![]), ("two".into(), vec![("C".into(), Value::Num(2.0))]),
                                 ("four".into(), vec![("C".into(), Value::Par("C_big".into()))])],
                observables: vec![obs("y", "S1", "0.1", Noise::Normal, Scale::Lin)],
                measurements: meas,
            }
        };
        let mut p = Problem::new(def(None)).unwrap();
        p.rtol = 1e-11;
        p.atol = 1e-14;
        let (v0, k) = (2.5f64, 0.4f64);
        let theta = [v0.log10(), k.log10()];
        let y = p.simulate_measurements(&theta).unwrap();
        for (q, ms) in p.measurements.iter().enumerate() {
            let c = match ms.condition.as_str() { "one" => 1.0, "two" => 2.0, _ => 4.0 };
            assert!((y[q] - s1(ms.time, c, v0, k)).abs() < 1e-8, "{} t {}: {} vs {}", ms.condition, ms.time, y[q], s1(ms.time, c, v0, k));
        }
        fd_check(&p, &theta, 1e-5);
        // a fixed parameter-table row sets the size for every condition without its own
        let p = Problem::new(def(Some(3.0))).unwrap();
        let y = p.simulate_measurements(&theta).unwrap();
        assert!((y[0] - s1(TIMES[1], 3.0, v0, k)).abs() < 1e-6, "{} vs {}", y[0], s1(TIMES[1], 3.0, v0, k));
        // estimating a size is refused
        let mut d = def(Some(3.0));
        d.params.last_mut().unwrap().estimate = true;
        assert!(Problem::new(d).err().unwrap().contains("cannot be estimated"));
    }

    #[test]
    fn placeholders() {
        // obsA = observableParameter1 * S1 + observableParameter2 with noise noiseParameter1, filled per condition:
        // the scale is a free parameter in c1 and c2 and the number 1 in c3; the offset 0.5 or the free 'offset'.
        // The same problem written with the parameter names directly must give the same objective and gradient.
        let mut ph = decay_problem(0.02);
        ph.params.push(par("sA", Scale::Log10, 0.1, 10.0, 1.0, true));
        ph.params.push(par("offset", Scale::Lin, -1.0, 1.0, 0.0, true));
        ph.params.push(par("sdA", Scale::Lin, 0.01, 1.0, 0.1, false));
        ph.observables[0].formula = "observableParameter1_obsA * S1 + observableParameter2_obsA".into();
        ph.observables[0].noise_formula = "noiseParameter1_obsA".into();
        let mut direct = ph.clone();
        direct.observables = vec![
            obs("obsA_c12", "sA * S1 + 0.5", "sdA", Noise::Normal, Scale::Lin),
            obs("obsA_c3", "1 * S1 + offset", "0.2", Noise::Normal, Scale::Lin),
            ph.observables[1].clone(),
        ];
        for (mp, md) in ph.measurements.iter_mut().zip(direct.measurements.iter_mut()) {
            if mp.observable != "obsA" { continue; }
            if mp.condition == "c3" {
                mp.observable_parameters = vec![Value::Num(1.0), Value::Par("offset".into())];
                mp.noise_parameters = vec![Value::Num(0.2)];
                md.observable = "obsA_c3".into();
            } else {
                mp.observable_parameters = vec![Value::Par("sA".into()), Value::Num(0.5)];
                mp.noise_parameters = vec![Value::Par("sdA".into())];
                md.observable = "obsA_c12".into();
            }
        }
        let mut p = Problem::new(ph).unwrap();
        let mut q = Problem::new(direct).unwrap();
        (p.rtol, p.atol, q.rtol, q.atol) = (1e-11, 1e-14, 1e-11, 1e-14);
        assert!(p.least_squares); // sdA is fixed
        let theta = [0.6f64.log10(), 0.15f64.log10(), 2.5f64.ln(), 12.0, 0.2, 0.3];
        let (a, b) = (p.evaluate(&theta, true, true).unwrap(), q.evaluate(&theta, true, true).unwrap());
        assert!((a.nllh - b.nllh).abs() < 1e-12 * a.nllh.abs(), "{} vs {}", a.nllh, b.nllh);
        for f in 0..6 { assert!((a.grad[f] - b.grad[f]).abs() < 1e-9 * (1.0 + a.grad[f].abs()), "{f}: {} vs {}", a.grad[f], b.grad[f]); }
        fd_check(&p, &theta, 1e-5);
        // a noise placeholder filled by a free parameter: no longer least squares
        let mut d = decay_problem(0.02);
        d.params.push(par("sdA", Scale::Log10, 0.01, 1.0, 0.1, true));
        d.observables[0].noise_formula = "noiseParameter1_obsA".into();
        for m in d.measurements.iter_mut().filter(|m| m.observable == "obsA") { m.noise_parameters = vec![Value::Par("sdA".into())]; }
        let mut p = Problem::new(d).unwrap();
        (p.rtol, p.atol) = (1e-11, 1e-14);
        assert!(!p.least_squares);
        fd_check(&p, &[0.6f64.log10(), 0.15f64.log10(), 2.5f64.ln(), 12.0, -1.0], 1e-5);
    }

    #[test]
    fn initial_assignment_through_assignment_rule() {
        // S1(0) = tot, where tot := a1 + 2 * a2 is an assignment rule (as in Zhao_QuantBiol2020)
        let ant = "model ar
            compartment C = 1
            species S1 in C
            J1: S1 -> ; k*S1
            tot := a1 + 2*a2
            S1 = tot
            a1 = 3; a2 = 1; k = 0.2
        end";
        let def = ProblemDef {
            sbml: sbml(ant),
            params: vec![par("a1", Scale::Log10, 0.1, 100.0, 3.0, true), par("k", Scale::Log10, 0.01, 10.0, 0.2, true)],
            conditions: vec![("c".into(), vec![])],
            observables: vec![obs("y", "S1", "0.1", Noise::Normal, Scale::Lin)],
            measurements: TIMES.iter().map(|&t| Measurement { condition: "c".into(), observable: "y".into(), time: t, value: 1.0, ..Default::default() }).collect(),
        };
        let mut p = Problem::new(def).unwrap();
        (p.rtol, p.atol) = (1e-11, 1e-14);
        let theta = [7f64.log10(), 0.3f64.log10()];
        let y = p.simulate_measurements(&theta).unwrap();
        for (q, &t) in TIMES.iter().enumerate() { assert!((y[q] - 9.0 * (-0.3 * t).exp()).abs() < 1e-8, "t {t}: {}", y[q]); }
        fd_check(&p, &theta, 1e-5);
    }

    #[test]
    fn switch_times() {
        // the inflow switches from k1 to k2 at time tsw: with tsw estimated the sensitivities jump at the switch
        // (s+ = s- + (f- - f+) dtau / dp); without the jump the gradient for tsw would be 0
        let ant = "model sw
            compartment C = 1
            species S1 in C
            J0: -> S1; piecewise(k1, time < tsw, k2)
            J1: S1 -> ; kd*S1
            S1 = 0; k1 = 2; k2 = 0.5; kd = 0.3; tsw = 4.5
        end";
        let def = |law: &str| ProblemDef {
            sbml: sbml(&ant.replace("piecewise(k1, time < tsw, k2)", law)),
            params: vec![par("k1", Scale::Log10, 0.01, 100.0, 2.0, true), par("kd", Scale::Log10, 0.01, 10.0, 0.3, true),
                         par("tsw", Scale::Lin, 1.0, 9.0, 4.5, true)],
            conditions: vec![("c".into(), vec![])],
            observables: vec![obs("y", "S1", "0.1", Noise::Normal, Scale::Lin)],
            measurements: TIMES[1..].iter().map(|&t| Measurement { condition: "c".into(), observable: "y".into(), time: t, value: 1.0, ..Default::default() }).collect(),
        };
        // the engine's time breakpoint (time < tsw), and the form Beer_MolBioSystems2014 uses (located as a root)
        for law in ["piecewise(k1, time < tsw, k2)", "piecewise(k1, time - tsw < 0, k2)"] {
            let mut p = Problem::new(def(law)).unwrap();
            (p.rtol, p.atol) = (1e-11, 1e-14);
            let theta = [0.4f64.log10(), 0.2f64.log10(), 4.3];
            let e = p.evaluate(&theta, true, true).unwrap();
            assert!(e.grad[2].abs() > 1.0, "{law}: d nllh / d tsw = {}", e.grad[2]);
            fd_check(&p, &theta, 1e-5);
        }
    }

    #[test]
    fn state_switch() {
        // the inflow switches when S1 crosses an estimated threshold: the switch time moves with every parameter
        // through the state, dtau = -(g_x s + g_p) / (g_x f-)
        let ant = "model ss
            compartment C = 1
            species S1 in C, S2 in C
            J0: -> S1; piecewise(k1, S1 < thr, 0.2*k1)
            J1: S1 -> S2; kd*S1
            J2: S2 -> ; 0.5*S2
            S1 = 0; S2 = 0; k1 = 2; kd = 0.1; thr = 5
        end";
        let def = ProblemDef {
            sbml: sbml(ant),
            params: vec![par("k1", Scale::Log10, 0.01, 100.0, 2.0, true), par("kd", Scale::Log10, 0.001, 10.0, 0.1, true),
                         par("thr", Scale::Lin, 1.0, 9.0, 5.0, true)],
            conditions: vec![("c".into(), vec![])],
            observables: vec![obs("y1", "S1", "0.1", Noise::Normal, Scale::Lin), obs("y2", "S2", "0.1", Noise::Normal, Scale::Lin)],
            measurements: TIMES[1..].iter().flat_map(|&t| ["y1", "y2"].map(|o| Measurement { condition: "c".into(), observable: o.into(), time: t, value: 1.0, ..Default::default() })).collect(),
        };
        let mut p = Problem::new(def).unwrap();
        (p.rtol, p.atol) = (1e-11, 1e-14);
        fd_check(&p, &[2.5f64.log10(), 0.12f64.log10(), 4.0], 1e-5);
    }

    #[test]
    fn refuses_unsupported() {
        let mut def = decay_problem(0.0);
        def.measurements[0].time = f64::INFINITY;
        assert!(Problem::new(def).err().unwrap().contains("not supported yet"));
        let mut def = decay_problem(0.0);
        def.observables[0].formula = "observableParameter1_obsA * S1".into(); // a placeholder no measurement fills
        assert!(Problem::new(def).err().unwrap().contains("has 1 placeholder"));
        let mut def = decay_problem(0.0);
        def.observables[0].formula = "observableParameter2_obsA * S1".into(); // numbered from 1
        assert!(Problem::new(def).err().unwrap().contains("observableParameter1_obsA is missing"));
        let mut def = decay_problem(0.0);
        def.observables[0].formula = "nope * S1".into();
        assert!(Problem::new(def).err().unwrap().contains("unknown symbol 'nope'"));
        let mut def = decay_problem(0.0);
        def.conditions[0].1.push(("nope".into(), Value::Num(1.0)));
        assert!(Problem::new(def).err().unwrap().contains("not a parameter"));
    }
}
