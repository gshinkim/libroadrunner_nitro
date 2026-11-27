//! Parameter estimation (spec: "Spec: parameter estimation in roadrunner_nitro", stage 1).
//!
//! - `problem`: problem definition (the native API; PEtab import produces the same `ProblemDef`), compiled once;
//!   negative log-likelihood, gradient, residuals and residual Jacobian with exact forward sensitivities.
//! - `trf`: trust-region reflective least squares with bounds.
//! - `optimize`: Latin hypercube multistart and Fisher-information intervals.
pub mod optimize;
pub mod petab;
pub mod problem;
pub mod trf;

pub use optimize::{fisher, fit, FitOptions, FitResult, Fisher, StartResult};
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
                meas.push(Measurement { condition: c.into(), observable: "obsA".into(), time: t, value: a + e });
                if t > 0.0 { meas.push(Measurement { condition: c.into(), observable: "obsB".into(), time: t, value: sc * b * (1.0 + 0.1 * e) }); }
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
            meas.push(Measurement { condition: "a".into(), observable: "y".into(), time: t, value: 3.0 + 0.1 * q as f64 });
            meas.push(Measurement { condition: "b".into(), observable: "y".into(), time: t, value: 2.0 + 0.2 * q as f64 });
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
        assert!(fit(&p, &FitOptions::default(), None).is_err());
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
    fn refuses_unsupported() {
        let mut def = decay_problem(0.0);
        def.measurements[0].time = f64::INFINITY;
        assert!(Problem::new(def).err().unwrap().contains("stage 2"));
        let mut def = decay_problem(0.0);
        def.observables[0].formula = "observableParameter1_obsA * S1".into();
        assert!(Problem::new(def).err().unwrap().contains("unknown symbol"));
        let mut def = decay_problem(0.0);
        def.conditions[0].1.push(("nope".into(), Value::Num(1.0)));
        assert!(Problem::new(def).err().unwrap().contains("not a parameter"));
    }
}
