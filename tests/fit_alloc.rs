//! Parameter estimation: no allocation in the objective's inner loop (spec: "Performance targets").
//!
//! A counting global allocator (this test binary only, one test, so nothing else allocates meanwhile) counts the
//! allocations of one objective + gradient + residual evaluation. The inner loop is the integration steps and
//! the per-measurement work, so the count must not grow with the number of measurements or with the number of
//! integration steps. Buffers come from a pool in the Problem, so after the first evaluation only the vectors
//! handed back (gradient, residuals, Jacobian) are allocated.
use roadrunner_nitro::fit::{Measurement, Noise, ObservableDef, ParamDef, Problem, ProblemDef, Scale, Value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { ALLOCS.fetch_add(1, Relaxed); System.alloc(l) }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { System.dealloc(p, l) }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 { ALLOCS.fetch_add(1, Relaxed); System.alloc_zeroed(l) }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 { ALLOCS.fetch_add(1, Relaxed); System.realloc(p, l, n) }
}

#[global_allocator]
static A: Counting = Counting;

const ANT: &str = "model mm
    compartment C = 1
    species S1 in C, S2 in C, S3 in C
    J0: -> S1; k0
    J1: S1 -> S2; Vm*S1/(Km + S1)
    J2: S2 -> S3; k2*S2
    J3: S3 -> ; k3*S3
    S1 = 10; S2 = 0; S3 = 0
    k0 = 0.3; Vm = 2; Km = 1; k2 = 0.5; k3 = 0.2
end";

/// Two conditions, two observables, n measurement times over [0, tend].
fn problem(n: usize, tend: f64) -> Problem {
    let sbml = antimony::sbml::write_sbml(&antimony::parse(ANT).document);
    let p = |id: &str, nominal: f64| ParamDef { id: id.into(), scale: Scale::Log10, lb: 1e-3, ub: 1e3, nominal, estimate: true };
    let mut meas = vec![];
    for c in ["a", "b"] {
        for q in 0..n {
            let t = tend * (q + 1) as f64 / n as f64;
            meas.push(Measurement { condition: c.into(), observable: "y2".into(), time: t, value: 1.0 });
            meas.push(Measurement { condition: c.into(), observable: "y3".into(), time: t, value: 0.5 });
        }
    }
    Problem::new(ProblemDef {
        sbml,
        params: vec![p("Vm", 2.0), p("Km", 1.0), p("k2", 0.5), p("k3", 0.2), p("S1_b", 5.0), p("scale", 1.0)],
        conditions: vec![("a".into(), vec![]), ("b".into(), vec![("S1".into(), Value::Par("S1_b".into()))])],
        observables: vec![
            ObservableDef { id: "y2".into(), formula: "S2".into(), noise_formula: "0.1".into(), noise: Noise::Normal, transform: Scale::Lin },
            ObservableDef { id: "y3".into(), formula: "scale*S3".into(), noise_formula: "0.1".into(), noise: Noise::Normal, transform: Scale::Log },
        ],
        measurements: meas,
    }).unwrap()
}

/// Allocations of one evaluation with gradient and residuals, after a warm-up evaluation.
fn allocs(p: &Problem) -> usize {
    let th = p.nominal();
    p.evaluate(&th, true, true).unwrap(); // compiles the sensitivity model, fills caches
    let before = ALLOCS.load(Relaxed);
    let e = p.evaluate(&th, true, true).unwrap();
    let n = ALLOCS.load(Relaxed) - before;
    assert!(e.nllh.is_finite());
    n
}

#[test]
fn allocations_do_not_grow_with_measurements_or_steps() {
    // run on one thread so the count is not affected by how rayon splits work
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        let base = allocs(&problem(5, 10.0));
        let more_meas = allocs(&problem(200, 10.0));
        let more_steps = allocs(&problem(5, 1000.0));
        println!("allocations per evaluation: 5 times {base}, 200 times {more_meas}, 5 times over 100x the span {more_steps}");
        assert_eq!(more_meas, base, "allocations grow with the number of measurements");
        assert_eq!(more_steps, base, "allocations grow with the number of integration steps");
        // what remains are the vectors handed back: gradient, residuals, Jacobian
        assert!(base <= 3, "{base} allocations per evaluation");
    });
}

