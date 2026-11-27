//! Benchmark: the same tape interpreted and compiled (Cranelift), on real models. Opt-in:
//!
//!     cargo test --release tape_bench -- --ignored --nocapture
//!
//! For each model, the solvers' main tape (all reaction rates and every nonzero Jacobian entry) is evaluated
//! by the interpreter and by the compiled code from identical registers, and the compile time is measured.
//! Models: tests/models/sosbench (committed) and, when downloaded, a spread of sizes from tests/models/biomodels.

use crate::jit::compile;
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

/// Evaluations per second of `f`, timed over at least `min` after a warm-up.
fn rate(mut f: impl FnMut(), min: Duration) -> f64 {
    for _ in 0..1000 { f(); }
    let (mut n, t0) = (0u64, Instant::now());
    loop {
        for _ in 0..1000 { f(); }
        n += 1000;
        let dt = t0.elapsed();
        if dt >= min { return n as f64 / dt.as_secs_f64(); }
    }
}

fn models() -> Vec<(String, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("models");
    let mut out = vec![];
    if let Ok(dirs) = std::fs::read_dir(root.join("sosbench")) {
        let mut dirs: Vec<_> = dirs.flatten().map(|d| d.path()).collect();
        dirs.sort();
        for d in dirs {
            if let Some(f) = std::fs::read_dir(&d).ok().and_then(|it| it.flatten().map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "xml")).max()) {
                out.push((format!("sosbench/{}", d.file_name().unwrap().to_string_lossy()), std::fs::read_to_string(f).unwrap()));
            }
        }
    }
    // a spread of sizes from the curated BioModels, when they have been downloaded
    if let Ok(files) = std::fs::read_dir(root.join("biomodels")) {
        let mut files: Vec<_> = files.flatten().map(|e| (e.metadata().map(|m| m.len()).unwrap_or(0), e.path())).collect();
        files.sort();
        let n = files.len();
        for k in 0..12 {
            if n == 0 { break; }
            let (_, p) = &files[(n - 1) * k / 11];
            out.push((p.file_stem().unwrap().to_string_lossy().into_owned(), std::fs::read_to_string(p).unwrap()));
        }
    }
    out
}

#[test]
#[ignore]
fn tape_bench() {
    let min = Duration::from_millis(300);
    println!("\n{:<28} {:>7} {:>11} {:>11} {:>8} {:>11} {:>11}", "model", "instr", "interp ns", "native ns",
             "speedup", "compile ms", "break-even");
    let mut speedups = vec![];
    for (name, sbml) in models() {
        let Ok((m, _)) = crate::sbml::from_sbml(&sbml) else { continue };
        let t = m.full_tape();
        if t.ins.is_empty() { continue; }
        // registers: the tape's constants plus a state and parameters of order one, then the parameter-only part
        let mut regs = t.regs();
        for r in regs.iter_mut().take(t.ns + t.np) { *r = 0.75; }
        t.prep(&mut regs);
        let mut a = regs.clone();
        let interp = rate(|| t.interpret(black_box(&mut a)), min);
        let c0 = Instant::now();
        let Some(j) = compile(&t.ins, &t.outs, t.nregs) else { println!("{name:<28} (did not compile)"); continue };
        let compile_s = c0.elapsed().as_secs_f64();
        let mut b = regs.clone();
        let native = rate(|| unsafe { (j.f)(black_box(b.as_mut_ptr())) }, min);
        // the same answers from both (as the engine requires)
        let (mut x, mut y) = (regs.clone(), regs.clone());
        t.interpret(&mut x);
        unsafe { (j.f)(y.as_mut_ptr()) };
        let same = t.outs.iter().all(|&o| x[o as usize].to_bits() == y[o as usize].to_bits()
            || (x[o as usize].is_nan() && y[o as usize].is_nan()));
        let (ti, tn) = (1e9 / interp, 1e9 / native);
        // evaluations after which compiling has paid for itself
        let even = if ti > tn { format!("{:.0}", compile_s * 1e9 / (ti - tn)) } else { "never".into() };
        speedups.push(ti / tn);
        println!("{:<28} {:>7} {:>11.0} {:>11.0} {:>7.1}x {:>11.2} {:>11}{}", name, t.ins.len(), ti, tn, ti / tn,
                 compile_s * 1e3, even, if same { "" } else { "  (results differ in the last bit)" });
    }
    speedups.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !speedups.is_empty() {
        println!("\ncompiled vs interpreted: median {:.1}x, range {:.1}x - {:.1}x over {} tapes", speedups[speedups.len() / 2],
                 speedups[0], speedups[speedups.len() - 1], speedups.len());
    }
}
