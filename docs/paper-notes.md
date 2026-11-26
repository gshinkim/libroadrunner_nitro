# Notes toward a paper: roadrunner_nitro

Working notes on the design and measured behaviour of roadrunner_nitro, kept for a possible paper. Each section
states what was measured, how, and on what machine, so the numbers can be reproduced or re-measured.

Machine for all measurements below unless stated: Windows 10, x86-64, Rust 1.99, release build with
`target-cpu=native`, Python 3.12. Reference engine: libroadrunner 2.11.0 (2.9.1 for the stiff benchmark).

## 1. How model equations are evaluated: an expression tape, interpreted then compiled

### Design

Every model's mathematics is turned into a single **tape**: a straight-line list of instructions, each of the
form "register d = operation(register a, register b)", for example `r17 = r4 * r9` or `r18 = exp(r17)`
(`src/expr.rs`, `Tape`). The registers hold the species values, the parameters, the time, constants and
intermediate results, in a flat array of 64-bit floats.

The tape is built in three steps:

1. Rate laws are parsed into a hash-consed expression graph (a DAG). Identical subexpressions anywhere in the
   model are stored once, and constant subexpressions are folded at build time.
2. Derivatives are taken symbolically on that graph, giving exact expressions for every nonzero entry of the
   Jacobian (and, for sensitivities, for derivatives with respect to parameters). No finite differences are
   used anywhere in the engine.
3. The graph is flattened into one instruction list whose outputs are all the reaction rates and all the
   nonzero Jacobian entries. One pass down the list evaluates them all, sharing every common subexpression.
   Instructions that depend only on parameters and constants are split off (`pre`) and evaluated once per
   parameter change rather than at every step.

There are no loops or branches in a tape (piecewise functions become select operations), which makes it easy
both to interpret and to compile.

### Execution: interpreted first, compiled once it pays

`Tape::run` interprets the tape for its first 2000 calls (`JIT_AFTER`), then compiles it once with Cranelift
into a native function and calls that from then on (`src/jit.rs`). If compilation fails, the tape stays
interpreted.

- **Interpreter** (`Tape::exec`): a loop over the instruction list applying each operation to the register
  array. It is cheap as interpreters go: every value is a plain f64 (no boxing, no type checks), registers are
  indexed directly (no name lookup, bounds checks removed), nothing is allocated per instruction, and the
  instruction list was already simplified at build time. What remains is per-instruction dispatch, and results
  going through memory rather than CPU registers. This is the overhead that makes general-purpose interpreters
  such as CPython slow (dynamic types, boxed numbers, name lookups) largely absent here.
- **Compiler** (`jit::compile`): emits one straight-line native function. Temporaries stay in SSA values (CPU
  registers where possible); only output registers are stored back. Cranelift's optimiser is switched off
  (`opt_level = "none"`) because the tape is already optimised at the expression level; Cranelift is chosen for
  compile speed rather than code quality. Elementary functions are called through Rust's libm, the same as the
  interpreter, so the two paths are meant to give bit-identical results.

The rationale for interpreting first: compilation has a fixed cost per model, which short runs (one-off
simulations, parameter scans, steady states, the many small augmented models built for sensitivities) never
recover. Interpreting until a tape has proved it will be used heavily means short jobs pay nothing for
compilation and long jobs lose only a bounded amount (a ski-rental argument).

### Contrast with libroadrunner

libroadrunner compiles every model with LLVM, with optimisation passes, at load time, after parsing it with
libSBML. Its code is well optimised, but the cost is paid before anything runs. Measured on the stiff
benchmark (section 2): load times of 46-1850 ms for libroadrunner against 0.8-17 ms for roadrunner_nitro, the
1850 ms being a 201-species chain, almost all of it compilation.

### Measurement: interpreted against compiled, same tape

Method: for each model, the solvers' main tape (all rates plus all nonzero Jacobian entries) is evaluated
repeatedly by the interpreter and by the compiled function from identical registers (state and parameters set
to 0.75, then the parameter-only part), each timed over at least 0.3 s after a warm-up, with
`std::hint::black_box` preventing the optimiser from removing the work. Compile time is measured for one
compilation. The break-even point is compile time / (interpreted time - compiled time), the number of
evaluations after which compiling has paid for itself. Models: the 7 committed `tests/models/sosbench` models and
11 curated BioModels chosen across the range of file sizes.

Reproduce: `cargo test --release --lib tape_bench -- --ignored --nocapture` (`src/tape_bench.rs`).

| Model | Instructions | Interpreted (ns/eval) | Compiled (ns/eval) | Speedup | Compile (ms) | Break-even (evals) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sosbench/00009 | 90 | 147 | 19 | 7.6x | 1.77 | 13830 |
| sosbench/00014 | 776 | 923 | 227 | 4.1x | 5.23 | 7519 |
| sosbench/00022 | 208 | 441 | 111 | 4.0x | 1.32 | 3984 |
| sosbench/00033 | 230 | 539 | 71 | 7.6x | 1.52 | 3243 |
| sosbench/10000 | 39 | 68 | 10 | 6.5x | 0.33 | 5852 |
| sosbench/10001 | 64 | 132 | 25 | 5.2x | 0.60 | 5644 |
| sosbench/repressilator | 39 | 60 | 11 | 5.5x | 0.30 | 6191 |
| BIOMD0000000486 | 2 | 4 | 3 | 1.5x | 0.25 | 178490 |
| BIOMD0000000052 | 15 | 19 | 8 | 2.5x | 0.34 | 30031 |
| BIOMD0000001024 | 15 | 17 | 4 | 4.5x | 0.26 | 19440 |
| BIOMD0000000771 | 29 | 129 | 71 | 1.8x | 0.37 | 6418 |
| BIOMD0000000733 | 59 | 100 | 22 | 4.6x | 0.48 | 6052 |
| BIOMD0000000243 | 53 | 64 | 19 | 3.4x | 0.52 | 11577 |
| BIOMD0000000007 | 201 | 390 | 43 | 9.1x | 1.10 | 3165 |
| BIOMD0000000075 | 116 | 210 | 55 | 3.8x | 0.95 | 6099 |
| BIOMD0000000144 | 351 | 698 | 83 | 8.4x | 1.95 | 3173 |
| BIOMD0000000625 | 119 | 220 | 30 | 7.4x | 0.82 | 4304 |
| BIOMD0000000997 | 140 | 413 | 107 | 3.9x | 0.81 | 2662 |

Findings:

- **Compiled code is a median 4.6x faster than the interpreter (range 1.5x-9.1x)**, consistent with the
  project's design estimate of 3-10x. The interpreter costs about 1-4.5 ns per instruction. Gains are smallest on
  tiny tapes (little work to speed up) and largest on tapes of about 100-350 instructions.
- **Compilation takes 0.25-5.2 ms per model**: a fixed cost of about 0.25 ms plus 4-17 microseconds per
  instruction in these models.
- **Break-even is typically 3000-14000 evaluations**, and far more for very small tapes. The engine compiles at
  2000, so on this machine it compiles somewhat earlier than break-even; the cost of that is at most a few
  milliseconds per model. The code's own comment ("compiling costs about as much as 2000 interpreted runs
  save") does not hold on this machine; it may have been measured elsewhere. Worth re-measuring on several machines
  before choosing the threshold for a paper.
- **The interpreted and compiled results were bit-identical on all 18 tapes.** (One Rust unit test,
  `jit_matches_interpreter`, does find a last-bit difference in `y^3` at y = 1.3 on this machine, so bit-identity
  is not guaranteed for every operation; see section 4.)

Open question for a paper: how much of a whole simulation's time is tape evaluation? For stiff models much of
the time goes into LU factorisations and Newton iterations, which dilutes any difference between interpreted
and compiled tapes. Measuring a full simulation with compilation disabled against enabled would answer it.

## 2. Speed and accuracy against libroadrunner on stiff models

Default settings on both engines (rtol 1e-6, atol 1e-12), same SBML, median of 20 runs after a warm-up;
libroadrunner 2.9.1. Reproduce: `python benchmarks/bench_stiff.py`, `python benchmarks/check_accuracy.py`.

| Model | Load, libroadrunner | Load, nitro | Simulate, libroadrunner | Simulate, nitro | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Robertson (3 species, rates 4e-2 to 3e7) | 46-124 ms | 0.8-9 ms | 4.3 ms | 0.25 ms | 17x |
| Oregonator (stiff oscillator) | 40 ms | 0.4 ms | 9.2 ms | 0.88 ms | 10x |
| 201-species stiff chain | 1850 ms | 17 ms | 22.4 ms | 3.5 ms | 6.4x |

Against a tight-tolerance reference (rtol 1e-10), roadrunner_nitro's default-tolerance error was equal to or
smaller than libroadrunner's on every model (Oregonator 2.5e-3 vs 4.0e-3; chain 3.4e-6 vs 6.3e-6).

## 3. Correctness against independent references

- **SBML Test Suite, semantic time courses** (commit bbcbc80), judged against the suite's expected results with
  each case's own tolerances: roadrunner_nitro passes 1754 of 1754 scored cases, libroadrunner 2.11 passes 1576.
  libroadrunner refuses algebraic rules (123 cases) and delay differential equations (50) and gives wrong results
  in 5. Not scored: flux-balance cases (34, not supported) and fast reactions (35, removed from SBML in L3V2).
  Reproduce: `python benchmarks/sbml_test_suite.py`.
- **libroadrunner 2.11's own Python test suite**, run unchanged against roadrunner_nitro: 425 pass, 0 fail, with 7
  known differences tracked (`docs/milestones-1-3-report.md`).
- **Exact derivatives versus finite differences.** In several cases the reference values in libroadrunner's
  tests are libroadrunner's own finite-difference results, and roadrunner_nitro's exact values differ from them
  by more than the tests' tolerances (for example an elasticity of exactly 0.01 recorded as 0.009999). On the
  curated BioModels, roadrunner_nitro's Jacobian of BIOMD0000000017 agrees with an independent Richardson-
  extrapolated finite-difference reference to 1.4e-6, while libroadrunner's is off by up to 11% in one entry.
- **Curated BioModels comparison**: in progress (`benchmarks/biomodels_compare.py`).

## 4. Caveats to state in a paper

- Measurements so far are on one Windows machine. The original author measured on Apple Silicon; numbers should
  be reported per platform.
- Builds use `target-cpu=native`, so timings reflect the build machine's instruction set.
- Bit-identity between interpreted and compiled tapes holds on all measured model tapes but not for every
  operation on this machine (`jit_matches_interpreter`: `y^3` differs in the last bit at y = 1.3). A simulation
  that crosses the 2000-evaluation threshold can therefore change its last-bit rounding partway through.
- Exact forward sensitivities miss the effect of parameters on the timing of discontinuities (for example
  `ceil(S * p)`); finite differences capture it.
