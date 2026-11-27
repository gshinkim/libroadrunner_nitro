//! Fitting problems (spec: "Problem definition", "Objective and gradients").
//!
//! A `ProblemDef` (SBML model, parameter table, conditions, observables, measurements: what PEtab import and the
//! native API both produce) is compiled once into a `Problem`:
//! - the engine model;
//! - a pre-tape giving the effective parameter values and initial state of a condition from its inputs
//!   (initial assignments re-evaluated, with exact derivatives);
//! - an observable tape giving every observable and its noise from (state, parameters, extra parameters, time),
//!   with exact derivatives;
//! - one augmented sensitivity model serving every (condition, free parameter) direction.
//!
//! `Problem::evaluate` gives the negative log-likelihood and its gradient, and for least-squares problems (normal
//! noise, noise fixed) the residuals and their Jacobian, all with respect to the free parameters on their scales.
use crate::expr::{parse, parse_sym, Dag, Op, Sym, Tape, N};
use crate::model::{Key, Model, SensSetup, SensState, Work};
use crate::sens::{remap, sens_setup};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f64::consts::{LN_10, PI};
use std::sync::{Arc, Mutex, OnceLock};

/// Parameter scale, also used for observable transformations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale { Lin, Log, Log10 }

impl Scale {
    pub fn parse(s: &str) -> Result<Scale, String> {
        match s.trim() {
            "" | "lin" => Ok(Scale::Lin),
            "log" => Ok(Scale::Log),
            "log10" => Ok(Scale::Log10),
            _ => Err(format!("unknown scale '{s}' (lin, log, log10)")),
        }
    }
    pub fn name(self) -> &'static str { match self { Scale::Lin => "lin", Scale::Log => "log", Scale::Log10 => "log10" } }
    /// Linear value of x given on this scale.
    pub fn unscale(self, x: f64) -> f64 { match self { Scale::Lin => x, Scale::Log => x.exp(), Scale::Log10 => 10f64.powf(x) } }
    /// Linear value v on this scale.
    pub fn scale(self, v: f64) -> f64 { match self { Scale::Lin => v, Scale::Log => v.ln(), Scale::Log10 => v.log10() } }
    /// d unscale(x) / dx.
    pub fn dunscale(self, x: f64) -> f64 {
        match self { Scale::Lin => 1.0, Scale::Log => x.exp(), Scale::Log10 => 10f64.powf(x) * LN_10 }
    }
    /// Observable transformation h(y) and h'(y).
    fn h(self, y: f64) -> (f64, f64) {
        match self { Scale::Lin => (y, 1.0), Scale::Log => (y.ln(), 1.0 / y), Scale::Log10 => (y.log10(), 1.0 / (y * LN_10)) }
    }
    /// log |d h / d y| at a measurement m: the change of variables term of the likelihood of m.
    fn log_jac(self, m: f64) -> f64 { match self { Scale::Lin => 0.0, Scale::Log => m.ln(), Scale::Log10 => (m * LN_10).ln() } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Noise { Normal, Laplace }

impl Noise {
    pub fn parse(s: &str) -> Result<Noise, String> {
        match s.trim() {
            "" | "normal" => Ok(Noise::Normal),
            "laplace" => Ok(Noise::Laplace),
            _ => Err(format!("unknown noise distribution '{s}' (normal, laplace)")),
        }
    }
}

/// A row of the parameter table. Bounds and nominal value are linear values (as PEtab); `scale` is the scale the
/// optimizer works on.
#[derive(Clone, Debug)]
pub struct ParamDef { pub id: String, pub scale: Scale, pub lb: f64, pub ub: f64, pub nominal: f64, pub estimate: bool }

/// A condition-table entry: a number, or a parameter-table id.
#[derive(Clone, Debug)]
pub enum Value { Num(f64), Par(String) }

#[derive(Clone, Debug)]
pub struct ObservableDef { pub id: String, pub formula: String, pub noise_formula: String, pub noise: Noise, pub transform: Scale }

#[derive(Clone, Debug, Default)]
pub struct Measurement {
    pub condition: String,
    pub observable: String,
    pub time: f64,
    pub value: f64,
    /// values of the observable's `observableParameter{n}_{observableId}` placeholders, n = 1, 2, ... (numbers or parameter ids)
    pub observable_parameters: Vec<Value>,
    /// values of its `noiseParameter{n}_{observableId}` placeholders
    pub noise_parameters: Vec<Value>,
}

/// A fitting problem before compilation.
#[derive(Clone, Debug, Default)]
pub struct ProblemDef {
    pub sbml: String,
    pub params: Vec<ParamDef>,
    /// (condition id, [(model parameter, species or compartment id, value)])
    pub conditions: Vec<(String, Vec<(String, Value)>)>,
    pub observables: Vec<ObservableDef>,
    pub measurements: Vec<Measurement>,
}

/// Where an input of a condition comes from: a constant, or a parameter-table row (its value times `f`).
#[derive(Clone, Copy, Debug)]
enum Slot { Const(f64), Par(usize, f64) }

struct Cond {
    id: String,
    /// pre-tape leaves: [model parameters (np) | initial states (nx)]
    leaf: Vec<Slot>,
    /// per state: its initial value is given (overrides an initial assignment)
    fixed_init: Vec<bool>,
    /// model parameters whose value is given here (a compartment size): kept even if an initial assignment sets them
    fixed_par: Vec<usize>,
    /// the compiled model: 0 the problem's, i > 0 compartment_models[i - 1] (compartment sizes set)
    model: usize,
    /// 0, then the distinct measurement times
    times: Vec<f64>,
    /// (time index, observable, measurement index)
    meas: Vec<(usize, usize, usize)>,
}


/// What `evaluate` returns. `grad`, `res`, `jac` are empty unless asked for; `jac` is row-major
/// (measurements x free parameters).
#[derive(Clone, Debug, Default)]
pub struct Eval { pub nllh: f64, pub grad: Vec<f64>, pub res: Vec<f64>, pub jac: Vec<f64> }


/// Scratch of one condition, reused across evaluations so the inner loop does not allocate.
struct CondScratch {
    w: Work,
    regs: Vec<f64>,   // pre-tape registers
    vals: Vec<f64>,   // [p_eff | x0]
    dpre: Vec<f64>,   // d [p_eff | x0] / d theta, (np + nx) x nf
    x: Vec<f64>,      // states at the output times, nt x nx
    oregs: Vec<f64>,  // observable-tape registers
    dleaf: Vec<(usize, usize, f64)>,
    dextra: Vec<(usize, usize, f64)>,
    /// forward sensitivities of every free parameter (made on the first gradient)
    st: Option<Box<SensState>>,
    /// per measurement of the condition: [y, sigma, dy/dtheta (nf), dsigma/dtheta (nf)]
    pred: Vec<f64>,
    err: Option<String>,
}

/// Everything one evaluation writes; evaluations take one from the problem's pool and give it back.
struct Scratch { pv: Vec<f64>, dpv: Vec<f64>, conds: Vec<CondScratch> }

pub struct Problem {
    pub model: Model,
    /// the model compiled again with the compartment sizes some conditions set (volumes are compiled into a model)
    compartment_models: Vec<Model>,
    pub params: Vec<ParamDef>,
    /// parameter-table rows estimated, in the order of the optimizer's vector
    pub free: Vec<usize>,
    pub observables: Vec<ObservableDef>,
    pub measurements: Vec<Measurement>,
    /// normal noise everywhere and no noise depending on a free parameter or the state: the objective is
    /// 0.5 * sum(res^2) + constant, so least-squares methods apply
    pub least_squares: bool,
    /// integrator tolerances of every simulation
    pub rtol: f64,
    pub atol: f64,
    /// parameter-table row of each extra parameter (rows that are not model symbols)
    extras: Vec<usize>,
    pre: Tape,
    pre_d: Vec<(u32, u32)>, // (row in [p_eff | x0], leaf) of each derivative output after the np + nx values
    obs: Tape,
    obs_d: Vec<Vec<(bool, u32, u32)>>, // per observable: (is sigma, leaf in [x | p_eff | extras | placeholders], tape output)
    /// first placeholder leaf of the observable tape (nx + np + ne) and its time register (after the placeholders)
    ph_base: usize,
    obs_time: usize,
    /// per measurement: (placeholder leaf - ph_base, the number or parameter filling it)
    ph: Vec<Vec<(u32, Slot)>>,
    conds: Vec<Cond>,
    base_keys: Vec<Key>,
    dir_ks: Vec<usize>,
    /// per compiled model (the problem's, then compartment_models)
    sens: Vec<OnceLock<Arc<SensSetup>>>,
    /// per parameter-table row: its index among the free parameters
    fpos: Vec<Option<usize>>,
    /// per measurement: (condition, place among the condition's measurements)
    loc: Vec<(usize, usize)>,
    state_outs: Vec<u32>,
    pool: Mutex<Vec<Scratch>>,
}

/// Rebuilds a tape in `dag`: leaves are given for registers [states | params], `time` for the time register.
/// Returns the node of every register.
fn lift_tape(t: &Tape, dag: &mut Dag, leaves: &[N], time: N) -> Vec<N> {
    let mut node = vec![0 as N; t.nregs];
    node[..leaves.len()].copy_from_slice(leaves);
    node[t.ns + t.np] = time;
    for &(r, x) in &t.consts { node[r as usize] = dag.cst(x); }
    for &(dst, op) in t.pre.iter().chain(&t.ins) { node[dst as usize] = dag.mk(remap(op, |r| node[r as usize])); }
    node
}

/// Value nodes and the nonzero derivatives of every root with respect to Var(0..nleaves): a tape whose outputs
/// are [roots | derivatives], and (root, leaf) of each derivative.
fn with_derivs(mut dag: Dag, roots: &[N], nleaves: usize) -> (Tape, Vec<(u32, u32)>) {
    let mut all = roots.to_vec();
    let mut at = vec![];
    for (q, &r) in roots.iter().enumerate() {
        let mut has_t = false;
        for l in dag.deps(r, &mut has_t) {
            let d = dag.diff(r, Op::Var(l));
            if dag.val(d) != Some(0.0) { at.push((q as u32, l)); all.push(d); }
        }
    }
    (Tape::new(&dag, &all, nleaves, 0), at)
}

/// `observableParameter{n}_{id}` -> (false, n), `noiseParameter{n}_{id}` -> (true, n), for observable `id`.
fn placeholder(name: &str, id: &str) -> Option<(bool, usize)> {
    let (noise, rest) = if let Some(r) = name.strip_prefix("observableParameter") { (false, r) }
                        else if let Some(r) = name.strip_prefix("noiseParameter") { (true, r) } else { return None };
    let (n, tail) = rest.split_once('_')?;
    if tail != id { return None; }
    n.parse::<usize>().ok().filter(|&n| n >= 1).map(|n| (noise, n))
}

impl Problem {
    pub fn new(def: ProblemDef) -> Result<Problem, String> {
        let (model, info) = crate::sbml::from_sbml(&def.sbml).map_err(|e| format!("model: {e}"))?;
        let m = &model;
        if m.has_events() { return Err("model: events are not supported yet (stage 4)".into()); }
        if m.has_delays() { return Err("model: delay() is not supported".into()); }
        if m.has_algebraic() { return Err("model: algebraic rules are not supported".into()); }
        let (nx, np) = (m.nx, m.pnames.len());

        // parameter table
        let mut pidx: HashMap<&str, usize> = HashMap::new();
        for (j, p) in def.params.iter().enumerate() {
            if pidx.insert(p.id.as_str(), j).is_some() { return Err(format!("parameter table: '{}' appears twice", p.id)); }
            if p.estimate {
                if !(p.lb <= p.ub) { return Err(format!("parameter table: '{}' has lowerBound > upperBound", p.id)); }
                if p.scale != Scale::Lin && !(p.lb > 0.0) { return Err(format!("parameter table: '{}' on {} scale needs lowerBound > 0", p.id, p.scale.name())); }
            }
            if !p.estimate && !p.nominal.is_finite() { return Err(format!("parameter table: fixed parameter '{}' has no nominal value", p.id)); }
        }
        let free: Vec<usize> = (0..def.params.len()).filter(|&j| def.params[j].estimate).collect();
        let is_model = |id: &str| m.outputs.iter().any(|o| o.0 == id);
        let extras: Vec<usize> = (0..def.params.len()).filter(|&j| !is_model(&def.params[j].id)).collect();
        let ne = extras.len();
        let key_of = |id: &str| m.resolve(&[id.to_string()]).ok().map(|k| k[0]);
        let state_comps: HashSet<&str> = m.state_ids.iter().filter_map(|s| info.comp_of.get(s).map(|c| c.as_str())).collect();
        // concentration per unit of a species' SBML value in model mc (1 / volume for amounts)
        let fac = |mc: &Model, i: usize| if info.hosu.contains(&m.state_ids[i]) { 1.0 / mc.vol[i] } else { 1.0 };

        // pre-tape: leaves [model parameters | initial states (concentrations)] -> [p_eff (np) | x0 (nx)]
        let ia: HashMap<&str, &str> = info.ia.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let mut dag = Dag::default();
        let mut memo: HashMap<String, N> = HashMap::new();
        /// Volume of species i: its compartment's parameter (so a compartment size set by a condition applies), or the
        /// compiled volume if the compartment is no parameter.
        fn vol_node(id: &str, i: usize, m: &Model, info: &crate::sbml::Info, dag: &mut Dag) -> N {
            match info.comp_of.get(id).and_then(|c| m.resolve(std::slice::from_ref(c)).ok()).map(|k| k[0]) {
                Some(Key::Par(k)) => dag.mk(Op::Var(k as u32)),
                _ => dag.cst(m.vol[i]),
            }
        }
        fn ia_node(id: &str, m: &Model, ia: &HashMap<&str, &str>, info: &crate::sbml::Info, dag: &mut Dag,
                   memo: &mut HashMap<String, N>, depth: usize) -> Result<N, String> {
            if let Some(&n) = memo.get(id) { return Ok(n); }
            if depth > 100 { return Err(format!("initial assignments: cycle through '{id}'")); }
            let np = m.pnames.len();
            let key = m.resolve(&[id.to_string()]).ok().map(|k| k[0]);
            let n = if let Some(f) = ia.get(id) {
                // SBML value of id (amounts for species with hasOnlySubstanceUnits)
                let mut err = None;
                let mut names = vec![];
                parse(f, &mut Dag::default(), |n| { names.push(n.to_string()); Some(Op::Const(0)) })?;
                let mut sub: HashMap<String, N> = HashMap::new();
                for nm in names {
                    if nm == "time" || nm == "__time__" { continue; }
                    match ia_node(&nm, m, ia, info, dag, memo, depth + 1) {
                        Ok(x) => {
                            // the formula sees SBML values: amounts for hasOnlySubstanceUnits species
                            let x = match m.resolve(&[nm.clone()]).ok().map(|k| k[0]) {
                                Some(Key::Spec(i)) if info.hosu.contains(&nm) => { let v = vol_node(&nm, i, m, info, dag); dag.mk(Op::Mul(x, v)) }
                                _ => x,
                            };
                            sub.insert(nm, x);
                        }
                        Err(e) => { err = Some(e); break; }
                    }
                }
                if let Some(e) = err { return Err(e); }
                let v = parse_sym(f, dag, |nm| if nm == "time" || nm == "__time__" { Some(Sym::Op(Op::Const(0f64.to_bits()))) } else { sub.get(nm).map(|&x| Sym::Node(x)) })
                    .map_err(|e| format!("initial assignment of '{id}': {e}"))?;
                match key {
                    Some(Key::Spec(i)) if info.hosu.contains(id) => { let c = vol_node(id, i, m, info, dag); dag.mk(Op::Div(v, c)) }
                    _ => v,
                }
            } else {
                match key {
                    Some(Key::Spec(i)) => dag.mk(Op::Var((np + i) as u32)),
                    Some(Key::Par(k)) => dag.mk(Op::Var(k as u32)),
                    // an assignment-rule variable: its rule at time 0, through the compiled rules
                    None => match m.outputs.iter().find(|o| o.0 == id) {
                        Some(&(_, r)) => aux_node(r, m, ia, info, dag, memo, depth + 1)?,
                        None => return Err(format!("initial assignments: unknown symbol '{id}'")),
                    },
                }
            };
            memo.insert(id.to_string(), n);
            Ok(n)
        }
        /// Register r of the aux tape at time 0, built backwards from r: states and parameters by their initial
        /// values (initial assignments included), only the registers r needs.
        fn aux_node(r: u32, m: &Model, ia: &HashMap<&str, &str>, info: &crate::sbml::Info, dag: &mut Dag,
                    memo: &mut HashMap<String, N>, depth: usize) -> Result<N, String> {
            let key = format!("#aux{r}");
            if let Some(&n) = memo.get(&key) { return Ok(n); }
            if depth > 400 { return Err("initial assignments: assignment rules nested too deeply".into()); }
            let (nx, np, t) = (m.nx, m.pnames.len(), m.aux_tape());
            let ru = r as usize;
            let n = if ru < nx {
                ia_node(&m.state_ids[ru], m, ia, info, dag, memo, depth + 1)?
            } else if ru < nx + np {
                ia_node(&m.pnames[ru - nx], m, ia, info, dag, memo, depth + 1)?
            } else if ru == nx + np {
                dag.cst(0.0) // time
            } else if let Some(&(_, x)) = t.consts.iter().find(|c| c.0 == r) {
                dag.cst(x)
            } else {
                let op = t.pre.iter().chain(&t.ins).find(|e| e.0 == r).map(|e| e.1)
                    .ok_or_else(|| format!("initial assignments: rule register {r} not found"))?;
                let mut kids: HashMap<N, N> = HashMap::new();
                for c in crate::expr::children(&op).iter().flatten() {
                    if !kids.contains_key(c) { let x = aux_node(*c, m, ia, info, dag, memo, depth + 1)?; kids.insert(*c, x); }
                }
                dag.mk(remap(op, |c| kids[&c]))
            };
            memo.insert(key, n);
            Ok(n)
        }
        let mut roots = Vec::with_capacity(np + nx);
        for k in 0..np {
            let id = &m.pnames[k];
            let is_state = matches!(key_of(id), Some(Key::Spec(_)));
            roots.push(if ia.contains_key(id.as_str()) && !is_state { ia_node(id, m, &ia, &info, &mut dag, &mut memo, 0)? } else { dag.mk(Op::Var(k as u32)) });
        }
        for i in 0..nx {
            let id = &m.state_ids[i];
            roots.push(if ia.contains_key(id.as_str()) { ia_node(id, m, &ia, &info, &mut dag, &mut memo, 0)? } else { dag.mk(Op::Var((np + i) as u32)) });
        }
        let (pre, pre_d) = with_derivs(dag, &roots, np + nx);

        // observable tape: leaves [x (nx) | p_eff (np) | extras (ne)], time
        let mut dag = Dag::default();
        let leaves: Vec<N> = (0..nx + np).map(|r| dag.mk(Op::Var(r as u32))).collect();
        let time = dag.mk(Op::Time);
        let mut aux: Option<Vec<N>> = None;
        let mut roots = vec![];
        // placeholders observableParameter{n}_{id} / noiseParameter{n}_{id}: one leaf each, after the extras;
        // per observable, its highest n of each kind and the leaf of each (kind, n)
        let ph_base = nx + np + ne;
        let mut ph_leaf: HashMap<(usize, bool, usize), u32> = HashMap::new();
        let mut ph_count: Vec<[usize; 2]> = vec![[0, 0]; def.observables.len()];
        let mut ph_nodes: Vec<(u32, N)> = vec![];
        for (oi, o) in def.observables.iter().enumerate() {
            for (what, src) in [("observableFormula", &o.formula), ("noiseFormula", &o.noise_formula)] {
                let mut names = vec![];
                parse(src, &mut Dag::default(), |n| { names.push(n.to_string()); Some(Op::Const(0)) })
                    .map_err(|e| format!("observable '{}': {what}: {e}", o.id))?;
                let mut sub: HashMap<String, N> = HashMap::new();
                for nm in names {
                    if sub.contains_key(&nm) { continue; }
                    let node = |id: &str, dag: &mut Dag, aux: &mut Option<Vec<N>>| -> Option<N> {
                        if let Some(&(_, r)) = m.outputs.iter().find(|o| o.0 == id) {
                            let r = r as usize;
                            Some(if r < nx + np { leaves[r] } else if r == nx + np { time } else {
                                aux.get_or_insert_with(|| lift_tape(m.aux_tape(), dag, &leaves, time))[r]
                            })
                        } else { None }
                    };
                    let n = if let Some(n) = node(&nm, &mut dag, &mut aux) {
                        // species: the SBML value (amount for hasOnlySubstanceUnits)
                        if let Some(c) = info.comp_of.get(&nm) {
                            let (is_amt, want_amt) = (info.amount_params.contains(&nm), info.hosu.contains(&nm));
                            if is_amt != want_amt {
                                let v = match node(c, &mut dag, &mut aux) { Some(v) => v, None => dag.cst(info.values.get(c).copied().unwrap_or(1.0)) };
                                dag.mk(if want_amt { Op::Mul(n, v) } else { Op::Div(n, v) })
                            } else { n }
                        } else { n }
                    } else if let Some(e) = extras.iter().position(|&j| def.params[j].id == nm) {
                        dag.mk(Op::Var((nx + np + e) as u32))
                    } else if nm == "time" || nm == "t" {
                        time
                    } else if let Some((noise, n)) = placeholder(&nm, &o.id) {
                        let next = ph_leaf.len() as u32;
                        let l = *ph_leaf.entry((oi, noise, n)).or_insert(next);
                        let c = &mut ph_count[oi][noise as usize];
                        *c = (*c).max(n);
                        match ph_nodes.iter().find(|e| e.0 == l) {
                            Some(e) => e.1,
                            None => { let x = dag.mk(Op::Var((ph_base + l as usize) as u32)); ph_nodes.push((l, x)); x }
                        }
                    } else {
                        return Err(format!("observable '{}': {what}: unknown symbol '{nm}'", o.id));
                    };
                    sub.insert(nm, n);
                }
                let r = parse_sym(src, &mut dag, |nm| sub.get(nm).map(|&x| Sym::Node(x)))
                    .map_err(|e| format!("observable '{}': {what}: {e}", o.id))?;
                roots.push(r);
            }
        }
        let nph = ph_leaf.len();
        for (oi, cnt) in ph_count.iter().enumerate() {
            for (noise, &m) in [false, true].iter().zip(cnt) {
                if let Some(n) = (1..=m).find(|&n| !ph_leaf.contains_key(&(oi, *noise, n))) {
                    return Err(format!("observable '{}': {}Parameter{n}_{} is missing (placeholders are numbered from 1)",
                                       def.observables[oi].id, if *noise { "noise" } else { "observable" }, def.observables[oi].id));
                }
            }
        }
        let (obs, at) = with_derivs(dag, &roots, ph_base + nph);
        let no = def.observables.len();
        let mut obs_d = vec![vec![]; no];
        for (q, &(row, leaf)) in at.iter().enumerate() {
            obs_d[row as usize / 2].push((row % 2 == 1, leaf, obs.outs[2 * no + q]));
        }

        // conditions
        let oidx: HashMap<&str, usize> = def.observables.iter().enumerate().map(|(i, o)| (o.id.as_str(), i)).collect();
        if oidx.len() != no { return Err("observable table: duplicate observableId".into()); }
        let slot_of = |v: &Value, f: f64, what: &str| -> Result<Option<Slot>, String> {
            Ok(match v {
                Value::Num(x) if x.is_nan() => None,
                Value::Num(x) => Some(Slot::Const(x * f)),
                Value::Par(id) => Some(Slot::Par(*pidx.get(id.as_str()).ok_or(format!("{what}: '{id}' is not in the parameter table"))?, f)),
            })
        };
        // compartment sizes: known values only (a size is compiled into the model); a fixed parameter-table row
        // sets it for every condition, a condition for itself
        let comp_value = |v: &Value, what: &str| -> Result<Option<f64>, String> {
            match v {
                Value::Num(x) if x.is_nan() => Ok(None),
                Value::Num(x) => Ok(Some(*x)),
                Value::Par(id) => {
                    let j = *pidx.get(id.as_str()).ok_or(format!("{what}: '{id}' is not in the parameter table"))?;
                    if def.params[j].estimate { return Err(format!("{what}: compartment sizes holding species cannot be estimated")); }
                    Ok(Some(def.params[j].nominal))
                }
            }
        };
        let mut base_comp: Vec<(String, f64)> = vec![];
        for p in &def.params {
            if state_comps.contains(p.id.as_str()) {
                if p.estimate { return Err(format!("parameter table: compartment '{}' holds species; compartment sizes cannot be estimated", p.id)); }
                base_comp.push((p.id.clone(), p.nominal));
            }
        }
        let mut compartment_models: Vec<Model> = vec![];
        let mut model_keys: Vec<Vec<(String, u64)>> = vec![vec![]];
        let mut conds = vec![];
        let mut cidx: HashMap<&str, usize> = HashMap::new();
        for (cid, entries) in &def.conditions {
            if cidx.insert(cid.as_str(), conds.len()).is_some() { return Err(format!("condition table: '{cid}' appears twice")); }
            // this condition's compartment sizes, and its compiled model
            let mut comp = base_comp.clone();
            for (target, v) in entries {
                if !state_comps.contains(target.as_str()) { continue; }
                let what = format!("condition '{cid}', column '{target}'");
                if let Some(x) = comp_value(v, &what)? {
                    if !(x > 0.0) { return Err(format!("{what}: a compartment size must be > 0")); }
                    comp.retain(|c| &c.0 != target);
                    comp.push((target.clone(), x));
                }
            }
            comp.sort_by(|a, b| a.0.cmp(&b.0));
            let key: Vec<(String, u64)> = comp.iter().map(|(c, x)| (c.clone(), x.to_bits())).collect();
            let mi = match model_keys.iter().position(|k| *k == key) {
                Some(i) => i,
                None => {
                    let mut sbml = def.sbml.clone();
                    for (c, x) in &comp { sbml = crate::sbml::edit::set_compartment_size(&sbml, c, *x).map_err(|e| format!("condition '{cid}': {e}"))?; }
                    let (mc, _) = crate::sbml::from_sbml(&sbml).map_err(|e| format!("condition '{cid}': model with its compartment sizes: {e}"))?;
                    if mc.pnames != m.pnames || mc.state_ids != m.state_ids {
                        return Err(format!("condition '{cid}': setting compartment sizes changes the model's structure"));
                    }
                    compartment_models.push(mc);
                    model_keys.push(key);
                    model_keys.len() - 1
                }
            };
            let mc = if mi == 0 { m } else { &compartment_models[mi - 1] };
            // defaults from that model, then the parameter table, then the condition
            let mut leaf: Vec<Slot> = (0..np).map(|k| Slot::Const(mc.pvals[k])).chain((0..nx).map(|i| Slot::Const(mc.init[i]))).collect();
            for (j, p) in def.params.iter().enumerate() {
                match key_of(&p.id) {
                    Some(Key::Par(k)) if state_comps.contains(p.id.as_str()) => leaf[k] = Slot::Const(mc.pvals[k]),
                    Some(Key::Par(k)) => leaf[k] = Slot::Par(j, 1.0),
                    Some(Key::Spec(i)) => leaf[np + i] = Slot::Par(j, fac(mc, i)),
                    None => {}
                }
            }
            let mut fixed_init = vec![false; nx];
            let mut fixed_par: Vec<usize> = comp.iter().filter_map(|(c, _)| match key_of(c) { Some(Key::Par(k)) => Some(k), _ => None }).collect();
            fixed_par.sort();
            for (target, v) in entries {
                let what = format!("condition '{cid}', column '{target}'");
                match key_of(target) {
                    Some(Key::Par(_)) if state_comps.contains(target.as_str()) => {} // in the compiled model
                    Some(Key::Par(k)) => if let Some(s) = slot_of(v, 1.0, &what)? { leaf[k] = s; },
                    Some(Key::Spec(i)) => if let Some(s) = slot_of(v, fac(mc, i), &what)? { leaf[np + i] = s; fixed_init[i] = true; },
                    None => return Err(format!("{what}: not a parameter, species or compartment of the model")),
                }
            }
            conds.push(Cond { id: cid.clone(), leaf, fixed_init, fixed_par, model: mi, times: vec![0.0], meas: vec![] });
        }
        let mut ph: Vec<Vec<(u32, Slot)>> = Vec::with_capacity(def.measurements.len());
        for (q, ms) in def.measurements.iter().enumerate() {
            let what = format!("measurement {}", q + 1);
            let &c = cidx.get(ms.condition.as_str()).ok_or(format!("{what}: unknown condition '{}'", ms.condition))?;
            let &o = oidx.get(ms.observable.as_str()).ok_or(format!("{what}: unknown observable '{}'", ms.observable))?;
            if ms.time.is_infinite() { return Err(format!("{what}: steady-state measurements (time inf) are not supported yet")); }
            if !(ms.time >= 0.0) { return Err(format!("{what}: time must be >= 0")); }
            if !ms.value.is_finite() { return Err(format!("{what}: measurement must be a finite number")); }
            if def.observables[o].transform != Scale::Lin && !(ms.value > 0.0) {
                return Err(format!("{what}: a {}-transformed observable needs measurements > 0", def.observables[o].transform.name()));
            }
            let mut fill = vec![];
            for (noise, vals) in [(false, &ms.observable_parameters), (true, &ms.noise_parameters)] {
                let need = ph_count[o][noise as usize];
                let col = if noise { "noiseParameters" } else { "observableParameters" };
                if vals.len() != need {
                    return Err(format!("{what}: {col} has {} value(s), observable '{}' has {need} placeholder(s)", vals.len(), ms.observable));
                }
                for (n, v) in vals.iter().enumerate() {
                    let w = format!("{what}, {col} {}", n + 1);
                    let sl = slot_of(v, 1.0, &w)?.ok_or(format!("{w}: a placeholder needs a value"))?;
                    fill.push((ph_leaf[&(o, noise, n + 1)], sl));
                }
            }
            ph.push(fill);
            conds[c].meas.push((0, o, q));
        }
        for c in &mut conds {
            let mut ts: Vec<f64> = c.meas.iter().map(|&(_, _, q)| def.measurements[q].time).collect();
            ts.push(0.0);
            ts.sort_by(f64::total_cmp);
            ts.dedup();
            for e in &mut c.meas { e.0 = ts.partition_point(|&t| t < def.measurements[e.2].time); }
            c.times = ts;
        }
        conds.retain(|c| !c.meas.is_empty());

        // which p_eff depend on a free parameter somewhere: those get a coefficient in the sensitivity model
        let is_free_leaf = |s: &Slot| matches!(s, Slot::Par(j, _) if def.params[*j].estimate);
        let mut dep = vec![false; np];
        for c in &conds {
            for &(row, l) in &pre_d {
                if (row as usize) < np && is_free_leaf(&c.leaf[l as usize]) { dep[row as usize] = true; }
            }
        }
        let dir_ks: Vec<usize> = (0..np).filter(|&k| dep[k]).collect();
        // A switch time in a rate law that depends on an estimated parameter: moving the switch moves the trajectory,
        // which forward sensitivities only capture with a jump at the switch (not implemented yet). Refused rather
        // than give a silently wrong gradient.
        // every comparison inside the rate laws (rules they use included): one involving time and a parameter that
        // depends on an estimated one is a switch whose time moves with that parameter (`time > t1`, `time - tau < 0`)
        {
            let mut d2 = Dag::default();
            let leaves: Vec<N> = (0..nx + np).map(|r| d2.mk(Op::Var(r as u32))).collect();
            let t = d2.mk(Op::Time);
            let rt = m.rates_tape();
            let node = lift_tape(rt, &mut d2, &leaves, t);
            let mut stack: Vec<N> = rt.outs.iter().map(|&r| node[r as usize]).collect();
            let mut seen = HashSet::new();
            while let Some(nd) = stack.pop() {
                if !seen.insert(nd) { continue; }
                let op = d2.nodes[nd as usize];
                if matches!(op, Op::Lt(..) | Op::Le(..) | Op::Eq(..) | Op::Ne(..)) {
                    let mut has_t = false;
                    let vars = d2.deps(nd, &mut has_t);
                    if has_t {
                        if let Some(k) = vars.iter().map(|&v| v as usize).find(|&k| k >= nx && dep[k - nx]) {
                            return Err(format!("parameter '{}' sets the time of a switch in the model (a piecewise or comparison with time) \
                                                and depends on an estimated parameter: exact gradients across such switches are not supported yet",
                                               m.pnames[k - nx]));
                        }
                    }
                }
                stack.extend(crate::expr::children(&op).iter().flatten());
            }
        }
        // least squares: normal noise, noise depending on no state and no free parameter
        let mut least_squares = def.observables.iter().all(|o| o.noise == Noise::Normal);
        let is_free = |s: &Slot| matches!(s, Slot::Par(j, _) if def.params[*j].estimate);
        for d in &obs_d {
            for &(sig, l, _) in d {
                if !sig { continue; }
                let l = l as usize;
                let free_dep = if l < nx { true } else if l < nx + np { dep[l - nx] } else if l < ph_base { def.params[extras[l - nx - np]].estimate } else {
                    let lp = (l - ph_base) as u32;
                    ph.iter().flatten().any(|&(pl, sl)| pl == lp && is_free(&sl))
                };
                if free_dep { least_squares = false; }
            }
        }
        let base_keys = (0..np).map(Key::Par).chain((0..nx).map(Key::Spec)).collect();
        let mut fpos = vec![None; def.params.len()];
        for (f, &j) in free.iter().enumerate() { fpos[j] = Some(f); }
        let mut loc = vec![(0, 0); def.measurements.len()];
        for (ci, c) in conds.iter().enumerate() {
            for (k, &(_, _, q)) in c.meas.iter().enumerate() { loc[q] = (ci, k); }
        }
        Ok(Problem {
            params: def.params, free, observables: def.observables, measurements: def.measurements, least_squares,
            rtol: 1e-8, atol: 1e-12, extras, pre, pre_d, obs, obs_d, ph_base, obs_time: ph_base + nph, ph, conds, base_keys, dir_ks,
            sens: (0..=compartment_models.len()).map(|_| OnceLock::new()).collect(), compartment_models,
            fpos, loc, state_outs: (0..nx as u32).collect(), pool: Mutex::new(vec![]), model,
        })
    }

    pub fn n_free(&self) -> usize { self.free.len() }
    pub fn free_ids(&self) -> Vec<String> { self.free.iter().map(|&j| self.params[j].id.clone()).collect() }
    /// Bounds of the free parameters on their scales.
    pub fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        self.free.iter().map(|&j| { let p = &self.params[j]; (p.scale.scale(p.lb), p.scale.scale(p.ub)) }).unzip()
    }
    /// Nominal values of the free parameters on their scales.
    pub fn nominal(&self) -> Vec<f64> { self.free.iter().map(|&j| { let p = &self.params[j]; p.scale.scale(p.nominal) }).collect() }

    fn model_of(&self, mi: usize) -> &Model { if mi == 0 { &self.model } else { &self.compartment_models[mi - 1] } }

    /// The parameter derivatives sensitivities need, for compiled model mi (built on first use).
    fn sens_setup(&self, mi: usize) -> Arc<SensSetup> {
        self.sens[mi].get_or_init(|| Arc::new(sens_setup(self.model_of(mi), &self.dir_ks))).clone()
    }

    fn new_scratch(&self) -> Scratch {
        let m = &self.model;
        let (nx, np, nf) = (m.nx, m.pnames.len(), self.free.len());
        let nl = np + nx;
        let conds = self.conds.iter().map(|c| {
            let nt = c.times.len();
            CondScratch {
                w: self.model_of(c.model).work(), regs: self.pre.regs(), vals: vec![0.0; nl], dpre: vec![0.0; nl * nf], x: vec![0.0; nt * nx],
                oregs: self.obs.regs(), dleaf: Vec::with_capacity(nl), dextra: Vec::with_capacity(self.extras.len()),
                st: None,
                pred: vec![0.0; c.meas.len() * (2 + 2 * nf)], err: None,
            }
        }).collect();
        Scratch { pv: vec![0.0; self.params.len()], dpv: vec![0.0; self.params.len()], conds }
    }

    /// Run `f` with a scratch from the pool (a new one if none is free) and give it back afterwards. Parallel
    /// evaluations (multistart) each hold their own, so they never wait for each other.
    fn with_scratch<T>(&self, f: impl FnOnce(&mut Scratch) -> T) -> T {
        let free = self.pool.lock().unwrap().pop();
        let mut s = free.unwrap_or_else(|| self.new_scratch());
        let r = f(&mut s);
        self.pool.lock().unwrap().push(s);
        r
    }

    /// One condition: fills cs.pred for its measurements, with the derivatives if `grad`.
    fn condition(&self, c: &Cond, cs: &mut CondScratch, pv: &[f64], dpv: &[f64], grad: bool) -> Result<(), String> {
        let m = self.model_of(c.model);
        let (nx, np, nf) = (m.nx, m.pnames.len(), self.free.len());
        let nl = np + nx;
        let CondScratch { w, regs, vals, dpre, x, oregs, dleaf, dextra, st, pred, .. } = cs;
        // pre-tape leaves and their derivatives
        dleaf.clear();
        for (l, s) in c.leaf.iter().enumerate() {
            regs[l] = match *s {
                Slot::Const(v) => v,
                Slot::Par(j, f) => {
                    if let Some(q) = self.fpos[j] { dleaf.push((l, q, f * dpv[j])); }
                    f * pv[j]
                }
            };
        }
        self.pre.run_all(regs);
        for (v, &r) in vals.iter_mut().zip(&self.pre.outs[..nl]) { *v = regs[r as usize]; }
        for i in 0..nx { if c.fixed_init[i] { vals[np + i] = regs[np + i]; } }
        for &k in &c.fixed_par { vals[k] = regs[k]; }
        if grad {
            dpre.fill(0.0);
            for (q, &(row, l)) in self.pre_d.iter().enumerate() {
                let row = row as usize;
                if (row >= np && c.fixed_init[row - np]) || (row < np && c.fixed_par.binary_search(&row).is_ok()) { continue; }
                let d = regs[self.pre.outs[nl + q] as usize];
                for &(ll, f, dl) in dleaf.iter() { if ll == l as usize { dpre[row * nf + f] += d * dl; } }
            }
            for &(l, f, dl) in dleaf.iter() { if l >= np && c.fixed_init[l - np] { dpre[l * nf + f] += dl; } }
        }
        // the model, with the sensitivities of every free parameter integrated alongside it if `grad`
        m.load(w, &self.base_keys, vals);
        let ok = if grad {
            let sv = st.get_or_insert_with(|| Box::new(m.sens_state(self.sens_setup(c.model), nf)));
            let nk = self.dir_ks.len();
            for f in 0..nf {
                for i in 0..nx { sv.s[f * nx + i] = dpre[(np + i) * nf + f]; }
                for (q, &k) in self.dir_ks.iter().enumerate() { sv.c[f * nk + q] = dpre[k * nf + f]; }
            }
            w.sens = st.take();
            let ok = m.simulate_sens(w, &c.times, self.rtol, self.atol, &self.state_outs, x);
            *st = w.sens.take();
            ok
        } else {
            m.simulate(w, &c.times, self.rtol, self.atol, &self.state_outs, x)
        };
        if !ok { return Err(format!("simulation failed in condition '{}'", c.id)); }
        // observables, noise and their derivatives
        oregs[nx..nx + np].copy_from_slice(&vals[..np]);
        dextra.clear();
        for (e, &j) in self.extras.iter().enumerate() {
            oregs[nx + np + e] = pv[j];
            if let Some(f) = self.fpos[j] { dextra.push((e, f, dpv[j])); }
        }
        self.obs.prep(oregs);
        let stride = 2 + 2 * nf;
        for (k, &(ti, o, q)) in c.meas.iter().enumerate() {
            oregs[..nx].copy_from_slice(&x[ti * nx..(ti + 1) * nx]);
            oregs[self.obs_time] = c.times[ti];
            for &(l, sl) in &self.ph[q] {
                oregs[self.ph_base + l as usize] = match sl { Slot::Const(v) => v, Slot::Par(j, f) => f * pv[j] };
            }
            self.obs.run(oregs);
            let p = &mut pred[k * stride..(k + 1) * stride];
            p[0] = oregs[self.obs.outs[2 * o] as usize];
            p[1] = oregs[self.obs.outs[2 * o + 1] as usize];
            if !grad { continue; }
            let (dy, ds) = p[2..].split_at_mut(nf);
            dy.fill(0.0);
            ds.fill(0.0);
            for &(is_sig, l, r) in &self.obs_d[o] {
                let d = oregs[r as usize];
                let acc = if is_sig { &mut ds[..] } else { &mut dy[..] };
                let l = l as usize;
                if l < nx {
                    if let Some(sv) = st.as_deref() { for f in 0..nf { acc[f] += d * sv.out[(ti * nf + f) * nx + l]; } }
                } else if l < nx + np {
                    let kk = l - nx;
                    for f in 0..nf { acc[f] += d * dpre[kk * nf + f]; }
                } else if l < self.ph_base {
                    for &(e, f, de) in dextra.iter() { if e == l - nx - np { acc[f] += d * de; } }
                } else if let Some(&(_, Slot::Par(j, fac))) = self.ph[q].iter().find(|e| e.0 as usize == l - self.ph_base) {
                    if let Some(f) = self.fpos[j] { acc[f] += d * fac * dpv[j]; }
                }
            }
        }
        Ok(())
    }

    /// Every condition at theta (in parallel), into s.
    fn run(&self, theta: &[f64], grad: bool, s: &mut Scratch) -> Result<(), String> {
        if theta.len() != self.free.len() { return Err(format!("expected {} parameter values, got {}", self.free.len(), theta.len())); }
        let Scratch { pv, dpv, conds } = s;
        for (j, p) in self.params.iter().enumerate() { pv[j] = p.nominal; dpv[j] = 0.0; }
        for (f, &j) in self.free.iter().enumerate() {
            let sc = self.params[j].scale;
            pv[j] = sc.unscale(theta[f]);
            dpv[j] = sc.dunscale(theta[f]);
        }
        let (pv, dpv) = (&*pv, &*dpv);
        conds.par_iter_mut().zip(self.conds.par_iter()).for_each(|(cs, c)| { cs.err = self.condition(c, cs, pv, dpv, grad).err(); });
        for cs in conds.iter_mut() { if let Some(e) = cs.err.take() { return Err(e); } }
        Ok(())
    }

    /// Negative log-likelihood at theta (free parameters on their scales), with its gradient if `grad`, and the
    /// residuals and their Jacobian if `res` (least-squares problems only). A failed simulation or an
    /// undefined value (log of a negative observable) is an error; optimizers treat it as a rejected point.
    /// Sums run over the measurements in table order, so the result does not depend on the number of threads.
    pub fn evaluate(&self, theta: &[f64], grad: bool, res: bool) -> Result<Eval, String> {
        if res && !self.least_squares { return Err("residuals need normal noise that depends on no free parameter".into()); }
        let nf = self.free.len();
        let stride = 2 + 2 * nf;
        self.with_scratch(|s| {
            self.run(theta, grad || res, s)?;
            let nm = self.measurements.len();
            let mut e = Eval { nllh: 0.0, ..Default::default() };
            if grad { e.grad = vec![0.0; nf]; }
            if res { e.res = Vec::with_capacity(nm); e.jac = Vec::with_capacity(nm * nf); }
            for (q, ms) in self.measurements.iter().enumerate() {
                let (ci, k) = self.loc[q];
                let o = &self.observables[self.conds[ci].meas[k].1];
                let p = &s.conds[ci].pred[k * stride..(k + 1) * stride];
                let (y, sig) = (p[0], p[1]);
                let (dy, ds) = (&p[2..2 + nf], &p[2 + nf..]);
                let (hy, dh) = o.transform.h(y);
                let (hm, _) = o.transform.h(ms.value);
                let r = hy - hm;
                if !r.is_finite() || !(sig > 0.0) || !sig.is_finite() {
                    return Err(format!("measurement {}: undefined likelihood (observable {y}, sigma {sig})", q + 1));
                }
                let lj = o.transform.log_jac(ms.value);
                match o.noise {
                    Noise::Normal => {
                        e.nllh += 0.5 * (2.0 * PI * sig * sig).ln() + lj + 0.5 * (r / sig).powi(2);
                        if grad {
                            for f in 0..nf {
                                e.grad[f] += ds[f] / sig + r / (sig * sig) * dh * dy[f] - r * r / (sig * sig * sig) * ds[f];
                            }
                        }
                    }
                    Noise::Laplace => {
                        e.nllh += (2.0 * sig).ln() + lj + r.abs() / sig;
                        if grad {
                            for f in 0..nf {
                                e.grad[f] += ds[f] / sig + r.signum() / sig * dh * dy[f] - r.abs() / (sig * sig) * ds[f];
                            }
                        }
                    }
                }
                if res {
                    e.res.push(r / sig);
                    e.jac.extend((0..nf).map(|f| dh * dy[f] / sig));
                }
            }
            Ok(e)
        })
    }

    /// Simulated observables (y) of every measurement at theta, in measurement order.
    pub fn simulate_measurements(&self, theta: &[f64]) -> Result<Vec<f64>, String> {
        let stride = 2 + 2 * self.free.len();
        self.with_scratch(|s| {
            self.run(theta, false, s)?;
            Ok(self.loc.iter().map(|&(ci, k)| s.conds[ci].pred[k * stride]).collect())
        })
    }
}
