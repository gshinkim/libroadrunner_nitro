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
use crate::model::{Key, Model};
use crate::sens::{remap, sens_model_dirs};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f64::consts::{LN_10, PI};
use std::sync::OnceLock;

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

#[derive(Clone, Debug)]
pub struct Measurement { pub condition: String, pub observable: String, pub time: f64, pub value: f64 }

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
    /// 0, then the distinct measurement times
    times: Vec<f64>,
    /// (time index, observable, measurement index)
    meas: Vec<(usize, usize, usize)>,
}

struct SensModel { m: Model, keys: Vec<Key>, souts: Vec<u32> }

/// What `evaluate` returns. `grad`, `res`, `jac` are empty unless asked for; `jac` is row-major
/// (measurements x free parameters).
#[derive(Clone, Debug, Default)]
pub struct Eval { pub nllh: f64, pub grad: Vec<f64>, pub res: Vec<f64>, pub jac: Vec<f64> }

/// Per measurement: (y, sigma, dy/dtheta, dsigma/dtheta).
type Pred = (f64, f64, Vec<f64>, Vec<f64>);

pub struct Problem {
    pub model: Model,
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
    obs_d: Vec<Vec<(bool, u32, u32)>>, // per observable: (is sigma, leaf in [x | p_eff | extras], tape output)
    conds: Vec<Cond>,
    base_keys: Vec<Key>,
    dir_ks: Vec<usize>,
    sens: OnceLock<Result<SensModel, String>>,
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
        // concentration per unit of a species' SBML value (1 / volume for amounts)
        let fac = |i: usize| if info.hosu.contains(&m.state_ids[i]) { 1.0 / m.vol[i] } else { 1.0 };

        // pre-tape: leaves [model parameters | initial states (concentrations)] -> [p_eff (np) | x0 (nx)]
        let ia: HashMap<&str, &str> = info.ia.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let mut dag = Dag::default();
        let mut memo: HashMap<String, N> = HashMap::new();
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
                                Some(Key::Spec(i)) if info.hosu.contains(&nm) => { let v = dag.cst(m.vol[i]); dag.mk(Op::Mul(x, v)) }
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
                    Some(Key::Spec(i)) if info.hosu.contains(id) => { let c = dag.cst(m.vol[i]); dag.mk(Op::Div(v, c)) }
                    _ => v,
                }
            } else {
                match key {
                    Some(Key::Spec(i)) => dag.mk(Op::Var((np + i) as u32)),
                    Some(Key::Par(k)) => dag.mk(Op::Var(k as u32)),
                    None => return Err(format!("initial assignments: '{id}' (an assignment rule variable?) is not supported")),
                }
            };
            memo.insert(id.to_string(), n);
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
        for o in &def.observables {
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
                    } else {
                        return Err(format!("observable '{}': {what}: unknown symbol '{nm}' (placeholders come in stage 2)", o.id));
                    };
                    sub.insert(nm, n);
                }
                let r = parse_sym(src, &mut dag, |nm| sub.get(nm).map(|&x| Sym::Node(x)))
                    .map_err(|e| format!("observable '{}': {what}: {e}", o.id))?;
                roots.push(r);
            }
        }
        let (obs, at) = with_derivs(dag, &roots, nx + np + ne);
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
        let mut base: Vec<Slot> = (0..np).map(|k| Slot::Const(m.pvals[k])).chain((0..nx).map(|i| Slot::Const(m.init[i]))).collect();
        for (j, p) in def.params.iter().enumerate() {
            match key_of(&p.id) {
                Some(Key::Par(k)) => {
                    if state_comps.contains(p.id.as_str()) { return Err(format!("parameter table: compartment '{}' holds species; compartment sizes cannot be set yet", p.id)); }
                    base[k] = Slot::Par(j, 1.0);
                }
                Some(Key::Spec(i)) => base[np + i] = Slot::Par(j, fac(i)),
                None => {}
            }
        }
        let mut conds = vec![];
        let mut cidx: HashMap<&str, usize> = HashMap::new();
        for (cid, entries) in &def.conditions {
            if cidx.insert(cid.as_str(), conds.len()).is_some() { return Err(format!("condition table: '{cid}' appears twice")); }
            let mut leaf = base.clone();
            let mut fixed_init = vec![false; nx];
            for (target, v) in entries {
                let what = format!("condition '{cid}', column '{target}'");
                match key_of(target) {
                    Some(Key::Par(k)) => {
                        if state_comps.contains(target.as_str()) { return Err(format!("{what}: compartment sizes cannot be set yet")); }
                        if let Some(s) = slot_of(v, 1.0, &what)? { leaf[k] = s; }
                    }
                    Some(Key::Spec(i)) => if let Some(s) = slot_of(v, fac(i), &what)? { leaf[np + i] = s; fixed_init[i] = true; },
                    None => return Err(format!("{what}: not a parameter, species or compartment of the model")),
                }
            }
            conds.push(Cond { id: cid.clone(), leaf, fixed_init, times: vec![0.0], meas: vec![] });
        }
        for (q, ms) in def.measurements.iter().enumerate() {
            let what = format!("measurement {}", q + 1);
            let &c = cidx.get(ms.condition.as_str()).ok_or(format!("{what}: unknown condition '{}'", ms.condition))?;
            let &o = oidx.get(ms.observable.as_str()).ok_or(format!("{what}: unknown observable '{}'", ms.observable))?;
            if ms.time.is_infinite() { return Err(format!("{what}: steady-state measurements (time inf) come in stage 2")); }
            if !(ms.time >= 0.0) { return Err(format!("{what}: time must be >= 0")); }
            if !ms.value.is_finite() { return Err(format!("{what}: measurement must be a finite number")); }
            if def.observables[o].transform != Scale::Lin && !(ms.value > 0.0) {
                return Err(format!("{what}: a {}-transformed observable needs measurements > 0", def.observables[o].transform.name()));
            }
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
        let is_free = |s: &Slot| matches!(s, Slot::Par(j, _) if def.params[*j].estimate);
        let mut dep = vec![false; np];
        for c in &conds {
            for &(row, l) in &pre_d {
                if (row as usize) < np && is_free(&c.leaf[l as usize]) { dep[row as usize] = true; }
            }
        }
        let dir_ks: Vec<usize> = (0..np).filter(|&k| dep[k]).collect();
        // least squares: normal noise, noise depending on no state and no free parameter
        let mut least_squares = def.observables.iter().all(|o| o.noise == Noise::Normal);
        for d in &obs_d {
            for &(sig, l, _) in d {
                if !sig { continue; }
                let l = l as usize;
                let free_dep = if l < nx { true } else if l < nx + np { dep[l - nx] } else { def.params[extras[l - nx - np]].estimate };
                if free_dep { least_squares = false; }
            }
        }
        let base_keys = (0..np).map(Key::Par).chain((0..nx).map(Key::Spec)).collect();
        Ok(Problem {
            params: def.params, free, observables: def.observables, measurements: def.measurements, least_squares,
            rtol: 1e-8, atol: 1e-12, extras, pre, pre_d, obs, obs_d, conds, base_keys, dir_ks, sens: OnceLock::new(), model,
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

    fn sens_model(&self) -> Result<&SensModel, String> {
        self.sens.get_or_init(|| {
            let m = &self.model;
            let a = sens_model_dirs(m, &self.dir_ks)?;
            let nx = m.nx;
            let names: Vec<String> = m.pnames.iter().cloned().chain(m.state_ids.iter().cloned())
                .chain((0..nx).map(|i| format!("__s{i}"))).chain((0..self.dir_ks.len()).map(|q| format!("__c{q}"))).collect();
            let keys = a.resolve(&names)?;
            let souts = keys[m.pnames.len() + nx..m.pnames.len() + 2 * nx].iter()
                .map(|k| match k { Key::Spec(i) => Ok(*i as u32), _ => Err("sensitivity model: unexpected layout".to_string()) })
                .collect::<Result<_, _>>()?;
            Ok(SensModel { m: a, keys, souts })
        }).as_ref().map_err(|e| e.clone())
    }

    /// Predictions of one condition: per measurement of it (in order), (y, sigma, dy/dtheta, dsigma/dtheta).
    fn condition(&self, c: &Cond, pv: &[f64], dpv: &[f64], grad: bool) -> Result<Vec<Pred>, String> {
        let m = &self.model;
        let (nx, np, nf, ne) = (m.nx, m.pnames.len(), self.free.len(), self.extras.len());
        let fpos: HashMap<usize, usize> = self.free.iter().enumerate().map(|(f, &j)| (j, f)).collect();
        // leaves and their derivatives
        let nl = np + nx;
        let mut regs = self.pre.regs();
        let mut dleaf: Vec<(usize, usize, f64)> = vec![]; // (leaf, free index, d leaf / d theta_f)
        for (l, s) in c.leaf.iter().enumerate() {
            regs[l] = match *s {
                Slot::Const(x) => x,
                Slot::Par(j, f) => {
                    if let Some(&q) = fpos.get(&j) { dleaf.push((l, q, f * dpv[j])); }
                    f * pv[j]
                }
            };
        }
        self.pre.run_all(&mut regs);
        let mut vals: Vec<f64> = self.pre.outs[..nl].iter().map(|&r| regs[r as usize]).collect();
        for i in 0..nx { if c.fixed_init[i] { vals[np + i] = regs[np + i]; } }
        // d [p_eff | x0] / d theta (nl x nf)
        let mut dpre = vec![0.0; if grad { nl * nf } else { 0 }];
        if grad {
            let fixed = |row: usize| row >= np && c.fixed_init[row - np];
            for (q, &(row, l)) in self.pre_d.iter().enumerate() {
                let row = row as usize;
                if fixed(row) { continue; }
                let d = regs[self.pre.outs[nl + q] as usize];
                for &(ll, f, dl) in &dleaf { if ll == l as usize { dpre[row * nf + f] += d * dl; } }
            }
            for &(l, f, dl) in &dleaf { if l >= np && c.fixed_init[l - np] { dpre[l * nf + f] += dl; } }
        }
        let fail = || format!("simulation failed in condition '{}'", c.id);
        let nt = c.times.len();
        let mut x = vec![0.0; nt * nx];
        let mut w = m.work();
        m.load(&mut w, &self.base_keys, &vals);
        if !m.simulate(&mut w, &c.times, self.rtol, self.atol, &(0..nx as u32).collect::<Vec<_>>(), &mut x) { return Err(fail()); }
        // sensitivities d x / d theta_f: nt x nx per free parameter (None: zero)
        let sens: Vec<Option<Vec<f64>>> = if grad {
            let sm = self.sens_model()?;
            (0..nf).into_par_iter().map(|f| {
                let s0: Vec<f64> = (0..nx).map(|i| dpre[(np + i) * nf + f]).collect();
                let cf: Vec<f64> = self.dir_ks.iter().map(|&k| dpre[k * nf + f]).collect();
                if s0.iter().chain(&cf).all(|&v| v == 0.0) { return Ok(None); }
                let v: Vec<f64> = vals.iter().copied().chain(s0).chain(cf).collect();
                let mut w = sm.m.work();
                sm.m.load(&mut w, &sm.keys, &v);
                let mut out = vec![0.0; nt * nx];
                if !sm.m.simulate(&mut w, &c.times, self.rtol, self.atol, &sm.souts, &mut out) { return Err(fail()); }
                Ok(Some(out))
            }).collect::<Result<_, String>>()?
        } else { vec![] };
        // observables
        let no = self.observables.len();
        let mut oregs = self.obs.regs();
        oregs[nx..nx + np].copy_from_slice(&vals[..np]);
        let mut dextra: Vec<(usize, usize, f64)> = vec![];
        for (e, &j) in self.extras.iter().enumerate() {
            oregs[nx + np + e] = pv[j];
            if let Some(&f) = fpos.get(&j) { dextra.push((e, f, dpv[j])); }
        }
        let mut done_prep = false;
        let mut out = Vec::with_capacity(c.meas.len());
        for &(ti, o, _) in &c.meas {
            oregs[..nx].copy_from_slice(&x[ti * nx..(ti + 1) * nx]);
            oregs[nx + np + ne] = c.times[ti];
            if !done_prep { self.obs.prep(&mut oregs); done_prep = true; }
            self.obs.run(&mut oregs);
            let (y, sig) = (oregs[self.obs.outs[2 * o] as usize], oregs[self.obs.outs[2 * o + 1] as usize]);
            let (mut dy, mut ds) = (vec![0.0; if grad { nf } else { 0 }], vec![0.0; if grad { nf } else { 0 }]);
            if grad {
                for &(is_sig, l, r) in &self.obs_d[o] {
                    let d = oregs[r as usize];
                    let acc = if is_sig { &mut ds } else { &mut dy };
                    let l = l as usize;
                    if l < nx {
                        for (f, s) in sens.iter().enumerate() { if let Some(s) = s { acc[f] += d * s[ti * nx + l]; } }
                    } else if l < nx + np {
                        let k = l - nx;
                        for f in 0..nf { acc[f] += d * dpre[k * nf + f]; }
                    } else {
                        for &(e, f, de) in &dextra { if e == l - nx - np { acc[f] += d * de; } }
                    }
                }
            }
            let _ = no;
            out.push((y, sig, dy, ds));
        }
        Ok(out)
    }

    /// Parameter values (linear, every table row) and d value / d (value on scale) for theta (free, on scale).
    fn values(&self, theta: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
        if theta.len() != self.free.len() { return Err(format!("expected {} parameter values, got {}", self.free.len(), theta.len())); }
        let mut pv: Vec<f64> = self.params.iter().map(|p| p.nominal).collect();
        let mut dpv = vec![0.0; pv.len()];
        for (f, &j) in self.free.iter().enumerate() {
            let s = self.params[j].scale;
            pv[j] = s.unscale(theta[f]);
            dpv[j] = s.dunscale(theta[f]);
        }
        Ok((pv, dpv))
    }

    /// Model predictions (y, sigma) and their derivatives for every measurement, in measurement order.
    fn predict(&self, theta: &[f64], grad: bool) -> Result<Vec<Pred>, String> {
        let (pv, dpv) = self.values(theta)?;
        let per: Vec<Vec<Pred>> = self.conds.par_iter().map(|c| self.condition(c, &pv, &dpv, grad)).collect::<Result<_, _>>()?;
        let mut all: Vec<Option<Pred>> = vec![None; self.measurements.len()];
        for (c, ps) in self.conds.iter().zip(per) {
            for (&(_, _, q), p) in c.meas.iter().zip(ps) { all[q] = Some(p); }
        }
        Ok(all.into_iter().map(|p| p.expect("every measurement belongs to a condition")).collect())
    }

    /// Negative log-likelihood at theta (free parameters on their scales), with its gradient if `grad`, and the
    /// residuals and their Jacobian if `res` (least-squares problems only). A failed simulation or an
    /// undefined value (log of a negative observable) is an error; optimizers treat it as a rejected point.
    pub fn evaluate(&self, theta: &[f64], grad: bool, res: bool) -> Result<Eval, String> {
        if res && !self.least_squares { return Err("residuals need normal noise that depends on no free parameter".into()); }
        let preds = self.predict(theta, grad || res)?;
        let nf = self.free.len();
        let mut e = Eval { nllh: 0.0, ..Default::default() };
        if grad { e.grad = vec![0.0; nf]; }
        if res { e.res = Vec::with_capacity(preds.len()); e.jac = Vec::with_capacity(preds.len() * nf); }
        for (q, (y, sig, dy, ds)) in preds.iter().enumerate() {
            let ms = &self.measurements[q];
            let o = &self.observables[self.obs_index(&ms.observable)];
            let (hy, dh) = o.transform.h(*y);
            let (hm, _) = o.transform.h(ms.value);
            let r = hy - hm;
            if !r.is_finite() || !(*sig > 0.0) || !sig.is_finite() {
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
    }

    fn obs_index(&self, id: &str) -> usize { self.observables.iter().position(|o| o.id == id).unwrap() }

    /// Simulated observables (y) of every measurement at theta, in measurement order.
    pub fn simulate_measurements(&self, theta: &[f64]) -> Result<Vec<f64>, String> {
        Ok(self.predict(theta, false)?.into_iter().map(|p| p.0).collect())
    }
}
