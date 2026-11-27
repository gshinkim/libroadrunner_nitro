//! PEtab v1 import, in Rust (spec: "PEtab import"): the YAML problem file, the parameter, condition,
//! observable and measurement tables, and the SBML model, into a `ProblemDef`.
//!
//! Stage 1 covers the core tables. Features of later stages (placeholders, pre-equilibration, priors,
//! steady-state measurements) are refused with an error naming the table, row and column, never ignored.
use super::problem::{Measurement, Noise, ObservableDef, ParamDef, ProblemDef, Scale, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The PEtab YAML subset PEtab files use: top-level and list-item `key: value`, block lists (`- x`) and
/// inline lists (`[a, b]`). Returns every key with its scalar values, in order.
fn yaml_values(text: &str) -> HashMap<String, Vec<String>> {
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    let mut cur: Option<String> = None;
    let unq = |s: &str| s.trim().trim_matches(|c| c == '"' || c == '\'').to_string();
    for raw in text.lines() {
        let line = match raw.find(" #") { Some(i) => &raw[..i], None => raw };
        let mut l = line.trim();
        if l.is_empty() || l.starts_with('#') || l == "---" { continue; }
        let mut item = false;
        while let Some(r) = l.strip_prefix("- ").or_else(|| (l == "-").then_some("")) { l = r.trim_start(); item = true; }
        if l.is_empty() { continue; }
        match l.split_once(':').filter(|(k, _)| !k.contains(' ') || k.starts_with('"')) {
            Some((k, v)) if !k.contains(['/', '\\']) => {
                let (k, v) = (unq(k), v.trim());
                if v.is_empty() { cur = Some(k.clone()); out.entry(k).or_default(); continue; }
                let e = out.entry(k.clone()).or_default();
                if let Some(inner) = v.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
                    e.extend(inner.split(',').map(unq).filter(|s| !s.is_empty()));
                } else { e.push(unq(v)); }
                cur = Some(k);
            }
            _ => if item { if let Some(k) = &cur { out.entry(k.clone()).or_default().push(unq(l)); } },
        }
    }
    out
}

/// A tab-separated table: header and rows (cells trimmed; short rows padded with empty cells).
struct Table { name: String, head: Vec<String>, rows: Vec<Vec<String>> }

impl Table {
    fn read(path: &Path, name: &str) -> Result<Table, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{name} table {}: {e}", path.display()))?;
        let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let head: Vec<String> = lines.next().ok_or(format!("{name} table: empty"))?.split('\t').map(|s| s.trim().to_string()).collect();
        let rows = lines.map(|l| {
            let mut r: Vec<String> = l.split('\t').map(|s| s.trim().to_string()).collect();
            r.resize(head.len().max(r.len()), String::new());
            r
        }).collect();
        Ok(Table { name: name.into(), head, rows })
    }
    fn col(&self, c: &str) -> Option<usize> { self.head.iter().position(|h| h == c) }
    fn need(&self, c: &str) -> Result<usize, String> { self.col(c).ok_or(format!("{} table: missing column '{c}'", self.name)) }
    fn at(&self, row: usize, c: &str) -> String {
        format!("{} table, row {} (line {}), column '{c}'", self.name, row + 1, row + 2)
    }
    /// Refuse a column that must be empty in stage 1.
    fn refuse(&self, c: &str, why: &str) -> Result<(), String> {
        if let Some(k) = self.col(c) {
            if let Some(r) = self.rows.iter().position(|row| !row[k].is_empty() && row[k] != "nan" && row[k] != "NaN") {
                return Err(format!("{}: {why}", self.at(r, c)));
            }
        }
        Ok(())
    }
}

fn num(s: &str, at: impl Fn() -> String) -> Result<f64, String> {
    match s.trim() {
        "" | "nan" | "NaN" | "NAN" => Ok(f64::NAN),
        "inf" | "Inf" | "INF" | "+inf" => Ok(f64::INFINITY),
        "-inf" | "-Inf" | "-INF" => Ok(f64::NEG_INFINITY),
        t => t.parse::<f64>().map_err(|_| format!("{}: '{t}' is not a number", at())),
    }
}

/// Load a PEtab v1 problem from its YAML file.
pub fn load(yaml: &Path) -> Result<ProblemDef, String> {
    let text = std::fs::read_to_string(yaml).map_err(|e| format!("PEtab YAML {}: {e}", yaml.display()))?;
    let y = yaml_values(&text);
    let dir = yaml.parent().unwrap_or(Path::new("."));
    let ver = y.get("format_version").and_then(|v| v.first()).cloned().unwrap_or_default();
    if !(ver == "1" || ver.starts_with("1.")) {
        return Err(format!("PEtab YAML: format_version '{ver}' (stage 1 reads version 1; PEtab v2 comes in stage 4)"));
    }
    let one = |k: &str| -> Result<PathBuf, String> {
        let v = y.get(k).filter(|v| !v.is_empty()).ok_or(format!("PEtab YAML: missing '{k}'"))?;
        if v.len() > 1 { return Err(format!("PEtab YAML: several {k} (one problem with one file each is supported)")); }
        Ok(dir.join(&v[0]))
    };
    let many = |k: &str| -> Result<Vec<PathBuf>, String> {
        let v = y.get(k).filter(|v| !v.is_empty()).ok_or(format!("PEtab YAML: missing '{k}'"))?;
        Ok(v.iter().map(|f| dir.join(f)).collect())
    };
    let sbml_path = one("sbml_files")?;
    let mut sbml = std::fs::read_to_string(&sbml_path).map_err(|e| format!("SBML {}: {e}", sbml_path.display()))?;
    // hierarchical models (SBML comp package): flattened, as RoadRunner does on load
    if sbml.contains("http://www.sbml.org/sbml/level3/version1/comp/") {
        sbml = sbml_io::comp::flatten(&sbml, sbml_path.parent()).map_err(|e| format!("SBML {}: comp flattening: {e}", sbml_path.display()))?;
    }
    let params = parameters(&Table::read(&one("parameter_file")?, "parameter")?)?;
    let mut conditions = vec![];
    for f in many("condition_files")? { conditions.extend(conditions_of(&Table::read(&f, "condition")?)?); }
    let mut observables = vec![];
    for f in many("observable_files")? { observables.extend(observables_of(&Table::read(&f, "observable")?)?); }
    let mut measurements = vec![];
    for f in many("measurement_files")? { measurements.extend(measurements_of(&Table::read(&f, "measurement")?)?); }
    Ok(ProblemDef { sbml, params, conditions, observables, measurements })
}

fn parameters(t: &Table) -> Result<Vec<ParamDef>, String> {
    let (id, sc, lb, ub, nom, est) = (t.need("parameterId")?, t.col("parameterScale"), t.col("lowerBound"), t.col("upperBound"), t.col("nominalValue"), t.need("estimate")?);
    for c in ["objectivePriorType", "objectivePriorParameters"] { t.refuse(c, "priors come in stage 3")?; }
    let mut out = vec![];
    for (r, row) in t.rows.iter().enumerate() {
        let get = |k: Option<usize>| k.map_or("", |k| row[k].as_str());
        let estimate = match row[est].as_str() { "1" => true, "0" => false, v => return Err(format!("{}: '{v}' (0 or 1)", t.at(r, "estimate"))) };
        let scale = Scale::parse(get(sc)).map_err(|e| format!("{}: {e}", t.at(r, "parameterScale")))?;
        let p = ParamDef {
            id: row[id].clone(), scale,
            lb: num(get(lb), || t.at(r, "lowerBound"))?, ub: num(get(ub), || t.at(r, "upperBound"))?,
            nominal: num(get(nom), || t.at(r, "nominalValue"))?, estimate,
        };
        if p.id.is_empty() { return Err(format!("{}: empty", t.at(r, "parameterId"))); }
        if estimate && (p.lb.is_nan() || p.ub.is_nan()) { return Err(format!("{}: estimated parameter needs both bounds", t.at(r, "lowerBound"))); }
        out.push(p);
    }
    Ok(out)
}

fn conditions_of(t: &Table) -> Result<Vec<(String, Vec<(String, Value)>)>, String> {
    let id = t.need("conditionId")?;
    let targets: Vec<usize> = (0..t.head.len()).filter(|&k| k != id && t.head[k] != "conditionName").collect();
    let mut out = vec![];
    for (r, row) in t.rows.iter().enumerate() {
        let mut e = vec![];
        for &k in &targets {
            let v = row[k].trim();
            let val = match v.parse::<f64>() {
                Ok(x) => Value::Num(x),
                Err(_) if v.is_empty() || v.eq_ignore_ascii_case("nan") => Value::Num(f64::NAN),
                Err(_) => Value::Par(v.to_string()),
            };
            e.push((t.head[k].clone(), val));
        }
        if row[id].is_empty() { return Err(format!("{}: empty", t.at(r, "conditionId"))); }
        out.push((row[id].clone(), e));
    }
    Ok(out)
}

fn observables_of(t: &Table) -> Result<Vec<ObservableDef>, String> {
    let (id, f, nf) = (t.need("observableId")?, t.need("observableFormula")?, t.need("noiseFormula")?);
    let (tr, nd) = (t.col("observableTransformation"), t.col("noiseDistribution"));
    let mut out = vec![];
    for (r, row) in t.rows.iter().enumerate() {
        let get = |k: Option<usize>| k.map_or("", |k| row[k].as_str());
        out.push(ObservableDef {
            id: row[id].clone(), formula: row[f].clone(), noise_formula: row[nf].clone(),
            transform: Scale::parse(get(tr)).map_err(|e| format!("{}: {e}", t.at(r, "observableTransformation")))?,
            noise: Noise::parse(get(nd)).map_err(|e| format!("{}: {e}", t.at(r, "noiseDistribution")))?,
        });
        if row[nf].is_empty() { return Err(format!("{}: empty", t.at(r, "noiseFormula"))); }
    }
    Ok(out)
}

/// A placeholder cell: values separated by ';', each a number or a parameter id ("" or "nan": none).
fn placeholder_values(cell: &str) -> Vec<Value> {
    let c = cell.trim();
    if c.is_empty() || c.eq_ignore_ascii_case("nan") { return vec![]; }
    c.split(';').map(|v| { let v = v.trim(); v.parse::<f64>().map(Value::Num).unwrap_or_else(|_| Value::Par(v.to_string())) }).collect()
}

fn measurements_of(t: &Table) -> Result<Vec<Measurement>, String> {
    let (o, c, tm, v) = (t.need("observableId")?, t.need("simulationConditionId")?, t.need("time")?, t.need("measurement")?);
    let (op, np) = (t.col("observableParameters"), t.col("noiseParameters"));
    t.refuse("preequilibrationConditionId", "pre-equilibration is not supported yet")?;
    let mut out = vec![];
    for (r, row) in t.rows.iter().enumerate() {
        let time = num(&row[tm], || t.at(r, "time"))?;
        if time.is_infinite() { return Err(format!("{}: steady-state measurements are not supported yet", t.at(r, "time"))); }
        out.push(Measurement {
            observable: row[o].clone(), condition: row[c].clone(), time,
            value: num(&row[v], || t.at(r, "measurement"))?,
            observable_parameters: op.map_or(vec![], |k| placeholder_values(&row[k])),
            noise_parameters: np.map_or(vec![], |k| placeholder_values(&row[k])),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaml_subset() {
        let y = yaml_values("format_version: 1\nparameter_file: parameters.tsv\nproblems:\n- condition_files:\n  - c.tsv\n  measurement_files: [m1.tsv, 'm2.tsv']\n  observable_files:\n    - o.tsv   # comment\n  sbml_files:\n  - model.xml\n");
        assert_eq!(y["format_version"], ["1"]);
        assert_eq!(y["parameter_file"], ["parameters.tsv"]);
        assert_eq!(y["condition_files"], ["c.tsv"]);
        assert_eq!(y["measurement_files"], ["m1.tsv", "m2.tsv"]);
        assert_eq!(y["observable_files"], ["o.tsv"]);
        assert_eq!(y["sbml_files"], ["model.xml"]);
    }

    fn write_problem(dir: &Path, measurements: &str) {
        let ant = "model decay
            compartment C = 1
            species S1 in C, S2 in C
            J1: S1 -> S2; k1*S1
            J2: S2 -> ; k2*S2
            S1 = 10; S2 = 0
            k1 = 0.5; k2 = 0.2
        end";
        std::fs::create_dir_all(dir).unwrap();
        let w = |f: &str, s: &str| std::fs::write(dir.join(f), s).unwrap();
        w("model.xml", &antimony::sbml::write_sbml(&antimony::parse(ant).document));
        w("problem.yaml", "format_version: 1\nparameter_file: parameters.tsv\nproblems:\n  - condition_files: [conditions.tsv]\n    measurement_files:\n      - measurements.tsv\n    observable_files:\n      - observables.tsv\n    sbml_files:\n      - model.xml\n");
        w("parameters.tsv", "parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\nk1\tlog10\t0.001\t100\t0.5\t1\nk2\tlog10\t0.001\t100\t0.2\t1\nscale\tlog\t0.1\t10\t2\t1\nS1_init\tlin\t1\t50\t15\t1\nsigma\tlin\t0\t1\t0.1\t0\n");
        w("conditions.tsv", "conditionId\tconditionName\tS1\tk2\nc1\t\t\t\nc2\tsecond\tS1_init\t\nc3\t\t\t0.3\n");
        w("observables.tsv", "observableId\tobservableFormula\tnoiseFormula\tobservableTransformation\tnoiseDistribution\nobsA\tS1\tsigma\tlin\tnormal\nobsB\tscale*S2\t0.05\tlog10\tnormal\n");
        w("measurements.tsv", measurements);
    }

    #[test]
    fn petab_matches_native_problem() {
        use crate::fit::{Problem, Value};
        let dir = std::env::temp_dir().join(format!("rrn_petab_{}", std::process::id()));
        let mut m = String::from("observableId\tsimulationConditionId\ttime\tmeasurement\n");
        let mut native = vec![];
        for (q, c) in ["c1", "c2", "c3"].iter().enumerate() {
            for (k, t) in [0.0, 1.0, 4.0].iter().enumerate() {
                let (a, b) = (5.0 + q as f64 + 0.5 * k as f64, 0.5 + 0.1 * (q + k) as f64);
                m += &format!("obsA\t{c}\t{t}\t{a}\n");
                native.push((c.to_string(), "obsA", *t, a));
                if *t > 0.0 { m += &format!("obsB\t{c}\t{t}\t{b}\n"); native.push((c.to_string(), "obsB", *t, b)); }
            }
        }
        write_problem(&dir, &m);
        let def = load(&dir.join("problem.yaml")).unwrap();
        assert_eq!(def.conditions[1].1.len(), 2);
        assert!(matches!(&def.conditions[1].1[0].1, Value::Par(p) if p == "S1_init"));
        assert!(matches!(def.conditions[0].1[0].1, Value::Num(x) if x.is_nan()));
        let p = Problem::new(def).unwrap();
        assert_eq!(p.free_ids(), ["k1", "k2", "scale", "S1_init"]);
        let theta = [0.6f64.log10(), 0.15f64.log10(), 2.5f64.ln(), 12.0];
        let e = p.evaluate(&theta, true, true).unwrap();
        // the same problem through the native definition
        let mut def = load(&dir.join("problem.yaml")).unwrap();
        def.measurements = native.into_iter().map(|(c, o, t, v)| super::Measurement { condition: c, observable: o.into(), time: t, value: v, ..Default::default() }).collect();
        let e2 = Problem::new(def).unwrap().evaluate(&theta, true, true).unwrap();
        assert_eq!(e.nllh.to_bits(), e2.nllh.to_bits());
        // placeholder columns are read: values separated by ';', numbers or parameter ids
        write_problem(&dir, "observableId\tsimulationConditionId\ttime\tmeasurement\tobservableParameters\tnoiseParameters\nobsA\tc1\t1\t2\tscale;2.5\t0.1\n");
        let def = load(&dir.join("problem.yaml")).unwrap();
        let m = &def.measurements[0];
        assert!(matches!(&m.observable_parameters[..], [Value::Par(a), Value::Num(b)] if a == "scale" && *b == 2.5));
        assert!(matches!(m.noise_parameters[..], [Value::Num(x)] if x == 0.1));
        // features not supported yet are refused with their place
        write_problem(&dir, "observableId\tpreequilibrationConditionId\tsimulationConditionId\ttime\tmeasurement\nobsA\tc1\tc2\t1\t2\n");
        let err = load(&dir.join("problem.yaml")).err().unwrap();
        assert!(err.contains("measurement table, row 1") && err.contains("preequilibrationConditionId"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
