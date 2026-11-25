# libroadrunner_nitro

The compute engine of [Tellurium Nitro](https://github.com/gshinkim/tellurium-nitro): libroadrunner's
simulation engine rewritten in Rust with a Cranelift JIT. No libroadrunner, no libsbml.

- **Python package** `libroadrunner_nitro` (`import roadrunner_nitro`), built with maturin.
- **C API** `libroadrunner_nitro_c_api`: all 350 functions of libroadrunner's C API (`rrc_api.h`,
  `rrc_types.h`, `rrc_utilities.h`), same names, arguments and memory rules.

This repository is the `rrs/` directory of tellurium-nitro, split out with its history.

## Build

Needs Rust (https://rustup.rs) and Python 3.9+.

```bash
pip install maturin
maturin develop --release          # Python package: import roadrunner_nitro
cargo build --release -p roadrunner_nitro_c_api   # C library in target/release/
```

`.cargo/config.toml` builds with `-C target-cpu=native`; remove it for portable binaries.

## C API

Headers are in `capi/include/` (libroadrunner's own, unchanged). Example: `capi/examples/simulate.c`.

```bash
python capi/tests/check_exports.py   # 350 declared / 350 exported
cc -I capi/include capi/examples/simulate.c \
   -L target/release -lroadrunner_nitro_c_api -Wl,-rpath,$PWD/target/release -o simulate
./simulate my_model.xml
```

## Tests

`tests/` compares results against stored libroadrunner references (`tests/reference/*.npz`). A few tests use
files from tellurium-nitro (`tellurium_nitro`, `benchmarks/`); `test_sbml_conversion.py` reads the SBML Test
Suite from `SBML_SEMANTIC_DIR`.

## Credits

`crates/antimony` and `crates/sbml-io` come from
[sys-bio/PathwayDesigner-rs](https://github.com/sys-bio/PathwayDesigner-rs) (see `crates/UPSTREAM`). The C
headers in `capi/include/` come from [sys-bio/roadrunner](https://github.com/sys-bio/roadrunner) (Apache 2.0,
University of Washington).

## License

Apache License 2.0, see `LICENSE`.
