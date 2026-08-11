//! Reader for the model registry at `model/models.toml` — the single place
//! that says which model files exist, where they live, and what each is
//! expected to produce.
//!
//! Before this existed, each of the four parity test files carried its own
//! `const MODEL_BIN: &str = "../../reference-c/..."` and its own expected
//! constants, and the C tools had their own compiled-in defaults. Adding a
//! second model meant finding every one of them, and nothing stopped two
//! benchmark runs from silently using different settings.
//!
//! ## Why hand-parsed instead of the `toml` crate
//!
//! The format is a deliberately small subset (see `models.toml`'s own
//! header): `key = value` inside `[section]` blocks, values being quoted
//! strings, integers, floats, or single-line integer arrays. Pulling in
//! `toml` costs six transitive crates in a workspace whose entire portable
//! core (`llm-core`) has exactly two (`half`, `libm`) — and the same
//! restricted shape is what lets `scripts/model.sh` read this file with
//! grep/sed so the C tools get the same values without a TOML parser at
//! all. Both readers depend on the format staying inside that subset; the
//! parser below rejects what it doesn't understand rather than guessing.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Where the repository root sits relative to the working directory Cargo
/// uses for this crate's tests and binaries (`engine/llm-host`). Every path
/// in `models.toml` is relative to the repo root, so they all get joined
/// onto this.
pub const REPO_ROOT: &str = "../..";

/// Path to the registry itself, relative to `REPO_ROOT`.
pub const MANIFEST_PATH: &str = "model/models.toml";

/// One model's entry. Optional fields are genuinely optional: not every
/// model has a golden-logit reference or validation tokens (see
/// `models.toml`).
#[derive(Debug, Clone, Default)]
pub struct Model {
    pub name: String,
    pub description: String,
    pub bin: String,
    pub bin_sha256: String,
    pub vocab: Option<String>,
    pub val_tokens: Option<String>,
    pub golden: Option<String>,
    pub prompt_ids: Vec<usize>,
    pub n_generate: Option<usize>,
    pub ppl_windows: Option<usize>,
    pub ppl_ce_fp32: Option<f64>,
    pub ppl_ce_int8: Option<f64>,
}

fn at_root(rel: &str) -> PathBuf {
    Path::new(REPO_ROOT).join(rel)
}

impl Model {
    /// Absolute-ish path to the weights, ready to hand to `fs::read`.
    pub fn bin_path(&self) -> PathBuf {
        at_root(&self.bin)
    }

    /// Panics with a pointed message rather than returning `None` — a test
    /// that asks for a field this model doesn't declare is a bug in the
    /// test or the registry, not a condition to handle at runtime.
    pub fn vocab_path(&self) -> PathBuf {
        at_root(
            self.vocab.as_ref().unwrap_or_else(|| {
                panic!("model '{}' has no `vocab` in {MANIFEST_PATH}", self.name)
            }),
        )
    }

    pub fn val_tokens_path(&self) -> PathBuf {
        at_root(self.val_tokens.as_ref().unwrap_or_else(|| {
            panic!(
                "model '{}' has no `val_tokens` in {MANIFEST_PATH}",
                self.name
            )
        }))
    }

    pub fn golden_path(&self) -> PathBuf {
        at_root(
            self.golden.as_ref().unwrap_or_else(|| {
                panic!("model '{}' has no `golden` in {MANIFEST_PATH}", self.name)
            }),
        )
    }
}

/// The parsed registry.
#[derive(Debug, Clone, Default)]
pub struct Manifest {
    pub default: String,
    pub models: BTreeMap<String, Model>,
}

impl Manifest {
    pub fn get(&self, name: &str) -> &Model {
        self.models.get(name).unwrap_or_else(|| {
            let known: Vec<&str> = self.models.keys().map(String::as_str).collect();
            panic!("no model '{name}' in {MANIFEST_PATH} (known: {known:?})")
        })
    }

    pub fn default_model(&self) -> &Model {
        self.get(&self.default)
    }
}

/// Reads and parses the registry. Panics on a malformed or missing file —
/// every caller is a test or a dev CLI, and a broken registry should stop
/// the run loudly rather than fall back to a guessed default that might
/// silently benchmark the wrong model.
pub fn load() -> Manifest {
    let path = at_root(MANIFEST_PATH);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "read {}: {e} (tools and tests run from engine/llm-host, where the \
             repo root is {REPO_ROOT})",
            path.display()
        )
    });
    parse(&text)
}

/// Convenience: the registry's default model.
pub fn default_model() -> Model {
    load().default_model().clone()
}

/// Convenience: one model by name.
pub fn model(name: &str) -> Model {
    load().get(name).clone()
}

fn strip_comment(line: &str) -> &str {
    // No `#` ever appears inside the string values this format carries
    // (paths, hex digests, prose descriptions), so a plain split is
    // sufficient and keeps the parser honest about its own limits — see
    // this module's doc comment.
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}

fn parse(text: &str) -> Manifest {
    let mut out = Manifest::default();
    let mut section: Option<String> = None;

    for (lineno, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix('[') {
            let name = rest
                .strip_suffix(']')
                .unwrap_or_else(|| panic!("{MANIFEST_PATH}:{}: unterminated [section]", lineno + 1))
                .trim()
                .to_string();
            out.models.entry(name.clone()).or_insert_with(|| Model {
                name: name.clone(),
                ..Default::default()
            });
            section = Some(name);
            continue;
        }

        let (key, value) = line.split_once('=').unwrap_or_else(|| {
            panic!(
                "{MANIFEST_PATH}:{}: expected `key = value`, got {line:?}",
                lineno + 1
            )
        });
        let key = key.trim();
        let value = value.trim();

        let Some(sec) = section.clone() else {
            // Top-level keys (before any [section]).
            if key == "default" {
                out.default = unquote(value, lineno);
            } else {
                panic!(
                    "{MANIFEST_PATH}:{}: unknown top-level key {key:?}",
                    lineno + 1
                );
            }
            continue;
        };

        let m = out.models.get_mut(&sec).expect("section inserted above");
        match key {
            "description" => m.description = unquote(value, lineno),
            "bin" => m.bin = unquote(value, lineno),
            "bin_sha256" => m.bin_sha256 = unquote(value, lineno),
            "vocab" => m.vocab = Some(unquote(value, lineno)),
            "val_tokens" => m.val_tokens = Some(unquote(value, lineno)),
            "golden" => m.golden = Some(unquote(value, lineno)),
            "prompt_ids" => m.prompt_ids = parse_int_array(value, lineno),
            "n_generate" => m.n_generate = Some(parse_num(value, lineno)),
            "ppl_windows" => m.ppl_windows = Some(parse_num(value, lineno)),
            "ppl_ce_fp32" => m.ppl_ce_fp32 = Some(parse_float(value, lineno)),
            "ppl_ce_int8" => m.ppl_ce_int8 = Some(parse_float(value, lineno)),
            other => panic!(
                "{MANIFEST_PATH}:{}: unknown key {other:?} in [{sec}] -- add it to \
                 manifest.rs's match if it's intentional, rather than letting it \
                 be silently ignored",
                lineno + 1
            ),
        }
    }

    if out.default.is_empty() {
        panic!("{MANIFEST_PATH}: no top-level `default = \"...\"`");
    }
    if !out.models.contains_key(&out.default) {
        panic!(
            "{MANIFEST_PATH}: default = {:?} names no [section] in this file",
            out.default
        );
    }
    out
}

fn unquote(v: &str, lineno: usize) -> String {
    v.strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or_else(|| {
            panic!(
                "{MANIFEST_PATH}:{}: expected a quoted string, got {v:?}",
                lineno + 1
            )
        })
        .to_string()
}

fn parse_num(v: &str, lineno: usize) -> usize {
    v.parse().unwrap_or_else(|e| {
        panic!(
            "{MANIFEST_PATH}:{}: expected an integer, got {v:?}: {e}",
            lineno + 1
        )
    })
}

fn parse_float(v: &str, lineno: usize) -> f64 {
    v.parse().unwrap_or_else(|e| {
        panic!(
            "{MANIFEST_PATH}:{}: expected a float, got {v:?}: {e}",
            lineno + 1
        )
    })
}

fn parse_int_array(v: &str, lineno: usize) -> Vec<usize> {
    let inner = v
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or_else(|| {
            panic!(
                "{MANIFEST_PATH}:{}: expected a single-line [a, b, c] array, got {v:?}",
                lineno + 1
            )
        });
    inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse().unwrap_or_else(|e| {
                panic!(
                    "{MANIFEST_PATH}:{}: bad array element {s:?}: {e}",
                    lineno + 1
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
# a comment
default = "alpha"

[alpha]
description = "first"     # trailing comment
bin         = "a/model.bin"
bin_sha256  = "deadbeef"
vocab       = "a/vocab.h"
prompt_ids  = [1, 22, 333]
n_generate  = 400
ppl_ce_fp32 = 2.6671

[beta]
description = "second"
bin         = "b/model.bin"
bin_sha256  = "cafe"
golden      = "b/golden.txt"
"#;

    #[test]
    fn parses_sections_scalars_and_arrays() {
        let m = parse(SAMPLE);
        assert_eq!(m.default, "alpha");
        assert_eq!(m.models.len(), 2);

        let a = m.get("alpha");
        assert_eq!(a.name, "alpha");
        assert_eq!(a.description, "first");
        assert_eq!(a.bin, "a/model.bin");
        assert_eq!(a.prompt_ids, vec![1, 22, 333]);
        assert_eq!(a.n_generate, Some(400));
        assert_eq!(a.ppl_ce_fp32, Some(2.6671));
        // Not declared for this model -> stays None rather than defaulting.
        assert_eq!(a.golden, None);
        assert_eq!(a.ppl_ce_int8, None);

        let b = m.get("beta");
        assert_eq!(b.golden.as_deref(), Some("b/golden.txt"));
        assert_eq!(b.vocab, None);
        assert!(b.prompt_ids.is_empty());
    }

    #[test]
    fn default_model_resolves() {
        assert_eq!(parse(SAMPLE).default_model().name, "alpha");
    }

    #[test]
    fn paths_are_joined_onto_the_repo_root() {
        let a = parse(SAMPLE).get("alpha").clone();
        assert_eq!(a.bin_path(), Path::new(REPO_ROOT).join("a/model.bin"));
        assert_eq!(a.vocab_path(), Path::new(REPO_ROOT).join("a/vocab.h"));
    }

    #[test]
    #[should_panic(expected = "unknown key")]
    fn unknown_key_is_an_error_not_a_silent_skip() {
        parse("default = \"x\"\n[x]\nbogus = \"y\"\n");
    }

    #[test]
    #[should_panic(expected = "names no [section]")]
    fn default_must_name_a_real_section() {
        parse("default = \"missing\"\n[x]\nbin = \"b\"\n");
    }

    #[test]
    #[should_panic(expected = "no model 'nope'")]
    fn unknown_model_lookup_lists_what_is_available() {
        parse(SAMPLE).get("nope");
    }

    /// The real registry must parse, and its declared files must exist —
    /// this is the test that catches a path typo or a model file that moved,
    /// which is otherwise only discovered by whichever parity test happens
    /// to read it first.
    #[test]
    fn real_registry_parses_and_its_files_exist() {
        let m = load();
        assert!(!m.models.is_empty());
        for (name, model) in &m.models {
            assert!(!model.bin.is_empty(), "model '{name}' declares no `bin`");
            assert!(
                model.bin_path().exists(),
                "model '{name}': bin {} does not exist",
                model.bin_path().display()
            );
            for (label, p) in [
                ("vocab", model.vocab.as_ref()),
                ("val_tokens", model.val_tokens.as_ref()),
                ("golden", model.golden.as_ref()),
            ] {
                if let Some(rel) = p {
                    assert!(
                        at_root(rel).exists(),
                        "model '{name}': {label} {rel} does not exist"
                    );
                }
            }
        }
    }
}
