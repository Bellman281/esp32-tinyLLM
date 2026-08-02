//! Parses the C-generated `vocab.h` (token id -> raw UTF-8 bytes table)
//! directly from its `.h` text, instead of `#include`-ing it like the C
//! tools do. `vocab.h` is data (produced by `src/gen_assets.py`), not
//! logic, so this is just a small text parser -- no model math involved,
//! and deliberately lives in `llm-host` rather than `llm-core` since it's
//! host/decode-side tooling, not part of the portable inference engine.

use std::fs;

pub struct Vocab {
    pub n: usize,
    blob: Vec<u8>,
    off: Vec<usize>,
}

impl Vocab {
    pub fn load(path: &str) -> Result<Vocab, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;

        let n = parse_define(&text, "VOCAB_N")
            .ok_or_else(|| "VOCAB_N not found".to_string())?;
        let blob = parse_array_u8(&text, "VOCAB_BLOB")
            .ok_or_else(|| "VOCAB_BLOB not found".to_string())?;
        let off = parse_array_usize(&text, "VOCAB_OFF")
            .ok_or_else(|| "VOCAB_OFF not found".to_string())?;

        if off.len() != n + 1 {
            return Err(format!(
                "VOCAB_OFF has {} entries, expected VOCAB_N+1 = {}",
                off.len(),
                n + 1
            ));
        }
        Ok(Vocab { n, blob, off })
    }

    /// Bytes for token `tok`, or `None` if out of range (mirrors the C
    /// `emit()`'s `if (tok >= VOCAB_N) return;` guard).
    pub fn bytes(&self, tok: usize) -> Option<&[u8]> {
        if tok >= self.n {
            return None;
        }
        Some(&self.blob[self.off[tok]..self.off[tok + 1]])
    }
}

fn parse_define(text: &str, name: &str) -> Option<usize> {
    let marker = format!("#define {name} ");
    let start = text.find(&marker)? + marker.len();
    let rest = &text[start..];
    let end = rest.find('\n').unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

/// Extracts the `{ ... }` initializer body of `static const ... NAME[..] =
/// { ... };` and parses it as a comma-separated list of `u8`.
fn parse_array_u8(text: &str, name: &str) -> Option<Vec<u8>> {
    parse_array_body(text, name)?
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u16>().ok().map(|v| v as u8)) // values are 0..255
        .collect()
}

fn parse_array_usize(text: &str, name: &str) -> Option<Vec<usize>> {
    parse_array_body(text, name)?
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().ok())
        .collect()
}

fn parse_array_body<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let decl = format!("{name}[");
    let decl_at = text.find(&decl)?;
    let brace_at = text[decl_at..].find('{')? + decl_at;
    let close_at = text[brace_at..].find("};")? + brace_at;
    Some(&text[brace_at + 1..close_at])
}
