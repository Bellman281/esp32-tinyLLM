//! A running digest over a generated token stream.
//!
//! WHY THIS EXISTS. This repo's central claim is that the Rust engine produces
//! output identical to the C reference it was ported from, and the parity
//! suite enforces that on the host to the bit: `golden.rs` against PyTorch
//! logits, `cli_parity.rs` byte-for-byte against the compiled C binary,
//! `ppl_parity.rs` on exact fp32 cross-entropy.
//!
//! None of those reach the firmware. `llm-firmware` runs a different code
//! path -- an int4-staged output head split across two LX7 cores, a dual-core
//! matvec override, per-core attention heads, and a greedy pick computed from
//! two running maxima instead of a scan -- and the only thing that had ever
//! checked its 400-token output was a human reading the story and deciding it
//! looked the same. Every performance figure in `benchmarks/device.toml` is
//! reported alongside "byte-identical output", and that phrase was resting on
//! that reading.
//!
//! So the device emits a number. `TokenDigest` folds every token id the
//! firmware prints into one 64-bit value, and the same function running on the
//! host over the same prompt and settings produces the expected value, pinned
//! in `model/models.toml` and gated by
//! `llm-host/tests/token_stream_digest.rs`. A mismatch is one glance instead
//! of a diff of 400 tokens of prose, and -- more to the point -- a mismatch is
//! *visible at all*, which it previously was not.
//!
//! WHY FNV-1a AND NOT A CHECKSUM WORTH THE NAME. This detects accidental
//! divergence between two implementations of the same arithmetic, not tampering.
//! FNV-1a is eight lines, has no tables, needs no allocation, and runs in
//! `no_std` on both sides -- which matters more here than collision resistance,
//! because the two things being compared differ by a code path rather than by
//! an adversary. A single flipped token changes the digest with overwhelming
//! probability, which is the whole requirement.
//!
//! WHY IDS AND NOT TEXT. The token ids are what the engine actually decides;
//! the text is a vocab lookup downstream of them. Digesting ids fails on the
//! first wrong *decision*, and it cannot be accidentally satisfied by two
//! different ids that happen to decode to the same bytes.

/// FNV-1a 64-bit offset basis and prime, per the reference specification.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Folds a sequence of token ids into a 64-bit value, in order.
///
/// Order-sensitive by construction: the same ids emitted in a different
/// sequence give a different digest, which is the point -- a decode that goes
/// wrong at step 200 and right again afterwards must not be able to hide.
#[derive(Clone, Copy, Debug)]
pub struct TokenDigest {
    hash: u64,
    count: u32,
}

impl Default for TokenDigest {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenDigest {
    pub const fn new() -> Self {
        Self {
            hash: FNV_OFFSET,
            count: 0,
        }
    }

    /// Fold in one token id.
    ///
    /// The id is hashed as a FIXED-WIDTH four-byte little-endian `u32`, never
    /// as a variable-length encoding. With a variable-length one, the stream
    /// `[0x0102, 0x03]` and the stream `[0x01, 0x0203]` would present the same
    /// bytes to the hash and collide trivially -- a real hazard here, since
    /// this vocab spans one- and two-byte id ranges. Four bytes always, so
    /// token boundaries are part of what is hashed.
    ///
    /// Ids above `u32::MAX` cannot occur (vocab is 32,768) and are truncated
    /// rather than checked, to keep this branch-free in a per-token path.
    pub fn push(&mut self, tok: usize) {
        for b in (tok as u32).to_le_bytes() {
            self.hash ^= b as u64;
            self.hash = self.hash.wrapping_mul(FNV_PRIME);
        }
        self.count += 1;
    }

    /// The digest so far.
    pub fn value(&self) -> u64 {
        self.hash
    }

    /// How many tokens have been folded in.
    ///
    /// Reported alongside the digest because a run that stops early -- the
    /// firmware breaks out of its loop at `pos >= seq_len` -- would otherwise
    /// present as a plain mismatch, and "wrong tokens" and "fewer tokens" want
    /// different investigations.
    pub fn count(&self) -> u32 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The empty digest is the offset basis, and folding is order-sensitive.
    #[test]
    fn digest_is_order_sensitive() {
        assert_eq!(TokenDigest::new().value(), FNV_OFFSET);
        assert_eq!(TokenDigest::new().count(), 0);

        let mut a = TokenDigest::new();
        let mut b = TokenDigest::new();
        for t in [7usize, 11, 13] {
            a.push(t);
        }
        for t in [7usize, 13, 11] {
            b.push(t);
        }
        assert_ne!(a.value(), b.value(), "reordering must change the digest");
        assert_eq!(a.count(), b.count());
    }

    /// The concatenation hazard the fixed-width encoding exists to prevent.
    #[test]
    fn token_boundaries_are_part_of_the_digest() {
        let mut a = TokenDigest::new();
        a.push(0x0102);
        a.push(0x03);
        let mut b = TokenDigest::new();
        b.push(0x01);
        b.push(0x0203);
        assert_ne!(a.value(), b.value());
    }

    /// A single changed token must change the digest, wherever it sits.
    ///
    /// `no_std`, so this walks fixed arrays rather than `Vec`.
    #[test]
    fn one_flipped_token_changes_the_digest() {
        const N: usize = 64;
        let mut base = [0usize; N];
        for (i, slot) in base.iter_mut().enumerate() {
            *slot = i * 397 + 1; // spread across the id range, not 0..64
        }
        let digest_of = |xs: &[usize; N]| {
            let mut d = TokenDigest::new();
            for &t in xs.iter() {
                d.push(t);
            }
            d.value()
        };
        let a = digest_of(&base);
        for i in 0..N {
            let mut c = base;
            c[i] += 1;
            assert_ne!(a, digest_of(&c), "flipping index {i} went undetected");
        }
    }
}
