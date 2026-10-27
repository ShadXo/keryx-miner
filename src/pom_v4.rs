//! PoM v4 (D=32 re-walk) — wire structs, constants, host proof-build. Byte-exact mirror of the
//! node's `consensus/core/src/pom_v4.rs`. Field order and salts MUST stay bit-identical.

use crate::pom::{blake, merkle_root, mix64, verify_merkle, WeightIndex};
use crate::pom_v3::{dot_i8, fold64, rho8, rho_tweak, snippet_fold};
use anyhow::{anyhow, Result};
use borsh::{BorshDeserialize, BorshSerialize};

pub const POM_V4_D: usize = 32;
pub const POM_V4_K: usize = 256;
pub const POM_V4_CHUNK_BYTES: usize = 32;
pub const POM_V4_TILE_BYTES: usize = POM_V4_D * POM_V4_D; // 1 KB
pub const POM_V4_TILE_CHUNKS: u64 = (POM_V4_TILE_BYTES / POM_V4_CHUNK_BYTES) as u64;
pub const POM_V4_SNIPPET_BYTES: usize = 32;
pub const POM_V4_TILE_SUBTREE_DEPTH: u32 = 5; // log2(POM_V4_TILE_CHUNKS)

/// sha256("keryx-v4-s0-row-salt")
pub const POM_V4_S0_ROW_SALT: u64 = 0x03421325594C3C51;
/// sha256("keryx-v4-offset-first-salt")
pub const POM_V4_OFFSET_FIRST_SALT: u64 = 0x6D1CCF96AC4D76F9;
/// sha256("keryx-v4-offset-step-salt")
pub const POM_V4_OFFSET_STEP_SALT: u64 = 0x89050E78D34609EF;

/// Merkle range proof for one tile (path from the tile's aligned subtree root up to R_T).
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomV4RangeProof {
    pub path: Vec<[u8; 32]>,
}

/// v4 walk witness — mirror of the node's `PomProofV4` (borsh field order).
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomProofV4 {
    pub tier: u8,
    pub tiles: Vec<Vec<u8>>,
    pub merkle: Vec<PomV4RangeProof>,
}

#[inline]
pub fn v4_first_offset(seed: u64, n_tiles: u64) -> u64 {
    mix64(seed ^ POM_V4_OFFSET_FIRST_SALT) % n_tiles
}

#[inline]
pub fn v4_next_offset(seed: u64, step: u64, snippet: &[u8; POM_V4_SNIPPET_BYTES], n_tiles: u64) -> u64 {
    mix64(seed ^ (step + 1).wrapping_mul(POM_V4_OFFSET_STEP_SALT) ^ snippet_fold(snippet)) % n_tiles
}

pub fn v4_initial_state(seed: u64) -> Vec<u8> {
    let mut s = vec![0u8; POM_V4_D * POM_V4_D];
    for r in 0..POM_V4_D {
        let mut h = mix64(seed ^ POM_V4_S0_ROW_SALT.wrapping_add(r as u64));
        for k4 in 0..POM_V4_D / 4 {
            h = mix64(h);
            s[r * POM_V4_D + k4 * 4..r * POM_V4_D + k4 * 4 + 4].copy_from_slice(&(h as u32).to_le_bytes());
        }
    }
    s
}

pub fn v4_transition(state: &[u8], tile: &[u8], step: u32) -> Vec<u8> {
    let mut next = vec![0u8; POM_V4_D * POM_V4_D];
    v4_transition_into(&mut next, state, tile, step);
    next
}

/// `dst = rho(src x tile)` without allocating; `dst` must not alias `src`. Ported from the
/// ocminer (suprnova) fork: the int8 dot is SIMD (AVX2 -> SSE4.1 -> NEON -> scalar), and the
/// feature check is made once per step for the whole 32x32 transition, so the
/// #[target_feature] dot inlines into the cell loop instead of being called 1,024 times.
/// Bit-identical on every path: each product is exact in i32 and the sum cannot overflow
/// (32 x 128^2 < 2^31), so the reduction order does not matter.
pub fn v4_transition_into(dst: &mut [u8], src: &[u8], tile: &[u8], step: u32) {
    assert_eq!(dst.len(), POM_V4_D * POM_V4_D);
    assert_eq!(src.len(), POM_V4_D * POM_V4_D);
    assert_eq!(tile.len(), POM_V4_TILE_BYTES);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { v4_transition_into_avx2(dst, src, tile, step) };
            return;
        }
        if is_x86_feature_detected!("sse4.1") {
            unsafe { v4_transition_into_sse41(dst, src, tile, step) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { v4_transition_into_neon(dst, src, tile, step) };
        return;
    }
    #[allow(unreachable_code)]
    v4_transition_into_scalar(dst, src, tile, step);
}

/// The reference transition: the fallback, and the oracle the SIMD paths are tested against.
fn v4_transition_into_scalar(dst: &mut [u8], src: &[u8], tile: &[u8], step: u32) {
    for x in 0..POM_V4_D {
        let row = &src[x * POM_V4_D..(x + 1) * POM_V4_D];
        for j in 0..POM_V4_D {
            let col = &tile[j * POM_V4_D..(j + 1) * POM_V4_D];
            dst[x * POM_V4_D + j] = rho8(dot_i8(row, col), rho_tweak(step, x as u32, j as u32));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot32_avx2(row: &[u8], col: &[u8]) -> i32 {
    use std::arch::x86_64::*;
    // 16 i8 -> 16 i16, madd pairwise -> 8 i32; twice for the 32-byte row.
    let r0 = _mm_loadu_si128(row.as_ptr() as *const __m128i);
    let c0 = _mm_loadu_si128(col.as_ptr() as *const __m128i);
    let p0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(r0), _mm256_cvtepi8_epi16(c0));
    let r1 = _mm_loadu_si128(row.as_ptr().add(16) as *const __m128i);
    let c1 = _mm_loadu_si128(col.as_ptr().add(16) as *const __m128i);
    let p1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(r1), _mm256_cvtepi8_epi16(c1));
    let sum = _mm256_add_epi32(p0, p1);
    let s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    let s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    let s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    _mm_cvtsi128_si32(s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dot32_sse41(row: &[u8], col: &[u8]) -> i32 {
    use std::arch::x86_64::*;
    let mut acc = _mm_setzero_si128();
    let mut off = 0usize;
    while off < POM_V4_D {
        let r = _mm_loadl_epi64(row.as_ptr().add(off) as *const __m128i);
        let c = _mm_loadl_epi64(col.as_ptr().add(off) as *const __m128i);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_cvtepi8_epi16(r), _mm_cvtepi8_epi16(c)));
        off += 8;
    }
    let acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 8));
    let acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 4));
    _mm_cvtsi128_si32(acc)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot32_neon(row: &[u8], col: &[u8]) -> i32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_s32(0);
    for half in [0usize, 16] {
        let a = vld1q_s8(row.as_ptr().add(half) as *const i8);
        let b = vld1q_s8(col.as_ptr().add(half) as *const i8);
        let a_lo = vmovl_s8(vget_low_s8(a));
        let a_hi = vmovl_s8(vget_high_s8(a));
        let b_lo = vmovl_s8(vget_low_s8(b));
        let b_hi = vmovl_s8(vget_high_s8(b));
        acc = vaddq_s32(acc, vmull_s16(vget_low_s16(a_lo), vget_low_s16(b_lo)));
        acc = vaddq_s32(acc, vmull_high_s16(a_lo, b_lo));
        acc = vaddq_s32(acc, vmull_s16(vget_low_s16(a_hi), vget_low_s16(b_hi)));
        acc = vaddq_s32(acc, vmull_high_s16(a_hi, b_hi));
    }
    vaddvq_s32(acc)
}

/// One SIMD transition body per target feature, so each inlines its own dot.
macro_rules! v4_transition_simd {
    ($name:ident, $feature:literal, $dot:ident) => {
        #[target_feature(enable = $feature)]
        unsafe fn $name(dst: &mut [u8], src: &[u8], tile: &[u8], step: u32) {
            for x in 0..POM_V4_D {
                let row = &src[x * POM_V4_D..(x + 1) * POM_V4_D];
                for j in 0..POM_V4_D {
                    let col = &tile[j * POM_V4_D..(j + 1) * POM_V4_D];
                    dst[x * POM_V4_D + j] = rho8($dot(row, col), rho_tweak(step, x as u32, j as u32));
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
v4_transition_simd!(v4_transition_into_avx2, "avx2", dot32_avx2);
#[cfg(target_arch = "x86_64")]
v4_transition_simd!(v4_transition_into_sse41, "sse4.1", dot32_sse41);
#[cfg(target_arch = "aarch64")]
v4_transition_simd!(v4_transition_into_neon, "neon", dot32_neon);

fn v4_state_leaves(state: &[u8]) -> Vec<[u8; 32]> {
    (0..POM_V4_D).map(|r| blake(&state[r * POM_V4_D..(r + 1) * POM_V4_D])).collect()
}

pub fn v4_state_root(state: &[u8]) -> [u8; 32] {
    merkle_root(&v4_state_leaves(state))
}

pub(crate) fn v4_tile_subtree_root(tile: &[u8]) -> [u8; 32] {
    let leaves: Vec<[u8; 32]> = tile.chunks(POM_V4_CHUNK_BYTES).map(blake).collect();
    merkle_root(&leaves)
}

/// Re-walk `seed` reading tiles from `index`, returning the proof and the derived `final_state`.
pub fn build_proof_v4(tier: u8, seed: u64, index: &WeightIndex) -> Result<(PomProofV4, u64)> {
    let n_tiles = index.n_chunks / POM_V4_TILE_CHUNKS;
    if n_tiles == 0 {
        return Err(anyhow!("blob too small for the v4 walk"));
    }
    let mut state = v4_initial_state(seed);
    let mut next = vec![0u8; POM_V4_D * POM_V4_D];
    let mut off = v4_first_offset(seed, n_tiles);
    let mut tiles = Vec::with_capacity(POM_V4_K);
    let mut merkle = Vec::with_capacity(POM_V4_K);
    for step in 1..=POM_V4_K as u64 {
        let mut tile = Vec::with_capacity(POM_V4_TILE_BYTES);
        for c in 0..POM_V4_TILE_CHUNKS {
            tile.extend_from_slice(&index.read_chunk_bytes(off * POM_V4_TILE_CHUNKS + c));
        }
        let snippet: [u8; 32] = tile[..POM_V4_SNIPPET_BYTES].try_into().unwrap();
        let path = index.merkle_path(off * POM_V4_TILE_CHUNKS)[POM_V4_TILE_SUBTREE_DEPTH as usize..].to_vec();
        merkle.push(PomV4RangeProof { path });
        v4_transition_into(&mut next, &state, &tile, step as u32);
        std::mem::swap(&mut state, &mut next);
        tiles.push(tile);
        if step < POM_V4_K as u64 {
            off = v4_next_offset(seed, step, &snippet, n_tiles);
        }
    }
    let final_state = fold64(&v4_state_root(&state));
    Ok((PomProofV4 { tier, tiles, merkle }, final_state))
}

/// Pre-submit self-check: re-walk the proof against `r_t` and return the derived `final_state`.
pub fn verify_proof_v4(seed: u64, proof: &PomProofV4, r_t: &[u8; 32], n_chunks: u64) -> Result<u64> {
    if proof.tiles.len() != POM_V4_K || proof.merkle.len() != POM_V4_K {
        return Err(anyhow!("v4 proof wrong shape"));
    }
    let n_tiles = n_chunks / POM_V4_TILE_CHUNKS;
    if n_tiles == 0 {
        return Err(anyhow!("blob too small"));
    }
    let mut state = v4_initial_state(seed);
    let mut next = vec![0u8; POM_V4_D * POM_V4_D];
    let mut off = v4_first_offset(seed, n_tiles);
    for step in 1..=POM_V4_K {
        let tile = &proof.tiles[step - 1];
        if tile.len() != POM_V4_TILE_BYTES {
            return Err(anyhow!("v4 tile wrong shape"));
        }
        if !verify_merkle(v4_tile_subtree_root(tile), off, &proof.merkle[step - 1].path, r_t) {
            return Err(anyhow!("v4 tile fails range proof at step {step}"));
        }
        v4_transition_into(&mut next, &state, tile, step as u32);
        std::mem::swap(&mut state, &mut next);
        if step < POM_V4_K {
            let snippet: [u8; 32] = tile[..POM_V4_SNIPPET_BYTES].try_into().unwrap();
            off = v4_next_offset(seed, step as u64, &snippet, n_tiles);
        }
    }
    Ok(fold64(&v4_state_root(&state)))
}

/// Test-only reference walk (host): S_0..=S_K + snippets + offsets over an in-RAM blob.
///
/// Exists to give the v4 GPU kernel the byte-exactness harness v3 has had since H6 and v4
/// shipped without. Without it a subtly wrong kernel is indistinguishable from a correct one
/// until the node rejects blocks — and any optimisation of the walk is unverifiable.
#[cfg(test)]
pub(crate) fn ref_walk(seed: u64, blob: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u64>) {
    let d2 = POM_V4_D * POM_V4_D;
    let n_tiles = (blob.len() / POM_V4_CHUNK_BYTES) as u64 / POM_V4_TILE_CHUNKS;
    assert!(n_tiles > 0, "blob too small for one v4 tile");
    let mut states = Vec::with_capacity((POM_V4_K + 1) * d2);
    states.extend_from_slice(&v4_initial_state(seed));
    let mut snippets = Vec::with_capacity(POM_V4_K * POM_V4_SNIPPET_BYTES);
    let mut offsets = Vec::with_capacity(POM_V4_K);
    let mut off = v4_first_offset(seed, n_tiles);
    for step in 1..=POM_V4_K as u32 {
        let tile = &blob[(off as usize) * POM_V4_TILE_BYTES..(off as usize + 1) * POM_V4_TILE_BYTES];
        let snippet: [u8; POM_V4_SNIPPET_BYTES] = tile[..POM_V4_SNIPPET_BYTES].try_into().unwrap();
        offsets.push(off);
        snippets.extend_from_slice(&snippet);
        let prev = states[(step as usize - 1) * d2..(step as usize) * d2].to_vec();
        // Scalar on purpose: the reference stays independent of the SIMD paths it checks.
        let mut next = vec![0u8; d2];
        v4_transition_into_scalar(&mut next, &prev, tile, step);
        states.extend_from_slice(&next);
        if (step as usize) < POM_V4_K {
            off = v4_next_offset(seed, step as u64, &snippet, n_tiles);
        }
    }
    (states, snippets, offsets)
}

/// Test-only blob sized for the v4 walk. Same generator as the v3 lockstep blob so the two
/// harnesses stay comparable; only the tile geometry differs.
#[cfg(test)]
pub(crate) fn lockstep_blob() -> Vec<u8> {
    let n_bytes = 64 * POM_V4_TILE_BYTES + 5 * POM_V4_CHUNK_BYTES;
    let mut blob = vec![0u8; n_bytes];
    let mut h = 0xDEADBEEFu64;
    for b in blob.iter_mut() {
        h = mix64(h);
        *b = h as u8;
    }
    blob
}

#[cfg(test)]
mod simd_tests {
    use super::*;

    fn fill(seed: u64, buf: &mut [u8]) {
        let mut h = seed;
        for b in buf.iter_mut() {
            h = mix64(h);
            *b = h as u8;
        }
    }

    /// Every dispatched path must equal the scalar oracle, including the i8 extremes where a
    /// widening or saturation mistake would show.
    #[test]
    fn simd_transition_matches_scalar() {
        let d2 = POM_V4_D * POM_V4_D;
        let mut cases: Vec<(Vec<u8>, Vec<u8>)> = vec![
            (vec![0x80; d2], vec![0x80; POM_V4_TILE_BYTES]),
            (vec![0x7F; d2], vec![0x80; POM_V4_TILE_BYTES]),
            (vec![0xFF; d2], vec![0x01; POM_V4_TILE_BYTES]),
        ];
        for s in 0..16u64 {
            let mut state = vec![0u8; d2];
            let mut tile = vec![0u8; POM_V4_TILE_BYTES];
            fill(s, &mut state);
            fill(s ^ 0xA5A5, &mut tile);
            cases.push((state, tile));
        }
        for (i, (state, tile)) in cases.iter().enumerate() {
            let step = 1 + i as u32 * 17;
            let mut want = vec![0u8; d2];
            v4_transition_into_scalar(&mut want, state, tile, step);
            let mut got = vec![0u8; d2];
            v4_transition_into(&mut got, state, tile, step);
            assert_eq!(got, want, "dispatched transition differs from scalar (case {i})");
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe { v4_transition_into_avx2(&mut got, state, tile, step) };
                    assert_eq!(got, want, "avx2 differs from scalar (case {i})");
                }
                if is_x86_feature_detected!("sse4.1") {
                    unsafe { v4_transition_into_sse41(&mut got, state, tile, step) };
                    assert_eq!(got, want, "sse4.1 differs from scalar (case {i})");
                }
            }
        }
    }
}
