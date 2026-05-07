/// OPoI inference module — Phase 2.
///
/// Detects AiRequest TXs in block templates and embeds the OPoI tag in
/// the coinbase extra_data.  Tag computation delegates to `keryx-inference`
/// which uses a fixed-point i32/i64 MLP — bit-exact on all hardware.

const KRX_AI_PREFIX: &[u8] = b"KRX:AI:1:";

/// Compute the Phase-2 OPoI tag for a coinbase.
///
/// Uses the fixed-point MLP from `keryx-inference` — bit-exact on all hardware.
/// The node's `validate_opoi_tag` verifies against this same model, so miners
/// and consensus always agree regardless of CPU/GPU architecture or SIMD support.
pub fn compute_opoi_tag(nonce_hex: &str) -> String {
    let nonce = u64::from_str_radix(nonce_hex, 16).unwrap_or(0);
    keryx_inference::tag_fixed(nonce)
}

// ---------------------------------------------------------------------------
// AiRequest TX scanning helpers.
// ---------------------------------------------------------------------------

/// Extract an AI inference request from a hex-encoded transaction payload.
///
/// Returns `(payload_prefix_16, prompt, max_tokens)` if the payload starts
/// with the KRX:AI:1: marker and contains a valid JSON body.
/// `payload_prefix_16` is the first 8 bytes of blake2b-256(payload_bytes) as
/// 16 lowercase hex chars — a deterministic identifier that does not require
/// the transaction ID (unavailable in GetBlockTemplateResponse).
pub fn extract_ai_request(payload_hex: &str) -> Option<(String, String, usize)> {
    if payload_hex.is_empty() {
        return None;
    }
    let bytes = hex::decode(payload_hex).ok()?;
    if bytes.len() <= KRX_AI_PREFIX.len() {
        return None;
    }
    if !bytes.starts_with(KRX_AI_PREFIX) {
        return None;
    }
    let json_str = std::str::from_utf8(&bytes[KRX_AI_PREFIX.len()..]).ok()?;
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let prompt = v["p"].as_str()?.to_string();
    let max_tokens = v["n"].as_u64().unwrap_or(128) as usize;
    // Compute stable prefix from blake2b(payload_bytes) — matches the indexer's payload_prefix.
    let hash = blake2b_simd::blake2b(&bytes);
    let prefix = hex::encode(&hash.as_bytes()[..8]);
    Some((prefix, prompt, max_tokens))
}

/// Run synthetic Phase-1 inference on a prompt.
///
/// Computes blake2b(prompt) and returns the first 8 bytes as 16 lowercase hex
/// chars.  Deterministic and cheap — proves the miner processed the prompt
/// without requiring a real neural network in Phase 1.
pub fn run_synthetic_inference(prompt: &str) -> String {
    use blake2b_simd::blake2b;
    let hash = blake2b(prompt.as_bytes());
    hex::encode(&hash.as_bytes()[..8])
}
