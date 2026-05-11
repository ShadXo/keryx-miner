# Keryx-Miner v0.2.3 — OPoI Phase 2 & Next-Gen GPU Support

## OPoI Phase 2: On-Chain Inference is Live

This release delivers the full OPoI Phase 2 pipeline — miners now perform real AI inference on-chain and earn escrow rewards automatically.

- **Real AI inference** — The miner runs a fixed-point Keryx-inference model (bit-exact, deterministic) and embeds the commitment directly in the coinbase tag. No more synthetic MLP — this is production-grade verifiable inference.
- **AiResponse as standalone transaction** — AI responses are now submitted as proper on-chain transactions with subnetwork filtering, making them queryable and auditable.
- **Automated escrow claim** — Just pass `--escrow-privkey` and the miner handles everything: tracking mature escrow outputs, building Schnorr-signed claim TXs, and broadcasting them via gRPC. No manual claiming needed.
- **OPoI enabled by default** — All miners contribute to the AI network from block 1. Use `--no-opoi` to opt out.
- **AI request detection** — The miner detects `KRX:AI:1:` requests in the mempool and embeds AI responses directly in the coinbase.

## GPU Support: RTX 40/50 & RDNA 3/4

Keryx-Miner now ships native CUDA PTX for the latest NVIDIA and AMD architectures:

- **RTX 40 series (Ada Lovelace, sm_89)** — dedicated PTX binary, no more JIT fallback
- **RTX 50 series (Blackwell, sm_100)** — dedicated PTX binary for next-gen GPUs
- **AMD RDNA 3 (RX 7000) & RDNA 4 (RX 9000)** — `v_dot4`/`v_dot8` optimized kernel path now active, significant performance uplift over the previous scalar fallback
- **nvml-wrapper updated to 0.10** — proper NVML support for modern GPUs (overclock/power-limit features)

## Escrow: Smarter Claim Management

The escrow claim system has been hardened against BlockDAG reorgs:

- **Orphan rejection handling** — Claims rejected because the coinbase block is off the selected chain (orphan) are no longer permanently slashed. They're retried with a 100-block cooldown.
- **Max 10 retries** — After 10 consecutive orphan rejections, the entry is slashed permanently to prevent infinite loops.
- **Sequence-lock retries** — Timing-race rejections are silently retried on the next block.
- **Clean logs** — Orphan and retry messages moved to debug level. Only real issues surface as warnings.

## Fee Fix

- **CLAIM_FEE_SOMPI** updated to 0.3 KRX minimum, matching the network's required minimum fee.

---

**Full Changelog**: v0.2.2...v0.2.3