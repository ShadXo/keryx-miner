#!/bin/sh
# Which PoM tiers fit a GPU — answered before starting the miner.
#
# PoM cannot use llama's CPU-offload escape hatch: the walk needs every chunk in DEVICE memory,
# so any layer llama leaves on the host gets uploaded again by the miner. "Fits with offload" is
# enough to serve inference and NOT enough to mine. That makes the fit a hard yes/no worth
# checking before a rig downloads 9 GB it cannot use.
#
#   ./tools/pom-fit.sh                  # every GPU nvidia-smi reports
#   ./tools/pom-fit.sh 12288            # a card you do not have in front of you (MiB)
#   ./tools/pom-fit.sh --gguf m.gguf    # size a specific model file instead of the table
#
# Weights and KV/workspace come from the per-model comments in src/models.rs, converted from the
# GB (10^9) those use to the GiB (2^30) that nvidia-smi and every card spec actually mean. Mixing
# the two costs ~7%, which is most of the margin on a tight fit.
#
# Assumes the default 4096 context. KERYX_LLAMA_CTX scales the KV part of the overhead roughly
# linearly -- but lowering it is not a way to make a model fit: an over-length request then fails
# outright rather than degrading (llama_decode fails, the wrapper returns 0, generate() -> None,
# and the OPoI request goes unanswered), which is what service-bond strikes punish.

set -eu

# tier | flag | model | weights GiB | KV+workspace GiB
TIERS='0|--very-light|Qwen3.5-9B-abliterated|6.05|1.21
1|--light|GLM-4-9B-0414|7.73|1.40
2|(default)|Gemma-4-12B-abliterated|9.11|1.86
3|--high|Qwen3.6-27B|15.37|2.33
4|--very-high|Kimi-Linear-48B|27.66|1.50'

# Reserved before your first allocation: CUDA context, plus more with a display attached.
RESERVE_GIB=0.7
# Demanded on top of a bare fit. Not superstition: a 12 GB card computes to 11.0 needed vs 11.3
# usable for tier 2 and still OOMs, so a fit inside this band is a no.
MARGIN_GIB=0.5

if [ "${1:-}" = "--gguf" ]; then
    [ -n "${2:-}" ] || { echo "usage: $0 --gguf <file.gguf>" >&2; exit 2; }
    [ -f "$2" ] || { echo "no such file: $2" >&2; exit 2; }
    bytes=$(wc -c < "$2")
    echo "$bytes" | awk -v f="$2" '{
        w = $1 / 1073741824
        printf "%s\n  weights %.2f GiB  +20%% KV/workspace  ->  needs %.1f GiB\n", f, w, w * 1.20
    }'
    exit 0
fi

if [ $# -ge 1 ]; then
    gpus="0,manual,$1"
else
    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "nvidia-smi not found — pass VRAM in MiB instead: $0 12288" >&2
        exit 2
    }
    gpus=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | tr -d ' ' | sed 's/,/,/g')
fi

echo "$gpus" | while IFS=, read -r idx name mib; do
    [ -n "${mib:-}" ] || continue
    echo "$TIERS" | awk -F'|' \
        -v idx="$idx" -v name="$name" -v mib="$mib" \
        -v reserve="$RESERVE_GIB" -v margin="$MARGIN_GIB" '
        BEGIN {
            total  = mib / 1024
            usable = total - reserve
            printf "\nGPU%s  %s  —  %.1f GiB total, ~%.1f usable\n", idx, name, total, usable
            printf "  %-4s %-13s %-24s %8s  %s\n", "tier", "flag", "model", "needs", "verdict"
        }
        {
            need = $4 + $5
            if (need <= usable - margin)      v = sprintf("fits (%.1f spare)", usable - need)
            else if (need <= usable)          v = "TOO TIGHT — treat as no"
            else                              v = "no"
            printf "  %-4s %-13s %-24s %6.1f GiB  %s\n", $1, $2, $3, need, v
        }'
done
