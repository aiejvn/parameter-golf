#!/usr/bin/env bash
set -euo pipefail

THREADS=4

echo "============================================================"
echo "  Parameter Golf — Prepare + Sweep"
echo "  CPU threads for tokenization: $THREADS / $(nproc)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Phase 0a: Download docs_selected.jsonl (needed to retokenize)
# ---------------------------------------------------------------------------
if [ ! -f "data/docs_selected.jsonl" ]; then
    echo ""
    echo "[Phase 0a] Downloading docs_selected.jsonl from HuggingFace..."
    python data/cached_challenge_fineweb.py --variant sp1024 --with-docs --train-shards 0
    echo "[Phase 0a] Done."
else
    echo ""
    echo "[Phase 0a] docs_selected.jsonl already present — skipping download."
fi

# ---------------------------------------------------------------------------
# Phase 0b: Build sp4096 + sp8192 tokenizers and dataset shards
# ---------------------------------------------------------------------------
SP4096_READY=0
SP8192_READY=0

if [ -f "data/tokenizers/fineweb_4096_bpe.model" ] && \
   ls data/datasets/fineweb10B_sp4096/fineweb_val_*.bin &>/dev/null; then
    echo ""
    echo "[Phase 0b] sp4096 already built — skipping."
    SP4096_READY=1
fi

if [ -f "data/tokenizers/fineweb_8192_bpe.model" ] && \
   ls data/datasets/fineweb10B_sp8192/fineweb_val_*.bin &>/dev/null; then
    echo ""
    echo "[Phase 0b] sp8192 already built — skipping."
    SP8192_READY=1
fi

if [ "$SP4096_READY" -eq 0 ] || [ "$SP8192_READY" -eq 0 ]; then
    echo ""
    echo "[Phase 0b] Building tokenizer(s) and dataset shards..."
    echo "           Threads: $THREADS"
    echo "           This will take several hours."
    echo ""

    MATCHED_FINEWEB_TOKENIZER_THREADS=$THREADS \
    MATCHED_FINEWEB_SP_BATCH_SIZE=1024 \
    python data/download_hf_docs_and_tokenize.py \
        --output-root ./data \
        --tokenizer-config data/tokenizer_specs_4096_8192.json \
        --reuse-sp-model "8192=records/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/fineweb_8192_bpe.model" \
        --skip-byte

    echo "[Phase 0b] Done."
fi

# ---------------------------------------------------------------------------
# Phase 1: Run the grid sweep
# ---------------------------------------------------------------------------
echo ""
echo "[Phase 1] Starting grid sweep..."
echo "          Stage 1: 18 runs (VOCAB x WD x MLP_MULT), Medusa=0.04"
echo "          Stage 2: 3 Medusa bracket runs at best config"
echo ""

python run_sweep.py

echo ""
echo "============================================================"
echo "  Sweep complete. Results:"
echo "    sweep_stage1_results.json"
echo "    sweep_stage2_results.json"
echo "============================================================"
