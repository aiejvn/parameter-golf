"""
Grid sweep runner for sweep_gpt_v2.py.

Stage 1 (18 runs, ~3h): VOCAB_SIZE × WEIGHT_DECAY × MLP_MULT, Medusa fixed at 0.04
Stage 2 (3 runs, ~30min): Medusa weight bracket at the Stage 1 winner

Usage:
    python run_sweep.py            # run Stage 1 then Stage 2
    python run_sweep.py --stage 1  # Stage 1 only
    python run_sweep.py --stage 2 --best VOCAB=4096,WD=0.105,MLP=3  # Stage 2 at specific config
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Tokenizer/dataset paths for each vocab size
# ---------------------------------------------------------------------------
VOCAB_CONFIG: dict[int, tuple[str, str]] = {
    1024: (
        "./data/tokenizers/fineweb_1024_bpe.model",
        "./data/datasets/fineweb10B_sp1024",
    ),
    4096: (
        "./data/tokenizers/fineweb_4096_bpe.model",
        "./data/datasets/fineweb10B_sp4096",
    ),
    8192: (
        "./data/tokenizers/fineweb_8192_bpe.model",
        "./data/datasets/fineweb10B_sp8192",
    ),
}

# ---------------------------------------------------------------------------
# Stage 1 grid
# ---------------------------------------------------------------------------
STAGE1_GRID: dict[str, list] = {
    "VOCAB_SIZE":   [1024, 4096, 8192],
    "WEIGHT_DECAY": [0.07, 0.105, 0.14],
    "MLP_MULT":     [2, 3],
}

STAGE1_FIXED: dict[str, str] = {
    "MAX_WALLCLOCK_SECONDS": "600",
    "MEDUSA_WEIGHT":         "0.04",
    "MEDUSA_K":              "2",
    "SEED":                  "1337",
}

# ---------------------------------------------------------------------------
# Stage 2 Medusa bracket
# ---------------------------------------------------------------------------
STAGE2_MEDUSA_WEIGHTS = [0.03, 0.04, 0.05]


def check_prereqs(vocab_sizes: list[int]) -> list[str]:
    """Return list of missing prerequisite paths."""
    missing = []
    for v in vocab_sizes:
        tok_path, data_path = VOCAB_CONFIG[v]
        if not Path(tok_path).is_file():
            missing.append(f"Tokenizer missing: {tok_path}")
        val_glob = list(Path(data_path).glob("fineweb_val_*.bin")) if Path(data_path).is_dir() else []
        if not val_glob:
            missing.append(f"Val shards missing: {data_path}/fineweb_val_*.bin")
    return missing


def make_run_id(params: dict[str, str]) -> str:
    parts = [f"{k}={v}" for k, v in sorted(params.items())]
    return "_".join(parts)


def parse_final_bpb(logfile: str) -> float | None:
    """Return the last val_bpb value logged in the file."""
    path = Path(logfile)
    if not path.is_file():
        return None
    bpb = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"val_bpb:([\d.]+)", line)
        if m:
            bpb = float(m.group(1))
    return bpb


def run_one(params: dict[str, str], fixed: dict[str, str], run_id: str) -> float | None:
    vocab_size = int(params["VOCAB_SIZE"])
    tok_path, data_path = VOCAB_CONFIG[vocab_size]

    env = {
        **os.environ,
        **fixed,
        **{k: str(v) for k, v in params.items()},
        "TOKENIZER_PATH": tok_path,
        "DATA_PATH":      data_path,
        "RUN_ID":         run_id,
    }

    logfile = f"logs/{run_id}.txt"
    print(f"\n{'='*70}")
    print(f"Run: {run_id}")
    print(f"{'='*70}")

    proc = subprocess.run(
        [sys.executable, "sweep_gpt_v2.py"],
        env=env,
    )
    if proc.returncode != 0:
        print(f"[WARN] Run exited with code {proc.returncode}: {run_id}")

    bpb = parse_final_bpb(logfile)
    print(f"Result: val_bpb={bpb}")
    return bpb


def run_stage1(skip_vocab: list[int] | None = None) -> list[dict]:
    vocab_list = [v for v in STAGE1_GRID["VOCAB_SIZE"] if not skip_vocab or v not in skip_vocab]

    # Check prereqs and skip missing vocab sizes
    missing = check_prereqs(vocab_list)
    if missing:
        print("WARNING: Some vocab sizes are missing prerequisites and will be skipped:")
        for m in missing:
            print(f"  {m}")
        # Determine which vocab sizes have all files
        available = []
        for v in vocab_list:
            tok_path, data_path = VOCAB_CONFIG[v]
            val_glob = list(Path(data_path).glob("fineweb_val_*.bin")) if Path(data_path).is_dir() else []
            if Path(tok_path).is_file() and val_glob:
                available.append(v)
        print(f"Running with available vocab sizes: {available}")
        vocab_list = available

    if not vocab_list:
        print("ERROR: No vocab sizes available. Run Phase 0 data download first.")
        sys.exit(1)

    grid = {
        "VOCAB_SIZE":   vocab_list,
        "WEIGHT_DECAY": STAGE1_GRID["WEIGHT_DECAY"],
        "MLP_MULT":     STAGE1_GRID["MLP_MULT"],
    }

    combos = list(itertools.product(*grid.values()))
    keys = list(grid.keys())
    total = len(combos)
    print(f"\nStage 1: {total} runs")

    results = []
    for i, combo in enumerate(combos, 1):
        params = {k: str(v) for k, v in zip(keys, combo)}
        run_id = "s1_" + make_run_id(params)
        print(f"\n[{i}/{total}]", end="")
        bpb = run_one(params, STAGE1_FIXED, run_id)
        results.append({**{k: v for k, v in zip(keys, combo)}, "val_bpb": bpb, "run_id": run_id})

    return results


def run_stage2(best: dict) -> list[dict]:
    print(f"\nStage 2: Medusa bracket at best config {best}")
    results = []
    for mw in STAGE2_MEDUSA_WEIGHTS:
        params = {
            "VOCAB_SIZE":   str(best["VOCAB_SIZE"]),
            "WEIGHT_DECAY": str(best["WEIGHT_DECAY"]),
            "MLP_MULT":     str(best["MLP_MULT"]),
        }
        fixed = {**STAGE1_FIXED, "MEDUSA_WEIGHT": str(mw)}
        run_id = f"s2_medusa={mw}_" + make_run_id(params)
        bpb = run_one(params, fixed, run_id)
        results.append({**best, "MEDUSA_WEIGHT": mw, "val_bpb": bpb, "run_id": run_id})
    return results


def print_leaderboard(results: list[dict], title: str = "Results") -> None:
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    sorted_results = sorted(
        [r for r in results if r.get("val_bpb") is not None],
        key=lambda r: r["val_bpb"],
    )
    for r in sorted_results:
        kv = "  ".join(f"{k}={v}" for k, v in r.items() if k not in ("run_id", "val_bpb"))
        print(f"  val_bpb={r['val_bpb']:.4f}  {kv}  [{r.get('run_id','')}]")
    failed = [r for r in results if r.get("val_bpb") is None]
    if failed:
        print(f"\n  Failed runs ({len(failed)}):")
        for r in failed:
            print(f"    {r.get('run_id','?')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid sweep runner for sweep_gpt_v2.py")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="Run only Stage 1 or Stage 2 (default: run both)")
    parser.add_argument("--best", type=str, default=None,
                        help="Override best config for Stage 2, e.g. VOCAB=4096,WD=0.105,MLP=3")
    parser.add_argument("--skip-vocab", type=int, nargs="+", default=None,
                        help="Skip specific vocab sizes (e.g. --skip-vocab 8192)")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)

    stage1_results: list[dict] = []
    best_config: dict = {}

    if args.stage in (None, 1):
        stage1_results = run_stage1(skip_vocab=args.skip_vocab)
        print_leaderboard(stage1_results, "Stage 1 Leaderboard")

        # Save Stage 1 results
        with open("sweep_stage1_results.json", "w") as f:
            json.dump(stage1_results, f, indent=2)
        print("\nStage 1 results saved to sweep_stage1_results.json")

        # Pick best config for Stage 2
        valid = [r for r in stage1_results if r.get("val_bpb") is not None]
        if valid:
            best = min(valid, key=lambda r: r["val_bpb"])
            best_config = {
                "VOCAB_SIZE":   best["VOCAB_SIZE"],
                "WEIGHT_DECAY": best["WEIGHT_DECAY"],
                "MLP_MULT":     best["MLP_MULT"],
            }
            print(f"\nBest Stage 1 config: {best_config}  val_bpb={best['val_bpb']:.4f}")

    if args.stage in (None, 2):
        # Allow manual override of best config
        if args.best:
            for token in args.best.split(","):
                k, v = token.split("=")
                key_map = {"VOCAB": "VOCAB_SIZE", "WD": "WEIGHT_DECAY", "MLP": "MLP_MULT"}
                best_config[key_map.get(k, k)] = int(v) if v.isdigit() else float(v)
        if not best_config:
            print("ERROR: No best config available for Stage 2. Run Stage 1 first or provide --best.")
            sys.exit(1)

        stage2_results = run_stage2(best_config)
        print_leaderboard(stage2_results, "Stage 2 Medusa Bracket Leaderboard")

        with open("sweep_stage2_results.json", "w") as f:
            json.dump(stage2_results, f, indent=2)
        print("Stage 2 results saved to sweep_stage2_results.json")

        all_results = stage1_results + stage2_results
        print_leaderboard(all_results, "Overall Leaderboard")


if __name__ == "__main__":
    main()
