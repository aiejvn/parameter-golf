"""
Post-training visualizer for the MDLM trainer.

Called from train_mdlm.py main() with in-memory data:
    post_training_visualizer(loss_log, results, model, val_np, sp)

Writes to ./viz/ (or out_dir):
  loss_curves.png       — train + val loss graph
  reconstructions.csv   — example_id, original, masked_input, reconstruction

Requirements: matplotlib, sentencepiece (already used by trainer)
"""

import csv
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── constants imported from trainer ─────────────────────────────────────────
from train_mdlm import SEQ_LEN, PAD_ID, MASK_ID, EOS_ID

MASK_FRAC = 0.40   # fraction of content tokens to mask for the reconstruction demo


# ── decode helpers ────────────────────────────────────────────────────────────
def decode_tokens(token_ids, sp):
    """Decode, skipping PAD and MASK."""
    ids = [int(t) for t in token_ids
           if int(t) < sp.vocab_size() and int(t) not in (PAD_ID, MASK_ID)]
    return sp.decode(ids)


def decode_masked(token_ids, sp):
    """Decode, replacing MASK_ID positions with '▮' inline."""
    parts = []
    run   = []
    for t in token_ids:
        t = int(t)
        if t == PAD_ID:
            break
        if t == MASK_ID:
            if run:
                parts.append(sp.decode(run))
                run = []
            parts.append("▮")
        elif t < sp.vocab_size():
            run.append(t)
    if run:
        parts.append(sp.decode(run))
    return "".join(parts)


# ── greedy reconstruction ─────────────────────────────────────────────────────
@torch.no_grad()
def reconstruct(model, x0, device, mask_frac=MASK_FRAC, sigma_val=0.5):
    """
    Randomly mask mask_frac of content tokens, run one forward pass,
    fill masked positions with argmax prediction.

    Returns (xt_np, recon_np) as 1-D int64 arrays.
    """
    model.eval()
    x0t        = torch.as_tensor(x0, dtype=torch.long).unsqueeze(0).to(device)
    is_special = (x0t == EOS_ID) | (x0t == PAD_ID)
    content_pos = (~is_special[0]).nonzero(as_tuple=True)[0]
    n_mask      = max(1, int(len(content_pos) * mask_frac))
    # random subset — avoids masking only the start of every sequence
    perm        = torch.randperm(len(content_pos), device=device)
    mask_pos    = content_pos[perm[:n_mask]]

    xt          = x0t.clone()
    xt[0, mask_pos] = MASK_ID

    sigma = torch.tensor([sigma_val], device=device, dtype=torch.float32)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        log_probs = model.subs_log_probs(xt, sigma)   # [1, T, V]

    pred             = log_probs[0].float().argmax(-1)
    recon            = xt[0].clone()
    recon[mask_pos]  = pred[mask_pos]

    return xt[0].cpu().numpy(), recon.cpu().numpy()


# ── loss curve graph ─────────────────────────────────────────────────────────
def plot_loss_curves(loss_log, results, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    train_steps  = np.array(loss_log["train_steps"])
    train_losses = np.array(loss_log["train_losses"])
    val_steps    = np.array(loss_log["val_steps"])
    val_losses   = np.array(loss_log["val_losses"])

    kernel = np.ones(50) / 50
    if len(train_losses) >= 50:
        smoothed     = np.convolve(train_losses, kernel, mode="valid")
        smooth_steps = train_steps[49:]
    else:
        smoothed, smooth_steps = train_losses, train_steps

    ax.plot(train_steps, train_losses, color="#aad4f5", alpha=0.35,
            linewidth=0.7, label="train (raw)")
    ax.plot(smooth_steps, smoothed,    color="#1f77b4", linewidth=1.8,
            label="train (50-step avg)")
    if len(val_steps):
        ax.plot(val_steps, val_losses, color="#ff7f0e", linewidth=2.0,
                marker="o", markersize=4, label="val loss")

    ax.set_xlabel("step"); ax.set_ylabel("MDLM loss")
    title = f"LLaDA v5 — {results.get('params', 0)//1_000_000:.0f}M params"
    if "val_bpb" in results:   title += f"  |  val_bpb={results['val_bpb']:.4f}"
    if "train_min" in results: title += f"  |  {results['train_min']:.1f} min"
    ax.set_title(title); ax.legend(framealpha=0.7); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def write_reconstructions_csv(model, seqs, sp, device, out_path):
    """
    Columns: example_id, original, masked_input, reconstruction
    """
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["example_id", "original", "masked_input", "reconstruction"])
        for i, x0 in enumerate(seqs):
            xt_np, recon_np = reconstruct(model, x0, device)
            orig  = decode_tokens(x0,      sp).replace("\n", " ")
            masked = decode_masked(xt_np,  sp).replace("\n", " ")
            recon  = decode_tokens(recon_np, sp).replace("\n", " ")
            w.writerow([i + 1, orig, masked, recon])
            if (i + 1) % 10 == 0:
                print(f"  reconstructions {i+1}/{len(seqs)}", flush=True)
    print(f"  Saved {out_path}")


# ── main entry point ──────────────────────────────────────────────────────────
def post_training_visualizer(loss_log, results, model, val_np, sp,
                              out_dir=None, n_examples=30, device="cuda"):
    """
    loss_log : dict  — train_steps, train_losses, val_steps, val_losses
    results  : dict  — val_bpb, train_loss, train_min, params, …
    model    : trained DiffusionLM (on device)
    val_np   : int64 ndarray of validation tokens
    sp       : SentencePieceProcessor
    """
    if out_dir is None:
        from datetime import datetime
        out_dir = os.path.join("./viz", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nVisualizer: writing outputs to {out_dir}", flush=True)

    # Sample n_examples non-overlapping windows from val_np
    n_avail    = max(1, (len(val_np) - 1) // SEQ_LEN)
    n_examples = min(n_examples, n_avail)
    np.random.seed(0)
    indices = np.sort(np.random.choice(n_avail, size=n_examples, replace=False))
    seqs    = np.stack([val_np[i*SEQ_LEN:(i+1)*SEQ_LEN].astype(np.int64) for i in indices])

    model.eval()
    plot_loss_curves(loss_log, results,
                     os.path.join(out_dir, "loss_curves.png"))
    write_reconstructions_csv(model, seqs, sp, device,
                              os.path.join(out_dir, "reconstructions.csv"))
    print("  Visualizer done.", flush=True)


# ── standalone usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, sys
    import sentencepiece as spm_lib

    loss_log_path = sys.argv[1] if len(sys.argv) > 1 else "./v5_loss_log.json"
    results_path  = sys.argv[2] if len(sys.argv) > 2 else "./v5_results.json"

    loss_log = json.load(open(loss_log_path))
    results  = json.load(open(results_path))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_mdlm import DiffusionLM, load_tokens, TOKENIZER_PATH, DEVICE  # noqa

    sp    = spm_lib.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    model = DiffusionLM().to(DEVICE).to(torch.bfloat16)
    model.load_state_dict(torch.load(os.path.expanduser("~/llada_v5.pt"), map_location=DEVICE))

    val_np = load_tokens("val")
    post_training_visualizer(loss_log, results, model, val_np, sp, device=DEVICE)
