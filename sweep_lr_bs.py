"""
Sweep learning rate, batch size, and optimizer (AdamW vs Muon) for the MDLM.
One shard, no rotation. Results written to sweep_lr_bs_results.csv.

Muon split (mirrors train_gpt.py):
  blocks.* 2-D weight matrices  → Muon  at lr
  embeddings, head, adaln, etc. → AdamW at lr * SCALAR_LR_RATIO
"""
import csv, glob, os, math, time, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_gpt import Muon

# ── fixed hyperparams ─────────────────────────────────────────────────────────
VOCAB_SIZE   = 1024
MASK_ID      = 1024
EOS_ID       = 1
PAD_ID       = 1025
TOTAL_VOCAB  = 1026
PADDED_VOCAB = 1088
DEVICE       = "cuda"; SEED = 42

NUM_LAYERS = 11; MODEL_DIM = 512; NUM_HEADS = 8; MLP_MULT = 3.0
SEQ_LEN    = 2048; GRAD_ACCUM = 4

TRAIN_STEPS    = 401
WARMUP_STEPS   = 300
WARMDOWN_STEPS = 1500

NOISE_EPS      = 1e-3
VAR_EVAL_STEPS = 32

DATA_DIR    = "data/datasets/fineweb10B_sp1024"
RESULTS_CSV = "sweep_lr_bs_results.csv"

SCALAR_LR_RATIO  = 0.1   # Adam lr for non-matrix params when using Muon
MUON_MOMENTUM    = 0.95
MUON_BACKEND_STEPS = 5

# ── sweep grid ────────────────────────────────────────────────────────────────
SWEEP_LR         = [1e-4, 3e-4, 6e-4, 1e-3, 3e-3]
SWEEP_BATCH_SIZE = [4, 8, 16, 32, 128, 256, 512]
SWEEP_OPTIMIZER  = ["adamw", "muon"]

torch.manual_seed(SEED); np.random.seed(SEED)
NEG_INF = -1e6


# ── noise schedule ────────────────────────────────────────────────────────────
def log_linear_noise(t, eps=NOISE_EPS):
    alpha = 1 - (1 - eps) * t
    sigma = -torch.log(alpha.clamp(min=1e-8))
    return sigma, alpha


# ── model ─────────────────────────────────────────────────────────────────────
def rms_norm(x): return F.rms_norm(x, (x.size(-1),))

def apply_rotary(x, cos, sin):
    d = x.shape[3] // 2; x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1*cos+x2*sin, x1*(-sin)+x2*cos], dim=3)

class TimestepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))
        half = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000)*torch.arange(half, dtype=torch.float32)/half))
    def forward(self, sigma):
        emb = sigma[:, None] * self.freqs[None, :]
        return self.mlp(torch.cat([emb.sin(), emb.cos()], dim=-1))

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads, self.hd = n_heads, dim // n_heads
        self.c_q    = nn.Linear(dim, dim, bias=False)
        self.c_k    = nn.Linear(dim, dim, bias=False)
        self.c_v    = nn.Linear(dim, dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.c_q(x).view(B, T, self.n_heads, self.hd)
        k = self.c_k(x).view(B, T, self.n_heads, self.hd)
        v = self.c_v(x).view(B, T, self.n_heads, self.hd)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        q, k = rms_norm(q), rms_norm(k)
        y = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=False)
        return self.c_proj(y.transpose(1,2).contiguous().view(B, T, -1))

class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim=128):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2*dim, bias=True)
    def forward(self, x, c):
        s, sh = self.proj(c).unsqueeze(1).chunk(2, dim=-1)
        return rms_norm(x) * (1+s) + sh

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_mult, cond_dim=128):
        super().__init__()
        self.attn       = Attention(dim, n_heads)
        self.adaln_attn = AdaLN(dim, cond_dim)
        self.adaln_mlp  = AdaLN(dim, cond_dim)
        hidden = int(dim * mlp_mult)
        self.mlp_fc   = nn.Linear(dim, hidden, bias=False)
        self.mlp_proj = nn.Linear(hidden, dim, bias=False)
    def forward(self, x, cos, sin, c):
        x = x + self.attn(self.adaln_attn(x, c), cos, sin)
        x = x + self.mlp_proj(F.relu(self.mlp_fc(self.adaln_mlp(x, c))).square())
        return x

class DiffusionLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte       = nn.Embedding(PADDED_VOCAB, MODEL_DIM)
        self.sigma_map = TimestepEmbedder(128)
        self.blocks    = nn.ModuleList([Block(MODEL_DIM, NUM_HEADS, MLP_MULT) for _ in range(NUM_LAYERS)])
        self.head      = nn.Linear(MODEL_DIM, PADDED_VOCAB, bias=False)
        hd = MODEL_DIM // NUM_HEADS
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        freqs = torch.outer(torch.arange(SEQ_LEN*2, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", freqs.cos()[None, :, None, :])
        self.register_buffer("sin", freqs.sin()[None, :, None, :])

    def forward_logits(self, xt, sigma):
        B, T = xt.shape
        x = self.wte(xt)
        c = F.silu(self.sigma_map(sigma)).to(dtype=x.dtype)
        cos, sin = self.cos[:, :T], self.sin[:, :T]
        for b in self.blocks:
            x = b(x, cos, sin, c)
        return self.head(rms_norm(x))[..., :TOTAL_VOCAB].float()

    def subs_log_probs(self, xt, sigma):
        logits = self.forward_logits(xt, sigma)
        logits[:, :, MASK_ID] = NEG_INF
        logits[:, :, PAD_ID]  = NEG_INF
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        frozen = torch.full_like(logits, NEG_INF)
        frozen.scatter_(-1, xt[..., None], 0.0)
        return torch.where((xt != MASK_ID)[..., None], frozen, logits)


# ── loss ──────────────────────────────────────────────────────────────────────
def mdlm_loss(model, x0):
    B = x0.shape[0]
    t = torch.rand(B // 2 + 1, device=x0.device)
    t = torch.cat([t, 1 - t])[:B].clamp(1e-5, 1 - 1e-5)
    sigma, alpha = log_linear_noise(t)
    is_special   = (x0 == EOS_ID) | (x0 == PAD_ID)
    move         = (torch.rand_like(x0.float()) < (1 - alpha)[:, None]) & ~is_special
    xt           = torch.where(move, MASK_ID, x0)
    log_probs    = model.subs_log_probs(xt, sigma)
    x0_safe      = x0.masked_fill(x0 == PAD_ID, 0)
    log_p_x0     = torch.gather(log_probs, -1, x0_safe[..., None]).squeeze(-1)
    dsigma       = (1 - NOISE_EPS) / alpha
    is_masked    = (xt == MASK_ID).float()
    content_mask = (x0 != PAD_ID).float()
    n_content    = content_mask.sum().clamp(min=1)
    return (dsigma[:, None] * (-log_p_x0) * is_masked * content_mask).sum() / n_content


# ── ELBO eval ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def variational_elbo_bits(model, x0, n_steps=VAR_EVAL_STEPS):
    B, _         = x0.shape
    total_bits   = torch.zeros(B, device=x0.device)
    is_special   = (x0 == EOS_ID) | (x0 == PAD_ID)
    content_mask = (x0 != PAD_ID).float()
    x0_safe      = x0.masked_fill(x0 == PAD_ID, 0)
    t_grid       = torch.arange(1, n_steps+1, device=x0.device, dtype=torch.float32) / n_steps
    sigma_grid, alpha_grid = log_linear_noise(t_grid)
    total_bits  += content_mask.sum(dim=-1) * float(alpha_grid[-1]) * math.log(VOCAB_SIZE) / math.log(2.0)
    alpha_prev   = 1.0
    for i in range(n_steps):
        alpha_curr  = alpha_grid[i]
        move        = (torch.rand_like(x0.float()) < (1 - alpha_curr)) & ~is_special
        xt          = torch.where(move, MASK_ID, x0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            log_probs = model.subs_log_probs(xt, sigma_grid[i].expand(B))
        log_p_x0    = torch.gather(log_probs.float(), -1, x0_safe[..., None]).squeeze(-1)
        reveal_prob = (alpha_prev - float(alpha_curr)) / max(1.0 - float(alpha_curr), 1e-12)
        total_bits += (reveal_prob * (-log_p_x0) * (xt == MASK_ID).float() * content_mask / math.log(2.0)).sum(dim=-1)
        alpha_prev  = float(alpha_curr)
    return total_bits


# ── data ──────────────────────────────────────────────────────────────────────
def _load_shard(path):
    with open(path, "rb") as f:
        f.read(256 * 4)
        return np.frombuffer(f.read(), dtype=np.uint16).astype(np.int64)

def load_one_shard(split):
    pattern = os.path.join(DATA_DIR, f"fineweb_{split}_*.bin")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shards: {pattern}")
    print(f"  Loading {split}: {os.path.basename(paths[0])}", flush=True)
    return _load_shard(paths[0])

def build_chunk_index(tokens_np):
    eos_pos = np.where(tokens_np == EOS_ID)[0]
    chunks  = []
    for k in range(len(eos_pos) - 1):
        start, end_eos = int(eos_pos[k]) + 1, int(eos_pos[k + 1])
        pos = start
        while pos < end_eos + 1:
            chunks.append((pos, min(pos + SEQ_LEN, end_eos + 1)))
            pos = min(pos + SEQ_LEN, end_eos + 1)
    return chunks

def sample_batch(tokens_np, chunks, batch_size):
    ki    = np.random.randint(0, len(chunks), size=batch_size)
    batch = np.full((batch_size, SEQ_LEN), PAD_ID, dtype=np.int64)
    for b, k in enumerate(ki):
        s, e = chunks[k]
        batch[b, :e - s] = tokens_np[s:e]
    return torch.from_numpy(batch).to(DEVICE)

def get_lr(step, lr):
    if step < WARMUP_STEPS:
        return lr * (step + 1) / WARMUP_STEPS
    elif step < TRAIN_STEPS - WARMDOWN_STEPS:
        return lr
    else:
        progress = (TRAIN_STEPS - step) / WARMDOWN_STEPS
        return lr * (0.1 + 0.9 * (0.5 * (1 + math.cos(math.pi * (1 - progress)))))


# ── optimizer builder ─────────────────────────────────────────────────────────
def build_optimizer(model, opt_name, lr):
    if opt_name == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                  weight_decay=0.1, fused=True)], None

    # Muon: 2-D block matrices → Muon; everything else → Adam
    matrix_params = [p for _, p in model.blocks.named_parameters() if p.ndim == 2]
    matrix_ids    = {id(p) for p in matrix_params}
    scalar_params = [p for p in model.parameters() if id(p) not in matrix_ids]

    opt_muon  = Muon(matrix_params, lr=lr, momentum=MUON_MOMENTUM,
                     backend_steps=MUON_BACKEND_STEPS)
    opt_adam  = torch.optim.AdamW(scalar_params, lr=lr * SCALAR_LR_RATIO,
                                  betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    return [opt_muon, opt_adam], lr * SCALAR_LR_RATIO

def set_lr(optimizers, opt_name, lr, scalar_lr):
    if opt_name == "adamw":
        for g in optimizers[0].param_groups: g["lr"] = lr
    else:
        for g in optimizers[0].param_groups: g["lr"] = lr          # Muon
        for g in optimizers[1].param_groups: g["lr"] = scalar_lr   # Adam


# ── single config training run ────────────────────────────────────────────────
def train_one_config(lr, batch_size, opt_name, train_np, train_chunks, val_np, val_chunks):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model    = DiffusionLM().to(DEVICE).to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    eff_batch = batch_size * GRAD_ACCUM
    print(f"\n{'='*60}", flush=True)
    print(f"  lr={lr:.0e}  batch={batch_size}  eff_batch={eff_batch}  opt={opt_name}  params={n_params:,}", flush=True)
    print(f"{'='*60}", flush=True)

    optimizers, _ = build_optimizer(model, opt_name, lr)
    t0 = time.time(); losses = []
    model.train()

    for step in range(TRAIN_STEPS):
        cur_lr = get_lr(step, lr)
        cur_scalar_lr = cur_lr * SCALAR_LR_RATIO if opt_name == "muon" else None
        set_lr(optimizers, opt_name, cur_lr, cur_scalar_lr)

        for opt in optimizers: opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(GRAD_ACCUM):
            batch = sample_batch(train_np, train_chunks, batch_size)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = mdlm_loss(model, batch) / GRAD_ACCUM
            loss.backward(); accum_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers: opt.step()
        losses.append(accum_loss)

        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                val_batch = sample_batch(val_np, val_chunks, batch_size * GRAD_ACCUM)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    val_loss = mdlm_loss(model, val_batch).item()
            model.train()
            avg     = float(np.mean(losses[-200:]))
            elapsed = time.time() - t0
            tok_s   = (step+1) * batch_size * GRAD_ACCUM * SEQ_LEN / elapsed
            print(f"  step {step:5d}/{TRAIN_STEPS} | loss={avg:.4f} | val={val_loss:.4f} | "
                  f"lr={cur_lr:.1e} | {tok_s/1e3:.0f}K tok/s | {elapsed:.0f}s", flush=True)

    final_train_loss = float(np.mean(losses[-100:]))

    # final val loss
    model.eval()
    with torch.no_grad():
        val_batch = sample_batch(val_np, val_chunks, batch_size * GRAD_ACCUM)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            final_val_loss = mdlm_loss(model, val_batch).item()

    # val bpb
    val_tokens = torch.from_numpy(val_np)
    total_bits = 0.0; total_content = 0
    n_seqs     = min(200, (len(val_np) - 1) // SEQ_LEN)
    with torch.no_grad():
        for i in range(n_seqs):
            x    = val_tokens[i*SEQ_LEN:(i+1)*SEQ_LEN].unsqueeze(0).to(DEVICE)
            total_bits    += variational_elbo_bits(model, x).sum().item()
            total_content += SEQ_LEN
    val_bpb = total_bits / max(total_content * 4.3, 1)

    train_time = time.time() - t0
    print(f"\n  DONE lr={lr:.0e} batch={batch_size} opt={opt_name} | "
          f"train={final_train_loss:.4f} val={final_val_loss:.4f} bpb={val_bpb:.4f} "
          f"| {train_time/60:.1f}min", flush=True)

    return {
        "lr":           lr,
        "batch_size":   batch_size,
        "eff_batch":    eff_batch,
        "optimizer":    opt_name,
        "params":       n_params,
        "train_loss":   round(final_train_loss, 6),
        "val_loss":     round(final_val_loss,   6),
        "val_bpb":      round(val_bpb,          6),
        "train_min":    round(train_time / 60,  2),
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    configs = list(itertools.product(SWEEP_LR, SWEEP_BATCH_SIZE, SWEEP_OPTIMIZER))
    print(f"MDLM lr × batch_size × optimizer sweep — {len(configs)} configs", flush=True)
    print(f"LR: {SWEEP_LR}", flush=True)
    print(f"Batch sizes: {SWEEP_BATCH_SIZE}", flush=True)
    print(f"Optimizers: {SWEEP_OPTIMIZER}\n", flush=True)

    train_np     = load_one_shard("train")
    val_np       = load_one_shard("val")
    train_chunks = build_chunk_index(train_np)
    val_chunks   = build_chunk_index(val_np)
    print(f"Train: {len(train_np):,} tokens, {len(train_chunks):,} chunks")
    print(f"Val:   {len(val_np):,} tokens\n")

    csv_fields = ["lr", "batch_size", "eff_batch", "optimizer", "params",
                  "train_loss", "val_loss", "val_bpb", "train_min"]
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    all_results = []
    for i, (lr, batch_size, opt_name) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] lr={lr:.0e}  batch={batch_size}  opt={opt_name}", flush=True)
        result = train_one_config(lr, batch_size, opt_name, train_np, train_chunks, val_np, val_chunks)
        all_results.append(result)
        with open(RESULTS_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(result)
        print(f"  → written to {RESULTS_CSV}", flush=True)

    # Summary table sorted by val_bpb
    all_results.sort(key=lambda r: r["val_bpb"])
    print(f"\n{'='*80}")
    print(f"  {'lr':>8}  {'bs':>4}  {'eff_bs':>6}  {'opt':>6}  {'train':>8}  {'val':>8}  {'bpb':>8}  {'min':>5}")
    print(f"  {'-'*8}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*5}")
    for r in all_results:
        print(f"  {r['lr']:>8.0e}  {r['batch_size']:>4}  {r['eff_batch']:>6}  {r['optimizer']:>6}  "
              f"{r['train_loss']:>8.4f}  {r['val_loss']:>8.4f}  {r['val_bpb']:>8.4f}  {r['train_min']:>5.1f}")
    print(f"{'='*80}", flush=True)

if __name__ == "__main__": main()
