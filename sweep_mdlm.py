"""
Sweep over attention head counts for the MDLM.
Trains one config per entry in SWEEP_N_HEADS on a single shard,
then writes results to sweep_results.csv.
"""
import csv, glob, os, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── fixed hyperparams ─────────────────────────────────────────────────────────
VOCAB_SIZE   = 1024
MASK_ID      = 1024
EOS_ID       = 1
PAD_ID       = 1025
TOTAL_VOCAB  = 1026
PADDED_VOCAB = 1088
DEVICE       = "cuda"; SEED = 42

NUM_LAYERS = 11; MODEL_DIM = 512; MLP_MULT = 3.0
SEQ_LEN = 2048; BATCH_SIZE = 8; GRAD_ACCUM = 4

TRAIN_STEPS    = 5
LR             = 6e-4
WARMUP_STEPS   = 300
WARMDOWN_STEPS = 1500

NOISE_EPS      = 1e-3
VAR_EVAL_STEPS = 32   # reduced for sweep speed; bump for final eval

DATA_DIR         = "data/datasets/fineweb10B_sp1024"
MAX_TRAIN_SHARDS = 0     # 0 = all available shards
SHARDS_IN_MEMORY = 1     # shards loaded at once when rotating
ROTATE_SHARDS    = True # True = shard rotation; False = single shard (default for sweep)
RESULTS_CSV      = "sweep_results.csv"

# ── sweep config ──────────────────────────────────────────────────────────────
SWEEP_N_HEADS = [2, 4, 8, 16, 32]

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
        self.attn      = Attention(dim, n_heads)
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
    def __init__(self, n_heads):
        super().__init__()
        self.wte       = nn.Embedding(PADDED_VOCAB, MODEL_DIM)
        self.sigma_map = TimestepEmbedder(128)
        self.blocks    = nn.ModuleList([Block(MODEL_DIM, n_heads, MLP_MULT) for _ in range(NUM_LAYERS)])
        self.head      = nn.Linear(MODEL_DIM, PADDED_VOCAB, bias=False)
        hd = MODEL_DIM // n_heads
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
        visible = (xt != MASK_ID)[..., None]
        return torch.where(visible, frozen, logits)


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
    B, L       = x0.shape
    total_bits = torch.zeros(B, device=x0.device)
    is_special   = (x0 == EOS_ID) | (x0 == PAD_ID)
    content_mask = (x0 != PAD_ID).float()
    x0_safe      = x0.masked_fill(x0 == PAD_ID, 0)
    t_grid = torch.arange(1, n_steps+1, device=x0.device, dtype=torch.float32) / n_steps
    sigma_grid, alpha_grid = log_linear_noise(t_grid)
    alpha_T    = alpha_grid[-1]
    n_content  = content_mask.sum(dim=-1)
    total_bits += n_content * float(alpha_T) * math.log(VOCAB_SIZE) / math.log(2.0)
    alpha_prev = 1.0
    for step in range(n_steps):
        alpha_curr  = alpha_grid[step]
        sigma_curr  = sigma_grid[step].expand(B)
        move        = (torch.rand_like(x0.float()) < (1 - alpha_curr)) & ~is_special
        xt          = torch.where(move, MASK_ID, x0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            log_probs = model.subs_log_probs(xt, sigma_curr)
        log_p_x0    = torch.gather(log_probs.float(), -1, x0_safe[..., None]).squeeze(-1)
        reveal_prob = (alpha_prev - float(alpha_curr)) / max(1.0 - float(alpha_curr), 1e-12)
        is_masked   = (xt == MASK_ID).float()
        total_bits += (reveal_prob * (-log_p_x0) * is_masked * content_mask / math.log(2.0)).sum(dim=-1)
        alpha_prev  = float(alpha_curr)
    return total_bits


# ── data ──────────────────────────────────────────────────────────────────────
def _load_shard(path):
    with open(path, "rb") as f:
        f.read(256 * 4)
        return np.frombuffer(f.read(), dtype=np.uint16).astype(np.int64)

def find_shards(split):
    pattern = os.path.join(DATA_DIR, f"fineweb_{split}_*.bin")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shards found: {pattern}")
    if split == "train" and MAX_TRAIN_SHARDS > 0:
        paths = paths[:MAX_TRAIN_SHARDS]
    return paths

def load_tokens(split):
    """Load all (or MAX_TRAIN_SHARDS) shards at once."""
    paths = find_shards(split)
    print(f"  Loading {len(paths)} {split} shard(s)...", flush=True)
    return np.concatenate([_load_shard(p) for p in paths])

class ShardedDataLoader:
    """
    Splits train shards into groups of SHARDS_IN_MEMORY.
    Call load_group(i) to load group i explicitly.
    Total training steps = TRAIN_STEPS * n_groups (one full pass per group).
    """
    def __init__(self):
        self.paths     = find_shards("train")
        self.n_shards  = len(self.paths)
        self.window    = min(SHARDS_IN_MEMORY, self.n_shards)
        self.n_groups  = math.ceil(self.n_shards / self.window)
        self.tokens_np = None
        self.chunks    = None
        print(f"  ShardedDataLoader: {self.n_shards} shards, "
              f"{self.window} per group, {self.n_groups} groups, "
              f"{TRAIN_STEPS} steps/group → {TRAIN_STEPS * self.n_groups} total steps", flush=True)

    def load_group(self, group_idx):
        import gc
        self.tokens_np = None
        self.chunks    = None
        gc.collect()
        start       = group_idx * self.window
        batch_paths = self.paths[start:start + self.window]
        print(f"\n  [group {group_idx}/{self.n_groups}] Loading: "
              f"{[os.path.basename(p) for p in batch_paths]}", flush=True)
        sizes = []
        for p in batch_paths:
            with open(p, "rb") as f:
                f.read(256 * 4)
                sizes.append((os.fstat(f.fileno()).st_size - 256 * 4) // 2)
        self.tokens_np = np.empty(sum(sizes), dtype=np.int64)
        offset = 0
        for p, n in zip(batch_paths, sizes):
            self.tokens_np[offset:offset + n] = _load_shard(p)
            offset += n
        gc.collect()
        self.chunks = build_chunk_index(self.tokens_np, SEQ_LEN)
        print(f"  {len(self.tokens_np):,} tokens, {len(self.chunks):,} chunks", flush=True)

    def sample_batch(self, batch_size, device):
        return sample_doc_batch(self.tokens_np, self.chunks, batch_size, device)

def build_chunk_index(tokens_np, seq_len):
    eos_positions = np.where(tokens_np == EOS_ID)[0]
    chunks = []
    for k in range(len(eos_positions) - 1):
        start   = int(eos_positions[k]) + 1
        end_eos = int(eos_positions[k + 1])
        pos = start
        while pos < end_eos + 1:
            chunk_end = min(pos + seq_len, end_eos + 1)
            chunks.append((pos, chunk_end))
            pos = chunk_end
    return chunks

def sample_doc_batch(tokens_np, chunks, batch_size, device):
    ki    = np.random.randint(0, len(chunks), size=batch_size)
    batch = np.full((batch_size, SEQ_LEN), PAD_ID, dtype=np.int64)
    for b, k in enumerate(ki):
        start, end = chunks[k]
        batch[b, :end - start] = tokens_np[start:end]
    return torch.from_numpy(batch).to(device)

def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    elif step < TRAIN_STEPS - WARMDOWN_STEPS:
        return LR
    else:
        progress = (TRAIN_STEPS - step) / WARMDOWN_STEPS
        return LR * (0.1 + 0.9 * (0.5 * (1 + math.cos(math.pi * (1 - progress)))))


# ── single config training run ────────────────────────────────────────────────
def train_one_config(n_heads, train_source, val_np, val_chunks):
    """
    train_source: ShardedDataLoader when ROTATE_SHARDS=True,
                  (train_np, train_chunks) tuple otherwise.
    """
    torch.manual_seed(SEED); np.random.seed(SEED)
    model    = DiffusionLM(n_heads).to(DEVICE).to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}", flush=True)
    print(f"  n_heads={n_heads}  hd={MODEL_DIM//n_heads}  params={n_params:,}", flush=True)
    print(f"{'='*60}", flush=True)

    if ROTATE_SHARDS:
        n_groups = train_source.n_groups
    else:
        train_np, train_chunks = train_source
        n_groups = 1

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                  weight_decay=0.1, fused=True)
    t0 = time.time(); losses = []
    global_step = 0
    model.train()

    for group in range(n_groups):
        if ROTATE_SHARDS:
            train_source.load_group(group)
        for step in range(TRAIN_STEPS):
            lr = get_lr(step)
            for g in optimizer.param_groups: g["lr"] = lr
            optimizer.zero_grad(set_to_none=True); accum_loss = 0.0
            for _ in range(GRAD_ACCUM):
                if ROTATE_SHARDS:
                    batch = train_source.sample_batch(BATCH_SIZE, DEVICE)
                else:
                    batch = sample_doc_batch(train_np, train_chunks, BATCH_SIZE, DEVICE)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = mdlm_loss(model, batch) / GRAD_ACCUM
                loss.backward(); accum_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); losses.append(accum_loss)

            if step % 1 == 0:
                model.eval()
                with torch.no_grad():
                    val_batch = sample_doc_batch(val_np, val_chunks, BATCH_SIZE * GRAD_ACCUM, DEVICE)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        val_loss = mdlm_loss(model, val_batch).item()
                    # 1-sequence ELBO sample for a quick bpb estimate
                    bpb_x   = val_batch[:1]
                    bits    = variational_elbo_bits(model, bpb_x, n_steps=16).sum().item()
                    n_cont  = (bpb_x != PAD_ID).sum().item()
                    bpb_est = bits / max(n_cont * 4.3, 1)
                model.train()
                avg     = float(np.mean(losses[-200:]))
                elapsed = time.time() - t0
                tok_s   = (global_step+1)*BATCH_SIZE*GRAD_ACCUM*SEQ_LEN/elapsed
                print(f"  [g{group}/{n_groups} s{step}/{TRAIN_STEPS}] gs={global_step} | "
                      f"loss={avg:.4f} | val={val_loss:.4f} | bpb~{bpb_est:.3f} | {tok_s/1e3:.0f}K tok/s | "
                      f"lr={lr:.1e} | {elapsed:.0f}s", flush=True)
            global_step += 1

    train_time = time.time() - t0
    final_train_loss = float(np.mean(losses[-100:]))

    # ── final val loss ────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        val_batch = sample_doc_batch(val_np, val_chunks, BATCH_SIZE * GRAD_ACCUM, DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            final_val_loss = mdlm_loss(model, val_batch).item()

    # ── val bpb (ELBO over fixed val windows) ────────────────────────────────
    val_tokens  = torch.from_numpy(val_np)
    total_bits  = 0.0; total_content = 0
    n_seqs      = min(200, (len(val_np) - 1) // SEQ_LEN)
    with torch.no_grad():
        for i in range(n_seqs):
            x    = val_tokens[i*SEQ_LEN:(i+1)*SEQ_LEN].unsqueeze(0).to(DEVICE)
            bits = variational_elbo_bits(model, x).sum().item()
            total_bits    += bits
            total_content += SEQ_LEN
    val_bpb = total_bits / max(total_content * 4.3, 1)

    print(f"\n  DONE n_heads={n_heads} | train_loss={final_train_loss:.4f} | "
          f"val_loss={final_val_loss:.4f} | val_bpb={val_bpb:.4f} | {train_time/60:.1f}min", flush=True)

    return {
        "n_heads":        n_heads,
        "head_dim":       MODEL_DIM // n_heads,
        "params":         n_params,
        "train_loss":     round(final_train_loss, 6),
        "val_loss":       round(final_val_loss,   6),
        "val_bpb":        round(val_bpb,          6),
        "train_min":      round(train_time / 60,  2),
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("MDLM attention head sweep", flush=True)
    print(f"Configs: {SWEEP_N_HEADS}", flush=True)
    print(f"ROTATE_SHARDS={ROTATE_SHARDS}  SHARDS_IN_MEMORY={SHARDS_IN_MEMORY}  "
          f"MAX_TRAIN_SHARDS={MAX_TRAIN_SHARDS}\n", flush=True)

    if ROTATE_SHARDS:
        train_source = ShardedDataLoader()
    else:
        train_np     = load_tokens("train")
        train_chunks = build_chunk_index(train_np, SEQ_LEN)
        train_source = (train_np, train_chunks)
        print(f"Train: {len(train_np):,} tokens, {len(train_chunks):,} chunks")

    val_np     = load_tokens("val")
    val_chunks = build_chunk_index(val_np, SEQ_LEN)
    print(f"Val:   {len(val_np):,} tokens\n")

    # Write CSV header up front so partial results survive a crash
    csv_fields = ["n_heads", "head_dim", "params", "train_loss", "val_loss", "val_bpb", "train_min"]
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    all_results = []
    for n_heads in SWEEP_N_HEADS:
        result = train_one_config(n_heads, train_source, val_np, val_chunks)
        all_results.append(result)
        # Append row immediately so results are saved even if sweep crashes later
        with open(RESULTS_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(result)
        print(f"  → written to {RESULTS_CSV}", flush=True)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  {'n_heads':>8}  {'head_dim':>8}  {'params':>10}  {'train_loss':>10}  {'val_loss':>8}  {'val_bpb':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for r in all_results:
        print(f"  {r['n_heads']:>8}  {r['head_dim']:>8}  {r['params']:>10,}  "
              f"{r['train_loss']:>10.4f}  {r['val_loss']:>8.4f}  {r['val_bpb']:>8.4f}")
    print(f"{'='*70}", flush=True)

if __name__ == "__main__": main()
