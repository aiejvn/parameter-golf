"""
Masked Diffusion Language Model (MDLM) for OpenAI Parameter Golf.
Bidirectional transformer with log-linear noise schedule, adaLN timestep
conditioning, and discrete absorbing-mask ELBO evaluation.

EOS learning: token 1 (<s> in SP1024) marks document boundaries.
EOS positions are never masked — they serve as visible anchors.
Positions after EOS are filled with PAD_ID (1025), also never masked,
and excluded from loss via content_mask.
"""
import glob, io, os, math, time, json, zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

VOCAB_SIZE  = 1024
MASK_ID     = 1024
EOS_ID      = 1       # <s> in SP1024 = document boundary marker
PAD_ID      = 1025    # dedicated padding token; never masked, never in loss
TOTAL_VOCAB = 1026    # real(1024) + MASK(1) + PAD(1)
PADDED_VOCAB = 1088   # embedding table size (unchanged, has headroom)
DEVICE = "cuda"; SEED = 42

NUM_LAYERS = 11; MODEL_DIM = 512; NUM_HEADS = 8; MLP_MULT = 3.0
SEQ_LEN = 2048; BATCH_SIZE = 8; GRAD_ACCUM = 4

TRAIN_STEPS = 6000
LR = 6e-4
WARMUP_STEPS = 300 # Note: this is part of training steps
WARMDOWN_STEPS = 1500

NOISE_EPS = 1e-3
VAR_EVAL_STEPS = 128  # higher = tighter bound

DATA_DIR         = "data/datasets/fineweb10B_sp1024"
MAX_TRAIN_SHARDS = 1  # set 0 to load all shards (80 = ~16 GB)
TOKENIZER_PATH   = "data/tokenizers/fineweb_1024_bpe.model"
EXPORT_PATH      = os.path.expanduser("~/llada_v5.bin")

INT8_CLIP_Q      = 99.99984 / 100.0
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536

torch.manual_seed(SEED); np.random.seed(SEED)
NEG_INF = -1e6


# ─── Log-linear noise schedule (from MDLM) ───
def log_linear_noise(t, eps=NOISE_EPS):
    """sigma(t) = -log(1 - (1-eps)*t), alpha(t) = exp(-sigma(t)) = 1 - (1-eps)*t"""
    alpha = 1 - (1 - eps) * t
    sigma = -torch.log(alpha.clamp(min=1e-8))
    return sigma, alpha


# ─── Model ───
def rms_norm(x): return F.rms_norm(x, (x.size(-1),))

def apply_rotary(x, cos, sin):
    d = x.shape[3] // 2; x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1*cos+x2*sin, x1*(-sin)+x2*cos], dim=3)

class TimestepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))
        half = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000)*torch.arange(half,dtype=torch.float32)/half))
    def forward(self, sigma):
        emb = sigma[:, None] * self.freqs[None, :]
        return self.mlp(torch.cat([emb.sin(), emb.cos()], dim=-1))

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads, self.hd = n_heads, dim // n_heads
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, dim, bias=False)
        self.c_v = nn.Linear(dim, dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.c_q(x).view(B,T,self.n_heads,self.hd)
        k = self.c_k(x).view(B,T,self.n_heads,self.hd)
        v = self.c_v(x).view(B,T,self.n_heads,self.hd)
        q, k = apply_rotary(q,cos,sin), apply_rotary(k,cos,sin)
        q, k = rms_norm(q), rms_norm(k)
        y = F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=False)
        return self.c_proj(y.transpose(1,2).contiguous().view(B,T,-1))

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
        self.attn = Attention(dim, n_heads)
        self.adaln_attn = AdaLN(dim, cond_dim)
        self.adaln_mlp = AdaLN(dim, cond_dim)
        hidden = int(dim * mlp_mult)
        self.mlp_fc = nn.Linear(dim, hidden, bias=False)
        self.mlp_proj = nn.Linear(hidden, dim, bias=False)
    def forward(self, x, cos, sin, c):
        x = x + self.attn(self.adaln_attn(x, c), cos, sin)
        x = x + self.mlp_proj(F.relu(self.mlp_fc(self.adaln_mlp(x, c))).square())
        return x

class DiffusionLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(PADDED_VOCAB, MODEL_DIM)
        self.sigma_map = TimestepEmbedder(128)
        self.blocks = nn.ModuleList([Block(MODEL_DIM, NUM_HEADS, MLP_MULT) for _ in range(NUM_LAYERS)])
        self.head = nn.Linear(MODEL_DIM, PADDED_VOCAB, bias=False)
        hd = MODEL_DIM // NUM_HEADS
        inv_freq = 1.0/(10000**(torch.arange(0,hd,2,dtype=torch.float32)/hd))
        freqs = torch.outer(torch.arange(SEQ_LEN*2, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", freqs.cos()[None,:,None,:])
        self.register_buffer("sin", freqs.sin()[None,:,None,:])

    def forward_logits(self, xt, sigma):
        """Raw logits (used for training)."""
        B, T = xt.shape
        x = self.wte(xt)
        c = F.silu(self.sigma_map(sigma)).to(dtype=x.dtype)
        cos, sin = self.cos[:,:T], self.sin[:,:T]
        for b in self.blocks:
            x = b(x, cos, sin, c)
        logits = self.head(rms_norm(x))[...,:TOTAL_VOCAB].float()
        return logits

    def subs_log_probs(self, xt, sigma):
        """MDLM substitution log probs with frozen visible tokens.

        Visible = any token that is not MASK_ID (includes EOS and PAD).
        MASK_ID and PAD_ID are blocked from being predicted.
        """
        logits = self.forward_logits(xt, sigma)
        logits[:, :, MASK_ID] = NEG_INF  # never predict MASK
        logits[:, :, PAD_ID]  = NEG_INF  # never predict PAD
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # Visible tokens (content, EOS, PAD): freeze to identity
        frozen = torch.full_like(logits, NEG_INF)
        frozen.scatter_(-1, xt[..., None], 0.0)
        visible = (xt != MASK_ID)[..., None]
        return torch.where(visible, frozen, logits)


# ─── Training loss (MDLM continuous-time ELBO) ───
def mdlm_loss(model, x0):
    B = x0.shape[0]
    # Antithetic sampling
    t = torch.rand(B // 2 + 1, device=x0.device)
    t = torch.cat([t, 1 - t])[:B].clamp(1e-5, 1 - 1e-5)

    sigma, alpha = log_linear_noise(t)
    move_chance = 1 - alpha

    # EOS and PAD are always visible — never enter the masked diffusion process
    is_special = (x0 == EOS_ID) | (x0 == PAD_ID)
    move = (torch.rand_like(x0.float()) < move_chance[:, None]) & ~is_special
    xt = torch.where(move, MASK_ID, x0)

    log_probs = model.subs_log_probs(xt, sigma)
    # PAD positions: gather at a safe index (0) to avoid -inf * 0 = nan
    x0_safe = x0.masked_fill(x0 == PAD_ID, 0)
    log_p_x0 = torch.gather(log_probs, -1, x0_safe[..., None]).squeeze(-1)

    dsigma = (1 - NOISE_EPS) / alpha

    is_masked    = (xt == MASK_ID).float()
    content_mask = (x0 != PAD_ID).float()  # 1 for real tokens + EOS, 0 for PAD

    n_content = content_mask.sum().clamp(min=1)
    loss = (dsigma[:, None] * (-log_p_x0) * is_masked * content_mask).sum() / n_content
    return loss


# ─── Discrete ELBO eval ───
@torch.no_grad()
def variational_elbo_bits(model, x0, n_steps=128):
    """Proper discrete absorbing-mask ELBO. Returns total bits for the batch."""
    B, L = x0.shape
    total_bits = torch.zeros(B, device=x0.device)

    is_special   = (x0 == EOS_ID) | (x0 == PAD_ID)
    content_mask = (x0 != PAD_ID).float()           # [B, L]
    x0_safe      = x0.masked_fill(x0 == PAD_ID, 0)  # safe gather index

    t_grid = torch.arange(1, n_steps+1, device=x0.device, dtype=torch.float32) / n_steps
    sigma_grid, alpha_grid = log_linear_noise(t_grid)

    # Terminal KL: only over content positions (not PAD)
    alpha_T   = alpha_grid[-1]
    n_content = content_mask.sum(dim=-1)  # [B]
    total_bits += n_content * float(alpha_T) * math.log(VOCAB_SIZE) / math.log(2.0)

    alpha_prev = 1.0
    for step in range(n_steps):
        alpha_curr  = alpha_grid[step]
        sigma_curr  = sigma_grid[step].expand(B)
        move_chance = 1 - alpha_curr

        move = (torch.rand_like(x0.float()) < move_chance) & ~is_special
        xt   = torch.where(move, MASK_ID, x0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            log_probs = model.subs_log_probs(xt, sigma_curr)
        log_p_x0 = torch.gather(log_probs.float(), -1, x0_safe[..., None]).squeeze(-1)

        reveal_prob = (alpha_prev - float(alpha_curr)) / max(1.0 - float(alpha_curr), 1e-12)
        is_masked   = (xt == MASK_ID).float()
        step_bits   = reveal_prob * (-log_p_x0) * is_masked * content_mask / math.log(2.0)
        total_bits += step_bits.sum(dim=-1)

        alpha_prev = float(alpha_curr)

    return total_bits  # [B] total bits per sequence


# ─── Data ───
def _load_shard(path):
    with open(path, "rb") as f:
        f.read(256 * 4)  # skip header
        return np.frombuffer(f.read(), dtype=np.uint16).astype(np.int64)


def load_tokens(split):
    import glob
    pattern = os.path.join(DATA_DIR, f"fineweb_{split}_*.bin")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shards found: {pattern}")
    if split == "train" and MAX_TRAIN_SHARDS > 0:
        paths = paths[:MAX_TRAIN_SHARDS]
    print(f"  Loading {len(paths)} {split} shard(s)...", flush=True)
    return np.concatenate([_load_shard(p) for p in paths])


def build_chunk_index(tokens_np, seq_len):
    """
    Split every document into contiguous chunks of at most seq_len tokens.

    Each document runs from one EOS_ID (exclusive) to the next (inclusive).
    Short docs  (≤ seq_len): one chunk — content + closing EOS.
    Long docs   (> seq_len): N full chunks of exactly seq_len (no EOS),
                             followed by one tail chunk ending at EOS.

    Returns a list of (start, end) pairs with end - start <= seq_len.
    Chunks that include the closing EOS will get PAD fill in sample_doc_batch;
    mid-doc chunks fill the full seq_len with no PAD needed.
    """
    eos_positions = np.where(tokens_np == EOS_ID)[0]
    chunks = []
    for k in range(len(eos_positions) - 1):
        start   = int(eos_positions[k]) + 1   # first content token
        end_eos = int(eos_positions[k + 1])   # closing EOS position (inclusive)
        pos = start
        while pos < end_eos + 1:
            chunk_end = min(pos + seq_len, end_eos + 1)
            chunks.append((pos, chunk_end))
            pos = chunk_end
    return chunks


def sample_doc_batch(tokens_np, chunks, batch_size, seq_len, device):
    """
    Sample batch_size chunks uniformly at random.
    Chunks shorter than seq_len (those ending at EOS) are right-padded with PAD_ID.
    """
    ki    = np.random.randint(0, len(chunks), size=batch_size)
    batch = np.full((batch_size, seq_len), PAD_ID, dtype=np.int64)
    for b, k in enumerate(ki):
        start, end = chunks[k]
        length = end - start
        batch[b, :length] = tokens_np[start:end]
    return torch.from_numpy(batch).to(device)


def get_lr(step):
    if step < WARMUP_STEPS: return LR * (step+1)/WARMUP_STEPS
    elif step < TRAIN_STEPS - WARMDOWN_STEPS: return LR
    else:
        progress = (TRAIN_STEPS - step) / WARMDOWN_STEPS
        return LR * (0.1 + 0.9*(0.5*(1+math.cos(math.pi*(1-progress)))))


# ─── Post-training quantization (int8 per-row, zlib export) ───
def _quantize_tensor(t):
    """Per-row int8 for matrices, per-tensor int8 for vectors/scalars."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1).clamp_min(1e-8)
        clipped  = t32.clamp(-clip_abs[:, None], clip_abs[:, None])
        scale    = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q        = clipped.div(scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        return q.contiguous(), scale.to(torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale    = torch.tensor(max(clip_abs / 127.0, 1e-8))
    q        = t32.clamp(-clip_abs, clip_abs).div(scale).round().clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale


def quantize_state_dict(state_dict):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            passthrough[name] = t.to(torch.float16) if t.is_floating_point() else t
        else:
            q, s = _quantize_tensor(t)
            quantized[name] = q; scales[name] = s
            dtypes[name] = str(t.dtype).removeprefix("torch.")
    return {"quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}


def dequantize_state_dict(obj):
    out = {}
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name].float()
        if s.ndim > 0:  # per-row scale
            out[name] = (q.float() * s[:, None]).to(dtype)
        else:
            out[name] = (q.float() * s.item()).to(dtype)
    for name, t in obj["passthrough"].items():
        out[name] = t.contiguous()
    return out


def export_model(model, path):
    """Quantize → torch-serialize → zlib-compress → write to path."""
    obj   = quantize_state_dict({k: v for k, v in model.state_dict().items()})
    buf   = io.BytesIO()
    torch.save(obj, buf)
    blob  = zlib.compress(buf.getvalue(), level=9)
    with open(path, "wb") as f:
        f.write(blob)
    raw_mb  = buf.tell() / 1e6
    comp_mb = len(blob) / 1e6
    print(f"  Exported: {raw_mb:.1f} MB → {comp_mb:.1f} MB (zlib-9) → {path}")
    return comp_mb


# ─── Competition BPB eval ───
def build_sentencepiece_luts(sp, vocab_size, device):
    """Byte-count lookup tables matching the competition's scoring."""
    table_size = max(int(sp.vocab_size()), vocab_size)
    base_bytes      = np.zeros(table_size, dtype=np.int16)
    has_lead_space  = np.zeros(table_size, dtype=np.bool_)
    is_boundary     = np.ones(table_size,  dtype=np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_lead_space[tid] = True; piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes,     dtype=torch.int16, device=device),
            torch.tensor(has_lead_space, dtype=torch.bool,  device=device),
            torch.tensor(is_boundary,    dtype=torch.bool,  device=device))


@torch.no_grad()
def competition_bpb_eval(model, val_np, sp, device, n_steps=VAR_EVAL_STEPS, max_seqs=500):
    """
    Compute competition val_bpb: ELBO bits / tokenizer bytes.
    Uses the same sentencepiece byte-counting as train_gpt.py eval_val.
    """
    base_lut, space_lut, boundary_lut = build_sentencepiece_luts(sp, VOCAB_SIZE, device)
    val_tokens = torch.from_numpy(val_np)
    n_seqs     = min(max_seqs, (len(val_np) - 1) // SEQ_LEN)
    total_bits = 0.0; total_bytes = 0.0

    model.eval()
    for i in range(n_seqs):
        x        = val_tokens[i*SEQ_LEN:(i+1)*SEQ_LEN].to(device)
        bits     = variational_elbo_bits(model, x.unsqueeze(0), n_steps=n_steps)
        total_bits += bits.sum().item()

        # Count bytes using tokenizer luts (competition formula)
        tgt  = x                           # target tokens [L]
        prev = torch.cat([x[:1], x[:-1]])  # previous tokens (for space detection)
        tok_bytes = base_lut[tgt].to(torch.int16)
        tok_bytes += (space_lut[tgt] & ~boundary_lut[prev]).to(torch.int16)
        total_bytes += tok_bytes.float().sum().item()

        if (i+1) % 50 == 0:
            print(f"  bpb_eval {i+1}/{n_seqs} | bpb={total_bits/max(total_bytes,1):.4f}", flush=True)

    return total_bits / max(total_bytes, 1.0)


# ─── Main ───
def main():
    print("="*60)
    print("  LLaDA v5: MDLM + EOS learning + post-EOS zeroing")
    print("="*60, flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_np     = load_tokens("train")
    val_np       = load_tokens("val")
    train_chunks = build_chunk_index(train_np, SEQ_LEN)
    n_docs = len(np.where(train_np == EOS_ID)[0])
    print(f"Train: {len(train_np):,} tokens, {n_docs:,} documents, {len(train_chunks):,} chunks")
    print(f"Val:   {len(val_np):,} tokens")

    model = DiffusionLM().to(DEVICE).to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {NUM_LAYERS}L {MODEL_DIM}d {NUM_HEADS}h — {n_params:,} params")
    print(f"EOS_ID={EOS_ID} never masked; PAD_ID={PAD_ID} always visible, excluded from loss")
    print(f"Eval: discrete ELBO ({VAR_EVAL_STEPS} steps)\n", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1, fused=True)
    val_chunks = build_chunk_index(val_np, SEQ_LEN)
    t0 = time.time(); losses = []
    model.train()

    for step in range(TRAIN_STEPS):
        lr = get_lr(step)
        for g in optimizer.param_groups: g['lr'] = lr
        optimizer.zero_grad(set_to_none=True); accum_loss = 0.0
        for _ in range(GRAD_ACCUM):
            batch = sample_doc_batch(train_np, train_chunks, BATCH_SIZE, SEQ_LEN, DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = mdlm_loss(model, batch) / GRAD_ACCUM
            loss.backward(); accum_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); losses.append(accum_loss)
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_batch = sample_doc_batch(val_np, val_chunks, BATCH_SIZE * GRAD_ACCUM, SEQ_LEN, DEVICE)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    val_loss = mdlm_loss(model, val_batch).item()
            model.train()
            avg     = np.mean(losses[-100:])
            elapsed = time.time() - t0
            tok_s   = (step+1)*BATCH_SIZE*GRAD_ACCUM*SEQ_LEN/elapsed
            print(f"  step {step:5d}/{TRAIN_STEPS} | loss={avg:.4f} | val={val_loss:.4f} | lr={lr:.1e} | {tok_s/1e3:.0f}K tok/s | {elapsed:.0f}s", flush=True)

    train_time = time.time() - t0
    print(f"\nTraining done: {train_time/60:.1f}m, loss={np.mean(losses[-100:]):.4f}", flush=True)

    # ── Export (quantized + zlib) ──
    print(f"\nExporting model to {EXPORT_PATH} ...", flush=True)
    comp_mb = export_model(model, EXPORT_PATH)

    # ── Competition BPB eval ──
    print(f"\nCompetition BPB eval ({VAR_EVAL_STEPS} steps, up to 500 seqs)...", flush=True)
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    model.eval()
    bpb = competition_bpb_eval(model, val_np, sp, DEVICE,
                               n_steps=VAR_EVAL_STEPS, max_seqs=500)

    print(f"\n{'='*60}")
    print(f"  COMPETITION BPB: {bpb:.4f}")
    print(f"  PR #820 MDLM:    1.625")
    print(f"  AR baseline:     1.22")
    print(f"  Export size:     {comp_mb:.1f} MB")
    print(f"{'='*60}", flush=True)

    json.dump({"val_bpb": bpb, "export_mb": comp_mb, "params": n_params,
               "train_min": train_time/60, "var_eval_steps": VAR_EVAL_STEPS,
               "train_loss": float(np.mean(losses[-100:]))},
              open(os.path.expanduser("~/v5_results.json"), "w"), indent=2)

if __name__ == "__main__": main()
