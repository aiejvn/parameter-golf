# Goal: Adaptive Length

To get adaptive length you'd need to either:

* Treat EOS as a learnable token and mask everything after the first predicted EOS at inference
* Use a separate length predictor before running the denoising chain
* Absorb into a "padding" token class instead of a mask token, with EOS marking the boundary

None of that is in the current code.

## EOS as Learnable Token:

__3/30/2026:__

New constants (lines 12-19):

EOS_ID = 1, PAD_ID = 1025, TOTAL_VOCAB = 1026 — PAD gets its own slot, no collision with MASK
subs_log_probs (line 117): blocks both MASK_ID and PAD_ID from being predicted. PAD and EOS are treated as visible (frozen to identity) since xt != MASK_ID for both.

mdlm_loss (lines 136-152):

is_special = (x0 == EOS_ID) | (x0 == PAD_ID) — neither gets masked
content_mask = (x0 != PAD_ID) — EOS contributes to loss, PAD doesn't
x0_safe remaps PAD positions to index 0 for gathering, preventing -inf * 0 = nan
variational_elbo_bits (lines 156-185): same is_special/content_mask treatment; terminal KL scaled by n_content not fixed L

Data pipeline (lines 188-220):

load_tokens now returns np.ndarray (not tensor) so build_eos_index can use np.where efficiently
build_eos_index finds all document boundaries in O(n)
sample_doc_batch samples document-aligned sequences with proper PAD fill

Issues:
- Docs over 2048 tokens - how to handle?

MDLM Loss: How It Works
The Forward Process (Noising)
At time t ∈ [0,1], each token is independently either kept or replaced with [MASK]:


P(x_t[i] = MASK  | x_0[i]) = 1 - α(t)
P(x_t[i] = x_0[i] | x_0[i]) = α(t)
With the log-linear schedule (line 26-30):


α(t) = 1 - (1-ε)t        ← linearly decays from ~1 to ~ε
σ(t) = -log α(t)          ← log-noise level, used to condition the model
At t=0: nothing masked. At t=1: everything masked (absorbing state).

---

The Training Loss (lines 127-147)
The continuous-time ELBO for absorbing diffusion is:


L = E_t [ dσ/dt · E_{x_t|x_0} [ -log p_θ(x_0 | x_t) · 𝟙[x_t = MASK] ] ]
Three terms to unpack:

1. -log p_θ(x_0 | x_t) — standard cross-entropy: how well can the model predict the original token given the noisy sequence. Only computed at masked positions (the indicator 𝟙). Visible tokens are frozen to identity — the model doesn't need to predict what it can already see.

2. dσ/dt — the importance weight from integrating over continuous time. For the log-linear schedule:


dσ/dt = (1-ε) / α(t)
This upweights high-t timesteps (lots of masking, harder) and downweights low-t (most tokens visible, trivial). In code this is dsigma = (1 - NOISE_EPS) / alpha.

3. Antithetic sampling (line 130-131) — instead of sampling t i.i.d., pairs (t, 1-t) are sampled. Variance cancels out between the easy and hard halves of the schedule — equivalent to stratified sampling for free.

Variational ELBO Bits: The Math
The ELBO bits is a Riemann sum approximation of the continuous ELBO, converted to bits. It gives an upper bound on the true bits-per-byte (tighter as n_steps increases).

Three components
Step KL (the main sum, lines 165-178):

For each discrete timestep k, the backward posterior for absorbing diffusion is:


q(x_{t-1} = x_0 | x_t = MASK, x_0)  =  (α_{k-1} - α_k) / (1 - α_k)  ← reveal probability
q(x_{t-1} = MASK | x_t = MASK, x_0) =  (1 - α_{k-1}) / (1 - α_k)     ← stay masked
reveal_prob = (α_{k-1} - α_k) / (1 - α_k) is "given a token is currently masked, what's the chance it gets unmasked at this step?" The KL between the true posterior and the model's denoising distribution simplifies to:


KL_k = reveal_prob · (-log p_θ(x_0 | x_t)) · 𝟙[x_t = MASK]
Converted to bits (/ log 2), summed over positions and steps → step_bits.

Terminal KL (lines 161-162):

At t=1, the ~ε=0.001 fraction of tokens still not masked have no information to condition on. The code charges log₂(VOCAB_SIZE) bits per such token (uniform prior penalty):


total_bits += L * float(alpha_T) * math.log(VOCAB_SIZE) / math.log(2.0)
Intuition for the bound: the model is being asked "at each denoising step, how surprised are you by the true token?" Summing that surprise in bits over all steps and positions gives the total information content — a valid upper bound on -log₂ p(x_0) by Jensen's inequality.

Architecture Overview
Component	Value
Layers	11
model_dim	512, 8 heads (head_dim=64)
MLP	3× expansion, relu²
Sequence length	2048
Attention	Bidirectional (is_causal=False)
Positional encoding	RoPE
QK norm	RMSNorm (same trick as GPT baseline)
Vocab	1025 (1024 real + MASK token)
Key differences from the autoregressive GPT:

Bidirectional attention — the model sees the full (partially masked) sequence at once, which is valid since at inference time you know which positions are masked
AdaLN timestep conditioning (lines 68-74) — σ(t) is sinusoidally embedded → MLP → produces per-layer scale s and shift sh applied after RMSNorm. Each layer gets different modulation based on how much noise is present. This replaces the attn_scale/mlp_scale/resid_mix learnable scalars in the GPT
No causal mask — the model can use future context to predict masked tokens, which is why diffusion LMs can potentially be better compressors: they condition on everything observable