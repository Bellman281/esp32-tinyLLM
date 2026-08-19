"""Tiny decoder-only transformer with pluggable embedding strategies.

The experiment this file exists to run: on an ESP32-S3, weights touched on every
token are the expensive ones - they must be reachable from fast memory, of which
there is very little - while flash is huge but only good for sparse random reads.
(The shipped runtime stages that dense core as int8 in PSRAM rather than SRAM;
what matters to this experiment is the split between every-token and per-token
weights, not which fast tier the former ends up in.) That is the same memory-tier shape Gemma's Per-Layer
Embeddings were designed for, three orders of magnitude down. So we hold the
*core* (dense, every-token) parameter count fixed and vary how extra
*table* (sparse, per-token-lookup) parameters are spent.

Arms:
  baseline    -- no table params at all. What a tiny LM looks like today.
  ple         -- faithful Gemma-4 PLE: per-layer inputs are (RMSNorm(proj(embed))
                 + table) / sqrt(2), gated into each layer's residual.
  ple_notable -- the same per-layer adapters and projection, but *no* lookup
                 table. Isolates whether flash-resident table params buy
                 anything, or whether the per-layer plumbing alone explains any
                 gain. Costs core params, buys no table params.
  fatembed    -- same table budget as `ple`, but a wide input embedding
                 factorised down to d_model once at the bottom. Decides whether
                 the *per-layer* part earns its keep or whether any sparse
                 embedding capacity would do.
  bigcore     -- spend the table budget on a wider core instead. Reference for
                 how much a thin core is actually costing us.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    arm: str = "baseline"
    vocab_size: int = 4096
    # Rows in the output head. None, or a value equal to vocab_size, means the
    # model writes the vocabulary it reads: the two are one id space and a
    # sampled index is a token id. A different value gives the model its own
    # output alphabet, so the head is a separate tensor and a sampled index is a
    # class that has to be mapped before it can be fed back in.
    out_vocab_size: int | None = None
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    ffn_hidden: int = 256
    seq_len: int = 512
    ple_dim: int = 64
    rope_theta: float = 10000.0

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def uses_per_layer(self):
        return self.arm in ("ple", "ple_notable")

    @property
    def resolved_out_vocab_size(self):
        """Rows in the output head. Defaults to the read vocabulary."""
        return self.vocab_size if self.out_vocab_size is None else self.out_vocab_size

    @property
    def output_ids_are_input_ids(self):
        """Whether a sampled index can be fed straight back in as a token."""
        return self.resolved_out_vocab_size == self.vocab_size

    @property
    def head_is_tied(self):
        """Whether the head shares storage with tok_emb.

        Two conditions, and they are not the same question: the alphabets must
        match, and the tensors must have the same width. fatembed widens tok_emb
        to table_width, so it always keeps a separate head even though it writes
        what it reads.
        """
        return self.arm != "fatembed" and self.output_ids_are_input_ids

    @property
    def table_width(self):
        """Width of the wide input embedding in the fatembed arm.

        Chosen so its table has exactly the same parameter count as the PLE arm's
        table: vocab x (n_layers * ple_dim).
        """
        return self.n_layers * self.ple_dim


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def build_rope(seq_len, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh)
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, : x.shape[2], :]
    sin = sin[None, None, : x.shape[2], :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)
        if cfg.uses_per_layer:
            # Gemma 4's Gemma4TextDecoderLayer, minus AltUp: squeeze the hidden
            # state to ple_dim, gate it elementwise *by* the per-layer input,
            # project back, norm, add to the residual. The multiply is what makes
            # this conditioning rather than a per-token bias -- either factor can
            # suppress the other.
            self.ple_gate = nn.Linear(cfg.d_model, cfg.ple_dim, bias=False)
            self.ple_proj = nn.Linear(cfg.ple_dim, cfg.d_model, bias=False)
            self.ple_norm = RMSNorm(cfg.d_model)

    def forward(self, x, cos, sin, ple=None):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        if ple is not None:
            g = F.gelu(self.ple_gate(x))
            x = x + self.ple_norm(self.ple_proj(g * ple))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        out_vocab = cfg.resolved_out_vocab_size
        if cfg.arm == "fatembed":
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.table_width)
            self.emb_down = nn.Linear(cfg.table_width, cfg.d_model, bias=False)
            self.head = nn.Linear(cfg.d_model, out_vocab, bias=False)
        else:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.head = nn.Linear(cfg.d_model, out_vocab, bias=False)
            if cfg.head_is_tied:
                self.head.weight = self.tok_emb.weight  # tied

        if cfg.uses_per_layer:
            # Context-aware half of the per-layer input: one projection of the
            # token embedding, reshaped to (layers, ple_dim) and normed per slice.
            self.ple_model_proj = nn.Linear(cfg.d_model, cfg.n_layers * cfg.ple_dim, bias=False)
            self.ple_proj_norm = RMSNorm(cfg.ple_dim)
        if cfg.arm == "ple":
            self.ple_table = nn.Embedding(cfg.vocab_size, cfg.n_layers * cfg.ple_dim)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.d_model)

        self.apply(self._init)
        for n, p in self.named_parameters():
            # Scale down residual-writing projections, GPT-2 style.
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))
        if cfg.arm == "fatembed":
            # Keep post-projection activations at the same scale as the other
            # arms' embeddings, otherwise this arm is handicapped by init alone.
            nn.init.normal_(self.emb_down.weight, std=cfg.table_width**-0.5)
        for block in self.blocks:
            # The per-layer branch ends in an RMSNorm, which would undo the small
            # residual init above and inject unit-scale noise from step 0. Zeroing
            # the norm gain makes the branch start as an exact no-op, so every arm
            # begins from the same function and the comparison measures learning
            # rather than initialisation luck.
            if cfg.uses_per_layer:
                nn.init.zeros_(block.ple_norm.weight)

        cos, sin = build_rope(cfg.seq_len, cfg.head_dim, cfg.rope_theta, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        cfg = self.cfg
        x = self.tok_emb(idx)
        if cfg.arm == "fatembed":
            x = self.emb_down(x)

        ple = None
        if cfg.uses_per_layer:
            B, T = idx.shape
            ple = self.ple_model_proj(x) * (cfg.d_model**-0.5)
            ple = self.ple_proj_norm(ple.view(B, T, cfg.n_layers, cfg.ple_dim))
            if cfg.arm == "ple":
                # embed_scale = sqrt(ple_dim), applied to the table rather than
                # baked into init -- undocumented in Gemma's config but load-bearing.
                table = self.ple_table(idx).view(B, T, cfg.n_layers, cfg.ple_dim)
                ple = (ple + table * (cfg.ple_dim**0.5)) * (2**-0.5)

        for i, block in enumerate(self.blocks):
            x = block(x, self.cos, self.sin, None if ple is None else ple[:, :, i])

        x = self.out_norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            # Reshape by the head's own width, not the read vocabulary. The
            # two differ whenever the output width differs from vocab_size.
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.resolved_out_vocab_size),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    # ---- parameter accounting -------------------------------------------------
    # Three tiers matching the ESP32's three, by ACCESS PATTERN not just speed:
    #   core     dense, random, every token   -> the scarce budget. The
    #            shipped runtime stages it as int8 in PSRAM, not SRAM. What
    #            SRAM holds is the hot working set: activations and norms.
    #   stream   the output head: dense but read as one sequential scan per
    #            token -> costs bandwidth, not SRAM; can live off-chip
    #   table    sparse, one row per token     -> memory-mapped flash
    # `core` deliberately EXCLUDES the head. Lumping the head in makes a large
    # vocabulary look unaffordable when it is merely slow, and blocks the whole
    # cheap-table-via-big-vocab design from being built at all. See param_budget() below.

    def param_budget(self):
        table = 0
        if self.cfg.arm == "ple":
            table += self.ple_table.weight.numel()
        if not self.cfg.head_is_tied:
            # A tied embedding is already accounted for through head.weight;
            # otherwise it is sparse table data, read one row per token.
            table += self.tok_emb.weight.numel()
        stream = self.head.weight.numel()
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue  # tied weights counted once
            seen.add(id(p))
            total += p.numel()
        return {"core": total - table - stream, "stream": stream,
                "table": table, "total": total}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        # Appending a sample to idx requires that a sampled index IS a token
        # id. Under a distinct output alphabet it is a class, and the class to
        # token map is model data rather than architecture, so the caller owns
        # both the map and the sampling loop.
        if not self.cfg.output_ids_are_input_ids:
            raise ValueError(
                f"generate() needs the head to span the input vocabulary, but "
                f"this model writes {self.cfg.resolved_out_vocab_size} classes "
                f"and reads {self.cfg.vocab_size} tokens; sample with the "
                f"output-class to input-token map instead"
            )
        for _ in range(max_new_tokens):
            idx_c = idx[:, -self.cfg.seq_len :]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def make_model(arm, target_core, base: Config = None, verbose=True, fixed_ffn=None):
    """Build `arm`.

    Default: solve ffn_hidden so every arm hits the same compute-core budget
    (core now excludes the output head -- see param_budget). Without this the
    comparison is worthless: each arm's plumbing costs a different amount of
    dense parameters, and the winner would just be whichever got a fatter FFN.

    fixed_ffn: skip the solver and pin ffn_hidden. Used by the corrected
    table-scaling sweep, where the point is to hold the FFN fixed and let core
    float with ple_dim -- so growing the table never cannibalises compute. The
    table's isolated contribution is then read as (ple - ple_notable) at each
    ple_dim, both sharing the same fixed FFN.
    """
    cfg = Config(**{**(base.__dict__ if base else {}), "arm": arm})
    if fixed_ffn is not None:
        cfg.ffn_hidden = fixed_ffn
        model = TinyLM(cfg)
        if verbose:
            b = model.param_budget()
            print(f"[{arm}] d_model={cfg.d_model} layers={cfg.n_layers} "
                  f"ffn={cfg.ffn_hidden} ple_dim={cfg.ple_dim} core={b['core']:,} "
                  f"stream={b['stream']:,} table={b['table']:,} total={b['total']:,}")
        return model
    table_budget = cfg.vocab_size * cfg.n_layers * cfg.ple_dim
    if arm == "bigcore":
        # Spend the table budget on width instead, so this arm's core equals every
        # other arm's core+table. Searched, since core scales ~quadratically in d.
        best = None
        for d in range(cfg.d_model, 8 * cfg.d_model, cfg.n_heads):
            trial = Config(**{**cfg.__dict__, "arm": "baseline", "d_model": d, "ffn_hidden": 2 * d})
            if TinyLM(trial).param_budget()["core"] <= target_core + table_budget:
                best = trial
            else:
                break
        cfg = best
    else:
        lo, hi = 1, 64 * cfg.d_model
        while lo < hi:
            mid = (lo + hi + 1) // 2
            cfg.ffn_hidden = mid
            if TinyLM(cfg).param_budget()["core"] <= target_core:
                lo = mid
            else:
                hi = mid - 1
        cfg.ffn_hidden = lo

    model = TinyLM(cfg)
    if verbose:
        b = model.param_budget()
        print(
            f"[{arm}] d_model={cfg.d_model} layers={cfg.n_layers} ffn={cfg.ffn_hidden} "
            f"core={b['core']:,} stream={b['stream']:,} table={b['table']:,} "
            f"total={b['total']:,}"
        )
    return model
