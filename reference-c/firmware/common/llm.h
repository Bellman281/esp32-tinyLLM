// Portable single-header inference for the PLE TinyLM. Same code runs on the
// host (verify against PyTorch golden) and on the ESP32-S3 (mmap'd flash). No
// dynamic arch: dims come from the model.bin header. Weights are bound in place
// from `base` by default; platforms may relocate hot tensors and override the
// output-head matvec.
//
// Matches src/model.py op-for-op: split-half RoPE, erf-GELU, SiLU-SwiGLU,
// RMSNorm(weight * x * rsqrt(mean(x^2)+eps)), tied input/output embedding, and
// the PLE input  (RMSNorm(proj(x)/sqrt(D)) + table[tok]*sqrt(P)) / sqrt(2)
// gated per layer as  x += RMSNorm(ple_proj(gelu(ple_gate(x)) * ple_l)).
#ifndef LLM_H
#define LLM_H
#include <stdint.h>
#include <math.h>
#include <string.h>

#define LLM_MAGIC 0x504C4531u
#define RMS_EPS 1e-6f

typedef struct {
  int vocab, dim, n_layers, n_heads, ffn, ple_dim, seq_len, group;
  float rope_theta;
} Cfg;

// A group-wise int4 tensor viewed in place: ragged packed nibbles (row-aligned
// to a byte) + fp16 group scales. Per-tensor group. No padding.
typedef struct {
  const uint8_t  *codes;   // rows*row_bytes, nibble = value+8, row_bytes=ceil(cols/2)
  const uint16_t *scales;  // rows*n_groups fp16
  int rows, cols, group, n_groups, row_bytes;
} QT;

// IEEE half -> float.
static inline float half2float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
  if (exp == 0) {
    if (man == 0) f = sign;
    else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400)) { man <<= 1; exp--; }
      man &= 0x3FF; f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out; memcpy(&out, &f, 4); return out;
}

typedef struct {
  Cfg c;
  QT tok_emb;             // [V, D]  (also the output head, tied)
  QT ple_model_proj;      // [L*P, D]
  const float *ple_proj_norm; // [P]
  QT ple_table;           // [V, L*P]
  const float *attn_norm[32]; // [D]
  QT qkv[32];             // [3D, D]
  QT attn_proj[32];       // [D, D]
  const float *ffn_norm[32];  // [D]
  QT gate[32], up[32], down[32];
  QT ple_gate[32];        // [P, D]
  QT ple_proj[32];        // [D, P]
  const float *ple_norm[32];  // [D]
  const float *out_norm;      // [D]
  void (*head_matvec)(const QT *, const float *, float *); // optional platform override
} Model;

// Advance a cursor over the file, binding one quant tensor. Reads the per-tensor
// group prefix, then ragged codes + fp16 scales.
static const uint8_t *bind_q(const uint8_t *p, QT *t, int rows, int cols) {
  int32_t group; memcpy(&group, p, 4); p += 4;
  t->rows = rows; t->cols = cols; t->group = group;
  t->n_groups = (cols + group - 1) / group;
  t->row_bytes = (cols + 1) / 2;
  t->codes = p;  p += (size_t)rows * t->row_bytes;
  t->scales = (const uint16_t *)p;  p += (size_t)rows * t->n_groups * 2;
  return p;
}
static const uint8_t *bind_f(const uint8_t *p, const float **t, int n) {
  *t = (const float *)p;  return p + (size_t)n * sizeof(float);
}

// Dequantize row r of a quant tensor into out[cols].
static inline void deq_row(const QT *t, int r, float *out) {
  const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
  const uint16_t *sc = t->scales + (size_t)r * t->n_groups;
  for (int gi = 0; gi < t->n_groups; gi++) {
    int begin = gi * t->group;
    int end = begin + t->group;
    if (end > t->cols) end = t->cols;
    float scale = half2float(sc[gi]);
    int j = begin;
    if ((j & 1) && j < end) {
      out[j] = (float)((row[j >> 1] >> 4) - 8) * scale;
      j++;
    }
    for (; j + 1 < end; j += 2) {
      uint8_t byte = row[j >> 1];
      out[j] = (float)((byte & 0xF) - 8) * scale;
      out[j + 1] = (float)((byte >> 4) - 8) * scale;
    }
    if (j < end) {
      uint8_t byte = row[j >> 1];
      int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
      out[j] = (float)(code - 8) * scale;
    }
  }
}

// y[row_begin:row_end] = W * x for a quant tensor W. Keeping the row range
// explicit lets platforms parallelize the large output head without changing
// any individual dot product.
static inline void matvec_q_range(const QT *t, const float *x, float *y,
                                  int row_begin, int row_end) {
  for (int r = row_begin; r < row_end; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    const uint16_t *sc = t->scales + (size_t)r * t->n_groups;
    float acc = 0.f;
    for (int gi = 0; gi < t->n_groups; gi++) {
      int begin = gi * t->group;
      int end = begin + t->group;
      if (end > t->cols) end = t->cols;
      float scale = half2float(sc[gi]);
      float group_acc = 0.f;
      int j = begin;
      if ((j & 1) && j < end) {
        group_acc += (float)((row[j >> 1] >> 4) - 8) * x[j];
        j++;
      }
      for (; j + 1 < end; j += 2) {
        uint8_t byte = row[j >> 1];
        group_acc += (float)((byte & 0xF) - 8) * x[j];
        group_acc += (float)((byte >> 4) - 8) * x[j + 1];
      }
      if (j < end) {
        uint8_t byte = row[j >> 1];
        int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
        group_acc += (float)(code - 8) * x[j];
      }
      acc += group_acc * scale;
    }
    y[r] = acc;
  }
}

// y[rows] = W * x[cols], W a quant tensor [rows, cols].
static inline void matvec_q(const QT *t, const float *x, float *y) {
  matvec_q_range(t, x, y, 0, t->rows);
}

// ---- int8-activation path (the SIMD-friendly form) -------------------------
// Quantize x to int8 once per call, then per output row do int8*int8 -> int32
// group dot products, scaled by (x_scale * group_scale). The int32 group sum is
// exactly what an S3 SIMD int8 dot instruction produces, so the device SIMD
// kernel is numerically identical to this scalar reference -- only faster. The
// ONLY approximation vs matvec_q is int8 activations, which the host golden
// measures. Split into quantize + ranged-dot so the parallel head can quantize
// once and let both cores read the shared int8 activations.
static void quantize_act(const float *x, int n, int8_t *xq, float *x_scale) {
  float xmax = 1e-8f;
  for (int j = 0; j < n; j++) { float a = fabsf(x[j]); if (a > xmax) xmax = a; }
  float inv = 127.f / xmax;
  for (int j = 0; j < n; j++) {
    int q = (int)lrintf(x[j] * inv);
    xq[j] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
  }
  *x_scale = xmax / 127.f;
}

static void matvec_q8_range(const QT *t, const int8_t *xq, float x_scale,
                            float *y, int row_begin, int row_end) {
  for (int r = row_begin; r < row_end; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    const uint16_t *sc = t->scales + (size_t)r * t->n_groups;
    float acc = 0.f;
    for (int gi = 0; gi < t->n_groups; gi++) {
      int begin = gi * t->group, end = begin + t->group;
      if (end > t->cols) end = t->cols;
      int32_t g = 0;                       // <-- the SIMD int8 dot lives here
      for (int j = begin; j < end; j++) {
        uint8_t byte = row[j >> 1];
        int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
        g += (code - 8) * (int)xq[j];
      }
      acc += (float)g * half2float(sc[gi]);
    }
    y[r] = acc * x_scale;
  }
}

static void matvec_q8(const QT *t, const float *x, float *y) {
  static int8_t xq[1024];  // max input dim across the model is 128
  float xs;
  quantize_act(x, t->cols, xq, &xs);
  matvec_q8_range(t, xq, xs, y, 0, t->rows);
}

// llm_forward dispatches through MATVEC so one flag flips the whole model.
#ifdef LLM_INT8_ACT
#define MATVEC matvec_q8
#else
#define MATVEC matvec_q
#endif

static inline void rmsnorm(const float *x, const float *w, int n, float *out) {
  float ss = 0.f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  float inv = 1.f / sqrtf(ss / n + RMS_EPS);
  for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * inv;
}
static inline float gelu(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678f)); }
static inline float silu(float x) { return x / (1.f + expf(-x)); }

// Parse header + bind all tensors. Returns 0 on ok, -1 on bad magic.
static int llm_load(const uint8_t *base, Model *m) {
  const uint8_t *p = base;
  uint32_t magic; memcpy(&magic, p, 4); p += 4;
  if (magic != LLM_MAGIC) return -1;
  int32_t hv[8]; memcpy(hv, p, 32); p += 32;
  m->c.vocab = hv[0]; m->c.dim = hv[1]; m->c.n_layers = hv[2]; m->c.n_heads = hv[3];
  m->c.ffn = hv[4]; m->c.ple_dim = hv[5]; m->c.seq_len = hv[6]; m->c.group = hv[7];
  memcpy(&m->c.rope_theta, p, 4); p += 4;
  m->head_matvec = NULL;
  int D = m->c.dim, L = m->c.n_layers, P = m->c.ple_dim, F = m->c.ffn, V = m->c.vocab;

  p = bind_q(p, &m->tok_emb, V, D);
  p = bind_q(p, &m->ple_model_proj, L * P, D);
  p = bind_f(p, &m->ple_proj_norm, P);
  p = bind_q(p, &m->ple_table, V, L * P);
  for (int i = 0; i < L; i++) {
    p = bind_f(p, &m->attn_norm[i], D);
    p = bind_q(p, &m->qkv[i], 3 * D, D);
    p = bind_q(p, &m->attn_proj[i], D, D);
    p = bind_f(p, &m->ffn_norm[i], D);
    p = bind_q(p, &m->gate[i], F, D);
    p = bind_q(p, &m->up[i], F, D);
    p = bind_q(p, &m->down[i], D, F);
    p = bind_q(p, &m->ple_gate[i], P, D);
    p = bind_q(p, &m->ple_proj[i], D, P);
    p = bind_f(p, &m->ple_norm[i], D);
  }
  p = bind_f(p, &m->out_norm, D);
  return 0;
}

// Scratch buffers, caller-allocated (host: malloc; device: PSRAM).
typedef struct {
  float *x, *h, *qkv, *att, *g1, *g2, *ple, *tmpP, *trow, *logits;
  float *scores; // [seq_len], reused by each attention head
  float *kcache, *vcache; // [L * seq_len * D]
#ifdef LLM_PROFILE
  struct {
    uint64_t input_us, attn_us, ffn_us, ple_us, head_us;
    uint32_t calls;
  } profile;
#endif
} Scratch;

#ifdef LLM_PROFILE
static void llm_profile_reset(Scratch *s) {
  memset(&s->profile, 0, sizeof(s->profile));
}
#endif

// One decode step: token at position pos -> logits[V]. KV cache persists across calls.
static void llm_forward(Model *m, int token, int pos, Scratch *s) {
  int D = m->c.dim, L = m->c.n_layers, P = m->c.ple_dim, F = m->c.ffn;
  int H = m->c.n_heads, Dh = D / H, S = m->c.seq_len;
#ifdef LLM_PROFILE
  uint64_t profile_t0 = (uint64_t)LLM_PROFILE_NOW();
#endif

  deq_row(&m->tok_emb, token, s->x);           // embedding

  // ---- per-layer input: (RMSNorm(proj(x)/sqrt(D)) + table[tok]*sqrt(P)) / sqrt(2)
  MATVEC(&m->ple_model_proj, s->x, s->tmpP); // [L*P]
  float dscale = 1.f / sqrtf((float)D);
  for (int i = 0; i < L * P; i++) s->tmpP[i] *= dscale;
  for (int l = 0; l < L; l++)
    rmsnorm(s->tmpP + l * P, m->ple_proj_norm, P, s->tmpP + l * P);
  deq_row(&m->ple_table, token, s->trow);      // [L*P]
  float sp = sqrtf((float)P), inv2 = 0.70710678f;
  for (int i = 0; i < L * P; i++)
    s->ple[i] = (s->tmpP[i] + s->trow[i] * sp) * inv2;

  // RoPE frequencies are identical across every head and layer at a position.
  // Reuse trow (dead after constructing s->ple) instead of recomputing the same
  // pow/cos/sin values L*H times.
  float *rope_c = s->trow, *rope_s = s->trow + Dh / 2;
  for (int i = 0; i < Dh / 2; i++) {
    float freq = powf(m->c.rope_theta, -2.f * i / Dh);
    rope_c[i] = cosf(pos * freq);
    rope_s[i] = sinf(pos * freq);
  }
#ifdef LLM_PROFILE
  uint64_t profile_t1 = (uint64_t)LLM_PROFILE_NOW();
  s->profile.input_us += profile_t1 - profile_t0;
#endif

  for (int l = 0; l < L; l++) {
    // ---- attention
    rmsnorm(s->x, m->attn_norm[l], D, s->h);
    MATVEC(&m->qkv[l], s->h, s->qkv);        // [3D]
    float *q = s->qkv, *k = s->qkv + D, *v = s->qkv + 2 * D;
    // split-half RoPE at position pos, per head
    for (int hh = 0; hh < H; hh++) {
      float *qh = q + hh * Dh, *kh = k + hh * Dh;
      for (int i = 0; i < Dh / 2; i++) {
        float c = rope_c[i], sn = rope_s[i];
        float q1 = qh[i], q2 = qh[i + Dh / 2];
        qh[i] = q1 * c - q2 * sn; qh[i + Dh / 2] = q2 * c + q1 * sn;
        float k1 = kh[i], k2 = kh[i + Dh / 2];
        kh[i] = k1 * c - k2 * sn; kh[i + Dh / 2] = k2 * c + k1 * sn;
      }
    }
    float *kc = s->kcache + (size_t)l * S * D, *vc = s->vcache + (size_t)l * S * D;
    memcpy(kc + (size_t)pos * D, k, D * sizeof(float));
    memcpy(vc + (size_t)pos * D, v, D * sizeof(float));
    // causal attention over 0..pos
    float scale = 1.f / sqrtf((float)Dh);
    for (int hh = 0; hh < H; hh++) {
      float *qh = q + hh * Dh;
      float *ao = s->att + hh * Dh;
      for (int i = 0; i < Dh; i++) ao[i] = 0.f;
      float maxs = -1e30f;
      // pass 1: max for stable softmax
      for (int t = 0; t <= pos; t++) {
        float *kt = kc + (size_t)t * D + hh * Dh, dot = 0.f;
        for (int i = 0; i < Dh; i++) dot += qh[i] * kt[i];
        dot *= scale;
        s->scores[t] = dot;
        if (dot > maxs) maxs = dot;
      }
      float denom = 0.f;
      for (int t = 0; t <= pos; t++) {
        float w = expf(s->scores[t] - maxs); denom += w;
        float *vt = vc + (size_t)t * D + hh * Dh;
        for (int i = 0; i < Dh; i++) ao[i] += w * vt[i];
      }
      for (int i = 0; i < Dh; i++) ao[i] /= denom;
    }
    MATVEC(&m->attn_proj[l], s->att, s->h);
    for (int i = 0; i < D; i++) s->x[i] += s->h[i];
#ifdef LLM_PROFILE
    uint64_t profile_t2 = (uint64_t)LLM_PROFILE_NOW();
    s->profile.attn_us += profile_t2 - profile_t1;
#endif

    // ---- SwiGLU FFN
    rmsnorm(s->x, m->ffn_norm[l], D, s->h);
    MATVEC(&m->gate[l], s->h, s->g1);
    MATVEC(&m->up[l], s->h, s->g2);
    for (int i = 0; i < F; i++) s->g1[i] = silu(s->g1[i]) * s->g2[i];
    MATVEC(&m->down[l], s->g1, s->h);
    for (int i = 0; i < D; i++) s->x[i] += s->h[i];
#ifdef LLM_PROFILE
    uint64_t profile_t3 = (uint64_t)LLM_PROFILE_NOW();
    s->profile.ffn_us += profile_t3 - profile_t2;
#endif

    // ---- PLE gate: x += RMSNorm(ple_proj(gelu(ple_gate(x)) * ple_l))
    MATVEC(&m->ple_gate[l], s->x, s->g2);    // [P]
    for (int i = 0; i < P; i++) s->g2[i] = gelu(s->g2[i]) * s->ple[l * P + i];
    MATVEC(&m->ple_proj[l], s->g2, s->h);    // [D]
    rmsnorm(s->h, m->ple_norm[l], D, s->h);
    for (int i = 0; i < D; i++) s->x[i] += s->h[i];
#ifdef LLM_PROFILE
    profile_t1 = (uint64_t)LLM_PROFILE_NOW();
    s->profile.ple_us += profile_t1 - profile_t3;
#endif
  }

  rmsnorm(s->x, m->out_norm, D, s->x);
  // output head (tied embedding): logits[v] = dot(tok_emb_row[v], x)
  if (m->head_matvec) m->head_matvec(&m->tok_emb, s->x, s->logits);
  else MATVEC(&m->tok_emb, s->x, s->logits);
#ifdef LLM_PROFILE
  s->profile.head_us += (uint64_t)LLM_PROFILE_NOW() - profile_t1;
  s->profile.calls++;
#endif
}

#endif
