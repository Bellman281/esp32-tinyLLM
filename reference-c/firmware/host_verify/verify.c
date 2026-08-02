// Host-side correctness check: run the portable C inference on the golden prompt
// and compare its last-position logits to PyTorch's (dumped by src/export.py).
// If these match to fp tolerance, the C port is correct and can go to the ESP32.
#include <stdio.h>
#include <stdlib.h>
#include "../common/llm.h"

static uint8_t *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *b = malloc(*n);
  if (fread(b, 1, *n, f) != *n) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f); return b;
}

int main(int argc, char **argv) {
  const char *bin = argc > 1 ? argv[1] : "firmware/model/model.bin";
  const char *gold = argc > 2 ? argv[2] : "firmware/model/golden.txt";
  size_t n;
  uint8_t *buf = read_file(bin, &n);
  Model m;
  if (llm_load(buf, &m)) { fprintf(stderr, "bad magic\n"); return 1; }
  printf("loaded: V=%d D=%d L=%d H=%d F=%d P=%d group=%d  (%.2f MB)\n",
         m.c.vocab, m.c.dim, m.c.n_layers, m.c.n_heads, m.c.ffn, m.c.ple_dim,
         m.c.group, n / 1e6);

  int D = m.c.dim, L = m.c.n_layers, P = m.c.ple_dim, F = m.c.ffn, V = m.c.vocab, S = m.c.seq_len;
  Scratch s;
  s.x = malloc(D * 4); s.h = malloc((F > D ? F : D) * 4);
  s.qkv = malloc(3 * D * 4); s.att = malloc(D * 4);
  s.g1 = malloc(F * 4); s.g2 = malloc((P > F ? P : F) * 4);
  s.ple = malloc(L * P * 4); s.tmpP = malloc(L * P * 4); s.trow = malloc(L * P * 4);
  s.logits = malloc(V * 4);
  s.scores = malloc(S * 4);
  s.kcache = malloc((size_t)L * S * D * 4);
  s.vcache = malloc((size_t)L * S * D * 4);

  // Golden prompt + reference logits from a simple text dump.
  FILE *gf = fopen(gold, "r");
  if (!gf) { perror(gold); return 1; }
  int plen; if (fscanf(gf, "%d", &plen) != 1) return 1;
  int *prompt = malloc(plen * sizeof(int));
  for (int i = 0; i < plen; i++) if (fscanf(gf, "%d", &prompt[i]) != 1) return 1;
  float *ref = malloc(V * sizeof(float));
  for (int i = 0; i < V; i++) if (fscanf(gf, "%f", &ref[i]) != 1) return 1;
  fclose(gf);

  for (int pos = 0; pos < plen; pos++) llm_forward(&m, prompt[pos], pos, &s);

  // compare last-position logits
  double maxabs = 0, sum2 = 0;
  int c_top = 0, r_top = 0;
  for (int i = 0; i < V; i++) {
    double d = (double)s.logits[i] - ref[i];
    if (fabs(d) > maxabs) maxabs = fabs(d);
    sum2 += d * d;
    if (s.logits[i] > s.logits[c_top]) c_top = i;
    if (ref[i] > ref[r_top]) r_top = i;
  }
  printf("sample logits (idx: C vs ref):\n");
  int probe[8] = {265, 14, 1, 12, 13, 100, 5000, 20000};
  for (int i = 0; i < 8; i++)
    printf("  [%5d]  C=%8.4f  ref=%8.4f\n", probe[i], s.logits[probe[i]], ref[probe[i]]);
  printf("logits: C top=%d  PyTorch top=%d\n", c_top, r_top);
  printf("max abs diff = %.5f   rms diff = %.6f\n", maxabs, sqrt(sum2 / V));
  printf(maxabs < 0.02 ? "PASS: C matches PyTorch golden\n"
                       : "FAIL: numerics diverge\n");
  return maxabs < 0.02 ? 0 : 2;
}
