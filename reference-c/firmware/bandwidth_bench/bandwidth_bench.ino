// ESP32-S3 memory-hierarchy bandwidth benchmark
// ---------------------------------------------------------------------------
// The one measurement the whole PLE-on-MCU idea rests on. Everything in the
// repo's RESULTS.md is Mac perplexity; this turns the estimated tok/s into a
// real one by measuring what a token actually costs to fetch on this chip.
//
// Per token, our deploy model (RESULTS.md) reads:
//   HEAD   ~1.5 MB, ONE SEQUENTIAL scan   -> bounded by streaming bandwidth
//   TABLE  ~450 B, 6 RANDOM row lookups   -> bounded by random-read LATENCY
//   CORE   0 (SRAM-resident)
// So we measure three things and combine them into a bandwidth-only tok/s ceiling:
//   1. PSRAM sequential read bandwidth      (where the head could live)
//   2. Internal-SRAM sequential bandwidth   (upper bound / sanity)
//   3. Memory-mapped flash RANDOM-read latency  <-- THE open question
//
// Timing uses the Xtensa cycle counter (esp_cpu_get_cycle_count) for
// cycle-accurate numbers, converted with the actual CPU frequency. We reach
// past Arduino into IDF (heap_caps, esp_partition, cache) so the numbers are
// legitimate, not abstraction-blurred.

#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_cpu.h"
#include "esp_clk_tree.h"
#include "spi_flash_mmap.h"

static uint32_t cpu_hz;

static inline uint32_t ccount() { return esp_cpu_get_cycle_count(); }
static double cyc_to_us(uint64_t c) { return (double)c * 1e6 / cpu_hz; }

// Prevent the compiler from optimising away read loops.
static volatile uint32_t g_sink;

// ---- sequential read bandwidth of a buffer ---------------------------------
static double seq_read_MBps(volatile uint32_t *buf, size_t bytes, int reps) {
  size_t n = bytes / 4;
  uint64_t best = UINT64_MAX;
  for (int r = 0; r < reps; r++) {
    uint32_t acc = 0;
    uint32_t t0 = ccount();
    for (size_t i = 0; i < n; i++) acc += buf[i];
    uint32_t t1 = ccount();
    g_sink = acc;
    uint64_t dc = (uint32_t)(t1 - t0);
    if (dc < best) best = dc;   // best-of-reps = least interrupted
  }
  double us = cyc_to_us(best);
  return (double)bytes / us;    // bytes/us == MB/s
}

// ---- flash random-read latency ---------------------------------------------
// Map a flash partition, then read one cache-line-sized row from pseudo-random
// offsets, defeating the XIP cache the way a per-token table lookup would.
static void flash_random_latency(const void *base, size_t span,
                                 size_t row_bytes, int n_reads,
                                 double *us_per_read, double *eff_MBps) {
  const uint8_t *p = (const uint8_t *)base;
  size_t mask_span = span - row_bytes;
  uint32_t rng = 0xC0FFEE;
  uint32_t acc = 0;

  // Warm nothing on purpose: we want cold-ish random access.
  uint32_t t0 = ccount();
  for (int i = 0; i < n_reads; i++) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;   // xorshift
    size_t off = (rng % mask_span) & ~(size_t)3;
    const volatile uint32_t *row = (const volatile uint32_t *)(p + off);
    for (size_t b = 0; b < row_bytes / 4; b++) acc += row[b];
  }
  uint32_t t1 = ccount();
  g_sink = acc;
  double us = cyc_to_us((uint32_t)(t1 - t0));
  *us_per_read = us / n_reads;
  *eff_MBps = (double)((uint64_t)n_reads * row_bytes) / us;
}

void setup() {
  Serial.begin(115200);
  esp_clk_tree_src_get_freq_hz(SOC_MOD_CLK_CPU,
      ESP_CLK_TREE_SRC_FREQ_PRECISION_CACHED, &cpu_hz);
}

// Re-run on a loop so a serial monitor connecting at any time catches a full run
// (native USB loses anything printed before the host opens the port).
void runBench() {
  Serial.println("\n===== ESP32-S3 memory bandwidth benchmark =====");
  Serial.printf("CPU: %.0f MHz\n", cpu_hz / 1e6);
  Serial.printf("PSRAM total: %u KB  free: %u KB\n",
      heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024,
      heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);
  Serial.printf("Internal free: %u KB\n",
      heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024);

  // --- 1. PSRAM sequential bandwidth (candidate home of the streamed head) ---
  const size_t PS_BYTES = 1024 * 1024;  // 1MB, ~ head size
  uint32_t *ps = (uint32_t *)heap_caps_malloc(PS_BYTES, MALLOC_CAP_SPIRAM);
  if (ps) {
    for (size_t i = 0; i < PS_BYTES / 4; i++) ps[i] = i * 2654435761u;
    double bw = seq_read_MBps(ps, PS_BYTES, 8);
    Serial.printf("\n[1] PSRAM  seq read : %7.1f MB/s  (1MB head scan = %.2f ms)\n",
        bw, PS_BYTES / bw / 1000.0);
    heap_caps_free(ps);
  } else Serial.println("[1] PSRAM alloc FAILED");

  // --- 2. Internal SRAM sequential bandwidth (upper bound) --------------------
  const size_t IS_BYTES = 128 * 1024;
  uint32_t *is = (uint32_t *)heap_caps_malloc(IS_BYTES, MALLOC_CAP_INTERNAL);
  if (is) {
    for (size_t i = 0; i < IS_BYTES / 4; i++) is[i] = i;
    double bw = seq_read_MBps(is, IS_BYTES, 16);
    Serial.printf("[2] SRAM   seq read : %7.1f MB/s\n", bw);
    heap_caps_free(is);
  } else Serial.println("[2] SRAM alloc FAILED");

  // --- 3. Flash random-read latency (THE open question) -----------------------
  // Map the largest data partition and probe random rows sized like a PLE table
  // row (6 layers * 128 dim * 4B fp = 3072B; and a 4-bit-ish 512B variant).
  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_ANY, ESP_PARTITION_SUBTYPE_ANY, NULL);
  // Prefer a big app/data partition to get a wide random span.
  esp_partition_iterator_t it = esp_partition_find(
      ESP_PARTITION_TYPE_APP, ESP_PARTITION_SUBTYPE_ANY, NULL);
  const esp_partition_t *big = part;
  for (; it != NULL; it = esp_partition_next(it)) {
    const esp_partition_t *c = esp_partition_get(it);
    if (c->size > big->size) big = c;
  }
  Serial.printf("\n[3] flash partition '%s' size %u KB @ 0x%06x\n",
      big->label, big->size / 1024, big->address);

  const void *mmap_ptr;
  spi_flash_mmap_handle_t h;
  size_t span = big->size < (2 << 20) ? big->size : (2 << 20);
  esp_err_t err = spi_flash_mmap(big->address, span,
      SPI_FLASH_MMAP_DATA, &mmap_ptr, &h);
  if (err == ESP_OK) {
    double us, mbps;
    for (size_t row : {512u, 3072u}) {
      flash_random_latency(mmap_ptr, span, row, 4000, &us, &mbps);
      Serial.printf("    random row %4u B : %6.2f us/read  (%.1f MB/s eff)\n",
          (unsigned)row, us, mbps);
    }
    // The number that matters: 6 random table rows per token.
    flash_random_latency(mmap_ptr, span, 512, 6000, &us, &mbps);
    Serial.printf("    -> 6 table rows/token ~= %.1f us (table cost per token)\n",
        us * 6);
    spi_flash_munmap(h);
  } else {
    Serial.printf("    mmap FAILED err=%d\n", err);
  }

  // --- verdict ----------------------------------------------------------------
  Serial.println("\n===== interpretation =====");
  Serial.println("head_ms = 1.5MB / PSRAM_MBps ; table_us = 6 * row_latency");
  Serial.println("tok/s ceiling ~= 1000 / (head_ms + table_us/1000)");
  Serial.println("If table_us << head_ms, the PLE table really is ~free. That");
  Serial.println("is the whole bet. Numbers above decide it.");
  Serial.println("\n(done)");
}

void loop() {
  runBench();
  delay(8000);
}
