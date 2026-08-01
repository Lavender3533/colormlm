// MoE Scheduler — C ABI for llama.cpp integration.
//
// This header is hand-written to match the Rust extern "C" exports in
// `moe_scheduler_c::lib`.  Keep both in sync.
//
// Threading: all functions are safe to call from multiple threads, but
// not from inside a llama.cpp callback that itself touches the same handle
// without external synchronization.

#ifndef MOE_SCHEDULER_H
#define MOE_SCHEDULER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MoeScheduler MoeScheduler;

// ─── Lifetime ──────────────────────────────────────────────────────────────

MoeScheduler* moe_scheduler_new(
    uint16_t n_layers,
    uint16_t n_experts,
    uint32_t vram_capacity,    // # of experts that fit in VRAM
    uint32_t prefetch_k_prime  // # of candidates to prefetch per call (over K)
);

void moe_scheduler_free(MoeScheduler* s);

// ─── Hot path ──────────────────────────────────────────────────────────────

// Feed one (token, layer, [expert ids]) record. Updates the matrix builder
// AND triggers next-layer prediction. Returns the number of commands written
// to `out_commands`. Each command is 4 uint16 values:
//
//   out[i*4+0] = action  (0 = Prefetch, 1 = Evict)
//   out[i*4+1] = expert layer
//   out[i*4+2] = expert id (within layer)
//   out[i*4+3] = source tier  (0=VRAM, 1=RAM, 2=SSD, 3=HDD, 4=NotLoaded)
//
// `out_capacity` is the max command count the buffer can hold (so the
// buffer must be at least 4 * out_capacity uint16s).
size_t moe_scheduler_observe_and_predict(
    MoeScheduler* s,
    uint32_t token_idx,
    uint16_t layer,
    const uint16_t* expert_ids,
    size_t n_experts,
    uint16_t* out_commands,
    size_t out_capacity
);

// Promote the current builder snapshot to be the active prediction matrix.
// Call periodically (e.g. every N tokens) to refresh predictions with new data.
void moe_scheduler_promote_snapshot(MoeScheduler* s);

// ─── Diagnostics ───────────────────────────────────────────────────────────

void moe_scheduler_get_stats(
    const MoeScheduler* s,
    uint64_t* out_total_observations,
    uint64_t* out_vram_hits,
    uint64_t* out_ram_hits,
    uint64_t* out_misses,
    uint64_t* out_total_accesses,
    uint32_t* out_n_in_vram
);

uint16_t moe_scheduler_n_layers(const MoeScheduler* s);
uint16_t moe_scheduler_n_experts(const MoeScheduler* s);

const char* moe_scheduler_version(void);

// Reserved for future: load matrix snapshot from disk. Returns 0 on success.
int32_t moe_scheduler_load_matrix(MoeScheduler* s, const char* path);

#ifdef __cplusplus
}
#endif

#endif  // MOE_SCHEDULER_H
