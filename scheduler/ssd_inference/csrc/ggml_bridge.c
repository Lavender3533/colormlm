// Minimal bridge: exposes ggml's Q4K dot + Q8K quantize for direct Rust FFI.
// Compiled via cc crate, linked statically. Zero ggml context overhead.

#include <stdint.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>

// ---- Q8_K block (from ggml) ----
typedef struct {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
} block_q8_K;

// ---- Q4_K block (from ggml) ----
typedef struct {
    uint16_t d;      // f16 super-block scale
    uint16_t dmin;   // f16 super-block min
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

// f16 → f32
static inline float fp16_to_fp32(uint16_t h) {
    union { uint32_t u; float f; } u;
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { u.u = sign << 31; return u.f; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= 0x3FF;
        u.u = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    } else if (exp == 31) {
        u.u = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        u.u = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    return u.f;
}

// Quantize fp32 row to Q8_K
void bridge_quantize_row_q8_K(const float *x, block_q8_K *y, int k) {
    int nb = k / 256;
    for (int i = 0; i < nb; i++) {
        float max = 0;
        for (int j = 0; j < 256; j++) {
            float ax = fabsf(x[i*256+j]);
            if (ax > max) max = ax;
        }
        float d = max / 127.0f;
        float id = (d != 0) ? 1.0f/d : 0.0f;
        y[i].d = d;
        for (int j = 0; j < 256; j++) {
            float v = x[i*256+j] * id;
            y[i].qs[j] = (int8_t)(roundf(v < -128 ? -128 : v > 127 ? 127 : v));
        }
        for (int j = 0; j < 16; j++) {
            int16_t sum = 0;
            for (int l = 0; l < 16; l++) sum += y[i].qs[j*16+l];
            y[i].bsums[j] = sum;
        }
    }
}

// Horizontal sum of __m256
static inline float hsum_float_8(__m256 x) {
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 lo = _mm256_castps256_ps128(x);
    lo = _mm_add_ps(lo, hi);
    hi = _mm_movehl_ps(lo, lo);
    lo = _mm_add_ps(lo, hi);
    hi = _mm_movehdup_ps(lo);
    lo = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(lo);
}

// Scale shuffle masks for Q4K (from ggml)
static const uint8_t k_shuffle[256] = {
    0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
    2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,
    4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,
    6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,
    8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
};

// Q4K × Q8K dot product — exact copy of ggml's AVX2 path
void bridge_vec_dot_q4_K_q8_K(int n, float *s, const void *vx, const void *vy) {
    const block_q4_K *x = (const block_q4_K *)vx;
    const block_q8_K *y = (const block_q8_K *)vy;
    int nb = n / 256;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t *q4 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
            _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(
            _mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);
        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < 4; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales,
                _mm256_loadu_si256((const __m256i*)(k_shuffle + 64*j)));
            const __m256i scale_h = _mm256_shuffle_epi8(scales,
                _mm256_loadu_si256((const __m256i*)(k_shuffle + 64*j + 32)));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + 32*j));
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)(q8 + 64*j));
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)(q8 + 64*j + 32));
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

// Matvec: y[n] = dot(W_q4k[n,:], x_q8[:])  for n = start..end
// Called from multiple threads for row-parallel execution.
void bridge_matvec_q4k_chunk(
    const void *w_q,    // Q4K weight bytes [n_out × bpr × 144]
    const void *x_q8,   // Q8K quantized x [bpr blocks]
    float *y,           // output [chunk_size]
    int n_start,        // first output row
    int n_end,          // last+1 output row
    int n_in            // input dimension
) {
    int bpr = n_in / 256;
    int row_bytes = bpr * sizeof(block_q4_K);
    for (int n = n_start; n < n_end; n++) {
        const void *row = (const char*)w_q + (size_t)n * row_bytes;
        bridge_vec_dot_q4_K_q8_K(n_in, &y[n - n_start], row, x_q8);
    }
}
