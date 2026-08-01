//! ggml-grade AVX2 integer SIMD Q4_K × Q8 dot product.
//! Direct translation of ggml's `ggml_vec_dot_q4_K_q8_K` inner loop.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Scale shuffle masks — broadcast scale[i] to all 16 lanes of i16
#[cfg(target_arch = "x86_64")]
static SCALE_SHUFFLE_K4: [[u8; 32]; 8] = [
    [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1, 0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
    [2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3, 2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3],
    [4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5, 4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5],
    [6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7, 6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7],
    [8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9, 8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9],
    [10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11, 10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11],
    [12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13, 12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13],
    [14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15, 14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15],
];

#[derive(Clone, Copy)]
struct SendPtr(*const u8);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

pub struct Q8Block {
    pub qs: [i8; 256],
    pub d: f32,
    pub bsums: [i16; 16], // per-16-weight sums (matches ggml block_q8_K)
}

pub fn quantize_x_q8(x: &[f32]) -> Vec<Q8Block> {
    let nb = x.len() / 256;
    let mut out = Vec::with_capacity(nb);
    for b in 0..nb {
        let xb = &x[b * 256..(b + 1) * 256];
        let max_abs = xb.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let d = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let id = 1.0 / d;
        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 16];
        for i in 0..256 {
            let v = (xb[i] * id).round().clamp(-128.0, 127.0) as i8;
            qs[i] = v;
            bsums[i / 16] += v as i16;
        }
        out.push(Q8Block { qs, d, bsums });
    }
    out
}

#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let s = (h >> 15) & 1;
    let e = (h >> 10) & 0x1F;
    let m = h & 0x3FF;
    if e == 0 {
        if m == 0 { return f32::from_bits((s as u32) << 31); }
        let mut e2: i32 = -14;
        let mut m2 = m as u32;
        while (m2 & 0x400) == 0 { m2 <<= 1; e2 -= 1; }
        m2 &= 0x3FF;
        f32::from_bits(((s as u32) << 31) | (((e2 + 127) as u32) << 23) | (m2 << 13))
    } else if e == 0x1F {
        f32::from_bits(((s as u32) << 31) | (0xFF << 23) | ((m as u32) << 13))
    } else {
        f32::from_bits(((s as u32) << 31) | ((e as u32 - 15 + 127) << 23) | ((m as u32) << 13))
    }
}

/// ggml-grade Q4K×Q8 block dot product. Direct translation of ggml's AVX2 path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn dot_q4k_q8_ggml(q4_blk: &[u8], q8: &Q8Block) -> f32 {
    let d_q4  = f16_to_f32(u16::from_le_bytes([q4_blk[0], q4_blk[1]]));
    let dmin  = f16_to_f32(u16::from_le_bytes([q4_blk[2], q4_blk[3]]));
    let d = q8.d * d_q4;
    let dmin_neg = -q8.d * dmin;

    // Decode scales using ggml's bit manipulation
    let kmask1: u32 = 0x3f3f3f3f;
    let kmask2: u32 = 0x0f0f0f0f;
    let kmask3: u32 = 0x03030303;
    let mut utmp = [0u32; 4];
    std::ptr::copy_nonoverlapping(q4_blk[4..].as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    let uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    let mins_and_scales = _mm256_cvtepu8_epi16(
        _mm_set_epi32(utmp[3] as i32, utmp[2] as i32, utmp[1] as i32, utmp[0] as i32));

    // Mins correction: dmin * sum(mins[i] * bsums_pairs[i])
    // bsums has 16 i16 values. Load as __m256i, hadd to get 8 pairs, then madd with mins.
    let q8sums = _mm256_loadu_si256(q8.bsums.as_ptr() as *const __m256i);
    let q8s = _mm_hadd_epi16(
        _mm256_extracti128_si256(q8sums, 0),
        _mm256_extracti128_si256(q8sums, 1));
    let mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
    let prod = _mm_madd_epi16(mins128, q8s);
    let min_sum = {
        let mut tmp = [0i32; 4];
        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, prod);
        tmp.iter().sum::<i32>() as f32
    };
    let acc_min = dmin_neg * min_sum;

    // Scale broadcast from low 128 bits
    let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
    let scales = _mm256_set_m128i(sc128, sc128);

    let m4 = _mm256_set1_epi8(0x0F);
    let mut sumi = _mm256_setzero_si256();
    let q4 = &q4_blk[16..144];

    for j in 0..4u32 {
        let shuf_l = _mm256_loadu_si256(SCALE_SHUFFLE_K4[2 * j as usize].as_ptr() as *const __m256i);
        let shuf_h = _mm256_loadu_si256(SCALE_SHUFFLE_K4[2 * j as usize + 1].as_ptr() as *const __m256i);
        let scale_l = _mm256_shuffle_epi8(scales, shuf_l);
        let scale_h = _mm256_shuffle_epi8(scales, shuf_h);

        let q4bits = _mm256_loadu_si256(q4.as_ptr().add(j as usize * 32) as *const __m256i);
        let q4l = _mm256_and_si256(q4bits, m4);
        let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

        let q8l = _mm256_loadu_si256(q8.qs.as_ptr().add(j as usize * 64) as *const __m256i);
        let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
        p16l = _mm256_madd_epi16(scale_l, p16l); // fused scale × dot → i32

        let q8h = _mm256_loadu_si256(q8.qs.as_ptr().add(j as usize * 64 + 32) as *const __m256i);
        let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
        p16h = _mm256_madd_epi16(scale_h, p16h);

        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
    }

    // Horizontal i32 sum
    let mut tmp = [0i32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, sumi);
    let isum: i32 = tmp.iter().sum();

    d * isum as f32 + acc_min
}

/// Public wrapper for use in par_chunks. Caller ensures AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn dot_q4k_q8_pub(q4_blk: &[u8], q8: &Q8Block) -> f32 {
    dot_q4k_q8_ggml(q4_blk, q8)
}

/// ggml-grade Q4K×Q8 matvec with pre-quantized Q8 x.
#[cfg(target_arch = "x86_64")]
pub fn matvec_q4k_preq8(w_q: &[u8], q8_x: &[Q8Block], n_out: usize, n_in: usize) -> Vec<f32> {
    let bpr = n_in / 256;
    let row_bytes = bpr * 144;
    let mut y = vec![0f32; n_out];
    for n in 0..n_out {
        let row = &w_q[n * row_bytes..(n + 1) * row_bytes];
        let mut acc = 0f32;
        for b in 0..bpr {
            acc += unsafe { dot_q4k_q8_ggml(&row[b * 144..(b + 1) * 144], &q8_x[b]) };
        }
        y[n] = acc;
    }
    y
}

/// Single-row dot for argmax (lm_head).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub unsafe fn dot_q4k_row_preq8(row_q: &[u8], q8_x: &[Q8Block]) -> f32 {
    let bpr = q8_x.len();
    let mut acc = 0f32;
    for b in 0..bpr {
        acc += dot_q4k_q8_ggml(&row_q[b * 144..(b + 1) * 144], &q8_x[b]);
    }
    acc
}

/// Batch matvec: process multiple experts' matmul in ONE rayon dispatch.
/// Equivalent to ggml's `mul_mat_id` — all threads process all experts' rows.
/// expert_ptrs: [n_experts] pointers to Q4K byte arrays (each is one expert weight)
/// q8_x: shared pre-quantized input
/// y: output [n_experts × n_out_per_expert]
#[cfg(target_arch = "x86_64")]
pub fn batch_matvec_q4k(
    expert_ptrs: &[usize],  // raw pointers as usize (Send+Sync)
    q8_x: &[Q8Block],
    n_out_per: usize,
    n_in: usize,
    y: &mut [f32],
) {
    use rayon::prelude::*;
    let bpr = n_in / 256;
    let row_bytes = bpr * 144;
    let n_exp = expert_ptrs.len();
    let total = n_exp * n_out_per;
    let chunk_sz = (total + 7) / 8;

    y.par_chunks_mut(chunk_sz).enumerate().for_each(|(ci, chunk)| {
        let base = ci * chunk_sz;
        for (li, val) in chunk.iter_mut().enumerate() {
            let global = base + li;
            if global >= total { break; }
            let ei = global / n_out_per;
            let row_idx = global % n_out_per;
            let w_ptr = expert_ptrs[ei];
            let row = unsafe {
                std::slice::from_raw_parts((w_ptr + row_idx * row_bytes) as *const u8, row_bytes)
            };
            let mut acc = 0f32;
            for b in 0..bpr {
                acc += unsafe { dot_q4k_q8_ggml(&row[b * 144..(b + 1) * 144], &q8_x[b]) };
            }
            *val = acc;
        }
    });
}

/// Original fp32 entry point (quantizes x internally).
#[cfg(target_arch = "x86_64")]
pub fn matvec_q4k_q8_avx2(w_q: &[u8], x: &[f32], n_out: usize, n_in: usize) -> Vec<f32> {
    let q8_x = quantize_x_q8(x);
    unsafe { matvec_q4k_preq8(w_q, &q8_x, n_out, n_in) }
}
