//! FFI bindings to ggml_bridge.c — AVX2 Q4K matvec with zero ggml overhead.

#[repr(C)]
struct BlockQ8K {
    d: f32,
    qs: [i8; 256],
    bsums: [i16; 16],
}

extern "C" {
    fn bridge_quantize_row_q8_K(x: *const f32, y: *mut BlockQ8K, k: i32);
    fn bridge_matvec_q4k_chunk(
        w_q: *const u8,
        x_q8: *const BlockQ8K,
        y: *mut f32,
        n_start: i32,
        n_end: i32,
        n_in: i32,
    );
}

const N_THREADS: usize = 8;

/// Q4K matvec: out[n_out] = W_q4k[n_out, n_in] × x_fp32[n_in]
///
/// Uses 8 threads with std::thread::scope (no rayon overhead).
/// x is quantized to Q8K once, then all threads share it.
pub fn cpu_matvec_q4k(
    weight_q4k: &[u8],
    x_fp32: &[f32],
    out: &mut [f32],
    n_in: usize,
    n_out: usize,
) {
    assert_eq!(x_fp32.len(), n_in);
    assert!(out.len() >= n_out);

    let n_blocks = n_in / 256;
    let mut x_q8_storage = vec![0u8; n_blocks * std::mem::size_of::<BlockQ8K>()];
    let x_q8 = x_q8_storage.as_mut_ptr() as *mut BlockQ8K;
    unsafe {
        bridge_quantize_row_q8_K(x_fp32.as_ptr(), x_q8, n_in as i32);
    }

    let chunk_size = (n_out + N_THREADS - 1) / N_THREADS;

    // Pack pointers into usize for Send-safe transfer to threads
    let w_addr = weight_q4k.as_ptr() as usize;
    let q8_addr = x_q8 as usize;
    let out_addr = out.as_mut_ptr() as usize;

    std::thread::scope(|s| {
        for t in 0..N_THREADS {
            let start = t * chunk_size;
            if start >= n_out {
                break;
            }
            let end = (start + chunk_size).min(n_out);
            s.spawn(move || unsafe {
                bridge_matvec_q4k_chunk(
                    w_addr as *const u8,
                    q8_addr as *const BlockQ8K,
                    (out_addr + start * 4) as *mut f32,
                    start as i32,
                    end as i32,
                    n_in as i32,
                );
            });
        }
    });
}
