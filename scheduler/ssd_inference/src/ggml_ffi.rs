//! FFI bridge to ggml — persistent context for repeated matmul.

use std::ffi::c_void;

pub const GGML_TYPE_F32: i32 = 0;
pub const GGML_TYPE_Q4_K: i32 = 12;

#[repr(C)]
pub struct GgmlInitParams {
    pub mem_size: usize,
    pub mem_buffer: *mut c_void,
    pub no_alloc: bool,
}

pub enum GgmlContext {}
pub enum GgmlTensor {}
pub enum GgmlCGraph {}
pub enum GgmlBackend {}
pub enum GgmlBackendBuffer {}

#[link(name = "ggml-base")]
extern "C" {
    fn ggml_init(params: GgmlInitParams) -> *mut GgmlContext;
    fn ggml_free(ctx: *mut GgmlContext);
    fn ggml_new_tensor_2d(ctx: *mut GgmlContext, type_: i32, ne0: i64, ne1: i64) -> *mut GgmlTensor;
    fn ggml_mul_mat(ctx: *mut GgmlContext, a: *mut GgmlTensor, b: *mut GgmlTensor) -> *mut GgmlTensor;
    fn ggml_new_graph(ctx: *mut GgmlContext) -> *mut GgmlCGraph;
    fn ggml_build_forward_expand(graph: *mut GgmlCGraph, tensor: *mut GgmlTensor);
    fn ggml_get_data(tensor: *const GgmlTensor) -> *mut c_void;
    fn ggml_backend_alloc_ctx_tensors(ctx: *mut GgmlContext, backend: *mut GgmlBackend) -> *mut GgmlBackendBuffer;
    fn ggml_backend_buffer_free(buffer: *mut GgmlBackendBuffer);
}

#[link(name = "ggml-cpu")]
extern "C" {
    fn ggml_backend_cpu_init() -> *mut GgmlBackend;
    fn ggml_backend_cpu_set_n_threads(backend: *mut GgmlBackend, n_threads: i32);
    fn ggml_graph_compute_with_ctx(ctx: *mut GgmlContext, graph: *mut GgmlCGraph, n_threads: i32) -> i32;
    fn ggml_backend_free(backend: *mut GgmlBackend);
}

/// Persistent ggml compute engine for MoE expert matmul.
/// Created once, reused every layer. Avoids per-call context/backend alloc.
pub struct GgmlMoe {
    backend: *mut GgmlBackend,
    n_threads: i32,
}

unsafe impl Send for GgmlMoe {}

impl GgmlMoe {
    pub fn new(n_threads: i32) -> Self {
        let backend = unsafe {
            let b = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(b, n_threads);
            b
        };
        Self { backend, n_threads }
    }

    /// Q4K matvec: y = W @ x where W is [n_out, n_in] Q4K, x is [n_in] fp32.
    pub fn matvec_q4k(&self, w_q: &[u8], x: &[f32], n_out: usize, n_in: usize) -> Vec<f32> {
        unsafe {
            let ctx = ggml_init(GgmlInitParams {
                mem_size: 256 * 1024,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            });
            let w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, n_in as i64, n_out as i64);
            let xv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in as i64, 1);
            let result = ggml_mul_mat(ctx, w, xv);

            let buffer = ggml_backend_alloc_ctx_tensors(ctx, self.backend);

            std::ptr::copy_nonoverlapping(w_q.as_ptr(), ggml_get_data(w) as *mut u8, w_q.len());
            std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, ggml_get_data(xv) as *mut u8, n_in * 4);

            let graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, result);
            ggml_graph_compute_with_ctx(ctx, graph, self.n_threads);

            let mut y = vec![0f32; n_out];
            std::ptr::copy_nonoverlapping(ggml_get_data(result) as *const f32, y.as_mut_ptr(), n_out);

            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            y
        }
    }
}

impl Drop for GgmlMoe {
    fn drop(&mut self) {
        unsafe { ggml_backend_free(self.backend); }
    }
}
