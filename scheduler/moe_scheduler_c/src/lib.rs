//! C ABI shim for the MoE scheduler.
//!
//! 暴露给 llama.cpp (C++) 调用,所有指针/数据布局都按 C 兼容设计。
//!
//! 命名约定:`moe_*` 前缀,所有函数都是线程安全的(内部用 Mutex)。
//!
//! 头文件 `moe_scheduler.h` 是配套的 C 接口声明。

use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;

use expert_cache::ExpertCache;
use predictor::{ActivationRecord, CooccurMatrix, MatrixBuilder};
use scheduler_core::{Scheduler, SchedulerConfig, SchedulerCommand};

/// 不透明 handle,C 端只持有指针。
pub struct MoeScheduler {
    builder: MatrixBuilder,
    scheduler: Scheduler,
    n_layers: u16,
    n_experts: u16,
}

#[no_mangle]
pub extern "C" fn moe_scheduler_new(
    n_layers: u16,
    n_experts: u16,
    vram_capacity: u32,
    prefetch_k_prime: u32,
) -> *mut MoeScheduler {
    if n_layers == 0 || n_experts == 0 {
        return ptr::null_mut();
    }

    let builder = MatrixBuilder::new(n_layers, n_experts);
    // 一开始用空矩阵,后面 train 后再 swap_matrix
    let empty_matrix = builder.build_snapshot();
    let cache = Arc::new(ExpertCache::new(
        n_layers,
        n_experts,
        vram_capacity,
        vram_capacity * 4,
    ));
    let mut config = SchedulerConfig::default();
    config.prefetch_k_prime = prefetch_k_prime as usize;
    let scheduler = Scheduler::new(empty_matrix, cache, config);

    Box::into_raw(Box::new(MoeScheduler {
        builder,
        scheduler,
        n_layers,
        n_experts,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn moe_scheduler_free(s: *mut MoeScheduler) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

/// 投递一条激活观测(builder 累加 + scheduler touch + 触发预测预取)。
///
/// 返回:写入 `out_commands` 的命令数量。`commands` 数组容量由 `out_capacity` 给出。
/// 命令格式:每条 4 个 u16:
///   [0] = action (0=Prefetch, 1=Evict)
///   [1] = expert layer
///   [2] = expert id
///   [3] = source tier (0=VRAM, 1=RAM, 2=SSD, 3=HDD, 4=NotLoaded)
#[no_mangle]
pub unsafe extern "C" fn moe_scheduler_observe_and_predict(
    s: *mut MoeScheduler,
    token_idx: u32,
    layer: u16,
    expert_ids: *const u16,
    n_experts: usize,
    out_commands: *mut u16,
    out_capacity: usize,
) -> usize {
    if s.is_null() || expert_ids.is_null() {
        return 0;
    }
    let s = &mut *s;
    if n_experts == 0 || n_experts > 16 {
        return 0;
    }

    // Defensive validation: layer and expert IDs must be in range. Upstream
    // (ggml graph callback) sometimes feeds garbage values when the source
    // tensor isn't fully populated yet (e.g., first decode batch on Vulkan).
    if layer >= s.n_layers {
        return 0;
    }
    let raw_slice = std::slice::from_raw_parts(expert_ids, n_experts);
    // Filter out-of-range expert IDs in place
    let mut clean_experts = [0u16; 16];
    let mut clean_n = 0usize;
    for &e in raw_slice {
        if e < s.n_experts {
            clean_experts[clean_n] = e;
            clean_n += 1;
        }
    }
    if clean_n == 0 {
        return 0;
    }
    let experts_slice = &clean_experts[..clean_n];

    // 1) 累加到 builder(在线训练矩阵)
    let mut record = ActivationRecord {
        timestamp_ns: 0,
        token_idx,
        layer,
        n_experts_used: clean_n as u8,
        _padding: 0,
        expert_ids: [0; 16],
        expert_weights: [0.0; 16],
    };
    for (i, &e) in experts_slice.iter().enumerate() {
        record.expert_ids[i] = e;
    }
    s.builder.observe(&record);

    // 2) scheduler 触发对下一层的预测预取
    let cmds = s.scheduler.on_layer_complete(layer, experts_slice);

    // 3) 写出命令到 C 缓冲
    let mut written = 0usize;
    for cmd in &cmds {
        if written >= out_capacity { break; }
        let (action, layer, expert, from) = match cmd {
            SchedulerCommand::PrefetchToVram { expert, currently_at } => {
                (0u16, expert.layer, expert.expert, *currently_at as u16)
            }
            SchedulerCommand::EvictFromVram { expert } => {
                (1u16, expert.layer, expert.expert, 0u16)
            }
        };
        let base = written * 4;
        *out_commands.add(base + 0) = action;
        *out_commands.add(base + 1) = layer;
        *out_commands.add(base + 2) = expert;
        *out_commands.add(base + 3) = from;
        written += 1;
    }
    written
}

/// 把 builder 当前累加状态做成新 snapshot,推送给 scheduler 用作下一轮预测。
/// 调用时机:训了一段时间想"上线"新矩阵时。
#[no_mangle]
pub unsafe extern "C" fn moe_scheduler_promote_snapshot(s: *mut MoeScheduler) {
    if s.is_null() { return; }
    let s = &mut *s;
    let snapshot = s.builder.build_snapshot();
    s.scheduler.swap_matrix(snapshot);
}

/// 输出当前累计统计(填充指针指向的 u64 字段;NULL 字段被忽略)。
#[no_mangle]
pub unsafe extern "C" fn moe_scheduler_get_stats(
    s: *const MoeScheduler,
    out_total_observations: *mut u64,
    out_vram_hits: *mut u64,
    out_ram_hits: *mut u64,
    out_misses: *mut u64,
    out_total_accesses: *mut u64,
    out_n_in_vram: *mut u32,
) {
    if s.is_null() { return; }
    let s = &*s;
    let stats = s.scheduler.cache().stats();

    if !out_total_observations.is_null() {
        *out_total_observations = s.builder.total_observations();
    }
    if !out_vram_hits.is_null() { *out_vram_hits = stats.vram_hits; }
    if !out_ram_hits.is_null() { *out_ram_hits = stats.ram_hits; }
    if !out_misses.is_null() { *out_misses = stats.misses; }
    if !out_total_accesses.is_null() { *out_total_accesses = stats.total_accesses; }
    if !out_n_in_vram.is_null() { *out_n_in_vram = stats.n_in_vram; }
}

/// 返回内部尺寸,方便 C 端 sanity check
#[no_mangle]
pub unsafe extern "C" fn moe_scheduler_n_layers(s: *const MoeScheduler) -> u16 {
    if s.is_null() { return 0; }
    (*s).n_layers
}

#[no_mangle]
pub unsafe extern "C" fn moe_scheduler_n_experts(s: *const MoeScheduler) -> u16 {
    if s.is_null() { return 0; }
    (*s).n_experts
}

/// 简单握手字符串(C 端可调用确认链接成功)。
#[no_mangle]
pub extern "C" fn moe_scheduler_version() -> *const c_char {
    static VERSION: &[u8] = b"moe_scheduler_c v0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

/// 从磁盘加载预训矩阵,把它推送给 scheduler 作为活跃预测矩阵。
/// 要求矩阵的 (n_layers, n_experts) 与 scheduler 创建时一致。
/// 返回 0 成功;非 0 失败:
///   -1 = NULL 指针
///   -2 = 路径不是有效 UTF-8
///   -3 = 加载文件出错(读不到 / 格式损坏)
///   -4 = 维度不匹配
#[no_mangle]
pub unsafe extern "C" fn moe_scheduler_load_matrix(
    s: *mut MoeScheduler,
    path: *const c_char,
) -> i32 {
    if s.is_null() || path.is_null() {
        return -1;
    }
    let s = &mut *s;

    let cstr = CStr::from_ptr(path);
    let path_str = match cstr.to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };

    let matrix = match predictor::load_matrix(path_str) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[moe_scheduler] load_matrix failed: {}", e);
            return -3;
        }
    };

    if matrix.n_layers() != s.n_layers || matrix.n_experts() != s.n_experts {
        eprintln!(
            "[moe_scheduler] dimension mismatch: expected {}×{}, got {}×{}",
            s.n_layers, s.n_experts,
            matrix.n_layers(), matrix.n_experts()
        );
        return -4;
    }

    s.scheduler.swap_matrix(matrix);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_destroy() {
        let s = moe_scheduler_new(16, 64, 32, 16);
        assert!(!s.is_null());
        unsafe {
            assert_eq!(moe_scheduler_n_layers(s), 16);
            assert_eq!(moe_scheduler_n_experts(s), 64);
            moe_scheduler_free(s);
        }
    }

    #[test]
    fn observe_and_predict_writes_commands() {
        let s = moe_scheduler_new(4, 8, 4, 4);
        assert!(!s.is_null());
        let experts: [u16; 2] = [1, 2];
        let mut out = [0u16; 64];
        unsafe {
            // First observation should generate prefetch commands for layer 1
            // (cold start: returns first K=4 expert ids, all not in VRAM)
            let n = moe_scheduler_observe_and_predict(
                s, 0, 0, experts.as_ptr(), 2,
                out.as_mut_ptr(), 16,
            );
            // For cold matrix, cold start returns 4 candidates;
            // they're all NotLoaded so 4 prefetch commands
            assert!(n >= 1, "expected at least 1 prefetch command, got {}", n);

            // Each command is 4 u16: (action, layer, expert, from)
            // First command should be a Prefetch (action=0) for layer 1
            assert_eq!(out[0], 0, "first command should be Prefetch");
            assert_eq!(out[1], 1, "first command should target layer 1");

            moe_scheduler_free(s);
        }
    }

    #[test]
    fn version_string_terminated() {
        let p = moe_scheduler_version();
        assert!(!p.is_null());
        let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap();
        assert!(s.starts_with("moe_scheduler_c"));
    }
}
