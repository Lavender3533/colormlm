//! Expert weight tiered cache.
//!
//! 抽象出"专家权重在哪一层存储介质"的状态管理,**不**实际加载/卸载权重
//! (那是 backend 实现的事 — Vulkan/CPU/etc.)。
//!
//! 角色:
//! - `Tier`:存储层级
//! - `ExpertId`:专家身份(layer + expert_within_layer)
//! - `ExpertCache`:维护 (ExpertId → Tier),提供 LRU 淘汰、容量约束、查询
//!
//! 调度器(scheduler-core)用这个来:
//! 1. 查"专家 X 当前在哪"(决定要不要预取)
//! 2. "请把专家 X 升到 VRAM"(请求 backend 搬运)
//! 3. "VRAM 满了,淘汰最冷的 K 个"(请求 backend 卸载)

pub mod tier;
pub mod cache;

pub use tier::{Tier, ExpertId};
pub use cache::{ExpertCache, CacheEvent, CacheStats};
