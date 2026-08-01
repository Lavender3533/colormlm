//! Router Predictor — MoE 共现矩阵
//!
//! 核心:给定当前层激活的专家集合,预测下一层最可能激活的专家。
//!
//! 三层结构:
//! - [`record`]:从 llama.cpp hook 来的原始激活记录
//! - [`matrix`]:只读快照,推理热路径用
//! - [`builder`]:原子累加器,后台收数据
//!
//! 双缓冲通过 [`arc_swap::ArcSwap`] 实现,见模块文档。

pub mod record;
pub mod matrix;
pub mod builder;
pub mod format;

pub use record::ActivationRecord;
pub use matrix::CooccurMatrix;
pub use builder::MatrixBuilder;
pub use format::{save as save_matrix, load as load_matrix, FormatError};

/// 矩阵的活跃只读句柄,无锁读
pub type ActiveMatrix = arc_swap::ArcSwap<CooccurMatrix>;
