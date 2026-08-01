//! 存储层级与专家 ID 类型。

use std::fmt;

/// 存储层级,从最快到最慢
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Tier {
    /// GPU 显存 — 最快,最贵
    Vram = 0,
    /// 系统 RAM — 中速,中等容量
    Ram = 1,
    /// NVMe SSD — 慢但容量大
    Ssd = 2,
    /// HDD — 最慢,容量大
    Hdd = 3,
    /// 未加载(磁盘上的原始权重还在,但当前不知道)
    NotLoaded = 4,
}

impl Tier {
    /// 这个层级访问数据的相对延迟(用于排序)。Vram=1, Ram=20, Ssd=10000, Hdd=1000000。
    pub fn relative_latency(self) -> u64 {
        match self {
            Tier::Vram => 1,
            Tier::Ram => 20,
            Tier::Ssd => 10_000,
            Tier::Hdd => 1_000_000,
            Tier::NotLoaded => u64::MAX,
        }
    }

    /// 这个层级是否被认为"立即可用"(无需异步加载)
    pub fn is_resident(self) -> bool {
        matches!(self, Tier::Vram | Tier::Ram)
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Tier::Vram => "VRAM",
            Tier::Ram => "RAM",
            Tier::Ssd => "SSD",
            Tier::Hdd => "HDD",
            Tier::NotLoaded => "NotLoaded",
        };
        f.write_str(s)
    }
}

/// 专家在模型里的全局唯一标识。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertId {
    pub layer: u16,
    pub expert: u16,
}

impl ExpertId {
    pub const fn new(layer: u16, expert: u16) -> Self {
        Self { layer, expert }
    }

    /// 用于稠密索引(layer * n_experts + expert)。
    /// 调用方需保证 expert 编号在范围内。
    pub fn dense_index(self, n_experts_per_layer: u16) -> usize {
        self.layer as usize * n_experts_per_layer as usize + self.expert as usize
    }
}

impl fmt::Display for ExpertId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}E{}", self.layer, self.expert)
    }
}
