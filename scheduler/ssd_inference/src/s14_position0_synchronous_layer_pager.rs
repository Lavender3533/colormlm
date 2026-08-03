//! FullDepth43 position0 的第一版同步逐层权重换页状态机。
//!
//! 本模块刻意不拥有 Vulkan 对象。它把第一版正确性路径收敛为严格的
//! `wait(previous compute) -> upload(current weights) -> reconfigure descriptors
//! -> submit(current compute)`。hidden 只以 GPU workspace A/B 槽位出现，API
//! 没有 host slice、readback 或 D2H 回调，因此适配器不能借换页把 hidden 拉回主机。
//!
//! 生产双页异步流水可以在该合同通过后替换同步 backend；本状态机本身不宣称性能。

use anyhow::{anyhow, bail, Result};
use std::{collections::HashSet, path::PathBuf};

pub const S14_POSITION0_SYNCHRONOUS_LAYER_COUNT: u8 = 43;
pub const S14_POSITION0_SYNCHRONOUS_BANKS: usize = 2;
pub const S14_POSITION0_SYNCHRONOUS_WEIGHT_ALIGNMENT: u64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14Position0DeviceHiddenSlot {
    A,
    B,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0DeviceHiddenBinding {
    pub input: S14Position0DeviceHiddenSlot,
    pub output: S14Position0DeviceHiddenSlot,
}

impl S14Position0DeviceHiddenBinding {
    pub fn for_layer(layer: u8) -> Result<Self> {
        if layer >= S14_POSITION0_SYNCHRONOUS_LAYER_COUNT {
            bail!("position0 hidden binding layer 越界: L{layer}");
        }
        Ok(if layer % 2 == 0 {
            Self {
                input: S14Position0DeviceHiddenSlot::A,
                output: S14Position0DeviceHiddenSlot::B,
            }
        } else {
            Self {
                input: S14Position0DeviceHiddenSlot::B,
                output: S14Position0DeviceHiddenSlot::A,
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum S14Position0StaticPage {
    Resident { layer: u8 },
    Streamed { bank: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum S14Position0WeightPageTarget {
    Static(S14Position0StaticPage),
    Routed { bank: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0WeightUpload {
    pub tensor: String,
    pub kind: String,
    pub expert_id: Option<u16>,
    pub source_path: PathBuf,
    pub sha256: String,
    pub destination_offset: u64,
    pub bytes: u64,
    pub target: S14Position0WeightPageTarget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0SynchronousLayerPlan {
    pub layer: u8,
    pub static_page: S14Position0StaticPage,
    pub static_page_bytes: u64,
    pub routed_bank: usize,
    pub routed_page_bytes: u64,
    /// 同时承担 descriptor 子范围合同。Resident 页不会在 token 内重复上传。
    pub static_weights: Vec<S14Position0WeightUpload>,
    pub routed_weights: Vec<S14Position0WeightUpload>,
    pub hidden: S14Position0DeviceHiddenBinding,
}

impl S14Position0SynchronousLayerPlan {
    pub fn validate(&self) -> Result<()> {
        if self.layer >= S14_POSITION0_SYNCHRONOUS_LAYER_COUNT {
            bail!("position0 synchronous layer 越界: L{}", self.layer);
        }
        let expected_bank = self.layer as usize % S14_POSITION0_SYNCHRONOUS_BANKS;
        if self.routed_bank != expected_bank {
            bail!(
                "L{} routed bank 漂移: actual={} expected={expected_bank}",
                self.layer,
                self.routed_bank
            );
        }
        match self.static_page {
            S14Position0StaticPage::Resident { layer } if layer == self.layer => {}
            S14Position0StaticPage::Streamed { bank } if bank == expected_bank => {}
            _ => bail!("L{} static page 身份/parity 漂移", self.layer),
        }
        if self.hidden != S14Position0DeviceHiddenBinding::for_layer(self.layer)? {
            bail!("L{} hidden A/B ping-pong 漂移", self.layer);
        }
        validate_page_bytes(self.static_page_bytes, "static")?;
        validate_page_bytes(self.routed_page_bytes, "routed")?;
        if self.static_weights.is_empty() || self.routed_weights.is_empty() {
            bail!("L{} static/routed 权重绑定不得为空", self.layer);
        }
        validate_uploads(
            &self.static_weights,
            S14Position0WeightPageTarget::Static(self.static_page),
            self.static_page_bytes,
            "static",
        )?;
        validate_uploads(
            &self.routed_weights,
            S14Position0WeightPageTarget::Routed {
                bank: self.routed_bank,
            },
            self.routed_page_bytes,
            "routed",
        )?;
        Ok(())
    }

    pub fn upload_request(&self) -> S14Position0LayerUploadRequest<'_> {
        let static_weights = match self.static_page {
            S14Position0StaticPage::Resident { .. } => &[],
            S14Position0StaticPage::Streamed { .. } => self.static_weights.as_slice(),
        };
        S14Position0LayerUploadRequest {
            layer: self.layer,
            static_page: self.static_page,
            static_page_bytes: self.static_page_bytes,
            static_weights,
            routed_bank: self.routed_bank,
            routed_page_bytes: self.routed_page_bytes,
            routed_weights: &self.routed_weights,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct S14Position0LayerUploadRequest<'a> {
    pub layer: u8,
    pub static_page: S14Position0StaticPage,
    pub static_page_bytes: u64,
    pub static_weights: &'a [S14Position0WeightUpload],
    pub routed_bank: usize,
    pub routed_page_bytes: u64,
    pub routed_weights: &'a [S14Position0WeightUpload],
}

impl S14Position0LayerUploadRequest<'_> {
    pub fn expected_static_upload_bytes(&self) -> Result<u64> {
        checked_upload_bytes(self.static_weights)
    }

    pub fn expected_routed_upload_bytes(&self) -> Result<u64> {
        checked_upload_bytes(self.routed_weights)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0LayerUploadReceipt {
    pub static_uploaded_bytes: u64,
    pub routed_uploaded_bytes: u64,
    /// 常驻 static 层只有 routed 一次等待；streamed static 首版允许 static+routed 两次。
    /// token 5 闭合后由双页 timeline 把它们替换为异步依赖。
    pub host_wait_calls: u32,
}

/// 主线 Vulkan 适配面的最小合同。
///
/// `upload_weights` 必须在返回前完成 staging→VRAM copy；`reconfigure_layer`
/// 只重绑当前权重页和 GPU hidden A/B workspace；`submit_layer` 不得读取 hidden。
pub trait S14Position0SynchronousLayerBackend {
    type ComputeTicket: Copy;

    fn wait_compute(&mut self, layer: u8, ticket: Self::ComputeTicket) -> Result<()>;

    fn upload_weights(
        &mut self,
        request: S14Position0LayerUploadRequest<'_>,
    ) -> Result<S14Position0LayerUploadReceipt>;

    fn reconfigure_layer(&mut self, plan: &S14Position0SynchronousLayerPlan) -> Result<()>;

    fn submit_layer(&mut self, layer: u8) -> Result<Self::ComputeTicket>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0LayerReconfigureReceipt<T> {
    pub layer: u8,
    pub waited_for_layer: Option<u8>,
    pub upload: S14Position0LayerUploadReceipt,
    pub compute_ticket: T,
    pub hidden: S14Position0DeviceHiddenBinding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0SynchronousLayerSummary {
    pub completed_layers: u8,
    pub compute_wait_calls: u32,
    pub upload_wait_calls: u32,
    pub static_uploaded_bytes: u64,
    pub routed_uploaded_bytes: u64,
    /// 该值固定为零；接口中不存在 hidden readback 操作。
    pub hidden_readback_bytes: u64,
}

pub struct S14Position0SynchronousLayerPager<T: Copy> {
    next_layer: u8,
    in_flight: Option<(u8, T)>,
    compute_wait_calls: u32,
    upload_wait_calls: u32,
    static_uploaded_bytes: u64,
    routed_uploaded_bytes: u64,
    finished: bool,
    poisoned: bool,
}

impl<T: Copy> Default for S14Position0SynchronousLayerPager<T> {
    fn default() -> Self {
        Self {
            next_layer: 0,
            in_flight: None,
            compute_wait_calls: 0,
            upload_wait_calls: 0,
            static_uploaded_bytes: 0,
            routed_uploaded_bytes: 0,
            finished: false,
            poisoned: false,
        }
    }
}

impl<T: Copy> S14Position0SynchronousLayerPager<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn next_layer(&self) -> Option<u8> {
        (!self.finished && self.next_layer < S14_POSITION0_SYNCHRONOUS_LAYER_COUNT)
            .then_some(self.next_layer)
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// 同步接通一层。L1 起先等待上一层 compute，因此 descriptor 和滚动页均可安全复用；
    /// hidden 仍留在同一个 GPU workspace，只从上一层 output 槽变成当前 input 槽。
    pub fn reconfigure_layer<B>(
        &mut self,
        backend: &mut B,
        plan: &S14Position0SynchronousLayerPlan,
    ) -> Result<S14Position0LayerReconfigureReceipt<T>>
    where
        B: S14Position0SynchronousLayerBackend<ComputeTicket = T>,
    {
        self.ensure_active()?;
        if let Err(error) = plan.validate() {
            return self.poison(error.context("validate synchronous layer plan"));
        }
        if plan.layer != self.next_layer {
            return self.poison(anyhow!(
                "position0 synchronous layer 顺序漂移: actual=L{} expected=L{}",
                plan.layer,
                self.next_layer
            ));
        }

        let waited_for_layer = if let Some((previous_layer, ticket)) = self.in_flight {
            if previous_layer + 1 != plan.layer {
                return self.poison(anyhow!("position0 in-flight layer 身份漂移"));
            }
            if let Err(error) = backend.wait_compute(previous_layer, ticket) {
                return self.poison(error.context(format!("wait L{previous_layer} compute")));
            }
            self.in_flight = None;
            self.compute_wait_calls = self
                .compute_wait_calls
                .checked_add(1)
                .ok_or_else(|| anyhow!("position0 compute wait counter overflow"))?;
            Some(previous_layer)
        } else {
            if plan.layer != 0 {
                return self.poison(anyhow!("L{} 缺少上一层 compute ticket", plan.layer));
            }
            None
        };

        let request = plan.upload_request();
        let expected_static = match request.expected_static_upload_bytes() {
            Ok(bytes) => bytes,
            Err(error) => return self.poison(error),
        };
        let expected_routed = match request.expected_routed_upload_bytes() {
            Ok(bytes) => bytes,
            Err(error) => return self.poison(error),
        };
        let upload = match backend.upload_weights(request) {
            Ok(receipt) => receipt,
            Err(error) => {
                return self.poison(error.context(format!("upload L{} weights", plan.layer)));
            }
        };
        if upload.static_uploaded_bytes != expected_static
            || upload.routed_uploaded_bytes != expected_routed
            || !(1..=2).contains(&upload.host_wait_calls)
        {
            return self.poison(anyhow!(
                "L{} synchronous upload receipt 漂移: actual={upload:?} expected_static={} expected_routed={} waits=1..=2",
                plan.layer,
                expected_static,
                expected_routed
            ));
        }
        if let Err(error) = backend.reconfigure_layer(plan) {
            return self.poison(error.context(format!("reconfigure L{}", plan.layer)));
        }
        let compute_ticket = match backend.submit_layer(plan.layer) {
            Ok(ticket) => ticket,
            Err(error) => return self.poison(error.context(format!("submit L{}", plan.layer))),
        };

        let upload_wait_calls = match self.upload_wait_calls.checked_add(upload.host_wait_calls) {
            Some(value) => value,
            None => return self.poison(anyhow!("position0 upload wait counter overflow")),
        };
        let static_uploaded_bytes = match self
            .static_uploaded_bytes
            .checked_add(upload.static_uploaded_bytes)
        {
            Some(value) => value,
            None => return self.poison(anyhow!("position0 static upload bytes overflow")),
        };
        let routed_uploaded_bytes = match self
            .routed_uploaded_bytes
            .checked_add(upload.routed_uploaded_bytes)
        {
            Some(value) => value,
            None => return self.poison(anyhow!("position0 routed upload bytes overflow")),
        };
        self.in_flight = Some((plan.layer, compute_ticket));
        self.next_layer += 1;
        self.upload_wait_calls = upload_wait_calls;
        self.static_uploaded_bytes = static_uploaded_bytes;
        self.routed_uploaded_bytes = routed_uploaded_bytes;
        Ok(S14Position0LayerReconfigureReceipt {
            layer: plan.layer,
            waited_for_layer,
            upload,
            compute_ticket,
            hidden: plan.hidden,
        })
    }

    /// 等待 L42，签发同步 43 层控制面回执。它只说明换页/顺序合同完成，
    /// 不代表 final head 或 token 数值门通过。
    pub fn finish<B>(&mut self, backend: &mut B) -> Result<S14Position0SynchronousLayerSummary>
    where
        B: S14Position0SynchronousLayerBackend<ComputeTicket = T>,
    {
        self.ensure_active()?;
        if self.next_layer != S14_POSITION0_SYNCHRONOUS_LAYER_COUNT {
            return self.poison(anyhow!(
                "position0 synchronous finish 过早: completed={} expected={}",
                self.next_layer,
                S14_POSITION0_SYNCHRONOUS_LAYER_COUNT
            ));
        }
        let (layer, ticket) = match self.in_flight {
            Some(in_flight) => in_flight,
            None => return self.poison(anyhow!("position0 synchronous finish 缺少 L42 ticket")),
        };
        if layer + 1 != S14_POSITION0_SYNCHRONOUS_LAYER_COUNT {
            return self.poison(anyhow!("position0 synchronous terminal layer 漂移"));
        }
        if let Err(error) = backend.wait_compute(layer, ticket) {
            return self.poison(error.context("wait terminal L42 compute"));
        }
        self.in_flight = None;
        self.compute_wait_calls = self
            .compute_wait_calls
            .checked_add(1)
            .ok_or_else(|| anyhow!("position0 compute wait counter overflow"))?;
        self.finished = true;
        Ok(S14Position0SynchronousLayerSummary {
            completed_layers: self.next_layer,
            compute_wait_calls: self.compute_wait_calls,
            upload_wait_calls: self.upload_wait_calls,
            static_uploaded_bytes: self.static_uploaded_bytes,
            routed_uploaded_bytes: self.routed_uploaded_bytes,
            hidden_readback_bytes: 0,
        })
    }

    fn ensure_active(&self) -> Result<()> {
        if self.poisoned {
            bail!("position0 synchronous layer pager 已 poisoned");
        }
        if self.finished {
            bail!("position0 synchronous layer pager 已完成");
        }
        Ok(())
    }

    fn poison<U>(&mut self, error: anyhow::Error) -> Result<U> {
        self.poisoned = true;
        Err(error)
    }
}

fn validate_page_bytes(bytes: u64, label: &str) -> Result<()> {
    if bytes == 0 || bytes % S14_POSITION0_SYNCHRONOUS_WEIGHT_ALIGNMENT != 0 {
        bail!("position0 {label} page bytes 必须为非零 256-byte 对齐");
    }
    Ok(())
}

fn validate_uploads(
    uploads: &[S14Position0WeightUpload],
    expected_target: S14Position0WeightPageTarget,
    capacity: u64,
    label: &str,
) -> Result<()> {
    let mut tensors = HashSet::with_capacity(uploads.len());
    let mut ranges = Vec::with_capacity(uploads.len());
    for upload in uploads {
        if upload.tensor.is_empty()
            || upload.kind.is_empty()
            || !upload.source_path.is_absolute()
            || !is_sha256(&upload.sha256)
            || upload.bytes == 0
            || upload.destination_offset % S14_POSITION0_SYNCHRONOUS_WEIGHT_ALIGNMENT != 0
            || upload.target != expected_target
            || !tensors.insert(upload.tensor.as_str())
        {
            bail!(
                "position0 {label} upload 身份/对齐/target 漂移: {}",
                upload.tensor
            );
        }
        let end = upload
            .destination_offset
            .checked_add(upload.bytes)
            .ok_or_else(|| anyhow!("position0 {label} upload end overflow"))?;
        if end > capacity {
            bail!("position0 {label} upload 越出 page: {}", upload.tensor);
        }
        ranges.push((upload.destination_offset, end, upload.tensor.as_str()));
    }
    ranges.sort_unstable_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            bail!(
                "position0 {label} upload 重叠: {} / {}",
                pair[0].2,
                pair[1].2
            );
        }
    }
    Ok(())
}

fn checked_upload_bytes(uploads: &[S14Position0WeightUpload]) -> Result<u64> {
    uploads.iter().try_fold(0u64, |total, upload| {
        total
            .checked_add(upload.bytes)
            .ok_or_else(|| anyhow!("position0 upload logical bytes overflow"))
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
