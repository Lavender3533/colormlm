//! S14 Starfold 单窗口 packed-row MXFP4 matvec 原语。

use crate::compute::{ComputePipeline, DescriptorBinder, ExternalStorageBuffer};
use crate::VulkanContext;
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;

pub const S14_STARFOLD_MXFP4_TILE_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_starfold_mxfp4_tile.spv"));

/// 与 production StarFold 双窗口合同一致；64 MiB 上限仍远低于旧 GiB union bank，
/// 8 MiB 默认值可让当前 4096×2048/2048×4096 投影各自一次整行提交完成。
pub const S14_STARFOLD_MXFP4_MAX_WINDOW_BYTES: u32 = 64 * 1024 * 1024;
pub const S14_STARFOLD_MXFP4_LOCAL_SIZE: u32 = 128;
pub const S14_STARFOLD_MXFP4_PAYLOAD_CONTRACT_VERSION: u32 = 1;
pub const S14_STARFOLD_MXFP4_ROW_ALIGNMENT: u64 = 64;
pub const S14_STARFOLD_MXFP4_RESERVED_SCALE: u8 = 0xff;

const F32_BYTES: u64 = 4;
const MXFP4_VALUES_PER_PACKED_BYTE: u32 = 2;
const MXFP4_VALUES_PER_SCALE: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldMxfp4TileShape {
    n: u32,
    k: u32,
    window_capacity_bytes: u32,
}

impl S14StarfoldMxfp4TileShape {
    pub fn new(n: u32, k: u32, window_capacity_bytes: u32) -> Result<Self> {
        if !matches!(n, 2048 | 4096) {
            bail!("S14 Starfold MXFP4 tile N 只允许 2048 或 4096");
        }
        if !matches!(k, 2048 | 4096) {
            bail!("S14 Starfold MXFP4 tile K 只允许 2048 或 4096");
        }
        if window_capacity_bytes == 0 || window_capacity_bytes > S14_STARFOLD_MXFP4_MAX_WINDOW_BYTES
        {
            bail!(
                "S14 Starfold MXFP4 window 必须位于 1..={S14_STARFOLD_MXFP4_MAX_WINDOW_BYTES} bytes"
            );
        }
        let shape = Self {
            n,
            k,
            window_capacity_bytes,
        };
        if u64::from(window_capacity_bytes) < shape.payload_row_bytes() {
            bail!(
                "S14 Starfold MXFP4 window 放不下一个完整 packed-weight/UE8M0 行: capacity={} row_bytes={}",
                window_capacity_bytes,
                shape.payload_row_bytes()
            );
        }
        if shape.packed_weight_row_bytes() % S14_STARFOLD_MXFP4_ROW_ALIGNMENT != 0
            || shape.scale_row_bytes() % S14_STARFOLD_MXFP4_ROW_ALIGNMENT != 0
        {
            bail!("S14 Starfold MXFP4 packed-weight/scale row alignment 漂移");
        }
        shape.input_bytes()?;
        shape.output_bytes()?;
        shape.tile(shape.tile_count() - 1)?;
        Ok(shape)
    }

    pub const fn n(self) -> u32 {
        self.n
    }

    pub const fn k(self) -> u32 {
        self.k
    }

    pub const fn window_capacity_bytes(self) -> u32 {
        self.window_capacity_bytes
    }

    pub const fn packed_weight_row_bytes(self) -> u64 {
        (self.k / MXFP4_VALUES_PER_PACKED_BYTE) as u64
    }

    pub const fn scale_row_bytes(self) -> u64 {
        (self.k / MXFP4_VALUES_PER_SCALE) as u64
    }

    pub const fn payload_row_bytes(self) -> u64 {
        self.packed_weight_row_bytes() + self.scale_row_bytes()
    }

    pub fn rows_per_full_tile(self) -> u32 {
        (u64::from(self.window_capacity_bytes) / self.payload_row_bytes()) as u32
    }

    pub fn tile_count(self) -> u32 {
        self.n.div_ceil(self.rows_per_full_tile())
    }

    pub fn tail_rows(self) -> u32 {
        let full_rows = self.rows_per_full_tile();
        let remainder = self.n % full_rows;
        if remainder == 0 {
            full_rows
        } else {
            remainder
        }
    }

    pub fn input_bytes(self) -> Result<u64> {
        checked_mul(u64::from(self.k), F32_BYTES, "input bytes")
    }

    pub fn output_bytes(self) -> Result<u64> {
        checked_mul(u64::from(self.n), F32_BYTES, "output bytes")
    }

    pub fn tile(self, tile_index: u32) -> Result<S14StarfoldMxfp4TileSpec> {
        let tile_count = self.tile_count();
        if tile_index >= tile_count {
            bail!("S14 Starfold MXFP4 tile index 越界: index={tile_index} count={tile_count}");
        }
        let rows_per_tile = self.rows_per_full_tile();
        let row_base = tile_index
            .checked_mul(rows_per_tile)
            .ok_or_else(|| anyhow!("S14 Starfold MXFP4 tile row_base overflow"))?;
        let rows = (self.n - row_base).min(rows_per_tile);
        let weight_bytes = checked_mul(
            u64::from(rows),
            self.packed_weight_row_bytes(),
            "tile packed-weight bytes",
        )?;
        let scale_bytes = checked_mul(
            u64::from(rows),
            self.scale_row_bytes(),
            "tile UE8M0 scale bytes",
        )?;
        let payload_bytes = weight_bytes
            .checked_add(scale_bytes)
            .ok_or_else(|| anyhow!("S14 Starfold MXFP4 tile payload bytes overflow"))?;
        if rows == 0
            || payload_bytes == 0
            || payload_bytes > u64::from(self.window_capacity_bytes)
            || payload_bytes % 4 != 0
            || weight_bytes % S14_STARFOLD_MXFP4_ROW_ALIGNMENT != 0
        {
            bail!("S14 Starfold MXFP4 tile 不是合法整行 payload");
        }
        Ok(S14StarfoldMxfp4TileSpec {
            tile_index,
            row_base,
            rows,
            weight_bytes,
            scale_bytes,
            payload_bytes,
        })
    }

    pub fn validate_complete_coverage(
        self,
        receipts: &[S14StarfoldMxfp4TileRecordingReceipt],
    ) -> Result<()> {
        let expected_count = self.tile_count() as usize;
        if receipts.len() != expected_count {
            bail!(
                "S14 Starfold MXFP4 tile 回执数量错误: expected={expected_count} actual={}",
                receipts.len()
            );
        }
        for (index, receipt) in receipts.iter().copied().enumerate() {
            receipt.validate()?;
            let expected = self.tile(index as u32)?;
            if receipt.shape != self
                || receipt.tile_index != expected.tile_index
                || receipt.row_base != expected.row_base
                || receipt.row_end != expected.row_end()?
                || receipt.rows != expected.rows
            {
                bail!("S14 Starfold MXFP4 tile 回执未形成无缝全局 N 行覆盖");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldMxfp4TileSpec {
    pub tile_index: u32,
    pub row_base: u32,
    pub rows: u32,
    pub weight_bytes: u64,
    pub scale_bytes: u64,
    pub payload_bytes: u64,
}

impl S14StarfoldMxfp4TileSpec {
    pub fn row_end(self) -> Result<u32> {
        self.row_base
            .checked_add(self.rows)
            .ok_or_else(|| anyhow!("S14 Starfold MXFP4 tile row_end overflow"))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct S14StarfoldMxfp4ExternalSlice {
    pub buffer: vk::Buffer,
    pub capacity_bytes: u64,
    pub offset: u64,
    pub logical_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct S14StarfoldMxfp4TileBindings {
    pub input_f32: S14StarfoldMxfp4ExternalSlice,
    pub raw_window: S14StarfoldMxfp4ExternalSlice,
    pub output_f32: S14StarfoldMxfp4ExternalSlice,
    pub scale_audit: S14StarfoldMxfp4ScaleAudit,
}

/// Host packer 对一个完整 `[packed rows][scale rows]` payload 的验收凭据。
/// 字段保持私有，只有实际扫描过对应 scale 区且未出现 `0xff` 才能构造。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldMxfp4ScaleAudit {
    shape: S14StarfoldMxfp4TileShape,
    tile_index: u32,
    scale_offset: u64,
    scale_bytes: u64,
    payload_bytes: u64,
}

impl S14StarfoldMxfp4ScaleAudit {
    pub fn scan_host_payload(
        shape: S14StarfoldMxfp4TileShape,
        tile_index: u32,
        payload: &[u8],
    ) -> Result<Self> {
        let shape = S14StarfoldMxfp4TileShape::new(shape.n, shape.k, shape.window_capacity_bytes)?;
        let spec = shape.tile(tile_index)?;
        let actual_bytes = u64::try_from(payload.len())
            .context("S14 Starfold MXFP4 host payload length 超出 u64")?;
        if actual_bytes != spec.payload_bytes {
            bail!(
                "S14 Starfold MXFP4 host payload 不是完整整行 tile: expected={} actual={actual_bytes}",
                spec.payload_bytes
            );
        }
        let scale_start = usize::try_from(spec.weight_bytes)
            .context("S14 Starfold MXFP4 scale offset 超出 usize")?;
        let scale_end = usize::try_from(spec.payload_bytes)
            .context("S14 Starfold MXFP4 payload bytes 超出 usize")?;
        if let Some(scale_index) = payload[scale_start..scale_end]
            .iter()
            .position(|byte| *byte == S14_STARFOLD_MXFP4_RESERVED_SCALE)
        {
            bail!("S14 Starfold MXFP4 host packer 拒绝 UE8M0 0xff: scale_index={scale_index}");
        }
        Ok(Self {
            shape,
            tile_index,
            scale_offset: spec.weight_bytes,
            scale_bytes: spec.scale_bytes,
            payload_bytes: spec.payload_bytes,
        })
    }

    fn validate_against(
        self,
        shape: S14StarfoldMxfp4TileShape,
        spec: S14StarfoldMxfp4TileSpec,
    ) -> Result<()> {
        if self.shape != shape
            || self.tile_index != spec.tile_index
            || self.scale_offset != spec.weight_bytes
            || self.scale_bytes != spec.scale_bytes
            || self.payload_bytes != spec.payload_bytes
        {
            bail!("S14 Starfold MXFP4 scale audit 与 resident tile 不同源");
        }
        Ok(())
    }
}

pub struct S14StarfoldMxfp4TileDispatch {
    binder: DescriptorBinder,
    shape: S14StarfoldMxfp4TileShape,
    spec: S14StarfoldMxfp4TileSpec,
}

impl S14StarfoldMxfp4TileDispatch {
    pub const fn shape(&self) -> S14StarfoldMxfp4TileShape {
        self.shape
    }

    pub const fn spec(&self) -> S14StarfoldMxfp4TileSpec {
        self.spec
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.binder.destroy(ctx);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldMxfp4TileRecordingReceipt {
    pub payload_contract_version: u32,
    pub shape: S14StarfoldMxfp4TileShape,
    pub tile_index: u32,
    pub row_base: u32,
    pub row_end: u32,
    pub rows: u32,
    pub weight_bytes: u64,
    pub scale_bytes: u64,
    pub payload_bytes: u64,
    pub global_output_byte_start: u64,
    pub global_output_byte_end: u64,
    pub workgroups_x: u32,
    pub pipeline_bind_calls: u32,
    pub descriptor_bind_calls: u32,
    pub push_constant_calls: u32,
    pub dispatch_calls: u32,
    pub serial_token_forward_calls: u32,
    pub host_ue8m0_scan_required: bool,
    pub host_ue8m0_scan_verified: bool,
}

impl S14StarfoldMxfp4TileRecordingReceipt {
    pub fn validate(self) -> Result<()> {
        let shape = S14StarfoldMxfp4TileShape::new(
            self.shape.n,
            self.shape.k,
            self.shape.window_capacity_bytes,
        )?;
        let expected = shape.tile(self.tile_index)?;
        let expected_row_end = expected.row_end()?;
        let expected_output_start = checked_mul(
            u64::from(expected.row_base),
            F32_BYTES,
            "receipt output byte start",
        )?;
        let expected_output_end = checked_mul(
            u64::from(expected_row_end),
            F32_BYTES,
            "receipt output byte end",
        )?;
        if self.payload_contract_version != S14_STARFOLD_MXFP4_PAYLOAD_CONTRACT_VERSION
            || self.shape != shape
            || self.row_base != expected.row_base
            || self.row_end != expected_row_end
            || self.rows != expected.rows
            || self.weight_bytes != expected.weight_bytes
            || self.scale_bytes != expected.scale_bytes
            || self.payload_bytes != expected.payload_bytes
            || self.global_output_byte_start != expected_output_start
            || self.global_output_byte_end != expected_output_end
            || self.workgroups_x != expected.rows
            || self.pipeline_bind_calls != 1
            || self.descriptor_bind_calls != 1
            || self.push_constant_calls != 1
            || self.dispatch_calls != 1
            || self.serial_token_forward_calls != 0
            || !self.host_ue8m0_scan_required
            || !self.host_ue8m0_scan_verified
        {
            bail!("S14 Starfold MXFP4 tile 回执不能证明精确 packed-row 覆盖");
        }
        Ok(())
    }
}

pub struct S14StarfoldMxfp4TileRecorder {
    pipeline: ComputePipeline,
}

impl S14StarfoldMxfp4TileRecorder {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_STARFOLD_MXFP4_TILE_SPV, 3, 16)
                .context("创建 S14 Starfold MXFP4 tile pipeline")?,
        })
    }

    pub fn bind_external_tile(
        &self,
        ctx: &VulkanContext,
        shape: S14StarfoldMxfp4TileShape,
        tile_index: u32,
        bindings: S14StarfoldMxfp4TileBindings,
    ) -> Result<S14StarfoldMxfp4TileDispatch> {
        let shape = S14StarfoldMxfp4TileShape::new(shape.n, shape.k, shape.window_capacity_bytes)?;
        let spec = shape.tile(tile_index)?;
        bindings.scale_audit.validate_against(shape, spec)?;
        validate_external_slice(bindings.input_f32, shape.input_bytes()?, "F32 input")?;
        validate_external_slice(bindings.raw_window, spec.payload_bytes, "raw MXFP4 window")?;
        validate_external_slice(bindings.output_f32, shape.output_bytes()?, "F32 output")?;
        require_non_overlapping(bindings.input_f32, bindings.raw_window, "input", "window")?;
        require_non_overlapping(bindings.input_f32, bindings.output_f32, "input", "output")?;
        require_non_overlapping(bindings.raw_window, bindings.output_f32, "window", "output")?;

        let binder = DescriptorBinder::new_with_external_offsets(
            ctx,
            &self.pipeline,
            &[
                (
                    external(bindings.input_f32),
                    bindings.input_f32.offset,
                    shape.input_bytes()?,
                ),
                (
                    external(bindings.raw_window),
                    bindings.raw_window.offset,
                    spec.payload_bytes,
                ),
                (
                    external(bindings.output_f32),
                    bindings.output_f32.offset,
                    shape.output_bytes()?,
                ),
            ],
        )
        .context("绑定 S14 Starfold MXFP4 tile external buffers")?;
        Ok(S14StarfoldMxfp4TileDispatch {
            binder,
            shape,
            spec,
        })
    }

    /// # Safety
    ///
    /// `command` 必须正在 recording，三个 external buffer 必须保活到 GPU 完成；调用前 host
    /// packer 必须拒绝 UE8M0 `0xff`，调用后由外层 graph 建立 output write barrier。
    pub unsafe fn record_tile(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14StarfoldMxfp4TileDispatch,
    ) -> Result<S14StarfoldMxfp4TileRecordingReceipt> {
        if command == vk::CommandBuffer::null() {
            bail!("S14 Starfold MXFP4 tile command 不能为空");
        }
        let spec = dispatch.spec;
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&spec.rows.to_le_bytes());
        push[4..8].copy_from_slice(&dispatch.shape.k.to_le_bytes());
        let scale_offset = u32::try_from(spec.weight_bytes)
            .context("S14 Starfold MXFP4 tile scale_offset 超出 u32")?;
        push[8..12].copy_from_slice(&scale_offset.to_le_bytes());
        push[12..16].copy_from_slice(&spec.row_base.to_le_bytes());
        unsafe {
            ctx.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.pipeline,
            );
            ctx.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.layout,
                0,
                &[dispatch.binder.set],
                &[],
            );
            ctx.device.cmd_push_constants(
                command,
                self.pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            ctx.device.cmd_dispatch(command, spec.rows, 1, 1);
        }

        let row_end = spec.row_end()?;
        let receipt = S14StarfoldMxfp4TileRecordingReceipt {
            payload_contract_version: S14_STARFOLD_MXFP4_PAYLOAD_CONTRACT_VERSION,
            shape: dispatch.shape,
            tile_index: spec.tile_index,
            row_base: spec.row_base,
            row_end,
            rows: spec.rows,
            weight_bytes: spec.weight_bytes,
            scale_bytes: spec.scale_bytes,
            payload_bytes: spec.payload_bytes,
            global_output_byte_start: checked_mul(
                u64::from(spec.row_base),
                F32_BYTES,
                "output byte start",
            )?,
            global_output_byte_end: checked_mul(u64::from(row_end), F32_BYTES, "output byte end")?,
            workgroups_x: spec.rows,
            pipeline_bind_calls: 1,
            descriptor_bind_calls: 1,
            push_constant_calls: 1,
            dispatch_calls: 1,
            serial_token_forward_calls: 0,
            host_ue8m0_scan_required: true,
            host_ue8m0_scan_verified: true,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

fn external(binding: S14StarfoldMxfp4ExternalSlice) -> ExternalStorageBuffer {
    ExternalStorageBuffer {
        buffer: binding.buffer,
        capacity: binding.capacity_bytes,
    }
}

fn validate_external_slice(
    binding: S14StarfoldMxfp4ExternalSlice,
    expected_logical_bytes: u64,
    label: &str,
) -> Result<()> {
    if binding.buffer == vk::Buffer::null()
        || binding.capacity_bytes == 0
        || binding.logical_bytes == 0
    {
        bail!("S14 Starfold MXFP4 {label} handle/capacity/payload 不能为空");
    }
    if binding.logical_bytes != expected_logical_bytes {
        bail!(
            "S14 Starfold MXFP4 {label} payload 合同错误: expected={expected_logical_bytes} actual={}",
            binding.logical_bytes
        );
    }
    let end = binding
        .offset
        .checked_add(binding.logical_bytes)
        .ok_or_else(|| anyhow!("S14 Starfold MXFP4 {label} range overflow"))?;
    if end > binding.capacity_bytes {
        bail!(
            "S14 Starfold MXFP4 {label} 越界: end={end} capacity={}",
            binding.capacity_bytes
        );
    }
    Ok(())
}

fn require_non_overlapping(
    left: S14StarfoldMxfp4ExternalSlice,
    right: S14StarfoldMxfp4ExternalSlice,
    left_label: &str,
    right_label: &str,
) -> Result<()> {
    if left.buffer != right.buffer {
        return Ok(());
    }
    let left_end = left
        .offset
        .checked_add(left.logical_bytes)
        .ok_or_else(|| anyhow!("S14 Starfold MXFP4 {left_label} alias range overflow"))?;
    let right_end = right
        .offset
        .checked_add(right.logical_bytes)
        .ok_or_else(|| anyhow!("S14 Starfold MXFP4 {right_label} alias range overflow"))?;
    if left.offset < right_end && right.offset < left_end {
        bail!("S14 Starfold MXFP4 {left_label}/{right_label} binding 重叠");
    }
    Ok(())
}

fn checked_mul(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("S14 Starfold MXFP4 {label} overflow"))
}
