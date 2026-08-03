//! FullDepth43 K=1/4/8 的 BF16 分块 lm-head 与跨 chunk GPU argmax 原语。
//!
//! 生产 head 为 `[129280,4096]` BF16，按 4096 行分成 32 块。每块只占用
//! 一个 4096-F32 logits 临时区；GPU accumulator 在块间保存全局 top-1 和
//! `next_expected_token`，因此不需要把 1.06GB 权重或完整 logits 同时常驻。
//! 所有 descriptor 都支持非零 offset。normalized input、chunk logits 和
//! accumulator 必须来自同一个 workspace 的互不重叠子范围。

use crate::compute::{
    storage_buffer_slices_overlap, ComputePipeline, DescriptorBinder, StorageBufferSlice,
};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{anyhow, bail, Result};
use ash::vk;

pub const S14_HEAD_VOCAB: u32 = 129_280;
pub const S14_HEAD_HIDDEN: u32 = 4_096;
pub const S14_HEAD_CHUNK_ROWS: u32 = 4_096;
pub const S14_HEAD_CHUNK_COUNT: u32 = 32;
pub const S14_HEAD_MAX_BATCH: u32 = 8;
pub const S14_HEAD_ARGMAX_WORDS: usize = 4;
pub const S14_HEAD_ARGMAX_BYTES: u64 = S14_HEAD_ARGMAX_WORDS as u64 * 4;

pub const S14_HEAD_STATUS_NON_FINITE: u32 = 1;
pub const S14_HEAD_STATUS_SEQUENCE: u32 = 2;
pub const S14_HEAD_STATUS_TOKEN_RANGE: u32 = 4;
pub const S14_HEAD_STATUS_EMPTY: u32 = 8;

const HEAD_CHUNK_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/s14_bf16_head_chunk.spv"));
const HEAD_CHUNK_ARGMAX_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_head_chunk_argmax.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14HeadChunkArgmaxShape {
    pub vocab: u32,
    pub hidden: u32,
    pub chunk_rows: u32,
    pub batch: u32,
}

impl S14HeadChunkArgmaxShape {
    pub fn production() -> Self {
        Self {
            vocab: S14_HEAD_VOCAB,
            hidden: S14_HEAD_HIDDEN,
            chunk_rows: S14_HEAD_CHUNK_ROWS,
            batch: 1,
        }
    }

    pub fn production_batched(batch: u32) -> Result<Self> {
        Self::new_batched(S14_HEAD_VOCAB, S14_HEAD_HIDDEN, S14_HEAD_CHUNK_ROWS, batch)
    }

    pub fn new(vocab: u32, hidden: u32, chunk_rows: u32) -> Result<Self> {
        Self::new_batched(vocab, hidden, chunk_rows, 1)
    }

    pub fn new_batched(vocab: u32, hidden: u32, chunk_rows: u32, batch: u32) -> Result<Self> {
        if vocab == 0
            || hidden == 0
            || chunk_rows == 0
            || chunk_rows > 65_535
            || batch == 0
            || batch > S14_HEAD_MAX_BATCH
        {
            bail!(
                "S14 head chunk shape requires non-zero vocab/hidden, chunk_rows in 1..=65535, and batch in 1..={S14_HEAD_MAX_BATCH}"
            );
        }
        let shape = Self {
            vocab,
            hidden,
            chunk_rows,
            batch,
        };
        shape.head_total_bytes()?;
        shape.max_chunk_weight_bytes()?;
        shape.normalized_input_bytes()?;
        shape.max_chunk_logits_bytes()?;
        shape.argmax_bytes()?;
        Ok(shape)
    }

    pub fn chunk_count(self) -> u32 {
        self.vocab.div_ceil(self.chunk_rows)
    }

    pub fn chunk(self, index: u32) -> Result<S14HeadChunkSpec> {
        if index >= self.chunk_count() {
            bail!(
                "S14 head chunk index out of range: index={index} chunks={}",
                self.chunk_count()
            );
        }
        let token_start = index
            .checked_mul(self.chunk_rows)
            .ok_or_else(|| anyhow!("S14 head chunk token start overflow"))?;
        let rows = (self.vocab - token_start).min(self.chunk_rows);
        Ok(S14HeadChunkSpec {
            index,
            token_start,
            rows,
        })
    }

    pub fn normalized_input_bytes(self) -> Result<u64> {
        checked_bytes(self.batch as u64, self.hidden as u64, "S14 head input rows")
            .and_then(|elements| checked_bytes(elements, 4, "S14 head normalized input"))
    }

    pub fn max_chunk_logits_bytes(self) -> Result<u64> {
        checked_bytes(
            self.batch as u64,
            self.chunk_rows as u64,
            "S14 head chunk logit rows",
        )
        .and_then(|elements| checked_bytes(elements, 4, "S14 head chunk logits"))
    }

    pub fn argmax_bytes(self) -> Result<u64> {
        checked_bytes(
            self.batch as u64,
            S14_HEAD_ARGMAX_BYTES,
            "S14 head batched argmax",
        )
    }

    pub fn max_chunk_weight_bytes(self) -> Result<u64> {
        checked_bytes(
            self.chunk_rows as u64,
            self.hidden as u64,
            "S14 head chunk elements",
        )
        .and_then(|elements| checked_bytes(elements, 2, "S14 head chunk weights"))
    }

    pub fn head_total_bytes(self) -> Result<u64> {
        checked_bytes(
            self.vocab as u64,
            self.hidden as u64,
            "S14 head total elements",
        )
        .and_then(|elements| checked_bytes(elements, 2, "S14 head total weights"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14HeadChunkSpec {
    pub index: u32,
    pub token_start: u32,
    pub rows: u32,
}

impl S14HeadChunkSpec {
    pub fn weight_bytes(self, shape: S14HeadChunkArgmaxShape) -> Result<u64> {
        checked_bytes(
            self.rows as u64,
            shape.hidden as u64,
            "S14 head chunk elements",
        )
        .and_then(|elements| checked_bytes(elements, 2, "S14 head chunk weights"))
    }

    pub fn logits_bytes(self, shape: S14HeadChunkArgmaxShape) -> Result<u64> {
        checked_bytes(
            shape.batch as u64,
            self.rows as u64,
            "S14 head chunk logit rows",
        )
        .and_then(|elements| checked_bytes(elements, 4, "S14 head chunk logits"))
    }
}

#[derive(Clone, Copy)]
pub struct S14HeadChunkWorkspace<'a> {
    pub normalized_input: StorageBufferSlice<'a>,
    pub chunk_logits: StorageBufferSlice<'a>,
    pub accumulator: StorageBufferSlice<'a>,
}

impl<'a> S14HeadChunkWorkspace<'a> {
    pub const fn new(
        workspace: &'a GpuBuffer,
        normalized_input_offset: u64,
        chunk_logits_offset: u64,
        accumulator_offset: u64,
    ) -> Self {
        Self {
            normalized_input: StorageBufferSlice {
                buffer: workspace,
                offset: normalized_input_offset,
            },
            chunk_logits: StorageBufferSlice {
                buffer: workspace,
                offset: chunk_logits_offset,
            },
            accumulator: StorageBufferSlice {
                buffer: workspace,
                offset: accumulator_offset,
            },
        }
    }
}

pub struct S14HeadChunkArgmaxPipeline {
    head: ComputePipeline,
    argmax: ComputePipeline,
}

pub struct S14HeadChunkArgmaxDispatch<'a> {
    head_binder: DescriptorBinder,
    argmax_binder: DescriptorBinder,
    shape: S14HeadChunkArgmaxShape,
    spec: S14HeadChunkSpec,
    workspace: S14HeadChunkWorkspace<'a>,
}

impl S14HeadChunkArgmaxDispatch<'_> {
    pub fn spec(&self) -> S14HeadChunkSpec {
        self.spec
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.argmax_binder.destroy(ctx);
        self.head_binder.destroy(ctx);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkspaceIdentity {
    buffer: vk::Buffer,
    normalized_input_offset: u64,
    chunk_logits_offset: u64,
    accumulator_offset: u64,
}

impl WorkspaceIdentity {
    fn from_workspace(workspace: S14HeadChunkWorkspace<'_>) -> Self {
        Self {
            buffer: workspace.normalized_input.buffer.handle(),
            normalized_input_offset: workspace.normalized_input.offset,
            chunk_logits_offset: workspace.chunk_logits.offset,
            accumulator_offset: workspace.accumulator.offset,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecorderState {
    New,
    Recording,
    Finished,
    Poisoned,
}

pub struct S14HeadChunkArgmaxRecorder {
    shape: S14HeadChunkArgmaxShape,
    state: RecorderState,
    next_chunk: u32,
    workspace: Option<WorkspaceIdentity>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14HeadChunkArgmaxRecordingReceipt {
    pub shape: S14HeadChunkArgmaxShape,
    pub submitted_chunks: u32,
    pub expected_next_token: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct S14HeadArgmaxResult {
    pub token_id: u32,
    pub logit: f32,
}

impl S14HeadChunkArgmaxPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            head: ComputePipeline::new(ctx, HEAD_CHUNK_SPV, 3, 16)?,
            argmax: ComputePipeline::new(ctx, HEAD_CHUNK_ARGMAX_SPV, 2, 16)?,
        })
    }

    /// 为一个逻辑 chunk 创建 descriptor。`head_weight.offset` 可以非零；绑定范围
    /// 只覆盖本 chunk 的真实行数。任何越界或与 workspace 的 alias 都在提交前拒绝。
    pub fn bind_chunk<'a>(
        &self,
        ctx: &VulkanContext,
        shape: S14HeadChunkArgmaxShape,
        chunk_index: u32,
        head_weight: StorageBufferSlice<'a>,
        workspace: S14HeadChunkWorkspace<'a>,
    ) -> Result<S14HeadChunkArgmaxDispatch<'a>> {
        let shape = S14HeadChunkArgmaxShape::new_batched(
            shape.vocab,
            shape.hidden,
            shape.chunk_rows,
            shape.batch,
        )?;
        let spec = shape.chunk(chunk_index)?;
        validate_workspace(shape, workspace)?;
        let weight_bytes = spec.weight_bytes(shape)?;
        let logits_bytes = spec.logits_bytes(shape)?;
        validate_slice(head_weight, weight_bytes, "S14 head chunk weight")?;
        for (slice, bytes, name) in [
            (
                workspace.normalized_input,
                shape.normalized_input_bytes()?,
                "normalized input",
            ),
            (workspace.chunk_logits, logits_bytes, "chunk logits"),
            (
                workspace.accumulator,
                shape.argmax_bytes()?,
                "argmax accumulator",
            ),
        ] {
            if storage_buffer_slices_overlap(head_weight, weight_bytes, slice, bytes)? {
                bail!("S14 head chunk weight overlaps workspace {name}");
            }
        }

        let head_binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.head,
            &[
                (head_weight.buffer, head_weight.offset, weight_bytes),
                (
                    workspace.normalized_input.buffer,
                    workspace.normalized_input.offset,
                    shape.normalized_input_bytes()?,
                ),
                (
                    workspace.chunk_logits.buffer,
                    workspace.chunk_logits.offset,
                    logits_bytes,
                ),
            ],
        )?;
        let argmax_binder = match DescriptorBinder::new_with_offsets(
            ctx,
            &self.argmax,
            &[
                (
                    workspace.chunk_logits.buffer,
                    workspace.chunk_logits.offset,
                    logits_bytes,
                ),
                (
                    workspace.accumulator.buffer,
                    workspace.accumulator.offset,
                    shape.argmax_bytes()?,
                ),
            ],
        ) {
            Ok(binder) => binder,
            Err(error) => {
                head_binder.destroy(ctx);
                return Err(error);
            }
        };
        Ok(S14HeadChunkArgmaxDispatch {
            head_binder,
            argmax_binder,
            shape,
            spec,
            workspace,
        })
    }

    /// 在已 recording 的 command buffer 中追加当前 chunk 的 head、barrier 和累计 argmax。
    /// 顺序合同由 recorder 与 GPU accumulator 双重检查。
    unsafe fn cmd_chunk(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14HeadChunkArgmaxDispatch<'_>,
    ) {
        ctx.device
            .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.head.pipeline);
        ctx.device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.head.layout,
            0,
            &[dispatch.head_binder.set],
            &[],
        );
        let head_push_words = [
            dispatch.spec.rows,
            dispatch.shape.hidden,
            dispatch.shape.batch,
            0,
        ];
        let head_push = words_as_bytes(&head_push_words);
        ctx.device.cmd_push_constants(
            command,
            self.head.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            head_push,
        );
        // 一个 workgroup 只读取一次 head 权重行，并在同一归约里累计全部 batch lane。
        ctx.device.cmd_dispatch(command, dispatch.spec.rows, 1, 1);
        compute_barrier(ctx, command);

        ctx.device.cmd_bind_pipeline(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.argmax.pipeline,
        );
        ctx.device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.argmax.layout,
            0,
            &[dispatch.argmax_binder.set],
            &[],
        );
        let argmax_push_words = [
            dispatch.spec.rows,
            dispatch.spec.token_start,
            dispatch.shape.vocab,
            dispatch.shape.batch,
        ];
        let argmax_push = words_as_bytes(&argmax_push_words);
        ctx.device.cmd_push_constants(
            command,
            self.argmax.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            argmax_push,
        );
        ctx.device.cmd_dispatch(command, 1, dispatch.shape.batch, 1);
        compute_barrier(ctx, command);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.argmax.destroy(ctx);
        self.head.destroy(ctx);
    }
}

impl S14HeadChunkArgmaxRecorder {
    pub fn new(shape: S14HeadChunkArgmaxShape) -> Result<Self> {
        let shape = S14HeadChunkArgmaxShape::new_batched(
            shape.vocab,
            shape.hidden,
            shape.chunk_rows,
            shape.batch,
        )?;
        Ok(Self {
            shape,
            state: RecorderState::New,
            next_chunk: 0,
            workspace: None,
        })
    }

    /// 将累计区清零并建立 transfer→compute 可见性。workspace buffer 必须带
    /// `TRANSFER_DST`；生产 position0 workspace 已满足该合同。
    ///
    /// # Safety
    /// `command` 必须处于 recording 状态，dispatch 资源必须活到提交完成。
    pub unsafe fn cmd_reset(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        first: &S14HeadChunkArgmaxDispatch<'_>,
    ) -> Result<()> {
        if self.state != RecorderState::New || first.shape != self.shape || first.spec.index != 0 {
            self.state = RecorderState::Poisoned;
            bail!("S14 head argmax reset phase/shape/first-chunk drift");
        }
        self.workspace = Some(WorkspaceIdentity::from_workspace(first.workspace));
        ctx.device.cmd_fill_buffer(
            command,
            first.workspace.accumulator.buffer.handle(),
            first.workspace.accumulator.offset,
            self.shape.argmax_bytes()?,
            0,
        );
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        ctx.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
        self.state = RecorderState::Recording;
        Ok(())
    }

    /// # Safety
    /// `command` 必须处于 recording 状态，dispatch 及其 buffer 必须活到提交完成。
    pub unsafe fn cmd_chunk(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        pipeline: &S14HeadChunkArgmaxPipeline,
        dispatch: &S14HeadChunkArgmaxDispatch<'_>,
    ) -> Result<()> {
        let expected = self.shape.chunk(self.next_chunk);
        let valid = self.state == RecorderState::Recording
            && dispatch.shape == self.shape
            && expected.as_ref().is_ok_and(|spec| *spec == dispatch.spec)
            && self.workspace == Some(WorkspaceIdentity::from_workspace(dispatch.workspace));
        if !valid {
            self.state = RecorderState::Poisoned;
            bail!(
                "S14 head chunk order/range/workspace drift: expected_chunk={} actual={:?}",
                self.next_chunk,
                dispatch.spec
            );
        }
        pipeline.cmd_chunk(ctx, command, dispatch);
        self.next_chunk += 1;
        Ok(())
    }

    pub fn finish_recording(&mut self) -> Result<S14HeadChunkArgmaxRecordingReceipt> {
        if self.state != RecorderState::Recording || self.next_chunk != self.shape.chunk_count() {
            self.state = RecorderState::Poisoned;
            bail!(
                "S14 head argmax incomplete recording: chunks={}/{}",
                self.next_chunk,
                self.shape.chunk_count()
            );
        }
        self.state = RecorderState::Finished;
        Ok(S14HeadChunkArgmaxRecordingReceipt {
            shape: self.shape,
            submitted_chunks: self.next_chunk,
            expected_next_token: self.shape.vocab,
        })
    }
}

/// fence 完成后解码累计区。任何 sticky status、未覆盖完整词表、越界 token
/// 或非有限 logit 都禁止发布 token。
pub fn decode_head_argmax(
    receipt: &S14HeadChunkArgmaxRecordingReceipt,
    words: [u32; S14_HEAD_ARGMAX_WORDS],
) -> Result<S14HeadArgmaxResult> {
    if receipt.shape.batch != 1 {
        bail!(
            "S14 scalar head decode requires batch=1, actual={}",
            receipt.shape.batch
        );
    }
    Ok(decode_batched_head_argmax(receipt, &words)?[0])
}

/// fence 完成后一次解码 K 行累计区。每行都有独立的顺序、状态和 top-1；
/// 任一行失败会拒绝整个 causal block，避免只发布部分 GPU head 结果。
pub fn decode_batched_head_argmax(
    receipt: &S14HeadChunkArgmaxRecordingReceipt,
    words: &[u32],
) -> Result<Vec<S14HeadArgmaxResult>> {
    let expected_words = usize::try_from(receipt.shape.batch)
        .map_err(|_| anyhow!("S14 head batch does not fit usize"))?
        .checked_mul(S14_HEAD_ARGMAX_WORDS)
        .ok_or_else(|| anyhow!("S14 head result word count overflow"))?;
    if words.len() != expected_words {
        bail!(
            "S14 head batched result size drift: words={} expected={expected_words}",
            words.len()
        );
    }
    if receipt.submitted_chunks != receipt.shape.chunk_count()
        || receipt.expected_next_token != receipt.shape.vocab
    {
        bail!("S14 head argmax receipt is incomplete");
    }
    let mut results = Vec::with_capacity(receipt.shape.batch as usize);
    for (lane, row) in words.chunks_exact(S14_HEAD_ARGMAX_WORDS).enumerate() {
        validate_head_argmax_status(row[3])?;
        if row[2] != receipt.expected_next_token {
            bail!(
                "S14 head argmax progress drift at batch lane {lane}: gpu_next={} expected={}",
                row[2],
                receipt.expected_next_token
            );
        }
        let token_id = row[0];
        let logit = f32::from_bits(row[1]);
        if token_id >= receipt.shape.vocab || !logit.is_finite() {
            bail!("S14 head argmax result token/logit drift at batch lane {lane}");
        }
        results.push(S14HeadArgmaxResult { token_id, logit });
    }
    Ok(results)
}

pub fn validate_head_argmax_status(code: u32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let known = S14_HEAD_STATUS_NON_FINITE
        | S14_HEAD_STATUS_SEQUENCE
        | S14_HEAD_STATUS_TOKEN_RANGE
        | S14_HEAD_STATUS_EMPTY;
    if code & !known != 0 {
        bail!("S14 head argmax returned unknown status bits 0x{code:08x}");
    }
    bail!("S14 head argmax rejected candidate, status=0x{code:08x}")
}

fn validate_workspace(
    shape: S14HeadChunkArgmaxShape,
    workspace: S14HeadChunkWorkspace<'_>,
) -> Result<()> {
    let handle = workspace.normalized_input.buffer.handle();
    if workspace.chunk_logits.buffer.handle() != handle
        || workspace.accumulator.buffer.handle() != handle
    {
        bail!("S14 head normalized/logits/argmax must share one workspace buffer");
    }
    let requirements = [
        (
            workspace.normalized_input,
            shape.normalized_input_bytes()?,
            "normalized input",
        ),
        (
            workspace.chunk_logits,
            shape.max_chunk_logits_bytes()?,
            "chunk logits",
        ),
        (
            workspace.accumulator,
            shape.argmax_bytes()?,
            "argmax accumulator",
        ),
    ];
    for (slice, bytes, name) in requirements {
        validate_slice(slice, bytes, name)?;
    }
    for left in 0..requirements.len() {
        for right in left + 1..requirements.len() {
            if storage_buffer_slices_overlap(
                requirements[left].0,
                requirements[left].1,
                requirements[right].0,
                requirements[right].1,
            )? {
                bail!(
                    "S14 head workspace slices overlap: {} / {}",
                    requirements[left].2,
                    requirements[right].2
                );
            }
        }
    }
    Ok(())
}

fn validate_slice(slice: StorageBufferSlice<'_>, bytes: u64, name: &str) -> Result<()> {
    let end = slice
        .offset
        .checked_add(bytes)
        .ok_or_else(|| anyhow!("{name} range overflow"))?;
    if bytes == 0 || end > slice.buffer.size() {
        bail!(
            "{name} out of bounds: offset={} bytes={} capacity={}",
            slice.offset,
            bytes,
            slice.buffer.size()
        );
    }
    Ok(())
}

#[cfg(test)]
fn validate_chunk_order(
    shape: S14HeadChunkArgmaxShape,
    next_chunk: u32,
    actual: S14HeadChunkSpec,
) -> Result<()> {
    let expected = shape.chunk(next_chunk)?;
    if expected != actual {
        bail!("S14 head chunk order/token range drift");
    }
    Ok(())
}

fn checked_bytes(left: u64, right: u64, name: &str) -> Result<u64> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("{name} byte/element overflow"))
}

fn words_as_bytes(words: &[u32; 4]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), 16) }
}

unsafe fn compute_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_shape_is_exactly_32_chunks() {
        let shape = S14HeadChunkArgmaxShape::production();
        assert_eq!(shape.chunk_count(), S14_HEAD_CHUNK_COUNT);
        assert_eq!(shape.head_total_bytes().unwrap(), 1_059_061_760);
        assert_eq!(shape.max_chunk_weight_bytes().unwrap(), 33_554_432);
        assert_eq!(shape.normalized_input_bytes().unwrap(), 16_384);
        assert_eq!(shape.max_chunk_logits_bytes().unwrap(), 16_384);
        assert_eq!(shape.argmax_bytes().unwrap(), 16);
        assert_eq!(shape.chunk(0).unwrap().rows, 4_096);
        assert_eq!(shape.chunk(31).unwrap().token_start, 126_976);
        assert_eq!(shape.chunk(31).unwrap().rows, 2_304);
    }

    #[test]
    fn k8_shape_scans_one_weight_chunk_for_eight_independent_rows() {
        let shape = S14HeadChunkArgmaxShape::production_batched(8).unwrap();
        assert_eq!(shape.chunk_count(), S14_HEAD_CHUNK_COUNT);
        assert_eq!(shape.max_chunk_weight_bytes().unwrap(), 33_554_432);
        assert_eq!(shape.normalized_input_bytes().unwrap(), 8 * 16_384);
        assert_eq!(shape.max_chunk_logits_bytes().unwrap(), 8 * 16_384);
        assert_eq!(shape.argmax_bytes().unwrap(), 8 * 16);
        assert!(S14HeadChunkArgmaxShape::production_batched(0).is_err());
        assert!(S14HeadChunkArgmaxShape::production_batched(9).is_err());
    }

    #[test]
    fn shape_chunk_order_and_status_fail_closed() {
        for shape in [(0, 4096, 4096), (32, 0, 8), (32, 8, 0), (32, 8, 65_536)] {
            assert!(S14HeadChunkArgmaxShape::new(shape.0, shape.1, shape.2).is_err());
        }
        let shape = S14HeadChunkArgmaxShape::new(11, 128, 4).unwrap();
        assert_eq!(shape.chunk_count(), 3);
        assert!(shape.chunk(3).is_err());
        validate_chunk_order(shape, 0, shape.chunk(0).unwrap()).unwrap();
        assert!(validate_chunk_order(shape, 0, shape.chunk(1).unwrap()).is_err());
        let malformed = S14HeadChunkSpec {
            index: 0,
            token_start: 1,
            rows: 4,
        };
        assert!(validate_chunk_order(shape, 0, malformed).is_err());
        validate_head_argmax_status(0).unwrap();
        for status in [1, 2, 4, 8, 15, 16] {
            assert!(validate_head_argmax_status(status).is_err());
        }
    }

    #[test]
    fn receipt_rejects_nan_out_of_range_and_incomplete_gpu_progress() {
        let shape = S14HeadChunkArgmaxShape::new(11, 128, 4).unwrap();
        let receipt = S14HeadChunkArgmaxRecordingReceipt {
            shape,
            submitted_chunks: 3,
            expected_next_token: 11,
        };
        let result = decode_head_argmax(&receipt, [7, 1.25f32.to_bits(), 11, 0]).unwrap();
        assert_eq!(result.token_id, 7);
        assert_eq!(result.logit, 1.25);
        assert!(decode_head_argmax(&receipt, [11, 1.0f32.to_bits(), 11, 0]).is_err());
        assert!(decode_head_argmax(&receipt, [7, f32::NAN.to_bits(), 11, 0]).is_err());
        assert!(decode_head_argmax(&receipt, [7, 1.0f32.to_bits(), 10, 0]).is_err());
        assert!(decode_head_argmax(&receipt, [7, 1.0f32.to_bits(), 11, 1]).is_err());
    }

    #[test]
    fn batched_decode_rejects_any_lane_drift() {
        let shape = S14HeadChunkArgmaxShape::new_batched(11, 128, 4, 4).unwrap();
        let receipt = S14HeadChunkArgmaxRecordingReceipt {
            shape,
            submitted_chunks: 3,
            expected_next_token: 11,
        };
        let mut words = Vec::new();
        for lane in 0..4u32 {
            words.extend_from_slice(&[lane, (lane as f32 + 0.5).to_bits(), 11, 0]);
        }
        let decoded = decode_batched_head_argmax(&receipt, &words).unwrap();
        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded[3].token_id, 3);
        words[2 * S14_HEAD_ARGMAX_WORDS + 3] = S14_HEAD_STATUS_SEQUENCE;
        assert!(decode_batched_head_argmax(&receipt, &words).is_err());
        assert!(decode_batched_head_argmax(&receipt, &words[..words.len() - 1]).is_err());
    }
}
