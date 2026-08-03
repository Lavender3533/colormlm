//! position4+ 的通用 compressed-indexer / sparse-attention Vulkan 合同。
//!
//! ratio4 index query 严格执行 position RoPE、Hadamard 与 FP4 QDQ；indexer
//! 按真实 score 输出 compressed block 顺序。attention 以逻辑顺序读取环形 window，
//! 再按 indexer 顺序读取 main compressed KV。ratio4 使用真实 indexer 排名；ratio128
//! 使用官方 block0..N-1 确定顺序。ratio4 的历史 logical length 可以超过
//! `INDEX_TOP_K=512`；此时 indexer 分页扫描全部历史，global merge 只选择最多512行
//! main KV进入连续 attention 工作集。当前文件固定分页/选择结构合同，global merge
//! Vulkan recorder 仍由上层 fail-closed。

use crate::compute::{
    storage_buffer_slices_overlap, ComputePipeline, DescriptorBinder, StorageBufferSlice,
};
use crate::s14_bf16_to_f32::{S14Bf16ToF32Dispatch, S14Bf16ToF32Pipeline, S14Bf16ToF32Shape};
use crate::s14_f32_to_bf16::{S14F32ToBf16Dispatch, S14F32ToBf16Pipeline, S14F32ToBf16Shape};
use crate::s14_position0_attention::{S14_POSITION0_HEADS, S14_POSITION0_HEAD_DIM};
use crate::s14_position1_attention::position_rope_cos_sin;
use crate::s14_ratio4_history_paging::{
    S14Ratio4HistoryLayout, S14Ratio4HistoryPage, S14_RATIO4_ATTENTION_TOP_K,
    S14_RATIO4_HISTORY_PAGE_ROWS,
};
use crate::s14_vulkan::{
    validate_e4m3fn_codes, validate_ue8m0_codes, S14Fp8Dispatch, S14MatvecShape,
    S14NumericPipelines,
};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{anyhow, bail, Result};
use ash::vk;

pub const S14_RATIO4_INDEX_QUERY_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_ratio4_index_query.spv"));
pub const S14_SPARSE_INDEXER_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_sparse_indexer.spv"));
pub const S14_SPARSE_ATTENTION_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_sparse_attention.spv"));

pub const S14_INDEX_HEADS: u32 = 64;
pub const S14_INDEX_HEAD_DIM: u32 = 128;
pub const S14_INDEX_QUERY_INPUT_DIM: u32 = 1024;
pub const S14_INDEX_QUERY_OUTPUT_DIM: u32 = S14_INDEX_HEADS * S14_INDEX_HEAD_DIM;
pub const S14_INDEX_TOP_K: u32 = 512;
pub const S14_WINDOW_ROWS: u32 = 128;
pub const S14_SPARSE_STATUS_INDEX_QUERY_NON_FINITE: u32 = 1;
pub const S14_SPARSE_STATUS_INDEX_QUERY_SCALE: u32 = 2;
pub const S14_SPARSE_STATUS_INDEX_QUERY_OUTPUT: u32 = 4;
pub const S14_SPARSE_STATUS_INDEX_SCORE_NON_FINITE: u32 = 8;
pub const S14_SPARSE_STATUS_INVALID_COMPRESSED_INDEX: u32 = 16;

const QUERY_BYTES: u64 = S14_POSITION0_HEADS as u64 * S14_POSITION0_HEAD_DIM as u64 * 2;
const KV_ROW_BYTES: u64 = S14_POSITION0_HEAD_DIM as u64 * 2;
const WINDOW_BYTES: u64 = S14_WINDOW_ROWS as u64 * KV_ROW_BYTES;
const INDEX_QUERY_BYTES: u64 = S14_INDEX_HEADS as u64 * S14_INDEX_HEAD_DIM as u64 * 2;
const INDEX_WEIGHT_BYTES: u64 = S14_INDEX_HEADS as u64 * 2;
const SINK_BYTES: u64 = S14_POSITION0_HEADS as u64 * 4;
const ROPE_BYTES: u64 = 32 * 2 * 4;
const STATUS_BYTES: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Ratio4IndexQueryShape {
    pub position: u32,
}

impl S14Ratio4IndexQueryShape {
    pub fn new(position: u32, compress_ratio: u16) -> Result<Self> {
        if compress_ratio != 4 {
            bail!("ratio4 index query 只允许 compress_ratio=4");
        }
        // position3 使用基数1特例；position127 的 ratio128 layer 走确定顺序，
        // 同一 token 的 ratio4 layers 仍必须运行真实 indexer。
        if position < 4 {
            bail!("ratio4 index query position 必须是4+");
        }
        Ok(Self { position })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14SparseAttentionShape {
    pub position: u32,
    pub window_start: u32,
    pub previous_count: u32,
    pub compressed_count: u32,
    /// ratio128 没有 indexer；官方按 block0..N-1 的确定顺序消费压缩历史。
    /// ratio4 必须保持 false 并读取真实 indexer 排名。
    pub implicit_compressed_indices: bool,
}

impl S14SparseAttentionShape {
    pub fn new(
        position: u32,
        window_start: u32,
        previous_count: u32,
        compressed_count: u32,
    ) -> Result<Self> {
        if position == 0 {
            bail!("通用 sparse attention 只允许 position1+");
        }
        let expected_previous = position.min(S14_WINDOW_ROWS - 1);
        let next_position = position
            .checked_add(1)
            .ok_or_else(|| anyhow!("sparse attention position overflow"))?;
        let expected_start = if next_position <= S14_WINDOW_ROWS {
            0
        } else {
            next_position % S14_WINDOW_ROWS
        };
        if previous_count != expected_previous
            || window_start != expected_start
            || compressed_count > S14_INDEX_TOP_K
            || previous_count + 1 + compressed_count > 640
        {
            bail!(
                "sparse attention shape 漂移: position={position} start={window_start}/{expected_start} previous={previous_count}/{expected_previous} compressed={compressed_count}"
            );
        }
        Ok(Self {
            position,
            window_start,
            previous_count,
            compressed_count,
            implicit_compressed_indices: false,
        })
    }

    /// position127/255/... 及其后续位置的 ratio128 确定顺序 attention。
    /// 当前 production 上层只会签发 position127；本构造器保留完整公式，且仍受
    /// 512 compressed rows 的现有 shader 容量硬门约束。
    pub fn new_ratio128(
        position: u32,
        window_start: u32,
        previous_count: u32,
        compressed_count: u32,
    ) -> Result<Self> {
        if position < 127 || compressed_count != (position + 1) / 128 {
            bail!(
                "ratio128 sparse attention identity 漂移: position={position} compressed={compressed_count}/{}",
                (position + 1) / 128
            );
        }
        let mut shape = Self::new(position, window_start, previous_count, compressed_count)?;
        shape.implicit_compressed_indices = true;
        Ok(shape)
    }

    /// 返回与官方 `get_window_topk_idxs + compressed_topk` 相同的物理 KV
    /// 消费顺序。compressed indexer 在当前 ABI 中保留全部 <=512 个 block，
    /// 因而 indices 必须是无重复的完整排列，不能携带越界或缺失 block。
    pub fn kv_sequence(self, compressed_indices: &[u32]) -> Result<Vec<S14SparseKvSource>> {
        let implicit;
        let compressed_indices = if self.implicit_compressed_indices {
            if !compressed_indices.is_empty() {
                bail!("ratio128 implicit compressed order 不得携带外部 indexer 排名");
            }
            implicit = (0..self.compressed_count).collect::<Vec<_>>();
            implicit.as_slice()
        } else {
            if compressed_indices.len() != self.compressed_count as usize {
                bail!("compressed index 数量与 shape 漂移");
            }
            compressed_indices
        };
        let mut seen = vec![false; self.compressed_count as usize];
        for &index in compressed_indices {
            let slot = seen
                .get_mut(index as usize)
                .ok_or_else(|| anyhow!("compressed index {index} 越界"))?;
            if *slot {
                bail!("compressed index {index} 重复");
            }
            *slot = true;
        }
        if seen.iter().any(|selected| !selected) {
            bail!("compressed index 不是完整 block 排列");
        }

        let mut sequence =
            Vec::with_capacity(self.previous_count as usize + 1 + self.compressed_count as usize);
        for logical in 0..self.previous_count {
            sequence.push(S14SparseKvSource::Window(
                (self.window_start + logical) % S14_WINDOW_ROWS,
            ));
        }
        sequence.push(S14SparseKvSource::Current);
        sequence.extend(
            compressed_indices
                .iter()
                .copied()
                .map(S14SparseKvSource::Compressed),
        );
        Ok(sequence)
    }
}

/// ratio4 历史超过512块后的两级长度合同：indexer 扫描 `logical_compressed_count`
/// 全历史页，attention 只消费 global top-k 聚合后的 `selected_count<=512` 连续行。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Ratio4PagedAttentionPlan {
    pub position: u32,
    pub logical_compressed_count: u32,
    pub selected_count: u32,
    pub indexer_pages: Vec<S14Ratio4HistoryPage>,
    pub attention_shape: S14SparseAttentionShape,
}

impl S14Ratio4PagedAttentionPlan {
    pub fn build(
        position: u32,
        window_start: u32,
        previous_count: u32,
        history: &S14Ratio4HistoryLayout,
    ) -> Result<Self> {
        let expected_logical = position
            .checked_add(1)
            .ok_or_else(|| anyhow!("ratio4 paged attention position overflow"))?
            / 4;
        if expected_logical == 0 || history.logical_len != expected_logical {
            bail!(
                "ratio4 paged attention logical length 漂移: position={position} history={}/{}",
                history.logical_len,
                expected_logical
            );
        }
        if history.pages.is_empty()
            || history.pages.iter().any(|page| {
                page.logical_len() == 0
                    || page.logical_len() > S14_RATIO4_HISTORY_PAGE_ROWS
                    || page.indexer_state_range.end - page.indexer_state_range.start
                        != u64::from(page.logical_len()) * u64::from(S14_INDEX_HEAD_DIM) * 2
            })
        {
            bail!("ratio4 paged attention indexer page layout 漂移");
        }
        let selected_count = expected_logical.min(S14_RATIO4_ATTENTION_TOP_K);
        let attention_shape =
            S14SparseAttentionShape::new(position, window_start, previous_count, selected_count)?;
        Ok(Self {
            position,
            logical_compressed_count: expected_logical,
            selected_count,
            indexer_pages: history.pages.clone(),
            attention_shape,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14SparseKvSource {
    Window(u32),
    Current,
    Compressed(u32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct S14SparseIndexerSelection {
    pub scores: Vec<f32>,
    pub indices: Vec<u32>,
}

/// compressed indexer 的纯 Rust 数值合同。它固定 Python/PyTorch 的运算顺序：
/// 每个 head 先做128维 BF16 dot；`weights_proj` 的 F32 accumulator 先按
/// BF16 RNE 发布，原位 scale 再按 BF16 RNE，随后才转 F32 与 ReLU score
/// 相乘并按 head0..63 累加；最终以稳定降序返回全部 block。
pub fn reference_sparse_indexer(
    query_bf16: &[u16],
    index_cache_bf16: &[u16],
    head_weights: &[f32],
    compressed_count: u32,
) -> Result<S14SparseIndexerSelection> {
    if compressed_count == 0 || compressed_count > S14_INDEX_TOP_K {
        bail!("reference ratio4 indexer compressed_count 必须是1..=512");
    }
    let expected_query = (S14_INDEX_HEADS * S14_INDEX_HEAD_DIM) as usize;
    let expected_cache = (compressed_count * S14_INDEX_HEAD_DIM) as usize;
    if query_bf16.len() != expected_query
        || index_cache_bf16.len() != expected_cache
        || head_weights.len() != S14_INDEX_HEADS as usize
    {
        bail!("reference ratio4 indexer shape 漂移");
    }
    if query_bf16
        .iter()
        .chain(index_cache_bf16)
        .any(|bits| !f32::from_bits(u32::from(*bits) << 16).is_finite())
        || head_weights.iter().any(|value| !value.is_finite())
    {
        bail!("reference ratio4 indexer 输入含 NaN/Inf");
    }

    let scale = f32::from_bits(0x3c35_04f3);
    let mut scores = vec![0.0f32; compressed_count as usize];
    for block in 0..compressed_count as usize {
        let mut score = 0.0f32;
        for head in 0..S14_INDEX_HEADS as usize {
            let mut dot = 0.0f32;
            for dimension in 0..S14_INDEX_HEAD_DIM as usize {
                let query = f32::from_bits(
                    u32::from(query_bf16[head * S14_INDEX_HEAD_DIM as usize + dimension]) << 16,
                );
                let cached = f32::from_bits(
                    u32::from(index_cache_bf16[block * S14_INDEX_HEAD_DIM as usize + dimension])
                        << 16,
                );
                dot += query * cached;
            }
            let projected_weight = bf16_to_f32(f32_to_bf16_rne(head_weights[head]));
            let scaled_weight = bf16_to_f32(f32_to_bf16_rne(projected_weight * scale));
            score += dot.max(0.0) * scaled_weight;
        }
        if !score.is_finite() {
            bail!("reference ratio4 indexer score 含 NaN/Inf");
        }
        scores[block] = score;
    }

    let mut indices: Vec<u32> = (0..compressed_count).collect();
    // 与 Vulkan selection sort 一致：严格大于才交换，精确 tie 保留较早 block。
    for output in 0..indices.len() {
        let mut best = output;
        for candidate in output + 1..indices.len() {
            if scores[indices[candidate] as usize] > scores[indices[best] as usize] {
                best = candidate;
            }
        }
        indices.swap(output, best);
    }
    let ordered_scores = indices
        .iter()
        .map(|&index| scores[index as usize])
        .collect();
    Ok(S14SparseIndexerSelection {
        scores: ordered_scores,
        indices,
    })
}

/// `indexer.wq_b → BF16 → position RoPE → normalized Hadamard → FP4 QDQ`
/// 的纯 Rust exact-audit 参考。只负责 index query 前半链，不包含
/// `weights_proj`、score 或 top-k。
pub fn reference_ratio4_index_query(
    qr_bf16: &[u16],
    wq_b_e4m3fn: &[u8],
    wq_b_ue8m0: &[u8],
    position: u32,
) -> Result<Vec<u16>> {
    S14Ratio4IndexQueryShape::new(position, 4)?;
    let input_scalars = S14_INDEX_QUERY_INPUT_DIM as usize;
    let output_scalars = S14_INDEX_QUERY_OUTPUT_DIM as usize;
    let weight_bytes = input_scalars
        .checked_mul(output_scalars)
        .ok_or_else(|| anyhow!("indexer.wq_b weight size overflow"))?;
    let k_groups = input_scalars.div_ceil(128);
    let scale_bytes = output_scalars.div_ceil(128) * k_groups;
    if qr_bf16.len() != input_scalars
        || wq_b_e4m3fn.len() != weight_bytes
        || wq_b_ue8m0.len() != scale_bytes
    {
        bail!(
            "indexer.wq_b ABI 漂移: qr={}/{} weight={}/{} scale={}/{}",
            qr_bf16.len(),
            input_scalars,
            wq_b_e4m3fn.len(),
            weight_bytes,
            wq_b_ue8m0.len(),
            scale_bytes
        );
    }
    validate_e4m3fn_codes(wq_b_e4m3fn)?;
    validate_ue8m0_codes(wq_b_ue8m0)?;
    if qr_bf16.iter().any(|bits| !bf16_to_f32(*bits).is_finite()) {
        bail!("indexer.wq_b qr BF16 含 NaN/Inf");
    }

    let input: Vec<f32> = qr_bf16.iter().copied().map(bf16_to_f32).collect();
    let mut projected = vec![0u16; output_scalars];
    // 与 s14_fp8_matvec_exact.comp 的 K=1024 路径一致。
    const REDUCTION_ORDER: [usize; 8] = [0, 1, 2, 3, 4, 5, 7, 6];
    for row in 0..output_scalars {
        let mut partial = [0.0f32; 8];
        let row_base = row * input_scalars;
        let scale_row = (row / 128) * k_groups;
        for lane in 0..8 {
            let mut accumulator = 0.0f32;
            for column in (lane..input_scalars).step_by(8) {
                let weight = decode_e4m3fn(wq_b_e4m3fn[row_base + column]);
                let scale = decode_ue8m0(wq_b_ue8m0[scale_row + column / 128]);
                accumulator = (weight * scale).mul_add(input[column], accumulator);
            }
            partial[lane] = accumulator;
        }
        let mut total = partial[REDUCTION_ORDER[0]];
        for &lane in &REDUCTION_ORDER[1..] {
            total += partial[lane];
        }
        if !total.is_finite() {
            bail!("indexer.wq_b projection row {row} 产生 NaN/Inf");
        }
        projected[row] = f32_to_bf16_rne(total);
    }

    let rope = position_rope_cos_sin(position, 4)?;
    for head in 0..S14_INDEX_HEADS as usize {
        let base = head * S14_INDEX_HEAD_DIM as usize;
        for pair in 0..32usize {
            let left_index = base + 64 + pair * 2;
            let right_index = left_index + 1;
            let left = bf16_to_f32(projected[left_index]);
            let right = bf16_to_f32(projected[right_index]);
            let cosine = rope[pair * 2];
            let sine = rope[pair * 2 + 1];
            projected[left_index] = f32_to_bf16_rne(left * cosine - right * sine);
            projected[right_index] = f32_to_bf16_rne(left * sine + right * cosine);
        }

        let mut values = [0.0f32; S14_INDEX_HEAD_DIM as usize];
        for (value, bits) in values.iter_mut().zip(&projected[base..base + 128]) {
            *value = bf16_to_f32(*bits);
        }
        let mut step = 1usize;
        while step < values.len() {
            for block in (0..values.len()).step_by(step * 2) {
                for offset in 0..step {
                    let left = values[block + offset];
                    let right = values[block + step + offset];
                    values[block + offset] = left + right;
                    values[block + step + offset] = left - right;
                }
            }
            step *= 2;
        }
        for value in &mut values {
            *value = bf16_to_f32(f32_to_bf16_rne(*value * f32::from_bits(0x3db5_04f3)));
        }
        for block in values.chunks_exact_mut(32) {
            let amax = block
                .iter()
                .map(|value| value.abs())
                .fold(0.0f32, f32::max)
                .max(6.0 * 2.0f32.powi(-126));
            let scale_exponent = (amax / 6.0).log2().ceil() as i32;
            let scale = 2.0f32.powi(scale_exponent);
            if !scale.is_finite() || scale <= 0.0 {
                bail!("index query FP4 scale 非法");
            }
            for value in block {
                let normalized = (*value / scale).clamp(-6.0, 6.0);
                *value = nearest_e2m1(normalized) * scale;
            }
        }
        for (slot, value) in projected[base..base + 128].iter_mut().zip(values) {
            if !value.is_finite() {
                bail!("index query FP4 输出含 NaN/Inf");
            }
            *slot = f32_to_bf16_rne(value);
        }
    }
    Ok(projected)
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return (bits >> 16) as u16;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn decode_e4m3fn(code: u8) -> f32 {
    let exponent = (code >> 3) & 0x0f;
    let mantissa = code & 7;
    let magnitude = if exponent == 0 {
        f32::from(mantissa) * 0.001953125
    } else {
        (1.0 + f32::from(mantissa) * 0.125) * 2.0f32.powi(i32::from(exponent) - 7)
    };
    if code & 0x80 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn decode_ue8m0(code: u8) -> f32 {
    2.0f32.powi(i32::from(code) - 127)
}

fn nearest_e2m1(value: f32) -> f32 {
    const LEVELS: [f32; 15] = [
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    ];
    let mut best = LEVELS[0];
    let mut distance = (value - best).abs();
    for &candidate in &LEVELS[1..] {
        let candidate_distance = (value - candidate).abs();
        if candidate_distance < distance {
            best = candidate;
            distance = candidate_distance;
        }
    }
    best
}

pub struct S14Ratio4IndexQueryPipeline {
    pipeline: ComputePipeline,
}

pub struct S14Ratio4IndexQueryDispatch {
    pub binder: DescriptorBinder,
    shape: S14Ratio4IndexQueryShape,
}

pub struct S14Ratio4IndexQueryChainDispatch {
    pub qr_to_f32: S14Bf16ToF32Dispatch,
    pub projection: S14Fp8Dispatch,
    pub projection_to_bf16: S14F32ToBf16Dispatch,
    pub postprocess: S14Ratio4IndexQueryDispatch,
}

impl S14Ratio4IndexQueryPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_RATIO4_INDEX_QUERY_SPV, 4, 4)?,
        })
    }

    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        input: StorageBufferSlice<'_>,
        rope: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
        shape: S14Ratio4IndexQueryShape,
    ) -> Result<S14Ratio4IndexQueryDispatch> {
        let shape = S14Ratio4IndexQueryShape::new(shape.position, 4)?;
        for (left, left_bytes, right, right_bytes, label) in [
            (
                input,
                INDEX_QUERY_BYTES,
                output,
                INDEX_QUERY_BYTES,
                "input/output",
            ),
            (
                input,
                INDEX_QUERY_BYTES,
                status,
                STATUS_BYTES,
                "input/status",
            ),
            (
                output,
                INDEX_QUERY_BYTES,
                status,
                STATUS_BYTES,
                "output/status",
            ),
            (rope, ROPE_BYTES, status, STATUS_BYTES, "rope/status"),
        ] {
            if storage_buffer_slices_overlap(left, left_bytes, right, right_bytes)? {
                bail!("ratio4 index query {label} slices 不得重叠");
            }
        }
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (input.buffer, input.offset, INDEX_QUERY_BYTES),
                (rope.buffer, rope.offset, ROPE_BYTES),
                (output.buffer, output.offset, INDEX_QUERY_BYTES),
                (status.buffer, status.offset, STATUS_BYTES),
            ],
        )?;
        Ok(S14Ratio4IndexQueryDispatch { binder, shape })
    }

    /// # Safety
    /// descriptor 资源必须活到 command 完成；调用前后由上层建立 compute barrier。
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Ratio4IndexQueryDispatch,
    ) {
        bind_dispatch(
            ctx,
            command,
            &self.pipeline,
            &dispatch.binder,
            S14_INDEX_HEADS,
            &dispatch.shape.position.to_le_bytes(),
        );
    }

    /// 绑定真实 `indexer.wq_b`：BF16 qr → F32 → packed FP8 matvec → BF16
    /// → position RoPE → normalized Hadamard → group32 E2M1 QDQ。
    /// `numeric_exact` 必须是 production 已有的 exact-audit pipeline，才能保持
    /// K=1024 的冻结归约顺序。
    #[allow(clippy::too_many_arguments)]
    pub fn bind_chain_arenas(
        &self,
        ctx: &VulkanContext,
        numeric_exact: &S14NumericPipelines,
        bf16_to_f32: &S14Bf16ToF32Pipeline,
        f32_to_bf16: &S14F32ToBf16Pipeline,
        weight_arena: &GpuBuffer,
        weight_arena_logical_bytes: u64,
        weight_offset: u64,
        weight_scale_offset: u64,
        workspace: &GpuBuffer,
        workspace_logical_bytes: u64,
        qr_bf16_offset: u64,
        qr_f32_offset: u64,
        projection_f32_offset: u64,
        projection_bf16_offset: u64,
        rope: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
        shape: S14Ratio4IndexQueryShape,
    ) -> Result<S14Ratio4IndexQueryChainDispatch> {
        let shape = S14Ratio4IndexQueryShape::new(shape.position, 4)?;
        let qr_to_f32 = bf16_to_f32.bind_slices(
            ctx,
            S14Bf16ToF32Shape::new(S14_INDEX_QUERY_INPUT_DIM)?,
            StorageBufferSlice {
                buffer: workspace,
                offset: qr_bf16_offset,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: qr_f32_offset,
            },
            status,
        )?;
        let projection = numeric_exact.bind_fp8_arenas(
            ctx,
            S14MatvecShape::new(S14_INDEX_QUERY_OUTPUT_DIM, S14_INDEX_QUERY_INPUT_DIM)?,
            weight_arena,
            weight_arena_logical_bytes,
            weight_offset,
            weight_scale_offset,
            workspace,
            workspace_logical_bytes,
            qr_f32_offset,
            projection_f32_offset,
        )?;
        let projection_to_bf16 = f32_to_bf16.bind_slices(
            ctx,
            S14F32ToBf16Shape::new(S14_INDEX_QUERY_OUTPUT_DIM)?,
            StorageBufferSlice {
                buffer: workspace,
                offset: projection_f32_offset,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: projection_bf16_offset,
            },
            status,
        )?;
        let postprocess = self.bind_slices(
            ctx,
            StorageBufferSlice {
                buffer: workspace,
                offset: projection_bf16_offset,
            },
            rope,
            output,
            status,
            shape,
        )?;
        Ok(S14Ratio4IndexQueryChainDispatch {
            qr_to_f32,
            projection,
            projection_to_bf16,
            postprocess,
        })
    }

    /// # Safety
    /// 所有绑定资源必须活到 command 完成；函数内部固定插入四阶段 compute barrier。
    pub unsafe fn cmd_chain(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        numeric_exact: &S14NumericPipelines,
        bf16_to_f32: &S14Bf16ToF32Pipeline,
        f32_to_bf16: &S14F32ToBf16Pipeline,
        dispatch: &S14Ratio4IndexQueryChainDispatch,
    ) {
        bf16_to_f32.cmd(ctx, command, &dispatch.qr_to_f32);
        compute_to_compute_barrier(ctx, command);
        numeric_exact.cmd_fp8_matvec(ctx, command, &dispatch.projection);
        compute_to_compute_barrier(ctx, command);
        f32_to_bf16.cmd(ctx, command, &dispatch.projection_to_bf16);
        compute_to_compute_barrier(ctx, command);
        self.cmd(ctx, command, &dispatch.postprocess);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

pub struct S14SparseIndexerPipeline {
    pipeline: ComputePipeline,
}

pub struct S14SparseIndexerDispatch {
    pub binder: DescriptorBinder,
    compressed_count: u32,
}

impl S14SparseIndexerPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_SPARSE_INDEXER_SPV, 6, 4)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        query: StorageBufferSlice<'_>,
        index_cache: StorageBufferSlice<'_>,
        head_weights: StorageBufferSlice<'_>,
        output_scores: StorageBufferSlice<'_>,
        output_indices: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
        compressed_count: u32,
    ) -> Result<S14SparseIndexerDispatch> {
        if compressed_count == 0 || compressed_count > S14_INDEX_TOP_K {
            bail!("ratio4 indexer compressed_count 必须是1..=512");
        }
        let cache_bytes = u64::from(compressed_count) * u64::from(S14_INDEX_HEAD_DIM) * 2;
        let output_bytes = u64::from(compressed_count) * 4;
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (query.buffer, query.offset, INDEX_QUERY_BYTES),
                (index_cache.buffer, index_cache.offset, cache_bytes),
                (head_weights.buffer, head_weights.offset, INDEX_WEIGHT_BYTES),
                (output_scores.buffer, output_scores.offset, output_bytes),
                (output_indices.buffer, output_indices.offset, output_bytes),
                (status.buffer, status.offset, STATUS_BYTES),
            ],
        )?;
        Ok(S14SparseIndexerDispatch {
            binder,
            compressed_count,
        })
    }

    /// # Safety
    /// descriptor 资源必须活到 command 完成；调用前后由上层建立 compute barrier。
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14SparseIndexerDispatch,
    ) {
        let push = dispatch.compressed_count.to_le_bytes();
        bind_dispatch(ctx, command, &self.pipeline, &dispatch.binder, 1, &push);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

pub struct S14SparseAttentionPipeline {
    pipeline: ComputePipeline,
}

pub struct S14SparseAttentionDispatch {
    pub binder: DescriptorBinder,
    shape: S14SparseAttentionShape,
}

impl S14SparseAttentionPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_SPARSE_ATTENTION_SPV, 9, 28)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        query: StorageBufferSlice<'_>,
        window_kv: StorageBufferSlice<'_>,
        current_kv: StorageBufferSlice<'_>,
        compressed_kv: StorageBufferSlice<'_>,
        compressed_indices: StorageBufferSlice<'_>,
        sink: StorageBufferSlice<'_>,
        rope: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
        shape: S14SparseAttentionShape,
    ) -> Result<S14SparseAttentionDispatch> {
        let optional_count = shape.compressed_count.max(1);
        let compressed_bytes = u64::from(optional_count) * KV_ROW_BYTES;
        let index_bytes = u64::from(optional_count) * 4;
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (query.buffer, query.offset, QUERY_BYTES),
                (window_kv.buffer, window_kv.offset, WINDOW_BYTES),
                (current_kv.buffer, current_kv.offset, KV_ROW_BYTES),
                (compressed_kv.buffer, compressed_kv.offset, compressed_bytes),
                (
                    compressed_indices.buffer,
                    compressed_indices.offset,
                    index_bytes,
                ),
                (sink.buffer, sink.offset, SINK_BYTES),
                (rope.buffer, rope.offset, ROPE_BYTES),
                (output.buffer, output.offset, QUERY_BYTES),
                (status.buffer, status.offset, STATUS_BYTES),
            ],
        )?;
        Ok(S14SparseAttentionDispatch { binder, shape })
    }

    /// # Safety
    /// descriptor 资源必须活到 command 完成；调用前后由上层建立 compute barrier。
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14SparseAttentionDispatch,
    ) {
        let shape = dispatch.shape;
        let mut push = [0u8; 28];
        for (index, value) in [
            S14_POSITION0_HEADS,
            S14_POSITION0_HEAD_DIM,
            shape.position,
            shape.window_start,
            shape.previous_count,
            shape.compressed_count,
            u32::from(shape.implicit_compressed_indices),
        ]
        .into_iter()
        .enumerate()
        {
            push[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        bind_dispatch(
            ctx,
            command,
            &self.pipeline,
            &dispatch.binder,
            S14_POSITION0_HEADS,
            &push,
        );
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

pub struct S14SparseAttentionPipelines {
    pub index_query: S14Ratio4IndexQueryPipeline,
    pub indexer: S14SparseIndexerPipeline,
    pub attention: S14SparseAttentionPipeline,
}

#[derive(Clone, Copy)]
pub struct S14Ratio4SparseAttentionBindings<'a> {
    pub raw_index_query: StorageBufferSlice<'a>,
    pub rope: StorageBufferSlice<'a>,
    pub processed_index_query: StorageBufferSlice<'a>,
    pub index_cache: StorageBufferSlice<'a>,
    pub head_weights: StorageBufferSlice<'a>,
    pub index_scores: StorageBufferSlice<'a>,
    pub compressed_indices: StorageBufferSlice<'a>,
    pub query: StorageBufferSlice<'a>,
    pub window_kv: StorageBufferSlice<'a>,
    pub current_kv: StorageBufferSlice<'a>,
    pub compressed_kv: StorageBufferSlice<'a>,
    pub sink: StorageBufferSlice<'a>,
    pub output: StorageBufferSlice<'a>,
    pub status: StorageBufferSlice<'a>,
}

pub struct S14SparseAttentionRecording {
    binders: Vec<DescriptorBinder>,
}

impl S14SparseAttentionRecording {
    pub fn destroy(self, ctx: &VulkanContext) {
        for binder in self.binders {
            binder.destroy(ctx);
        }
    }
}

impl S14SparseAttentionPipelines {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        let index_query = S14Ratio4IndexQueryPipeline::new(ctx)?;
        let indexer = match S14SparseIndexerPipeline::new(ctx) {
            Ok(value) => value,
            Err(error) => {
                index_query.destroy(ctx);
                return Err(error);
            }
        };
        let attention = match S14SparseAttentionPipeline::new(ctx) {
            Ok(value) => value,
            Err(error) => {
                indexer.destroy(ctx);
                index_query.destroy(ctx);
                return Err(error);
            }
        };
        Ok(Self {
            index_query,
            indexer,
            attention,
        })
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.attention.destroy(ctx);
        self.indexer.destroy(ctx);
        self.index_query.destroy(ctx);
    }

    /// 在一个已 recording 的 command buffer 中闭合 ratio4 compressed sparse
    /// attention：index-query 后处理 → score/ReLU/weight/top-k → 按真实顺序 attention。
    /// 三个阶段之间只插入 GPU compute barrier，不产生 host wait/readback。
    ///
    /// # Safety
    /// 所有 slice 与返回的 descriptor 必须活到 command 完成；status 必须在调用前清零。
    pub unsafe fn record_ratio4(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        bindings: S14Ratio4SparseAttentionBindings<'_>,
        shape: S14SparseAttentionShape,
    ) -> Result<S14SparseAttentionRecording> {
        if shape.compressed_count == 0 {
            bail!("ratio4 compressed sparse attention 至少需要1个 compressed block");
        }
        let mut binders = Vec::with_capacity(3);
        let result = (|| -> Result<()> {
            let dispatch = self.index_query.bind_slices(
                ctx,
                bindings.raw_index_query,
                bindings.rope,
                bindings.processed_index_query,
                bindings.status,
                S14Ratio4IndexQueryShape::new(shape.position, 4)?,
            )?;
            self.index_query.cmd(ctx, command, &dispatch);
            binders.push(dispatch.binder);
            compute_to_compute_barrier(ctx, command);

            let dispatch = self.indexer.bind_slices(
                ctx,
                bindings.processed_index_query,
                bindings.index_cache,
                bindings.head_weights,
                bindings.index_scores,
                bindings.compressed_indices,
                bindings.status,
                shape.compressed_count,
            )?;
            self.indexer.cmd(ctx, command, &dispatch);
            binders.push(dispatch.binder);
            compute_to_compute_barrier(ctx, command);

            let dispatch = self.attention.bind_slices(
                ctx,
                bindings.query,
                bindings.window_kv,
                bindings.current_kv,
                bindings.compressed_kv,
                bindings.compressed_indices,
                bindings.sink,
                bindings.rope,
                bindings.output,
                bindings.status,
                shape,
            )?;
            self.attention.cmd(ctx, command, &dispatch);
            binders.push(dispatch.binder);
            Ok(())
        })();
        if let Err(error) = result {
            for binder in binders.drain(..) {
                binder.destroy(ctx);
            }
            return Err(error);
        }
        Ok(S14SparseAttentionRecording { binders })
    }
}

unsafe fn compute_to_compute_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
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

unsafe fn bind_dispatch(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    binder: &DescriptorBinder,
    groups_x: u32,
    push: &[u8],
) {
    ctx.device
        .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
    ctx.device.cmd_bind_descriptor_sets(
        command,
        vk::PipelineBindPoint::COMPUTE,
        pipeline.layout,
        0,
        &[binder.set],
        &[],
    );
    if !push.is_empty() {
        ctx.device.cmd_push_constants(
            command,
            pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push,
        );
    }
    ctx.device.cmd_dispatch(command, groups_x, 1, 1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn sha256_bf16(values: &[u16]) -> String {
        let mut hasher = Sha256::new();
        for value in values {
            hasher.update(value.to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    #[test]
    fn ratio4_index_query_shape_covers_first_ring_cycle() {
        assert_eq!(
            S14Ratio4IndexQueryShape::new(4, 4).unwrap(),
            S14Ratio4IndexQueryShape { position: 4 }
        );
        assert!(S14Ratio4IndexQueryShape::new(126, 4).is_ok());
        assert!(S14Ratio4IndexQueryShape::new(127, 4).is_ok());
        assert!(S14Ratio4IndexQueryShape::new(128, 4).is_ok());
        assert!(S14Ratio4IndexQueryShape::new(254, 4).is_ok());
        assert!(S14Ratio4IndexQueryShape::new(255, 4).is_ok());
        assert!(S14Ratio4IndexQueryShape::new(2047, 4).is_ok());
        assert!(S14Ratio4IndexQueryShape::new(2050, 4).is_ok());
        assert!(S14Ratio4IndexQueryShape::new(2051, 4).is_ok());
        assert!(S14Ratio4IndexQueryShape::new(200_000, 4).is_ok());
        assert!(S14Ratio4IndexQueryShape::new(3, 4).is_err());
        assert!(S14Ratio4IndexQueryShape::new(4, 128).is_err());
    }

    #[test]
    fn ratio4_index_query_reference_matches_independent_torch_bf16_fixture() {
        let mut qr = vec![0u16; S14_INDEX_QUERY_INPUT_DIM as usize];
        qr[0] = 0x3f80; // +1.0
        let mut weight =
            vec![0u8; (S14_INDEX_QUERY_OUTPUT_DIM * S14_INDEX_QUERY_INPUT_DIM) as usize];
        for row in 0..S14_INDEX_QUERY_OUTPUT_DIM as usize {
            weight[row * S14_INDEX_QUERY_INPUT_DIM as usize] = 0x38; // E4M3FN +1.0
        }
        let scale = vec![
            127u8;
            (S14_INDEX_QUERY_OUTPUT_DIM / 128 * S14_INDEX_QUERY_INPUT_DIM / 128)
                as usize
        ];

        let output = reference_ratio4_index_query(&qr, &weight, &scale, 4).unwrap();
        assert_eq!(output.len(), S14_INDEX_QUERY_OUTPUT_DIM as usize);
        assert_eq!(
            sha256_bf16(&output),
            "121f068d6b9c977ea0bb4a4282f8e63755cd89288bbe7092a7daa095dc9c3bb5"
        );
    }

    #[test]
    fn sparse_attention_shape_covers_position4_position7_and_ring_wrap() {
        assert_eq!(
            S14SparseAttentionShape::new(4, 0, 4, 1).unwrap(),
            S14SparseAttentionShape {
                position: 4,
                window_start: 0,
                previous_count: 4,
                compressed_count: 1,
                implicit_compressed_indices: false,
            }
        );
        assert!(S14SparseAttentionShape::new(7, 0, 7, 2).is_ok());
        assert!(S14SparseAttentionShape::new(127, 0, 127, 32).is_ok());
        assert!(S14SparseAttentionShape::new(128, 1, 127, 32).is_ok());
        assert!(S14SparseAttentionShape::new(128, 0, 127, 32).is_err());
        assert!(S14SparseAttentionShape::new(1, 0, 0, 0).is_err());
        assert!(S14SparseAttentionShape::new(2048, 1, 127, 513).is_err());
    }

    #[test]
    fn position2051_separates_513_history_rows_from_512_attention_rows() {
        let mut state = polaris_s14_runner::NativeState::decode_layout_for(
            polaris_s14_runner::GraphProfile::FullDepth43NativeTop6,
            4096,
        )
        .unwrap();
        state.position = 2051;
        let history = S14Ratio4HistoryLayout::build(&state, 2, 513).unwrap();
        let plan = S14Ratio4PagedAttentionPlan::build(2051, 4, 127, &history).unwrap();
        assert_eq!(plan.logical_compressed_count, 513);
        assert_eq!(plan.selected_count, 512);
        assert_eq!(plan.indexer_pages.len(), 2);
        assert_eq!(plan.indexer_pages[0].logical_rows, 0..512);
        assert_eq!(plan.indexer_pages[1].logical_rows, 512..513);
        assert_eq!(plan.attention_shape.compressed_count, 512);
        assert!(S14Ratio4PagedAttentionPlan::build(
            2051,
            4,
            127,
            &S14Ratio4HistoryLayout::build(&state, 2, 512).unwrap()
        )
        .is_err());
    }

    #[test]
    fn kv_sequence_preserves_ring_order_and_real_compressed_ranking() {
        let shape = S14SparseAttentionShape::new(128, 1, 127, 3).unwrap();
        let sequence = shape.kv_sequence(&[2, 0, 1]).unwrap();
        assert_eq!(sequence[0], S14SparseKvSource::Window(1));
        assert_eq!(sequence[126], S14SparseKvSource::Window(127));
        assert_eq!(sequence[127], S14SparseKvSource::Current);
        assert_eq!(
            &sequence[128..],
            &[
                S14SparseKvSource::Compressed(2),
                S14SparseKvSource::Compressed(0),
                S14SparseKvSource::Compressed(1),
            ]
        );
        assert!(shape.kv_sequence(&[0, 0, 1]).is_err());
        assert!(shape.kv_sequence(&[0, 1, 3]).is_err());

        let ratio128 = S14SparseAttentionShape::new_ratio128(127, 0, 127, 1).unwrap();
        let ratio128_sequence = ratio128.kv_sequence(&[]).unwrap();
        assert_eq!(ratio128_sequence[127], S14SparseKvSource::Current);
        assert_eq!(ratio128_sequence[128], S14SparseKvSource::Compressed(0));
        assert!(ratio128.kv_sequence(&[0]).is_err());
    }

    #[test]
    fn reference_indexer_applies_relu_scaled_weights_and_stable_topk() {
        let mut query = vec![0u16; (S14_INDEX_HEADS * S14_INDEX_HEAD_DIM) as usize];
        query[0] = 0x3f80; // head0/d0 = +1
        query[S14_INDEX_HEAD_DIM as usize] = 0xbf80; // head1/d0 = -1
        let mut cache = vec![0u16; 4 * S14_INDEX_HEAD_DIM as usize];
        cache[0] = 0x3f80; // block0 scores through head0
        cache[S14_INDEX_HEAD_DIM as usize] = 0xbf80; // block1 scores through head1
        let mut weights = vec![0.0f32; S14_INDEX_HEADS as usize];
        weights[0] = 2.0;
        weights[1] = 1.0;

        let selection = reference_sparse_indexer(&query, &cache, &weights, 4).unwrap();
        assert_eq!(selection.indices, vec![0, 1, 2, 3]);
        let scale = f32::from_bits(0x3c35_04f3);
        let scaled_two = bf16_to_f32(f32_to_bf16_rne(2.0 * scale));
        let scaled_one = bf16_to_f32(f32_to_bf16_rne(scale));
        assert_eq!(selection.scores[0].to_bits(), scaled_two.to_bits());
        assert_eq!(selection.scores[1].to_bits(), scaled_one.to_bits());
        assert_eq!(selection.scores[2], 0.0);
        assert_eq!(selection.scores[3], 0.0);

        cache[2 * S14_INDEX_HEAD_DIM as usize] = 0x4000; // block2/head0 dot=2
        let selection = reference_sparse_indexer(&query, &cache, &weights, 4).unwrap();
        assert_eq!(selection.indices, vec![2, 0, 1, 3]);
    }
}
