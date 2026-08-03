//! S14 route 后处理的真实 Vulkan 数值门。
//!
//! 真实 L0 物理 ID 只从 `tid2eid` I64 payload 的当前 token 行在线派生；
//! manifest/capture 记录的 expected expert IDs 从不作为 GPU 或 CPU oracle 输入。
//! L3 使用真实 bias payload，并与 synthetic logits 共同验证 bias top-6。

use anyhow::{bail, Context, Result};
use ash::vk;
use serde_json::Value;
use sha2::{Digest, Sha256};
use ssd_inference::{
    s14_route_postprocess::{
        postprocess_s14_route, S14RouteBias, S14RoutePostprocessKind, S14RoutePostprocessOutput,
    },
    s14_route_postprocess_gpu::{
        validate_route_postprocess_gpu_status, S14RouteBufferSlice, S14RoutePostprocessGpuBindings,
        S14RoutePostprocessGpuDispatch, S14RoutePostprocessGpuMode, S14RoutePostprocessGpuPipeline,
        S14_ROUTE_GPU_BIAS_BYTES, S14_ROUTE_GPU_EXPERTS, S14_ROUTE_GPU_LOGITS_BYTES,
        S14_ROUTE_GPU_OUTPUT_BYTES, S14_ROUTE_GPU_STATUS_BYTES, S14_ROUTE_GPU_STATUS_INVALID_MODE,
        S14_ROUTE_GPU_STATUS_INVALID_PHYSICAL_ID, S14_ROUTE_GPU_STATUS_NON_FINITE_BIAS,
        S14_ROUTE_GPU_TOP_K,
    },
    GpuBuffer, VulkanContext,
};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

const MANIFEST_RELATIVE: &str =
    "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json";
const MANIFEST_FORMAT: &str = "polaris-fulldepth43-position0-whole-token-manifest-v1";
const TID2EID_TENSOR: &str = "layers.0.ffn.gate.tid2eid";
const L3_BIAS_TENSOR: &str = "layers.3.ffn.gate.bias";
const FLOAT_TOLERANCE: f32 = 2.0e-5;

#[derive(Debug, Clone, Copy)]
struct ArenaLayout {
    logits: u64,
    aux: u64,
    expert_ids: u64,
    weights: u64,
    selected_scores: u64,
    ranking_scores: u64,
    status: u64,
    total: u64,
}

impl ArenaLayout {
    fn new(alignment: u64) -> Result<Self> {
        let mut cursor = alignment;
        let logits = alloc_span(&mut cursor, S14_ROUTE_GPU_LOGITS_BYTES, alignment)?;
        let aux = alloc_span(&mut cursor, S14_ROUTE_GPU_BIAS_BYTES, alignment)?;
        let expert_ids = alloc_span(&mut cursor, S14_ROUTE_GPU_OUTPUT_BYTES, alignment)?;
        let weights = alloc_span(&mut cursor, S14_ROUTE_GPU_OUTPUT_BYTES, alignment)?;
        let selected_scores = alloc_span(&mut cursor, S14_ROUTE_GPU_OUTPUT_BYTES, alignment)?;
        let ranking_scores = alloc_span(&mut cursor, S14_ROUTE_GPU_OUTPUT_BYTES, alignment)?;
        let status = alloc_span(&mut cursor, S14_ROUTE_GPU_STATUS_BYTES, alignment)?;
        Ok(Self {
            logits,
            aux,
            expert_ids,
            weights,
            selected_scores,
            ranking_scores,
            status,
            total: cursor,
        })
    }

    fn bindings<'a>(&self, arena: &'a GpuBuffer) -> S14RoutePostprocessGpuBindings<'a> {
        S14RoutePostprocessGpuBindings {
            logits: S14RouteBufferSlice::new(arena, self.logits),
            aux: S14RouteBufferSlice::new(arena, self.aux),
            expert_ids: S14RouteBufferSlice::new(arena, self.expert_ids),
            weights: S14RouteBufferSlice::new(arena, self.weights),
            selected_scores: S14RouteBufferSlice::new(arena, self.selected_scores),
            ranking_scores: S14RouteBufferSlice::new(arena, self.ranking_scores),
            status: S14RouteBufferSlice::new(arena, self.status),
        }
    }
}

#[derive(Debug)]
struct GpuOutput {
    status: u32,
    expert_ids: [u32; S14_ROUTE_GPU_TOP_K],
    weights: [f32; S14_ROUTE_GPU_TOP_K],
    selected_scores: [f32; S14_ROUTE_GPU_TOP_K],
    ranking_scores: [f32; S14_ROUTE_GPU_TOP_K],
}

#[derive(Debug)]
struct RealInputs {
    token_id: usize,
    physical_ids: [u32; S14_ROUTE_GPU_TOP_K],
    l3_bias: Vec<f32>,
    tid2eid_sha256: String,
    l3_bias_sha256: String,
}

fn main() -> Result<()> {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(MANIFEST_RELATIVE);
    let real = load_real_inputs(&manifest_path)?;
    let ctx = VulkanContext::init()?;
    let properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    let alignment = u64::from(properties.limits.min_storage_buffer_offset_alignment.max(1));
    let layout = ArenaLayout::new(alignment)?;
    let arena = host_buffer(&ctx, layout.total)?;
    let pipeline = S14RoutePostprocessGpuPipeline::new(&ctx)?;

    // 非零、设备对齐 descriptor offset 是这道门的核心 ABI，不允许退回 offset=0。
    if [
        layout.logits,
        layout.aux,
        layout.expert_ids,
        layout.weights,
        layout.selected_scores,
        layout.ranking_scores,
        layout.status,
    ]
    .iter()
    .any(|offset| *offset == 0 || *offset % alignment != 0)
    {
        bail!("S14 route arena produced zero/misaligned descriptor offset");
    }

    let mut overlap = layout.bindings(&arena);
    overlap.expert_ids = overlap.weights;
    if pipeline
        .bind_with_offsets(&ctx, S14RoutePostprocessGpuMode::BiasTop6, overlap)
        .is_ok()
    {
        bail!("S14 route wrapper accepted overlapping descriptor slices");
    }
    let mut offset_rejected = true;
    if alignment > 1 {
        let mut misaligned = layout.bindings(&arena);
        misaligned.logits.offset += 1;
        offset_rejected = pipeline
            .bind_with_offsets(&ctx, S14RoutePostprocessGpuMode::BiasTop6, misaligned)
            .is_err();
        if !offset_rejected {
            bail!("S14 route wrapper accepted a misaligned descriptor offset");
        }
    }

    let synthetic_bias_logits = vec![0.0f32; S14_ROUTE_GPU_EXPERTS];
    let synthetic_bias = vec![0.0f32; S14_ROUTE_GPU_EXPERTS];
    let cpu_tie = postprocess_s14_route(
        3,
        &synthetic_bias_logits,
        S14RoutePostprocessKind::ScoreTop6 {
            bias: Some(S14RouteBias::F32(&synthetic_bias)),
        },
    )?;
    let tie_gpu = run_case(
        &ctx,
        &pipeline,
        &arena,
        layout,
        S14RoutePostprocessGpuMode::BiasTop6,
        &synthetic_bias_logits,
        bytemuck::cast_slice(&synthetic_bias),
        None,
    )?;
    assert_cpu_parity("synthetic_bias_tie", &tie_gpu, &cpu_tie)?;
    if tie_gpu.expert_ids != [0, 1, 2, 3, 4, 5] {
        bail!("equal-score tie did not resolve to lower expert IDs");
    }

    let physical_logits = synthetic_logits(17, 53, 8.0);
    let synthetic_physical_ids = [254u32, 222, 33, 161, 38, 40];
    let synthetic_physical_cpu = cpu_physical(&physical_logits, &synthetic_physical_ids)?;
    let physical_gpu = run_case(
        &ctx,
        &pipeline,
        &arena,
        layout,
        S14RoutePostprocessGpuMode::PhysicalIds,
        &physical_logits,
        bytemuck::cast_slice(&synthetic_physical_ids),
        None,
    )?;
    assert_cpu_parity("synthetic_physical", &physical_gpu, &synthetic_physical_cpu)?;

    let real_l0_logits = synthetic_logits(29, 71, 11.0);
    let real_l0_cpu = cpu_physical(&real_l0_logits, &real.physical_ids)?;
    let real_l0_gpu = run_case(
        &ctx,
        &pipeline,
        &arena,
        layout,
        S14RoutePostprocessGpuMode::PhysicalIds,
        &real_l0_logits,
        bytemuck::cast_slice(&real.physical_ids),
        None,
    )?;
    assert_cpu_parity("real_l0_tid2eid", &real_l0_gpu, &real_l0_cpu)?;

    let real_l3_logits = synthetic_logits(43, 97, 13.0);
    let real_l3_cpu = postprocess_s14_route(
        3,
        &real_l3_logits,
        S14RoutePostprocessKind::ScoreTop6 {
            bias: Some(S14RouteBias::F32(&real.l3_bias)),
        },
    )?;
    let started = Instant::now();
    let real_l3_gpu = run_case(
        &ctx,
        &pipeline,
        &arena,
        layout,
        S14RoutePostprocessGpuMode::BiasTop6,
        &real_l3_logits,
        bytemuck::cast_slice(&real.l3_bias),
        None,
    )?;
    let real_l3_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert_cpu_parity("real_l3_bias", &real_l3_gpu, &real_l3_cpu)?;

    let mut nan_bias = synthetic_bias.clone();
    nan_bias[127] = f32::NAN;
    let nan_bias_gpu = run_case(
        &ctx,
        &pipeline,
        &arena,
        layout,
        S14RoutePostprocessGpuMode::BiasTop6,
        &synthetic_bias_logits,
        bytemuck::cast_slice(&nan_bias),
        None,
    )?;
    assert_fail_closed(
        "nan_bias",
        &nan_bias_gpu,
        S14_ROUTE_GPU_STATUS_NON_FINITE_BIAS,
    )?;

    let duplicate_ids = [7u32, 13, 7, 19, 23, 29];
    let duplicate_gpu = run_case(
        &ctx,
        &pipeline,
        &arena,
        layout,
        S14RoutePostprocessGpuMode::PhysicalIds,
        &physical_logits,
        bytemuck::cast_slice(&duplicate_ids),
        None,
    )?;
    assert_fail_closed(
        "duplicate_physical_id",
        &duplicate_gpu,
        S14_ROUTE_GPU_STATUS_INVALID_PHYSICAL_ID,
    )?;

    let invalid_mode_gpu = run_case(
        &ctx,
        &pipeline,
        &arena,
        layout,
        S14RoutePostprocessGpuMode::BiasTop6,
        &synthetic_bias_logits,
        bytemuck::cast_slice(&synthetic_bias),
        Some(99),
    )?;
    assert_fail_closed(
        "invalid_mode",
        &invalid_mode_gpu,
        S14_ROUTE_GPU_STATUS_INVALID_MODE,
    )?;

    println!(
        "status=pass gpu=\"{}\" descriptor_alignment={} nonzero_offsets=7 overlap_rejected=1 misaligned_rejected={} synthetic_bias_parity=pass tie_low_id=pass synthetic_physical_parity=pass real_l0_token_id={} real_l0_ids={:?} tid2eid_sha256={} real_l3_bias_parity=pass l3_ids={:?} l3_bias_sha256={} fail_closed_nan_bias=pass fail_closed_duplicate_id=pass fail_closed_invalid_mode=pass real_l3_wall_ms={real_l3_wall_ms:.4} capture_expected_ids_used_as_input=false",
        ctx.gpu_name,
        alignment,
        u8::from(offset_rejected),
        real.token_id,
        real.physical_ids,
        real.tid2eid_sha256,
        real_l3_gpu.expert_ids,
        real.l3_bias_sha256,
    );

    arena.destroy(&ctx);
    pipeline.destroy(&ctx);
    Ok(())
}

fn run_case(
    ctx: &VulkanContext,
    pipeline: &S14RoutePostprocessGpuPipeline,
    arena: &GpuBuffer,
    layout: ArenaLayout,
    mode: S14RoutePostprocessGpuMode,
    logits: &[f32],
    aux: &[u8],
    raw_mode: Option<u32>,
) -> Result<GpuOutput> {
    if logits.len() != S14_ROUTE_GPU_EXPERTS || aux.len() as u64 != mode.aux_bytes() {
        bail!("numeric case input shape drift");
    }
    let sentinel_u32 = [0xffff_ffffu32; S14_ROUTE_GPU_TOP_K];
    let sentinel_f32 = [12345.0f32; S14_ROUTE_GPU_TOP_K];
    unsafe {
        arena.write_at(layout.logits as usize, bytemuck::cast_slice(logits));
        arena.write_at(layout.aux as usize, aux);
        arena.write_at(
            layout.expert_ids as usize,
            bytemuck::cast_slice(&sentinel_u32),
        );
        for offset in [
            layout.weights,
            layout.selected_scores,
            layout.ranking_scores,
        ] {
            arena.write_at(offset as usize, bytemuck::cast_slice(&sentinel_f32));
        }
        arena.write_at(layout.status as usize, bytemuck::bytes_of(&0u32));
    }

    let dispatch = pipeline.bind_with_offsets(ctx, mode, layout.bindings(arena))?;
    dispatch_once(ctx, pipeline, &dispatch, raw_mode)?;
    let output = GpuOutput {
        status: mapped_one::<u32>(arena, layout.status),
        expert_ids: mapped_array::<u32, S14_ROUTE_GPU_TOP_K>(arena, layout.expert_ids),
        weights: mapped_array::<f32, S14_ROUTE_GPU_TOP_K>(arena, layout.weights),
        selected_scores: mapped_array::<f32, S14_ROUTE_GPU_TOP_K>(arena, layout.selected_scores),
        ranking_scores: mapped_array::<f32, S14_ROUTE_GPU_TOP_K>(arena, layout.ranking_scores),
    };
    dispatch.binder.destroy(ctx);
    Ok(output)
}

fn dispatch_once(
    ctx: &VulkanContext,
    pipeline: &S14RoutePostprocessGpuPipeline,
    dispatch: &S14RoutePostprocessGpuDispatch,
    raw_mode: Option<u32>,
) -> Result<()> {
    unsafe {
        let pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.qf_graphics),
            None,
        )?;
        let command = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        let fence = ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)?;
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        if let Some(raw_mode) = raw_mode {
            pipeline.cmd_raw_mode_for_validation(ctx, command, dispatch, raw_mode);
        } else {
            pipeline.cmd(ctx, command, dispatch);
        }
        ctx.device.end_command_buffer(command)?;
        let commands = [command];
        ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            fence,
        )?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    Ok(())
}

fn assert_cpu_parity(label: &str, gpu: &GpuOutput, cpu: &S14RoutePostprocessOutput) -> Result<()> {
    validate_route_postprocess_gpu_status(gpu.status)?;
    let expected_ids = cpu.expert_ids.map(u32::from);
    if gpu.expert_ids != expected_ids {
        bail!(
            "{label} expert ID mismatch: gpu={:?} cpu={:?}",
            gpu.expert_ids,
            expected_ids
        );
    }
    for (field, actual, expected) in [
        ("weights", &gpu.weights[..], &cpu.weights[..]),
        (
            "selected_scores",
            &gpu.selected_scores[..],
            &cpu.selected_scores[..],
        ),
        (
            "ranking_scores",
            &gpu.ranking_scores[..],
            &cpu.selected_ranking_scores[..],
        ),
    ] {
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        if !max_abs.is_finite() || max_abs > FLOAT_TOLERANCE {
            bail!("{label} {field} max_abs={max_abs} exceeds {FLOAT_TOLERANCE}");
        }
    }
    Ok(())
}

fn assert_fail_closed(label: &str, output: &GpuOutput, expected_status: u32) -> Result<()> {
    if output.status != expected_status
        || validate_route_postprocess_gpu_status(output.status).is_ok()
    {
        bail!(
            "{label} status mismatch: actual=0x{:08x} expected=0x{expected_status:08x}",
            output.status
        );
    }
    if output.expert_ids.iter().any(|value| *value != 0)
        || output.weights.iter().any(|value| value.to_bits() != 0)
        || output
            .selected_scores
            .iter()
            .any(|value| value.to_bits() != 0)
        || output
            .ranking_scores
            .iter()
            .any(|value| value.to_bits() != 0)
    {
        bail!("{label} leaked nonzero output after fail-closed status");
    }
    Ok(())
}

fn cpu_physical(
    logits: &[f32],
    ids: &[u32; S14_ROUTE_GPU_TOP_K],
) -> Result<S14RoutePostprocessOutput> {
    let ids_u16: Vec<u16> = ids
        .iter()
        .map(|value| u16::try_from(*value).context("physical ID exceeds u16"))
        .collect::<Result<_>>()?;
    Ok(postprocess_s14_route(
        0,
        logits,
        S14RoutePostprocessKind::Tid2EidPhysical {
            expert_ids: &ids_u16,
        },
    )?)
}

fn synthetic_logits(multiplier: usize, modulus: usize, divisor: f32) -> Vec<f32> {
    (0..S14_ROUTE_GPU_EXPERTS)
        .map(|expert| {
            let centered = (expert * multiplier % modulus) as i32 - (modulus as i32 / 2);
            centered as f32 / divisor
        })
        .collect()
}

fn load_real_inputs(manifest_path: &Path) -> Result<RealInputs> {
    let manifest_bytes = fs::read(manifest_path)
        .with_context(|| format!("read real whole-token manifest {}", manifest_path.display()))?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    if manifest["format"].as_str() != Some(MANIFEST_FORMAT) {
        bail!("whole-token manifest format drift");
    }
    let token_id = usize::try_from(
        manifest["input_token_id"]
            .as_u64()
            .context("manifest input_token_id missing")?,
    )?;
    let tid2eid = router_asset(&manifest, 0, TID2EID_TENSOR)?;
    validate_asset(tid2eid, "I64", &[129_280, 6], 6_205_440)?;
    let tid2eid_bytes = read_verified_asset(tid2eid)?;
    let row_start = token_id
        .checked_mul(S14_ROUTE_GPU_TOP_K * 8)
        .context("tid2eid row offset overflow")?;
    let row_end = row_start + S14_ROUTE_GPU_TOP_K * 8;
    if row_end > tid2eid_bytes.len() {
        bail!("input token {token_id} exceeds tid2eid row count");
    }
    let mut physical_ids = [0u32; S14_ROUTE_GPU_TOP_K];
    let mut unique = HashSet::new();
    for (slot, chunk) in tid2eid_bytes[row_start..row_end]
        .chunks_exact(8)
        .enumerate()
    {
        let value = i64::from_le_bytes(chunk.try_into().unwrap());
        if !(0..S14_ROUTE_GPU_EXPERTS as i64).contains(&value) {
            bail!("real tid2eid[{token_id},{slot}]={value} is outside 0..255");
        }
        let id = value as u32;
        if !unique.insert(id) {
            bail!("real tid2eid row has duplicate expert {id}");
        }
        physical_ids[slot] = id;
    }

    let l3_bias_asset = router_asset(&manifest, 3, L3_BIAS_TENSOR)?;
    validate_asset(l3_bias_asset, "F32", &[256], 1024)?;
    let l3_bias_bytes = read_verified_asset(l3_bias_asset)?;
    let l3_bias: Vec<f32> = l3_bias_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if l3_bias.len() != S14_ROUTE_GPU_EXPERTS || l3_bias.iter().any(|value| !value.is_finite()) {
        bail!("real L3 bias shape/non-finite drift");
    }
    Ok(RealInputs {
        token_id,
        physical_ids,
        l3_bias,
        tid2eid_sha256: tid2eid["sha256"].as_str().unwrap().to_owned(),
        l3_bias_sha256: l3_bias_asset["sha256"].as_str().unwrap().to_owned(),
    })
}

fn router_asset<'a>(manifest: &'a Value, layer: usize, tensor: &str) -> Result<&'a Value> {
    let layer_row = manifest["layers"]
        .as_array()
        .and_then(|layers| layers.get(layer))
        .with_context(|| format!("manifest L{layer} missing"))?;
    layer_row["assets"]["router"]
        .as_array()
        .and_then(|assets| {
            assets
                .iter()
                .find(|asset| asset["tensor"].as_str() == Some(tensor))
        })
        .with_context(|| format!("manifest router asset {tensor} missing"))
}

fn validate_asset(asset: &Value, dtype: &str, shape: &[u64], bytes: u64) -> Result<()> {
    let actual_shape: Vec<u64> = asset["shape"]
        .as_array()
        .context("asset shape missing")?
        .iter()
        .map(|value| value.as_u64().context("asset shape is not u64"))
        .collect::<Result<_>>()?;
    if asset["dtype"].as_str() != Some(dtype)
        || actual_shape != shape
        || asset["bytes"].as_u64() != Some(bytes)
        || asset["sha256"].as_str().map(str::len) != Some(64)
    {
        bail!(
            "real router asset contract drift: tensor={}",
            asset["tensor"]
        );
    }
    Ok(())
}

fn read_verified_asset(asset: &Value) -> Result<Vec<u8>> {
    let path = PathBuf::from(asset["path"].as_str().context("asset path missing")?);
    let expected_bytes = usize::try_from(asset["bytes"].as_u64().context("asset bytes missing")?)?;
    let expected_sha = asset["sha256"].as_str().context("asset sha256 missing")?;
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let actual_sha = sha256_bytes(&bytes);
    if bytes.len() != expected_bytes || actual_sha != expected_sha {
        bail!(
            "real asset size/SHA drift: path={} bytes={} expected_bytes={} sha={} expected_sha={}",
            path.display(),
            bytes.len(),
            expected_bytes,
            actual_sha,
            expected_sha
        );
    }
    Ok(bytes)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn alloc_span(cursor: &mut u64, bytes: u64, alignment: u64) -> Result<u64> {
    let start = align_up(*cursor, alignment)?;
    *cursor = start
        .checked_add(bytes)
        .context("route arena size overflow")?;
    Ok(start)
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("invalid storage buffer alignment {alignment}");
    }
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("alignment overflow")
}

fn host_buffer(ctx: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new(
        ctx,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )
}

fn mapped_one<T: Copy>(buffer: &GpuBuffer, offset: u64) -> T {
    unsafe { *((buffer.mapped() as *const u8).add(offset as usize) as *const T) }
}

fn mapped_array<T: Copy, const N: usize>(buffer: &GpuBuffer, offset: u64) -> [T; N] {
    unsafe {
        std::slice::from_raw_parts(
            (buffer.mapped() as *const u8).add(offset as usize) as *const T,
            N,
        )
        .try_into()
        .unwrap()
    }
}
