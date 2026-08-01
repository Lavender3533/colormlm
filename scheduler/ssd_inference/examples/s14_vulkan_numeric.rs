//! Real-byte numerical and timing evidence for the Polaris S14 Vulkan matvecs.
//!
//! No weights are downloaded. The program only reads the frozen local ABI
//! samples under `D:/models/Polaris-S14/abi_samples`.
//!
//! Run:
//!   cargo run --release --offline -p ssd_inference --example s14_vulkan_numeric

use anyhow::{bail, Context, Result};
use ash::vk;
use rayon::prelude::*;
use serde::Deserialize;
use ssd_inference::buffer::GpuBuffer;
use ssd_inference::device::VulkanContext;
use ssd_inference::s14_vulkan::{
    validate_e4m3fn_codes, validate_ue8m0_codes, S14MatvecShape, S14NumericPipelines,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

const MODEL_DIR: &str = "D:/models/Polaris-S14";
const ROUTE_MANIFEST: &str = "D:/models/Polaris-S14/l42_synthetic_route_manifest.json";
const MOE_REFERENCE: &str = "D:/models/Polaris-S14/l42_real_moe_reference.json";
const RANGE_CACHE: &str = "D:/models/Polaris-S14/range_cache";
const ITERATIONS: u32 = 100;

struct DeviceBuffers {
    upload_x: GpuBuffer,
    upload_weight: GpuBuffer,
    upload_scale: GpuBuffer,
    x: GpuBuffer,
    weight: GpuBuffer,
    scale: GpuBuffer,
    y: GpuBuffer,
    readback: GpuBuffer,
    x_bytes: u64,
    weight_bytes: u64,
    scale_bytes: u64,
    y_bytes: u64,
}

impl DeviceBuffers {
    fn new(ctx: &VulkanContext, x: &[f32], weight: &[u8], scale: &[u8], n: usize) -> Result<Self> {
        if weight.len() % 4 != 0 || scale.len() % 4 != 0 {
            bail!("real S14 uint-word buffers must be four-byte aligned");
        }
        let x_bytes = std::mem::size_of_val(x) as u64;
        let weight_bytes = weight.len() as u64;
        let scale_bytes = scale.len() as u64;
        let y_bytes = (n * std::mem::size_of::<f32>()) as u64;

        let upload_x = GpuBuffer::new_staging(ctx, x_bytes)?;
        let upload_weight = GpuBuffer::new_staging(ctx, weight_bytes)?;
        let upload_scale = GpuBuffer::new_staging(ctx, scale_bytes)?;
        unsafe {
            upload_x.write_at(0, bytemuck::cast_slice(x));
            upload_weight.write_at(0, weight);
            upload_scale.write_at(0, scale);
        }

        let input_usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let x_device = GpuBuffer::new_vram(ctx, x_bytes, input_usage)?;
        let weight_device = GpuBuffer::new_vram(ctx, weight_bytes, input_usage)?;
        let scale_device = GpuBuffer::new_vram(ctx, scale_bytes, input_usage)?;
        let y = GpuBuffer::new_vram(
            ctx,
            y_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        let readback = GpuBuffer::new(
            ctx,
            y_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?;

        Ok(Self {
            upload_x,
            upload_weight,
            upload_scale,
            x: x_device,
            weight: weight_device,
            scale: scale_device,
            y,
            readback,
            x_bytes,
            weight_bytes,
            scale_bytes,
            y_bytes,
        })
    }

    fn upload(&self, ctx: &VulkanContext) -> Result<()> {
        unsafe {
            let pool = make_command_pool(ctx)?;
            let cb = allocate_command_buffer(ctx, pool)?;
            ctx.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            copy(ctx, cb, &self.upload_x, &self.x, self.x_bytes);
            copy(
                ctx,
                cb,
                &self.upload_weight,
                &self.weight,
                self.weight_bytes,
            );
            copy(ctx, cb, &self.upload_scale, &self.scale, self.scale_bytes);
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            ctx.device.end_command_buffer(cb)?;
            submit_and_wait(ctx, cb)?;
            ctx.device.destroy_command_pool(pool, None);
        }
        Ok(())
    }

    fn output(&self, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(self.readback.mapped() as *const f32, n).to_vec() }
    }

    fn destroy(&self, ctx: &VulkanContext) {
        self.readback.destroy(ctx);
        self.y.destroy(ctx);
        self.scale.destroy(ctx);
        self.weight.destroy(ctx);
        self.x.destroy(ctx);
        self.upload_scale.destroy(ctx);
        self.upload_weight.destroy(ctx);
        self.upload_x.destroy(ctx);
    }
}

struct Timing {
    iterations: u32,
    gpu_kernel_ms_mean: f64,
    submit_readback_sync_ms: f64,
}

struct ErrorStats {
    max_abs: f32,
    mean_abs: f64,
    rmse: f64,
    max_rel_for_abs_ref_gt_1e_5: f32,
}

struct MatrixRun {
    actual: Vec<f32>,
    expected: Vec<f32>,
}

#[derive(Deserialize)]
struct RouteManifest {
    format: String,
    layer: u32,
    input: String,
    expert_ids: Vec<u32>,
    route_weights: Vec<f32>,
    weight_sum: f32,
    entries: Vec<RouteEntry>,
}

#[derive(Deserialize)]
struct RouteEntry {
    tensor: String,
    expert_id: u32,
    bytes: u64,
    path: PathBuf,
}

#[derive(Deserialize)]
struct MoeReference {
    format: String,
    layer: u32,
    input: String,
    expert_ids: Vec<u32>,
    route_weights: Vec<f32>,
    expert_stats: Vec<ExpertReference>,
    routed_weighted: FrozenStats,
    shared: FrozenStats,
    moe_sum: FrozenStats,
}

#[derive(Deserialize)]
struct ExpertReference {
    expert_id: u32,
    route_weight: f32,
    l2: f64,
    mean: f64,
    maxabs: f32,
}

#[derive(Deserialize)]
struct FrozenStats {
    l2: f64,
    mean: f64,
    maxabs: f32,
    f32_le_sha256: String,
}

#[derive(Deserialize)]
struct BaseManifest {
    format: String,
    layer: u32,
    entry_count: usize,
    bytes: u64,
    entries: Vec<BaseEntry>,
}

#[derive(Deserialize)]
struct BaseEntry {
    tensor: String,
    bytes: u64,
    path: PathBuf,
}

struct BasicStats {
    l2: f64,
    mean: f64,
    maxabs: f32,
}

#[derive(Deserialize)]
struct CacheSidecar {
    identity: CacheIdentity,
    bytes: u64,
}

#[derive(Deserialize)]
struct CacheIdentity {
    source_file: String,
    start: u64,
    end: u64,
}

#[derive(Clone, Copy)]
struct SharedRange {
    component: &'static str,
    kind: &'static str,
    start: u64,
    end: u64,
    bytes: u64,
}

const SHARED_RANGES: [SharedRange; 6] = [
    SharedRange {
        component: "w1",
        kind: "scale",
        start: 228_290_160,
        end: 228_290_671,
        bytes: 512,
    },
    SharedRange {
        component: "w2",
        kind: "scale",
        start: 228_290_672,
        end: 228_291_183,
        bytes: 512,
    },
    SharedRange {
        component: "w3",
        kind: "scale",
        start: 228_291_184,
        end: 228_291_695,
        bytes: 512,
    },
    SharedRange {
        component: "w1",
        kind: "weight",
        start: 343_635_056,
        end: 352_023_663,
        bytes: 8_388_608,
    },
    SharedRange {
        component: "w2",
        kind: "weight",
        start: 352_023_664,
        end: 360_412_271,
        bytes: 8_388_608,
    },
    SharedRange {
        component: "w3",
        kind: "weight",
        start: 360_412_272,
        end: 368_800_879,
        bytes: 8_388_608,
    },
];

fn main() -> Result<()> {
    println!("Polaris S14 Vulkan numerical kernels (real local ABI slices)");
    let ctx = VulkanContext::init()?;
    println!("GPU: {}", ctx.gpu_name);
    if !ctx.gpu_name.contains("RX 5700 XT") {
        bail!("evidence run requires RX 5700 XT, found {}", ctx.gpu_name);
    }
    let queue_props = unsafe {
        ctx.instance
            .get_physical_device_queue_family_properties(ctx.physical)
    };
    let timestamp_bits = queue_props[ctx.qf_graphics as usize].timestamp_valid_bits;
    if timestamp_bits == 0 {
        bail!("graphics/compute queue does not expose timestamp queries");
    }
    let physical_properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    let timestamp_period_ns = physical_properties.limits.timestamp_period as f64;
    println!(
        "Vulkan device: vendor=0x{:04x}, device=0x{:04x}, driver_version=0x{:08x}, \
         timestamp_valid_bits={}, timestamp_period_ns={timestamp_period_ns}",
        physical_properties.vendor_id,
        physical_properties.device_id,
        physical_properties.driver_version,
        timestamp_bits,
    );

    let pipelines = S14NumericPipelines::new(&ctx)?;
    run_route_first_moe(&ctx, &pipelines, timestamp_bits, timestamp_period_ns)?;
    run_real_fp8_wq_a(&ctx, &pipelines, timestamp_bits, timestamp_period_ns)?;
    pipelines.destroy(&ctx);
    println!("claim: real GPU matvec parity only; no token/s and no full S14 forward claim");
    Ok(())
}

fn run_route_first_moe(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
) -> Result<()> {
    let manifest: RouteManifest =
        serde_json::from_slice(&std::fs::read(ROUTE_MANIFEST).context("read L42 route manifest")?)
            .context("parse L42 route manifest")?;
    let reference: MoeReference =
        serde_json::from_slice(&std::fs::read(MOE_REFERENCE).context("read L42 MoE reference")?)
            .context("parse L42 MoE reference")?;
    validate_route_contract(&manifest, &reference)?;
    println!(
        "route-first L42 top6={:?}, weight_sum={:.9}, cached_ranges={}",
        manifest.expert_ids,
        manifest.weight_sum,
        manifest.entries.len()
    );

    let x: Vec<f32> = (0..4096).map(|i| ((i as f32) * 0.013).sin()).collect();
    let mut routed_actual = vec![0.0f32; 4096];
    let mut routed_expected = vec![0.0f32; 4096];

    for (&expert_id, &route_weight) in manifest.expert_ids.iter().zip(&manifest.route_weights) {
        let (w1, s1) = load_route_pair(&manifest, expert_id, "w1")?;
        let gate = run_mxfp4_matrix(
            ctx,
            pipelines,
            timestamp_bits,
            timestamp_period_ns,
            &format!("MXFP4 L42/E{expert_id} w1"),
            &x,
            &w1,
            &s1,
            S14MatvecShape::new(2048, 4096)?,
        )?;
        let (w3, s3) = load_route_pair(&manifest, expert_id, "w3")?;
        let up = run_mxfp4_matrix(
            ctx,
            pipelines,
            timestamp_bits,
            timestamp_period_ns,
            &format!("MXFP4 L42/E{expert_id} w3"),
            &x,
            &w3,
            &s3,
            S14MatvecShape::new(2048, 4096)?,
        )?;
        let hidden_actual = swiglu_limit(&gate.actual, &up.actual)?;
        let hidden_expected = swiglu_limit(&gate.expected, &up.expected)?;

        let (w2, s2) = load_route_pair(&manifest, expert_id, "w2")?;
        let down = run_mxfp4_matrix(
            ctx,
            pipelines,
            timestamp_bits,
            timestamp_period_ns,
            &format!("MXFP4 L42/E{expert_id} w2"),
            &hidden_actual,
            &w2,
            &s2,
            S14MatvecShape::new(4096, 2048)?,
        )?;
        let expert_expected = cpu_mxfp4_matvec(
            &hidden_expected,
            &w2,
            &s2,
            S14MatvecShape::new(4096, 2048)?.validate_mxfp4()?,
        );
        let expert_error = error_stats(&down.actual, &expert_expected)?;
        enforce_error(&expert_error, 1.5e-4, 2.0e-5)?;
        let expert_reference = reference
            .expert_stats
            .iter()
            .find(|item| item.expert_id == expert_id)
            .ok_or_else(|| anyhow::anyhow!("missing E{expert_id} frozen reference"))?;
        if (expert_reference.route_weight - route_weight).abs() > f32::EPSILON {
            bail!("E{expert_id} route weight drift");
        }
        compare_frozen(
            &format!("routed E{expert_id}"),
            &down.actual,
            &FrozenStats {
                l2: expert_reference.l2,
                mean: expert_reference.mean,
                maxabs: expert_reference.maxabs,
                f32_le_sha256: "per-expert hash not recorded".into(),
            },
        )?;
        println!(
            "routed E{expert_id} end_to_end_error: max_abs={:.9e}, rmse={:.9e}",
            expert_error.max_abs, expert_error.rmse
        );
        for i in 0..4096 {
            routed_actual[i] += route_weight * down.actual[i];
            routed_expected[i] += route_weight * expert_expected[i];
        }
    }

    let routed_error = error_stats(&routed_actual, &routed_expected)?;
    enforce_error(&routed_error, 2.0e-4, 2.5e-5)?;
    compare_frozen(
        "routed weighted top6",
        &routed_actual,
        &reference.routed_weighted,
    )?;
    println!(
        "routed weighted error: max_abs={:.9e}, rmse={:.9e}, frozen_cpu_sha={}",
        routed_error.max_abs, routed_error.rmse, reference.routed_weighted.f32_le_sha256
    );

    let (shared_actual, shared_expected) =
        run_shared_fp8(ctx, pipelines, timestamp_bits, timestamp_period_ns, &x)?;
    let shared_error = error_stats(&shared_actual, &shared_expected)?;
    enforce_error(&shared_error, 2.0e-4, 2.5e-5)?;
    compare_frozen("shared FP8", &shared_actual, &reference.shared)?;
    println!(
        "shared FP8 error: max_abs={:.9e}, rmse={:.9e}, frozen_cpu_sha={}",
        shared_error.max_abs, shared_error.rmse, reference.shared.f32_le_sha256
    );

    let moe_actual: Vec<f32> = routed_actual
        .iter()
        .zip(&shared_actual)
        .map(|(&routed, &shared)| routed + shared)
        .collect();
    let moe_expected: Vec<f32> = routed_expected
        .iter()
        .zip(&shared_expected)
        .map(|(&routed, &shared)| routed + shared)
        .collect();
    let moe_error = error_stats(&moe_actual, &moe_expected)?;
    enforce_error(&moe_error, 3.0e-4, 4.0e-5)?;
    compare_frozen("L42 MoE sum", &moe_actual, &reference.moe_sum)?;
    println!(
        "L42 MoE sum error: max_abs={:.9e}, rmse={:.9e}, frozen_cpu_sha={}",
        moe_error.max_abs, moe_error.rmse, reference.moe_sum.f32_le_sha256
    );
    Ok(())
}

fn run_real_fp8_wq_a(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
) -> Result<()> {
    let path = Path::new(MODEL_DIR).join("l42_base_cache_manifest.json");
    let manifest: BaseManifest = serde_json::from_slice(&std::fs::read(&path)?)?;
    if manifest.format != "polaris-l42-base-cache-snapshot-v1"
        || manifest.layer != 42
        || manifest.entry_count != 34
        || manifest.entries.len() != 34
        || manifest.bytes != 142_131_800
    {
        bail!("L42 base cache manifest contract drift");
    }
    let weight_entry = base_entry(&manifest, "layers.42.attn.wq_a.weight")?;
    let scale_entry = base_entry(&manifest, "layers.42.attn.wq_a.scale")?;
    let weight = read_exact(&weight_entry.path, weight_entry.bytes as usize)?;
    let scale = read_exact(&scale_entry.path, scale_entry.bytes as usize)?;
    let shape = S14MatvecShape::new(1024, 4096)?.validate_fp8()?;
    let x: Vec<f32> = (0..shape.k).map(|i| ((i as f32) * 0.013).sin()).collect();
    run_fp8_matrix(
        ctx,
        pipelines,
        timestamp_bits,
        timestamp_period_ns,
        "FP8 L42 base-cache wq_a",
        &x,
        &weight,
        &scale,
        shape,
    )?;
    Ok(())
}

fn validate_route_contract(manifest: &RouteManifest, reference: &MoeReference) -> Result<()> {
    let expected_ids = [48, 153, 144, 83, 221, 127];
    if manifest.format != "polaris-l42-synthetic-route-cache-v1"
        || reference.format != "polaris-l42-real-routed-moe-reference-v1"
        || manifest.layer != 42
        || reference.layer != 42
        || manifest.input != "sin(arange(4096)*0.013)"
        || reference.input != manifest.input
        || manifest.expert_ids != expected_ids
        || reference.expert_ids != manifest.expert_ids
        || reference.route_weights != manifest.route_weights
        || manifest.entries.len() != 36
        || manifest.weight_sum.to_bits() != 1.5f32.to_bits()
    {
        bail!("L42 route/reference contract drift");
    }
    let sum: f32 = manifest.route_weights.iter().sum();
    if (sum - 1.5).abs() > 2.0e-7 {
        bail!("L42 route weights sum to {sum}, expected 1.5");
    }
    Ok(())
}

fn load_route_pair(
    manifest: &RouteManifest,
    expert_id: u32,
    component: &str,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let prefix = format!("layers.42.ffn.experts.{expert_id}.{component}");
    let weight = route_entry(manifest, expert_id, &format!("{prefix}.weight"))?;
    let scale = route_entry(manifest, expert_id, &format!("{prefix}.scale"))?;
    if weight.bytes != 4_194_304 || scale.bytes != 262_144 {
        bail!("E{expert_id}/{component} byte contract drift");
    }
    Ok((
        read_exact(&weight.path, weight.bytes as usize)?,
        read_exact(&scale.path, scale.bytes as usize)?,
    ))
}

fn route_entry<'a>(
    manifest: &'a RouteManifest,
    expert_id: u32,
    tensor: &str,
) -> Result<&'a RouteEntry> {
    manifest
        .entries
        .iter()
        .find(|entry| entry.expert_id == expert_id && entry.tensor == tensor)
        .ok_or_else(|| anyhow::anyhow!("route manifest missing {tensor}"))
}

fn base_entry<'a>(manifest: &'a BaseManifest, tensor: &str) -> Result<&'a BaseEntry> {
    manifest
        .entries
        .iter()
        .find(|entry| entry.tensor == tensor)
        .ok_or_else(|| anyhow::anyhow!("base manifest missing {tensor}"))
}

fn run_mxfp4_matrix(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    label: &str,
    x: &[f32],
    weight: &[u8],
    scale: &[u8],
    shape: S14MatvecShape,
) -> Result<MatrixRun> {
    let shape = shape.validate_mxfp4()?;
    validate_ue8m0_codes(scale)?;
    let cpu_start = Instant::now();
    let expected = cpu_mxfp4_matvec(x, weight, scale, shape);
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;
    let buffers = DeviceBuffers::new(ctx, x, weight, scale, shape.n as usize)?;
    buffers.upload(ctx)?;
    let dispatch = pipelines.bind_mxfp4(
        ctx,
        shape,
        &buffers.x,
        &buffers.weight,
        &buffers.scale,
        &buffers.y,
    )?;
    let timing = benchmark(
        ctx,
        &buffers,
        timestamp_bits,
        timestamp_period_ns,
        |cb| unsafe { pipelines.cmd_mxfp4_matvec(ctx, cb, &dispatch) },
    )?;
    let actual = buffers.output(shape.n as usize);
    let error = error_stats(&actual, &expected)?;
    print_evidence(label, shape, cpu_ms, &timing, &error);
    enforce_error(&error, 2.5e-5, 3.5e-6)?;
    dispatch.binder.destroy(ctx);
    buffers.destroy(ctx);
    Ok(MatrixRun { actual, expected })
}

fn run_fp8_matrix(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    label: &str,
    x: &[f32],
    weight: &[u8],
    scale: &[u8],
    shape: S14MatvecShape,
) -> Result<MatrixRun> {
    let shape = shape.validate_fp8()?;
    validate_e4m3fn_codes(weight)?;
    validate_ue8m0_codes(scale)?;
    let cpu_start = Instant::now();
    let expected = cpu_fp8_matvec(x, weight, scale, shape);
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;
    let buffers = DeviceBuffers::new(ctx, x, weight, scale, shape.n as usize)?;
    buffers.upload(ctx)?;
    let dispatch = pipelines.bind_fp8(
        ctx,
        shape,
        &buffers.x,
        &buffers.weight,
        &buffers.scale,
        &buffers.y,
    )?;
    let timing = benchmark(
        ctx,
        &buffers,
        timestamp_bits,
        timestamp_period_ns,
        |cb| unsafe { pipelines.cmd_fp8_matvec(ctx, cb, &dispatch) },
    )?;
    let actual = buffers.output(shape.n as usize);
    let error = error_stats(&actual, &expected)?;
    print_evidence(label, shape, cpu_ms, &timing, &error);
    enforce_error(&error, 2.5e-5, 3.5e-6)?;
    dispatch.binder.destroy(ctx);
    buffers.destroy(ctx);
    Ok(MatrixRun { actual, expected })
}

fn run_shared_fp8(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    x: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (w1, s1) = load_shared_pair("w1")?;
    let gate = run_fp8_matrix(
        ctx,
        pipelines,
        timestamp_bits,
        timestamp_period_ns,
        "FP8 L42 shared w1",
        x,
        &w1,
        &s1,
        S14MatvecShape::new(2048, 4096)?,
    )?;
    let (w3, s3) = load_shared_pair("w3")?;
    let up = run_fp8_matrix(
        ctx,
        pipelines,
        timestamp_bits,
        timestamp_period_ns,
        "FP8 L42 shared w3",
        x,
        &w3,
        &s3,
        S14MatvecShape::new(2048, 4096)?,
    )?;
    let hidden_actual = swiglu_limit(&gate.actual, &up.actual)?;
    let hidden_expected = swiglu_limit(&gate.expected, &up.expected)?;
    let (w2, s2) = load_shared_pair("w2")?;
    let down = run_fp8_matrix(
        ctx,
        pipelines,
        timestamp_bits,
        timestamp_period_ns,
        "FP8 L42 shared w2",
        &hidden_actual,
        &w2,
        &s2,
        S14MatvecShape::new(4096, 2048)?,
    )?;
    let expected = cpu_fp8_matvec(
        &hidden_expected,
        &w2,
        &s2,
        S14MatvecShape::new(4096, 2048)?.validate_fp8()?,
    );
    Ok((down.actual, expected))
}

fn load_shared_pair(component: &str) -> Result<(Vec<u8>, Vec<u8>)> {
    let weight = SHARED_RANGES
        .iter()
        .find(|range| range.component == component && range.kind == "weight")
        .ok_or_else(|| anyhow::anyhow!("missing shared {component} weight range"))?;
    let scale = SHARED_RANGES
        .iter()
        .find(|range| range.component == component && range.kind == "scale")
        .ok_or_else(|| anyhow::anyhow!("missing shared {component} scale range"))?;
    let weight_path = find_cached_range(*weight)?;
    let scale_path = find_cached_range(*scale)?;
    Ok((
        read_exact(&weight_path, weight.bytes as usize)?,
        read_exact(&scale_path, scale.bytes as usize)?,
    ))
}

fn find_cached_range(range: SharedRange) -> Result<PathBuf> {
    for item in std::fs::read_dir(RANGE_CACHE).context("read Polaris range cache")? {
        let path = item?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(encoded) = std::fs::read(&path) else {
            continue;
        };
        let Ok(sidecar) = serde_json::from_slice::<CacheSidecar>(&encoded) else {
            continue;
        };
        if sidecar.identity.source_file == "model-00044-of-00048.safetensors"
            && sidecar.identity.start == range.start
            && sidecar.identity.end == range.end
            && sidecar.bytes == range.bytes
        {
            let payload = path.with_extension("bin");
            if payload.is_file() {
                return Ok(payload);
            }
        }
    }
    bail!(
        "local cache is missing shared {} {} range {}-{}; downloads are forbidden",
        range.component,
        range.kind,
        range.start,
        range.end
    )
}

fn swiglu_limit(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        bail!("SwiGLU gate/up shape mismatch");
    }
    Ok(gate
        .iter()
        .zip(up)
        .map(|(&gate, &up)| {
            let gate = gate.min(10.0);
            let up = up.clamp(-10.0, 10.0);
            (gate / (1.0 + (-gate).exp())) * up
        })
        .collect())
}

fn basic_stats(values: &[f32]) -> Result<BasicStats> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        bail!("stats require non-empty finite values");
    }
    let square_sum: f64 = values
        .iter()
        .map(|&value| (value as f64) * (value as f64))
        .sum();
    let sum: f64 = values.iter().map(|&value| value as f64).sum();
    let maxabs = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    Ok(BasicStats {
        l2: square_sum.sqrt(),
        mean: sum / values.len() as f64,
        maxabs,
    })
}

fn compare_frozen(label: &str, actual: &[f32], frozen: &FrozenStats) -> Result<()> {
    let stats = basic_stats(actual)?;
    let l2_delta = (stats.l2 - frozen.l2).abs();
    let mean_delta = (stats.mean - frozen.mean).abs();
    let maxabs_delta = (stats.maxabs - frozen.maxabs).abs();
    println!(
        "{label} frozen stats: l2={:.9} (delta={l2_delta:.3e}), mean={:.9e} \
         (delta={mean_delta:.3e}), maxabs={:.9} (delta={maxabs_delta:.3e})",
        stats.l2, stats.mean, stats.maxabs
    );
    if l2_delta > 2.0e-3 || mean_delta > 2.0e-5 || maxabs_delta > 5.0e-4 {
        bail!("{label} drifted from frozen real-page reference");
    }
    Ok(())
}

fn benchmark(
    ctx: &VulkanContext,
    buffers: &DeviceBuffers,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    mut record_dispatch: impl FnMut(vk::CommandBuffer),
) -> Result<Timing> {
    unsafe {
        let pool = make_command_pool(ctx)?;
        let cb = allocate_command_buffer(ctx, pool)?;
        let queries = ctx.device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2),
            None,
        )?;
        ctx.device.begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        ctx.device.cmd_reset_query_pool(cb, queries, 0, 2);
        ctx.device
            .cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, queries, 0);
        let serial_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE);
        for iteration in 0..ITERATIONS {
            record_dispatch(cb);
            if iteration + 1 < ITERATIONS {
                // Repeated evidence dispatches target the same output buffer.
                // Serialize WAW accesses so the benchmark remains Vulkan-valid.
                ctx.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[serial_barrier],
                    &[],
                    &[],
                );
            }
        }
        ctx.device
            .cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, queries, 1);
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
        copy(ctx, cb, &buffers.y, &buffers.readback, buffers.y_bytes);
        ctx.device.end_command_buffer(cb)?;

        let cpu_start = Instant::now();
        submit_and_wait(ctx, cb)?;
        let submit_readback_sync_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;
        let mut ticks = [0u64; 2];
        ctx.device.get_query_pool_results(
            queries,
            0,
            &mut ticks,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )?;
        let mask = if timestamp_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << timestamp_bits) - 1
        };
        let elapsed_ticks = ticks[1].wrapping_sub(ticks[0]) & mask;
        let gpu_kernel_ms_mean =
            elapsed_ticks as f64 * timestamp_period_ns / 1_000_000.0 / ITERATIONS as f64;
        ctx.device.destroy_query_pool(queries, None);
        ctx.device.destroy_command_pool(pool, None);
        Ok(Timing {
            iterations: ITERATIONS,
            gpu_kernel_ms_mean,
            submit_readback_sync_ms,
        })
    }
}

fn print_evidence(
    label: &str,
    shape: S14MatvecShape,
    cpu_ms: f64,
    timing: &Timing,
    error: &ErrorStats,
) {
    println!(
        "{label} [N={},K={}]: cpu_ref_ms={cpu_ms:.6}, iterations={}, \
         gpu_kernel_plus_serial_barrier_ms_mean={:.7}, submit_readback_sync_ms={:.6}, \
         max_abs={:.9e}, mean_abs={:.9e}, rmse={:.9e}, max_rel_abs_ref_gt_1e-5={:.9e}",
        shape.n,
        shape.k,
        timing.iterations,
        timing.gpu_kernel_ms_mean,
        timing.submit_readback_sync_ms,
        error.max_abs,
        error.mean_abs,
        error.rmse,
        error.max_rel_for_abs_ref_gt_1e_5,
    );
}

fn cpu_mxfp4_matvec(
    x: &[f32],
    packed_weight: &[u8],
    scale: &[u8],
    shape: S14MatvecShape,
) -> Vec<f32> {
    let k = shape.k as usize;
    let packed_k = k / 2;
    let groups = k / 32;
    (0..shape.n as usize)
        .into_par_iter()
        .map(|n| {
            let mut sum = 0.0f32;
            for group in 0..groups {
                let s = ue8m0(scale[n * groups + group]);
                let pair_base = group * 16;
                for pair_in_group in 0..16 {
                    let pair = pair_base + pair_in_group;
                    let codes = packed_weight[n * packed_k + pair];
                    let k0 = pair * 2;
                    sum += x[k0] * e2m1(codes & 0x0f) * s;
                    sum += x[k0 + 1] * e2m1(codes >> 4) * s;
                }
            }
            sum
        })
        .collect()
}

fn cpu_fp8_matvec(x: &[f32], weight: &[u8], scale: &[u8], shape: S14MatvecShape) -> Vec<f32> {
    let k = shape.k as usize;
    let k_groups = k.div_ceil(128);
    (0..shape.n as usize)
        .into_par_iter()
        .map(|n| {
            let scale_row = (n / 128) * k_groups;
            let weight_row = n * k;
            let mut sum = 0.0f32;
            for k_index in 0..k {
                let s = ue8m0(scale[scale_row + k_index / 128]);
                sum += x[k_index] * e4m3fn(weight[weight_row + k_index]) * s;
            }
            sum
        })
        .collect()
}

fn e2m1(code: u8) -> f32 {
    const MAGNITUDE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let value = MAGNITUDE[(code & 7) as usize];
    if code & 8 == 0 {
        value
    } else {
        -value
    }
}

fn ue8m0(code: u8) -> f32 {
    2.0f32.powi(code as i32 - 127)
}

fn e4m3fn(code: u8) -> f32 {
    let exponent = (code >> 3) & 0x0f;
    let mantissa = code & 7;
    let magnitude = if exponent == 0 {
        mantissa as f32 * 2.0f32.powi(-9)
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
    };
    if code & 0x80 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn error_stats(actual: &[f32], expected: &[f32]) -> Result<ErrorStats> {
    if actual.len() != expected.len() || actual.is_empty() {
        bail!("error metric input shape mismatch");
    }
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut sum_square = 0.0f64;
    let mut max_rel = 0.0f32;
    for (&a, &e) in actual.iter().zip(expected) {
        if !a.is_finite() || !e.is_finite() {
            bail!("non-finite numerical result: actual={a}, expected={e}");
        }
        let delta = (a - e).abs();
        max_abs = max_abs.max(delta);
        sum_abs += delta as f64;
        sum_square += (delta as f64) * (delta as f64);
        if e.abs() > 1.0e-5 {
            max_rel = max_rel.max(delta / e.abs());
        }
    }
    Ok(ErrorStats {
        max_abs,
        mean_abs: sum_abs / actual.len() as f64,
        rmse: (sum_square / actual.len() as f64).sqrt(),
        max_rel_for_abs_ref_gt_1e_5: max_rel,
    })
}

fn enforce_error(error: &ErrorStats, max_abs: f32, rmse: f64) -> Result<()> {
    if error.max_abs > max_abs || error.rmse > rmse {
        bail!(
            "Vulkan parity exceeded tolerance: max_abs={} (limit {}), rmse={} (limit {})",
            error.max_abs,
            max_abs,
            error.rmse,
            rmse
        );
    }
    Ok(())
}

fn read_exact(path: &Path, expected_bytes: usize) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() != expected_bytes {
        bail!(
            "{} is {} B, expected {} B",
            path.display(),
            bytes.len(),
            expected_bytes
        );
    }
    Ok(bytes)
}

unsafe fn make_command_pool(ctx: &VulkanContext) -> Result<vk::CommandPool> {
    Ok(ctx.device.create_command_pool(
        &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.qf_graphics),
        None,
    )?)
}

unsafe fn allocate_command_buffer(
    ctx: &VulkanContext,
    pool: vk::CommandPool,
) -> Result<vk::CommandBuffer> {
    Ok(ctx.device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1),
    )?[0])
}

unsafe fn copy(
    ctx: &VulkanContext,
    cb: vk::CommandBuffer,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    bytes: u64,
) {
    let region = vk::BufferCopy::default().size(bytes);
    ctx.device
        .cmd_copy_buffer(cb, src.handle(), dst.handle(), &[region]);
}

unsafe fn submit_and_wait(ctx: &VulkanContext, cb: vk::CommandBuffer) -> Result<()> {
    let fence = ctx
        .device
        .create_fence(&vk::FenceCreateInfo::default(), None)?;
    let command_buffers = [cb];
    ctx.device.queue_submit(
        ctx.q_graphics,
        &[vk::SubmitInfo::default().command_buffers(&command_buffers)],
        fence,
    )?;
    ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
    ctx.device.destroy_fence(fence, None);
    Ok(())
}
