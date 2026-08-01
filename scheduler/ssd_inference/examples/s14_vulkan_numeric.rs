//! Real L42 numerical and timing evidence for the Polaris S14 Vulkan matvecs.
//!
//! No weights are downloaded. The program requires inputs captured by the
//! hash-verifying L42 CPU reference and rejects synthetic fallback data.
//!
//! Run:
//!   python -X utf8 fast16/research/polaris_meridian_v1/l42_real_reference/l42_reference.py \
//!     --capture-dir <fresh-dir>
//!   $env:POLARIS_S14_L42_CAPTURE_DIR = "<fresh-dir>"
//!   cargo run --release --offline -p ssd_inference --example s14_vulkan_numeric

use anyhow::{bail, Context, Result};
use ash::vk;
use rayon::prelude::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use ssd_inference::buffer::GpuBuffer;
use ssd_inference::device::VulkanContext;
use ssd_inference::s14_vulkan::{
    validate_e4m3fn_codes, validate_ue8m0_codes, S14Fp8Dispatch, S14MatvecShape,
    S14MoeAccumulateDispatch, S14Mxfp4Dispatch, S14NumericPipelines, S14RouteMixDispatch,
    S14SwigluLimitDispatch,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

const MODEL_DIR: &str = "D:/models/Polaris-S14";
const ROUTE_MANIFEST: &str = "D:/models/Polaris-S14/l42_real_layer_route_manifest.json";
const S14_MANIFEST: &str = "D:/models/Polaris-S14/s14_base_cache_manifest.json";
const CAPTURE_DIR_ENV: &str = "POLARIS_S14_L42_CAPTURE_DIR";
const REVISION: &str = "7872f01b1d1fe23eabc4c98b48bffcef5a386062";
const REAL_EXPERT_ID: u32 = 126;
const AMD_VENDOR_ID: u32 = 0x1002;
const NAVI10_DEVICE_ID: u32 = 0x731f;
const ITERATIONS: u32 = 100;
const MOE_BATCH_ITERATIONS: u32 = 1;

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

struct UploadedBuffer {
    staging: GpuBuffer,
    device: GpuBuffer,
    bytes: u64,
}

impl UploadedBuffer {
    fn new(ctx: &VulkanContext, bytes: &[u8]) -> Result<Self> {
        let byte_count = bytes.len() as u64;
        let staging = GpuBuffer::new_staging(ctx, byte_count)?;
        unsafe { staging.write_at(0, bytes) };
        let device = GpuBuffer::new_vram(
            ctx,
            byte_count,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        Ok(Self {
            staging,
            device,
            bytes: byte_count,
        })
    }

    unsafe fn cmd_upload(&self, ctx: &VulkanContext, cb: vk::CommandBuffer) {
        copy(ctx, cb, &self.staging, &self.device, self.bytes);
    }

    fn destroy(&self, ctx: &VulkanContext) {
        self.device.destroy(ctx);
        self.staging.destroy(ctx);
    }
}

#[allow(dead_code)]
struct ExpertChainBuffers {
    x: UploadedBuffer,
    w1: UploadedBuffer,
    s1: UploadedBuffer,
    w3: UploadedBuffer,
    s3: UploadedBuffer,
    w2: UploadedBuffer,
    s2: UploadedBuffer,
    gate: GpuBuffer,
    up: GpuBuffer,
    hidden: GpuBuffer,
    down: GpuBuffer,
    routed: GpuBuffer,
    readback: GpuBuffer,
}

#[allow(dead_code)]
impl ExpertChainBuffers {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &VulkanContext,
        x: &[f32],
        w1: &[u8],
        s1: &[u8],
        w3: &[u8],
        s3: &[u8],
        w2: &[u8],
        s2: &[u8],
    ) -> Result<Self> {
        let x = UploadedBuffer::new(ctx, bytemuck::cast_slice(x))?;
        let w1 = UploadedBuffer::new(ctx, w1)?;
        let s1 = UploadedBuffer::new(ctx, s1)?;
        let w3 = UploadedBuffer::new(ctx, w3)?;
        let s3 = UploadedBuffer::new(ctx, s3)?;
        let w2 = UploadedBuffer::new(ctx, w2)?;
        let s2 = UploadedBuffer::new(ctx, s2)?;
        let intermediate_bytes = 2048 * std::mem::size_of::<f32>() as u64;
        let output_bytes = 4096 * std::mem::size_of::<f32>() as u64;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let gate = GpuBuffer::new_vram(ctx, intermediate_bytes, storage)?;
        let up = GpuBuffer::new_vram(ctx, intermediate_bytes, storage)?;
        let hidden = GpuBuffer::new_vram(ctx, intermediate_bytes, storage)?;
        let down = GpuBuffer::new_vram(ctx, output_bytes, storage)?;
        let routed = GpuBuffer::new_vram(
            ctx,
            output_bytes,
            storage | vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        let readback = GpuBuffer::new(
            ctx,
            output_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?;
        Ok(Self {
            x,
            w1,
            s1,
            w3,
            s3,
            w2,
            s2,
            gate,
            up,
            hidden,
            down,
            routed,
            readback,
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
            for buffer in [
                &self.x, &self.w1, &self.s1, &self.w3, &self.s3, &self.w2, &self.s2,
            ] {
                buffer.cmd_upload(ctx, cb);
            }
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

    fn output(&self) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(self.readback.mapped() as *const f32, 4096).to_vec() }
    }

    fn destroy(&self, ctx: &VulkanContext) {
        self.readback.destroy(ctx);
        self.routed.destroy(ctx);
        self.down.destroy(ctx);
        self.hidden.destroy(ctx);
        self.up.destroy(ctx);
        self.gate.destroy(ctx);
        self.s2.destroy(ctx);
        self.w2.destroy(ctx);
        self.s3.destroy(ctx);
        self.w3.destroy(ctx);
        self.s1.destroy(ctx);
        self.w1.destroy(ctx);
        self.x.destroy(ctx);
    }
}

struct MoePayload {
    expert_id: Option<u32>,
    mix_weight: f32,
    w1: Vec<u8>,
    s1: Vec<u8>,
    w3: Vec<u8>,
    s3: Vec<u8>,
    w2: Vec<u8>,
    s2: Vec<u8>,
}

struct GpuMoeWeights {
    w1: UploadedBuffer,
    s1: UploadedBuffer,
    w3: UploadedBuffer,
    s3: UploadedBuffer,
    w2: UploadedBuffer,
    s2: UploadedBuffer,
}

impl GpuMoeWeights {
    fn new(ctx: &VulkanContext, payload: &MoePayload) -> Result<Self> {
        Ok(Self {
            w1: UploadedBuffer::new(ctx, &payload.w1)?,
            s1: UploadedBuffer::new(ctx, &payload.s1)?,
            w3: UploadedBuffer::new(ctx, &payload.w3)?,
            s3: UploadedBuffer::new(ctx, &payload.s3)?,
            w2: UploadedBuffer::new(ctx, &payload.w2)?,
            s2: UploadedBuffer::new(ctx, &payload.s2)?,
        })
    }

    unsafe fn cmd_upload(&self, ctx: &VulkanContext, cb: vk::CommandBuffer) {
        for buffer in [&self.w1, &self.s1, &self.w3, &self.s3, &self.w2, &self.s2] {
            buffer.cmd_upload(ctx, cb);
        }
    }

    fn destroy(&self, ctx: &VulkanContext) {
        self.s2.destroy(ctx);
        self.w2.destroy(ctx);
        self.s3.destroy(ctx);
        self.w3.destroy(ctx);
        self.s1.destroy(ctx);
        self.w1.destroy(ctx);
    }
}

struct MoeBatchBuffers {
    x: UploadedBuffer,
    routed: Vec<GpuMoeWeights>,
    shared: GpuMoeWeights,
    gate: GpuBuffer,
    up: GpuBuffer,
    hidden: GpuBuffer,
    down: GpuBuffer,
    accumulator: GpuBuffer,
    readback: GpuBuffer,
}

impl MoeBatchBuffers {
    fn new(
        ctx: &VulkanContext,
        x: &[f32],
        routed: &[MoePayload],
        shared: &MoePayload,
    ) -> Result<Self> {
        let x = UploadedBuffer::new(ctx, bytemuck::cast_slice(x))?;
        let routed = routed
            .iter()
            .map(|payload| GpuMoeWeights::new(ctx, payload))
            .collect::<Result<Vec<_>>>()?;
        let shared = GpuMoeWeights::new(ctx, shared)?;
        let intermediate_bytes = 2048 * std::mem::size_of::<f32>() as u64;
        let output_bytes = 4096 * std::mem::size_of::<f32>() as u64;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let gate = GpuBuffer::new_vram(ctx, intermediate_bytes, storage)?;
        let up = GpuBuffer::new_vram(ctx, intermediate_bytes, storage)?;
        let hidden = GpuBuffer::new_vram(ctx, intermediate_bytes, storage)?;
        let down = GpuBuffer::new_vram(ctx, output_bytes, storage)?;
        let accumulator = GpuBuffer::new_vram(
            ctx,
            output_bytes,
            storage | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        let readback = GpuBuffer::new(
            ctx,
            output_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?;
        Ok(Self {
            x,
            routed,
            shared,
            gate,
            up,
            hidden,
            down,
            accumulator,
            readback,
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
            self.x.cmd_upload(ctx, cb);
            for weights in &self.routed {
                weights.cmd_upload(ctx, cb);
            }
            self.shared.cmd_upload(ctx, cb);
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

    fn output(&self) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(self.readback.mapped() as *const f32, 4096).to_vec() }
    }

    fn destroy(&self, ctx: &VulkanContext) {
        self.readback.destroy(ctx);
        self.accumulator.destroy(ctx);
        self.down.destroy(ctx);
        self.hidden.destroy(ctx);
        self.up.destroy(ctx);
        self.gate.destroy(ctx);
        self.shared.destroy(ctx);
        for weights in &self.routed {
            weights.destroy(ctx);
        }
        self.x.destroy(ctx);
    }
}

struct RoutedMoeDispatch {
    w1: S14Mxfp4Dispatch,
    w3: S14Mxfp4Dispatch,
    w2: S14Mxfp4Dispatch,
    accumulate: S14MoeAccumulateDispatch,
}

struct SharedMoeDispatch {
    w1: S14Fp8Dispatch,
    w3: S14Fp8Dispatch,
    w2: S14Fp8Dispatch,
    accumulate: S14MoeAccumulateDispatch,
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

#[derive(Deserialize)]
struct RouteManifest {
    format: String,
    revision: String,
    layer: u32,
    route_source: String,
    expert_ids: Vec<u32>,
    route_weights: Vec<f32>,
    entry_count: usize,
    bytes: u64,
    entries: Vec<RouteEntry>,
}

#[derive(Deserialize)]
struct RouteEntry {
    tensor: String,
    expert_id: u32,
    bytes: u64,
    path: PathBuf,
    sha256: String,
}

#[derive(Deserialize)]
struct CaptureManifest {
    format: String,
    revision: String,
    layer: u32,
    expert_id: u32,
    source_f32_le_sha256: CaptureSourceHashes,
    asset_integrity: CaptureAssetIntegrity,
    inputs: Vec<CaptureInput>,
}

#[derive(Deserialize)]
struct CaptureSourceHashes {
    ffn_input: String,
}

#[derive(Deserialize)]
struct CaptureAssetIntegrity {
    hashes_checked: usize,
    payload_bytes: u64,
    payload_files: usize,
    manifest_sha256: CaptureManifestHashes,
}

#[derive(Deserialize)]
struct CaptureManifestHashes {
    base: String,
    route: String,
    s14: String,
}

#[derive(Deserialize)]
struct CaptureInput {
    name: String,
    file: PathBuf,
    shape: Vec<usize>,
    bytes: usize,
    f32_le_sha256: String,
}

#[derive(Deserialize)]
struct BaseManifest {
    format: String,
    revision: String,
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
    sha256: String,
}

#[derive(Deserialize)]
struct S14Manifest {
    format: String,
    revision: String,
    entry_count: usize,
    bytes: u64,
    entries: Vec<BaseEntry>,
}

fn main() -> Result<()> {
    println!("Polaris S14 Vulkan numerical kernels (hash-verified real L42 payloads)");
    let capture_dir = std::env::var_os(CAPTURE_DIR_ENV)
        .map(PathBuf::from)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{CAPTURE_DIR_ENV} is required; synthetic activation fallback is forbidden"
            )
        })?;
    let capture = load_capture_manifest(&capture_dir)?;
    let expert_w1_w3_input =
        load_capture_input(&capture, &capture_dir, "expert_126_w1_w3", &[1, 1, 4096])?;
    let _expert_w2_input =
        load_capture_input(&capture, &capture_dir, "expert_126_w2", &[1, 1, 2048])?;
    let wq_a_input = load_capture_input(&capture, &capture_dir, "wq_a", &[1, 1, 4096])?;

    let ctx = VulkanContext::init()?;
    println!("GPU: {}", ctx.gpu_name);
    let queue_props = unsafe {
        ctx.instance
            .get_physical_device_queue_family_properties(ctx.physical)
    };
    let timestamp_bits = queue_props[ctx.qf_graphics as usize].timestamp_valid_bits;
    if timestamp_bits == 0 {
        bail!("graphics/compute queue does not expose timestamp queries");
    }
    let physical_properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    if physical_properties.vendor_id != AMD_VENDOR_ID
        || physical_properties.device_id != NAVI10_DEVICE_ID
    {
        bail!(
            "evidence run requires RX 5700 XT PCI id 0x{AMD_VENDOR_ID:04x}:0x{NAVI10_DEVICE_ID:04x}; found 0x{:04x}:0x{:04x} ({})",
            physical_properties.vendor_id,
            physical_properties.device_id,
            ctx.gpu_name
        );
    }
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
    run_real_moe_batch(
        &ctx,
        &pipelines,
        timestamp_bits,
        timestamp_period_ns,
        &expert_w1_w3_input,
    )?;
    run_real_fp8_wq_a(
        &ctx,
        &pipelines,
        timestamp_bits,
        timestamp_period_ns,
        &wq_a_input,
    )?;
    pipelines.destroy(&ctx);
    println!(
        "claim: real L42 packed-matvec and minimal GPU-resident top-6 routed + shared MoE batch parity; no full official expert/layer, token/s, or full S14 forward claim"
    );
    Ok(())
}

fn run_real_moe_batch(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    w1_w3_input: &[f32],
) -> Result<()> {
    let manifest: RouteManifest =
        serde_json::from_slice(&std::fs::read(ROUTE_MANIFEST).context("read L42 route manifest")?)
            .context("parse L42 route manifest")?;
    validate_route_contract(&manifest)?;
    println!(
        "real L42 route top6={:?}, weight_sum={:.9}, cached_ranges={}",
        manifest.expert_ids,
        manifest.route_weights.iter().sum::<f32>(),
        manifest.entries.len()
    );

    let routed = manifest
        .expert_ids
        .iter()
        .zip(&manifest.route_weights)
        .map(|(&expert_id, &mix_weight)| {
            let (w1, s1) = load_route_pair(&manifest, expert_id, "w1")?;
            let (w3, s3) = load_route_pair(&manifest, expert_id, "w3")?;
            let (w2, s2) = load_route_pair(&manifest, expert_id, "w2")?;
            Ok(MoePayload {
                expert_id: Some(expert_id),
                mix_weight,
                w1,
                s1,
                w3,
                s3,
                w2,
                s2,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let s14_manifest: S14Manifest = serde_json::from_slice(
        &std::fs::read(S14_MANIFEST).context("read S14 base cache manifest")?,
    )
    .context("parse S14 base cache manifest")?;
    if s14_manifest.format != "polaris-s14-base-cache-snapshot-v1"
        || s14_manifest.revision != REVISION
        || s14_manifest.entry_count != 503
        || s14_manifest.entries.len() != 503
        || s14_manifest.bytes != 2_194_713_552
    {
        bail!("S14 base cache manifest contract drift");
    }
    let (w1, s1) = load_shared_pair(&s14_manifest, "w1")?;
    let (w3, s3) = load_shared_pair(&s14_manifest, "w3")?;
    let (w2, s2) = load_shared_pair(&s14_manifest, "w2")?;
    let shared = MoePayload {
        expert_id: None,
        mix_weight: 1.0,
        w1,
        s1,
        w3,
        s3,
        w2,
        s2,
    };

    run_top6_shared_moe_batch(
        ctx,
        pipelines,
        timestamp_bits,
        timestamp_period_ns,
        w1_w3_input,
        &routed,
        &shared,
    )
    .context("real L42 top-6 routed + shared GPU-resident MoE batch")?;
    Ok(())
}

fn run_real_fp8_wq_a(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    input: &[f32],
) -> Result<()> {
    let path = Path::new(MODEL_DIR).join("l42_base_cache_manifest.json");
    let manifest: BaseManifest = serde_json::from_slice(&std::fs::read(&path)?)?;
    if manifest.format != "polaris-l42-base-cache-snapshot-v1"
        || manifest.revision != REVISION
        || manifest.layer != 42
        || manifest.entry_count != 34
        || manifest.entries.len() != 34
        || manifest.bytes != 142_131_800
    {
        bail!("L42 base cache manifest contract drift");
    }
    let weight_entry = base_entry(&manifest, "layers.42.attn.wq_a.weight")?;
    let scale_entry = base_entry(&manifest, "layers.42.attn.wq_a.scale")?;
    let weight = read_verified_payload(
        &weight_entry.path,
        weight_entry.bytes as usize,
        &weight_entry.sha256,
        &weight_entry.tensor,
    )?;
    let scale = read_verified_payload(
        &scale_entry.path,
        scale_entry.bytes as usize,
        &scale_entry.sha256,
        &scale_entry.tensor,
    )?;
    let shape = S14MatvecShape::new(1024, 4096)?.validate_fp8()?;
    run_fp8_matrix(
        ctx,
        pipelines,
        timestamp_bits,
        timestamp_period_ns,
        "FP8 real L42 base-cache wq_a",
        input,
        &weight,
        &scale,
        shape,
    )?;
    Ok(())
}

fn validate_route_contract(manifest: &RouteManifest) -> Result<()> {
    let expected_ids = [126, 12, 205, 149, 227, 174];
    let expected_weights = [
        0.2747795581817627,
        0.2491425722837448,
        0.24045628309249878,
        0.24615329504013062,
        0.25386279821395874,
        0.23560553789138794,
    ];
    if manifest.format != "polaris-l42-real-layer-route-cache-v1"
        || manifest.revision != REVISION
        || manifest.layer != 42
        || manifest.route_source != "l42_real_attention_route.json"
        || manifest.expert_ids != expected_ids
        || manifest.route_weights != expected_weights
        || manifest.entry_count != 36
        || manifest.entries.len() != 36
        || manifest.bytes != 80_216_064
    {
        bail!("real L42 route manifest contract drift");
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
        read_verified_payload(
            &weight.path,
            weight.bytes as usize,
            &weight.sha256,
            &weight.tensor,
        )?,
        read_verified_payload(
            &scale.path,
            scale.bytes as usize,
            &scale.sha256,
            &scale.tensor,
        )?,
    ))
}

fn load_shared_pair(manifest: &S14Manifest, component: &str) -> Result<(Vec<u8>, Vec<u8>)> {
    let prefix = format!("layers.42.ffn.shared_experts.{component}");
    let weight = s14_entry(manifest, &format!("{prefix}.weight"))?;
    let scale = s14_entry(manifest, &format!("{prefix}.scale"))?;
    if weight.bytes != 8_388_608 || scale.bytes != 512 {
        bail!("shared/{component} byte contract drift");
    }
    Ok((
        read_verified_payload(
            &weight.path,
            weight.bytes as usize,
            &weight.sha256,
            &weight.tensor,
        )?,
        read_verified_payload(
            &scale.path,
            scale.bytes as usize,
            &scale.sha256,
            &scale.tensor,
        )?,
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

fn s14_entry<'a>(manifest: &'a S14Manifest, tensor: &str) -> Result<&'a BaseEntry> {
    manifest
        .entries
        .iter()
        .find(|entry| entry.tensor == tensor)
        .ok_or_else(|| anyhow::anyhow!("S14 manifest missing {tensor}"))
}

fn run_top6_shared_moe_batch(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    x: &[f32],
    routed: &[MoePayload],
    shared: &MoePayload,
) -> Result<()> {
    if x.len() != 4096 || routed.len() != 6 || shared.expert_id.is_some() {
        bail!("real L42 MoE batch shape/route contract drift");
    }
    let up_shape = S14MatvecShape::new(2048, 4096)?.validate_mxfp4()?;
    let down_shape = S14MatvecShape::new(4096, 2048)?.validate_mxfp4()?;
    let shared_up_shape = S14MatvecShape::new(2048, 4096)?.validate_fp8()?;
    let shared_down_shape = S14MatvecShape::new(4096, 2048)?.validate_fp8()?;
    for payload in routed {
        if payload.expert_id.is_none()
            || !payload.mix_weight.is_finite()
            || payload.mix_weight < 0.0
        {
            bail!("routed MoE payload identity/weight contract drift");
        }
        for scale in [&payload.s1, &payload.s3, &payload.s2] {
            validate_ue8m0_codes(scale)?;
        }
    }
    for weight in [&shared.w1, &shared.w3, &shared.w2] {
        validate_e4m3fn_codes(weight)?;
    }
    for scale in [&shared.s1, &shared.s3, &shared.s2] {
        validate_ue8m0_codes(scale)?;
    }

    let cpu_start = Instant::now();
    let mut expected = vec![0.0f32; 4096];
    for payload in routed {
        let gate = cpu_mxfp4_matvec(x, &payload.w1, &payload.s1, up_shape);
        let up = cpu_mxfp4_matvec(x, &payload.w3, &payload.s3, up_shape);
        let hidden = swiglu_limit(&gate, &up)?;
        let down = cpu_mxfp4_matvec(&hidden, &payload.w2, &payload.s2, down_shape);
        for (accumulator, value) in expected.iter_mut().zip(down) {
            *accumulator += payload.mix_weight * value;
        }
    }
    let shared_gate = cpu_fp8_matvec(x, &shared.w1, &shared.s1, shared_up_shape);
    let shared_up = cpu_fp8_matvec(x, &shared.w3, &shared.s3, shared_up_shape);
    let shared_hidden = swiglu_limit(&shared_gate, &shared_up)?;
    let shared_down = cpu_fp8_matvec(&shared_hidden, &shared.w2, &shared.s2, shared_down_shape);
    for (accumulator, value) in expected.iter_mut().zip(shared_down) {
        *accumulator += value;
    }
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

    let buffers = MoeBatchBuffers::new(ctx, x, routed, shared)?;
    buffers.upload(ctx)?;
    let swiglu =
        pipelines.bind_swiglu_limit(ctx, 2048, &buffers.gate, &buffers.up, &buffers.hidden)?;
    let routed_dispatches = buffers
        .routed
        .iter()
        .zip(routed)
        .map(|(weights, payload)| {
            Ok(RoutedMoeDispatch {
                w1: pipelines.bind_mxfp4(
                    ctx,
                    up_shape,
                    &buffers.x.device,
                    &weights.w1.device,
                    &weights.s1.device,
                    &buffers.gate,
                )?,
                w3: pipelines.bind_mxfp4(
                    ctx,
                    up_shape,
                    &buffers.x.device,
                    &weights.w3.device,
                    &weights.s3.device,
                    &buffers.up,
                )?,
                w2: pipelines.bind_mxfp4(
                    ctx,
                    down_shape,
                    &buffers.hidden,
                    &weights.w2.device,
                    &weights.s2.device,
                    &buffers.down,
                )?,
                accumulate: pipelines.bind_moe_accumulate(
                    ctx,
                    4096,
                    payload.mix_weight,
                    &buffers.down,
                    &buffers.accumulator,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let shared_dispatch = SharedMoeDispatch {
        w1: pipelines.bind_fp8(
            ctx,
            shared_up_shape,
            &buffers.x.device,
            &buffers.shared.w1.device,
            &buffers.shared.s1.device,
            &buffers.gate,
        )?,
        w3: pipelines.bind_fp8(
            ctx,
            shared_up_shape,
            &buffers.x.device,
            &buffers.shared.w3.device,
            &buffers.shared.s3.device,
            &buffers.up,
        )?,
        w2: pipelines.bind_fp8(
            ctx,
            shared_down_shape,
            &buffers.hidden,
            &buffers.shared.w2.device,
            &buffers.shared.s2.device,
            &buffers.down,
        )?,
        accumulate: pipelines.bind_moe_accumulate(
            ctx,
            4096,
            1.0,
            &buffers.down,
            &buffers.accumulator,
        )?,
    };

    let timing = benchmark_top6_shared_moe_batch(
        ctx,
        pipelines,
        &buffers,
        timestamp_bits,
        timestamp_period_ns,
        &swiglu,
        &routed_dispatches,
        &shared_dispatch,
    )?;
    let actual = buffers.output();
    let error = error_stats(&actual, &expected)?;
    enforce_error(&error, 1.0e-3, 1.5e-4)?;
    println!(
        "GPU-resident real L42 top6+shared minimal MoE batch: ids={:?}, weights={:?}, cpu_ref_ms={cpu_ms:.6}, iterations={}, gpu_fill_plus_35_dispatch_barriers_ms_mean={:.7}, submit_readback_sync_ms={:.6}, max_abs={:.9e}, mean_abs={:.9e}, rmse={:.9e}, max_rel_abs_ref_gt_1e-5={:.9e}",
        routed
            .iter()
            .map(|payload| payload.expert_id.unwrap())
            .collect::<Vec<_>>(),
        routed
            .iter()
            .map(|payload| payload.mix_weight)
            .collect::<Vec<_>>(),
        timing.iterations,
        timing.gpu_kernel_ms_mean,
        timing.submit_readback_sync_ms,
        error.max_abs,
        error.mean_abs,
        error.rmse,
        error.max_rel_for_abs_ref_gt_1e_5,
    );

    shared_dispatch.accumulate.binder.destroy(ctx);
    shared_dispatch.w2.binder.destroy(ctx);
    shared_dispatch.w3.binder.destroy(ctx);
    shared_dispatch.w1.binder.destroy(ctx);
    for dispatch in routed_dispatches.into_iter().rev() {
        dispatch.accumulate.binder.destroy(ctx);
        dispatch.w2.binder.destroy(ctx);
        dispatch.w3.binder.destroy(ctx);
        dispatch.w1.binder.destroy(ctx);
    }
    swiglu.binder.destroy(ctx);
    buffers.destroy(ctx);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn benchmark_top6_shared_moe_batch(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    buffers: &MoeBatchBuffers,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    swiglu: &S14SwigluLimitDispatch,
    routed: &[RoutedMoeDispatch],
    shared: &SharedMoeDispatch,
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
        let shader_raw = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        let shader_serial = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        for iteration in 0..MOE_BATCH_ITERATIONS {
            if iteration > 0 {
                let compute_to_fill = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                ctx.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[compute_to_fill],
                    &[],
                    &[],
                );
            }
            ctx.device.cmd_fill_buffer(
                cb,
                buffers.accumulator.handle(),
                0,
                4096 * std::mem::size_of::<f32>() as u64,
                0,
            );
            let fill_to_compute = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[fill_to_compute],
                &[],
                &[],
            );

            for dispatch in routed {
                pipelines.cmd_mxfp4_matvec(ctx, cb, &dispatch.w1);
                pipelines.cmd_mxfp4_matvec(ctx, cb, &dispatch.w3);
                ctx.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[shader_raw],
                    &[],
                    &[],
                );
                pipelines.cmd_swiglu_limit(ctx, cb, swiglu);
                ctx.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[shader_raw],
                    &[],
                    &[],
                );
                pipelines.cmd_mxfp4_matvec(ctx, cb, &dispatch.w2);
                ctx.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[shader_raw],
                    &[],
                    &[],
                );
                pipelines.cmd_moe_accumulate(ctx, cb, &dispatch.accumulate);
                ctx.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[shader_serial],
                    &[],
                    &[],
                );
            }

            pipelines.cmd_fp8_matvec(ctx, cb, &shared.w1);
            pipelines.cmd_fp8_matvec(ctx, cb, &shared.w3);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[shader_raw],
                &[],
                &[],
            );
            pipelines.cmd_swiglu_limit(ctx, cb, swiglu);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[shader_raw],
                &[],
                &[],
            );
            pipelines.cmd_fp8_matvec(ctx, cb, &shared.w2);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[shader_raw],
                &[],
                &[],
            );
            pipelines.cmd_moe_accumulate(ctx, cb, &shared.accumulate);
        }
        ctx.device
            .cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, queries, 1);
        let readback_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[readback_barrier],
            &[],
            &[],
        );
        copy(
            ctx,
            cb,
            &buffers.accumulator,
            &buffers.readback,
            4096 * std::mem::size_of::<f32>() as u64,
        );
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
            elapsed_ticks as f64 * timestamp_period_ns / 1_000_000.0 / MOE_BATCH_ITERATIONS as f64;
        ctx.device.destroy_query_pool(queries, None);
        ctx.device.destroy_command_pool(pool, None);
        Ok(Timing {
            iterations: MOE_BATCH_ITERATIONS,
            gpu_kernel_ms_mean,
            submit_readback_sync_ms,
        })
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
fn run_mxfp4_expert_chain(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    route_weight: f32,
    x: &[f32],
    w1: &[u8],
    s1: &[u8],
    w3: &[u8],
    s3: &[u8],
    w2: &[u8],
    s2: &[u8],
) -> Result<()> {
    let up_shape = S14MatvecShape::new(2048, 4096)?.validate_mxfp4()?;
    let down_shape = S14MatvecShape::new(4096, 2048)?.validate_mxfp4()?;
    if x.len() != up_shape.k as usize {
        bail!("real E126 chain input shape drift");
    }
    for scale in [s1, s3, s2] {
        validate_ue8m0_codes(scale)?;
    }

    let cpu_start = Instant::now();
    let gate = cpu_mxfp4_matvec(x, w1, s1, up_shape);
    let up = cpu_mxfp4_matvec(x, w3, s3, up_shape);
    let hidden = swiglu_limit(&gate, &up)?;
    let down = cpu_mxfp4_matvec(&hidden, w2, s2, down_shape);
    let expected: Vec<f32> = down.iter().map(|value| route_weight * value).collect();
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

    let buffers = ExpertChainBuffers::new(ctx, x, w1, s1, w3, s3, w2, s2)?;
    buffers.upload(ctx)?;
    let w1_dispatch = pipelines.bind_mxfp4(
        ctx,
        up_shape,
        &buffers.x.device,
        &buffers.w1.device,
        &buffers.s1.device,
        &buffers.gate,
    )?;
    let w3_dispatch = pipelines.bind_mxfp4(
        ctx,
        up_shape,
        &buffers.x.device,
        &buffers.w3.device,
        &buffers.s3.device,
        &buffers.up,
    )?;
    let swiglu_dispatch =
        pipelines.bind_swiglu_limit(ctx, 2048, &buffers.gate, &buffers.up, &buffers.hidden)?;
    let w2_dispatch = pipelines.bind_mxfp4(
        ctx,
        down_shape,
        &buffers.hidden,
        &buffers.w2.device,
        &buffers.s2.device,
        &buffers.down,
    )?;
    let route_dispatch =
        pipelines.bind_route_mix(ctx, 4096, route_weight, &buffers.down, &buffers.routed)?;
    let timing = benchmark_expert_chain(
        ctx,
        pipelines,
        &buffers,
        timestamp_bits,
        timestamp_period_ns,
        &w1_dispatch,
        &w3_dispatch,
        &swiglu_dispatch,
        &w2_dispatch,
        &route_dispatch,
    )?;
    let actual = buffers.output();
    let error = error_stats(&actual, &expected)?;
    enforce_error(&error, 2.0e-4, 2.5e-5)?;
    println!(
        "GPU-resident real L42/E126 chain [w1,w3,clamp-SwiGLU,w2,route_mix]: route_weight={route_weight:.9}, cpu_ref_ms={cpu_ms:.6}, iterations={}, gpu_chain_dispatch_plus_barriers_ms_mean={:.7}, submit_readback_sync_ms={:.6}, max_abs={:.9e}, mean_abs={:.9e}, rmse={:.9e}, max_rel_abs_ref_gt_1e-5={:.9e}",
        timing.iterations,
        timing.gpu_kernel_ms_mean,
        timing.submit_readback_sync_ms,
        error.max_abs,
        error.mean_abs,
        error.rmse,
        error.max_rel_for_abs_ref_gt_1e_5,
    );

    route_dispatch.binder.destroy(ctx);
    w2_dispatch.binder.destroy(ctx);
    swiglu_dispatch.binder.destroy(ctx);
    w3_dispatch.binder.destroy(ctx);
    w1_dispatch.binder.destroy(ctx);
    buffers.destroy(ctx);
    Ok(())
}

#[allow(dead_code, clippy::too_many_arguments)]
fn benchmark_expert_chain(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    buffers: &ExpertChainBuffers,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    w1: &S14Mxfp4Dispatch,
    w3: &S14Mxfp4Dispatch,
    swiglu: &S14SwigluLimitDispatch,
    w2: &S14Mxfp4Dispatch,
    route: &S14RouteMixDispatch,
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
        let read_after_write = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        let next_iteration = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        for iteration in 0..ITERATIONS {
            pipelines.cmd_mxfp4_matvec(ctx, cb, w1);
            pipelines.cmd_mxfp4_matvec(ctx, cb, w3);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[read_after_write],
                &[],
                &[],
            );
            pipelines.cmd_swiglu_limit(ctx, cb, swiglu);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[read_after_write],
                &[],
                &[],
            );
            pipelines.cmd_mxfp4_matvec(ctx, cb, w2);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[read_after_write],
                &[],
                &[],
            );
            pipelines.cmd_route_mix(ctx, cb, route);
            if iteration + 1 < ITERATIONS {
                ctx.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[next_iteration],
                    &[],
                    &[],
                );
            }
        }
        ctx.device
            .cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, queries, 1);
        let readback_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[readback_barrier],
            &[],
            &[],
        );
        copy(
            ctx,
            cb,
            &buffers.routed,
            &buffers.readback,
            4096 * std::mem::size_of::<f32>() as u64,
        );
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
) -> Result<()> {
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
    Ok(())
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

fn load_capture_manifest(capture_dir: &Path) -> Result<CaptureManifest> {
    let capture_root = capture_dir
        .canonicalize()
        .with_context(|| format!("resolve capture directory {}", capture_dir.display()))?;
    let path = capture_root.join("capture_manifest.json");
    let manifest: CaptureManifest = serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
    )
    .context("parse real L42 capture manifest")?;
    if manifest.format != "polaris-l42-real-vulkan-input-capture-v1"
        || manifest.revision != REVISION
        || manifest.layer != 42
        || manifest.expert_id != REAL_EXPERT_ID
        || manifest.source_f32_le_sha256.ffn_input
            != "7e2d3167e3782eca8d762c3cc92d53bb9d64a65c7b18d37d16797ff39f611ad4"
        || manifest.asset_integrity.hashes_checked != 76
        || manifest.asset_integrity.payload_files != 76
        || manifest.asset_integrity.payload_bytes == 0
        || manifest.inputs.len() != 3
    {
        bail!("real L42 capture contract drift; synthetic fallback is forbidden");
    }

    let expected_manifests = [
        (
            Path::new(MODEL_DIR).join("l42_base_cache_manifest.json"),
            &manifest.asset_integrity.manifest_sha256.base,
            "base",
        ),
        (
            PathBuf::from(ROUTE_MANIFEST),
            &manifest.asset_integrity.manifest_sha256.route,
            "route",
        ),
        (
            Path::new(MODEL_DIR).join("s14_base_cache_manifest.json"),
            &manifest.asset_integrity.manifest_sha256.s14,
            "s14",
        ),
    ];
    for (manifest_path, expected_sha, label) in expected_manifests {
        let actual_sha = sha256_file(&manifest_path)?;
        if actual_sha != *expected_sha {
            bail!("captured {label} manifest SHA-256 drift");
        }
        println!("capture source {label}_manifest_sha256={actual_sha}");
    }
    println!(
        "capture integrity: {} real payload files / {} bytes hash-verified by CPU reference",
        manifest.asset_integrity.hashes_checked, manifest.asset_integrity.payload_bytes
    );
    Ok(manifest)
}

fn load_capture_input(
    manifest: &CaptureManifest,
    capture_dir: &Path,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>> {
    let matches: Vec<&CaptureInput> = manifest
        .inputs
        .iter()
        .filter(|input| input.name == name)
        .collect();
    if matches.len() != 1 {
        bail!("capture must contain exactly one {name} input");
    }
    let input = matches[0];
    if input.shape != expected_shape {
        bail!("capture {name} shape drift: {:?}", input.shape);
    }
    let expected_bytes = expected_shape.iter().try_fold(4usize, |bytes, &dim| {
        bytes
            .checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("capture {name} byte count overflow"))
    })?;
    if input.bytes != expected_bytes || input.file.components().count() != 1 {
        bail!("capture {name} byte/path contract drift");
    }
    let capture_root = capture_dir.canonicalize()?;
    let path = capture_root.join(&input.file).canonicalize()?;
    if !path.starts_with(&capture_root) {
        bail!("capture {name} escapes capture directory");
    }
    let bytes = std::fs::read(&path)?;
    if bytes.len() != expected_bytes || sha256_bytes(&bytes) != input.f32_le_sha256 {
        bail!("capture {name} bytes or SHA-256 drift");
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        bail!("capture {name} contains non-finite activation");
    }
    println!(
        "real activation {name}_f32_le_sha256={}",
        input.f32_le_sha256
    );
    Ok(values)
}

fn read_verified_payload(
    path: &Path,
    expected_bytes: usize,
    expected_sha256: &str,
    tensor: &str,
) -> Result<Vec<u8>> {
    if expected_sha256.len() != 64
        || !expected_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{tensor} has invalid manifest SHA-256");
    }
    let cache_root = Path::new(MODEL_DIR).join("range_cache").canonicalize()?;
    let resolved = path
        .canonicalize()
        .with_context(|| format!("resolve real payload {}", path.display()))?;
    if resolved.extension().and_then(|value| value.to_str()) != Some("bin")
        || !resolved.starts_with(&cache_root)
    {
        bail!("{tensor} payload escapes frozen range_cache");
    }
    let bytes = std::fs::read(&resolved).with_context(|| format!("read {tensor}"))?;
    let actual_sha256 = sha256_bytes(&bytes);
    if bytes.len() != expected_bytes || actual_sha256 != expected_sha256 {
        bail!("{tensor} payload size/SHA-256 drift");
    }
    println!("real payload {tensor} sha256={actual_sha256}");
    Ok(bytes)
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(sha256_bytes(
        &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
    ))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
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
