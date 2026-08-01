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
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ssd_inference::buffer::GpuBuffer;
use ssd_inference::device::VulkanContext;
use ssd_inference::s14_vulkan::{
    validate_e4m3fn_codes, validate_ue8m0_codes, S14Bf16AccumulateDispatch, S14Fp8Dispatch,
    S14MatvecShape, S14MoeAccumulateDispatch, S14Mxfp4Dispatch, S14NumericPipelines,
    S14OfficialExpertPrepareDispatch, S14RouteMixDispatch, S14SwigluLimitDispatch,
};
use ssd_inference::{VerifiedPayloadCache, VerifiedPayloadCacheStats};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const MODEL_DIR: &str = "D:/models/Polaris-S14";
const ROUTE_MANIFEST: &str = "D:/models/Polaris-S14/l42_real_layer_route_manifest.json";
const S14_MANIFEST: &str = "D:/models/Polaris-S14/s14_base_cache_manifest.json";
const CAPTURE_DIR_ENV: &str = "POLARIS_S14_L42_CAPTURE_DIR";
const FULLDEPTH_BRIDGE_DIR_ENV: &str = "POLARIS_FULLDEPTH43_VULKAN_BRIDGE_DIR";
const FULLDEPTH_BRIDGE_EVIDENCE_ENV: &str = "POLARIS_FULLDEPTH43_VULKAN_EVIDENCE";
const REVISION: &str = "7872f01b1d1fe23eabc4c98b48bffcef5a386062";
const REAL_EXPERT_ID: u32 = 126;
const AMD_VENDOR_ID: u32 = 0x1002;
const NAVI10_DEVICE_ID: u32 = 0x731f;
const ITERATIONS: u32 = 100;
const MOE_BATCH_ITERATIONS: u32 = 1;
const FULLDEPTH_BRIDGE_ITERATIONS: u32 = 20;
const WRITEBACK_WORKER_ARG: &str = "--fulldepth43-writeback-worker";
const WRITEBACK_PROTOCOL: &str = "polaris-fulldepth43-vulkan-writeback-v1";
const WRITEBACK_OUTPUT_FILE: &str = "vulkan_moe_branch.bf16le.bin";
const PAYLOAD_CACHE_GIB_ENV: &str = "POLARIS_VERIFIED_PAYLOAD_CACHE_GIB";
const DEFAULT_PAYLOAD_CACHE_GIB: usize = 10;
const MIN_PAYLOAD_CACHE_GIB: usize = 8;
const MAX_PAYLOAD_CACHE_GIB: usize = 12;

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
    w1: Arc<[u8]>,
    s1: Arc<[u8]>,
    w3: Arc<[u8]>,
    s3: Arc<[u8]>,
    w2: Arc<[u8]>,
    s2: Arc<[u8]>,
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

struct RoutedOfficialDispatch {
    w1: S14Mxfp4Dispatch,
    w3: S14Mxfp4Dispatch,
    prepare: S14OfficialExpertPrepareDispatch,
    w2: S14Mxfp4Dispatch,
    accumulate: S14Bf16AccumulateDispatch,
}

struct SharedOfficialDispatch {
    w1: S14Fp8Dispatch,
    w3: S14Fp8Dispatch,
    prepare: S14OfficialExpertPrepareDispatch,
    w2: S14Fp8Dispatch,
    accumulate: S14Bf16AccumulateDispatch,
}

#[derive(Serialize)]
struct Timing {
    iterations: u32,
    gpu_kernel_ms_mean: f64,
    submit_readback_sync_ms: f64,
}

#[derive(Serialize)]
struct ErrorStats {
    max_abs: f32,
    mean_abs: f64,
    rmse: f64,
    max_rel_for_abs_ref_gt_1e_5: f32,
    max_abs_reference: f32,
    rmse_reference: f64,
    relative_rmse: f64,
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
struct FullDepthBridgeManifest {
    format: String,
    revision: String,
    profile: String,
    layer: u32,
    position: u32,
    input_token_id: u32,
    completed_layers_before_capture: Vec<u32>,
    route_source: String,
    expert_ids: Vec<u32>,
    route_weights: Vec<f32>,
    route_weight_sum: f32,
    source_ffn_input_f32_le_sha256: String,
    input: CaptureInput,
    payload_count: usize,
    payload_bytes: u64,
    payloads: Vec<FullDepthBridgePayload>,
    reference_semantics: String,
}

#[derive(Deserialize)]
struct FullDepthBridgePayload {
    tensor: String,
    kind: String,
    expert_id: Option<u32>,
    dtype: String,
    shape: Vec<usize>,
    bytes: u64,
    path: PathBuf,
    sha256: String,
}

#[derive(Serialize)]
struct MoeBatchResult {
    cpu_reference_ms: f64,
    timing: Timing,
    error: ErrorStats,
    input_f32_le_sha256: String,
    cpu_reference_output_f32_le_sha256: String,
    gpu_output_f32_le_sha256: String,
    tolerance: MoeBatchTolerance,
    reference_semantics: &'static str,
}

#[derive(Serialize)]
struct MoeBatchTolerance {
    max_abs_limit: f32,
    rmse_limit: f64,
    relative_scale_factor: f64,
    passed: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WritebackRequest {
    protocol: String,
    op: String,
    request_id: String,
    manifest: PathBuf,
}

#[derive(Serialize)]
struct WritebackOutput {
    path: PathBuf,
    dtype: &'static str,
    shape: [usize; 3],
    bytes: usize,
    sha256: String,
}

#[derive(Serialize)]
struct WritebackResponse {
    protocol: &'static str,
    request_id: String,
    ok: bool,
    device: String,
    manifest_sha256: String,
    layer: u32,
    position: u32,
    input_token_id: u32,
    output: WritebackOutput,
    gpu_kernel_ms: f64,
    wall_ms: f64,
    payload_cache: WritebackPayloadCacheTelemetry,
    boundaries: [&'static str; 5],
    expansion_status: &'static str,
    claim_limit: &'static str,
}

#[derive(Debug, Serialize, PartialEq)]
struct WritebackPayloadCacheTelemetry {
    capacity_bytes: u64,
    entries: usize,
    current_bytes: u64,
    peak_bytes: u64,
    request_hits: u64,
    request_misses: u64,
    request_disk_bytes_read: u64,
    request_bytes_served: u64,
    total_hits: u64,
    total_misses: u64,
    total_evictions: u64,
    total_disk_bytes_read: u64,
    total_bytes_served: u64,
    total_hit_rate: f64,
}

impl WritebackPayloadCacheTelemetry {
    fn between(
        cache: &VerifiedPayloadCache,
        before: VerifiedPayloadCacheStats,
        after: VerifiedPayloadCacheStats,
    ) -> Result<Self> {
        Ok(Self {
            capacity_bytes: cache.capacity_bytes() as u64,
            entries: cache.len(),
            current_bytes: after.current_bytes,
            peak_bytes: after.peak_bytes,
            request_hits: monotonic_delta(after.hits, before.hits, "hits")?,
            request_misses: monotonic_delta(after.misses, before.misses, "misses")?,
            request_disk_bytes_read: monotonic_delta(
                after.disk_bytes_read,
                before.disk_bytes_read,
                "disk_bytes_read",
            )?,
            request_bytes_served: monotonic_delta(
                after.bytes_served,
                before.bytes_served,
                "bytes_served",
            )?,
            total_hits: after.hits,
            total_misses: after.misses,
            total_evictions: after.evictions,
            total_disk_bytes_read: after.disk_bytes_read,
            total_bytes_served: after.bytes_served,
            total_hit_rate: after.hit_rate(),
        })
    }
}

fn monotonic_delta(after: u64, before: u64, name: &str) -> Result<u64> {
    after
        .checked_sub(before)
        .ok_or_else(|| anyhow::anyhow!("verified payload cache {name} counter regressed"))
}

#[derive(Serialize)]
struct WritebackError<'a> {
    protocol: &'static str,
    request_id: &'a str,
    ok: bool,
    error: String,
    poisoned: bool,
}

struct OfficialMoeResult {
    branch_bf16: Vec<u16>,
    gpu_kernel_ms: f64,
}

#[derive(Serialize)]
struct FullDepthBridgeDeviceEvidence {
    name: String,
    vendor_id: String,
    device_id: String,
    driver_version_raw: String,
    timestamp_valid_bits: u32,
    timestamp_period_ns: f64,
}

#[derive(Serialize)]
struct FullDepthBridgeEvidence {
    format: &'static str,
    revision: &'static str,
    source_manifest_sha256: String,
    layer: u32,
    position: u32,
    input_token_id: u32,
    expert_ids: Vec<u32>,
    route_weights: Vec<f32>,
    route_weight_sum: f32,
    payload_count: usize,
    payload_bytes: u64,
    device: FullDepthBridgeDeviceEvidence,
    result: MoeBatchResult,
    bridge_wall_ms_including_payload_read_upload_pipeline_and_readback: f64,
    expansion_status: &'static str,
    claim_limit: &'static str,
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
    if std::env::args().any(|value| value == WRITEBACK_WORKER_ARG) {
        return run_fulldepth_writeback_worker();
    }
    if let Some(path) = std::env::var_os(FULLDEPTH_BRIDGE_DIR_ENV) {
        return run_fulldepth_bridge(PathBuf::from(path));
    }
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

fn run_fulldepth_writeback_worker() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    if properties.vendor_id != AMD_VENDOR_ID || properties.device_id != NAVI10_DEVICE_ID {
        bail!(
            "FullDepth43 writeback worker requires RX 5700 XT; found 0x{:04x}:0x{:04x} ({})",
            properties.vendor_id,
            properties.device_id,
            ctx.gpu_name
        );
    }
    let queue_properties = unsafe {
        ctx.instance
            .get_physical_device_queue_family_properties(ctx.physical)
    };
    let timestamp_bits = queue_properties[ctx.qf_graphics as usize].timestamp_valid_bits;
    if timestamp_bits == 0 {
        bail!("FullDepth43 writeback worker requires timestamp queries");
    }
    let timestamp_period_ns = properties.limits.timestamp_period as f64;
    let pipelines = S14NumericPipelines::new(&ctx)?;
    let payload_cache_capacity = payload_cache_capacity_bytes()?;
    let mut payload_cache = VerifiedPayloadCache::new(payload_cache_capacity)?;
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(
        &mut stdout,
        &serde_json::json!({
            "protocol": WRITEBACK_PROTOCOL,
            "op": "hello",
            "ready": true,
            "device": ctx.gpu_name,
            "vendor_id": format!("0x{:04x}", properties.vendor_id),
            "device_id": format!("0x{:04x}", properties.device_id),
            "persistent_context": true,
            "official_boundary_graph": true,
            "verified_payload_cache": true,
            "payload_cache_capacity_bytes": payload_cache_capacity,
        }),
    )?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;

    let stdin = std::io::stdin();
    for line in BufReader::new(stdin.lock()).lines() {
        let line = line?;
        if line.len() > 65_536 {
            bail!("writeback request exceeds 64 KiB");
        }
        let parsed: Result<WritebackRequest> =
            serde_json::from_str(&line).context("parse writeback JSON request");
        let request = match parsed {
            Ok(value) => value,
            Err(error) => {
                serde_json::to_writer(
                    &mut stdout,
                    &WritebackError {
                        protocol: WRITEBACK_PROTOCOL,
                        request_id: "unknown",
                        ok: false,
                        error: format!("{error:#}"),
                        poisoned: true,
                    },
                )?;
                stdout.write_all(b"\n")?;
                stdout.flush()?;
                pipelines.destroy(&ctx);
                bail!("writeback worker poisoned by invalid request");
            }
        };
        if request.protocol != WRITEBACK_PROTOCOL
            || request.op != "execute_single_layer"
            || request.request_id.is_empty()
            || request.request_id.len() > 128
            || !request
                .request_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
        {
            serde_json::to_writer(
                &mut stdout,
                &WritebackError {
                    protocol: WRITEBACK_PROTOCOL,
                    request_id: &request.request_id,
                    ok: false,
                    error: "writeback request contract drift".to_string(),
                    poisoned: true,
                },
            )?;
            stdout.write_all(b"\n")?;
            stdout.flush()?;
            pipelines.destroy(&ctx);
            bail!("writeback worker poisoned by contract drift");
        }

        let request_id = request.request_id.clone();
        match execute_writeback_request(
            &ctx,
            &pipelines,
            timestamp_bits,
            timestamp_period_ns,
            &mut payload_cache,
            request,
        ) {
            Ok(response) => {
                serde_json::to_writer(&mut stdout, &response)?;
                stdout.write_all(b"\n")?;
                stdout.flush()?;
            }
            Err(error) => {
                serde_json::to_writer(
                    &mut stdout,
                    &WritebackError {
                        protocol: WRITEBACK_PROTOCOL,
                        request_id: &request_id,
                        ok: false,
                        error: format!("{error:#}"),
                        poisoned: true,
                    },
                )?;
                stdout.write_all(b"\n")?;
                stdout.flush()?;
                pipelines.destroy(&ctx);
                return Err(error.context("writeback worker poisoned"));
            }
        }
    }
    pipelines.destroy(&ctx);
    Ok(())
}

fn execute_writeback_request(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    payload_cache: &mut VerifiedPayloadCache,
    request: WritebackRequest,
) -> Result<WritebackResponse> {
    let started = Instant::now();
    let cache_before = payload_cache.stats();
    let manifest_path = request
        .manifest
        .canonicalize()
        .with_context(|| format!("resolve writeback manifest {}", request.manifest.display()))?;
    if manifest_path.file_name().and_then(|value| value.to_str()) != Some("bridge_manifest.json") {
        bail!("writeback manifest filename drift");
    }
    let capture_root = manifest_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("writeback manifest has no parent"))?
        .canonicalize()?;
    let manifest_bytes = std::fs::read(&manifest_path)?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let manifest: FullDepthBridgeManifest =
        serde_json::from_slice(&manifest_bytes).context("parse FullDepth43 writeback manifest")?;
    validate_writeback_manifest(&manifest)?;
    let input = load_fulldepth_bridge_input(&manifest, &capture_root)?;
    let routed = manifest
        .expert_ids
        .iter()
        .zip(&manifest.route_weights)
        .map(|(&expert_id, &mix_weight)| {
            let (w1, s1) =
                load_fulldepth_bridge_pair_cached(&manifest, expert_id, "w1", payload_cache)?;
            let (w3, s3) =
                load_fulldepth_bridge_pair_cached(&manifest, expert_id, "w3", payload_cache)?;
            let (w2, s2) =
                load_fulldepth_bridge_pair_cached(&manifest, expert_id, "w2", payload_cache)?;
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
    let (w1, s1) = load_fulldepth_bridge_shared_pair_cached(&manifest, "w1", payload_cache)?;
    let (w3, s3) = load_fulldepth_bridge_shared_pair_cached(&manifest, "w3", payload_cache)?;
    let (w2, s2) = load_fulldepth_bridge_shared_pair_cached(&manifest, "w2", payload_cache)?;
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
    let result = run_official_top6_shared_moe_batch(
        ctx,
        pipelines,
        timestamp_bits,
        timestamp_period_ns,
        &input,
        &routed,
        &shared,
    )?;
    let mut output_bytes = Vec::with_capacity(result.branch_bf16.len() * 2);
    for value in result.branch_bf16 {
        output_bytes.extend_from_slice(&value.to_le_bytes());
    }
    if output_bytes.len() != 4096 * 2 {
        bail!("writeback BF16 output byte count drift");
    }
    let output_path = capture_root.join(WRITEBACK_OUTPUT_FILE);
    if output_path.exists() {
        bail!("refuse to overwrite existing writeback output");
    }
    let temporary = capture_root.join(format!("{WRITEBACK_OUTPUT_FILE}.tmp"));
    if temporary.exists() {
        bail!("stale writeback temporary output exists");
    }
    {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&output_bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(&temporary, &output_path)?;
    let output_path = output_path.canonicalize()?;
    if !output_path.starts_with(&capture_root) {
        bail!("writeback output escaped capture directory");
    }
    let cache_after = payload_cache.stats();
    let payload_cache_telemetry =
        WritebackPayloadCacheTelemetry::between(payload_cache, cache_before, cache_after)?;
    Ok(WritebackResponse {
        protocol: WRITEBACK_PROTOCOL,
        request_id: request.request_id,
        ok: true,
        device: ctx.gpu_name.clone(),
        manifest_sha256,
        layer: manifest.layer,
        position: manifest.position,
        input_token_id: manifest.input_token_id,
        output: WritebackOutput {
            path: output_path,
            dtype: "bf16_le",
            shape: [1, 1, 4096],
            bytes: output_bytes.len(),
            sha256: sha256_bytes(&output_bytes),
        },
        gpu_kernel_ms: result.gpu_kernel_ms,
        wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        payload_cache: payload_cache_telemetry,
        boundaries: [
            "w1_w3_output_round_to_bf16",
            "limited_swiglu_f32",
            "route_weight_before_w2_then_bf16",
            "e4m3fn_group128_activation_requantize",
            "w2_output_round_to_bf16_then_fp32_accumulate",
        ],
        expansion_status: "single_real_layer_writeback_only",
        claim_limit: "One FullDepth43 MoE branch computed by the official-boundary Vulkan graph; no full-layer, full-token, speed, or quality claim.",
    })
}

fn payload_cache_capacity_bytes() -> Result<usize> {
    let gib = match std::env::var(PAYLOAD_CACHE_GIB_ENV) {
        Ok(value) => value
            .parse::<usize>()
            .with_context(|| format!("parse {PAYLOAD_CACHE_GIB_ENV}={value:?}"))?,
        Err(std::env::VarError::NotPresent) => DEFAULT_PAYLOAD_CACHE_GIB,
        Err(error) => return Err(error).context(PAYLOAD_CACHE_GIB_ENV),
    };
    if !(MIN_PAYLOAD_CACHE_GIB..=MAX_PAYLOAD_CACHE_GIB).contains(&gib) {
        bail!(
            "{PAYLOAD_CACHE_GIB_ENV} must be between {MIN_PAYLOAD_CACHE_GIB} and {MAX_PAYLOAD_CACHE_GIB} GiB"
        );
    }
    gib.checked_mul(1024 * 1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("verified payload cache byte capacity overflow"))
}

fn validate_writeback_manifest(manifest: &FullDepthBridgeManifest) -> Result<()> {
    let expected_prefix: Vec<u32> = (0..manifest.layer).collect();
    if manifest.format != "polaris-fulldepth43-vulkan-bridge-capture-v1"
        || manifest.revision != REVISION
        || manifest.profile != "fulldepth43_native_top6"
        || manifest.layer > 42
        || manifest.position != 0
        || manifest.completed_layers_before_capture != expected_prefix
        || manifest.route_source.is_empty()
        || manifest.expert_ids.len() != 6
        || manifest.route_weights.len() != 6
        || manifest.payload_count != 42
        || manifest.payloads.len() != 42
        || manifest.input.name != "ffn_input_activation_quant"
        || manifest.input.shape != [1, 1, 4096]
        || manifest.input.bytes != 4096 * std::mem::size_of::<f32>()
        || manifest.source_ffn_input_f32_le_sha256.len() != 64
    {
        bail!("FullDepth43 writeback manifest contract drift");
    }
    let mut unique_experts = manifest.expert_ids.clone();
    unique_experts.sort_unstable();
    unique_experts.dedup();
    if unique_experts.len() != 6
        || manifest
            .route_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        || (manifest.route_weights.iter().sum::<f32>() - 1.5).abs() > 2.0e-6
        || (manifest.route_weight_sum - 1.5).abs() > 2.0e-6
    {
        bail!("FullDepth43 writeback route is not a valid native top-6");
    }
    let observed_payload_bytes = manifest
        .payloads
        .iter()
        .try_fold(0u64, |total, payload| total.checked_add(payload.bytes))
        .ok_or_else(|| anyhow::anyhow!("FullDepth43 writeback payload byte overflow"))?;
    if observed_payload_bytes != manifest.payload_bytes {
        bail!("FullDepth43 writeback payload byte total drift");
    }
    Ok(())
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let rounded = if exponent == 0x7f80_0000 {
        bits
    } else {
        bits.wrapping_add(0x0000_7fff + ((bits >> 16) & 1))
    };
    (rounded >> 16) as u16
}

fn run_official_top6_shared_moe_batch(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    x: &[f32],
    routed: &[MoePayload],
    shared: &MoePayload,
) -> Result<OfficialMoeResult> {
    if x.len() != 4096 || routed.len() != 6 || shared.expert_id.is_some() {
        bail!("official FullDepth43 MoE shape/route contract drift");
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
            bail!("official routed payload contract drift");
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

    let buffers = MoeBatchBuffers::new(ctx, x, routed, shared)?;
    buffers.upload(ctx)?;
    let routed_dispatches = buffers
        .routed
        .iter()
        .zip(routed)
        .map(|(weights, payload)| {
            Ok(RoutedOfficialDispatch {
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
                prepare: pipelines.bind_official_expert_prepare(
                    ctx,
                    2048,
                    payload.mix_weight,
                    &buffers.gate,
                    &buffers.up,
                    &buffers.hidden,
                )?,
                w2: pipelines.bind_mxfp4(
                    ctx,
                    down_shape,
                    &buffers.hidden,
                    &weights.w2.device,
                    &weights.s2.device,
                    &buffers.down,
                )?,
                accumulate: pipelines.bind_bf16_accumulate(
                    ctx,
                    4096,
                    &buffers.down,
                    &buffers.accumulator,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let shared_dispatch = SharedOfficialDispatch {
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
        prepare: pipelines.bind_official_expert_prepare(
            ctx,
            2048,
            1.0,
            &buffers.gate,
            &buffers.up,
            &buffers.hidden,
        )?,
        w2: pipelines.bind_fp8(
            ctx,
            shared_down_shape,
            &buffers.hidden,
            &buffers.shared.w2.device,
            &buffers.shared.s2.device,
            &buffers.down,
        )?,
        accumulate: pipelines.bind_bf16_accumulate(
            ctx,
            4096,
            &buffers.down,
            &buffers.accumulator,
        )?,
    };

    let gpu_kernel_ms = record_official_moe_once(
        ctx,
        pipelines,
        &buffers,
        timestamp_bits,
        timestamp_period_ns,
        &routed_dispatches,
        &shared_dispatch,
    )?;
    let output = buffers.output();
    if output.iter().any(|value| !value.is_finite()) {
        bail!("official Vulkan MoE output contains non-finite values");
    }
    let branch_bf16 = output.into_iter().map(f32_to_bf16_bits).collect();

    shared_dispatch.accumulate.binder.destroy(ctx);
    shared_dispatch.w2.binder.destroy(ctx);
    shared_dispatch.prepare.binder.destroy(ctx);
    shared_dispatch.w3.binder.destroy(ctx);
    shared_dispatch.w1.binder.destroy(ctx);
    for dispatch in routed_dispatches.into_iter().rev() {
        dispatch.accumulate.binder.destroy(ctx);
        dispatch.w2.binder.destroy(ctx);
        dispatch.prepare.binder.destroy(ctx);
        dispatch.w3.binder.destroy(ctx);
        dispatch.w1.binder.destroy(ctx);
    }
    buffers.destroy(ctx);
    Ok(OfficialMoeResult {
        branch_bf16,
        gpu_kernel_ms,
    })
}

fn record_official_moe_once(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    buffers: &MoeBatchBuffers,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    routed: &[RoutedOfficialDispatch],
    shared: &SharedOfficialDispatch,
) -> Result<f64> {
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
        let shader_raw = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        let shader_serial = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);

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
            pipelines.cmd_official_expert_prepare(ctx, cb, &dispatch.prepare);
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
            pipelines.cmd_bf16_accumulate(ctx, cb, &dispatch.accumulate);
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
        pipelines.cmd_official_expert_prepare(ctx, cb, &shared.prepare);
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
        pipelines.cmd_bf16_accumulate(ctx, cb, &shared.accumulate);
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
        submit_and_wait(ctx, cb)?;
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
        let elapsed_ms = elapsed_ticks as f64 * timestamp_period_ns / 1_000_000.0;
        ctx.device.destroy_query_pool(queries, None);
        ctx.device.destroy_command_pool(pool, None);
        Ok(elapsed_ms)
    }
}

fn run_fulldepth_bridge(capture_dir: PathBuf) -> Result<()> {
    let bridge_started = Instant::now();
    let capture_root = capture_dir
        .canonicalize()
        .with_context(|| format!("resolve FullDepth43 bridge {}", capture_dir.display()))?;
    let manifest_path = capture_root.join("bridge_manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let manifest: FullDepthBridgeManifest = serde_json::from_slice(&manifest_bytes)
        .context("parse FullDepth43 Vulkan bridge manifest")?;
    let expected_prefix: Vec<u32> = (0..manifest.layer).collect();
    if manifest.format != "polaris-fulldepth43-vulkan-bridge-capture-v1"
        || manifest.revision != REVISION
        || manifest.profile != "fulldepth43_native_top6"
        || manifest.layer > 42
        || manifest.position != 0
        || manifest.completed_layers_before_capture != expected_prefix
        || manifest.route_source.is_empty()
        || manifest.expert_ids.len() != 6
        || manifest.route_weights.len() != 6
        || manifest.payload_count != 42
        || manifest.payloads.len() != 42
        || manifest.input.name != "ffn_input_activation_quant"
        || manifest.input.shape != [1, 1, 4096]
        || manifest.input.bytes != 4096 * std::mem::size_of::<f32>()
        || manifest.source_ffn_input_f32_le_sha256.len() != 64
        || !manifest.reference_semantics.contains("FullDepth43 live")
    {
        bail!("FullDepth43 Vulkan bridge contract drift");
    }
    let mut unique_experts = manifest.expert_ids.clone();
    unique_experts.sort_unstable();
    unique_experts.dedup();
    if unique_experts.len() != 6
        || manifest
            .route_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        || (manifest.route_weights.iter().sum::<f32>() - 1.5).abs() > 2.0e-6
        || (manifest.route_weight_sum - 1.5).abs() > 2.0e-6
    {
        bail!("FullDepth43 bridge route is not a valid native top-6");
    }
    let observed_payload_bytes = manifest
        .payloads
        .iter()
        .try_fold(0u64, |total, payload| total.checked_add(payload.bytes))
        .ok_or_else(|| anyhow::anyhow!("FullDepth43 bridge payload byte overflow"))?;
    if observed_payload_bytes != manifest.payload_bytes {
        bail!("FullDepth43 bridge payload byte total drift");
    }

    let input = load_fulldepth_bridge_input(&manifest, &capture_root)?;
    println!(
        "FullDepth43 live L{} position{} token{} input_sha256={} source_ffn_sha256={}",
        manifest.layer,
        manifest.position,
        manifest.input_token_id,
        manifest.input.f32_le_sha256,
        manifest.source_ffn_input_f32_le_sha256,
    );
    println!(
        "FullDepth43 live route top6={:?}, weights={:?}, source={}, payloads={} / {} bytes",
        manifest.expert_ids,
        manifest.route_weights,
        manifest.route_source,
        manifest.payload_count,
        manifest.payload_bytes,
    );

    let routed = manifest
        .expert_ids
        .iter()
        .zip(&manifest.route_weights)
        .map(|(&expert_id, &mix_weight)| {
            let (w1, s1) = load_fulldepth_bridge_pair(&manifest, expert_id, "w1")?;
            let (w3, s3) = load_fulldepth_bridge_pair(&manifest, expert_id, "w3")?;
            let (w2, s2) = load_fulldepth_bridge_pair(&manifest, expert_id, "w2")?;
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
    let (w1, s1) = load_fulldepth_bridge_shared_pair(&manifest, "w1")?;
    let (w3, s3) = load_fulldepth_bridge_shared_pair(&manifest, "w3")?;
    let (w2, s2) = load_fulldepth_bridge_shared_pair(&manifest, "w2")?;
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

    let ctx = VulkanContext::init()?;
    let queue_props = unsafe {
        ctx.instance
            .get_physical_device_queue_family_properties(ctx.physical)
    };
    let timestamp_bits = queue_props[ctx.qf_graphics as usize].timestamp_valid_bits;
    let physical_properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    if physical_properties.vendor_id != AMD_VENDOR_ID
        || physical_properties.device_id != NAVI10_DEVICE_ID
        || timestamp_bits == 0
    {
        bail!(
            "FullDepth43 bridge requires timestamp-capable RX 5700 XT; found 0x{:04x}:0x{:04x} ({})",
            physical_properties.vendor_id,
            physical_properties.device_id,
            ctx.gpu_name
        );
    }
    let timestamp_period_ns = physical_properties.limits.timestamp_period as f64;
    println!(
        "FullDepth43 Vulkan bridge GPU={} vendor=0x{:04x} device=0x{:04x} driver=0x{:08x} timestamp_bits={} timestamp_period_ns={}",
        ctx.gpu_name,
        physical_properties.vendor_id,
        physical_properties.device_id,
        physical_properties.driver_version,
        timestamp_bits,
        timestamp_period_ns,
    );
    let pipelines = S14NumericPipelines::new(&ctx)?;
    let result = run_top6_shared_moe_batch(
        &ctx,
        &pipelines,
        timestamp_bits,
        timestamp_period_ns,
        &input,
        &routed,
        &shared,
        FULLDEPTH_BRIDGE_ITERATIONS,
    )?;
    pipelines.destroy(&ctx);

    let evidence = FullDepthBridgeEvidence {
        format: "polaris-fulldepth43-vulkan-bridge-evidence-v1",
        revision: REVISION,
        source_manifest_sha256: manifest_sha256,
        layer: manifest.layer,
        position: manifest.position,
        input_token_id: manifest.input_token_id,
        expert_ids: manifest.expert_ids,
        route_weights: manifest.route_weights,
        route_weight_sum: manifest.route_weight_sum,
        payload_count: manifest.payload_count,
        payload_bytes: manifest.payload_bytes,
        device: FullDepthBridgeDeviceEvidence {
            name: ctx.gpu_name.clone(),
            vendor_id: format!("0x{:04x}", physical_properties.vendor_id),
            device_id: format!("0x{:04x}", physical_properties.device_id),
            driver_version_raw: format!("0x{:08x}", physical_properties.driver_version),
            timestamp_valid_bits: timestamp_bits,
            timestamp_period_ns,
        },
        result,
        bridge_wall_ms_including_payload_read_upload_pipeline_and_readback: bridge_started
            .elapsed()
            .as_secs_f64()
            * 1000.0,
        expansion_status: "single_real_layer_only",
        claim_limit: "Real FullDepth43 live layer activation and native top-6 payloads executed by the existing bounded Vulkan minimal MoE batch. This is not official BF16/requantized expert parity and is not wired into token commit.",
    };
    let evidence_path = std::env::var_os(FULLDEPTH_BRIDGE_EVIDENCE_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| capture_root.join("vulkan_evidence.json"));
    if let Some(parent) = evidence_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&evidence_path, serde_json::to_vec_pretty(&evidence)?)?;
    println!("FullDepth43 Vulkan evidence={}", evidence_path.display());
    Ok(())
}

fn load_fulldepth_bridge_input(
    manifest: &FullDepthBridgeManifest,
    capture_root: &Path,
) -> Result<Vec<f32>> {
    if manifest.input.file.components().count() != 1 {
        bail!("FullDepth43 bridge input must be a capture-local file");
    }
    let path = capture_root
        .join(&manifest.input.file)
        .canonicalize()
        .context("resolve FullDepth43 bridge input")?;
    if !path.starts_with(capture_root) {
        bail!("FullDepth43 bridge input escapes capture directory");
    }
    let bytes = std::fs::read(&path)?;
    if bytes.len() != manifest.input.bytes || sha256_bytes(&bytes) != manifest.input.f32_le_sha256 {
        bail!("FullDepth43 bridge input bytes/SHA-256 drift");
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    if values.len() != 4096 || values.iter().any(|value| !value.is_finite()) {
        bail!("FullDepth43 bridge input shape/value drift");
    }
    Ok(values)
}

fn fulldepth_bridge_payload<'a>(
    manifest: &'a FullDepthBridgeManifest,
    tensor: &str,
    kind: &str,
    expert_id: Option<u32>,
) -> Result<&'a FullDepthBridgePayload> {
    let matches: Vec<&FullDepthBridgePayload> = manifest
        .payloads
        .iter()
        .filter(|payload| {
            payload.tensor == tensor && payload.kind == kind && payload.expert_id == expert_id
        })
        .collect();
    if matches.len() != 1 {
        bail!("FullDepth43 bridge must contain exactly one {tensor}");
    }
    Ok(matches[0])
}

fn load_fulldepth_bridge_pair(
    manifest: &FullDepthBridgeManifest,
    expert_id: u32,
    component: &str,
) -> Result<(Arc<[u8]>, Arc<[u8]>)> {
    let (weight, scale) = fulldepth_bridge_pair_entries(manifest, expert_id, component)?;
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

fn load_fulldepth_bridge_pair_cached(
    manifest: &FullDepthBridgeManifest,
    expert_id: u32,
    component: &str,
    cache: &mut VerifiedPayloadCache,
) -> Result<(Arc<[u8]>, Arc<[u8]>)> {
    let (weight, scale) = fulldepth_bridge_pair_entries(manifest, expert_id, component)?;
    Ok((
        read_verified_payload_cached(
            cache,
            &weight.path,
            weight.bytes as usize,
            &weight.sha256,
            &weight.tensor,
        )?,
        read_verified_payload_cached(
            cache,
            &scale.path,
            scale.bytes as usize,
            &scale.sha256,
            &scale.tensor,
        )?,
    ))
}

fn fulldepth_bridge_pair_entries<'a>(
    manifest: &'a FullDepthBridgeManifest,
    expert_id: u32,
    component: &str,
) -> Result<(&'a FullDepthBridgePayload, &'a FullDepthBridgePayload)> {
    let prefix = format!(
        "layers.{}.ffn.experts.{expert_id}.{component}",
        manifest.layer
    );
    let weight = fulldepth_bridge_payload(
        manifest,
        &format!("{prefix}.weight"),
        "routed",
        Some(expert_id),
    )?;
    let scale = fulldepth_bridge_payload(
        manifest,
        &format!("{prefix}.scale"),
        "routed",
        Some(expert_id),
    )?;
    if weight.dtype != "I8"
        || scale.dtype != "F8_E8M0"
        || weight.bytes != 4_194_304
        || scale.bytes != 262_144
        || weight.shape.iter().product::<usize>() != weight.bytes as usize
        || scale.shape.iter().product::<usize>() != scale.bytes as usize
    {
        bail!("FullDepth43 bridge E{expert_id}/{component} physical ABI drift");
    }
    Ok((weight, scale))
}

fn load_fulldepth_bridge_shared_pair(
    manifest: &FullDepthBridgeManifest,
    component: &str,
) -> Result<(Arc<[u8]>, Arc<[u8]>)> {
    let (weight, scale) = fulldepth_bridge_shared_pair_entries(manifest, component)?;
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

fn load_fulldepth_bridge_shared_pair_cached(
    manifest: &FullDepthBridgeManifest,
    component: &str,
    cache: &mut VerifiedPayloadCache,
) -> Result<(Arc<[u8]>, Arc<[u8]>)> {
    let (weight, scale) = fulldepth_bridge_shared_pair_entries(manifest, component)?;
    Ok((
        read_verified_payload_cached(
            cache,
            &weight.path,
            weight.bytes as usize,
            &weight.sha256,
            &weight.tensor,
        )?,
        read_verified_payload_cached(
            cache,
            &scale.path,
            scale.bytes as usize,
            &scale.sha256,
            &scale.tensor,
        )?,
    ))
}

fn fulldepth_bridge_shared_pair_entries<'a>(
    manifest: &'a FullDepthBridgeManifest,
    component: &str,
) -> Result<(&'a FullDepthBridgePayload, &'a FullDepthBridgePayload)> {
    let prefix = format!("layers.{}.ffn.shared_experts.{component}", manifest.layer);
    let weight = fulldepth_bridge_payload(manifest, &format!("{prefix}.weight"), "shared", None)?;
    let scale = fulldepth_bridge_payload(manifest, &format!("{prefix}.scale"), "shared", None)?;
    if weight.dtype != "F8_E4M3"
        || scale.dtype != "F8_E8M0"
        || weight.bytes != 8_388_608
        || scale.bytes != 512
        || weight.shape.iter().product::<usize>() != weight.bytes as usize
        || scale.shape.iter().product::<usize>() != scale.bytes as usize
    {
        bail!("FullDepth43 bridge shared/{component} physical ABI drift");
    }
    Ok((weight, scale))
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

    let _ = run_top6_shared_moe_batch(
        ctx,
        pipelines,
        timestamp_bits,
        timestamp_period_ns,
        w1_w3_input,
        &routed,
        &shared,
        MOE_BATCH_ITERATIONS,
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
) -> Result<(Arc<[u8]>, Arc<[u8]>)> {
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

fn load_shared_pair(manifest: &S14Manifest, component: &str) -> Result<(Arc<[u8]>, Arc<[u8]>)> {
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
    iterations: u32,
) -> Result<MoeBatchResult> {
    if x.len() != 4096 || routed.len() != 6 || shared.expert_id.is_some() || iterations == 0 {
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
        iterations,
    )?;
    let actual = buffers.output();
    let error = error_stats(&actual, &expected)?;
    let relative_scale_factor = 2.5e-5f64;
    let max_abs_limit =
        1.0e-3f32.max((error.max_abs_reference as f64 * relative_scale_factor) as f32);
    let rmse_limit = 1.5e-4f64.max(error.rmse_reference * relative_scale_factor);
    enforce_error(&error, max_abs_limit, rmse_limit)?;
    println!(
        "GPU-resident real L42 top6+shared minimal MoE batch: ids={:?}, weights={:?}, cpu_ref_ms={cpu_ms:.6}, iterations={}, gpu_fill_plus_35_dispatch_barriers_ms_mean={:.7}, submit_readback_sync_ms={:.6}, max_abs={:.9e}, mean_abs={:.9e}, rmse={:.9e}, max_abs_ref={:.9e}, rmse_ref={:.9e}, relative_rmse={:.9e}, max_rel_abs_ref_gt_1e-5={:.9e}, max_abs_limit={:.9e}, rmse_limit={:.9e}",
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
        error.max_abs_reference,
        error.rmse_reference,
        error.relative_rmse,
        error.max_rel_for_abs_ref_gt_1e_5,
        max_abs_limit,
        rmse_limit,
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
    Ok(MoeBatchResult {
        cpu_reference_ms: cpu_ms,
        timing,
        error,
        input_f32_le_sha256: sha256_f32_le(x),
        cpu_reference_output_f32_le_sha256: sha256_f32_le(&expected),
        gpu_output_f32_le_sha256: sha256_f32_le(&actual),
        tolerance: MoeBatchTolerance {
            max_abs_limit,
            rmse_limit,
            relative_scale_factor,
            passed: true,
        },
        reference_semantics: "F32 packed decode and route-weight-after-w2 accumulation for the existing bounded Vulkan minimal top6+shared chain",
    })
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
    iterations: u32,
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
        for iteration in 0..iterations {
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
            elapsed_ticks as f64 * timestamp_period_ns / 1_000_000.0 / iterations as f64;
        ctx.device.destroy_query_pool(queries, None);
        ctx.device.destroy_command_pool(pool, None);
        Ok(Timing {
            iterations,
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
    let mut max_abs_reference = 0.0f32;
    let mut sum_reference_square = 0.0f64;
    for (&a, &e) in actual.iter().zip(expected) {
        if !a.is_finite() || !e.is_finite() {
            bail!("non-finite numerical result: actual={a}, expected={e}");
        }
        let delta = (a - e).abs();
        max_abs_reference = max_abs_reference.max(e.abs());
        sum_reference_square += (e as f64) * (e as f64);
        max_abs = max_abs.max(delta);
        sum_abs += delta as f64;
        sum_square += (delta as f64) * (delta as f64);
        if e.abs() > 1.0e-5 {
            max_rel = max_rel.max(delta / e.abs());
        }
    }
    let rmse = (sum_square / actual.len() as f64).sqrt();
    let rmse_reference = (sum_reference_square / actual.len() as f64).sqrt();
    Ok(ErrorStats {
        max_abs,
        mean_abs: sum_abs / actual.len() as f64,
        rmse,
        max_rel_for_abs_ref_gt_1e_5: max_rel,
        max_abs_reference,
        rmse_reference,
        relative_rmse: rmse / rmse_reference.max(f64::MIN_POSITIVE),
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
) -> Result<Arc<[u8]>> {
    if expected_sha256.len() != 64
        || !expected_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{tensor} has invalid manifest SHA-256");
    }
    let cache_root = Path::new(MODEL_DIR).join("range_cache").canonicalize()?;
    let resolved = resolve_verified_payload_path(path, &cache_root, tensor)?;
    let bytes = std::fs::read(&resolved).with_context(|| format!("read {tensor}"))?;
    let actual_sha256 = sha256_bytes(&bytes);
    if bytes.len() != expected_bytes || actual_sha256 != expected_sha256 {
        bail!("{tensor} payload size/SHA-256 drift");
    }
    eprintln!("real payload {tensor} sha256={actual_sha256}");
    Ok(bytes.into())
}

fn read_verified_payload_cached(
    cache: &mut VerifiedPayloadCache,
    path: &Path,
    expected_bytes: usize,
    expected_sha256: &str,
    tensor: &str,
) -> Result<Arc<[u8]>> {
    let cache_root = Path::new(MODEL_DIR).join("range_cache").canonicalize()?;
    read_verified_payload_cached_with_root(
        cache,
        &cache_root,
        path,
        expected_bytes,
        expected_sha256,
        tensor,
    )
}

fn read_verified_payload_cached_with_root(
    cache: &mut VerifiedPayloadCache,
    cache_root: &Path,
    path: &Path,
    expected_bytes: usize,
    expected_sha256: &str,
    tensor: &str,
) -> Result<Arc<[u8]>> {
    let resolved = resolve_verified_payload_path(path, cache_root, tensor)?;
    cache
        .load_verified(&resolved, expected_bytes, expected_sha256)
        .with_context(|| format!("load cached verified payload {tensor}"))
}

fn resolve_verified_payload_path(path: &Path, cache_root: &Path, tensor: &str) -> Result<PathBuf> {
    let resolved = path
        .canonicalize()
        .with_context(|| format!("resolve real payload {}", path.display()))?;
    if resolved.extension().and_then(|value| value.to_str()) != Some("bin")
        || !resolved.starts_with(cache_root)
    {
        bail!("{tensor} payload escapes frozen range_cache");
    }
    Ok(resolved)
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

fn sha256_f32_le(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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

#[cfg(test)]
mod writeback_payload_cache_tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct FixtureDir(PathBuf);

    impl FixtureDir {
        fn new() -> Self {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "polaris-writeback-payload-cache-{}-{stamp}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for FixtureDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn worker_payload_reader_hits_arc_cache_without_second_read_or_hash() {
        let fixture = FixtureDir::new();
        let range_cache = fixture.0.join("range_cache");
        std::fs::create_dir(&range_cache).unwrap();
        let range_cache = range_cache.canonicalize().unwrap();
        let payload_path = range_cache.join("expert.bin");
        std::fs::write(&payload_path, b"frozen").unwrap();
        let expected_sha256 = sha256_bytes(b"frozen");
        let mut cache = VerifiedPayloadCache::new(64).unwrap();

        let before_first = cache.stats();
        let first = read_verified_payload_cached_with_root(
            &mut cache,
            &range_cache,
            &payload_path,
            6,
            &expected_sha256,
            "layers.0.ffn.experts.0.w1.weight",
        )
        .unwrap();
        let after_first = cache.stats();
        let first_telemetry =
            WritebackPayloadCacheTelemetry::between(&cache, before_first, after_first).unwrap();
        assert_eq!(first_telemetry.request_hits, 0);
        assert_eq!(first_telemetry.request_misses, 1);
        assert_eq!(first_telemetry.request_disk_bytes_read, 6);
        assert_eq!(first_telemetry.request_bytes_served, 6);

        // 同一路径被外部改写后，当前 worker 的已验证不可变副本仍然可信；
        // 第二次请求必须命中 Arc，不能再次读盘或重新哈希损坏内容。
        std::fs::write(&payload_path, b"damage").unwrap();
        let before_second = cache.stats();
        let second = read_verified_payload_cached_with_root(
            &mut cache,
            &range_cache,
            &payload_path,
            6,
            &expected_sha256,
            "layers.0.ffn.experts.0.w1.weight",
        )
        .unwrap();
        let after_second = cache.stats();
        let second_telemetry =
            WritebackPayloadCacheTelemetry::between(&cache, before_second, after_second).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(&*second, b"frozen");
        assert_eq!(second_telemetry.request_hits, 1);
        assert_eq!(second_telemetry.request_misses, 0);
        assert_eq!(second_telemetry.request_disk_bytes_read, 0);
        assert_eq!(second_telemetry.request_bytes_served, 6);
        assert_eq!(second_telemetry.total_hits, 1);
        assert_eq!(second_telemetry.total_misses, 1);
        assert_eq!(second_telemetry.total_disk_bytes_read, 6);
        assert_eq!(second_telemetry.total_hit_rate, 0.5);
    }

    #[test]
    fn worker_payload_reader_rejects_files_outside_frozen_range_cache() {
        let fixture = FixtureDir::new();
        let range_cache = fixture.0.join("range_cache");
        std::fs::create_dir(&range_cache).unwrap();
        let range_cache = range_cache.canonicalize().unwrap();
        let outside = fixture.0.join("outside.bin");
        std::fs::write(&outside, b"outside").unwrap();
        let mut cache = VerifiedPayloadCache::new(64).unwrap();
        let error = read_verified_payload_cached_with_root(
            &mut cache,
            &range_cache,
            &outside,
            7,
            &sha256_bytes(b"outside"),
            "escape",
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("escapes frozen range_cache"));
        assert_eq!(cache.stats().requests, 0);
    }
}
