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
    S14GroupedFp8Bf16Dispatch, S14GroupedMatvecShape, S14MatvecShape, S14MoeAccumulateDispatch,
    S14Mxfp4Dispatch, S14NumericPipelines, S14OfficialExpertPrepareDispatch, S14RouteMixDispatch,
    S14SwigluLimitDispatch,
};
use ssd_inference::{VerifiedPayloadCache, VerifiedPayloadCacheStats};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
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
const PRODUCTION_WORKER_ARG: &str = "--fulldepth43-production-worker";
const FP8_PROJECTION_WORKER_ARG: &str = "--l42-wq-a-fp8-projection-worker";
const FULLDEPTH_FP8_ATTENTION_WORKER_ARG: &str = "--fulldepth43-packed-fp8-attention-worker";
const FP8_PROJECTION_SUITE_ARG: &str = "--l42-standard-fp8-projection-suite";
const WO_A_GROUPED_SUITE_ARG: &str = "--l42-wo-a-grouped-suite";
const WRITEBACK_PROTOCOL: &str = "polaris-fulldepth43-vulkan-writeback-v1";
const FP8_PROJECTION_PROTOCOL: &str = "polaris-fulldepth43-packed-fp8-projection-v1";
const FULLDEPTH_FP8_ATTENTION_PROTOCOL: &str = "polaris-fulldepth43-packed-fp8-attention-v1";
const FP8_PROJECTION_FIXTURE_DIR_ENV: &str = "POLARIS_L42_FP8_PROJECTION_FIXTURE_DIR";
const WO_A_GROUPED_FIXTURE_DIR_ENV: &str = "POLARIS_L42_WO_A_FIXTURE_DIR";
const WRITEBACK_OUTPUT_FILE: &str = "vulkan_moe_branch.bf16le.bin";
const PAYLOAD_CACHE_GIB_ENV: &str = "POLARIS_VERIFIED_PAYLOAD_CACHE_GIB";
const DEFAULT_PAYLOAD_CACHE_GIB: usize = 10;
const MIN_PAYLOAD_CACHE_GIB: usize = 8;
const MAX_PAYLOAD_CACHE_GIB: usize = 12;
const GPU_PAYLOAD_CACHE_GIB_ENV: &str = "POLARIS_GPU_PAYLOAD_CACHE_GIB";
// The experimental resident cache regressed the real sequential FullDepth43
// workload and could OOM at 6 GiB. Production therefore keeps the original
// per-request upload path unless an operator explicitly opts in with 1-7 GiB.
const DEFAULT_GPU_PAYLOAD_CACHE_GIB: usize = 0;
const MIN_GPU_PAYLOAD_CACHE_GIB: usize = 1;
const MAX_GPU_PAYLOAD_CACHE_GIB: usize = 7;
const GPU_VRAM_HARD_LIMIT_GIB: usize = 8;
const OFFICIAL_ROUTED_EXPERT_COUNT: usize = 6;
const REUSABLE_GPU_SLOT_MAX_LOGICAL_BYTES: u64 = 128 * 1024 * 1024;
const WRITEBACK_DIAGNOSTIC_DIR_ENV: &str = "POLARIS_FULLDEPTH43_WRITEBACK_DIAGNOSTIC_DIR";
const FULLDEPTH43_CATALOG_FILE: &str = "D:/models/Polaris-S14/fulldepth43_native_top6_catalog.json";
const FULLDEPTH43_CATALOG_SHA256: &str =
    "ca619984d4a46ad1a3701d2b4035766ea40c3a3dbedd3a474ce1df7aad4d0049";
const L42_WQ_A_PROJECTION: &str = "layers.42.attn.wq_a";
const L42_WQ_A_WEIGHT_SHA256: &str =
    "1efcea39938dfadc143c41813bc32327a9bb5369b2b612feac76d9dfb8001ce7";
const L42_WQ_A_SCALE_SHA256: &str =
    "dfb4085717aa527f8affa5a1640c5f806867c5ba6e0301d170f387be8b6660cf";
const L42_WQ_A_INPUT_SHA256: &str =
    "47156935b19ca5483f0e92d2284eaa6a9417686978dc4b41ca893ee162f37577";
const L42_WQ_A_OUTPUT_SHA256: &str =
    "76469fd163f5db49de956eff9b29087afa4caa97d566be80bab9d9119facb0b8";
const FP8_PROJECTION_ARENA_MAX_BYTES: u64 = 64 * 1024 * 1024;
const FULLDEPTH_FP8_PAYLOAD_CACHE_BYTES: usize = 256 * 1024 * 1024;
const FULLDEPTH_FP8_GPU_SLOT_MAX_ENTRIES: usize = 43 * 6;
const FULLDEPTH_FP8_GPU_SLOT_MAX_RESIDENT_BYTES: u64 = 5 * 1024 * 1024 * 1024;
const FULLDEPTH_FP8_MAX_POSITION: u32 = 1_048_575;
const L42_WO_A_CAPTURE_MANIFEST_SHA256: &str =
    "f91fc9fd40a95ee04984376b7a80544db8decc88bc5f139b9a6fd382fa4bb43d";
const L42_WO_A_WEIGHT_SHA256: &str =
    "4986ee3f7fb2758199940aa3007d876d32d0fcb4568149a6f57c8d816f758df5";
const L42_WO_A_SCALE_SHA256: &str =
    "d689681d2d32663037fd4799d1c49d2c4439138d1dd97768dd2030a6d42b7ac9";
const L42_WO_A_INPUT_SHA256: &str =
    "eee925360c8709263a0cdfa3986c2d3ee91a38c4e4589a7220064b489ad40060";
const L42_WO_A_OUTPUT_SHA256: &str =
    "2be0aa3b4b67aae58f62a77d2a255d6240b5baf3d71f37c9084fd890741d2eb9";
const L42_WO_A_REQUANTIZED_SHA256: &str =
    "94b3f7fd24ee36b8553ed513d1986ef49162c053bd6dbf62f98b9579e20ea3f0";
const L42_WO_B_OUTPUT_SHA256: &str =
    "84ce63ca9233b07bea99741f9982accac17bc65025b0098b7017acd7dab6db10";

#[derive(Debug, Clone, Copy)]
struct FrozenFp8Projection {
    name: &'static str,
    file_stem: &'static str,
    n: u32,
    k: u32,
    weight_sha256: &'static str,
    scale_sha256: &'static str,
    input_sha256: &'static str,
    output_sha256: &'static str,
}

const L42_STANDARD_FP8_PROJECTIONS: [FrozenFp8Projection; 5] = [
    FrozenFp8Projection {
        name: "layers.42.attn.wq_a",
        file_stem: "wq_a",
        n: 1024,
        k: 4096,
        weight_sha256: L42_WQ_A_WEIGHT_SHA256,
        scale_sha256: L42_WQ_A_SCALE_SHA256,
        input_sha256: L42_WQ_A_INPUT_SHA256,
        output_sha256: L42_WQ_A_OUTPUT_SHA256,
    },
    FrozenFp8Projection {
        name: "layers.42.attn.wkv",
        file_stem: "wkv",
        n: 512,
        k: 4096,
        weight_sha256: "77dd49fd2396568513c3d397b7dfc65d54c9dd3fcd2223bfef2a8bdad5f652ed",
        scale_sha256: "9fc73bfdab7bf74ecb69af224adcefca194ce379842402e334a7547653a66abe",
        input_sha256: L42_WQ_A_INPUT_SHA256,
        output_sha256: "3cc7f8f4264c6448dd32f9044c0d001107f06d57209a91a80fa56bdda59dd541",
    },
    FrozenFp8Projection {
        name: "layers.42.attn.wq_b",
        file_stem: "wq_b",
        n: 32768,
        k: 1024,
        weight_sha256: "533f57bca168206b55ae28d8e852ca1fb270a0978c57ae5fdbf278d45b85f45c",
        scale_sha256: "6192668ac70e241e49ff8a04bb74f32d337575a4c259aa3eaa9e7dd5dcf1c15f",
        input_sha256: "4ceb243521589b40b930c63b03da362163dfdc7fe12c0b76397100ec4b4c58e1",
        output_sha256: "284391a5a45d6a5367060ecd444a21770e69fa7949455bea6823317f4fb43c04",
    },
    FrozenFp8Projection {
        name: "layers.42.attn.indexer.wq_b",
        file_stem: "indexer-wq_b",
        n: 8192,
        k: 1024,
        weight_sha256: "b1da5eb69957925039b13a3b22d4132a7441cffdf19a38c491a60f645cdd83f3",
        scale_sha256: "98202fdf7f65cdc616c6f4ecfbcd8e194f5890b21b0b351320309706d9f952e9",
        input_sha256: "4ceb243521589b40b930c63b03da362163dfdc7fe12c0b76397100ec4b4c58e1",
        output_sha256: "d9adda7639665267be4fac36e2a74755bb5d730a4a2a8734695198fc4f331501",
    },
    FrozenFp8Projection {
        name: "layers.42.attn.wo_b",
        file_stem: "wo_b",
        n: 4096,
        k: 8192,
        weight_sha256: "07237a368057a84e20b13783b4e5a0b70d39d7a26b183924f34e87395465d112",
        scale_sha256: "f13bf9653a967f3e3a9d3e24a5b19c2fd713bcda573c01556befe28e1550dfee",
        input_sha256: "94b3f7fd24ee36b8553ed513d1986ef49162c053bd6dbf62f98b9579e20ea3f0",
        output_sha256: "84ce63ca9233b07bea99741f9982accac17bc65025b0098b7017acd7dab6db10",
    },
];

const L42_WO_A_GROUPED_PROJECTION: FrozenFp8Projection = FrozenFp8Projection {
    name: "layers.42.attn.wo_a",
    file_stem: "wo_a_grouped",
    n: 8192,
    k: 4096,
    weight_sha256: L42_WO_A_WEIGHT_SHA256,
    scale_sha256: L42_WO_A_SCALE_SHA256,
    input_sha256: L42_WO_A_INPUT_SHA256,
    output_sha256: L42_WO_A_OUTPUT_SHA256,
};

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
    staging: Option<GpuBuffer>,
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
            staging: Some(staging),
            device,
            bytes: byte_count,
        })
    }

    unsafe fn cmd_upload(&self, ctx: &VulkanContext, cb: vk::CommandBuffer) {
        let staging = self
            .staging
            .as_ref()
            .expect("uploaded buffer staging was already released");
        copy(ctx, cb, staging, &self.device, self.bytes);
    }

    fn release_staging(&mut self, ctx: &VulkanContext) {
        if let Some(staging) = self.staging.take() {
            staging.destroy(ctx);
        }
    }

    fn destroy(&self, ctx: &VulkanContext) {
        self.device.destroy(ctx);
        if let Some(staging) = &self.staging {
            staging.destroy(ctx);
        }
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GpuTensorIdentity {
    tensor: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GpuMoeIdentity {
    tensors: [GpuTensorIdentity; 6],
}

impl GpuMoeIdentity {
    fn bytes(&self) -> Result<u64> {
        self.tensors.iter().try_fold(0u64, |total, tensor| {
            total
                .checked_add(tensor.bytes)
                .ok_or_else(|| anyhow::anyhow!("GPU payload identity byte overflow"))
        })
    }
}

struct MoePayload {
    expert_id: Option<u32>,
    mix_weight: f32,
    gpu_identity: Option<GpuMoeIdentity>,
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

    fn upload_and_release_staging(&mut self, ctx: &VulkanContext) -> Result<()> {
        unsafe {
            let pool = make_command_pool(ctx)?;
            let cb = allocate_command_buffer(ctx, pool)?;
            ctx.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.cmd_upload(ctx, cb);
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
        for buffer in [
            &mut self.w1,
            &mut self.s1,
            &mut self.w3,
            &mut self.s3,
            &mut self.w2,
            &mut self.s2,
        ] {
            buffer.release_staging(ctx);
        }
        Ok(())
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct GpuPayloadCacheStats {
    requests: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    current_bytes: u64,
    peak_bytes: u64,
    uploaded_bytes: u64,
}

struct GpuPayloadCacheEntry {
    weights: GpuMoeWeights,
    bytes: u64,
    last_touch: u64,
}

struct GpuPayloadCache {
    capacity_bytes: u64,
    current_bytes: u64,
    clock: u64,
    entries: HashMap<GpuMoeIdentity, GpuPayloadCacheEntry>,
    stats: GpuPayloadCacheStats,
}

impl GpuPayloadCache {
    fn new(capacity_bytes: u64) -> Result<Self> {
        let hard_limit = (GPU_VRAM_HARD_LIMIT_GIB as u64)
            .checked_mul(1024 * 1024 * 1024)
            .ok_or_else(|| anyhow::anyhow!("GPU VRAM hard limit overflow"))?;
        if capacity_bytes == 0 || capacity_bytes > hard_limit {
            bail!("GPU payload cache capacity must be within the 8 GiB VRAM hard limit");
        }
        Ok(Self {
            capacity_bytes,
            current_bytes: 0,
            clock: 0,
            entries: HashMap::new(),
            stats: GpuPayloadCacheStats::default(),
        })
    }

    fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn stats(&self) -> GpuPayloadCacheStats {
        self.stats
    }

    fn ensure(&mut self, ctx: &VulkanContext, payload: &MoePayload) -> Result<()> {
        let identity = payload.gpu_identity.as_ref().ok_or_else(|| {
            anyhow::anyhow!("GPU resident payload is missing strict SHA identity")
        })?;
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("GPU payload cache clock overflow"))?;
        self.stats.requests = self
            .stats
            .requests
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("GPU payload cache request counter overflow"))?;
        if let Some(entry) = self.entries.get_mut(identity) {
            entry.last_touch = self.clock;
            self.stats.hits += 1;
            return Ok(());
        }

        let identity_bytes = identity.bytes()?;
        let payload_bytes = [
            payload.w1.len(),
            payload.s1.len(),
            payload.w3.len(),
            payload.s3.len(),
            payload.w2.len(),
            payload.s2.len(),
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            total
                .checked_add(bytes as u64)
                .ok_or_else(|| anyhow::anyhow!("GPU payload byte overflow"))
        })?;
        if identity_bytes != payload_bytes {
            bail!("GPU payload identity byte count drift");
        }
        if payload_bytes > self.capacity_bytes {
            bail!("single GPU payload exceeds configured resident cache capacity");
        }
        while self.current_bytes + payload_bytes > self.capacity_bytes {
            self.evict_oldest(ctx)?;
        }

        let mut weights = GpuMoeWeights::new(ctx, payload)?;
        weights.upload_and_release_staging(ctx)?;
        self.current_bytes += payload_bytes;
        self.stats.misses += 1;
        self.stats.current_bytes = self.current_bytes;
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.current_bytes);
        self.stats.uploaded_bytes = self
            .stats
            .uploaded_bytes
            .checked_add(payload_bytes)
            .ok_or_else(|| anyhow::anyhow!("GPU payload uploaded byte counter overflow"))?;
        self.entries.insert(
            identity.clone(),
            GpuPayloadCacheEntry {
                weights,
                bytes: payload_bytes,
                last_touch: self.clock,
            },
        );
        Ok(())
    }

    fn get(&self, identity: &GpuMoeIdentity) -> Result<&GpuMoeWeights> {
        self.entries
            .get(identity)
            .map(|entry| &entry.weights)
            .ok_or_else(|| anyhow::anyhow!("GPU payload cache identity disappeared after ensure"))
    }

    fn evict_oldest(&mut self, ctx: &VulkanContext) -> Result<()> {
        let key = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_touch)
            .map(|(identity, _)| identity.clone())
            .ok_or_else(|| anyhow::anyhow!("GPU payload cache has no eviction candidate"))?;
        let entry = self
            .entries
            .remove(&key)
            .ok_or_else(|| anyhow::anyhow!("GPU payload cache eviction index drift"))?;
        entry.weights.destroy(ctx);
        self.current_bytes = self
            .current_bytes
            .checked_sub(entry.bytes)
            .ok_or_else(|| anyhow::anyhow!("GPU payload cache byte ledger underflow"))?;
        self.stats.evictions += 1;
        self.stats.current_bytes = self.current_bytes;
        Ok(())
    }

    fn destroy(&mut self, ctx: &VulkanContext) {
        for (_, entry) in self.entries.drain() {
            entry.weights.destroy(ctx);
        }
        self.current_bytes = 0;
        self.stats.current_bytes = 0;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MoeWeightByteLayout {
    w1: u64,
    s1: u64,
    w3: u64,
    s3: u64,
    w2: u64,
    s2: u64,
}

impl MoeWeightByteLayout {
    fn total(self) -> Result<u64> {
        [self.w1, self.s1, self.w3, self.s3, self.w2, self.s2]
            .into_iter()
            .try_fold(0u64, |total, bytes| {
                total
                    .checked_add(bytes)
                    .ok_or_else(|| anyhow::anyhow!("reusable GPU slot byte count overflow"))
            })
    }

    fn from_payload(payload: &MoePayload) -> Self {
        Self {
            w1: payload.w1.len() as u64,
            s1: payload.s1.len() as u64,
            w3: payload.w3.len() as u64,
            s3: payload.s3.len() as u64,
            w2: payload.w2.len() as u64,
            s2: payload.s2.len() as u64,
        }
    }
}

fn official_moe_weight_layouts() -> Result<(MoeWeightByteLayout, MoeWeightByteLayout)> {
    let routed_up = S14MatvecShape::new(2048, 4096)?.validate_mxfp4()?;
    let routed_down = S14MatvecShape::new(4096, 2048)?.validate_mxfp4()?;
    let shared_up = S14MatvecShape::new(2048, 4096)?.validate_fp8()?;
    let shared_down = S14MatvecShape::new(4096, 2048)?.validate_fp8()?;
    Ok((
        MoeWeightByteLayout {
            w1: routed_up.mxfp4_weight_bytes()?,
            s1: routed_up.mxfp4_scale_bytes()?,
            w3: routed_up.mxfp4_weight_bytes()?,
            s3: routed_up.mxfp4_scale_bytes()?,
            w2: routed_down.mxfp4_weight_bytes()?,
            s2: routed_down.mxfp4_scale_bytes()?,
        },
        MoeWeightByteLayout {
            w1: shared_up.fp8_weight_bytes()?,
            s1: shared_up.fp8_scale_bytes()?,
            w3: shared_up.fp8_weight_bytes()?,
            s3: shared_up.fp8_scale_bytes()?,
            w2: shared_down.fp8_weight_bytes()?,
            s2: shared_down.fp8_scale_bytes()?,
        },
    ))
}

fn require_exact_moe_weight_layout(
    payload: &MoePayload,
    expected: MoeWeightByteLayout,
    label: &str,
) -> Result<()> {
    let actual = MoeWeightByteLayout::from_payload(payload);
    if actual != expected {
        bail!("{label} payload shape drift: expected {expected:?}, observed {actual:?}");
    }
    Ok(())
}

/// A persistently mapped staging buffer paired with one fixed-size VRAM slot.
/// Unlike `UploadedBuffer`, its staging allocation is never released between
/// requests and its contents are always overwritten before the next upload.
struct ReusableUploadedBuffer {
    staging: GpuBuffer,
    device: GpuBuffer,
    bytes: u64,
}

impl ReusableUploadedBuffer {
    fn new(ctx: &VulkanContext, bytes: u64) -> Result<Self> {
        if bytes == 0 || bytes > REUSABLE_GPU_SLOT_MAX_LOGICAL_BYTES {
            bail!("reusable GPU upload buffer size is outside the fixed safety bound");
        }
        Ok(Self {
            staging: GpuBuffer::new_staging(ctx, bytes)?,
            device: GpuBuffer::new_vram(
                ctx,
                bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            )?,
            bytes,
        })
    }

    fn rewrite_staging(&self, bytes: &[u8], label: &str) -> Result<()> {
        if bytes.len() as u64 != self.bytes {
            bail!(
                "{label} reusable upload size drift: expected {}, observed {}",
                self.bytes,
                bytes.len()
            );
        }
        unsafe { self.staging.write_at(0, bytes) };
        Ok(())
    }

    unsafe fn cmd_upload(&self, ctx: &VulkanContext, cb: vk::CommandBuffer) {
        copy(ctx, cb, &self.staging, &self.device, self.bytes);
    }

    fn destroy(&self, ctx: &VulkanContext) {
        self.device.destroy(ctx);
        self.staging.destroy(ctx);
    }
}

struct ReusableMoeWeights {
    w1: ReusableUploadedBuffer,
    s1: ReusableUploadedBuffer,
    w3: ReusableUploadedBuffer,
    s3: ReusableUploadedBuffer,
    w2: ReusableUploadedBuffer,
    s2: ReusableUploadedBuffer,
}

impl ReusableMoeWeights {
    fn new(ctx: &VulkanContext, layout: MoeWeightByteLayout) -> Result<Self> {
        Ok(Self {
            w1: ReusableUploadedBuffer::new(ctx, layout.w1)?,
            s1: ReusableUploadedBuffer::new(ctx, layout.s1)?,
            w3: ReusableUploadedBuffer::new(ctx, layout.w3)?,
            s3: ReusableUploadedBuffer::new(ctx, layout.s3)?,
            w2: ReusableUploadedBuffer::new(ctx, layout.w2)?,
            s2: ReusableUploadedBuffer::new(ctx, layout.s2)?,
        })
    }

    fn rewrite_staging(&self, payload: &MoePayload, label: &str) -> Result<()> {
        self.w1
            .rewrite_staging(&payload.w1, &format!("{label}.w1"))?;
        self.s1
            .rewrite_staging(&payload.s1, &format!("{label}.s1"))?;
        self.w3
            .rewrite_staging(&payload.w3, &format!("{label}.w3"))?;
        self.s3
            .rewrite_staging(&payload.s3, &format!("{label}.s3"))?;
        self.w2
            .rewrite_staging(&payload.w2, &format!("{label}.w2"))?;
        self.s2
            .rewrite_staging(&payload.s2, &format!("{label}.s2"))?;
        Ok(())
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

/// Default, non-resident FullDepth43 upload path. The slot owns exactly one
/// layer's six routed experts plus shared expert and workspace. Every request
/// rewrites all 42 mapped tensor staging buffers before copying them to the
/// same fixed VRAM buffers; no payload identity or layer content is retained.
struct ReusableOfficialMoeSlot {
    x: ReusableUploadedBuffer,
    routed: Vec<ReusableMoeWeights>,
    shared: ReusableMoeWeights,
    gate: GpuBuffer,
    up: GpuBuffer,
    hidden: GpuBuffer,
    down: GpuBuffer,
    accumulator: GpuBuffer,
    readback: GpuBuffer,
    logical_device_bytes: u64,
    logical_staging_bytes: u64,
}

impl ReusableOfficialMoeSlot {
    fn logical_byte_counts() -> Result<(u64, u64)> {
        let (routed_layout, shared_layout) = official_moe_weight_layouts()?;
        let x_bytes = 4096 * std::mem::size_of::<f32>() as u64;
        let routed_bytes = routed_layout
            .total()?
            .checked_mul(OFFICIAL_ROUTED_EXPERT_COUNT as u64)
            .ok_or_else(|| anyhow::anyhow!("reusable routed slot byte count overflow"))?;
        let shared_bytes = shared_layout.total()?;
        let staging_bytes = x_bytes
            .checked_add(routed_bytes)
            .and_then(|bytes| bytes.checked_add(shared_bytes))
            .ok_or_else(|| anyhow::anyhow!("reusable staging byte count overflow"))?;
        let intermediate_bytes = 2048 * std::mem::size_of::<f32>() as u64;
        let output_bytes = 4096 * std::mem::size_of::<f32>() as u64;
        let workspace_bytes = intermediate_bytes
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_add(output_bytes.checked_mul(3)?))
            .ok_or_else(|| anyhow::anyhow!("reusable workspace byte count overflow"))?;
        let device_bytes = staging_bytes
            .checked_add(workspace_bytes)
            .ok_or_else(|| anyhow::anyhow!("reusable device byte count overflow"))?;
        Ok((device_bytes, staging_bytes))
    }

    fn new(ctx: &VulkanContext) -> Result<Self> {
        let (routed_layout, shared_layout) = official_moe_weight_layouts()?;
        let (logical_device_bytes, logical_staging_bytes) = Self::logical_byte_counts()?;
        if logical_device_bytes > REUSABLE_GPU_SLOT_MAX_LOGICAL_BYTES {
            bail!(
                "fixed reusable GPU slot requires {logical_device_bytes} bytes, above the {} byte safety bound",
                REUSABLE_GPU_SLOT_MAX_LOGICAL_BYTES
            );
        }
        let x_bytes = 4096 * std::mem::size_of::<f32>() as u64;
        let x = ReusableUploadedBuffer::new(ctx, x_bytes)?;
        let routed = (0..OFFICIAL_ROUTED_EXPERT_COUNT)
            .map(|_| ReusableMoeWeights::new(ctx, routed_layout))
            .collect::<Result<Vec<_>>>()?;
        let shared = ReusableMoeWeights::new(ctx, shared_layout)?;
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
            logical_device_bytes,
            logical_staging_bytes,
        })
    }

    fn rewrite_and_upload(
        &self,
        ctx: &VulkanContext,
        x: &[f32],
        routed: &[MoePayload],
        shared: &MoePayload,
    ) -> Result<()> {
        if x.len() != 4096
            || routed.len() != OFFICIAL_ROUTED_EXPERT_COUNT
            || routed.iter().any(|payload| payload.expert_id.is_none())
            || shared.expert_id.is_some()
        {
            bail!("reusable official MoE slot request contract drift");
        }
        let (routed_layout, shared_layout) = official_moe_weight_layouts()?;
        self.x
            .rewrite_staging(bytemuck::cast_slice(x), "activation")?;
        for (index, (slot, payload)) in self.routed.iter().zip(routed).enumerate() {
            require_exact_moe_weight_layout(payload, routed_layout, &format!("routed[{index}]"))?;
            slot.rewrite_staging(payload, &format!("routed[{index}]"))?;
        }
        require_exact_moe_weight_layout(shared, shared_layout, "shared")?;
        self.shared.rewrite_staging(shared, "shared")?;

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
        for weights in self.routed.iter().rev() {
            weights.destroy(ctx);
        }
        self.x.destroy(ctx);
    }
}

/// Per-request activation/workspace buffers. Expert weights live in the
/// worker-level `GpuPayloadCache` and are not reallocated or reuploaded here.
struct OfficialMoeWorkspace {
    x: UploadedBuffer,
    gate: GpuBuffer,
    up: GpuBuffer,
    hidden: GpuBuffer,
    down: GpuBuffer,
    accumulator: GpuBuffer,
    readback: GpuBuffer,
}

impl OfficialMoeWorkspace {
    fn new(ctx: &VulkanContext, x: &[f32]) -> Result<Self> {
        let x = UploadedBuffer::new(ctx, bytemuck::cast_slice(x))?;
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
            gate,
            up,
            hidden,
            down,
            accumulator,
            readback,
        })
    }

    fn upload_activation(&self, ctx: &VulkanContext) -> Result<()> {
        unsafe {
            let pool = make_command_pool(ctx)?;
            let cb = allocate_command_buffer(ctx, pool)?;
            ctx.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.x.cmd_upload(ctx, cb);
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
    gpu_payload_cache: WritebackGpuPayloadCacheTelemetry,
    reusable_gpu_slot: WritebackReusableGpuSlotTelemetry,
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

#[derive(Debug, Serialize, PartialEq)]
struct WritebackGpuPayloadCacheTelemetry {
    enabled: bool,
    capacity_bytes: u64,
    entries: usize,
    current_bytes: u64,
    peak_bytes: u64,
    request_hits: u64,
    request_misses: u64,
    request_uploaded_bytes: u64,
    total_hits: u64,
    total_misses: u64,
    total_evictions: u64,
    total_uploaded_bytes: u64,
    total_hit_rate: f64,
    strict_sha_identity: bool,
}

#[derive(Debug, Serialize, PartialEq)]
struct WritebackReusableGpuSlotTelemetry {
    enabled: bool,
    logical_device_bytes: u64,
    logical_staging_bytes: u64,
    request_uploads: u64,
    request_uploaded_bytes: u64,
    weight_tensor_slots_reused: usize,
    workspace_reused: bool,
    strict_fixed_shapes: bool,
    resident_cache_isolated: bool,
}

impl WritebackReusableGpuSlotTelemetry {
    fn for_successful_request(
        slot_logical_bytes: Option<(u64, u64)>,
        resident_cache_enabled: bool,
    ) -> Result<Self> {
        match (slot_logical_bytes, resident_cache_enabled) {
            (Some((logical_device_bytes, logical_staging_bytes)), false) => Ok(Self {
                enabled: true,
                logical_device_bytes,
                logical_staging_bytes,
                request_uploads: 1,
                request_uploaded_bytes: logical_staging_bytes,
                weight_tensor_slots_reused: (OFFICIAL_ROUTED_EXPERT_COUNT + 1) * 6,
                workspace_reused: true,
                strict_fixed_shapes: true,
                resident_cache_isolated: true,
            }),
            (None, true) => Ok(Self {
                enabled: false,
                logical_device_bytes: 0,
                logical_staging_bytes: 0,
                request_uploads: 0,
                request_uploaded_bytes: 0,
                weight_tensor_slots_reused: 0,
                workspace_reused: false,
                strict_fixed_shapes: true,
                resident_cache_isolated: true,
            }),
            _ => bail!("GPU resident cache and reusable upload slot telemetry mode drift"),
        }
    }
}

impl WritebackGpuPayloadCacheTelemetry {
    fn disabled() -> Self {
        Self {
            enabled: false,
            capacity_bytes: 0,
            entries: 0,
            current_bytes: 0,
            peak_bytes: 0,
            request_hits: 0,
            request_misses: 0,
            request_uploaded_bytes: 0,
            total_hits: 0,
            total_misses: 0,
            total_evictions: 0,
            total_uploaded_bytes: 0,
            total_hit_rate: 0.0,
            strict_sha_identity: true,
        }
    }

    fn between(
        cache: &GpuPayloadCache,
        before: GpuPayloadCacheStats,
        after: GpuPayloadCacheStats,
    ) -> Result<Self> {
        let total_requests = after
            .hits
            .checked_add(after.misses)
            .ok_or_else(|| anyhow::anyhow!("GPU payload cache request total overflow"))?;
        Ok(Self {
            enabled: true,
            capacity_bytes: cache.capacity_bytes(),
            entries: cache.len(),
            current_bytes: after.current_bytes,
            peak_bytes: after.peak_bytes,
            request_hits: monotonic_delta(after.hits, before.hits, "gpu_hits")?,
            request_misses: monotonic_delta(after.misses, before.misses, "gpu_misses")?,
            request_uploaded_bytes: monotonic_delta(
                after.uploaded_bytes,
                before.uploaded_bytes,
                "gpu_uploaded_bytes",
            )?,
            total_hits: after.hits,
            total_misses: after.misses,
            total_evictions: after.evictions,
            total_uploaded_bytes: after.uploaded_bytes,
            total_hit_rate: if total_requests == 0 {
                0.0
            } else {
                after.hits as f64 / total_requests as f64
            },
            strict_sha_identity: true,
        })
    }
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
    diagnostics: Option<OfficialMoeDiagnostics>,
}

struct ExpertDiagnostics {
    expert_id: Option<u32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    hidden: Vec<f32>,
    down: Vec<f32>,
}

struct OfficialMoeDiagnostics {
    routed: Vec<ExpertDiagnostics>,
    shared: ExpertDiagnostics,
    accumulator: Vec<f32>,
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ProjectionArenaView {
    path: PathBuf,
    offset: u64,
    bytes: u64,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fp8ProjectionSpec {
    name: String,
    n: u32,
    k: u32,
    activation_contract: String,
    output_rounding: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fp8ProjectionRequest {
    protocol: String,
    op: String,
    request_id: String,
    revision: String,
    profile: String,
    layer: u32,
    position: u32,
    arena_epoch: u64,
    input_sha256: String,
    projection: Fp8ProjectionSpec,
    input: ProjectionArenaView,
    output: ProjectionArenaView,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct FullDepthFp8AssetView {
    tensor: String,
    path: PathBuf,
    bytes: u64,
    sha256: String,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct FullDepthFp8ProjectionSpec {
    name: String,
    kernel: String,
    n: u32,
    k: u32,
    groups: Option<u32>,
    n_per_group: Option<u32>,
    activation_contract: String,
    output_rounding: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FullDepthFp8AttentionRequest {
    protocol: String,
    op: String,
    request_id: String,
    revision: String,
    profile: String,
    layer: u32,
    position: u32,
    arena_epoch: u64,
    input_sha256: String,
    projection: FullDepthFp8ProjectionSpec,
    weight: FullDepthFp8AssetView,
    scale: FullDepthFp8AssetView,
    input: ProjectionArenaView,
    output: ProjectionArenaView,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FullDepthFp8AttentionBatchItem {
    projection: FullDepthFp8ProjectionSpec,
    weight: FullDepthFp8AssetView,
    scale: FullDepthFp8AssetView,
    output: ProjectionArenaView,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FullDepthFp8AttentionBatchRequest {
    protocol: String,
    op: String,
    request_id: String,
    revision: String,
    profile: String,
    layer: u32,
    position: u32,
    arena_epoch: u64,
    input_sha256: String,
    input: ProjectionArenaView,
    projections: Vec<FullDepthFp8AttentionBatchItem>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FullDepthFp8AttentionOutputChainRequantization {
    format: String,
    group_size: u32,
    amax_floor: f32,
    max_finite: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FullDepthFp8AttentionOutputChainStage {
    projection: FullDepthFp8ProjectionSpec,
    weight: FullDepthFp8AssetView,
    scale: FullDepthFp8AssetView,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FullDepthFp8AttentionOutputChainRequest {
    protocol: String,
    op: String,
    request_id: String,
    revision: String,
    profile: String,
    layer: u32,
    position: u32,
    arena_epoch: u64,
    input_sha256: String,
    input: ProjectionArenaView,
    projections: Vec<FullDepthFp8AttentionOutputChainStage>,
    requantization: FullDepthFp8AttentionOutputChainRequantization,
    output: ProjectionArenaView,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum FullDepthFp8AttentionWorkerRequest {
    Single(FullDepthFp8AttentionRequest),
    SharedInputBatch(FullDepthFp8AttentionBatchRequest),
    OutputChain(FullDepthFp8AttentionOutputChainRequest),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FullDepthFp8Kernel {
    Standard(S14MatvecShape),
    GroupedWoA(S14GroupedMatvecShape),
}

impl FullDepthFp8Kernel {
    fn input_elements(self) -> Result<u64> {
        match self {
            Self::Standard(shape) => Ok(shape.k as u64),
            Self::GroupedWoA(shape) => Ok(shape.groups as u64 * shape.k as u64),
        }
    }

    fn output_elements(self) -> Result<u64> {
        match self {
            Self::Standard(shape) => Ok(shape.n as u64),
            Self::GroupedWoA(shape) => Ok(shape.flat_n()? as u64),
        }
    }

    fn weight_bytes(self) -> Result<u64> {
        match self {
            Self::Standard(shape) => shape.fp8_weight_bytes(),
            Self::GroupedWoA(shape) => shape.fp8_weight_bytes(),
        }
    }

    fn scale_bytes(self) -> Result<u64> {
        match self {
            Self::Standard(shape) => shape.fp8_scale_bytes(),
            Self::GroupedWoA(shape) => shape.fp8_scale_bytes(),
        }
    }

    fn input_shape(self) -> Vec<usize> {
        match self {
            Self::Standard(shape) => vec![1, 1, shape.k as usize],
            Self::GroupedWoA(shape) => {
                vec![1, 1, shape.groups as usize, shape.k as usize]
            }
        }
    }

    fn output_shape(self) -> Result<Vec<usize>> {
        Ok(vec![1, 1, self.output_elements()? as usize])
    }

    fn weight_shape(self) -> Result<Vec<usize>> {
        match self {
            Self::Standard(shape) => Ok(vec![shape.n as usize, shape.k as usize]),
            Self::GroupedWoA(shape) => Ok(vec![shape.flat_n()? as usize, shape.k as usize]),
        }
    }

    fn scale_shape(self) -> Result<Vec<usize>> {
        match self {
            Self::Standard(shape) => Ok(vec![
                shape.n.div_ceil(128) as usize,
                shape.k.div_ceil(128) as usize,
            ]),
            Self::GroupedWoA(shape) => Ok(vec![
                shape.flat_n()?.div_ceil(128) as usize,
                shape.k.div_ceil(128) as usize,
            ]),
        }
    }

    fn numeric_mode(self) -> &'static str {
        match self {
            Self::Standard(_) => "packed_fp8_e4m3_ue8m0_bf16_output",
            Self::GroupedWoA(_) => "grouped_packed_fp8_e4m3_ue8m0_bf16_input_output",
        }
    }
}

fn fulldepth_fp8_slot_key(
    kernel: FullDepthFp8Kernel,
    request: &FullDepthFp8AttentionRequest,
) -> String {
    format!(
        "{}:{}:{}:{}:{}",
        kernel.numeric_mode(),
        request.weight.tensor,
        request.weight.sha256,
        request.scale.tensor,
        request.scale.sha256
    )
}

fn reserve_fulldepth_fp8_gpu_slot(
    current_entries: usize,
    current_resident_bytes: u64,
    request: &FullDepthFp8AttentionRequest,
) -> Result<u64> {
    let next_entries = current_entries
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("FullDepth43 GPU slot entry overflow"))?;
    if next_entries > FULLDEPTH_FP8_GPU_SLOT_MAX_ENTRIES {
        bail!(
            "FullDepth43 GPU slot entry budget exceeded: {} > {}",
            next_entries,
            FULLDEPTH_FP8_GPU_SLOT_MAX_ENTRIES
        );
    }
    let next_resident_bytes = current_resident_bytes
        .checked_add(request.weight.bytes)
        .and_then(|value| value.checked_add(request.scale.bytes))
        .ok_or_else(|| anyhow::anyhow!("FullDepth43 GPU slot byte overflow"))?;
    if next_resident_bytes > FULLDEPTH_FP8_GPU_SLOT_MAX_RESIDENT_BYTES {
        bail!(
            "FullDepth43 GPU slot resident budget exceeded: {} > {}",
            next_resident_bytes,
            FULLDEPTH_FP8_GPU_SLOT_MAX_RESIDENT_BYTES
        );
    }
    Ok(next_resident_bytes)
}

#[derive(Serialize)]
struct FullDepthFp8AttentionResponse {
    protocol: &'static str,
    request_id: String,
    ok: bool,
    revision: &'static str,
    profile: &'static str,
    layer: u32,
    position: u32,
    arena_epoch: u64,
    projection: FullDepthFp8ProjectionSpec,
    input: ProjectionArenaView,
    output_written: ProjectionArenaView,
    input_sha256: String,
    output_sha256: String,
    weight_sha256: String,
    scale_sha256: String,
    catalog_sha256: &'static str,
    payload_hash_verified: bool,
    gpu_slot_cache_hit: bool,
    gpu_slot_cache_entries: usize,
    gpu_slot_resident_bytes: u64,
    payload_uploaded_bytes: u64,
    activation_uploaded_bytes: u64,
    numeric_mode: &'static str,
    output_rounding: &'static str,
}

#[derive(Serialize)]
struct FullDepthFp8AttentionBatchItemResponse {
    projection: FullDepthFp8ProjectionSpec,
    output_written: ProjectionArenaView,
    output_sha256: String,
    weight_sha256: String,
    scale_sha256: String,
    payload_hash_verified: bool,
    gpu_slot_cache_hit: bool,
    gpu_slot_resident_bytes: u64,
    payload_uploaded_bytes: u64,
    numeric_mode: &'static str,
    output_rounding: &'static str,
}

#[derive(Serialize)]
struct FullDepthFp8AttentionBatchResponse {
    protocol: &'static str,
    request_id: String,
    ok: bool,
    revision: &'static str,
    profile: &'static str,
    layer: u32,
    position: u32,
    arena_epoch: u64,
    input: ProjectionArenaView,
    input_sha256: String,
    outputs: Vec<FullDepthFp8AttentionBatchItemResponse>,
    catalog_sha256: &'static str,
    gpu_slot_cache_entries: usize,
    activation_uploaded_bytes: u64,
}

#[derive(Serialize)]
struct FullDepthFp8AttentionOutputChainSlotResponse {
    projection: FullDepthFp8ProjectionSpec,
    weight_sha256: String,
    scale_sha256: String,
    payload_hash_verified: bool,
    gpu_slot_cache_hit: bool,
    gpu_slot_cache_entries: usize,
    gpu_slot_resident_bytes: u64,
    payload_uploaded_bytes: u64,
    activation_uploaded_bytes: u64,
    numeric_mode: &'static str,
    output_rounding: &'static str,
}

#[derive(Serialize)]
struct FullDepthFp8AttentionOutputChainResponse {
    protocol: &'static str,
    request_id: String,
    ok: bool,
    revision: &'static str,
    profile: &'static str,
    layer: u32,
    position: u32,
    arena_epoch: u64,
    input: ProjectionArenaView,
    output_written: ProjectionArenaView,
    input_sha256: String,
    wo_a_output_sha256: String,
    requantized_activation_sha256: String,
    output_sha256: String,
    requantization: FullDepthFp8AttentionOutputChainRequantization,
    slots: Vec<FullDepthFp8AttentionOutputChainSlotResponse>,
    catalog_sha256: &'static str,
    gpu_slot_cache_entries: usize,
    numeric_mode: &'static str,
    output_rounding: &'static str,
}

#[derive(Serialize)]
#[serde(untagged)]
enum FullDepthFp8AttentionWorkerResponse {
    Single(FullDepthFp8AttentionResponse),
    SharedInputBatch(FullDepthFp8AttentionBatchResponse),
    OutputChain(FullDepthFp8AttentionOutputChainResponse),
}

#[derive(Serialize)]
struct Fp8ProjectionResponse {
    protocol: &'static str,
    request_id: String,
    ok: bool,
    revision: &'static str,
    profile: &'static str,
    layer: u32,
    position: u32,
    arena_epoch: u64,
    projection: &'static str,
    input: ProjectionArenaView,
    output_written: ProjectionArenaView,
    input_sha256: &'static str,
    output_sha256: &'static str,
    weight_sha256: &'static str,
    scale_sha256: &'static str,
    catalog_sha256: &'static str,
    weight_resident: bool,
    static_uploaded_bytes: u64,
    request_uploaded_bytes: u64,
    numeric_mode: &'static str,
    output_rounding: &'static str,
}

#[derive(Serialize)]
struct Fp8ProjectionError<'a> {
    protocol: &'static str,
    request_id: &'a str,
    ok: bool,
    error: String,
    poisoned: bool,
}

fn validate_l42_wq_a_request(request: &Fp8ProjectionRequest, expected_epoch: u64) -> Result<()> {
    if request.protocol != FP8_PROJECTION_PROTOCOL
        || request.op != "execute_fp8_projection"
        || request.revision != REVISION
        || request.profile != "fulldepth43_native_top6"
        || request.layer != 42
        || request.position != 0
        || request.arena_epoch != expected_epoch
        || request.input_sha256 != L42_WQ_A_INPUT_SHA256
        || request.projection.name != L42_WQ_A_PROJECTION
        || request.projection.n != 1024
        || request.projection.k != 4096
        || request.projection.activation_contract != "cpu_e4m3fn_quant_dequant_f32"
        || request.projection.output_rounding != "bf16_rne_then_f32_le"
        || request.request_id.is_empty()
        || request.request_id.len() > 128
        || !request
            .request_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    {
        bail!("L42 wq_a FP8 projection request identity drift");
    }
    if request.input.dtype != "f32_le"
        || request.input.shape != [1, 1, 4096]
        || request.input.bytes != 4096 * 4
        || request.output.dtype != "f32_le_bf16_rounded"
        || request.output.shape != [1, 1, 1024]
        || request.output.bytes != 1024 * 4
    {
        bail!("L42 wq_a FP8 projection arena tensor contract drift");
    }
    Ok(())
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn expected_fulldepth_fp8_kernel(
    layer: u32,
    projection: &FullDepthFp8ProjectionSpec,
) -> Result<FullDepthFp8Kernel> {
    let prefix = format!("layers.{layer}.attn.");
    let suffix = projection
        .name
        .strip_prefix(&prefix)
        .ok_or_else(|| anyhow::anyhow!("FullDepth43 projection layer/name mismatch"))?;
    let expected = match suffix {
        "wq_a" => ("standard", 1024, 4096),
        "wkv" => ("standard", 512, 4096),
        "wq_b" => ("standard", 32768, 1024),
        "indexer.wq_b" => ("standard", 8192, 1024),
        "wo_b" => ("standard", 4096, 8192),
        "wo_a" => ("grouped_wo_a", 8192, 4096),
        _ => bail!("FullDepth43 projection is not an approved packed-FP8 attention tensor"),
    };
    if projection.kernel != expected.0
        || projection.n != expected.1
        || projection.k != expected.2
        || projection.output_rounding != "bf16_rne_then_f32_le"
    {
        bail!("FullDepth43 projection kernel/shape/rounding drift");
    }
    match suffix {
        "wo_a" => {
            if projection.groups != Some(8)
                || projection.n_per_group != Some(1024)
                || projection.activation_contract != "bf16_carrying_f32_per_group"
            {
                bail!("FullDepth43 wo_a grouped activation contract drift");
            }
            Ok(FullDepthFp8Kernel::GroupedWoA(
                S14GroupedMatvecShape::new(8, 1024, 4096)?.validate_fp8_bf16_weight()?,
            ))
        }
        _ => {
            if projection.groups.is_some()
                || projection.n_per_group.is_some()
                || projection.activation_contract != "cpu_e4m3fn_quant_dequant_f32"
            {
                bail!("FullDepth43 standard FP8 activation contract drift");
            }
            Ok(FullDepthFp8Kernel::Standard(
                S14MatvecShape::new(expected.1, expected.2)?.validate_fp8()?,
            ))
        }
    }
}

fn validate_fulldepth_fp8_attention_request(
    request: &FullDepthFp8AttentionRequest,
    expected_epoch: u64,
) -> Result<FullDepthFp8Kernel> {
    if request.protocol != FULLDEPTH_FP8_ATTENTION_PROTOCOL
        || request.op != "execute_fp8_attention"
        || request.revision != REVISION
        || request.profile != "fulldepth43_native_top6"
        || request.layer > 42
        || request.position > FULLDEPTH_FP8_MAX_POSITION
        || request.arena_epoch != expected_epoch
        || request.request_id.is_empty()
        || request.request_id.len() > 128
        || !request
            .request_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
        || !is_lower_sha256(&request.input_sha256)
    {
        bail!("FullDepth43 packed-FP8 attention request identity drift");
    }
    let kernel = expected_fulldepth_fp8_kernel(request.layer, &request.projection)?;
    let weight_tensor = format!("{}.weight", request.projection.name);
    let scale_tensor = format!("{}.scale", request.projection.name);
    if request.weight.tensor != weight_tensor
        || request.scale.tensor != scale_tensor
        || request.weight.dtype != "F8_E4M3"
        || request.scale.dtype != "F8_E8M0"
        || request.weight.bytes != kernel.weight_bytes()?
        || request.scale.bytes != kernel.scale_bytes()?
        || request.weight.shape != kernel.weight_shape()?
        || request.scale.shape != kernel.scale_shape()?
        || !is_lower_sha256(&request.weight.sha256)
        || !is_lower_sha256(&request.scale.sha256)
        || request
            .weight
            .path
            .extension()
            .and_then(|value| value.to_str())
            != Some("bin")
        || request
            .scale
            .path
            .extension()
            .and_then(|value| value.to_str())
            != Some("bin")
    {
        bail!("FullDepth43 packed-FP8 asset contract drift");
    }
    let input_bytes = kernel
        .input_elements()?
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("FullDepth43 FP8 input byte overflow"))?;
    let output_bytes = kernel
        .output_elements()?
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("FullDepth43 FP8 output byte overflow"))?;
    if request.input.dtype != "f32_le"
        || request.input.shape != kernel.input_shape()
        || request.input.bytes != input_bytes
        || request.output.dtype != "f32_le_bf16_rounded"
        || request.output.shape != kernel.output_shape()?
        || request.output.bytes != output_bytes
    {
        bail!("FullDepth43 packed-FP8 arena tensor contract drift");
    }
    Ok(kernel)
}

fn validate_fulldepth_fp8_attention_batch_request(
    request: &FullDepthFp8AttentionBatchRequest,
    expected_epoch: u64,
) -> Result<[(FullDepthFp8AttentionRequest, FullDepthFp8Kernel); 2]> {
    if request.protocol != FULLDEPTH_FP8_ATTENTION_PROTOCOL
        || request.op != "execute_fp8_attention_shared_batch"
        || request.revision != REVISION
        || request.profile != "fulldepth43_native_top6"
        || request.layer > 42
        || request.position > FULLDEPTH_FP8_MAX_POSITION
        || request.arena_epoch != expected_epoch
        || request.request_id.is_empty()
        || request.request_id.len() > 128
        || !request
            .request_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
        || !is_lower_sha256(&request.input_sha256)
        || request.projections.len() != 2
    {
        bail!("FullDepth43 shared-input FP8 batch request identity drift");
    }

    let expected_prefix = format!("layers.{}.attn.", request.layer);
    let first_suffix = request.projections[0]
        .projection
        .name
        .strip_prefix(&expected_prefix)
        .ok_or_else(|| anyhow::anyhow!("FullDepth43 batch first projection layer drift"))?;
    let second_suffix = request.projections[1]
        .projection
        .name
        .strip_prefix(&expected_prefix)
        .ok_or_else(|| anyhow::anyhow!("FullDepth43 batch second projection layer drift"))?;
    if !matches!(
        (first_suffix, second_suffix),
        ("wq_a", "wkv") | ("wq_b", "indexer.wq_b")
    ) {
        bail!("FullDepth43 shared-input FP8 batch projection order is not approved");
    }

    let to_single = |item: &FullDepthFp8AttentionBatchItem| FullDepthFp8AttentionRequest {
        protocol: request.protocol.clone(),
        op: "execute_fp8_attention".to_string(),
        request_id: request.request_id.clone(),
        revision: request.revision.clone(),
        profile: request.profile.clone(),
        layer: request.layer,
        position: request.position,
        arena_epoch: request.arena_epoch,
        input_sha256: request.input_sha256.clone(),
        projection: item.projection.clone(),
        weight: item.weight.clone(),
        scale: item.scale.clone(),
        input: request.input.clone(),
        output: item.output.clone(),
    };
    let first = to_single(&request.projections[0]);
    let second = to_single(&request.projections[1]);
    let first_kernel = validate_fulldepth_fp8_attention_request(&first, expected_epoch)?;
    let second_kernel = validate_fulldepth_fp8_attention_request(&second, expected_epoch)?;
    if !matches!(first_kernel, FullDepthFp8Kernel::Standard(_))
        || !matches!(second_kernel, FullDepthFp8Kernel::Standard(_))
        || first_kernel.input_shape() != second_kernel.input_shape()
    {
        bail!("FullDepth43 shared-input FP8 batch kernel/input contract drift");
    }
    Ok([(first, first_kernel), (second, second_kernel)])
}

fn validate_fulldepth_fp8_attention_output_chain_request(
    request: &FullDepthFp8AttentionOutputChainRequest,
    expected_epoch: u64,
) -> Result<[(FullDepthFp8AttentionRequest, FullDepthFp8Kernel); 2]> {
    if request.protocol != FULLDEPTH_FP8_ATTENTION_PROTOCOL
        || request.op != "execute_fp8_attention_output_chain"
        || request.revision != REVISION
        || request.profile != "fulldepth43_native_top6"
        || request.layer > 42
        || request.position > FULLDEPTH_FP8_MAX_POSITION
        || request.arena_epoch != expected_epoch
        || request.request_id.is_empty()
        || request.request_id.len() > 128
        || !request
            .request_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
        || !is_lower_sha256(&request.input_sha256)
        || request.projections.len() != 2
    {
        bail!("FullDepth43 FP8 output-chain request identity drift");
    }
    if request.requantization.format != "e4m3fn_group128_quantize_dequantize_f32"
        || request.requantization.group_size != 128
        || request.requantization.amax_floor != 1.0e-4
        || request.requantization.max_finite != 448.0
    {
        bail!("FullDepth43 FP8 output-chain requantization contract drift");
    }

    let prefix = format!("layers.{}.attn.", request.layer);
    let suffixes: Vec<&str> =
        request
            .projections
            .iter()
            .map(|stage| {
                stage.projection.name.strip_prefix(&prefix).ok_or_else(|| {
                    anyhow::anyhow!("FullDepth43 output-chain projection layer drift")
                })
            })
            .collect::<Result<_>>()?;
    if suffixes != ["wo_a", "wo_b"] {
        bail!("FullDepth43 FP8 output-chain projection order must be wo_a then wo_b");
    }

    let intermediate = ProjectionArenaView {
        path: request.input.path.clone(),
        offset: 0,
        bytes: 8192 * 4,
        dtype: "f32_le_bf16_rounded".to_string(),
        shape: vec![1, 1, 8192],
    };
    let requantized = ProjectionArenaView {
        path: request.input.path.clone(),
        offset: 0,
        bytes: 8192 * 4,
        dtype: "f32_le".to_string(),
        shape: vec![1, 1, 8192],
    };
    let make_request = |stage: &FullDepthFp8AttentionOutputChainStage,
                        input: ProjectionArenaView,
                        output: ProjectionArenaView| {
        FullDepthFp8AttentionRequest {
            protocol: request.protocol.clone(),
            op: "execute_fp8_attention".to_string(),
            request_id: request.request_id.clone(),
            revision: request.revision.clone(),
            profile: request.profile.clone(),
            layer: request.layer,
            position: request.position,
            arena_epoch: request.arena_epoch,
            input_sha256: request.input_sha256.clone(),
            projection: stage.projection.clone(),
            weight: stage.weight.clone(),
            scale: stage.scale.clone(),
            input,
            output,
        }
    };
    let first = make_request(&request.projections[0], request.input.clone(), intermediate);
    let second = make_request(&request.projections[1], requantized, request.output.clone());
    let first_kernel = validate_fulldepth_fp8_attention_request(&first, expected_epoch)?;
    let second_kernel = validate_fulldepth_fp8_attention_request(&second, expected_epoch)?;
    if !matches!(first_kernel, FullDepthFp8Kernel::GroupedWoA(_))
        || !matches!(second_kernel, FullDepthFp8Kernel::Standard(_))
    {
        bail!("FullDepth43 FP8 output-chain kernel contract drift");
    }
    Ok([(first, first_kernel), (second, second_kernel)])
}

fn validate_fulldepth_fp8_attention_catalog(
    document: &serde_json::Value,
    request: &FullDepthFp8AttentionRequest,
    kernel: FullDepthFp8Kernel,
) -> Result<()> {
    if document.get("format").and_then(serde_json::Value::as_str)
        != Some("polaris-fulldepth43-native-top6-catalog-v1")
        || document.get("repo").and_then(serde_json::Value::as_str)
            != Some("deepseek-ai/DeepSeek-V4-Flash-0731")
        || document.get("revision").and_then(serde_json::Value::as_str) != Some(REVISION)
        || document
            .get("download_authorized")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
    {
        bail!("FullDepth43 catalog identity drift");
    }
    let entries = document
        .pointer(&format!("/layers/{}/non_expert", request.layer))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("FullDepth43 catalog lacks requested layer"))?;
    let require = |asset: &FullDepthFp8AssetView| -> Result<()> {
        let matches: Vec<&serde_json::Value> = entries
            .iter()
            .filter(|entry| {
                entry.get("tensor").and_then(serde_json::Value::as_str)
                    == Some(asset.tensor.as_str())
            })
            .collect();
        if matches.len() != 1 {
            bail!(
                "FullDepth43 catalog must contain exactly one {}",
                asset.tensor
            );
        }
        let entry = matches[0];
        let shape: Option<Vec<usize>> = entry
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .map(|values| values.iter().map(serde_json::Value::as_u64).collect())
            .flatten()
            .map(|values: Vec<u64>| values.into_iter().map(|value| value as usize).collect());
        if entry.get("kind").and_then(serde_json::Value::as_str) != Some("non_expert")
            || entry.get("layer").and_then(serde_json::Value::as_u64) != Some(request.layer as u64)
            || entry.get("dtype").and_then(serde_json::Value::as_str) != Some(asset.dtype.as_str())
            || entry.get("bytes").and_then(serde_json::Value::as_u64) != Some(asset.bytes)
            || shape.as_deref() != Some(asset.shape.as_slice())
        {
            bail!("FullDepth43 catalog asset byte/shape/dtype drift");
        }
        Ok(())
    };
    require(&request.weight)?;
    require(&request.scale)?;
    if request.weight.bytes != kernel.weight_bytes()?
        || request.scale.bytes != kernel.scale_bytes()?
    {
        bail!("FullDepth43 catalog kernel byte contract drift");
    }
    Ok(())
}

fn validate_fulldepth_fp8_cache_proofs(
    document: &serde_json::Value,
    request: &FullDepthFp8AttentionRequest,
    cache_root: &Path,
) -> Result<()> {
    let entries = document
        .pointer(&format!("/layers/{}/non_expert", request.layer))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("FullDepth43 catalog lacks requested layer"))?;
    let validate = |asset: &FullDepthFp8AssetView| -> Result<()> {
        let entry = entries
            .iter()
            .find(|entry| {
                entry.get("tensor").and_then(serde_json::Value::as_str)
                    == Some(asset.tensor.as_str())
            })
            .ok_or_else(|| anyhow::anyhow!("FullDepth43 catalog lacks {}", asset.tensor))?;
        let payload = asset.path.canonicalize()?;
        if payload.parent() != Some(cache_root)
            || payload.extension().and_then(|value| value.to_str()) != Some("bin")
        {
            bail!("FullDepth43 payload is outside canonical range_cache");
        }
        let cache_key = payload
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| anyhow::anyhow!("FullDepth43 payload cache key is missing"))?;
        if !is_lower_sha256(cache_key) {
            bail!("FullDepth43 payload cache key is not lowercase SHA-256");
        }
        let metadata_path = cache_root.join(format!("{cache_key}.json"));
        let metadata: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&metadata_path)
                .with_context(|| format!("read {}", metadata_path.display()))?,
        )?;
        let expected_identity = serde_json::json!({
            "repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": REVISION,
            "source_file": entry.get("file").and_then(serde_json::Value::as_str),
            "source_file_bytes": entry.get("file_bytes").and_then(serde_json::Value::as_u64),
            "start": entry.get("start").and_then(serde_json::Value::as_u64),
            "end": entry.get("end").and_then(serde_json::Value::as_u64),
            "header_tensor_table_sha256": entry
                .get("header_tensor_table_sha256")
                .and_then(serde_json::Value::as_str),
        });
        if metadata.get("format").and_then(serde_json::Value::as_str)
            != Some("polaris-s14-range-cache-entry-v1")
            || metadata
                .get("cache_key")
                .and_then(serde_json::Value::as_str)
                != Some(cache_key)
            || metadata.get("identity") != Some(&expected_identity)
            || metadata.get("bytes").and_then(serde_json::Value::as_u64) != Some(asset.bytes)
            || metadata
                .get("observed_sha256")
                .and_then(serde_json::Value::as_str)
                != Some(asset.sha256.as_str())
        {
            bail!("FullDepth43 payload cache proof/catalog identity drift");
        }
        let authoritative = metadata
            .get("authoritative")
            .and_then(serde_json::Value::as_bool)
            == Some(true);
        if authoritative {
            if metadata
                .get("hash_authority")
                .and_then(serde_json::Value::as_str)
                != Some("official_lock")
                || metadata
                    .get("expected_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(asset.sha256.as_str())
            {
                bail!("FullDepth43 authoritative payload proof is inconsistent");
            }
        } else if metadata
            .get("hash_authority")
            .and_then(serde_json::Value::as_str)
            != Some("tofu")
            || !metadata
                .get("expected_sha256")
                .is_some_and(serde_json::Value::is_null)
        {
            bail!("FullDepth43 TOFU payload proof attempts to masquerade as authoritative");
        }
        Ok(())
    };
    validate(&request.weight)?;
    validate(&request.scale)?;
    Ok(())
}

fn validate_l42_projection_catalog(
    document: &serde_json::Value,
    projection: FrozenFp8Projection,
) -> Result<()> {
    if document.get("format").and_then(serde_json::Value::as_str)
        != Some("polaris-fulldepth43-native-top6-catalog-v1")
        || document.get("repo").and_then(serde_json::Value::as_str)
            != Some("deepseek-ai/DeepSeek-V4-Flash-0731")
        || document.get("revision").and_then(serde_json::Value::as_str) != Some(REVISION)
        || document
            .get("download_authorized")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
    {
        bail!("FullDepth43 catalog identity drift");
    }
    let entries = document
        .pointer("/layers/42/non_expert")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("FullDepth43 catalog lacks L42 non-expert entries"))?;
    let require = |tensor: &str, dtype: &str, shape: &[u64], bytes: u64| -> Result<()> {
        let matches: Vec<&serde_json::Value> = entries
            .iter()
            .filter(|entry| entry.get("tensor").and_then(serde_json::Value::as_str) == Some(tensor))
            .collect();
        if matches.len() != 1 {
            bail!("FullDepth43 catalog must contain exactly one {tensor}");
        }
        let entry = matches[0];
        let observed_shape = entry
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("{tensor} catalog shape is missing"))?;
        let observed_shape: Option<Vec<u64>> = observed_shape
            .iter()
            .map(serde_json::Value::as_u64)
            .collect();
        if entry.get("kind").and_then(serde_json::Value::as_str) != Some("non_expert")
            || entry.get("layer").and_then(serde_json::Value::as_u64) != Some(42)
            || entry.get("dtype").and_then(serde_json::Value::as_str) != Some(dtype)
            || entry.get("bytes").and_then(serde_json::Value::as_u64) != Some(bytes)
            || observed_shape.as_deref() != Some(shape)
        {
            bail!("{tensor} catalog byte/shape/dtype drift");
        }
        Ok(())
    };
    let shape = S14MatvecShape::new(projection.n, projection.k)?.validate_fp8()?;
    let weight_name = format!("{}.weight", projection.name);
    let scale_name = format!("{}.scale", projection.name);
    require(
        &weight_name,
        "F8_E4M3",
        &[projection.n as u64, projection.k as u64],
        shape.fp8_weight_bytes()?,
    )?;
    require(
        &scale_name,
        "F8_E8M0",
        &[(projection.n / 128) as u64, (projection.k / 128) as u64],
        shape.fp8_scale_bytes()?,
    )?;
    Ok(())
}

fn validate_l42_wq_a_catalog(document: &serde_json::Value) -> Result<()> {
    validate_l42_projection_catalog(document, L42_STANDARD_FP8_PROJECTIONS[0])
}

fn load_l42_projection_assets(projection: FrozenFp8Projection) -> Result<(Arc<[u8]>, Arc<[u8]>)> {
    let catalog_path = Path::new(FULLDEPTH43_CATALOG_FILE);
    let catalog_bytes =
        std::fs::read(catalog_path).with_context(|| format!("read {}", catalog_path.display()))?;
    if sha256_bytes(&catalog_bytes) != FULLDEPTH43_CATALOG_SHA256 {
        bail!("FullDepth43 catalog SHA-256 drift");
    }
    let catalog: serde_json::Value = serde_json::from_slice(&catalog_bytes)?;
    validate_l42_projection_catalog(&catalog, projection)?;

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
    let shape = S14MatvecShape::new(projection.n, projection.k)?.validate_fp8()?;
    let weight_name = format!("{}.weight", projection.name);
    let scale_name = format!("{}.scale", projection.name);
    let weight_entry = base_entry(&manifest, &weight_name)?;
    let scale_entry = base_entry(&manifest, &scale_name)?;
    if weight_entry.bytes != shape.fp8_weight_bytes()?
        || weight_entry.sha256 != projection.weight_sha256
        || scale_entry.bytes != shape.fp8_scale_bytes()?
        || scale_entry.sha256 != projection.scale_sha256
    {
        bail!("L42 {} payload whitelist drift", projection.name);
    }
    let weight = read_verified_payload(
        &weight_entry.path,
        weight_entry.bytes as usize,
        projection.weight_sha256,
        &weight_entry.tensor,
    )?;
    let scale = read_verified_payload(
        &scale_entry.path,
        scale_entry.bytes as usize,
        projection.scale_sha256,
        &scale_entry.tensor,
    )?;
    validate_e4m3fn_codes(&weight)?;
    validate_ue8m0_codes(&scale)?;
    Ok((weight, scale))
}

fn load_l42_wq_a_projection_assets() -> Result<(Arc<[u8]>, Arc<[u8]>)> {
    load_l42_projection_assets(L42_STANDARD_FP8_PROJECTIONS[0])
}

fn checked_arena_end(view: &ProjectionArenaView) -> Result<u64> {
    view.offset
        .checked_add(view.bytes)
        .ok_or_else(|| anyhow::anyhow!("FP8 projection arena view byte overflow"))
}

fn resolve_projection_arena(
    input: &ProjectionArenaView,
    output: &ProjectionArenaView,
) -> Result<PathBuf> {
    if input.path.as_os_str().is_empty() || output.path.as_os_str().is_empty() {
        bail!("FP8 projection arena path is empty");
    }
    let input_path = input.path.canonicalize()?;
    let output_path = output.path.canonicalize()?;
    if input_path != output_path
        || input_path.extension().and_then(|value| value.to_str()) != Some("bin")
        || input.offset % 4 != 0
        || output.offset % 4 != 0
    {
        bail!("FP8 projection arena path/alignment drift");
    }
    let file_bytes = input_path.metadata()?.len();
    let input_end = checked_arena_end(input)?;
    let output_end = checked_arena_end(output)?;
    if file_bytes == 0
        || file_bytes > FP8_PROJECTION_ARENA_MAX_BYTES
        || input_end > file_bytes
        || output_end > file_bytes
        || input.offset < output_end && output.offset < input_end
    {
        bail!("FP8 projection arena bounds/overlap drift");
    }
    Ok(input_path)
}

fn arena_views_overlap(left: &ProjectionArenaView, right: &ProjectionArenaView) -> Result<bool> {
    Ok(left.offset < checked_arena_end(right)? && right.offset < checked_arena_end(left)?)
}

fn resolve_projection_batch_arena(
    input: &ProjectionArenaView,
    outputs: [&ProjectionArenaView; 2],
) -> Result<PathBuf> {
    if input.path.as_os_str().is_empty()
        || outputs
            .iter()
            .any(|output| output.path.as_os_str().is_empty())
    {
        bail!("FP8 shared-input batch arena path is empty");
    }
    let input_path = input.path.canonicalize()?;
    let output_paths = [
        outputs[0].path.canonicalize()?,
        outputs[1].path.canonicalize()?,
    ];
    if output_paths.iter().any(|path| path != &input_path)
        || input_path.extension().and_then(|value| value.to_str()) != Some("bin")
        || input.offset % 4 != 0
        || outputs.iter().any(|output| output.offset % 4 != 0)
    {
        bail!("FP8 shared-input batch arena path/alignment drift");
    }
    let file_bytes = input_path.metadata()?.len();
    let output_ends = [
        checked_arena_end(outputs[0])?,
        checked_arena_end(outputs[1])?,
    ];
    if file_bytes == 0
        || file_bytes > FP8_PROJECTION_ARENA_MAX_BYTES
        || checked_arena_end(input)? > file_bytes
        || output_ends.iter().any(|end| *end > file_bytes)
    {
        bail!("FP8 shared-input batch arena bounds drift");
    }
    for output in outputs {
        if arena_views_overlap(input, output)? {
            bail!("FP8 shared-input batch input/output overlap drift");
        }
    }
    if arena_views_overlap(outputs[0], outputs[1])? {
        bail!("FP8 shared-input batch outputs overlap");
    }
    Ok(input_path)
}

fn read_projection_input(path: &Path, view: &ProjectionArenaView) -> Result<Vec<f32>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(view.offset))?;
    let mut payload = vec![0u8; view.bytes as usize];
    file.read_exact(&mut payload)?;
    if sha256_bytes(&payload) != L42_WQ_A_INPUT_SHA256 {
        bail!("L42 wq_a frozen input SHA-256 drift");
    }
    let values: Vec<f32> = payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if values.len() != 4096 || values.iter().any(|value| !value.is_finite()) {
        bail!("L42 wq_a input shape/non-finite drift");
    }
    Ok(values)
}

fn round_f32_to_bf16_f32(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x0000_7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

fn nearest_e4m3fn_code(value: f32) -> Result<u8> {
    if !value.is_finite() || value.abs() > 448.0 {
        bail!("E4M3FN quantization input is non-finite or out of range");
    }
    let negative = value.is_sign_negative();
    let magnitude = value.abs();
    // Positive finite E4M3FN codes are monotonic over 0x00..=0x7e;
    // 0x7f is NaN. Binary-search the first representable value >= magnitude
    // instead of scanning all 127 codes for every activation element.
    let mut left = 0usize;
    let mut right = 0x7fusize;
    while left < right {
        let middle = (left + right) / 2;
        if e4m3fn(middle as u8) < magnitude {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    let upper = left.min(0x7e) as u8;
    let lower = upper.saturating_sub(1);
    let lower_distance = (magnitude - e4m3fn(lower)).abs();
    let upper_distance = (e4m3fn(upper) - magnitude).abs();
    // Ties use the destination mantissa's least-significant bit, matching
    // torch.float8_e4m3fn round-to-nearest-ties-to-even.
    let best_code = if upper_distance < lower_distance
        || (upper_distance == lower_distance && upper & 1 == 0 && lower & 1 != 0)
    {
        upper
    } else {
        lower
    };
    Ok(if negative {
        best_code | 0x80
    } else {
        best_code
    })
}

fn official_group128_e4m3fn_activation_quantize_dequantize(values: &[f32]) -> Result<Vec<f32>> {
    const GROUP_SIZE: usize = 128;
    const AMAX_FLOOR: f32 = 1.0e-4;
    const MAX_FINITE: f32 = 448.0;
    if values.is_empty()
        || values.len() % GROUP_SIZE != 0
        || values.iter().any(|value| !value.is_finite())
    {
        bail!("group-128 E4M3FN activation input contract drift");
    }
    let mut output = Vec::with_capacity(values.len());
    for block in values.chunks_exact(GROUP_SIZE) {
        let amax = block
            .iter()
            .fold(0.0f32, |maximum, value| maximum.max(value.abs()))
            .max(AMAX_FLOOR);
        let scale_exponent = (amax / MAX_FINITE).log2().ceil() as i32;
        let scale = 2.0f32.powi(scale_exponent);
        if !scale.is_finite() || scale <= 0.0 {
            bail!("group-128 E4M3FN activation scale drift");
        }
        for value in block {
            let normalized = (*value / scale).clamp(-MAX_FINITE, MAX_FINITE);
            let code = nearest_e4m3fn_code(normalized)?;
            output.push(e4m3fn(code) * scale);
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        bail!("group-128 E4M3FN activation output is non-finite");
    }
    Ok(output)
}

fn validate_frozen_l42_output_chain_hashes(
    layer: u32,
    input_sha256: &str,
    wo_a_output_sha256: &str,
    requantized_activation_sha256: &str,
    output_sha256: &str,
) -> Result<()> {
    if layer == 42
        && input_sha256 == L42_WO_A_INPUT_SHA256
        && (wo_a_output_sha256 != L42_WO_A_OUTPUT_SHA256
            || requantized_activation_sha256 != L42_WO_A_REQUANTIZED_SHA256
            || output_sha256 != L42_WO_B_OUTPUT_SHA256)
    {
        bail!("frozen L42 FP8 output-chain SHA-256 drift");
    }
    Ok(())
}

fn write_projection_output(path: &Path, view: &ProjectionArenaView, values: &[f32]) -> Result<()> {
    if values.len() * std::mem::size_of::<f32>() != view.bytes as usize
        || values.iter().any(|value| !value.is_finite())
    {
        bail!("L42 wq_a output shape/non-finite drift");
    }
    let mut payload = Vec::with_capacity(view.bytes as usize);
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    if sha256_bytes(&payload) != L42_WQ_A_OUTPUT_SHA256 {
        bail!("L42 wq_a BF16-rounded output SHA-256 drift");
    }
    let mut file = OpenOptions::new().write(true).open(path)?;
    file.seek(SeekFrom::Start(view.offset))?;
    file.write_all(&payload)?;
    file.flush()?;
    Ok(())
}

fn read_fulldepth_fp8_input(
    path: &Path,
    view: &ProjectionArenaView,
    expected_sha256: &str,
    kernel: FullDepthFp8Kernel,
) -> Result<Vec<f32>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(view.offset))?;
    let mut payload = vec![0u8; view.bytes as usize];
    file.read_exact(&mut payload)?;
    if sha256_bytes(&payload) != expected_sha256 {
        bail!("FullDepth43 FP8 activation SHA-256 drift");
    }
    let values: Vec<f32> = payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if values.len() as u64 != kernel.input_elements()?
        || values.iter().any(|value| !value.is_finite())
    {
        bail!("FullDepth43 FP8 activation shape/non-finite drift");
    }
    if matches!(kernel, FullDepthFp8Kernel::GroupedWoA(_))
        && values.iter().any(|value| value.to_bits() & 0xffff != 0)
    {
        bail!("FullDepth43 grouped wo_a input is not BF16-carrying F32");
    }
    Ok(values)
}

fn write_fulldepth_fp8_output(
    path: &Path,
    view: &ProjectionArenaView,
    values: &[f32],
    kernel: FullDepthFp8Kernel,
) -> Result<String> {
    if values.len() as u64 != kernel.output_elements()?
        || values.len() * std::mem::size_of::<f32>() != view.bytes as usize
        || values.iter().any(|value| !value.is_finite())
        || values.iter().any(|value| value.to_bits() & 0xffff != 0)
    {
        bail!("FullDepth43 FP8 BF16 output contract drift");
    }
    let mut payload = Vec::with_capacity(view.bytes as usize);
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let output_sha256 = sha256_bytes(&payload);
    let mut file = OpenOptions::new().write(true).open(path)?;
    file.seek(SeekFrom::Start(view.offset))?;
    file.write_all(&payload)?;
    file.flush()?;
    Ok(output_sha256)
}

const PROJECTION_UPLOAD_X: usize = 0;
const PROJECTION_UPLOAD_WEIGHT: usize = 1;
const PROJECTION_UPLOAD_SCALE: usize = 2;
const PROJECTION_X: usize = 3;
const PROJECTION_WEIGHT: usize = 4;
const PROJECTION_SCALE: usize = 5;
const PROJECTION_Y: usize = 6;
const PROJECTION_READBACK: usize = 7;

/// Construction-safe owner for the fixed buffers used by the projection
/// worker. `GpuBuffer` deliberately has explicit destruction, so this small
/// arena makes partially constructed workers fail without leaking Vulkan
/// memory.
struct ProjectionBufferArena<'a> {
    ctx: &'a VulkanContext,
    buffers: Vec<GpuBuffer>,
}

impl<'a> ProjectionBufferArena<'a> {
    fn new(ctx: &'a VulkanContext) -> Self {
        Self {
            ctx,
            buffers: Vec::with_capacity(8),
        }
    }

    fn push(&mut self, buffer: GpuBuffer) {
        self.buffers.push(buffer);
    }

    fn get(&self, index: usize) -> &GpuBuffer {
        &self.buffers[index]
    }
}

impl Drop for ProjectionBufferArena<'_> {
    fn drop(&mut self) {
        for buffer in self.buffers.iter().rev() {
            buffer.destroy(self.ctx);
        }
    }
}

/// One exact packed-FP8 projection slot. The descriptor, command buffer,
/// fence, packed weight and scale all survive across requests. Only the
/// shape-sized activation staging buffer is rewritten for each token.
struct PersistentFp8ProjectionSlot<'a> {
    ctx: &'a VulkanContext,
    shape: S14MatvecShape,
    buffers: ProjectionBufferArena<'a>,
    dispatch: Option<S14Fp8Dispatch>,
    command_pool: Option<vk::CommandPool>,
    command_buffer: Option<vk::CommandBuffer>,
    fence: Option<vk::Fence>,
}

impl<'a> PersistentFp8ProjectionSlot<'a> {
    fn new(
        ctx: &'a VulkanContext,
        pipelines: &S14NumericPipelines,
        shape: S14MatvecShape,
        weight: &[u8],
        scale: &[u8],
    ) -> Result<Self> {
        let shape = shape.validate_fp8()?;
        if weight.len() as u64 != shape.fp8_weight_bytes()?
            || scale.len() as u64 != shape.fp8_scale_bytes()?
        {
            bail!("L42 packed-FP8 persistent payload byte drift");
        }
        let x_bytes = shape.fp32_input_bytes()?;
        let weight_bytes = shape.fp8_weight_bytes()?;
        let scale_bytes = shape.fp8_scale_bytes()?;
        let y_bytes = shape.fp32_output_bytes()?;
        let storage_dst = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;

        let mut buffers = ProjectionBufferArena::new(ctx);
        buffers.push(GpuBuffer::new_staging(ctx, x_bytes)?);
        buffers.push(GpuBuffer::new_staging(ctx, weight_bytes)?);
        buffers.push(GpuBuffer::new_staging(ctx, scale_bytes)?);
        buffers.push(GpuBuffer::new_vram(ctx, x_bytes, storage_dst)?);
        buffers.push(GpuBuffer::new_vram(ctx, weight_bytes, storage_dst)?);
        buffers.push(GpuBuffer::new_vram(ctx, scale_bytes, storage_dst)?);
        buffers.push(GpuBuffer::new_vram(
            ctx,
            y_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        )?);
        buffers.push(GpuBuffer::new(
            ctx,
            y_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?);
        if buffers.buffers.len() != 8 {
            bail!("L42 packed-FP8 persistent buffer layout drift");
        }
        unsafe {
            buffers.get(PROJECTION_UPLOAD_WEIGHT).write_at(0, weight);
            buffers.get(PROJECTION_UPLOAD_SCALE).write_at(0, scale);
        }

        let mut slot = Self {
            ctx,
            shape,
            buffers,
            dispatch: None,
            command_pool: None,
            command_buffer: None,
            fence: None,
        };
        unsafe {
            let pool = ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.qf_graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            slot.command_pool = Some(pool);
            slot.command_buffer = Some(allocate_command_buffer(ctx, pool)?);
            slot.fence = Some(
                ctx.device
                    .create_fence(&vk::FenceCreateInfo::default(), None)?,
            );
        }
        slot.dispatch = Some(pipelines.bind_fp8(
            ctx,
            shape,
            slot.buffers.get(PROJECTION_X),
            slot.buffers.get(PROJECTION_WEIGHT),
            slot.buffers.get(PROJECTION_SCALE),
            slot.buffers.get(PROJECTION_Y),
        )?);
        slot.upload_static_payload()?;
        Ok(slot)
    }

    fn begin_recording(&self) -> Result<vk::CommandBuffer> {
        let cb = self
            .command_buffer
            .ok_or_else(|| anyhow::anyhow!("L42 packed-FP8 command buffer is unavailable"))?;
        let fence = self
            .fence
            .ok_or_else(|| anyhow::anyhow!("L42 packed-FP8 fence is unavailable"))?;
        unsafe {
            self.ctx.device.reset_fences(&[fence])?;
            self.ctx
                .device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            self.ctx.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        Ok(cb)
    }

    fn submit_and_wait(&self, cb: vk::CommandBuffer) -> Result<()> {
        let fence = self
            .fence
            .ok_or_else(|| anyhow::anyhow!("L42 packed-FP8 fence is unavailable"))?;
        let command_buffers = [cb];
        unsafe {
            self.ctx.device.end_command_buffer(cb)?;
            self.ctx.device.queue_submit(
                self.ctx.q_graphics,
                &[vk::SubmitInfo::default().command_buffers(&command_buffers)],
                fence,
            )?;
            self.ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }
        Ok(())
    }

    fn upload_static_payload(&self) -> Result<()> {
        let cb = self.begin_recording()?;
        unsafe {
            copy(
                self.ctx,
                cb,
                self.buffers.get(PROJECTION_UPLOAD_WEIGHT),
                self.buffers.get(PROJECTION_WEIGHT),
                self.shape.fp8_weight_bytes()?,
            );
            copy(
                self.ctx,
                cb,
                self.buffers.get(PROJECTION_UPLOAD_SCALE),
                self.buffers.get(PROJECTION_SCALE),
                self.shape.fp8_scale_bytes()?,
            );
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        self.submit_and_wait(cb)
    }

    fn execute(&self, pipelines: &S14NumericPipelines, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.shape.k as usize || input.iter().any(|value| !value.is_finite()) {
            bail!("L42 packed-FP8 persistent activation contract drift");
        }
        unsafe {
            self.buffers
                .get(PROJECTION_UPLOAD_X)
                .write_at(0, bytemuck::cast_slice(input));
        }
        let cb = self.begin_recording()?;
        unsafe {
            copy(
                self.ctx,
                cb,
                self.buffers.get(PROJECTION_UPLOAD_X),
                self.buffers.get(PROJECTION_X),
                self.shape.fp32_input_bytes()?,
            );
            let upload_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[upload_barrier],
                &[],
                &[],
            );
            let dispatch = self
                .dispatch
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("L42 packed-FP8 dispatch is unavailable"))?;
            pipelines.cmd_fp8_matvec(self.ctx, cb, dispatch);
            let readback_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            self.ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[readback_barrier],
                &[],
                &[],
            );
            copy(
                self.ctx,
                cb,
                self.buffers.get(PROJECTION_Y),
                self.buffers.get(PROJECTION_READBACK),
                self.shape.fp32_output_bytes()?,
            );
        }
        self.submit_and_wait(cb)?;
        let raw = unsafe {
            std::slice::from_raw_parts(
                self.buffers.get(PROJECTION_READBACK).mapped() as *const f32,
                self.shape.n as usize,
            )
            .to_vec()
        };
        if raw.iter().any(|value| !value.is_finite()) {
            bail!("L42 wq_a Vulkan output contains non-finite values");
        }
        Ok(raw.into_iter().map(round_f32_to_bf16_f32).collect())
    }
}

impl Drop for PersistentFp8ProjectionSlot<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
        }
        if let Some(dispatch) = self.dispatch.take() {
            dispatch.binder.destroy(self.ctx);
        }
        unsafe {
            if let Some(fence) = self.fence.take() {
                self.ctx.device.destroy_fence(fence, None);
            }
            if let Some(pool) = self.command_pool.take() {
                self.ctx.device.destroy_command_pool(pool, None);
            }
        }
    }
}

struct PersistentGroupedFp8ProjectionSlot<'a> {
    ctx: &'a VulkanContext,
    shape: S14GroupedMatvecShape,
    buffers: ProjectionBufferArena<'a>,
    dispatch: Option<S14GroupedFp8Bf16Dispatch>,
    command_pool: Option<vk::CommandPool>,
    command_buffer: Option<vk::CommandBuffer>,
    fence: Option<vk::Fence>,
}

impl<'a> PersistentGroupedFp8ProjectionSlot<'a> {
    fn new(
        ctx: &'a VulkanContext,
        pipelines: &S14NumericPipelines,
        shape: S14GroupedMatvecShape,
        weight: &[u8],
        scale: &[u8],
    ) -> Result<Self> {
        let shape = shape.validate_fp8_bf16_weight()?;
        if weight.len() as u64 != shape.fp8_weight_bytes()?
            || scale.len() as u64 != shape.fp8_scale_bytes()?
        {
            bail!("FullDepth43 grouped persistent payload byte drift");
        }
        let x_bytes = shape.fp32_input_bytes()?;
        let weight_bytes = shape.fp8_weight_bytes()?;
        let scale_bytes = shape.fp8_scale_bytes()?;
        let y_bytes = shape.fp32_output_bytes()?;
        let storage_dst = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let mut buffers = ProjectionBufferArena::new(ctx);
        buffers.push(GpuBuffer::new_staging(ctx, x_bytes)?);
        buffers.push(GpuBuffer::new_staging(ctx, weight_bytes)?);
        buffers.push(GpuBuffer::new_staging(ctx, scale_bytes)?);
        buffers.push(GpuBuffer::new_vram(ctx, x_bytes, storage_dst)?);
        buffers.push(GpuBuffer::new_vram(ctx, weight_bytes, storage_dst)?);
        buffers.push(GpuBuffer::new_vram(ctx, scale_bytes, storage_dst)?);
        buffers.push(GpuBuffer::new_vram(
            ctx,
            y_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        )?);
        buffers.push(GpuBuffer::new(
            ctx,
            y_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?);
        unsafe {
            buffers.get(PROJECTION_UPLOAD_WEIGHT).write_at(0, weight);
            buffers.get(PROJECTION_UPLOAD_SCALE).write_at(0, scale);
        }
        let mut slot = Self {
            ctx,
            shape,
            buffers,
            dispatch: None,
            command_pool: None,
            command_buffer: None,
            fence: None,
        };
        unsafe {
            let pool = ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.qf_graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            slot.command_pool = Some(pool);
            slot.command_buffer = Some(allocate_command_buffer(ctx, pool)?);
            slot.fence = Some(
                ctx.device
                    .create_fence(&vk::FenceCreateInfo::default(), None)?,
            );
        }
        slot.dispatch = Some(pipelines.bind_grouped_fp8_bf16_weight(
            ctx,
            shape,
            slot.buffers.get(PROJECTION_X),
            slot.buffers.get(PROJECTION_WEIGHT),
            slot.buffers.get(PROJECTION_SCALE),
            slot.buffers.get(PROJECTION_Y),
        )?);
        slot.upload_static_payload()?;
        Ok(slot)
    }

    fn begin_recording(&self) -> Result<vk::CommandBuffer> {
        let command_buffer = self
            .command_buffer
            .ok_or_else(|| anyhow::anyhow!("FullDepth43 grouped command buffer unavailable"))?;
        let fence = self
            .fence
            .ok_or_else(|| anyhow::anyhow!("FullDepth43 grouped fence unavailable"))?;
        unsafe {
            self.ctx.device.reset_fences(&[fence])?;
            self.ctx
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.ctx.device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        Ok(command_buffer)
    }

    fn submit_and_wait(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
        let fence = self
            .fence
            .ok_or_else(|| anyhow::anyhow!("FullDepth43 grouped fence unavailable"))?;
        let command_buffers = [command_buffer];
        unsafe {
            self.ctx.device.end_command_buffer(command_buffer)?;
            self.ctx.device.queue_submit(
                self.ctx.q_graphics,
                &[vk::SubmitInfo::default().command_buffers(&command_buffers)],
                fence,
            )?;
            self.ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }
        Ok(())
    }

    fn upload_static_payload(&self) -> Result<()> {
        let command_buffer = self.begin_recording()?;
        unsafe {
            copy(
                self.ctx,
                command_buffer,
                self.buffers.get(PROJECTION_UPLOAD_WEIGHT),
                self.buffers.get(PROJECTION_WEIGHT),
                self.shape.fp8_weight_bytes()?,
            );
            copy(
                self.ctx,
                command_buffer,
                self.buffers.get(PROJECTION_UPLOAD_SCALE),
                self.buffers.get(PROJECTION_SCALE),
                self.shape.fp8_scale_bytes()?,
            );
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        self.submit_and_wait(command_buffer)
    }

    fn execute(&self, pipelines: &S14NumericPipelines, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() * std::mem::size_of::<f32>() != self.shape.fp32_input_bytes()? as usize
            || input.iter().any(|value| !value.is_finite())
            || input.iter().any(|value| value.to_bits() & 0xffff != 0)
        {
            bail!("FullDepth43 grouped persistent activation contract drift");
        }
        unsafe {
            self.buffers
                .get(PROJECTION_UPLOAD_X)
                .write_at(0, bytemuck::cast_slice(input));
        }
        let command_buffer = self.begin_recording()?;
        unsafe {
            copy(
                self.ctx,
                command_buffer,
                self.buffers.get(PROJECTION_UPLOAD_X),
                self.buffers.get(PROJECTION_X),
                self.shape.fp32_input_bytes()?,
            );
            let upload_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[upload_barrier],
                &[],
                &[],
            );
            let dispatch = self
                .dispatch
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("FullDepth43 grouped dispatch unavailable"))?;
            pipelines.cmd_grouped_fp8_bf16_weight_matvec(self.ctx, command_buffer, dispatch);
            let readback_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            self.ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[readback_barrier],
                &[],
                &[],
            );
            copy(
                self.ctx,
                command_buffer,
                self.buffers.get(PROJECTION_Y),
                self.buffers.get(PROJECTION_READBACK),
                self.shape.fp32_output_bytes()?,
            );
        }
        self.submit_and_wait(command_buffer)?;
        let output = unsafe {
            std::slice::from_raw_parts(
                self.buffers.get(PROJECTION_READBACK).mapped() as *const f32,
                self.shape.flat_n()? as usize,
            )
            .to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            bail!("FullDepth43 grouped persistent output contains non-finite values");
        }
        Ok(output.into_iter().map(round_f32_to_bf16_f32).collect())
    }
}

impl Drop for PersistentGroupedFp8ProjectionSlot<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
        }
        if let Some(dispatch) = self.dispatch.take() {
            dispatch.binder.destroy(self.ctx);
        }
        unsafe {
            if let Some(fence) = self.fence.take() {
                self.ctx.device.destroy_fence(fence, None);
            }
            if let Some(pool) = self.command_pool.take() {
                self.ctx.device.destroy_command_pool(pool, None);
            }
        }
    }
}

fn execute_fulldepth_grouped_fp8_once(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    shape: S14GroupedMatvecShape,
    input: &[f32],
    weight: &[u8],
    scale: &[u8],
) -> Result<Vec<f32>> {
    let shape = shape.validate_fp8_bf16_weight()?;
    if input.len() * 4 != shape.fp32_input_bytes()? as usize
        || weight.len() != shape.fp8_weight_bytes()? as usize
        || scale.len() != shape.fp8_scale_bytes()? as usize
    {
        bail!("FullDepth43 grouped FP8 tensor byte contract drift");
    }
    let buffers = DeviceBuffers::new(ctx, input, weight, scale, shape.flat_n()? as usize)?;
    let result = (|| -> Result<Vec<f32>> {
        buffers.upload(ctx)?;
        let dispatch = pipelines.bind_grouped_fp8_bf16_weight(
            ctx,
            shape,
            &buffers.x,
            &buffers.weight,
            &buffers.scale,
            &buffers.y,
        )?;
        let execution = (|| -> Result<Vec<f32>> {
            let pool = unsafe { make_command_pool(ctx)? };
            let recorded = (|| -> Result<()> {
                let cb = unsafe { allocate_command_buffer(ctx, pool)? };
                unsafe {
                    ctx.device.begin_command_buffer(
                        cb,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )?;
                    pipelines.cmd_grouped_fp8_bf16_weight_matvec(ctx, cb, &dispatch);
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
                    copy(
                        ctx,
                        cb,
                        &buffers.y,
                        &buffers.readback,
                        shape.fp32_output_bytes()?,
                    );
                    ctx.device.end_command_buffer(cb)?;
                    submit_and_wait(ctx, cb)?;
                }
                Ok(())
            })();
            unsafe { ctx.device.destroy_command_pool(pool, None) };
            recorded?;
            let raw = buffers.output(shape.flat_n()? as usize);
            if raw.iter().any(|value| !value.is_finite()) {
                bail!("FullDepth43 grouped FP8 output contains non-finite values");
            }
            Ok(raw.into_iter().map(round_f32_to_bf16_f32).collect())
        })();
        dispatch.binder.destroy(ctx);
        execution
    })();
    buffers.destroy(ctx);
    result
}

fn execute_fulldepth_fp8_once(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    kernel: FullDepthFp8Kernel,
    input: &[f32],
    weight: &[u8],
    scale: &[u8],
) -> Result<Vec<f32>> {
    validate_e4m3fn_codes(weight)?;
    validate_ue8m0_codes(scale)?;
    match kernel {
        FullDepthFp8Kernel::Standard(shape) => {
            let slot = PersistentFp8ProjectionSlot::new(ctx, pipelines, shape, weight, scale)?;
            slot.execute(pipelines, input)
        }
        FullDepthFp8Kernel::GroupedWoA(shape) => {
            execute_fulldepth_grouped_fp8_once(ctx, pipelines, shape, input, weight, scale)
        }
    }
}

fn write_fp8_projection_error(
    stdout: &mut impl Write,
    request_id: &str,
    error: &anyhow::Error,
) -> Result<()> {
    serde_json::to_writer(
        &mut *stdout,
        &Fp8ProjectionError {
            protocol: FP8_PROJECTION_PROTOCOL,
            request_id,
            ok: false,
            error: format!("{error:#}"),
            poisoned: true,
        },
    )?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn run_l42_wq_a_fp8_projection_loop(
    pipelines: &S14NumericPipelines,
    slot: &PersistentFp8ProjectionSlot<'_>,
) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(
        &mut stdout,
        &serde_json::json!({
            "protocol": FP8_PROJECTION_PROTOCOL,
            "op": "hello",
            "ready": true,
            "revision": REVISION,
            "profile": "fulldepth43_native_top6",
            "layer": 42,
            "position": 0,
            "projection": {
                "name": L42_WQ_A_PROJECTION,
                "n": 1024,
                "k": 4096,
                "activation_contract": "cpu_e4m3fn_quant_dequant_f32",
                "output_rounding": "bf16_rne_then_f32_le",
            },
            "arena_transport": "shared_binary_file",
            "weight_resident": true,
            "weight_sha256": L42_WQ_A_WEIGHT_SHA256,
            "scale_sha256": L42_WQ_A_SCALE_SHA256,
            "catalog_sha256": FULLDEPTH43_CATALOG_SHA256,
            "input_sha256": L42_WQ_A_INPUT_SHA256,
            "output_sha256": L42_WQ_A_OUTPUT_SHA256,
            "numeric_mode": "packed_fp8_e4m3_ue8m0_exact_audit",
        }),
    )?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;

    let mut expected_epoch = 0u64;
    let stdin = std::io::stdin();
    for line_result in BufReader::new(stdin.lock()).lines() {
        let line = match line_result {
            Ok(value) => value,
            Err(error) => {
                let error = anyhow::Error::new(error).context("read FP8 projection request");
                write_fp8_projection_error(&mut stdout, "unknown", &error)?;
                return Err(error.context("FP8 projection worker poisoned"));
            }
        };
        if line.len() > 65_536 {
            let error = anyhow::anyhow!("FP8 projection request exceeds 64 KiB");
            write_fp8_projection_error(&mut stdout, "unknown", &error)?;
            return Err(error.context("FP8 projection worker poisoned"));
        }
        let request: Fp8ProjectionRequest = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                let error = anyhow::Error::new(error).context("parse FP8 projection JSON request");
                write_fp8_projection_error(&mut stdout, "unknown", &error)?;
                return Err(error.context("FP8 projection worker poisoned"));
            }
        };
        let request_id = request.request_id.clone();
        let response = (|| -> Result<Fp8ProjectionResponse> {
            validate_l42_wq_a_request(&request, expected_epoch)?;
            let arena = resolve_projection_arena(&request.input, &request.output)?;
            let input = read_projection_input(&arena, &request.input)?;
            let output = slot.execute(pipelines, &input)?;
            write_projection_output(&arena, &request.output, &output)?;
            Ok(Fp8ProjectionResponse {
                protocol: FP8_PROJECTION_PROTOCOL,
                request_id: request.request_id,
                ok: true,
                revision: REVISION,
                profile: "fulldepth43_native_top6",
                layer: 42,
                position: 0,
                arena_epoch: request.arena_epoch,
                projection: L42_WQ_A_PROJECTION,
                input: request.input,
                output_written: request.output,
                input_sha256: L42_WQ_A_INPUT_SHA256,
                output_sha256: L42_WQ_A_OUTPUT_SHA256,
                weight_sha256: L42_WQ_A_WEIGHT_SHA256,
                scale_sha256: L42_WQ_A_SCALE_SHA256,
                catalog_sha256: FULLDEPTH43_CATALOG_SHA256,
                weight_resident: true,
                static_uploaded_bytes: 4_194_560,
                request_uploaded_bytes: 16_384,
                numeric_mode: "packed_fp8_e4m3_ue8m0_exact_audit",
                output_rounding: "bf16_rne_then_f32_le",
            })
        })();
        match response {
            Ok(response) => {
                serde_json::to_writer(&mut stdout, &response)?;
                stdout.write_all(b"\n")?;
                stdout.flush()?;
                expected_epoch = expected_epoch
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("FP8 projection arena epoch overflow"))?;
            }
            Err(error) => {
                write_fp8_projection_error(&mut stdout, &request_id, &error)?;
                return Err(error.context("FP8 projection worker poisoned"));
            }
        }
    }
    Ok(())
}

fn run_l42_wq_a_fp8_projection_worker() -> Result<()> {
    let (weight, scale) = load_l42_wq_a_projection_assets()?;
    let ctx = VulkanContext::init()?;
    let properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    if properties.vendor_id != AMD_VENDOR_ID || properties.device_id != NAVI10_DEVICE_ID {
        bail!(
            "L42 wq_a exact FP8 worker requires RX 5700 XT; found 0x{:04x}:0x{:04x} ({})",
            properties.vendor_id,
            properties.device_id,
            ctx.gpu_name
        );
    }
    let pipelines = S14NumericPipelines::new_exact_audit(&ctx)?;
    let slot = match PersistentFp8ProjectionSlot::new(
        &ctx,
        &pipelines,
        S14MatvecShape::new(1024, 4096)?,
        &weight,
        &scale,
    ) {
        Ok(value) => value,
        Err(error) => {
            pipelines.destroy(&ctx);
            return Err(error.context("initialize persistent L42 wq_a FP8 projection slot"));
        }
    };
    let result = run_l42_wq_a_fp8_projection_loop(&pipelines, &slot);
    drop(slot);
    pipelines.destroy(&ctx);
    result
}

fn write_fulldepth_fp8_attention_error(
    stdout: &mut impl Write,
    request_id: &str,
    error: &anyhow::Error,
) -> Result<()> {
    serde_json::to_writer(
        &mut *stdout,
        &Fp8ProjectionError {
            protocol: FULLDEPTH_FP8_ATTENTION_PROTOCOL,
            request_id,
            ok: false,
            error: format!("{error:#}"),
            poisoned: true,
        },
    )?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

struct FullDepthFp8SlotExecution {
    output: Vec<f32>,
    gpu_slot_cache_hit: bool,
    gpu_slot_cache_entries: usize,
    gpu_slot_resident_bytes: u64,
    payload_uploaded_bytes: u64,
}

#[allow(clippy::too_many_arguments)]
fn execute_fulldepth_fp8_with_slot_cache<'ctx>(
    ctx: &'ctx VulkanContext,
    pipelines: &S14NumericPipelines,
    payload_cache: &mut VerifiedPayloadCache,
    cache_root: &Path,
    standard_slots: &mut HashMap<String, PersistentFp8ProjectionSlot<'ctx>>,
    grouped_slots: &mut HashMap<String, PersistentGroupedFp8ProjectionSlot<'ctx>>,
    gpu_slot_resident_bytes: &mut u64,
    request: &FullDepthFp8AttentionRequest,
    kernel: FullDepthFp8Kernel,
    input: &[f32],
) -> Result<FullDepthFp8SlotExecution> {
    let slot_key = fulldepth_fp8_slot_key(kernel, request);
    let (output, gpu_slot_cache_hit, payload_uploaded_bytes) = match kernel {
        FullDepthFp8Kernel::Standard(shape) => {
            let hit = standard_slots.contains_key(&slot_key);
            if !hit {
                let next_resident_bytes = reserve_fulldepth_fp8_gpu_slot(
                    standard_slots.len() + grouped_slots.len(),
                    *gpu_slot_resident_bytes,
                    request,
                )?;
                let weight = read_verified_payload_cached_with_root(
                    payload_cache,
                    cache_root,
                    &request.weight.path,
                    request.weight.bytes as usize,
                    &request.weight.sha256,
                    &request.weight.tensor,
                )?;
                let scale = read_verified_payload_cached_with_root(
                    payload_cache,
                    cache_root,
                    &request.scale.path,
                    request.scale.bytes as usize,
                    &request.scale.sha256,
                    &request.scale.tensor,
                )?;
                validate_e4m3fn_codes(&weight)?;
                validate_ue8m0_codes(&scale)?;
                let slot =
                    PersistentFp8ProjectionSlot::new(ctx, pipelines, shape, &weight, &scale)?;
                *gpu_slot_resident_bytes = next_resident_bytes;
                standard_slots.insert(slot_key.clone(), slot);
            }
            let slot = standard_slots
                .get(&slot_key)
                .ok_or_else(|| anyhow::anyhow!("FullDepth43 GPU slot insertion failed"))?;
            (
                slot.execute(pipelines, input)?,
                hit,
                if hit {
                    0
                } else {
                    request.weight.bytes + request.scale.bytes
                },
            )
        }
        FullDepthFp8Kernel::GroupedWoA(shape) => {
            let hit = grouped_slots.contains_key(&slot_key);
            if !hit {
                let next_resident_bytes = reserve_fulldepth_fp8_gpu_slot(
                    standard_slots.len() + grouped_slots.len(),
                    *gpu_slot_resident_bytes,
                    request,
                )?;
                let weight = read_verified_payload_cached_with_root(
                    payload_cache,
                    cache_root,
                    &request.weight.path,
                    request.weight.bytes as usize,
                    &request.weight.sha256,
                    &request.weight.tensor,
                )?;
                let scale = read_verified_payload_cached_with_root(
                    payload_cache,
                    cache_root,
                    &request.scale.path,
                    request.scale.bytes as usize,
                    &request.scale.sha256,
                    &request.scale.tensor,
                )?;
                validate_e4m3fn_codes(&weight)?;
                validate_ue8m0_codes(&scale)?;
                let slot = PersistentGroupedFp8ProjectionSlot::new(
                    ctx, pipelines, shape, &weight, &scale,
                )?;
                *gpu_slot_resident_bytes = next_resident_bytes;
                grouped_slots.insert(slot_key.clone(), slot);
            }
            let slot = grouped_slots
                .get(&slot_key)
                .ok_or_else(|| anyhow::anyhow!("FullDepth43 grouped GPU slot insertion failed"))?;
            (
                slot.execute(pipelines, input)?,
                hit,
                if hit {
                    0
                } else {
                    request.weight.bytes + request.scale.bytes
                },
            )
        }
    };
    Ok(FullDepthFp8SlotExecution {
        output,
        gpu_slot_cache_hit,
        gpu_slot_cache_entries: standard_slots.len() + grouped_slots.len(),
        gpu_slot_resident_bytes: *gpu_slot_resident_bytes,
        payload_uploaded_bytes,
    })
}

fn run_fulldepth_fp8_attention_loop(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    catalog: &serde_json::Value,
) -> Result<()> {
    let cache_root = Path::new(MODEL_DIR).join("range_cache").canonicalize()?;
    let mut payload_cache = VerifiedPayloadCache::new(FULLDEPTH_FP8_PAYLOAD_CACHE_BYTES)?;
    let mut standard_slots: HashMap<String, PersistentFp8ProjectionSlot<'_>> = HashMap::new();
    let mut grouped_slots: HashMap<String, PersistentGroupedFp8ProjectionSlot<'_>> = HashMap::new();
    let mut gpu_slot_resident_bytes = 0u64;
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(
        &mut stdout,
        &serde_json::json!({
            "protocol": FULLDEPTH_FP8_ATTENTION_PROTOCOL,
            "op": "hello",
            "ready": true,
            "revision": REVISION,
            "profile": "fulldepth43_native_top6",
            "layers": {"min": 0, "max": 42},
            "position_max": FULLDEPTH_FP8_MAX_POSITION,
            "projections": ["wq_a", "wkv", "wq_b", "indexer.wq_b", "wo_b", "wo_a"],
            "kernels": ["standard", "grouped_wo_a"],
            "arena_transport": "shared_binary_file",
            "catalog_sha256": FULLDEPTH43_CATALOG_SHA256,
            "payload_root": cache_root,
            "payload_cache_bytes": FULLDEPTH_FP8_PAYLOAD_CACHE_BYTES,
            "output_rounding": "bf16_rne_then_f32_le",
        }),
    )?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;

    let mut expected_epoch = 0u64;
    let stdin = std::io::stdin();
    for line_result in BufReader::new(stdin.lock()).lines() {
        let line = match line_result {
            Ok(value) => value,
            Err(error) => {
                let error = anyhow::Error::new(error).context("read FullDepth43 FP8 request");
                write_fulldepth_fp8_attention_error(&mut stdout, "unknown", &error)?;
                return Err(error.context("FullDepth43 FP8 worker poisoned"));
            }
        };
        if line.len() > 65_536 {
            let error = anyhow::anyhow!("FullDepth43 FP8 request exceeds 64 KiB");
            write_fulldepth_fp8_attention_error(&mut stdout, "unknown", &error)?;
            return Err(error.context("FullDepth43 FP8 worker poisoned"));
        }
        let request: FullDepthFp8AttentionWorkerRequest = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                let error = anyhow::Error::new(error).context("parse FullDepth43 FP8 JSON request");
                write_fulldepth_fp8_attention_error(&mut stdout, "unknown", &error)?;
                return Err(error.context("FullDepth43 FP8 worker poisoned"));
            }
        };
        let request_id = match &request {
            FullDepthFp8AttentionWorkerRequest::Single(request) => request.request_id.clone(),
            FullDepthFp8AttentionWorkerRequest::SharedInputBatch(request) => {
                request.request_id.clone()
            }
            FullDepthFp8AttentionWorkerRequest::OutputChain(request) => request.request_id.clone(),
        };
        let response = (|| -> Result<FullDepthFp8AttentionWorkerResponse> {
            match request {
                FullDepthFp8AttentionWorkerRequest::Single(request) => {
                    let kernel =
                        validate_fulldepth_fp8_attention_request(&request, expected_epoch)?;
                    validate_fulldepth_fp8_attention_catalog(catalog, &request, kernel)?;
                    validate_fulldepth_fp8_cache_proofs(catalog, &request, &cache_root)?;
                    let arena = resolve_projection_arena(&request.input, &request.output)?;
                    let input = read_fulldepth_fp8_input(
                        &arena,
                        &request.input,
                        &request.input_sha256,
                        kernel,
                    )?;
                    let execution = execute_fulldepth_fp8_with_slot_cache(
                        ctx,
                        pipelines,
                        &mut payload_cache,
                        &cache_root,
                        &mut standard_slots,
                        &mut grouped_slots,
                        &mut gpu_slot_resident_bytes,
                        &request,
                        kernel,
                        &input,
                    )?;
                    let output_sha256 = write_fulldepth_fp8_output(
                        &arena,
                        &request.output,
                        &execution.output,
                        kernel,
                    )?;
                    Ok(FullDepthFp8AttentionWorkerResponse::Single(
                        FullDepthFp8AttentionResponse {
                            protocol: FULLDEPTH_FP8_ATTENTION_PROTOCOL,
                            request_id: request.request_id,
                            ok: true,
                            revision: REVISION,
                            profile: "fulldepth43_native_top6",
                            layer: request.layer,
                            position: request.position,
                            arena_epoch: request.arena_epoch,
                            projection: request.projection,
                            input: request.input,
                            output_written: request.output,
                            input_sha256: request.input_sha256,
                            output_sha256,
                            weight_sha256: request.weight.sha256,
                            scale_sha256: request.scale.sha256,
                            catalog_sha256: FULLDEPTH43_CATALOG_SHA256,
                            payload_hash_verified: true,
                            gpu_slot_cache_hit: execution.gpu_slot_cache_hit,
                            gpu_slot_cache_entries: execution.gpu_slot_cache_entries,
                            gpu_slot_resident_bytes: execution.gpu_slot_resident_bytes,
                            payload_uploaded_bytes: execution.payload_uploaded_bytes,
                            activation_uploaded_bytes: kernel
                                .input_elements()?
                                .checked_mul(4)
                                .ok_or_else(|| {
                                    anyhow::anyhow!("FullDepth43 request byte overflow")
                                })?,
                            numeric_mode: kernel.numeric_mode(),
                            output_rounding: "bf16_rne_then_f32_le",
                        },
                    ))
                }
                FullDepthFp8AttentionWorkerRequest::SharedInputBatch(request) => {
                    let [(first, first_kernel), (second, second_kernel)] =
                        validate_fulldepth_fp8_attention_batch_request(&request, expected_epoch)?;
                    for (item, kernel) in [(&first, first_kernel), (&second, second_kernel)] {
                        validate_fulldepth_fp8_attention_catalog(catalog, item, kernel)?;
                        validate_fulldepth_fp8_cache_proofs(catalog, item, &cache_root)?;
                    }
                    let arena = resolve_projection_batch_arena(
                        &request.input,
                        [&first.output, &second.output],
                    )?;
                    let input = read_fulldepth_fp8_input(
                        &arena,
                        &request.input,
                        &request.input_sha256,
                        first_kernel,
                    )?;

                    // Capacity is checked for both misses before either child is executed,
                    // so a batch cannot partially populate the resident cache and then
                    // discover that its second projection exceeds the frozen budget.
                    let mut projected_entries = standard_slots.len() + grouped_slots.len();
                    let mut projected_resident_bytes = gpu_slot_resident_bytes;
                    for (item, kernel) in [(&first, first_kernel), (&second, second_kernel)] {
                        let key = fulldepth_fp8_slot_key(kernel, item);
                        let hit = match kernel {
                            FullDepthFp8Kernel::Standard(_) => standard_slots.contains_key(&key),
                            FullDepthFp8Kernel::GroupedWoA(_) => grouped_slots.contains_key(&key),
                        };
                        if !hit {
                            projected_resident_bytes = reserve_fulldepth_fp8_gpu_slot(
                                projected_entries,
                                projected_resident_bytes,
                                item,
                            )?;
                            projected_entries =
                                projected_entries.checked_add(1).ok_or_else(|| {
                                    anyhow::anyhow!("FullDepth43 batch GPU slot count overflow")
                                })?;
                        }
                    }

                    let first_execution = execute_fulldepth_fp8_with_slot_cache(
                        ctx,
                        pipelines,
                        &mut payload_cache,
                        &cache_root,
                        &mut standard_slots,
                        &mut grouped_slots,
                        &mut gpu_slot_resident_bytes,
                        &first,
                        first_kernel,
                        &input,
                    )?;
                    let second_execution = execute_fulldepth_fp8_with_slot_cache(
                        ctx,
                        pipelines,
                        &mut payload_cache,
                        &cache_root,
                        &mut standard_slots,
                        &mut grouped_slots,
                        &mut gpu_slot_resident_bytes,
                        &second,
                        second_kernel,
                        &input,
                    )?;
                    let first_output_sha256 = write_fulldepth_fp8_output(
                        &arena,
                        &first.output,
                        &first_execution.output,
                        first_kernel,
                    )?;
                    let second_output_sha256 = write_fulldepth_fp8_output(
                        &arena,
                        &second.output,
                        &second_execution.output,
                        second_kernel,
                    )?;
                    let shared_activation_bytes = request.input.bytes;
                    let outputs = vec![
                        FullDepthFp8AttentionBatchItemResponse {
                            projection: first.projection,
                            output_written: first.output,
                            output_sha256: first_output_sha256,
                            weight_sha256: first.weight.sha256,
                            scale_sha256: first.scale.sha256,
                            payload_hash_verified: true,
                            gpu_slot_cache_hit: first_execution.gpu_slot_cache_hit,
                            gpu_slot_resident_bytes: first_execution.gpu_slot_resident_bytes,
                            payload_uploaded_bytes: first_execution.payload_uploaded_bytes,
                            numeric_mode: first_kernel.numeric_mode(),
                            output_rounding: "bf16_rne_then_f32_le",
                        },
                        FullDepthFp8AttentionBatchItemResponse {
                            projection: second.projection,
                            output_written: second.output,
                            output_sha256: second_output_sha256,
                            weight_sha256: second.weight.sha256,
                            scale_sha256: second.scale.sha256,
                            payload_hash_verified: true,
                            gpu_slot_cache_hit: second_execution.gpu_slot_cache_hit,
                            gpu_slot_resident_bytes: second_execution.gpu_slot_resident_bytes,
                            payload_uploaded_bytes: second_execution.payload_uploaded_bytes,
                            numeric_mode: second_kernel.numeric_mode(),
                            output_rounding: "bf16_rne_then_f32_le",
                        },
                    ];
                    Ok(FullDepthFp8AttentionWorkerResponse::SharedInputBatch(
                        FullDepthFp8AttentionBatchResponse {
                            protocol: FULLDEPTH_FP8_ATTENTION_PROTOCOL,
                            request_id: request.request_id,
                            ok: true,
                            revision: REVISION,
                            profile: "fulldepth43_native_top6",
                            layer: request.layer,
                            position: request.position,
                            arena_epoch: request.arena_epoch,
                            input: request.input,
                            input_sha256: request.input_sha256,
                            outputs,
                            catalog_sha256: FULLDEPTH43_CATALOG_SHA256,
                            gpu_slot_cache_entries: second_execution.gpu_slot_cache_entries,
                            activation_uploaded_bytes: shared_activation_bytes,
                        },
                    ))
                }
                FullDepthFp8AttentionWorkerRequest::OutputChain(request) => {
                    let [(wo_a, wo_a_kernel), (wo_b, wo_b_kernel)] =
                        validate_fulldepth_fp8_attention_output_chain_request(
                            &request,
                            expected_epoch,
                        )?;
                    for (stage, kernel) in [(&wo_a, wo_a_kernel), (&wo_b, wo_b_kernel)] {
                        validate_fulldepth_fp8_attention_catalog(catalog, stage, kernel)?;
                        validate_fulldepth_fp8_cache_proofs(catalog, stage, &cache_root)?;
                    }
                    let arena = resolve_projection_arena(&request.input, &request.output)?;
                    let input = read_fulldepth_fp8_input(
                        &arena,
                        &request.input,
                        &request.input_sha256,
                        wo_a_kernel,
                    )?;

                    // Reserve both misses before executing wo_a. A rejected wo_b slot
                    // must not leave a half-admitted output-chain resident on the GPU.
                    let mut projected_entries = standard_slots.len() + grouped_slots.len();
                    let mut projected_resident_bytes = gpu_slot_resident_bytes;
                    for (stage, kernel) in [(&wo_a, wo_a_kernel), (&wo_b, wo_b_kernel)] {
                        let key = fulldepth_fp8_slot_key(kernel, stage);
                        let hit = match kernel {
                            FullDepthFp8Kernel::Standard(_) => standard_slots.contains_key(&key),
                            FullDepthFp8Kernel::GroupedWoA(_) => grouped_slots.contains_key(&key),
                        };
                        if !hit {
                            projected_resident_bytes = reserve_fulldepth_fp8_gpu_slot(
                                projected_entries,
                                projected_resident_bytes,
                                stage,
                            )?;
                            projected_entries =
                                projected_entries.checked_add(1).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "FullDepth43 output-chain GPU slot count overflow"
                                    )
                                })?;
                        }
                    }

                    let wo_a_execution = execute_fulldepth_fp8_with_slot_cache(
                        ctx,
                        pipelines,
                        &mut payload_cache,
                        &cache_root,
                        &mut standard_slots,
                        &mut grouped_slots,
                        &mut gpu_slot_resident_bytes,
                        &wo_a,
                        wo_a_kernel,
                        &input,
                    )?;
                    let wo_a_output_sha256 = sha256_f32_le(&wo_a_execution.output);
                    let requantized = official_group128_e4m3fn_activation_quantize_dequantize(
                        &wo_a_execution.output,
                    )?;
                    let requantized_activation_sha256 = sha256_f32_le(&requantized);
                    let wo_b_execution = execute_fulldepth_fp8_with_slot_cache(
                        ctx,
                        pipelines,
                        &mut payload_cache,
                        &cache_root,
                        &mut standard_slots,
                        &mut grouped_slots,
                        &mut gpu_slot_resident_bytes,
                        &wo_b,
                        wo_b_kernel,
                        &requantized,
                    )?;
                    let output_sha256 = sha256_f32_le(&wo_b_execution.output);
                    validate_frozen_l42_output_chain_hashes(
                        request.layer,
                        &request.input_sha256,
                        &wo_a_output_sha256,
                        &requantized_activation_sha256,
                        &output_sha256,
                    )?;
                    let written_sha256 = write_fulldepth_fp8_output(
                        &arena,
                        &request.output,
                        &wo_b_execution.output,
                        wo_b_kernel,
                    )?;
                    if written_sha256 != output_sha256 {
                        bail!("FullDepth43 output-chain final write SHA-256 drift");
                    }

                    let slots = vec![
                        FullDepthFp8AttentionOutputChainSlotResponse {
                            projection: wo_a.projection,
                            weight_sha256: wo_a.weight.sha256,
                            scale_sha256: wo_a.scale.sha256,
                            payload_hash_verified: true,
                            gpu_slot_cache_hit: wo_a_execution.gpu_slot_cache_hit,
                            gpu_slot_cache_entries: wo_a_execution.gpu_slot_cache_entries,
                            gpu_slot_resident_bytes: wo_a_execution.gpu_slot_resident_bytes,
                            payload_uploaded_bytes: wo_a_execution.payload_uploaded_bytes,
                            activation_uploaded_bytes: wo_a_kernel
                                .input_elements()?
                                .checked_mul(4)
                                .ok_or_else(|| {
                                    anyhow::anyhow!("FullDepth43 wo_a activation byte overflow")
                                })?,
                            numeric_mode: wo_a_kernel.numeric_mode(),
                            output_rounding: "bf16_rne_then_f32_le",
                        },
                        FullDepthFp8AttentionOutputChainSlotResponse {
                            projection: wo_b.projection,
                            weight_sha256: wo_b.weight.sha256,
                            scale_sha256: wo_b.scale.sha256,
                            payload_hash_verified: true,
                            gpu_slot_cache_hit: wo_b_execution.gpu_slot_cache_hit,
                            gpu_slot_cache_entries: wo_b_execution.gpu_slot_cache_entries,
                            gpu_slot_resident_bytes: wo_b_execution.gpu_slot_resident_bytes,
                            payload_uploaded_bytes: wo_b_execution.payload_uploaded_bytes,
                            activation_uploaded_bytes: wo_b_kernel
                                .input_elements()?
                                .checked_mul(4)
                                .ok_or_else(|| {
                                    anyhow::anyhow!("FullDepth43 wo_b activation byte overflow")
                                })?,
                            numeric_mode: wo_b_kernel.numeric_mode(),
                            output_rounding: "bf16_rne_then_f32_le",
                        },
                    ];
                    Ok(FullDepthFp8AttentionWorkerResponse::OutputChain(
                        FullDepthFp8AttentionOutputChainResponse {
                            protocol: FULLDEPTH_FP8_ATTENTION_PROTOCOL,
                            request_id: request.request_id,
                            ok: true,
                            revision: REVISION,
                            profile: "fulldepth43_native_top6",
                            layer: request.layer,
                            position: request.position,
                            arena_epoch: request.arena_epoch,
                            input: request.input,
                            output_written: request.output,
                            input_sha256: request.input_sha256,
                            wo_a_output_sha256,
                            requantized_activation_sha256,
                            output_sha256,
                            requantization: request.requantization,
                            slots,
                            catalog_sha256: FULLDEPTH43_CATALOG_SHA256,
                            gpu_slot_cache_entries: wo_b_execution.gpu_slot_cache_entries,
                            numeric_mode: "grouped_wo_a_then_e4m3fn_group128_then_wo_b",
                            output_rounding: "bf16_rne_then_f32_le",
                        },
                    ))
                }
            }
        })();
        match response {
            Ok(response) => {
                serde_json::to_writer(&mut stdout, &response)?;
                stdout.write_all(b"\n")?;
                stdout.flush()?;
                expected_epoch = expected_epoch
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("FullDepth43 FP8 arena epoch overflow"))?;
            }
            Err(error) => {
                write_fulldepth_fp8_attention_error(&mut stdout, &request_id, &error)?;
                return Err(error.context("FullDepth43 FP8 worker poisoned"));
            }
        }
    }
    Ok(())
}

fn run_fulldepth_fp8_attention_worker() -> Result<()> {
    let catalog_path = Path::new(FULLDEPTH43_CATALOG_FILE);
    let catalog_bytes =
        std::fs::read(catalog_path).with_context(|| format!("read {}", catalog_path.display()))?;
    if sha256_bytes(&catalog_bytes) != FULLDEPTH43_CATALOG_SHA256 {
        bail!("FullDepth43 catalog SHA-256 drift");
    }
    let catalog: serde_json::Value = serde_json::from_slice(&catalog_bytes)?;
    let ctx = VulkanContext::init()?;
    let properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    if properties.vendor_id != AMD_VENDOR_ID || properties.device_id != NAVI10_DEVICE_ID {
        bail!(
            "FullDepth43 packed-FP8 worker requires RX 5700 XT; found 0x{:04x}:0x{:04x} ({})",
            properties.vendor_id,
            properties.device_id,
            ctx.gpu_name
        );
    }
    let pipelines = S14NumericPipelines::new_exact_audit(&ctx)?;
    let result = run_fulldepth_fp8_attention_loop(&ctx, &pipelines, &catalog);
    pipelines.destroy(&ctx);
    result
}

fn validate_l42_projection_fixture_manifest(
    document: &serde_json::Value,
    projection: FrozenFp8Projection,
) -> Result<()> {
    if document.get("format").and_then(serde_json::Value::as_str)
        != Some("polaris-l42-packed-fp8-projection-fixtures-v1")
        || document.get("repo").and_then(serde_json::Value::as_str)
            != Some("deepseek-ai/DeepSeek-V4-Flash-0731")
        || document.get("revision").and_then(serde_json::Value::as_str) != Some(REVISION)
        || document.get("layer").and_then(serde_json::Value::as_u64) != Some(42)
        || document
            .get("catalog_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(FULLDEPTH43_CATALOG_SHA256)
        || document
            .get("projection_count")
            .and_then(serde_json::Value::as_u64)
            != Some(L42_STANDARD_FP8_PROJECTIONS.len() as u64)
        || document
            .get("layer_output_sha256")
            .and_then(serde_json::Value::as_str)
            != Some("853b8b947a3f7a275cf748d7e97a311ebb22323cd0c2f3e5e973f27b04388895")
    {
        bail!("L42 packed-FP8 fixture manifest identity drift");
    }
    let entries = document
        .get("projections")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("L42 packed-FP8 fixture projections are missing"))?;
    if entries.len() != L42_STANDARD_FP8_PROJECTIONS.len() {
        bail!("L42 packed-FP8 fixture projection count drift");
    }
    let matches: Vec<&serde_json::Value> = entries
        .iter()
        .filter(|entry| {
            entry.get("projection").and_then(serde_json::Value::as_str) == Some(projection.name)
        })
        .collect();
    if matches.len() != 1 {
        bail!("fixture must contain exactly one {}", projection.name);
    }
    let entry = matches[0];
    let shape = S14MatvecShape::new(projection.n, projection.k)?.validate_fp8()?;
    let input = entry
        .get("input")
        .ok_or_else(|| anyhow::anyhow!("{} fixture input is missing", projection.name))?;
    let output = entry
        .get("output")
        .ok_or_else(|| anyhow::anyhow!("{} fixture output is missing", projection.name))?;
    let weight = entry
        .get("weight")
        .ok_or_else(|| anyhow::anyhow!("{} fixture weight is missing", projection.name))?;
    let scale = entry
        .get("scale")
        .ok_or_else(|| anyhow::anyhow!("{} fixture scale is missing", projection.name))?;
    let expected_input_shape = vec![1, 1, projection.k as u64];
    let expected_output_shape = vec![1, 1, projection.n as u64];
    let expected_weight_shape = vec![projection.n as u64, projection.k as u64];
    let expected_scale_shape = vec![(projection.n / 128) as u64, (projection.k / 128) as u64];
    let parse_shape = |value: &serde_json::Value| -> Option<Vec<u64>> {
        value
            .get("shape")?
            .as_array()?
            .iter()
            .map(serde_json::Value::as_u64)
            .collect()
    };
    if entry.get("n").and_then(serde_json::Value::as_u64) != Some(projection.n as u64)
        || entry.get("k").and_then(serde_json::Value::as_u64) != Some(projection.k as u64)
        || entry
            .get("activation_contract")
            .and_then(serde_json::Value::as_str)
            != Some("cpu_e4m3fn_quant_dequant_f32")
        || entry
            .get("output_rounding")
            .and_then(serde_json::Value::as_str)
            != Some("bf16_rne_then_f32_le")
        || input.get("bytes").and_then(serde_json::Value::as_u64) != Some(shape.fp32_input_bytes()?)
        || input.get("sha256").and_then(serde_json::Value::as_str) != Some(projection.input_sha256)
        || parse_shape(input).as_deref() != Some(expected_input_shape.as_slice())
        || output.get("bytes").and_then(serde_json::Value::as_u64)
            != Some(shape.fp32_output_bytes()?)
        || output.get("sha256").and_then(serde_json::Value::as_str)
            != Some(projection.output_sha256)
        || parse_shape(output).as_deref() != Some(expected_output_shape.as_slice())
        || weight.get("bytes").and_then(serde_json::Value::as_u64)
            != Some(shape.fp8_weight_bytes()?)
        || weight.get("sha256").and_then(serde_json::Value::as_str)
            != Some(projection.weight_sha256)
        || parse_shape(weight).as_deref() != Some(expected_weight_shape.as_slice())
        || scale.get("bytes").and_then(serde_json::Value::as_u64) != Some(shape.fp8_scale_bytes()?)
        || scale.get("sha256").and_then(serde_json::Value::as_str) != Some(projection.scale_sha256)
        || parse_shape(scale).as_deref() != Some(expected_scale_shape.as_slice())
    {
        bail!("{} fixture tensor/SHA contract drift", projection.name);
    }
    Ok(())
}

fn read_l42_projection_fixture(
    root: &Path,
    projection: FrozenFp8Projection,
    kind: &str,
    element_count: usize,
    expected_sha256: &str,
) -> Result<Vec<f32>> {
    if kind != "input" && kind != "output.bf16-f32le" {
        bail!("unsupported L42 projection fixture kind");
    }
    let filename = if kind == "input" {
        format!("{}.input.f32le.bin", projection.file_stem)
    } else {
        format!("{}.output.bf16-f32le.bin", projection.file_stem)
    };
    let path = root.join(filename);
    let resolved = path
        .canonicalize()
        .with_context(|| format!("resolve {} fixture", projection.name))?;
    if resolved.parent() != Some(root)
        || resolved.extension().and_then(|value| value.to_str()) != Some("bin")
    {
        bail!("{} fixture path escaped frozen directory", projection.name);
    }
    let payload = std::fs::read(&resolved)?;
    if payload.len() != element_count * std::mem::size_of::<f32>()
        || sha256_bytes(&payload) != expected_sha256
    {
        bail!("{} {kind} fixture byte/SHA drift", projection.name);
    }
    let values: Vec<f32> = payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if values.len() != element_count || values.iter().any(|value| !value.is_finite()) {
        bail!("{} {kind} fixture shape/non-finite drift", projection.name);
    }
    Ok(values)
}

fn f32_le_sha256(values: &[f32]) -> String {
    let mut payload = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    sha256_bytes(&payload)
}

fn run_l42_standard_fp8_projection_suite() -> Result<()> {
    let root = PathBuf::from(
        std::env::var_os(FP8_PROJECTION_FIXTURE_DIR_ENV)
            .ok_or_else(|| anyhow::anyhow!("{FP8_PROJECTION_FIXTURE_DIR_ENV} is required"))?,
    )
    .canonicalize()?;
    if !root.is_dir() {
        bail!("L42 packed-FP8 fixture root is not a directory");
    }
    let manifest_path = root.join("manifest.json");
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
    for projection in L42_STANDARD_FP8_PROJECTIONS {
        validate_l42_projection_fixture_manifest(&manifest, projection)?;
    }

    let ctx = VulkanContext::init()?;
    let properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    if properties.vendor_id != AMD_VENDOR_ID || properties.device_id != NAVI10_DEVICE_ID {
        bail!("L42 FP8 suite requires RX 5700 XT");
    }
    let pipelines = S14NumericPipelines::new_exact_audit(&ctx)?;
    let result = (|| -> Result<Vec<serde_json::Value>> {
        let mut reports = Vec::with_capacity(L42_STANDARD_FP8_PROJECTIONS.len());
        for projection in L42_STANDARD_FP8_PROJECTIONS {
            let shape = S14MatvecShape::new(projection.n, projection.k)?.validate_fp8()?;
            let input = read_l42_projection_fixture(
                &root,
                projection,
                "input",
                projection.k as usize,
                projection.input_sha256,
            )?;
            let expected_output = read_l42_projection_fixture(
                &root,
                projection,
                "output.bf16-f32le",
                projection.n as usize,
                projection.output_sha256,
            )?;
            let load_started = Instant::now();
            let (weight, scale) = load_l42_projection_assets(projection)?;
            let slot = PersistentFp8ProjectionSlot::new(&ctx, &pipelines, shape, &weight, &scale)?;
            let load_and_upload_ms = load_started.elapsed().as_secs_f64() * 1000.0;
            let execute_started = Instant::now();
            let output = slot.execute(&pipelines, &input)?;
            let execute_ms = execute_started.elapsed().as_secs_f64() * 1000.0;
            let output_sha256 = f32_le_sha256(&output);
            if output_sha256 != projection.output_sha256 || output != expected_output {
                let mismatch_count = output
                    .iter()
                    .zip(&expected_output)
                    .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
                    .count();
                let first_mismatch = output
                    .iter()
                    .zip(&expected_output)
                    .enumerate()
                    .find(|(_, (actual, expected))| actual.to_bits() != expected.to_bits())
                    .map(|(index, (actual, expected))| {
                        format!(
                            "index={index}, actual={actual:?} (0x{:08x}), expected={expected:?} (0x{:08x})",
                            actual.to_bits(),
                            expected.to_bits()
                        )
                    })
                    .unwrap_or_else(|| "none".to_string());
                if std::env::var_os("POLARIS_L42_FP8_DUMP_DRIFT").is_some() {
                    let mut payload = Vec::with_capacity(output.len() * 4);
                    for value in &output {
                        payload.extend_from_slice(&value.to_le_bytes());
                    }
                    std::fs::write(
                        root.join(format!(
                            "{}.gpu-actual.bf16-f32le.bin",
                            projection.file_stem
                        )),
                        payload,
                    )?;
                }
                bail!(
                    "{} GPU BF16 output drift: mismatch_count={mismatch_count}/{}, actual_sha256={output_sha256}, expected_sha256={}, first_mismatch=({first_mismatch})",
                    projection.name,
                    output.len(),
                    projection.output_sha256,
                );
            }
            drop(slot);
            reports.push(serde_json::json!({
                "projection": projection.name,
                "n": projection.n,
                "k": projection.k,
                "weight_sha256": projection.weight_sha256,
                "scale_sha256": projection.scale_sha256,
                "input_sha256": projection.input_sha256,
                "output_sha256": output_sha256,
                "bf16_elements_exact": projection.n,
                "load_verify_upload_ms": load_and_upload_ms,
                "execute_readback_round_verify_ms": execute_ms,
            }));
        }
        Ok(reports)
    })();
    pipelines.destroy(&ctx);
    let reports = result?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "format": "polaris-l42-standard-packed-fp8-vulkan-suite-v1",
            "status": "complete",
            "device": ctx.gpu_name,
            "revision": REVISION,
            "numeric_mode": "packed_fp8_e4m3_ue8m0_exact_audit",
            "projection_count": reports.len(),
            "projections": reports,
            "claim_limit": "L42 standard projection exact suite; wo_a grouped and full-token integration remain unproven",
        }))?
    );
    Ok(())
}

fn capture_shape(value: &serde_json::Value) -> Option<Vec<u64>> {
    value
        .get("shape")?
        .as_array()?
        .iter()
        .map(serde_json::Value::as_u64)
        .collect()
}

fn require_l42_wo_a_capture_entry(
    document: &serde_json::Value,
    name: &str,
    file: &str,
    shape: &[u64],
    bytes: u64,
    sha256: &str,
) -> Result<()> {
    let entries = document
        .get("inputs")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("L42 wo_a capture inputs are missing"))?;
    let matches: Vec<&serde_json::Value> = entries
        .iter()
        .filter(|entry| entry.get("name").and_then(serde_json::Value::as_str) == Some(name))
        .collect();
    if matches.len() != 1 {
        bail!("L42 wo_a capture must contain exactly one {name}");
    }
    let entry = matches[0];
    if entry.get("file").and_then(serde_json::Value::as_str) != Some(file)
        || entry.get("bytes").and_then(serde_json::Value::as_u64) != Some(bytes)
        || entry
            .get("f32_le_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(sha256)
        || capture_shape(entry).as_deref() != Some(shape)
    {
        bail!("L42 wo_a capture {name} file/shape/byte/SHA drift");
    }
    Ok(())
}

fn validate_l42_wo_a_capture_manifest(document: &serde_json::Value) -> Result<()> {
    if document.get("format").and_then(serde_json::Value::as_str)
        != Some("polaris-l42-real-vulkan-input-capture-v1")
        || document.get("repo").and_then(serde_json::Value::as_str)
            != Some("deepseek-ai/DeepSeek-V4-Flash-0731")
        || document.get("revision").and_then(serde_json::Value::as_str) != Some(REVISION)
        || document.get("layer").and_then(serde_json::Value::as_u64) != Some(42)
        || document.get("expert_id").and_then(serde_json::Value::as_u64)
            != Some(REAL_EXPERT_ID as u64)
        || document
            .pointer("/source_f32_le_sha256/ffn_input")
            .and_then(serde_json::Value::as_str)
            != Some("7e2d3167e3782eca8d762c3cc92d53bb9d64a65c7b18d37d16797ff39f611ad4")
        || document
            .pointer("/asset_integrity/hashes_checked")
            .and_then(serde_json::Value::as_u64)
            != Some(76)
        || document
            .pointer("/asset_integrity/payload_bytes")
            .and_then(serde_json::Value::as_u64)
            != Some(247_515_224)
        || document
            .pointer("/asset_integrity/payload_files")
            .and_then(serde_json::Value::as_u64)
            != Some(76)
        || document
            .pointer("/asset_integrity/manifest_sha256/base")
            .and_then(serde_json::Value::as_str)
            != Some("5e86fa2145c1e3b5f4f8efdcd4fdcca5b966d7cb43ca0ea592294a542ac086ed")
        || document
            .pointer("/asset_integrity/manifest_sha256/route")
            .and_then(serde_json::Value::as_str)
            != Some("feccc1b5dde256c9ad750985b4ab5446732a1f4deec796976198e95798c64b86")
        || document
            .pointer("/asset_integrity/manifest_sha256/s14")
            .and_then(serde_json::Value::as_str)
            != Some("f6ea01a0df591f272d4e5addb96e16e8eae79b6e9326a84ef6cbaab818c6a2b5")
        || document
            .get("inputs")
            .and_then(serde_json::Value::as_array)
            .map(Vec::len)
            != Some(5)
        || document.get("semantics").and_then(serde_json::Value::as_str)
            != Some(
                "Inputs captured from the hash-verified real L42 inline reference after the official UE8M0/E4M3FN activation quantization step.",
            )
    {
        bail!("L42 wo_a capture manifest identity/integrity drift");
    }
    require_l42_wo_a_capture_entry(
        document,
        "wq_a",
        "wq_a.f32le.bin",
        &[1, 1, 4096],
        16_384,
        L42_WQ_A_INPUT_SHA256,
    )?;
    require_l42_wo_a_capture_entry(
        document,
        "wo_a_grouped_input_bf16",
        "wo_a_grouped_input_bf16.f32le.bin",
        &[1, 1, 8, 4096],
        131_072,
        L42_WO_A_INPUT_SHA256,
    )?;
    require_l42_wo_a_capture_entry(
        document,
        "wo_a_grouped_output_bf16",
        "wo_a_grouped_output_bf16.f32le.bin",
        &[1, 1, 8192],
        32_768,
        L42_WO_A_OUTPUT_SHA256,
    )?;
    require_l42_wo_a_capture_entry(
        document,
        "expert_126_w1_w3",
        "expert_126_w1_w3.f32le.bin",
        &[1, 1, 4096],
        16_384,
        "1a2006c1b79b31bc3db3540ef730b08547c6148e224074b0ba712e2c3b9d7c9f",
    )?;
    require_l42_wo_a_capture_entry(
        document,
        "expert_126_w2",
        "expert_126_w2.f32le.bin",
        &[1, 1, 2048],
        8_192,
        "7795b1df61092c44883579a0107d02e693303fd16e926fcbd1baa154b49a9a31",
    )?;
    Ok(())
}

fn read_l42_wo_a_capture_f32(
    root: &Path,
    filename: &str,
    element_count: usize,
    expected_sha256: &str,
) -> Result<Vec<f32>> {
    let relative = Path::new(filename);
    if relative.components().count() != 1
        || relative.extension().and_then(|value| value.to_str()) != Some("bin")
    {
        bail!("L42 wo_a capture filename contract drift");
    }
    let path = root.join(relative).canonicalize()?;
    if path.parent() != Some(root) {
        bail!("L42 wo_a capture payload escaped fixture directory");
    }
    let payload = std::fs::read(&path)?;
    if payload.len() != element_count * std::mem::size_of::<f32>()
        || sha256_bytes(&payload) != expected_sha256
    {
        bail!("L42 wo_a capture payload byte/SHA drift");
    }
    let values: Vec<f32> = payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if values.len() != element_count
        || values.iter().any(|value| !value.is_finite())
        || values.iter().any(|value| value.to_bits() & 0xffff != 0)
    {
        bail!("L42 wo_a capture payload is not finite BF16-rounded F32");
    }
    Ok(values)
}

fn execute_l42_wo_a_grouped(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    input: &[f32],
    weight: &[u8],
    scale: &[u8],
    expected_output: &[f32],
) -> Result<serde_json::Value> {
    let shape = S14GroupedMatvecShape::new(8, 1024, 4096)?.validate_fp8_bf16_weight()?;
    if input.len() * 4 != shape.fp32_input_bytes()? as usize
        || weight.len() != shape.fp8_weight_bytes()? as usize
        || scale.len() != shape.fp8_scale_bytes()? as usize
        || expected_output.len() * 4 != shape.fp32_output_bytes()? as usize
    {
        bail!("L42 wo_a grouped tensor byte contract drift");
    }
    validate_e4m3fn_codes(weight)?;
    validate_ue8m0_codes(scale)?;
    let buffers = DeviceBuffers::new(ctx, input, weight, scale, shape.flat_n()? as usize)?;
    let result = (|| -> Result<serde_json::Value> {
        let upload_started = Instant::now();
        buffers.upload(ctx)?;
        let upload_ms = upload_started.elapsed().as_secs_f64() * 1000.0;
        let dispatch = pipelines.bind_grouped_fp8_bf16_weight(
            ctx,
            shape,
            &buffers.x,
            &buffers.weight,
            &buffers.scale,
            &buffers.y,
        )?;
        let execution = (|| -> Result<serde_json::Value> {
            let timing = benchmark(
                ctx,
                &buffers,
                timestamp_bits,
                timestamp_period_ns,
                |cb| unsafe { pipelines.cmd_grouped_fp8_bf16_weight_matvec(ctx, cb, &dispatch) },
            )?;
            let output: Vec<f32> = buffers
                .output(shape.flat_n()? as usize)
                .into_iter()
                .map(round_f32_to_bf16_f32)
                .collect();
            let output_sha256 = f32_le_sha256(&output);
            let exact_elements = output
                .iter()
                .zip(expected_output)
                .filter(|(actual, expected)| actual.to_bits() == expected.to_bits())
                .count();
            if output_sha256 != L42_WO_A_OUTPUT_SHA256 || exact_elements != expected_output.len() {
                let first_mismatch = output
                    .iter()
                    .zip(expected_output)
                    .enumerate()
                    .find(|(_, (actual, expected))| actual.to_bits() != expected.to_bits())
                    .map(|(index, (actual, expected))| {
                        format!(
                            "index={index}, actual={actual:?}/0x{:08x}, expected={expected:?}/0x{:08x}",
                            actual.to_bits(),
                            expected.to_bits()
                        )
                    })
                    .unwrap_or_else(|| "none".to_string());
                bail!(
                    "L42 wo_a grouped BF16 output drift: exact={exact_elements}/{}, actual_sha256={output_sha256}, expected_sha256={}, first_mismatch=({first_mismatch})",
                    expected_output.len(),
                    L42_WO_A_OUTPUT_SHA256,
                );
            }
            Ok(serde_json::json!({
                "groups": shape.groups,
                "n_per_group": shape.n_per_group,
                "k": shape.k,
                "input_elements": input.len(),
                "output_elements": output.len(),
                "bf16_elements_exact": exact_elements,
                "input_sha256": L42_WO_A_INPUT_SHA256,
                "weight_sha256": L42_WO_A_WEIGHT_SHA256,
                "scale_sha256": L42_WO_A_SCALE_SHA256,
                "output_sha256": output_sha256,
                "upload_ms": upload_ms,
                "iterations": timing.iterations,
                "gpu_kernel_ms_mean": timing.gpu_kernel_ms_mean,
                "submit_readback_sync_ms": timing.submit_readback_sync_ms,
            }))
        })();
        dispatch.binder.destroy(ctx);
        execution
    })();
    buffers.destroy(ctx);
    result
}

fn run_l42_wo_a_grouped_suite() -> Result<()> {
    let suite_started = Instant::now();
    let root = PathBuf::from(
        std::env::var_os(WO_A_GROUPED_FIXTURE_DIR_ENV)
            .ok_or_else(|| anyhow::anyhow!("{WO_A_GROUPED_FIXTURE_DIR_ENV} is required"))?,
    )
    .canonicalize()?;
    if !root.is_dir() {
        bail!("L42 wo_a fixture root is not a directory");
    }
    let manifest_path = root.join("capture_manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path)?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    if manifest_sha256 != L42_WO_A_CAPTURE_MANIFEST_SHA256 {
        bail!("L42 wo_a capture manifest SHA-256 drift");
    }
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)?;
    validate_l42_wo_a_capture_manifest(&manifest)?;
    let input = read_l42_wo_a_capture_f32(
        &root,
        "wo_a_grouped_input_bf16.f32le.bin",
        8 * 4096,
        L42_WO_A_INPUT_SHA256,
    )?;
    let expected_output = read_l42_wo_a_capture_f32(
        &root,
        "wo_a_grouped_output_bf16.f32le.bin",
        8192,
        L42_WO_A_OUTPUT_SHA256,
    )?;
    let load_started = Instant::now();
    let (weight, scale) = load_l42_projection_assets(L42_WO_A_GROUPED_PROJECTION)?;
    let load_verify_ms = load_started.elapsed().as_secs_f64() * 1000.0;

    let ctx = VulkanContext::init()?;
    let properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    if properties.vendor_id != AMD_VENDOR_ID || properties.device_id != NAVI10_DEVICE_ID {
        bail!("L42 wo_a grouped suite requires RX 5700 XT");
    }
    let queue_properties = unsafe {
        ctx.instance
            .get_physical_device_queue_family_properties(ctx.physical)
    };
    let timestamp_bits = queue_properties[ctx.qf_graphics as usize].timestamp_valid_bits;
    if timestamp_bits == 0 {
        bail!("L42 wo_a grouped suite requires timestamp queries");
    }
    let timestamp_period_ns = properties.limits.timestamp_period as f64;
    let pipelines = S14NumericPipelines::new_exact_audit(&ctx)?;
    let result = execute_l42_wo_a_grouped(
        &ctx,
        &pipelines,
        timestamp_bits,
        timestamp_period_ns,
        &input,
        &weight,
        &scale,
        &expected_output,
    );
    pipelines.destroy(&ctx);
    let mut result = result?;
    result["load_verify_ms"] = serde_json::json!(load_verify_ms);
    result["suite_wall_ms"] = serde_json::json!(suite_started.elapsed().as_secs_f64() * 1000.0);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "format": "polaris-l42-wo-a-grouped-packed-fp8-vulkan-suite-v1",
            "status": "complete",
            "device": ctx.gpu_name,
            "revision": REVISION,
            "fixture_manifest_sha256": manifest_sha256,
            "numeric_mode": "grouped_packed_fp8_e4m3_ue8m0_bf16_weight_exact_audit",
            "result": result,
            "claim_limit": "L42 wo_a grouped exact numerical suite; full attention/token integration remains separate",
        }))?
    );
    Ok(())
}

fn main() -> Result<()> {
    if std::env::args().any(|value| value == FULLDEPTH_FP8_ATTENTION_WORKER_ARG) {
        return run_fulldepth_fp8_attention_worker();
    }
    if std::env::args().any(|value| value == WO_A_GROUPED_SUITE_ARG) {
        return run_l42_wo_a_grouped_suite();
    }
    if std::env::args().any(|value| value == FP8_PROJECTION_SUITE_ARG) {
        return run_l42_standard_fp8_projection_suite();
    }
    if std::env::args().any(|value| value == FP8_PROJECTION_WORKER_ARG) {
        return run_l42_wq_a_fp8_projection_worker();
    }
    if std::env::args().any(|value| value == WRITEBACK_WORKER_ARG) {
        return run_fulldepth_writeback_worker(true);
    }
    if std::env::args().any(|value| value == PRODUCTION_WORKER_ARG) {
        return run_fulldepth_writeback_worker(false);
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

fn run_fulldepth_writeback_worker(exact_audit: bool) -> Result<()> {
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
    let pipelines = if exact_audit {
        S14NumericPipelines::new_exact_audit(&ctx)?
    } else {
        S14NumericPipelines::new(&ctx)?
    };
    let numeric_mode = if exact_audit {
        "exact_audit_12_lane_mxfp4_8_lane_fp8"
    } else {
        "fast_production_128_lane"
    };
    let payload_cache_capacity = payload_cache_capacity_bytes()?;
    let mut payload_cache = VerifiedPayloadCache::new(payload_cache_capacity)?;
    let gpu_payload_cache_capacity = gpu_payload_cache_capacity_bytes()?;
    let mut gpu_payload_cache = gpu_payload_cache_capacity
        .map(GpuPayloadCache::new)
        .transpose()?;
    // The two GPU weight modes are deliberately disjoint. Default operation
    // owns one bounded upload slot; explicit resident-cache operation owns no
    // reusable slot, so stale slot contents can never masquerade as a hit.
    let reusable_gpu_slot = if gpu_payload_cache.is_none() {
        Some(ReusableOfficialMoeSlot::new(&ctx)?)
    } else {
        None
    };
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
            "gpu_payload_cache": gpu_payload_cache.is_some(),
            "gpu_payload_cache_capacity_bytes": gpu_payload_cache_capacity.unwrap_or(0),
            "reusable_gpu_upload_slot": reusable_gpu_slot.is_some(),
            "reusable_gpu_upload_slot_device_bytes": reusable_gpu_slot
                .as_ref()
                .map(|slot| slot.logical_device_bytes)
                .unwrap_or(0),
            "reusable_gpu_upload_slot_staging_bytes": reusable_gpu_slot
                .as_ref()
                .map(|slot| slot.logical_staging_bytes)
                .unwrap_or(0),
            "gpu_vram_hard_limit_bytes": GPU_VRAM_HARD_LIMIT_GIB as u64 * 1024 * 1024 * 1024,
            "gpu_payload_identity": "tensor+bytes+sha256",
            "numeric_mode": numeric_mode,
            "production_default_shader_unchanged": true,
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
                if let Some(cache) = gpu_payload_cache.as_mut() {
                    cache.destroy(&ctx);
                }
                if let Some(slot) = reusable_gpu_slot.as_ref() {
                    slot.destroy(&ctx);
                }
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
            if let Some(cache) = gpu_payload_cache.as_mut() {
                cache.destroy(&ctx);
            }
            if let Some(slot) = reusable_gpu_slot.as_ref() {
                slot.destroy(&ctx);
            }
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
            &mut gpu_payload_cache,
            reusable_gpu_slot.as_ref(),
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
                if let Some(cache) = gpu_payload_cache.as_mut() {
                    cache.destroy(&ctx);
                }
                if let Some(slot) = reusable_gpu_slot.as_ref() {
                    slot.destroy(&ctx);
                }
                pipelines.destroy(&ctx);
                return Err(error.context("writeback worker poisoned"));
            }
        }
    }
    if let Some(cache) = gpu_payload_cache.as_mut() {
        cache.destroy(&ctx);
    }
    if let Some(slot) = reusable_gpu_slot.as_ref() {
        slot.destroy(&ctx);
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
    gpu_payload_cache: &mut Option<GpuPayloadCache>,
    reusable_gpu_slot: Option<&ReusableOfficialMoeSlot>,
    request: WritebackRequest,
) -> Result<WritebackResponse> {
    let started = Instant::now();
    let cache_before = payload_cache.stats();
    let gpu_cache_before = gpu_payload_cache.as_ref().map(GpuPayloadCache::stats);
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
                gpu_identity: Some(gpu_moe_identity(&manifest, Some(expert_id))?),
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
        gpu_identity: Some(gpu_moe_identity(&manifest, None)?),
        w1,
        s1,
        w3,
        s3,
        w2,
        s2,
    };
    let diagnostic_dir = std::env::var_os(WRITEBACK_DIAGNOSTIC_DIR_ENV).map(PathBuf::from);
    let result = run_official_top6_shared_moe_batch(
        ctx,
        pipelines,
        timestamp_bits,
        timestamp_period_ns,
        &input,
        &routed,
        &shared,
        gpu_payload_cache.as_mut(),
        reusable_gpu_slot,
        diagnostic_dir.is_some(),
    )?;
    if let (Some(root), Some(diagnostics)) = (diagnostic_dir, result.diagnostics.as_ref()) {
        std::fs::create_dir_all(&root)?;
        for expert in diagnostics.routed.iter().chain([&diagnostics.shared]) {
            let label = match expert.expert_id {
                Some(expert_id) => format!("routed_e{expert_id}"),
                None => "shared".to_string(),
            };
            for (stage, values) in [
                ("gate", expert.gate.as_slice()),
                ("up", expert.up.as_slice()),
                ("hidden", expert.hidden.as_slice()),
                ("down", expert.down.as_slice()),
            ] {
                let path = root.join(format!("{label}_{stage}_gpu.f32le.bin"));
                if path.exists() {
                    bail!(
                        "refuse to overwrite writeback diagnostic {}",
                        path.display()
                    );
                }
                std::fs::write(path, bytemuck::cast_slice(values))?;
            }
        }
        let accumulator_path = root.join("accumulator_gpu.f32le.bin");
        if accumulator_path.exists() {
            bail!(
                "refuse to overwrite writeback diagnostic {}",
                accumulator_path.display()
            );
        }
        std::fs::write(
            accumulator_path,
            bytemuck::cast_slice(&diagnostics.accumulator),
        )?;
    }
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
    let gpu_payload_cache_telemetry = match (gpu_payload_cache.as_ref(), gpu_cache_before) {
        (Some(cache), Some(before)) => {
            WritebackGpuPayloadCacheTelemetry::between(cache, before, cache.stats())?
        }
        (None, None) => WritebackGpuPayloadCacheTelemetry::disabled(),
        _ => bail!("GPU payload cache enablement changed within one request"),
    };
    let reusable_gpu_slot_telemetry = WritebackReusableGpuSlotTelemetry::for_successful_request(
        reusable_gpu_slot.map(|slot| (slot.logical_device_bytes, slot.logical_staging_bytes)),
        gpu_payload_cache.is_some(),
    )?;
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
        gpu_payload_cache: gpu_payload_cache_telemetry,
        reusable_gpu_slot: reusable_gpu_slot_telemetry,
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

fn gpu_payload_cache_capacity_bytes() -> Result<Option<u64>> {
    let gib = match std::env::var(GPU_PAYLOAD_CACHE_GIB_ENV) {
        Ok(value) => value
            .parse::<usize>()
            .with_context(|| format!("parse {GPU_PAYLOAD_CACHE_GIB_ENV}={value:?}"))?,
        Err(std::env::VarError::NotPresent) => DEFAULT_GPU_PAYLOAD_CACHE_GIB,
        Err(error) => return Err(error).context(GPU_PAYLOAD_CACHE_GIB_ENV),
    };
    if gib == 0 {
        return Ok(None);
    }
    if !(MIN_GPU_PAYLOAD_CACHE_GIB..=MAX_GPU_PAYLOAD_CACHE_GIB).contains(&gib) {
        bail!(
            "{GPU_PAYLOAD_CACHE_GIB_ENV} must be between {MIN_GPU_PAYLOAD_CACHE_GIB} and {MAX_GPU_PAYLOAD_CACHE_GIB} GiB; one GiB remains reserved under the 8 GiB VRAM hard limit"
        );
    }
    let bytes = (gib as u64)
        .checked_mul(1024 * 1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("GPU payload cache byte capacity overflow"))?;
    Ok(Some(bytes))
}

fn validate_writeback_manifest(manifest: &FullDepthBridgeManifest) -> Result<()> {
    let expected_prefix: Vec<u32> = (0..manifest.layer).collect();
    if manifest.format != "polaris-fulldepth43-vulkan-bridge-capture-v1"
        || manifest.revision != REVISION
        || manifest.profile != "fulldepth43_native_top6"
        || manifest.layer > 42
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

fn official_branch_bf16(output: &[f32]) -> Result<Vec<u16>> {
    if output.len() != 4096 {
        bail!("official Vulkan MoE output shape drift");
    }
    if output.iter().any(|value| !value.is_finite()) {
        bail!("official Vulkan MoE output contains non-finite values");
    }
    Ok(output.iter().copied().map(f32_to_bf16_bits).collect())
}

fn run_official_top6_shared_moe_batch(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    x: &[f32],
    routed: &[MoePayload],
    shared: &MoePayload,
    gpu_payload_cache: Option<&mut GpuPayloadCache>,
    reusable_gpu_slot: Option<&ReusableOfficialMoeSlot>,
    capture_shared_diagnostics: bool,
) -> Result<OfficialMoeResult> {
    match (gpu_payload_cache, reusable_gpu_slot) {
        (Some(cache), None) => run_official_top6_shared_moe_batch_cached(
            ctx,
            pipelines,
            timestamp_bits,
            timestamp_period_ns,
            x,
            routed,
            shared,
            cache,
            capture_shared_diagnostics,
        ),
        (None, Some(slot)) => run_official_top6_shared_moe_batch_reusable(
            ctx,
            pipelines,
            timestamp_bits,
            timestamp_period_ns,
            x,
            routed,
            shared,
            slot,
            capture_shared_diagnostics,
        ),
        _ => bail!("GPU resident cache and reusable upload slot mode drift"),
    }
}

fn run_official_top6_shared_moe_batch_cached(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    x: &[f32],
    routed: &[MoePayload],
    shared: &MoePayload,
    gpu_payload_cache: &mut GpuPayloadCache,
    capture_shared_diagnostics: bool,
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

    for payload in routed {
        gpu_payload_cache.ensure(ctx, payload)?;
    }
    gpu_payload_cache.ensure(ctx, shared)?;
    let routed_weights = routed
        .iter()
        .map(|payload| {
            let identity = payload
                .gpu_identity
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("routed GPU payload lost strict SHA identity"))?;
            gpu_payload_cache.get(identity)
        })
        .collect::<Result<Vec<_>>>()?;
    let shared_identity = shared
        .gpu_identity
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("shared GPU payload lost strict SHA identity"))?;
    let shared_weights = gpu_payload_cache.get(shared_identity)?;

    let buffers = OfficialMoeWorkspace::new(ctx, x)?;
    buffers.upload_activation(ctx)?;
    let routed_dispatches = routed_weights
        .into_iter()
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
            &shared_weights.w1.device,
            &shared_weights.s1.device,
            &buffers.gate,
        )?,
        w3: pipelines.bind_fp8(
            ctx,
            shared_up_shape,
            &buffers.x.device,
            &shared_weights.w3.device,
            &shared_weights.s3.device,
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
            &shared_weights.w2.device,
            &shared_weights.s2.device,
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
        &buffers.accumulator,
        &buffers.readback,
        timestamp_bits,
        timestamp_period_ns,
        &routed_dispatches,
        &shared_dispatch,
    )?;
    let output = buffers.output();
    let branch_bf16 = official_branch_bf16(&output)?;
    let diagnostics = if capture_shared_diagnostics {
        let shared = ExpertDiagnostics {
            expert_id: None,
            gate: read_device_f32(ctx, &buffers.gate, &buffers.readback, 2048)?,
            up: read_device_f32(ctx, &buffers.up, &buffers.readback, 2048)?,
            hidden: read_device_f32(ctx, &buffers.hidden, &buffers.readback, 2048)?,
            down: read_device_f32(ctx, &buffers.down, &buffers.readback, 4096)?,
        };
        let mut routed_diagnostics = Vec::with_capacity(routed_dispatches.len());
        for (dispatch, payload) in routed_dispatches.iter().zip(routed) {
            record_routed_diagnostic_once(ctx, pipelines, dispatch)?;
            routed_diagnostics.push(ExpertDiagnostics {
                expert_id: payload.expert_id,
                gate: read_device_f32(ctx, &buffers.gate, &buffers.readback, 2048)?,
                up: read_device_f32(ctx, &buffers.up, &buffers.readback, 2048)?,
                hidden: read_device_f32(ctx, &buffers.hidden, &buffers.readback, 2048)?,
                down: read_device_f32(ctx, &buffers.down, &buffers.readback, 4096)?,
            });
        }
        Some(OfficialMoeDiagnostics {
            routed: routed_diagnostics,
            shared,
            accumulator: output,
        })
    } else {
        None
    };

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
        diagnostics,
    })
}

fn run_official_top6_shared_moe_batch_reusable(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    x: &[f32],
    routed: &[MoePayload],
    shared: &MoePayload,
    buffers: &ReusableOfficialMoeSlot,
    capture_shared_diagnostics: bool,
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

    // This is the production default. The fixed slot survives, but every byte
    // of activation and weight staging is rewritten and uploaded on every
    // request. It therefore removes allocator churn without becoming a
    // resident payload cache or retaining layer identity.
    buffers.rewrite_and_upload(ctx, x, routed, shared)?;
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
        &buffers.accumulator,
        &buffers.readback,
        timestamp_bits,
        timestamp_period_ns,
        &routed_dispatches,
        &shared_dispatch,
    )?;
    let output = buffers.output();
    let branch_bf16 = official_branch_bf16(&output)?;
    let diagnostics = if capture_shared_diagnostics {
        let shared = ExpertDiagnostics {
            expert_id: None,
            gate: read_device_f32(ctx, &buffers.gate, &buffers.readback, 2048)?,
            up: read_device_f32(ctx, &buffers.up, &buffers.readback, 2048)?,
            hidden: read_device_f32(ctx, &buffers.hidden, &buffers.readback, 2048)?,
            down: read_device_f32(ctx, &buffers.down, &buffers.readback, 4096)?,
        };
        let mut routed_diagnostics = Vec::with_capacity(routed_dispatches.len());
        for (dispatch, payload) in routed_dispatches.iter().zip(routed) {
            record_routed_diagnostic_once(ctx, pipelines, dispatch)?;
            routed_diagnostics.push(ExpertDiagnostics {
                expert_id: payload.expert_id,
                gate: read_device_f32(ctx, &buffers.gate, &buffers.readback, 2048)?,
                up: read_device_f32(ctx, &buffers.up, &buffers.readback, 2048)?,
                hidden: read_device_f32(ctx, &buffers.hidden, &buffers.readback, 2048)?,
                down: read_device_f32(ctx, &buffers.down, &buffers.readback, 4096)?,
            });
        }
        Some(OfficialMoeDiagnostics {
            routed: routed_diagnostics,
            shared,
            accumulator: output,
        })
    } else {
        None
    };

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
    Ok(OfficialMoeResult {
        branch_bf16,
        gpu_kernel_ms,
        diagnostics,
    })
}

fn record_routed_diagnostic_once(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    dispatch: &RoutedOfficialDispatch,
) -> Result<()> {
    unsafe {
        let pool = make_command_pool(ctx)?;
        let cb = allocate_command_buffer(ctx, pool)?;
        ctx.device.begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        let read_after_write = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        pipelines.cmd_mxfp4_matvec(ctx, cb, &dispatch.w1);
        pipelines.cmd_mxfp4_matvec(ctx, cb, &dispatch.w3);
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[read_after_write],
            &[],
            &[],
        );
        pipelines.cmd_official_expert_prepare(ctx, cb, &dispatch.prepare);
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[read_after_write],
            &[],
            &[],
        );
        pipelines.cmd_mxfp4_matvec(ctx, cb, &dispatch.w2);
        ctx.device.end_command_buffer(cb)?;
        submit_and_wait(ctx, cb)?;
        ctx.device.destroy_command_pool(pool, None);
    }
    Ok(())
}

fn read_device_f32(
    ctx: &VulkanContext,
    source: &GpuBuffer,
    readback: &GpuBuffer,
    count: usize,
) -> Result<Vec<f32>> {
    let bytes = count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("diagnostic readback byte overflow"))?
        as u64;
    if bytes > source.size() || bytes > readback.size() {
        bail!("diagnostic readback exceeds buffer capacity");
    }
    unsafe {
        let pool = make_command_pool(ctx)?;
        let cb = allocate_command_buffer(ctx, pool)?;
        ctx.device.begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
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
        copy(ctx, cb, source, readback, bytes);
        ctx.device.end_command_buffer(cb)?;
        submit_and_wait(ctx, cb)?;
        ctx.device.destroy_command_pool(pool, None);
        Ok(std::slice::from_raw_parts(readback.mapped() as *const f32, count).to_vec())
    }
}

fn record_official_moe_once(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    accumulator: &GpuBuffer,
    readback: &GpuBuffer,
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
            accumulator.handle(),
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
            accumulator,
            readback,
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
                gpu_identity: None,
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
        gpu_identity: None,
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

fn gpu_moe_identity(
    manifest: &FullDepthBridgeManifest,
    expert_id: Option<u32>,
) -> Result<GpuMoeIdentity> {
    let mut tensors = Vec::with_capacity(6);
    for component in ["w1", "w3", "w2"] {
        let (weight, scale) = match expert_id {
            Some(expert_id) => fulldepth_bridge_pair_entries(manifest, expert_id, component)?,
            None => fulldepth_bridge_shared_pair_entries(manifest, component)?,
        };
        for payload in [weight, scale] {
            if payload.sha256.len() != 64
                || !payload
                    .sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                bail!("GPU payload identity requires lowercase SHA-256");
            }
            tensors.push(GpuTensorIdentity {
                tensor: payload.tensor.clone(),
                bytes: payload.bytes,
                sha256: payload.sha256.clone(),
            });
        }
    }
    let tensors: [GpuTensorIdentity; 6] = tensors
        .try_into()
        .map_err(|_| anyhow::anyhow!("GPU payload identity tensor count drift"))?;
    Ok(GpuMoeIdentity { tensors })
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
                gpu_identity: None,
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
        gpu_identity: None,
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

    fn fake_gpu_identity(seed: char) -> GpuMoeIdentity {
        let tensors = std::array::from_fn(|index| GpuTensorIdentity {
            tensor: format!("layers.0.ffn.experts.0.t{index}"),
            bytes: (index + 1) as u64,
            sha256: seed.to_string().repeat(64),
        });
        GpuMoeIdentity { tensors }
    }

    fn fake_payload(layout: MoeWeightByteLayout, mix_weight: f32) -> MoePayload {
        let bytes = |count: u64| Arc::<[u8]>::from(vec![0u8; count as usize]);
        MoePayload {
            expert_id: Some(0),
            mix_weight,
            gpu_identity: None,
            w1: bytes(layout.w1),
            s1: bytes(layout.s1),
            w3: bytes(layout.w3),
            s3: bytes(layout.s3),
            w2: bytes(layout.w2),
            s2: bytes(layout.s2),
        }
    }

    #[test]
    fn reusable_gpu_slot_has_exact_bounded_official_layout() {
        let (routed, shared) = official_moe_weight_layouts().unwrap();
        assert_eq!(
            routed,
            MoeWeightByteLayout {
                w1: 4_194_304,
                s1: 262_144,
                w3: 4_194_304,
                s3: 262_144,
                w2: 4_194_304,
                s2: 262_144,
            }
        );
        assert_eq!(
            shared,
            MoeWeightByteLayout {
                w1: 8_388_608,
                s1: 512,
                w3: 8_388_608,
                s3: 512,
                w2: 8_388_608,
                s2: 512,
            }
        );
        assert_eq!(routed.total().unwrap(), 13_369_344);
        assert_eq!(shared.total().unwrap(), 25_167_360);
        let (device_bytes, staging_bytes) = ReusableOfficialMoeSlot::logical_byte_counts().unwrap();
        assert_eq!(staging_bytes, 105_399_808);
        assert_eq!(device_bytes, 105_473_536);
        assert!(device_bytes <= REUSABLE_GPU_SLOT_MAX_LOGICAL_BYTES);
    }

    #[test]
    fn reusable_gpu_slot_rejects_tensor_shape_drift_without_freezing_route_weight() {
        let layout = MoeWeightByteLayout {
            w1: 1,
            s1: 2,
            w3: 3,
            s3: 4,
            w2: 5,
            s2: 6,
        };
        let low_route_weight = fake_payload(layout, 0.05);
        let high_route_weight = fake_payload(layout, 0.75);
        require_exact_moe_weight_layout(&low_route_weight, layout, "low").unwrap();
        require_exact_moe_weight_layout(&high_route_weight, layout, "high").unwrap();
        assert_ne!(low_route_weight.mix_weight, high_route_weight.mix_weight);

        let mut drift = fake_payload(layout, 0.25);
        drift.s2 = Arc::from(vec![0u8; 7]);
        let error = require_exact_moe_weight_layout(&drift, layout, "drift").unwrap_err();
        assert!(format!("{error:#}").contains("payload shape drift"));
    }

    #[test]
    fn reusable_gpu_slot_telemetry_is_exact_and_mode_exclusive() {
        let (device_bytes, staging_bytes) = ReusableOfficialMoeSlot::logical_byte_counts().unwrap();
        let reusable = WritebackReusableGpuSlotTelemetry::for_successful_request(
            Some((device_bytes, staging_bytes)),
            false,
        )
        .unwrap();
        assert!(reusable.enabled);
        assert_eq!(reusable.request_uploads, 1);
        assert_eq!(reusable.request_uploaded_bytes, 105_399_808);
        assert_eq!(reusable.weight_tensor_slots_reused, 42);
        assert!(reusable.workspace_reused);
        assert!(reusable.resident_cache_isolated);

        let resident =
            WritebackReusableGpuSlotTelemetry::for_successful_request(None, true).unwrap();
        assert!(!resident.enabled);
        assert_eq!(resident.request_uploaded_bytes, 0);
        assert!(WritebackReusableGpuSlotTelemetry::for_successful_request(
            Some((device_bytes, staging_bytes)),
            true,
        )
        .is_err());
        assert!(WritebackReusableGpuSlotTelemetry::for_successful_request(None, false).is_err());
    }

    #[test]
    fn resident_and_reusable_paths_share_exact_bf16_output_contract() {
        let mut output = vec![0.0f32; 4096];
        output[0] = 1.0;
        output[1] = -2.0;
        output[2] = f32::from_bits(0x3f80_8000);
        let rounded = official_branch_bf16(&output).unwrap();
        assert_eq!(rounded.len(), 4096);
        assert_eq!(rounded[0], 0x3f80);
        assert_eq!(rounded[1], 0xc000);
        // Round-to-nearest-even at the exact BF16 halfway point.
        assert_eq!(rounded[2], 0x3f80);

        output[17] = f32::NAN;
        assert!(official_branch_bf16(&output).is_err());
        assert!(official_branch_bf16(&output[..4095]).is_err());
    }

    #[test]
    fn gpu_payload_identity_is_sha_and_size_strict() {
        let first = fake_gpu_identity('a');
        let same = fake_gpu_identity('a');
        let different_sha = fake_gpu_identity('b');
        assert_eq!(first, same);
        assert_ne!(first, different_sha);
        assert_eq!(first.bytes().unwrap(), 21);

        let mut different_size = same;
        different_size.tensors[3].bytes += 1;
        assert_ne!(first, different_size);
    }

    #[test]
    fn gpu_payload_cache_enforces_eight_gib_hard_limit() {
        let gib = 1024_u64 * 1024 * 1024;
        assert!(GpuPayloadCache::new(7 * gib).is_ok());
        assert!(GpuPayloadCache::new(8 * gib).is_ok());
        assert!(GpuPayloadCache::new(8 * gib + 1).is_err());
        assert!(GpuPayloadCache::new(0).is_err());
    }

    #[test]
    fn gpu_payload_cache_is_disabled_by_default() {
        assert_eq!(DEFAULT_GPU_PAYLOAD_CACHE_GIB, 0);
        let telemetry = WritebackGpuPayloadCacheTelemetry::disabled();
        assert!(!telemetry.enabled);
        assert_eq!(telemetry.capacity_bytes, 0);
        assert_eq!(telemetry.entries, 0);
        assert_eq!(telemetry.total_uploaded_bytes, 0);
    }

    #[test]
    fn gpu_payload_telemetry_reports_request_deltas() {
        let gib = 1024_u64 * 1024 * 1024;
        let mut cache = GpuPayloadCache::new(gib).unwrap();
        let before = cache.stats();
        cache.stats = GpuPayloadCacheStats {
            requests: 7,
            hits: 4,
            misses: 3,
            evictions: 1,
            current_bytes: 96,
            peak_bytes: 128,
            uploaded_bytes: 64,
        };
        let telemetry =
            WritebackGpuPayloadCacheTelemetry::between(&cache, before, cache.stats()).unwrap();
        assert_eq!(telemetry.request_hits, 4);
        assert_eq!(telemetry.request_misses, 3);
        assert_eq!(telemetry.request_uploaded_bytes, 64);
        assert_eq!(telemetry.total_hit_rate, 4.0 / 7.0);
        assert!(telemetry.strict_sha_identity);
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

#[cfg(test)]
mod fp8_projection_worker_tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct ProjectionFixtureDir(PathBuf);

    impl ProjectionFixtureDir {
        fn new() -> Self {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "polaris-fp8-projection-{}-{stamp}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for ProjectionFixtureDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn request(path: PathBuf) -> Fp8ProjectionRequest {
        Fp8ProjectionRequest {
            protocol: FP8_PROJECTION_PROTOCOL.to_string(),
            op: "execute_fp8_projection".to_string(),
            request_id: "l42-wq-a-0".to_string(),
            revision: REVISION.to_string(),
            profile: "fulldepth43_native_top6".to_string(),
            layer: 42,
            position: 0,
            arena_epoch: 0,
            input_sha256: L42_WQ_A_INPUT_SHA256.to_string(),
            projection: Fp8ProjectionSpec {
                name: L42_WQ_A_PROJECTION.to_string(),
                n: 1024,
                k: 4096,
                activation_contract: "cpu_e4m3fn_quant_dequant_f32".to_string(),
                output_rounding: "bf16_rne_then_f32_le".to_string(),
            },
            input: ProjectionArenaView {
                path: path.clone(),
                offset: 0,
                bytes: 16_384,
                dtype: "f32_le".to_string(),
                shape: vec![1, 1, 4096],
            },
            output: ProjectionArenaView {
                path,
                offset: 16_384,
                bytes: 4_096,
                dtype: "f32_le_bf16_rounded".to_string(),
                shape: vec![1, 1, 1024],
            },
        }
    }

    fn catalog_fixture() -> serde_json::Value {
        serde_json::json!({
            "format": "polaris-fulldepth43-native-top6-catalog-v1",
            "repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": REVISION,
            "download_authorized": false,
            "layers": {
                "42": {
                    "non_expert": [
                        {
                            "tensor": "layers.42.attn.wq_a.weight",
                            "kind": "non_expert",
                            "layer": 42,
                            "dtype": "F8_E4M3",
                            "shape": [1024, 4096],
                            "bytes": 4_194_304
                        },
                        {
                            "tensor": "layers.42.attn.wq_a.scale",
                            "kind": "non_expert",
                            "layer": 42,
                            "dtype": "F8_E8M0",
                            "shape": [8, 32],
                            "bytes": 256
                        }
                    ]
                }
            }
        })
    }

    fn fulldepth_request(layer: u32, suffix: &str) -> FullDepthFp8AttentionRequest {
        let (kernel, n, k, groups, n_per_group, activation_contract, input_shape) = match suffix {
            "wo_a" => (
                "grouped_wo_a",
                8192,
                4096,
                Some(8),
                Some(1024),
                "bf16_carrying_f32_per_group",
                vec![1, 1, 8, 4096],
            ),
            "wq_a" => (
                "standard",
                1024,
                4096,
                None,
                None,
                "cpu_e4m3fn_quant_dequant_f32",
                vec![1, 1, 4096],
            ),
            "wkv" => (
                "standard",
                512,
                4096,
                None,
                None,
                "cpu_e4m3fn_quant_dequant_f32",
                vec![1, 1, 4096],
            ),
            "wq_b" => (
                "standard",
                32768,
                1024,
                None,
                None,
                "cpu_e4m3fn_quant_dequant_f32",
                vec![1, 1, 1024],
            ),
            "indexer.wq_b" => (
                "standard",
                8192,
                1024,
                None,
                None,
                "cpu_e4m3fn_quant_dequant_f32",
                vec![1, 1, 1024],
            ),
            "wo_b" => (
                "standard",
                4096,
                8192,
                None,
                None,
                "cpu_e4m3fn_quant_dequant_f32",
                vec![1, 1, 8192],
            ),
            _ => panic!("unsupported test projection"),
        };
        let projection_name = format!("layers.{layer}.attn.{suffix}");
        let spec = FullDepthFp8ProjectionSpec {
            name: projection_name.clone(),
            kernel: kernel.to_string(),
            n,
            k,
            groups,
            n_per_group,
            activation_contract: activation_contract.to_string(),
            output_rounding: "bf16_rne_then_f32_le".to_string(),
        };
        let kernel_shape = expected_fulldepth_fp8_kernel(layer, &spec).unwrap();
        FullDepthFp8AttentionRequest {
            protocol: FULLDEPTH_FP8_ATTENTION_PROTOCOL.to_string(),
            op: "execute_fp8_attention".to_string(),
            request_id: format!("l{layer}-{suffix}-0"),
            revision: REVISION.to_string(),
            profile: "fulldepth43_native_top6".to_string(),
            layer,
            position: 17,
            arena_epoch: 0,
            input_sha256: "1".repeat(64),
            projection: spec,
            weight: FullDepthFp8AssetView {
                tensor: format!("{projection_name}.weight"),
                path: PathBuf::from("range_cache/weight.bin"),
                bytes: kernel_shape.weight_bytes().unwrap(),
                sha256: "2".repeat(64),
                dtype: "F8_E4M3".to_string(),
                shape: kernel_shape.weight_shape().unwrap(),
            },
            scale: FullDepthFp8AssetView {
                tensor: format!("{projection_name}.scale"),
                path: PathBuf::from("range_cache/scale.bin"),
                bytes: kernel_shape.scale_bytes().unwrap(),
                sha256: "3".repeat(64),
                dtype: "F8_E8M0".to_string(),
                shape: kernel_shape.scale_shape().unwrap(),
            },
            input: ProjectionArenaView {
                path: PathBuf::from("arena.bin"),
                offset: 0,
                bytes: kernel_shape.input_elements().unwrap() * 4,
                dtype: "f32_le".to_string(),
                shape: input_shape,
            },
            output: ProjectionArenaView {
                path: PathBuf::from("arena.bin"),
                offset: kernel_shape.input_elements().unwrap() * 4,
                bytes: kernel_shape.output_elements().unwrap() * 4,
                dtype: "f32_le_bf16_rounded".to_string(),
                shape: kernel_shape.output_shape().unwrap(),
            },
        }
    }

    fn fulldepth_batch_request(
        layer: u32,
        first_suffix: &str,
        second_suffix: &str,
    ) -> FullDepthFp8AttentionBatchRequest {
        let first = fulldepth_request(layer, first_suffix);
        let mut second = fulldepth_request(layer, second_suffix);
        assert_eq!(first.input.shape, second.input.shape);
        assert_eq!(first.input.bytes, second.input.bytes);
        second.output.offset = checked_arena_end(&first.output).unwrap();
        let item = |request: &FullDepthFp8AttentionRequest| FullDepthFp8AttentionBatchItem {
            projection: request.projection.clone(),
            weight: request.weight.clone(),
            scale: request.scale.clone(),
            output: request.output.clone(),
        };
        FullDepthFp8AttentionBatchRequest {
            protocol: FULLDEPTH_FP8_ATTENTION_PROTOCOL.to_string(),
            op: "execute_fp8_attention_shared_batch".to_string(),
            request_id: format!(
                "l{layer}-{}-{}-0",
                first_suffix.replace('.', "-"),
                second_suffix.replace('.', "-")
            ),
            revision: REVISION.to_string(),
            profile: "fulldepth43_native_top6".to_string(),
            layer,
            position: 17,
            arena_epoch: 0,
            input_sha256: first.input_sha256.clone(),
            input: first.input.clone(),
            projections: vec![item(&first), item(&second)],
        }
    }

    fn fulldepth_output_chain_request(layer: u32) -> FullDepthFp8AttentionOutputChainRequest {
        let wo_a = fulldepth_request(layer, "wo_a");
        let mut wo_b = fulldepth_request(layer, "wo_b");
        wo_b.output.offset = checked_arena_end(&wo_a.input).unwrap();
        let stage =
            |request: &FullDepthFp8AttentionRequest| FullDepthFp8AttentionOutputChainStage {
                projection: request.projection.clone(),
                weight: request.weight.clone(),
                scale: request.scale.clone(),
            };
        FullDepthFp8AttentionOutputChainRequest {
            protocol: FULLDEPTH_FP8_ATTENTION_PROTOCOL.to_string(),
            op: "execute_fp8_attention_output_chain".to_string(),
            request_id: format!("l{layer}-attention-output-chain-0"),
            revision: REVISION.to_string(),
            profile: "fulldepth43_native_top6".to_string(),
            layer,
            position: 17,
            arena_epoch: 0,
            input_sha256: "1".repeat(64),
            input: wo_a.input.clone(),
            projections: vec![stage(&wo_a), stage(&wo_b)],
            requantization: FullDepthFp8AttentionOutputChainRequantization {
                format: "e4m3fn_group128_quantize_dequantize_f32".to_string(),
                group_size: 128,
                amax_floor: 1.0e-4,
                max_finite: 448.0,
            },
            output: wo_b.output,
        }
    }

    fn fulldepth_catalog(request: &FullDepthFp8AttentionRequest) -> serde_json::Value {
        serde_json::json!({
            "format": "polaris-fulldepth43-native-top6-catalog-v1",
            "repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": REVISION,
            "download_authorized": false,
            "layers": {
                request.layer.to_string(): {
                    "non_expert": [
                        {
                            "tensor": request.weight.tensor,
                            "kind": "non_expert",
                            "layer": request.layer,
                            "dtype": request.weight.dtype,
                            "shape": request.weight.shape,
                            "bytes": request.weight.bytes
                        },
                        {
                            "tensor": request.scale.tensor,
                            "kind": "non_expert",
                            "layer": request.layer,
                            "dtype": request.scale.dtype,
                            "shape": request.scale.shape,
                            "bytes": request.scale.bytes
                        }
                    ]
                }
            }
        })
    }

    #[test]
    fn frozen_l42_wq_a_request_contract_accepts_only_expected_epoch() {
        let request = request(PathBuf::from("arena.bin"));
        validate_l42_wq_a_request(&request, 0).unwrap();
        assert!(validate_l42_wq_a_request(&request, 1).is_err());
    }

    #[test]
    fn fulldepth_fp8_protocol_accepts_dynamic_standard_and_grouped_requests() {
        for request in [fulldepth_request(7, "wq_a"), fulldepth_request(42, "wo_a")] {
            let kernel = validate_fulldepth_fp8_attention_request(&request, 0).unwrap();
            validate_fulldepth_fp8_attention_catalog(
                &fulldepth_catalog(&request),
                &request,
                kernel,
            )
            .unwrap();
        }
    }

    #[test]
    fn fulldepth_fp8_slot_identity_binds_kernel_weight_and_scale() {
        let request = fulldepth_request(7, "wq_a");
        let kernel = validate_fulldepth_fp8_attention_request(&request, 0).unwrap();
        let original = fulldepth_fp8_slot_key(kernel, &request);

        let mut changed_scale = request.clone();
        changed_scale.scale.sha256 = "4".repeat(64);
        assert_ne!(
            original,
            fulldepth_fp8_slot_key(kernel, &changed_scale),
            "scale identity drift must never hit an existing GPU slot"
        );

        let grouped = fulldepth_request(7, "wo_a");
        let grouped_kernel = validate_fulldepth_fp8_attention_request(&grouped, 0).unwrap();
        assert_ne!(original, fulldepth_fp8_slot_key(grouped_kernel, &grouped));
    }

    #[test]
    fn fulldepth_fp8_shared_batch_accepts_only_two_approved_projection_pairs() {
        for request in [
            fulldepth_batch_request(7, "wq_a", "wkv"),
            fulldepth_batch_request(42, "wq_b", "indexer.wq_b"),
        ] {
            let validated = validate_fulldepth_fp8_attention_batch_request(&request, 0).unwrap();
            assert_eq!(validated[0].0.input, validated[1].0.input);
            assert_eq!(validated[0].1.input_shape(), validated[1].1.input_shape());
        }
    }

    #[test]
    fn fulldepth_fp8_shared_batch_rejects_order_duplicate_layer_and_epoch_drift() {
        let valid = fulldepth_batch_request(7, "wq_a", "wkv");

        let mut reversed = valid.clone();
        reversed.projections.swap(0, 1);
        assert!(validate_fulldepth_fp8_attention_batch_request(&reversed, 0).is_err());

        let mut duplicate = valid.clone();
        duplicate.projections[1] = duplicate.projections[0].clone();
        assert!(validate_fulldepth_fp8_attention_batch_request(&duplicate, 0).is_err());

        let mut cross_layer = valid.clone();
        cross_layer.projections[1].projection.name = "layers.8.attn.wkv".to_string();
        assert!(validate_fulldepth_fp8_attention_batch_request(&cross_layer, 0).is_err());

        assert!(validate_fulldepth_fp8_attention_batch_request(&valid, 1).is_err());
    }

    #[test]
    fn fulldepth_fp8_shared_batch_scale_identity_cannot_hit_existing_slot() {
        let request = fulldepth_batch_request(7, "wq_a", "wkv");
        let original = validate_fulldepth_fp8_attention_batch_request(&request, 0).unwrap();
        let first_key = fulldepth_fp8_slot_key(original[0].1, &original[0].0);
        let second_key = fulldepth_fp8_slot_key(original[1].1, &original[1].0);

        let mut changed_scale = request;
        changed_scale.projections[1].scale.sha256 = "4".repeat(64);
        let changed = validate_fulldepth_fp8_attention_batch_request(&changed_scale, 0).unwrap();
        assert_eq!(
            first_key,
            fulldepth_fp8_slot_key(changed[0].1, &changed[0].0)
        );
        assert_ne!(
            second_key,
            fulldepth_fp8_slot_key(changed[1].1, &changed[1].0),
            "batch scale identity drift must never hit an existing GPU slot"
        );
    }

    #[test]
    fn fulldepth_fp8_shared_batch_arena_rejects_overlapping_outputs() {
        let fixture = ProjectionFixtureDir::new();
        let path = fixture.0.join("arena.bin");
        let mut request = fulldepth_batch_request(7, "wq_a", "wkv");
        request.input.path = path.clone();
        for item in &mut request.projections {
            item.output.path = path.clone();
        }
        let arena_bytes = checked_arena_end(&request.projections[1].output).unwrap();
        std::fs::write(&path, vec![0u8; arena_bytes as usize]).unwrap();
        assert_eq!(
            resolve_projection_batch_arena(
                &request.input,
                [
                    &request.projections[0].output,
                    &request.projections[1].output,
                ],
            )
            .unwrap(),
            path.canonicalize().unwrap()
        );

        request.projections[1].output.offset = request.projections[0].output.offset;
        assert!(resolve_projection_batch_arena(
            &request.input,
            [
                &request.projections[0].output,
                &request.projections[1].output,
            ],
        )
        .is_err());
    }

    #[test]
    fn fulldepth_fp8_output_chain_accepts_only_grouped_wo_a_then_wo_b() {
        let request = fulldepth_output_chain_request(42);
        let [(wo_a, wo_a_kernel), (wo_b, wo_b_kernel)] =
            validate_fulldepth_fp8_attention_output_chain_request(&request, 0).unwrap();
        assert_eq!(wo_a.projection.name, "layers.42.attn.wo_a");
        assert_eq!(wo_b.projection.name, "layers.42.attn.wo_b");
        assert!(matches!(wo_a_kernel, FullDepthFp8Kernel::GroupedWoA(_)));
        assert!(matches!(wo_b_kernel, FullDepthFp8Kernel::Standard(_)));
        assert_eq!(wo_a_kernel.output_elements().unwrap(), 8192);
        assert_eq!(wo_b_kernel.input_elements().unwrap(), 8192);
        assert_eq!(wo_b_kernel.output_elements().unwrap(), 4096);
    }

    #[test]
    fn fulldepth_fp8_output_chain_rejects_order_epoch_and_requantization_drift() {
        let valid = fulldepth_output_chain_request(7);
        assert!(validate_fulldepth_fp8_attention_output_chain_request(&valid, 1).is_err());

        let mut reversed = valid.clone();
        reversed.projections.swap(0, 1);
        assert!(validate_fulldepth_fp8_attention_output_chain_request(&reversed, 0).is_err());

        let mut duplicate = valid.clone();
        duplicate.projections[1] = duplicate.projections[0].clone();
        assert!(validate_fulldepth_fp8_attention_output_chain_request(&duplicate, 0).is_err());

        let mut wrong_group = valid.clone();
        wrong_group.requantization.group_size = 64;
        assert!(validate_fulldepth_fp8_attention_output_chain_request(&wrong_group, 0).is_err());

        let mut wrong_format = valid.clone();
        wrong_format.requantization.format = "silu_then_wo_b".to_string();
        assert!(validate_fulldepth_fp8_attention_output_chain_request(&wrong_format, 0).is_err());

        let mut wrong_output = valid;
        wrong_output.output.shape = vec![1, 1, 8192];
        assert!(validate_fulldepth_fp8_attention_output_chain_request(&wrong_output, 0).is_err());
    }

    #[test]
    fn fulldepth_fp8_output_chain_arena_is_single_file_aligned_and_disjoint() {
        let fixture = ProjectionFixtureDir::new();
        let path = fixture.0.join("arena.bin");
        let mut request = fulldepth_output_chain_request(7);
        request.input.path = path.clone();
        request.output.path = path.clone();
        let arena_bytes = checked_arena_end(&request.output).unwrap();
        std::fs::write(&path, vec![0u8; arena_bytes as usize]).unwrap();
        assert_eq!(
            resolve_projection_arena(&request.input, &request.output).unwrap(),
            path.canonicalize().unwrap()
        );

        request.output.offset = request.input.bytes - 4;
        assert!(resolve_projection_arena(&request.input, &request.output).is_err());
    }

    #[test]
    fn official_group128_e4m3fn_requantization_matches_torch_ties_to_even_vector() {
        let mut values = vec![0.0f32; 128];
        values[0] = 1.0;
        values[1] = 1.0625;
        values[2] = 1.1875;
        values[3] = -1.0625;
        values[4] = -1.1875;
        values[127] = 448.0;
        let requantized = official_group128_e4m3fn_activation_quantize_dequantize(&values).unwrap();
        assert_eq!(&requantized[..5], &[1.0, 1.0, 1.25, -1.0, -1.25]);
        assert_eq!(requantized[127], 448.0);
        assert_eq!(
            sha256_f32_le(&requantized),
            "47e7582ed13a087604d6b45a3f3d4de813b312233ae39f56220fb04259179455"
        );
    }

    #[test]
    fn frozen_l42_output_chain_hash_gate_covers_all_three_boundaries() {
        validate_frozen_l42_output_chain_hashes(
            42,
            L42_WO_A_INPUT_SHA256,
            L42_WO_A_OUTPUT_SHA256,
            L42_WO_A_REQUANTIZED_SHA256,
            L42_WO_B_OUTPUT_SHA256,
        )
        .unwrap();
        assert!(validate_frozen_l42_output_chain_hashes(
            42,
            L42_WO_A_INPUT_SHA256,
            L42_WO_A_OUTPUT_SHA256,
            L42_WO_A_REQUANTIZED_SHA256,
            &"0".repeat(64),
        )
        .is_err());
    }

    #[test]
    fn fulldepth_fp8_slot_reservation_is_entry_and_byte_bounded() {
        let request = fulldepth_request(7, "wq_a");
        let payload_bytes = request.weight.bytes + request.scale.bytes;
        assert_eq!(
            reserve_fulldepth_fp8_gpu_slot(
                FULLDEPTH_FP8_GPU_SLOT_MAX_ENTRIES - 1,
                FULLDEPTH_FP8_GPU_SLOT_MAX_RESIDENT_BYTES - payload_bytes,
                &request,
            )
            .unwrap(),
            FULLDEPTH_FP8_GPU_SLOT_MAX_RESIDENT_BYTES
        );
        assert!(
            reserve_fulldepth_fp8_gpu_slot(FULLDEPTH_FP8_GPU_SLOT_MAX_ENTRIES, 0, &request,)
                .is_err()
        );
        assert!(reserve_fulldepth_fp8_gpu_slot(
            0,
            FULLDEPTH_FP8_GPU_SLOT_MAX_RESIDENT_BYTES - payload_bytes + 1,
            &request,
        )
        .is_err());
    }

    #[test]
    fn fulldepth_fp8_protocol_rejects_epoch_identity_path_and_shape_drift() {
        let request = fulldepth_request(7, "wq_a");
        assert!(validate_fulldepth_fp8_attention_request(&request, 1).is_err());

        let mut bad_name = request.clone();
        bad_name.projection.name = "layers.8.attn.wq_a".to_string();
        assert!(validate_fulldepth_fp8_attention_request(&bad_name, 0).is_err());

        let mut bad_path = request.clone();
        bad_path.weight.path = PathBuf::from("range_cache/weight.dat");
        assert!(validate_fulldepth_fp8_attention_request(&bad_path, 0).is_err());

        let mut bad_sha = request.clone();
        bad_sha.scale.sha256 = "A".repeat(64);
        assert!(validate_fulldepth_fp8_attention_request(&bad_sha, 0).is_err());

        let mut bad_shape = request;
        bad_shape.output.shape = vec![1, 1, 1023];
        assert!(validate_fulldepth_fp8_attention_request(&bad_shape, 0).is_err());
    }

    #[test]
    fn fulldepth_fp8_catalog_rejects_dtype_and_shape_drift() {
        let request = fulldepth_request(42, "wo_a");
        let kernel = validate_fulldepth_fp8_attention_request(&request, 0).unwrap();
        let mut bad_dtype = fulldepth_catalog(&request);
        bad_dtype["layers"]["42"]["non_expert"][1]["dtype"] = serde_json::json!("F8_E4M3");
        assert!(validate_fulldepth_fp8_attention_catalog(&bad_dtype, &request, kernel).is_err());

        let mut bad_shape = fulldepth_catalog(&request);
        bad_shape["layers"]["42"]["non_expert"][0]["shape"] = serde_json::json!([8191, 4096]);
        assert!(validate_fulldepth_fp8_attention_catalog(&bad_shape, &request, kernel).is_err());
    }

    #[test]
    fn frozen_l42_wq_a_request_rejects_identity_drift() {
        let mut bad_projection = request(PathBuf::from("arena.bin"));
        bad_projection.projection.name = "layers.42.attn.wo_a".to_string();
        assert!(validate_l42_wq_a_request(&bad_projection, 0).is_err());

        let mut bad_layer = request(PathBuf::from("arena.bin"));
        bad_layer.layer = 41;
        assert!(validate_l42_wq_a_request(&bad_layer, 0).is_err());

        let mut bad_sha = request(PathBuf::from("arena.bin"));
        bad_sha.input_sha256 = "0".repeat(64);
        assert!(validate_l42_wq_a_request(&bad_sha, 0).is_err());
    }

    #[test]
    fn projection_arena_rejects_overlap_and_out_of_bounds() {
        let fixture = ProjectionFixtureDir::new();
        let path = fixture.0.join("arena.bin");
        std::fs::write(&path, vec![0u8; 32_768]).unwrap();
        let valid = request(path.clone());
        assert_eq!(
            resolve_projection_arena(&valid.input, &valid.output).unwrap(),
            path.canonicalize().unwrap()
        );

        let mut overlap = request(path.clone());
        overlap.output.offset = 16_380;
        assert!(resolve_projection_arena(&overlap.input, &overlap.output).is_err());

        let mut out_of_bounds = request(path);
        out_of_bounds.output.offset = 32_768;
        assert!(resolve_projection_arena(&out_of_bounds.input, &out_of_bounds.output).is_err());
    }

    #[test]
    fn catalog_whitelist_rejects_shape_and_dtype_drift() {
        let valid = catalog_fixture();
        validate_l42_wq_a_catalog(&valid).unwrap();

        let mut bad_shape = valid.clone();
        bad_shape["layers"]["42"]["non_expert"][0]["shape"] = serde_json::json!([1024, 4095]);
        assert!(validate_l42_wq_a_catalog(&bad_shape).is_err());

        let mut bad_dtype = valid;
        bad_dtype["layers"]["42"]["non_expert"][1]["dtype"] = serde_json::json!("F8_E4M3");
        assert!(validate_l42_wq_a_catalog(&bad_dtype).is_err());
    }

    #[test]
    fn bf16_rounding_is_round_to_nearest_ties_to_even() {
        assert_eq!(
            round_f32_to_bf16_f32(f32::from_bits(0x3f80_8000)).to_bits(),
            0x3f80_0000
        );
        assert_eq!(
            round_f32_to_bf16_f32(f32::from_bits(0x3f81_8000)).to_bits(),
            0x3f82_0000
        );
        assert_eq!(
            round_f32_to_bf16_f32(f32::from_bits(0xbf80_8000)).to_bits(),
            0xbf80_0000
        );
    }

    #[test]
    fn frozen_projection_byte_and_hash_contract_is_exact() {
        let shape = S14MatvecShape::new(1024, 4096)
            .unwrap()
            .validate_fp8()
            .unwrap();
        assert_eq!(shape.fp32_input_bytes().unwrap(), 16_384);
        assert_eq!(shape.fp8_weight_bytes().unwrap(), 4_194_304);
        assert_eq!(shape.fp8_scale_bytes().unwrap(), 256);
        assert_eq!(shape.fp32_output_bytes().unwrap(), 4_096);
        for hash in [
            FULLDEPTH43_CATALOG_SHA256,
            L42_WQ_A_WEIGHT_SHA256,
            L42_WQ_A_SCALE_SHA256,
            L42_WQ_A_INPUT_SHA256,
            L42_WQ_A_OUTPUT_SHA256,
        ] {
            assert_eq!(hash.len(), 64);
            assert!(hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
        }
    }

    #[test]
    fn standard_projection_suite_contract_covers_five_distinct_shapes() {
        let observed: Vec<(&str, u32, u32)> = L42_STANDARD_FP8_PROJECTIONS
            .iter()
            .map(|projection| (projection.name, projection.n, projection.k))
            .collect();
        assert_eq!(
            observed,
            vec![
                ("layers.42.attn.wq_a", 1024, 4096),
                ("layers.42.attn.wkv", 512, 4096),
                ("layers.42.attn.wq_b", 32768, 1024),
                ("layers.42.attn.indexer.wq_b", 8192, 1024),
                ("layers.42.attn.wo_b", 4096, 8192),
            ]
        );
        for projection in L42_STANDARD_FP8_PROJECTIONS {
            let shape = S14MatvecShape::new(projection.n, projection.k)
                .unwrap()
                .validate_fp8()
                .unwrap();
            assert_eq!(
                shape.fp8_weight_bytes().unwrap(),
                projection.n as u64 * projection.k as u64
            );
            for digest in [
                projection.weight_sha256,
                projection.scale_sha256,
                projection.input_sha256,
                projection.output_sha256,
            ] {
                assert_eq!(digest.len(), 64);
            }
        }
    }

    #[test]
    fn standard_projection_fixture_manifest_rejects_sha_drift() {
        let projections: Vec<serde_json::Value> = L42_STANDARD_FP8_PROJECTIONS
            .iter()
            .map(|projection| {
                let shape = S14MatvecShape::new(projection.n, projection.k)
                    .unwrap()
                    .validate_fp8()
                    .unwrap();
                serde_json::json!({
                    "projection": projection.name,
                    "n": projection.n,
                    "k": projection.k,
                    "activation_contract": "cpu_e4m3fn_quant_dequant_f32",
                    "output_rounding": "bf16_rne_then_f32_le",
                    "input": {
                        "shape": [1, 1, projection.k],
                        "bytes": shape.fp32_input_bytes().unwrap(),
                        "sha256": projection.input_sha256,
                    },
                    "output": {
                        "shape": [1, 1, projection.n],
                        "bytes": shape.fp32_output_bytes().unwrap(),
                        "sha256": projection.output_sha256,
                    },
                    "weight": {
                        "shape": [projection.n, projection.k],
                        "bytes": shape.fp8_weight_bytes().unwrap(),
                        "sha256": projection.weight_sha256,
                    },
                    "scale": {
                        "shape": [projection.n / 128, projection.k / 128],
                        "bytes": shape.fp8_scale_bytes().unwrap(),
                        "sha256": projection.scale_sha256,
                    },
                })
            })
            .collect();
        let mut manifest = serde_json::json!({
            "format": "polaris-l42-packed-fp8-projection-fixtures-v1",
            "repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": REVISION,
            "layer": 42,
            "catalog_sha256": FULLDEPTH43_CATALOG_SHA256,
            "projection_count": 5,
            "projections": projections,
            "layer_output_sha256": "853b8b947a3f7a275cf748d7e97a311ebb22323cd0c2f3e5e973f27b04388895",
        });
        for projection in L42_STANDARD_FP8_PROJECTIONS {
            validate_l42_projection_fixture_manifest(&manifest, projection).unwrap();
        }
        manifest["projections"][4]["output"]["sha256"] = serde_json::json!("0".repeat(64));
        assert!(validate_l42_projection_fixture_manifest(
            &manifest,
            L42_STANDARD_FP8_PROJECTIONS[4]
        )
        .is_err());
    }

    #[test]
    fn frozen_l42_wo_a_grouped_byte_and_hash_contract_is_exact() {
        let shape = S14GroupedMatvecShape::new(8, 1024, 4096)
            .unwrap()
            .validate_fp8_bf16_weight()
            .unwrap();
        assert_eq!(shape.fp32_input_bytes().unwrap(), 131_072);
        assert_eq!(shape.fp8_weight_bytes().unwrap(), 33_554_432);
        assert_eq!(shape.fp8_scale_bytes().unwrap(), 2_048);
        assert_eq!(shape.fp32_output_bytes().unwrap(), 32_768);
        for hash in [
            L42_WO_A_CAPTURE_MANIFEST_SHA256,
            L42_WO_A_INPUT_SHA256,
            L42_WO_A_WEIGHT_SHA256,
            L42_WO_A_SCALE_SHA256,
            L42_WO_A_OUTPUT_SHA256,
        ] {
            assert_eq!(hash.len(), 64);
            assert!(hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
        }
    }

    #[test]
    fn l42_wo_a_capture_manifest_accepts_frozen_contract_and_rejects_output_drift() {
        let manifest = serde_json::json!({
            "format": "polaris-l42-real-vulkan-input-capture-v1",
            "repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": REVISION,
            "layer": 42,
            "expert_id": REAL_EXPERT_ID,
            "source_f32_le_sha256": {
                "ffn_input": "7e2d3167e3782eca8d762c3cc92d53bb9d64a65c7b18d37d16797ff39f611ad4"
            },
            "asset_integrity": {
                "hashes_checked": 76,
                "payload_bytes": 247_515_224,
                "payload_files": 76,
                "manifest_sha256": {
                    "base": "5e86fa2145c1e3b5f4f8efdcd4fdcca5b966d7cb43ca0ea592294a542ac086ed",
                    "route": "feccc1b5dde256c9ad750985b4ab5446732a1f4deec796976198e95798c64b86",
                    "s14": "f6ea01a0df591f272d4e5addb96e16e8eae79b6e9326a84ef6cbaab818c6a2b5"
                }
            },
            "inputs": [
                {
                    "name": "wq_a",
                    "file": "wq_a.f32le.bin",
                    "shape": [1, 1, 4096],
                    "bytes": 16_384,
                    "f32_le_sha256": L42_WQ_A_INPUT_SHA256
                },
                {
                    "name": "wo_a_grouped_input_bf16",
                    "file": "wo_a_grouped_input_bf16.f32le.bin",
                    "shape": [1, 1, 8, 4096],
                    "bytes": 131_072,
                    "f32_le_sha256": L42_WO_A_INPUT_SHA256
                },
                {
                    "name": "wo_a_grouped_output_bf16",
                    "file": "wo_a_grouped_output_bf16.f32le.bin",
                    "shape": [1, 1, 8192],
                    "bytes": 32_768,
                    "f32_le_sha256": L42_WO_A_OUTPUT_SHA256
                },
                {
                    "name": "expert_126_w1_w3",
                    "file": "expert_126_w1_w3.f32le.bin",
                    "shape": [1, 1, 4096],
                    "bytes": 16_384,
                    "f32_le_sha256": "1a2006c1b79b31bc3db3540ef730b08547c6148e224074b0ba712e2c3b9d7c9f"
                },
                {
                    "name": "expert_126_w2",
                    "file": "expert_126_w2.f32le.bin",
                    "shape": [1, 1, 2048],
                    "bytes": 8_192,
                    "f32_le_sha256": "7795b1df61092c44883579a0107d02e693303fd16e926fcbd1baa154b49a9a31"
                }
            ],
            "semantics": "Inputs captured from the hash-verified real L42 inline reference after the official UE8M0/E4M3FN activation quantization step."
        });
        validate_l42_wo_a_capture_manifest(&manifest).unwrap();

        let mut bad_sha = manifest.clone();
        bad_sha["inputs"][2]["f32_le_sha256"] = serde_json::json!("0".repeat(64));
        assert!(validate_l42_wo_a_capture_manifest(&bad_sha).is_err());

        let mut bad_shape = manifest;
        bad_shape["inputs"][2]["shape"] = serde_json::json!([1, 1, 8191]);
        assert!(validate_l42_wo_a_capture_manifest(&bad_shape).is_err());
    }
}
