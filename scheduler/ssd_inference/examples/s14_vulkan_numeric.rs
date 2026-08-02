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
    S14Mxfp4Dispatch, S14NumericPipelines, S14OfficialExpertPrepareDispatch,
    S14RaggedBranchOffsets, S14RaggedMatvecShape, S14RaggedProjection, S14RouteMixDispatch,
    S14SwigluLimitDispatch,
};
use ssd_inference::verified_payload_cache::{
    VerifiedPayloadRequest, VERIFIED_PAYLOAD_BATCH_MAX_TASKS,
};
use ssd_inference::{VerifiedPayloadCache, VerifiedPayloadCacheStats};
use std::collections::{BTreeMap, HashMap, HashSet};
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
const CAUSAL_BLOCK_LAYER_OUTPUT_FILE: &str = "vulkan_moe_block_branches.bf16le.bin";
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
const SHARED_GPU_PAYLOAD_CACHE_GIB_ENV: &str = "POLARIS_SHARED_GPU_PAYLOAD_CACHE_GIB";
const DEFAULT_SHARED_GPU_PAYLOAD_CACHE_GIB: usize = 0;
const SHARED_GPU_PAYLOAD_CACHE_GIB: usize = 2;
const GPU_VRAM_HARD_LIMIT_GIB: usize = 8;
const OFFICIAL_ROUTED_EXPERT_COUNT: usize = 6;
const CAUSAL_BLOCK_BATCH4_POSITIONS: usize = 4;
const CAUSAL_BLOCK_BATCH4_ROUTED_BRANCHES: usize =
    CAUSAL_BLOCK_BATCH4_POSITIONS * OFFICIAL_ROUTED_EXPERT_COUNT;
const REUSABLE_GPU_SLOT_MAX_LOGICAL_BYTES: u64 = 128 * 1024 * 1024;
const CAUSAL_BLOCK_UNION_ARENA_MAX_BYTES: u64 = 1024 * 1024 * 1024;
const WRITEBACK_DIAGNOSTIC_DIR_ENV: &str = "POLARIS_FULLDEPTH43_WRITEBACK_DIAGNOSTIC_DIR";
const WRITEBACK_BATCH_VERIFY_ENV: &str = "POLARIS_FULLDEPTH43_BATCH_VERIFY_PAYLOADS";
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct GpuTensorIdentity {
    tensor: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
        upload_shared: bool,
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
        if upload_shared {
            self.shared.rewrite_staging(shared, "shared")?;
        }

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
            if upload_shared {
                self.shared.cmd_upload(ctx, cb);
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

    fn rewrite_and_upload_activation(&self, ctx: &VulkanContext, x: &[f32]) -> Result<()> {
        if x.len() != 4096 {
            bail!("causal block reusable activation shape drift");
        }
        self.x
            .rewrite_staging(bytemuck::cast_slice(x), "causal block activation")?;
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
        self.shared.destroy(ctx);
        for weights in self.routed.iter().rev() {
            weights.destroy(ctx);
        }
        self.x.destroy(ctx);
    }
}

/// Persistent K=4 workspace for one grouped causal-block command graph.
/// The large weights stay in the union arena; only compact inputs, metadata,
/// route weights and branch intermediates live here.
struct ReusableCausalBlockBatch4MoeSlot {
    x: ReusableUploadedBuffer,
    routed_metadata: ReusableUploadedBuffer,
    shared_metadata: ReusableUploadedBuffer,
    routed_route_weights: ReusableUploadedBuffer,
    shared_route_weights: ReusableUploadedBuffer,
    routed_gate: GpuBuffer,
    routed_up: GpuBuffer,
    routed_hidden: GpuBuffer,
    routed_down: GpuBuffer,
    shared_gate: GpuBuffer,
    shared_up: GpuBuffer,
    shared_hidden: GpuBuffer,
    shared_down: GpuBuffer,
    output: GpuBuffer,
    readback: GpuBuffer,
}

impl ReusableCausalBlockBatch4MoeSlot {
    fn new(ctx: &VulkanContext) -> Result<Self> {
        let x_bytes = CAUSAL_BLOCK_BATCH4_POSITIONS as u64 * 4096 * 4;
        let routed_metadata_bytes = CAUSAL_BLOCK_BATCH4_ROUTED_BRANCHES as u64 * 6 * 4;
        let shared_metadata_bytes = CAUSAL_BLOCK_BATCH4_POSITIONS as u64 * 6 * 4;
        let routed_route_bytes = CAUSAL_BLOCK_BATCH4_ROUTED_BRANCHES as u64 * 4;
        let shared_route_bytes = CAUSAL_BLOCK_BATCH4_POSITIONS as u64 * 4;
        let routed_intermediate_bytes = CAUSAL_BLOCK_BATCH4_ROUTED_BRANCHES as u64 * 2048 * 4;
        let routed_output_bytes = CAUSAL_BLOCK_BATCH4_ROUTED_BRANCHES as u64 * 4096 * 4;
        let shared_intermediate_bytes = CAUSAL_BLOCK_BATCH4_POSITIONS as u64 * 2048 * 4;
        let output_bytes = CAUSAL_BLOCK_BATCH4_POSITIONS as u64 * 4096 * 4;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        Ok(Self {
            x: ReusableUploadedBuffer::new(ctx, x_bytes)?,
            routed_metadata: ReusableUploadedBuffer::new(ctx, routed_metadata_bytes)?,
            shared_metadata: ReusableUploadedBuffer::new(ctx, shared_metadata_bytes)?,
            routed_route_weights: ReusableUploadedBuffer::new(ctx, routed_route_bytes)?,
            shared_route_weights: ReusableUploadedBuffer::new(ctx, shared_route_bytes)?,
            routed_gate: GpuBuffer::new_vram(ctx, routed_intermediate_bytes, storage)?,
            routed_up: GpuBuffer::new_vram(ctx, routed_intermediate_bytes, storage)?,
            routed_hidden: GpuBuffer::new_vram(ctx, routed_intermediate_bytes, storage)?,
            routed_down: GpuBuffer::new_vram(ctx, routed_output_bytes, storage)?,
            shared_gate: GpuBuffer::new_vram(ctx, shared_intermediate_bytes, storage)?,
            shared_up: GpuBuffer::new_vram(ctx, shared_intermediate_bytes, storage)?,
            shared_hidden: GpuBuffer::new_vram(ctx, shared_intermediate_bytes, storage)?,
            shared_down: GpuBuffer::new_vram(ctx, output_bytes, storage)?,
            output: GpuBuffer::new_vram(
                ctx,
                output_bytes,
                storage | vk::BufferUsageFlags::TRANSFER_SRC,
            )?,
            readback: GpuBuffer::new(
                ctx,
                output_bytes,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                true,
            )?,
        })
    }

    fn rewrite_staging(
        &self,
        prepared: &[PreparedCausalBlockPosition],
        routed_metadata: &[S14RaggedBranchOffsets],
        shared_metadata: &[S14RaggedBranchOffsets],
    ) -> Result<()> {
        if prepared.len() != CAUSAL_BLOCK_BATCH4_POSITIONS
            || routed_metadata.len() != CAUSAL_BLOCK_BATCH4_ROUTED_BRANCHES
            || shared_metadata.len() != CAUSAL_BLOCK_BATCH4_POSITIONS
        {
            bail!("causal-block batch4 workspace input shape drift");
        }
        let activations = prepared
            .iter()
            .flat_map(|position| position.input.iter().copied())
            .collect::<Vec<_>>();
        let routed_weights = prepared
            .iter()
            .flat_map(|position| position.routed.iter().map(|payload| payload.mix_weight))
            .collect::<Vec<_>>();
        let shared_weights = vec![1.0f32; CAUSAL_BLOCK_BATCH4_POSITIONS];
        let metadata_words = |rows: &[S14RaggedBranchOffsets]| {
            rows.iter().flat_map(|row| row.words()).collect::<Vec<_>>()
        };
        let routed_words = metadata_words(routed_metadata);
        let shared_words = metadata_words(shared_metadata);
        self.x
            .rewrite_staging(bytemuck::cast_slice(&activations), "batch4 activations")?;
        self.routed_metadata.rewrite_staging(
            bytemuck::cast_slice(&routed_words),
            "batch4 routed metadata",
        )?;
        self.shared_metadata.rewrite_staging(
            bytemuck::cast_slice(&shared_words),
            "batch4 shared metadata",
        )?;
        self.routed_route_weights.rewrite_staging(
            bytemuck::cast_slice(&routed_weights),
            "batch4 routed route weights",
        )?;
        self.shared_route_weights.rewrite_staging(
            bytemuck::cast_slice(&shared_weights),
            "batch4 shared route weights",
        )?;
        Ok(())
    }

    unsafe fn cmd_upload_inputs(&self, ctx: &VulkanContext, cb: vk::CommandBuffer) {
        self.x.cmd_upload(ctx, cb);
        self.routed_metadata.cmd_upload(ctx, cb);
        self.shared_metadata.cmd_upload(ctx, cb);
        self.routed_route_weights.cmd_upload(ctx, cb);
        self.shared_route_weights.cmd_upload(ctx, cb);
    }

    fn output_rows(&self) -> Vec<Vec<f32>> {
        let values = unsafe {
            std::slice::from_raw_parts(
                self.readback.mapped() as *const f32,
                CAUSAL_BLOCK_BATCH4_POSITIONS * 4096,
            )
        };
        values.chunks_exact(4096).map(<[f32]>::to_vec).collect()
    }

    fn destroy(&self, ctx: &VulkanContext) {
        self.readback.destroy(ctx);
        self.output.destroy(ctx);
        self.shared_down.destroy(ctx);
        self.shared_hidden.destroy(ctx);
        self.shared_up.destroy(ctx);
        self.shared_gate.destroy(ctx);
        self.routed_down.destroy(ctx);
        self.routed_hidden.destroy(ctx);
        self.routed_up.destroy(ctx);
        self.routed_gate.destroy(ctx);
        self.shared_route_weights.destroy(ctx);
        self.routed_route_weights.destroy(ctx);
        self.shared_metadata.destroy(ctx);
        self.routed_metadata.destroy(ctx);
        self.x.destroy(ctx);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CausalBlockArenaSpan {
    offset: u64,
    bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CausalBlockMoeArenaView {
    w1: CausalBlockArenaSpan,
    s1: CausalBlockArenaSpan,
    w3: CausalBlockArenaSpan,
    s3: CausalBlockArenaSpan,
    w2: CausalBlockArenaSpan,
    s2: CausalBlockArenaSpan,
}

impl TryFrom<&CausalBlockMoeArenaView> for S14RaggedBranchOffsets {
    type Error = anyhow::Error;

    fn try_from(view: &CausalBlockMoeArenaView) -> Result<Self> {
        let offset = |span: CausalBlockArenaSpan, label: &str| {
            u32::try_from(span.offset)
                .with_context(|| format!("{label} causal-block arena offset exceeds u32"))
        };
        Ok(Self {
            w1: offset(view.w1, "w1")?,
            s1: offset(view.s1, "s1")?,
            w3: offset(view.w3, "w3")?,
            s3: offset(view.s3, "s3")?,
            w2: offset(view.w2, "w2")?,
            s2: offset(view.s2, "s2")?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CausalBlockUnionArenaPlanEntry {
    view: CausalBlockMoeArenaView,
    is_shared: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CausalBlockUnionArenaPlan {
    entries: BTreeMap<GpuMoeIdentity, CausalBlockUnionArenaPlanEntry>,
    logical_payload_bytes: u64,
    arena_bytes: u64,
    alignment_bytes: u64,
    unique_routed_identities: usize,
    shared_identities: usize,
}

fn checked_align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("causal block union arena alignment must be a non-zero power of two");
    }
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|candidate| candidate & !mask)
        .ok_or_else(|| anyhow::anyhow!("causal block union arena alignment overflow"))
}

fn place_causal_block_arena_span(
    cursor: &mut u64,
    bytes: u64,
    alignment: u64,
    hard_limit: u64,
) -> Result<CausalBlockArenaSpan> {
    if bytes == 0 || bytes % 4 != 0 {
        bail!("causal block union arena tensor size must be positive and four-byte aligned");
    }
    let offset = checked_align_up(*cursor, alignment)?;
    let end = offset
        .checked_add(bytes)
        .ok_or_else(|| anyhow::anyhow!("causal block union arena span overflow"))?;
    if end > hard_limit {
        bail!("causal block union arena requires {end} bytes, above {hard_limit} byte hard limit");
    }
    *cursor = end;
    Ok(CausalBlockArenaSpan { offset, bytes })
}

fn plan_causal_block_union_arena<I>(
    rows: I,
    alignment: u64,
    hard_limit: u64,
) -> Result<CausalBlockUnionArenaPlan>
where
    I: IntoIterator<Item = (GpuMoeIdentity, MoeWeightByteLayout, bool)>,
{
    if hard_limit == 0 || hard_limit > (GPU_VRAM_HARD_LIMIT_GIB as u64) * 1024 * 1024 * 1024 {
        bail!("causal block union arena hard limit exceeds the fixed 8 GiB contract");
    }
    let alignment = alignment.max(4);
    if !alignment.is_power_of_two() {
        bail!("causal block union arena device alignment must be a power of two");
    }
    let mut unique = BTreeMap::<GpuMoeIdentity, (MoeWeightByteLayout, bool)>::new();
    for (identity, layout, is_shared) in rows {
        match unique.get(&identity) {
            Some((previous_layout, previous_shared))
                if *previous_layout != layout || *previous_shared != is_shared =>
            {
                bail!("causal block union arena duplicate identity layout drift")
            }
            Some(_) => {}
            None => {
                unique.insert(identity, (layout, is_shared));
            }
        }
    }
    if unique.is_empty() {
        bail!("causal block union arena cannot be empty");
    }

    let mut cursor = 0u64;
    let mut logical_payload_bytes = 0u64;
    let mut entries = BTreeMap::new();
    let mut unique_routed_identities = 0usize;
    let mut shared_identities = 0usize;
    for (identity, (layout, is_shared)) in unique {
        logical_payload_bytes = logical_payload_bytes
            .checked_add(layout.total()?)
            .ok_or_else(|| anyhow::anyhow!("causal block union payload byte overflow"))?;
        let view = CausalBlockMoeArenaView {
            w1: place_causal_block_arena_span(&mut cursor, layout.w1, alignment, hard_limit)?,
            s1: place_causal_block_arena_span(&mut cursor, layout.s1, alignment, hard_limit)?,
            w3: place_causal_block_arena_span(&mut cursor, layout.w3, alignment, hard_limit)?,
            s3: place_causal_block_arena_span(&mut cursor, layout.s3, alignment, hard_limit)?,
            w2: place_causal_block_arena_span(&mut cursor, layout.w2, alignment, hard_limit)?,
            s2: place_causal_block_arena_span(&mut cursor, layout.s2, alignment, hard_limit)?,
        };
        if is_shared {
            shared_identities += 1;
        } else {
            unique_routed_identities += 1;
        }
        entries.insert(identity, CausalBlockUnionArenaPlanEntry { view, is_shared });
    }
    if shared_identities != 1 {
        bail!("causal block union arena requires exactly one shared identity");
    }
    Ok(CausalBlockUnionArenaPlan {
        entries,
        logical_payload_bytes,
        arena_bytes: cursor,
        alignment_bytes: alignment,
        unique_routed_identities,
        shared_identities,
    })
}

fn pack_causal_block_arena_span(
    packed: &mut [u8],
    span: CausalBlockArenaSpan,
    payload: &[u8],
    label: &str,
) -> Result<()> {
    if payload.len() as u64 != span.bytes {
        bail!("{label} causal block arena payload length drift");
    }
    let start = usize::try_from(span.offset).context("causal block arena offset exceeds usize")?;
    let end_u64 = span
        .offset
        .checked_add(span.bytes)
        .ok_or_else(|| anyhow::anyhow!("{label} causal block arena span overflow"))?;
    let end = usize::try_from(end_u64).context("causal block arena end exceeds usize")?;
    let target = packed
        .get_mut(start..end)
        .ok_or_else(|| anyhow::anyhow!("{label} causal block arena span escaped allocation"))?;
    target.copy_from_slice(payload);
    Ok(())
}

/// Worker-owned fixed buffers for causal-block union weights. The first block
/// request allocates the maximum bounded staging/device pair once; later
/// requests only rewrite and upload the used prefix.
struct ReusableCausalBlockUnionArena {
    staging: GpuBuffer,
    device: GpuBuffer,
    logical_capacity_bytes: u64,
}

impl ReusableCausalBlockUnionArena {
    fn new(ctx: &VulkanContext) -> Result<Self> {
        let logical_capacity_bytes = CAUSAL_BLOCK_UNION_ARENA_MAX_BYTES;
        let staging = GpuBuffer::new_staging(ctx, logical_capacity_bytes)?;
        let device = match GpuBuffer::new_vram(
            ctx,
            logical_capacity_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        ) {
            Ok(device) => device,
            Err(error) => {
                staging.destroy(ctx);
                return Err(error.context("allocate reusable causal-block device arena"));
            }
        };
        if staging.mapped().is_null()
            || staging.size() < logical_capacity_bytes
            || device.size() < logical_capacity_bytes
        {
            device.destroy(ctx);
            staging.destroy(ctx);
            bail!("reusable causal-block union arena allocation drift");
        }
        Ok(Self {
            staging,
            device,
            logical_capacity_bytes,
        })
    }

    fn destroy(&self, ctx: &VulkanContext) {
        self.device.destroy(ctx);
        self.staging.destroy(ctx);
    }
}

struct CausalBlockUnionArena<'a> {
    buffers: &'a mut ReusableCausalBlockUnionArena,
    plan: CausalBlockUnionArenaPlan,
    host_pack_ms: f64,
    arena_allocate_ms: f64,
    arena_upload_ms: f64,
    buffers_allocated_this_request: bool,
}

impl<'a> CausalBlockUnionArena<'a> {
    fn prepare(
        ctx: &VulkanContext,
        buffers: &'a mut ReusableCausalBlockUnionArena,
        payloads: &[&MoePayload],
        arena_allocate_ms: f64,
        buffers_allocated_this_request: bool,
    ) -> Result<Self> {
        let alignment = unsafe {
            ctx.instance
                .get_physical_device_properties(ctx.physical)
                .limits
                .min_storage_buffer_offset_alignment
        }
        .max(4);
        let (routed_layout, shared_layout) = official_moe_weight_layouts()?;
        let mut unique_payloads = BTreeMap::<GpuMoeIdentity, &MoePayload>::new();
        for payload in payloads {
            let identity = payload.gpu_identity.as_ref().ok_or_else(|| {
                anyhow::anyhow!("causal block arena payload lost strict SHA identity")
            })?;
            let expected = if payload.expert_id.is_some() {
                routed_layout
            } else {
                shared_layout
            };
            require_exact_moe_weight_layout(payload, expected, "causal block union payload")?;
            match unique_payloads.get(identity) {
                Some(previous)
                    if previous.expert_id.is_some() != payload.expert_id.is_some()
                        || MoeWeightByteLayout::from_payload(previous)
                            != MoeWeightByteLayout::from_payload(payload) =>
                {
                    bail!("causal block arena duplicate payload identity drift")
                }
                Some(_) => {}
                None => {
                    unique_payloads.insert(identity.clone(), payload);
                }
            }
        }
        let plan = plan_causal_block_union_arena(
            unique_payloads.iter().map(|(identity, payload)| {
                (
                    identity.clone(),
                    MoeWeightByteLayout::from_payload(payload),
                    payload.expert_id.is_none(),
                )
            }),
            alignment,
            CAUSAL_BLOCK_UNION_ARENA_MAX_BYTES,
        )?;

        if plan.arena_bytes > buffers.logical_capacity_bytes
            || plan.arena_bytes > buffers.staging.size()
            || plan.arena_bytes > buffers.device.size()
        {
            bail!("causal block union plan exceeds reusable arena capacity");
        }
        let packed_len = usize::try_from(plan.arena_bytes)
            .context("causal block union arena exceeds host address space")?;

        let host_pack_started = Instant::now();
        let pack_result = (|| -> Result<()> {
            let packed =
                unsafe { std::slice::from_raw_parts_mut(buffers.staging.mapped(), packed_len) };
            if plan.arena_bytes != plan.logical_payload_bytes {
                packed.fill(0);
            }
            for (identity, payload) in &unique_payloads {
                let entry = plan.entries.get(identity).ok_or_else(|| {
                    anyhow::anyhow!("causal block arena plan lost payload identity")
                })?;
                if payload.expert_id.is_some() {
                    for scale in [&payload.s1, &payload.s3, &payload.s2] {
                        validate_ue8m0_codes(scale)?;
                    }
                } else {
                    for weight in [&payload.w1, &payload.w3, &payload.w2] {
                        validate_e4m3fn_codes(weight)?;
                    }
                    for scale in [&payload.s1, &payload.s3, &payload.s2] {
                        validate_ue8m0_codes(scale)?;
                    }
                }
                pack_causal_block_arena_span(packed, entry.view.w1, &payload.w1, "w1")?;
                pack_causal_block_arena_span(packed, entry.view.s1, &payload.s1, "s1")?;
                pack_causal_block_arena_span(packed, entry.view.w3, &payload.w3, "w3")?;
                pack_causal_block_arena_span(packed, entry.view.s3, &payload.s3, "s3")?;
                pack_causal_block_arena_span(packed, entry.view.w2, &payload.w2, "w2")?;
                pack_causal_block_arena_span(packed, entry.view.s2, &payload.s2, "s2")?;
            }
            Ok(())
        })();
        if let Err(error) = pack_result {
            return Err(error);
        }
        let host_pack_ms = host_pack_started.elapsed().as_secs_f64() * 1000.0;

        let upload_started = Instant::now();
        unsafe {
            let pool = make_command_pool(ctx)?;
            let cb = allocate_command_buffer(ctx, pool)?;
            ctx.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            copy(ctx, cb, &buffers.staging, &buffers.device, plan.arena_bytes);
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
        let arena_upload_ms = upload_started.elapsed().as_secs_f64() * 1000.0;
        Ok(Self {
            buffers,
            plan,
            host_pack_ms,
            arena_allocate_ms,
            arena_upload_ms,
            buffers_allocated_this_request,
        })
    }

    fn view(&self, payload: &MoePayload) -> Result<&CausalBlockMoeArenaView> {
        let identity = payload
            .gpu_identity
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("causal block arena lookup lost strict SHA identity"))?;
        self.plan
            .entries
            .get(identity)
            .map(|entry| &entry.view)
            .ok_or_else(|| anyhow::anyhow!("causal block arena identity missing from fixed plan"))
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
#[serde(deny_unknown_fields)]
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

#[derive(Clone, Deserialize)]
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
    manifest: Option<PathBuf>,
    capture_root: Option<PathBuf>,
    manifest_json: Option<String>,
    manifest_sha256: Option<String>,
    manifests: Option<Vec<PathBuf>>,
    #[serde(default)]
    batch_verify_payloads: bool,
}

#[derive(Serialize)]
struct CausalBlockLayerOutput {
    position: u32,
    input_token_id: u32,
    manifest_sha256: String,
    expert_ids: Vec<u32>,
    output: CausalBlockOutputView,
}

#[derive(Serialize)]
struct CausalBlockOutputView {
    path: PathBuf,
    offset: usize,
    dtype: &'static str,
    shape: [usize; 3],
    bytes: usize,
    sha256: String,
}

#[derive(Serialize)]
struct CausalBlockLayerReplayResponse {
    protocol: &'static str,
    request_id: String,
    ok: bool,
    device: String,
    mode: &'static str,
    layer: u32,
    block_size: usize,
    positions: Vec<u32>,
    total_routed_references: usize,
    unique_routed_experts: usize,
    reused_routed_references: usize,
    shared_payload_uploads: usize,
    outputs: Vec<CausalBlockLayerOutput>,
    #[serde(flatten)]
    payload_verification: PayloadVerificationReceipt,
    payload_cache: WritebackPayloadCacheTelemetry,
    /// Compatibility projection for older report readers. The authoritative
    /// upload contract is `gpu_union_arena`; this field preserves the previous
    /// hit/miss byte counters without claiming per-identity allocations.
    gpu_payload_cache: WritebackGpuPayloadCacheTelemetry,
    gpu_union_arena: CausalBlockUnionArenaTelemetry,
    gpu_kernel_ms: f64,
    wall_ms: f64,
    speed_eligible_verifier: bool,
    claim_limit: &'static str,
}

#[derive(Serialize)]
struct WritebackOutput {
    path: PathBuf,
    dtype: &'static str,
    shape: [usize; 3],
    bytes: usize,
    sha256: String,
}

#[derive(Clone, Copy)]
struct VerifiedPayloadIdentity<'a> {
    tensor: &'a str,
    bytes: u64,
    sha256: &'a str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct PayloadVerificationReceipt {
    verification_owner: &'static str,
    verified_count: usize,
    verified_bytes: u64,
    payload_identity_sha256: String,
    payload_identity_contract: &'static str,
    verified_before_compute: bool,
    verification_scope: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct WritebackBatchVerificationReceipt {
    enabled: bool,
    batch_entries: usize,
    batch_hits: u64,
    batch_misses: u64,
    batch_disk_bytes_read: u64,
    concurrency_limit: usize,
    followup_cached_loader_hits: u64,
    all_verified_before_compute: bool,
}

struct PendingWritebackBatchVerification {
    receipt: WritebackBatchVerificationReceipt,
    followup_before: Option<VerifiedPayloadCacheStats>,
}

impl PendingWritebackBatchVerification {
    fn disabled() -> Self {
        Self {
            receipt: WritebackBatchVerificationReceipt {
                enabled: false,
                batch_entries: 0,
                batch_hits: 0,
                batch_misses: 0,
                batch_disk_bytes_read: 0,
                concurrency_limit: VERIFIED_PAYLOAD_BATCH_MAX_TASKS,
                followup_cached_loader_hits: 0,
                all_verified_before_compute: false,
            },
            followup_before: None,
        }
    }

    fn finish_before_compute(
        mut self,
        payload_cache: &VerifiedPayloadCache,
    ) -> Result<WritebackBatchVerificationReceipt> {
        let Some(before) = self.followup_before else {
            return Ok(self.receipt);
        };
        let after = payload_cache.stats();
        let hits = monotonic_delta(after.hits, before.hits, "batch_followup_hits")?;
        let misses = monotonic_delta(after.misses, before.misses, "batch_followup_misses")?;
        let disk_bytes = monotonic_delta(
            after.disk_bytes_read,
            before.disk_bytes_read,
            "batch_followup_disk_bytes_read",
        )?;
        if hits != self.receipt.batch_entries as u64 || misses != 0 || disk_bytes != 0 {
            bail!(
                "batch-preverified payloads were not complete cache hits: hits={} expected={} misses={} disk_bytes={}",
                hits,
                self.receipt.batch_entries,
                misses,
                disk_bytes
            );
        }
        self.receipt.followup_cached_loader_hits = hits;
        self.receipt.all_verified_before_compute = true;
        Ok(self.receipt)
    }
}

fn payload_verification_receipt<'a>(
    payloads: impl IntoIterator<Item = VerifiedPayloadIdentity<'a>>,
) -> Result<PayloadVerificationReceipt> {
    let mut payloads: Vec<_> = payloads.into_iter().collect();
    if payloads.is_empty() {
        bail!("payload verification receipt cannot be empty");
    }
    payloads.sort_unstable_by(|left, right| left.tensor.cmp(right.tensor));
    let mut verified_bytes = 0u64;
    let mut previous_tensor: Option<&str> = None;
    let mut identity = Sha256::new();
    identity.update(b"polaris-rust-vulkan-payload-identity-v1\0");
    for payload in &payloads {
        if payload.tensor.is_empty()
            || payload.bytes == 0
            || payload.sha256.len() != 64
            || !payload
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            bail!(
                "invalid payload verification identity for {}",
                payload.tensor
            );
        }
        if previous_tensor == Some(payload.tensor) {
            bail!(
                "duplicate payload verification identity for {}",
                payload.tensor
            );
        }
        previous_tensor = Some(payload.tensor);
        verified_bytes = verified_bytes
            .checked_add(payload.bytes)
            .ok_or_else(|| anyhow::anyhow!("verified payload byte total overflow"))?;

        let tensor_bytes = payload.tensor.as_bytes();
        let tensor_len = u64::try_from(tensor_bytes.len())
            .context("payload identity tensor name length overflow")?;
        identity.update(tensor_len.to_le_bytes());
        identity.update(tensor_bytes);
        identity.update(payload.bytes.to_le_bytes());
        identity.update(payload.sha256.as_bytes());
    }

    Ok(PayloadVerificationReceipt {
        verification_owner: "rust_vulkan_worker",
        verified_count: payloads.len(),
        verified_bytes,
        payload_identity_sha256: format!("{:x}", identity.finalize()),
        payload_identity_contract:
            "sha256(v1_nul || sorted(length_le64(tensor),tensor,bytes_le64,expected_sha256_ascii))",
        verified_before_compute: true,
        verification_scope: "all_listed_payloads_before_corresponding_gpu_compute",
    })
}

#[derive(Serialize)]
struct WritebackResponse {
    protocol: &'static str,
    request_id: String,
    ok: bool,
    device: String,
    manifest_sha256: String,
    manifest_transport: &'static str,
    layer: u32,
    position: u32,
    input_token_id: u32,
    #[serde(flatten)]
    payload_verification: PayloadVerificationReceipt,
    batch_payload_verification: WritebackBatchVerificationReceipt,
    output: WritebackOutput,
    gpu_kernel_ms: f64,
    wall_ms: f64,
    payload_cache: WritebackPayloadCacheTelemetry,
    gpu_payload_cache: WritebackGpuPayloadCacheTelemetry,
    shared_gpu_payload_cache: WritebackGpuPayloadCacheTelemetry,
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
struct CausalBlockUnionArenaTelemetry {
    enabled: bool,
    persistent_worker_buffers: bool,
    logical_capacity_bytes: u64,
    logical_payload_bytes: u64,
    arena_bytes: u64,
    alignment_bytes: u64,
    padding_bytes: u64,
    staging_allocations: u64,
    device_allocations: u64,
    upload_submissions: u64,
    copy_commands: u64,
    copy_regions: u64,
    actual_uploaded_bytes: u64,
    unique_tensor_views: usize,
    unique_routed_identities: usize,
    shared_identities: usize,
    reused_routed_references: usize,
    host_pack_ms: f64,
    arena_allocate_ms: f64,
    arena_upload_ms: f64,
    strict_sha_identity: bool,
    hard_limit_bytes: u64,
}

impl CausalBlockUnionArenaTelemetry {
    fn from_arena(
        arena: &CausalBlockUnionArena<'_>,
        reused_routed_references: usize,
    ) -> Result<Self> {
        Self::from_plan(
            &arena.plan,
            reused_routed_references,
            arena.host_pack_ms,
            arena.arena_allocate_ms,
            arena.arena_upload_ms,
            arena.buffers_allocated_this_request,
            arena.buffers.logical_capacity_bytes,
        )
    }

    fn from_plan(
        plan: &CausalBlockUnionArenaPlan,
        reused_routed_references: usize,
        host_pack_ms: f64,
        arena_allocate_ms: f64,
        arena_upload_ms: f64,
        buffers_allocated_this_request: bool,
        logical_capacity_bytes: u64,
    ) -> Result<Self> {
        if !host_pack_ms.is_finite()
            || host_pack_ms < 0.0
            || !arena_allocate_ms.is_finite()
            || arena_allocate_ms < 0.0
            || !arena_upload_ms.is_finite()
            || arena_upload_ms < 0.0
        {
            bail!("causal block union arena telemetry timing drift");
        }
        if logical_capacity_bytes < plan.arena_bytes
            || logical_capacity_bytes > CAUSAL_BLOCK_UNION_ARENA_MAX_BYTES
        {
            bail!("causal block union arena telemetry capacity drift");
        }
        let padding_bytes = plan
            .arena_bytes
            .checked_sub(plan.logical_payload_bytes)
            .ok_or_else(|| anyhow::anyhow!("causal block union arena padding underflow"))?;
        Ok(Self {
            enabled: true,
            persistent_worker_buffers: true,
            logical_capacity_bytes,
            logical_payload_bytes: plan.logical_payload_bytes,
            arena_bytes: plan.arena_bytes,
            alignment_bytes: plan.alignment_bytes,
            padding_bytes,
            staging_allocations: u64::from(buffers_allocated_this_request),
            device_allocations: u64::from(buffers_allocated_this_request),
            upload_submissions: 1,
            copy_commands: 1,
            copy_regions: 1,
            actual_uploaded_bytes: plan.arena_bytes,
            unique_tensor_views: plan.entries.len() * 6,
            unique_routed_identities: plan.unique_routed_identities,
            shared_identities: plan.shared_identities,
            reused_routed_references,
            host_pack_ms,
            arena_allocate_ms,
            arena_upload_ms,
            strict_sha_identity: true,
            hard_limit_bytes: CAUSAL_BLOCK_UNION_ARENA_MAX_BYTES,
        })
    }
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
    shared_resident_cache_hybrid: bool,
}

impl WritebackReusableGpuSlotTelemetry {
    fn for_successful_request(
        slot_logical_bytes: Option<(u64, u64)>,
        resident_cache_enabled: bool,
        shared_resident_cache_enabled: bool,
    ) -> Result<Self> {
        match (
            slot_logical_bytes,
            resident_cache_enabled,
            shared_resident_cache_enabled,
        ) {
            (Some((logical_device_bytes, logical_staging_bytes)), false, false) => Ok(Self {
                enabled: true,
                logical_device_bytes,
                logical_staging_bytes,
                request_uploads: 1,
                request_uploaded_bytes: logical_staging_bytes,
                weight_tensor_slots_reused: (OFFICIAL_ROUTED_EXPERT_COUNT + 1) * 6,
                workspace_reused: true,
                strict_fixed_shapes: true,
                resident_cache_isolated: true,
                shared_resident_cache_hybrid: false,
            }),
            (Some((logical_device_bytes, logical_staging_bytes)), false, true) => {
                let (_, shared_layout) = official_moe_weight_layouts()?;
                let uploaded_bytes = logical_staging_bytes
                    .checked_sub(shared_layout.total()?)
                    .ok_or_else(|| anyhow::anyhow!("hybrid reusable upload byte underflow"))?;
                Ok(Self {
                    enabled: true,
                    logical_device_bytes,
                    logical_staging_bytes,
                    request_uploads: 1,
                    request_uploaded_bytes: uploaded_bytes,
                    weight_tensor_slots_reused: OFFICIAL_ROUTED_EXPERT_COUNT * 6,
                    workspace_reused: true,
                    strict_fixed_shapes: true,
                    resident_cache_isolated: true,
                    shared_resident_cache_hybrid: true,
                })
            }
            (None, true, false) => Ok(Self {
                enabled: false,
                logical_device_bytes: 0,
                logical_staging_bytes: 0,
                request_uploads: 0,
                request_uploaded_bytes: 0,
                weight_tensor_slots_reused: 0,
                workspace_reused: false,
                strict_fixed_shapes: true,
                resident_cache_isolated: true,
                shared_resident_cache_hybrid: false,
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
    #[serde(flatten)]
    payload_verification: PayloadVerificationReceipt,
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
    #[serde(flatten)]
    payload_verification: PayloadVerificationReceipt,
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
    #[serde(flatten)]
    payload_verification: PayloadVerificationReceipt,
    gpu_slot_cache_entries: usize,
    activation_uploaded_bytes: u64,
}

#[derive(Serialize)]
struct FullDepthFp8AttentionOutputChainSlotResponse {
    projection: FullDepthFp8ProjectionSpec,
    weight_sha256: String,
    scale_sha256: String,
    payload_hash_verified: bool,
    #[serde(flatten)]
    payload_verification: PayloadVerificationReceipt,
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
    #[serde(flatten)]
    payload_verification: PayloadVerificationReceipt,
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
    payload_verification: PayloadVerificationReceipt,
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
    let payload_verification = payload_verification_receipt([
        VerifiedPayloadIdentity {
            tensor: &request.weight.tensor,
            bytes: request.weight.bytes,
            sha256: &request.weight.sha256,
        },
        VerifiedPayloadIdentity {
            tensor: &request.scale.tensor,
            bytes: request.scale.bytes,
            sha256: &request.scale.sha256,
        },
    ])?;
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
        payload_verification,
        gpu_slot_cache_hit,
        gpu_slot_cache_entries: standard_slots.len() + grouped_slots.len(),
        gpu_slot_resident_bytes: *gpu_slot_resident_bytes,
        payload_uploaded_bytes,
    })
}

fn fulldepth_fp8_payload_verification_receipt(
    requests: &[&FullDepthFp8AttentionRequest],
) -> Result<PayloadVerificationReceipt> {
    let payloads = requests.iter().flat_map(|request| {
        [
            VerifiedPayloadIdentity {
                tensor: request.weight.tensor.as_str(),
                bytes: request.weight.bytes,
                sha256: request.weight.sha256.as_str(),
            },
            VerifiedPayloadIdentity {
                tensor: request.scale.tensor.as_str(),
                bytes: request.scale.bytes,
                sha256: request.scale.sha256.as_str(),
            },
        ]
    });
    payload_verification_receipt(payloads)
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
                            payload_verification: execution.payload_verification,
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
                    let payload_verification =
                        fulldepth_fp8_payload_verification_receipt(&[&first, &second])?;
                    let shared_activation_bytes = request.input.bytes;
                    let outputs = vec![
                        FullDepthFp8AttentionBatchItemResponse {
                            projection: first.projection,
                            output_written: first.output,
                            output_sha256: first_output_sha256,
                            weight_sha256: first.weight.sha256,
                            scale_sha256: first.scale.sha256,
                            payload_hash_verified: true,
                            payload_verification: first_execution.payload_verification,
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
                            payload_verification: second_execution.payload_verification,
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
                            payload_verification,
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
                    let payload_verification =
                        fulldepth_fp8_payload_verification_receipt(&[&wo_a, &wo_b])?;

                    let slots = vec![
                        FullDepthFp8AttentionOutputChainSlotResponse {
                            projection: wo_a.projection,
                            weight_sha256: wo_a.weight.sha256,
                            scale_sha256: wo_a.scale.sha256,
                            payload_hash_verified: true,
                            payload_verification: wo_a_execution.payload_verification,
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
                            payload_verification: wo_b_execution.payload_verification,
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
                            payload_verification,
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
    let shared_gpu_payload_cache_capacity = shared_gpu_payload_cache_capacity_bytes()?;
    if gpu_payload_cache_capacity.is_some() && shared_gpu_payload_cache_capacity.is_some() {
        bail!("general and shared-only GPU payload caches are mutually exclusive");
    }
    let mut shared_gpu_payload_cache = shared_gpu_payload_cache_capacity
        .map(GpuPayloadCache::new)
        .transpose()?;
    let causal_block_layer_replay_available =
        gpu_payload_cache.is_none() && shared_gpu_payload_cache.is_none();
    // The two GPU weight modes are deliberately disjoint. Default operation
    // owns one bounded upload slot; explicit resident-cache operation owns no
    // reusable slot, so stale slot contents can never masquerade as a hit.
    let reusable_gpu_slot = if gpu_payload_cache.is_none() {
        Some(ReusableOfficialMoeSlot::new(&ctx)?)
    } else {
        None
    };
    let causal_batch4_workspace = if causal_block_layer_replay_available {
        Some(ReusableCausalBlockBatch4MoeSlot::new(&ctx)?)
    } else {
        None
    };
    let mut reusable_causal_block_arena: Option<ReusableCausalBlockUnionArena> = None;
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
            "inline_manifest_json": true,
            "causal_block_layer_replay": causal_block_layer_replay_available,
            "causal_block_sizes": [4, 8],
            "batch_payload_verification": true,
            "batch_payload_verification_concurrency_limit": VERIFIED_PAYLOAD_BATCH_MAX_TASKS,
            "payload_cache_capacity_bytes": payload_cache_capacity,
            "gpu_payload_cache": gpu_payload_cache.is_some(),
            "gpu_payload_cache_capacity_bytes": gpu_payload_cache_capacity.unwrap_or(0),
            "shared_gpu_payload_cache": shared_gpu_payload_cache.is_some(),
            "shared_gpu_payload_cache_capacity_bytes": shared_gpu_payload_cache_capacity.unwrap_or(0),
            "reusable_gpu_upload_slot": reusable_gpu_slot.is_some(),
            "reusable_gpu_upload_slot_device_bytes": reusable_gpu_slot
                .as_ref()
                .map(|slot| slot.logical_device_bytes)
                .unwrap_or(0),
            "reusable_gpu_upload_slot_staging_bytes": reusable_gpu_slot
                .as_ref()
                .map(|slot| slot.logical_staging_bytes)
                .unwrap_or(0),
            "reusable_causal_block_union_arena": causal_block_layer_replay_available,
            "reusable_causal_block_union_arena_capacity_bytes":
                CAUSAL_BLOCK_UNION_ARENA_MAX_BYTES,
            "causal_block_grouped_gpu_batch4": causal_batch4_workspace.is_some(),
            "causal_block_grouped_gpu_dispatches": 9,
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
                if let Some(cache) = shared_gpu_payload_cache.as_mut() {
                    cache.destroy(&ctx);
                }
                if let Some(slot) = reusable_gpu_slot.as_ref() {
                    slot.destroy(&ctx);
                }
                if let Some(slot) = causal_batch4_workspace.as_ref() {
                    slot.destroy(&ctx);
                }
                if let Some(arena) = reusable_causal_block_arena.as_ref() {
                    arena.destroy(&ctx);
                }
                pipelines.destroy(&ctx);
                bail!("writeback worker poisoned by invalid request");
            }
        };
        if request.protocol != WRITEBACK_PROTOCOL
            || !matches!(
                request.op.as_str(),
                "execute_single_layer"
                    | "execute_single_layer_inline_manifest"
                    | "execute_causal_block_layer_replay"
            )
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
            if let Some(cache) = shared_gpu_payload_cache.as_mut() {
                cache.destroy(&ctx);
            }
            if let Some(slot) = reusable_gpu_slot.as_ref() {
                slot.destroy(&ctx);
            }
            if let Some(slot) = causal_batch4_workspace.as_ref() {
                slot.destroy(&ctx);
            }
            if let Some(arena) = reusable_causal_block_arena.as_ref() {
                arena.destroy(&ctx);
            }
            pipelines.destroy(&ctx);
            bail!("writeback worker poisoned by contract drift");
        }

        let request_id = request.request_id.clone();
        let execution: Result<serde_json::Value> = if request.op
            == "execute_causal_block_layer_replay"
        {
            (|| -> Result<serde_json::Value> {
                let request_started = Instant::now();
                if gpu_payload_cache.is_some() || shared_gpu_payload_cache.is_some() {
                    bail!(
                        "causal block union arena is mutually exclusive with resident GPU caches"
                    );
                }
                let causal_workspace = reusable_gpu_slot.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "causal block union arena requires the persistent reusable workspace"
                    )
                })?;
                let causal_batch4_workspace =
                    causal_batch4_workspace.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("causal block grouped K=4 workspace is unavailable")
                    })?;
                let (arena_allocate_ms, buffers_allocated_this_request) =
                    if reusable_causal_block_arena.is_none() {
                        let allocate_started = Instant::now();
                        reusable_causal_block_arena =
                            Some(ReusableCausalBlockUnionArena::new(&ctx)?);
                        (allocate_started.elapsed().as_secs_f64() * 1000.0, true)
                    } else {
                        (0.0, false)
                    };
                let causal_arena_buffers =
                    reusable_causal_block_arena.as_mut().ok_or_else(|| {
                        anyhow::anyhow!("reusable causal block union arena disappeared")
                    })?;
                execute_causal_block_layer_replay(
                    &ctx,
                    &pipelines,
                    timestamp_bits,
                    timestamp_period_ns,
                    &mut payload_cache,
                    causal_workspace,
                    causal_batch4_workspace,
                    causal_arena_buffers,
                    arena_allocate_ms,
                    buffers_allocated_this_request,
                    request_started,
                    request,
                )
                .and_then(|response| {
                    serde_json::to_value(response).context("serialize causal block layer response")
                })
            })()
        } else {
            execute_writeback_request(
                &ctx,
                &pipelines,
                timestamp_bits,
                timestamp_period_ns,
                &mut payload_cache,
                &mut gpu_payload_cache,
                &mut shared_gpu_payload_cache,
                reusable_gpu_slot.as_ref(),
                request,
            )
            .and_then(|response| {
                serde_json::to_value(response).context("serialize writeback response")
            })
        };
        match execution {
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
                if let Some(cache) = shared_gpu_payload_cache.as_mut() {
                    cache.destroy(&ctx);
                }
                if let Some(slot) = reusable_gpu_slot.as_ref() {
                    slot.destroy(&ctx);
                }
                if let Some(slot) = causal_batch4_workspace.as_ref() {
                    slot.destroy(&ctx);
                }
                if let Some(arena) = reusable_causal_block_arena.as_ref() {
                    arena.destroy(&ctx);
                }
                pipelines.destroy(&ctx);
                return Err(error.context("writeback worker poisoned"));
            }
        }
    }
    if let Some(cache) = gpu_payload_cache.as_mut() {
        cache.destroy(&ctx);
    }
    if let Some(cache) = shared_gpu_payload_cache.as_mut() {
        cache.destroy(&ctx);
    }
    if let Some(slot) = reusable_gpu_slot.as_ref() {
        slot.destroy(&ctx);
    }
    if let Some(slot) = causal_batch4_workspace.as_ref() {
        slot.destroy(&ctx);
    }
    if let Some(arena) = reusable_causal_block_arena.as_ref() {
        arena.destroy(&ctx);
    }
    pipelines.destroy(&ctx);
    Ok(())
}

fn writeback_batch_verify_enabled_from(
    request_switch: bool,
    env_value: Option<&std::ffi::OsStr>,
) -> Result<bool> {
    match env_value {
        None => Ok(request_switch),
        Some(value) if value == "0" => Ok(request_switch),
        Some(value) if value == "1" => Ok(true),
        Some(value) => bail!(
            "{WRITEBACK_BATCH_VERIFY_ENV} must be exactly 0 or 1, got {:?}",
            value
        ),
    }
}

fn writeback_batch_verify_enabled(request_switch: bool) -> Result<bool> {
    let env_value = std::env::var_os(WRITEBACK_BATCH_VERIFY_ENV);
    writeback_batch_verify_enabled_from(request_switch, env_value.as_deref())
}

fn prepare_writeback_batch_requests(
    payloads: &[FullDepthBridgePayload],
    expected_count: usize,
    cache_root: &Path,
) -> Result<Vec<VerifiedPayloadRequest>> {
    if payloads.len() != expected_count || expected_count == 0 {
        bail!(
            "writeback batch payload count drift: actual={} expected={}",
            payloads.len(),
            expected_count
        );
    }
    payloads
        .iter()
        .map(|payload| {
            let path = resolve_verified_payload_path(&payload.path, cache_root, &payload.tensor)?;
            let expected_bytes = usize::try_from(payload.bytes)
                .context("writeback batch payload byte count overflows usize")?;
            Ok(VerifiedPayloadRequest {
                path,
                expected_bytes,
                expected_sha256: payload.sha256.clone(),
            })
        })
        .collect()
}

fn preverify_writeback_payload_batch(
    manifest: &FullDepthBridgeManifest,
    capture_root: &Path,
    cache_root: &Path,
    payload_cache: &mut VerifiedPayloadCache,
    enabled: bool,
) -> Result<PendingWritebackBatchVerification> {
    if !enabled {
        return Ok(PendingWritebackBatchVerification::disabled());
    }
    if !capture_root.is_dir() {
        bail!("writeback capture root is not a directory");
    }
    let canonical_capture_root = capture_root
        .canonicalize()
        .context("resolve writeback capture root for batch verification")?;
    if canonical_capture_root != capture_root {
        bail!("writeback capture root must already be canonical");
    }
    let canonical_cache_root = cache_root
        .canonicalize()
        .context("resolve range cache root for batch verification")?;
    if canonical_cache_root != cache_root || !canonical_cache_root.is_dir() {
        bail!("writeback range cache root must be a canonical directory");
    }

    let requests = prepare_writeback_batch_requests(
        &manifest.payloads,
        manifest.payload_count,
        &canonical_cache_root,
    )?;
    let before = payload_cache.stats();
    let verified = payload_cache.load_verified_batch(&requests)?;
    if verified.len() != requests.len()
        || verified
            .iter()
            .zip(&requests)
            .any(|(payload, request)| payload.len() != request.expected_bytes)
    {
        bail!("writeback batch verification output count/size drift");
    }
    let after = payload_cache.stats();
    let batch_hits = monotonic_delta(after.hits, before.hits, "batch_verify_hits")?;
    let batch_misses = monotonic_delta(after.misses, before.misses, "batch_verify_misses")?;
    let batch_disk_bytes_read = monotonic_delta(
        after.disk_bytes_read,
        before.disk_bytes_read,
        "batch_verify_disk_bytes_read",
    )?;
    if batch_hits
        .checked_add(batch_misses)
        .ok_or_else(|| anyhow::anyhow!("batch verification request count overflow"))?
        != requests.len() as u64
    {
        bail!("writeback batch verification request accounting drift");
    }

    Ok(PendingWritebackBatchVerification {
        receipt: WritebackBatchVerificationReceipt {
            enabled: true,
            batch_entries: requests.len(),
            batch_hits,
            batch_misses,
            batch_disk_bytes_read,
            concurrency_limit: VERIFIED_PAYLOAD_BATCH_MAX_TASKS,
            followup_cached_loader_hits: 0,
            all_verified_before_compute: false,
        },
        followup_before: Some(after),
    })
}

struct LoadedCausalBlockManifest {
    capture_root: PathBuf,
    manifest_sha256: String,
    manifest: FullDepthBridgeManifest,
}

struct PreparedCausalBlockPosition {
    input: Vec<f32>,
    routed: Vec<MoePayload>,
    shared: MoePayload,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CausalBlockBatch4RouteAssignment {
    route_slot: u32,
    route_weight: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct CausalBlockBatch4RouteGroup {
    rows: [Option<CausalBlockBatch4RouteAssignment>; 4],
}

#[derive(Debug, Clone, PartialEq)]
struct CausalBlockBatch4Schedule {
    routed: BTreeMap<GpuMoeIdentity, CausalBlockBatch4RouteGroup>,
    shared: GpuMoeIdentity,
}

fn plan_causal_block_batch4_schedule(
    prepared: &[PreparedCausalBlockPosition],
) -> Result<CausalBlockBatch4Schedule> {
    if prepared.len() != CAUSAL_BLOCK_BATCH4_POSITIONS {
        bail!("identity-grouped causal-block execution requires exactly K=4");
    }
    let shared = prepared[0]
        .shared
        .gpu_identity
        .clone()
        .ok_or_else(|| anyhow::anyhow!("batch4 shared expert lost strict SHA identity"))?;
    let mut routed = BTreeMap::<GpuMoeIdentity, CausalBlockBatch4RouteGroup>::new();
    for (row, position) in prepared.iter().enumerate() {
        if position.input.len() != 4096
            || position.routed.len() != OFFICIAL_ROUTED_EXPERT_COUNT
            || position.shared.expert_id.is_some()
            || position.shared.gpu_identity.as_ref() != Some(&shared)
        {
            bail!("batch4 position shape or shared identity drift");
        }
        for (route_slot, payload) in position.routed.iter().enumerate() {
            let identity = payload
                .gpu_identity
                .clone()
                .ok_or_else(|| anyhow::anyhow!("batch4 routed expert lost strict SHA identity"))?;
            if payload.expert_id.is_none()
                || !payload.mix_weight.is_finite()
                || payload.mix_weight < 0.0
            {
                bail!("batch4 routed slot contract drift");
            }
            let group = routed
                .entry(identity)
                .or_insert_with(|| CausalBlockBatch4RouteGroup { rows: [None; 4] });
            if group.rows[row].is_some() {
                bail!("batch4 row routes the same strict expert identity more than once");
            }
            group.rows[row] = Some(CausalBlockBatch4RouteAssignment {
                route_slot: route_slot as u32,
                route_weight: payload.mix_weight,
            });
        }
    }
    if routed.is_empty() || routed.len() > CAUSAL_BLOCK_BATCH4_ROUTED_BRANCHES {
        bail!("batch4 unique routed identity count drift");
    }
    Ok(CausalBlockBatch4Schedule { routed, shared })
}

fn validate_causal_block_layer_sequence(
    manifests: &[&FullDepthBridgeManifest],
) -> Result<(u32, Vec<u32>)> {
    if !matches!(manifests.len(), 4 | 8) {
        bail!("causal block layer replay only supports K=4/8");
    }
    let layer = manifests[0].layer;
    let first_position = manifests[0].position;
    let mut positions = Vec::with_capacity(manifests.len());
    for (offset, manifest) in manifests.iter().enumerate() {
        let expected_position = first_position
            .checked_add(u32::try_from(offset).context("causal block position offset overflow")?)
            .ok_or_else(|| anyhow::anyhow!("causal block position overflow"))?;
        if manifest.layer != layer || manifest.position != expected_position {
            bail!(
                "causal block layer/position sequence drift: expected L{layer}/P{expected_position}, got L{}/P{}",
                manifest.layer,
                manifest.position
            );
        }
        positions.push(manifest.position);
    }
    Ok((layer, positions))
}

fn same_payload_identity(left: &FullDepthBridgePayload, right: &FullDepthBridgePayload) -> bool {
    left.tensor == right.tensor
        && left.kind == right.kind
        && left.expert_id == right.expert_id
        && left.dtype == right.dtype
        && left.shape == right.shape
        && left.bytes == right.bytes
        && left.path == right.path
        && left.sha256 == right.sha256
}

fn validate_causal_block_request_contract(request: &WritebackRequest) -> Result<&[PathBuf]> {
    if request.op != "execute_causal_block_layer_replay"
        || request.manifest.is_some()
        || request.capture_root.is_some()
        || request.manifest_json.is_some()
        || request.manifest_sha256.is_some()
        || !request.batch_verify_payloads
    {
        bail!("causal block layer request mixed single-layer fields or disabled verification");
    }
    let paths = request
        .manifests
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("causal block layer request missing manifests"))?;
    if !matches!(paths.len(), 4 | 8) {
        bail!("causal block manifest path count must be K=4/8");
    }
    Ok(paths)
}

fn resolve_causal_block_manifest_paths(paths: &[PathBuf]) -> Result<Vec<(PathBuf, PathBuf)>> {
    if !matches!(paths.len(), 4 | 8) {
        bail!("causal block manifest path count must be K=4/8");
    }
    let mut roots = HashSet::with_capacity(paths.len());
    let mut resolved_paths = Vec::with_capacity(paths.len());
    for path in paths {
        let resolved = path
            .canonicalize()
            .with_context(|| format!("resolve causal block manifest {}", path.display()))?;
        if resolved.file_name().and_then(|value| value.to_str()) != Some("bridge_manifest.json") {
            bail!("causal block manifest filename drift");
        }
        let capture_root = resolved
            .parent()
            .ok_or_else(|| anyhow::anyhow!("causal block manifest has no parent"))?
            .canonicalize()?;
        if !roots.insert(capture_root.clone()) {
            bail!("causal block capture roots must be distinct");
        }
        resolved_paths.push((resolved, capture_root));
    }
    Ok(resolved_paths)
}

fn merge_causal_block_payload_identity(
    unique_payloads: &mut BTreeMap<String, FullDepthBridgePayload>,
    payload: &FullDepthBridgePayload,
) -> Result<()> {
    match unique_payloads.get(&payload.tensor) {
        Some(previous) if !same_payload_identity(previous, payload) => {
            bail!(
                "causal block duplicate tensor identity drift: {}",
                payload.tensor
            )
        }
        Some(_) => {}
        None => {
            unique_payloads.insert(payload.tensor.clone(), payload.clone());
        }
    }
    Ok(())
}

fn load_causal_block_manifests(paths: &[PathBuf]) -> Result<Vec<LoadedCausalBlockManifest>> {
    let resolved_paths = resolve_causal_block_manifest_paths(paths)?;
    let mut loaded = Vec::with_capacity(resolved_paths.len());
    for (resolved, capture_root) in resolved_paths {
        let bytes = std::fs::read(&resolved)?;
        let manifest_sha256 = sha256_bytes(&bytes);
        let manifest: FullDepthBridgeManifest = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse causal block manifest {}", resolved.display()))?;
        validate_writeback_manifest(&manifest)?;
        loaded.push(LoadedCausalBlockManifest {
            capture_root,
            manifest_sha256,
            manifest,
        });
    }
    let manifests: Vec<&FullDepthBridgeManifest> =
        loaded.iter().map(|entry| &entry.manifest).collect();
    validate_causal_block_layer_sequence(&manifests)?;
    Ok(loaded)
}

fn execute_causal_block_layer_replay(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    payload_cache: &mut VerifiedPayloadCache,
    workspace: &ReusableOfficialMoeSlot,
    batch4_workspace: &ReusableCausalBlockBatch4MoeSlot,
    arena_buffers: &mut ReusableCausalBlockUnionArena,
    arena_allocate_ms: f64,
    buffers_allocated_this_request: bool,
    started: Instant,
    request: WritebackRequest,
) -> Result<CausalBlockLayerReplayResponse> {
    let cache_before = payload_cache.stats();
    validate_causal_block_request_contract(&request)?;
    let WritebackRequest {
        protocol: _,
        op,
        request_id,
        manifest,
        capture_root,
        manifest_json,
        manifest_sha256,
        manifests,
        batch_verify_payloads,
    } = request;
    debug_assert_eq!(op, "execute_causal_block_layer_replay");
    debug_assert!(manifest.is_none());
    debug_assert!(capture_root.is_none());
    debug_assert!(manifest_json.is_none());
    debug_assert!(manifest_sha256.is_none());
    debug_assert!(batch_verify_payloads);
    let manifest_paths =
        manifests.ok_or_else(|| anyhow::anyhow!("causal block layer request missing manifests"))?;
    let loaded = load_causal_block_manifests(&manifest_paths)?;
    let manifest_refs: Vec<&FullDepthBridgeManifest> =
        loaded.iter().map(|entry| &entry.manifest).collect();
    let (layer, positions) = validate_causal_block_layer_sequence(&manifest_refs)?;

    let mut unique_payloads = BTreeMap::<String, FullDepthBridgePayload>::new();
    let mut unique_experts = HashSet::<u32>::new();
    for entry in &loaded {
        for &expert_id in &entry.manifest.expert_ids {
            unique_experts.insert(expert_id);
        }
        for payload in &entry.manifest.payloads {
            merge_causal_block_payload_identity(&mut unique_payloads, payload)?;
        }
    }
    let cache_root = Path::new(MODEL_DIR).join("range_cache").canonicalize()?;
    let payload_rows: Vec<FullDepthBridgePayload> = unique_payloads.into_values().collect();
    let requests =
        prepare_writeback_batch_requests(&payload_rows, payload_rows.len(), &cache_root)?;
    let verified = payload_cache.load_verified_batch(&requests)?;
    if verified.len() != requests.len()
        || verified
            .iter()
            .zip(&requests)
            .any(|(payload, request)| payload.len() != request.expected_bytes)
    {
        bail!("causal block union payload verification drift");
    }
    let payload_verification =
        payload_verification_receipt(payload_rows.iter().map(|payload| VerifiedPayloadIdentity {
            tensor: &payload.tensor,
            bytes: payload.bytes,
            sha256: &payload.sha256,
        }))?;
    let expected_verified_bytes = payload_rows.iter().try_fold(0u64, |total, payload| {
        total
            .checked_add(payload.bytes)
            .ok_or_else(|| anyhow::anyhow!("causal block payload byte overflow"))
    })?;
    if payload_verification.verified_count != payload_rows.len()
        || payload_verification.verified_bytes != expected_verified_bytes
    {
        bail!("causal block payload verification receipt drift");
    }

    let block_output_path = loaded[0].capture_root.join(CAUSAL_BLOCK_LAYER_OUTPUT_FILE);
    if block_output_path.exists() {
        bail!("refuse to overwrite causal block layer output");
    }

    let prepared = loaded
        .iter()
        .map(|entry| {
            let input = load_fulldepth_bridge_input(&entry.manifest, &entry.capture_root)?;
            let routed = entry
                .manifest
                .expert_ids
                .iter()
                .zip(&entry.manifest.route_weights)
                .map(|(&expert_id, &mix_weight)| {
                    let (w1, s1) = load_fulldepth_bridge_pair_cached(
                        &entry.manifest,
                        expert_id,
                        "w1",
                        payload_cache,
                    )?;
                    let (w3, s3) = load_fulldepth_bridge_pair_cached(
                        &entry.manifest,
                        expert_id,
                        "w3",
                        payload_cache,
                    )?;
                    let (w2, s2) = load_fulldepth_bridge_pair_cached(
                        &entry.manifest,
                        expert_id,
                        "w2",
                        payload_cache,
                    )?;
                    Ok(MoePayload {
                        expert_id: Some(expert_id),
                        mix_weight,
                        gpu_identity: Some(gpu_moe_identity(&entry.manifest, Some(expert_id))?),
                        w1,
                        s1,
                        w3,
                        s3,
                        w2,
                        s2,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let (w1, s1) =
                load_fulldepth_bridge_shared_pair_cached(&entry.manifest, "w1", payload_cache)?;
            let (w3, s3) =
                load_fulldepth_bridge_shared_pair_cached(&entry.manifest, "w3", payload_cache)?;
            let (w2, s2) =
                load_fulldepth_bridge_shared_pair_cached(&entry.manifest, "w2", payload_cache)?;
            Ok(PreparedCausalBlockPosition {
                input,
                routed,
                shared: MoePayload {
                    expert_id: None,
                    mix_weight: 1.0,
                    gpu_identity: Some(gpu_moe_identity(&entry.manifest, None)?),
                    w1,
                    s1,
                    w3,
                    s3,
                    w2,
                    s2,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let all_payloads = prepared
        .iter()
        .flat_map(|position| {
            position
                .routed
                .iter()
                .chain(std::iter::once(&position.shared))
        })
        .collect::<Vec<_>>();
    let arena = CausalBlockUnionArena::prepare(
        ctx,
        arena_buffers,
        &all_payloads,
        arena_allocate_ms,
        buffers_allocated_this_request,
    )?;
    if arena.plan.unique_routed_identities != unique_experts.len() {
        bail!("causal block union arena routed identity count drift");
    }

    let execution = (|| -> Result<(Vec<Vec<u16>>, f64)> {
        if prepared.len() == CAUSAL_BLOCK_BATCH4_POSITIONS {
            return run_official_causal_block_batch4_ragged(
                ctx,
                pipelines,
                timestamp_bits,
                timestamp_period_ns,
                &prepared,
                &arena,
                batch4_workspace,
            );
        }
        let mut outputs = Vec::with_capacity(loaded.len());
        let mut gpu_kernel_ms = 0.0;
        for position in &prepared {
            let result = run_official_top6_shared_moe_batch_union_arena(
                ctx,
                pipelines,
                timestamp_bits,
                timestamp_period_ns,
                &position.input,
                &position.routed,
                &position.shared,
                &arena,
                workspace,
            )?;
            gpu_kernel_ms += result.gpu_kernel_ms;
            outputs.push(result.branch_bf16);
        }
        Ok((outputs, gpu_kernel_ms))
    })();
    let (branch_outputs, gpu_kernel_ms) = execution?;

    let mut block_bytes = Vec::with_capacity(branch_outputs.len() * 4096 * 2);
    let mut output_rows = Vec::with_capacity(branch_outputs.len());
    for (entry, values) in loaded.iter().zip(&branch_outputs) {
        if values.len() != 4096 {
            bail!("causal block BF16 output shape drift");
        }
        let offset = block_bytes.len();
        let mut row_bytes = Vec::with_capacity(4096 * 2);
        for value in values {
            row_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let row_sha256 = sha256_bytes(&row_bytes);
        block_bytes.extend_from_slice(&row_bytes);
        output_rows.push(CausalBlockLayerOutput {
            position: entry.manifest.position,
            input_token_id: entry.manifest.input_token_id,
            manifest_sha256: entry.manifest_sha256.clone(),
            expert_ids: entry.manifest.expert_ids.clone(),
            output: CausalBlockOutputView {
                path: block_output_path.clone(),
                offset,
                dtype: "bf16_le",
                shape: [1, 1, 4096],
                bytes: row_bytes.len(),
                sha256: row_sha256,
            },
        });
    }
    let temporary = loaded[0]
        .capture_root
        .join(format!("{CAUSAL_BLOCK_LAYER_OUTPUT_FILE}.tmp"));
    if temporary.exists() {
        bail!("stale causal block layer output temporary exists");
    }
    let write_result = (|| -> Result<PathBuf> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&block_bytes)?;
        file.sync_all()?;
        std::fs::rename(&temporary, &block_output_path)?;
        let canonical = block_output_path.canonicalize()?;
        if !canonical.starts_with(&loaded[0].capture_root) {
            bail!("causal block output escaped first capture root");
        }
        Ok(canonical)
    })();
    let canonical_output = match write_result {
        Ok(value) => value,
        Err(error) => {
            let _ = std::fs::remove_file(&temporary);
            return Err(error.context("publish causal block layer output"));
        }
    };
    for row in &mut output_rows {
        row.output.path = canonical_output.clone();
    }

    let cache_after = payload_cache.stats();
    let payload_cache =
        WritebackPayloadCacheTelemetry::between(payload_cache, cache_before, cache_after)?;
    let total_routed_references = loaded.len() * OFFICIAL_ROUTED_EXPERT_COUNT;
    let unique_routed_experts = unique_experts.len();
    let reused_routed_references = total_routed_references - unique_routed_experts;
    let gpu_union_arena =
        CausalBlockUnionArenaTelemetry::from_arena(&arena, reused_routed_references)?;
    let total_gpu_references = loaded.len() * (OFFICIAL_ROUTED_EXPERT_COUNT + 1);
    let unique_gpu_identities = arena.plan.entries.len();
    let request_hits = u64::try_from(total_gpu_references - unique_gpu_identities)
        .context("causal block arena hit count exceeds u64")?;
    let request_misses = u64::try_from(unique_gpu_identities)
        .context("causal block arena miss count exceeds u64")?;
    let request_total = request_hits + request_misses;
    let gpu_payload_cache = WritebackGpuPayloadCacheTelemetry {
        enabled: true,
        capacity_bytes: arena.plan.arena_bytes,
        entries: unique_gpu_identities,
        current_bytes: arena.plan.arena_bytes,
        peak_bytes: arena.plan.arena_bytes,
        request_hits,
        request_misses,
        request_uploaded_bytes: arena.plan.arena_bytes,
        total_hits: request_hits,
        total_misses: request_misses,
        total_evictions: 0,
        total_uploaded_bytes: arena.plan.arena_bytes,
        total_hit_rate: if request_total == 0 {
            0.0
        } else {
            request_hits as f64 / request_total as f64
        },
        strict_sha_identity: true,
    };
    let shared_payload_uploads = arena.plan.shared_identities;
    Ok(CausalBlockLayerReplayResponse {
        protocol: WRITEBACK_PROTOCOL,
        request_id,
        ok: true,
        device: ctx.gpu_name.clone(),
        mode: "causal_block_layer_replay",
        layer,
        block_size: loaded.len(),
        positions,
        total_routed_references,
        unique_routed_experts,
        reused_routed_references,
        shared_payload_uploads,
        outputs: output_rows,
        payload_verification,
        payload_cache,
        gpu_payload_cache,
        gpu_union_arena,
        gpu_kernel_ms,
        wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        speed_eligible_verifier: false,
        claim_limit: "One same-layer K=4/8 replay request with union SHA verification and one fixed arena upload for all unique expert/shared identities; not yet a 43-layer causal verifier or token/s result.",
    })
}

fn execute_writeback_request(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    payload_cache: &mut VerifiedPayloadCache,
    gpu_payload_cache: &mut Option<GpuPayloadCache>,
    shared_gpu_payload_cache: &mut Option<GpuPayloadCache>,
    reusable_gpu_slot: Option<&ReusableOfficialMoeSlot>,
    request: WritebackRequest,
) -> Result<WritebackResponse> {
    let started = Instant::now();
    let cache_before = payload_cache.stats();
    let gpu_cache_before = gpu_payload_cache.as_ref().map(GpuPayloadCache::stats);
    let shared_gpu_cache_before = shared_gpu_payload_cache
        .as_ref()
        .map(GpuPayloadCache::stats);
    let WritebackRequest {
        protocol: _,
        op,
        request_id,
        manifest: manifest_path,
        capture_root: requested_capture_root,
        manifest_json,
        manifest_sha256: requested_manifest_sha256,
        manifests,
        batch_verify_payloads,
    } = request;
    if manifests.is_some() {
        bail!("single-layer writeback request mixed causal block manifests");
    }
    let batch_verify_enabled = writeback_batch_verify_enabled(batch_verify_payloads)?;
    let (capture_root, manifest_bytes, manifest_sha256, manifest_transport) = match op.as_str() {
        "execute_single_layer" => {
            if requested_capture_root.is_some()
                || manifest_json.is_some()
                || requested_manifest_sha256.is_some()
            {
                bail!("file writeback request mixed inline manifest fields");
            }
            let requested_path = manifest_path
                .ok_or_else(|| anyhow::anyhow!("file writeback request missing manifest"))?;
            let resolved_path = requested_path.canonicalize().with_context(|| {
                format!("resolve writeback manifest {}", requested_path.display())
            })?;
            if resolved_path.file_name().and_then(|value| value.to_str())
                != Some("bridge_manifest.json")
            {
                bail!("writeback manifest filename drift");
            }
            let root = resolved_path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("writeback manifest has no parent"))?
                .canonicalize()?;
            let bytes = std::fs::read(&resolved_path)?;
            let digest = sha256_bytes(&bytes);
            (root, bytes, digest, "capture_file")
        }
        "execute_single_layer_inline_manifest" => {
            if manifest_path.is_some() {
                bail!("inline writeback request mixed file manifest field");
            }
            let root = requested_capture_root
                .ok_or_else(|| anyhow::anyhow!("inline writeback request missing capture_root"))?
                .canonicalize()
                .context("resolve inline writeback capture_root")?;
            if !root.is_dir() {
                bail!("inline writeback capture_root is not a directory");
            }
            let json = manifest_json
                .ok_or_else(|| anyhow::anyhow!("inline writeback request missing manifest_json"))?;
            let expected = requested_manifest_sha256.ok_or_else(|| {
                anyhow::anyhow!("inline writeback request missing manifest_sha256")
            })?;
            let bytes = json.into_bytes();
            let digest = sha256_bytes(&bytes);
            if expected.len() != 64 || expected != digest {
                bail!("inline writeback manifest SHA-256 drift");
            }
            (root, bytes, digest, "inline_json")
        }
        _ => bail!("unsupported writeback request op"),
    };
    let manifest: FullDepthBridgeManifest =
        serde_json::from_slice(&manifest_bytes).context("parse FullDepth43 writeback manifest")?;
    validate_writeback_manifest(&manifest)?;
    let input = load_fulldepth_bridge_input(&manifest, &capture_root)?;
    let cache_root = Path::new(MODEL_DIR).join("range_cache").canonicalize()?;
    let pending_batch_verification = preverify_writeback_payload_batch(
        &manifest,
        &capture_root,
        &cache_root,
        payload_cache,
        batch_verify_enabled,
    )?;
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
    let batch_payload_verification =
        pending_batch_verification.finish_before_compute(payload_cache)?;
    // Every payload Arc above is returned only after VerifiedPayloadCache has
    // matched the complete file bytes to the manifest SHA-256.  Freeze the
    // exact verified identity set before the first Vulkan MoE dispatch.
    let payload_verification =
        payload_verification_receipt(manifest.payloads.iter().map(|payload| {
            VerifiedPayloadIdentity {
                tensor: &payload.tensor,
                bytes: payload.bytes,
                sha256: &payload.sha256,
            }
        }))?;
    if payload_verification.verified_count != manifest.payload_count
        || payload_verification.verified_bytes != manifest.payload_bytes
    {
        bail!("writeback payload verification receipt drift");
    }
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
        shared_gpu_payload_cache.as_mut(),
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
    let shared_gpu_payload_cache_telemetry =
        match (shared_gpu_payload_cache.as_ref(), shared_gpu_cache_before) {
            (Some(cache), Some(before)) => {
                WritebackGpuPayloadCacheTelemetry::between(cache, before, cache.stats())?
            }
            (None, None) => WritebackGpuPayloadCacheTelemetry::disabled(),
            _ => bail!("shared GPU payload cache enablement changed within one request"),
        };
    let reusable_gpu_slot_telemetry = WritebackReusableGpuSlotTelemetry::for_successful_request(
        reusable_gpu_slot.map(|slot| (slot.logical_device_bytes, slot.logical_staging_bytes)),
        gpu_payload_cache.is_some(),
        shared_gpu_payload_cache.is_some(),
    )?;
    Ok(WritebackResponse {
        protocol: WRITEBACK_PROTOCOL,
        request_id,
        ok: true,
        device: ctx.gpu_name.clone(),
        manifest_sha256,
        manifest_transport,
        layer: manifest.layer,
        position: manifest.position,
        input_token_id: manifest.input_token_id,
        payload_verification,
        batch_payload_verification,
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
        shared_gpu_payload_cache: shared_gpu_payload_cache_telemetry,
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

fn shared_gpu_payload_cache_capacity_bytes() -> Result<Option<u64>> {
    match std::env::var(SHARED_GPU_PAYLOAD_CACHE_GIB_ENV) {
        Ok(value) => shared_gpu_payload_cache_capacity_bytes_from(Some(&value)),
        Err(std::env::VarError::NotPresent) => shared_gpu_payload_cache_capacity_bytes_from(None),
        Err(error) => Err(error).context(SHARED_GPU_PAYLOAD_CACHE_GIB_ENV),
    }
}

fn shared_gpu_payload_cache_capacity_bytes_from(value: Option<&str>) -> Result<Option<u64>> {
    let gib = match value {
        Some(value) => value
            .parse::<usize>()
            .with_context(|| format!("parse {SHARED_GPU_PAYLOAD_CACHE_GIB_ENV}={value:?}"))?,
        None => DEFAULT_SHARED_GPU_PAYLOAD_CACHE_GIB,
    };
    if gib == 0 {
        return Ok(None);
    }
    if gib != SHARED_GPU_PAYLOAD_CACHE_GIB {
        bail!(
            "{SHARED_GPU_PAYLOAD_CACHE_GIB_ENV} must be exactly 0 or {SHARED_GPU_PAYLOAD_CACHE_GIB} GiB"
        );
    }
    let bytes = (gib as u64)
        .checked_mul(1024 * 1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("shared GPU payload cache byte capacity overflow"))?;
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
    shared_gpu_payload_cache: Option<&mut GpuPayloadCache>,
    reusable_gpu_slot: Option<&ReusableOfficialMoeSlot>,
    capture_shared_diagnostics: bool,
) -> Result<OfficialMoeResult> {
    match (
        gpu_payload_cache,
        shared_gpu_payload_cache,
        reusable_gpu_slot,
    ) {
        (Some(cache), None, None) => run_official_top6_shared_moe_batch_cached(
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
        (None, None, Some(slot)) => run_official_top6_shared_moe_batch_reusable(
            ctx,
            pipelines,
            timestamp_bits,
            timestamp_period_ns,
            x,
            routed,
            shared,
            slot,
            None,
            capture_shared_diagnostics,
        ),
        (None, Some(shared_cache), Some(slot)) => {
            shared_cache.ensure(ctx, shared)?;
            let identity = shared
                .gpu_identity
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("shared GPU payload lost strict SHA identity"))?;
            let resident = shared_cache.get(identity)?;
            run_official_top6_shared_moe_batch_reusable(
                ctx,
                pipelines,
                timestamp_bits,
                timestamp_period_ns,
                x,
                routed,
                shared,
                slot,
                Some(resident),
                capture_shared_diagnostics,
            )
        }
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

fn run_official_causal_block_batch4_ragged(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    prepared: &[PreparedCausalBlockPosition],
    arena: &CausalBlockUnionArena<'_>,
    workspace: &ReusableCausalBlockBatch4MoeSlot,
) -> Result<(Vec<Vec<u16>>, f64)> {
    let _identity_schedule = plan_causal_block_batch4_schedule(prepared)?;
    let routed_metadata = prepared
        .iter()
        .flat_map(|position| position.routed.iter())
        .map(|payload| S14RaggedBranchOffsets::try_from(arena.view(payload)?))
        .collect::<Result<Vec<_>>>()?;
    let shared_metadata = prepared
        .iter()
        .map(|position| S14RaggedBranchOffsets::try_from(arena.view(&position.shared)?))
        .collect::<Result<Vec<_>>>()?;
    workspace.rewrite_staging(prepared, &routed_metadata, &shared_metadata)?;

    let routed_shape = |n, k, projection| {
        S14RaggedMatvecShape::new(
            CAUSAL_BLOCK_BATCH4_ROUTED_BRANCHES as u32,
            if projection == S14RaggedProjection::W2 {
                1
            } else {
                OFFICIAL_ROUTED_EXPERT_COUNT as u32
            },
            n,
            k,
            projection,
        )
    };
    let shared_shape = |n, k, projection| {
        S14RaggedMatvecShape::new(CAUSAL_BLOCK_BATCH4_POSITIONS as u32, 1, n, k, projection)
    };
    let routed_w1 = pipelines.bind_ragged_mxfp4_weight_arena(
        ctx,
        routed_shape(2048, 4096, S14RaggedProjection::W1)?,
        &workspace.x.device,
        &arena.buffers.device,
        arena.plan.arena_bytes,
        &workspace.routed_metadata.device,
        &routed_metadata,
        &workspace.routed_gate,
    )?;
    let routed_w3 = pipelines.bind_ragged_mxfp4_weight_arena(
        ctx,
        routed_shape(2048, 4096, S14RaggedProjection::W3)?,
        &workspace.x.device,
        &arena.buffers.device,
        arena.plan.arena_bytes,
        &workspace.routed_metadata.device,
        &routed_metadata,
        &workspace.routed_up,
    )?;
    let routed_prepare = pipelines.bind_batched_official_expert_prepare(
        ctx,
        CAUSAL_BLOCK_BATCH4_ROUTED_BRANCHES as u32,
        2048,
        &workspace.routed_gate,
        &workspace.routed_up,
        &workspace.routed_route_weights.device,
        &workspace.routed_hidden,
    )?;
    let routed_w2 = pipelines.bind_ragged_mxfp4_weight_arena(
        ctx,
        routed_shape(4096, 2048, S14RaggedProjection::W2)?,
        &workspace.routed_hidden,
        &arena.buffers.device,
        arena.plan.arena_bytes,
        &workspace.routed_metadata.device,
        &routed_metadata,
        &workspace.routed_down,
    )?;
    let shared_w1 = pipelines.bind_ragged_fp8_weight_arena(
        ctx,
        shared_shape(2048, 4096, S14RaggedProjection::W1)?,
        &workspace.x.device,
        &arena.buffers.device,
        arena.plan.arena_bytes,
        &workspace.shared_metadata.device,
        &shared_metadata,
        &workspace.shared_gate,
    )?;
    let shared_w3 = pipelines.bind_ragged_fp8_weight_arena(
        ctx,
        shared_shape(2048, 4096, S14RaggedProjection::W3)?,
        &workspace.x.device,
        &arena.buffers.device,
        arena.plan.arena_bytes,
        &workspace.shared_metadata.device,
        &shared_metadata,
        &workspace.shared_up,
    )?;
    let shared_prepare = pipelines.bind_batched_official_expert_prepare(
        ctx,
        CAUSAL_BLOCK_BATCH4_POSITIONS as u32,
        2048,
        &workspace.shared_gate,
        &workspace.shared_up,
        &workspace.shared_route_weights.device,
        &workspace.shared_hidden,
    )?;
    let shared_w2 = pipelines.bind_ragged_fp8_weight_arena(
        ctx,
        shared_shape(4096, 2048, S14RaggedProjection::W2)?,
        &workspace.shared_hidden,
        &arena.buffers.device,
        arena.plan.arena_bytes,
        &workspace.shared_metadata.device,
        &shared_metadata,
        &workspace.shared_down,
    )?;
    let reduce = pipelines.bind_exact_order_block_reduce(
        ctx,
        CAUSAL_BLOCK_BATCH4_POSITIONS as u32,
        &workspace.routed_down,
        &workspace.shared_down,
        &workspace.output,
    )?;

    let execution = (|| -> Result<f64> {
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
            workspace.cmd_upload_inputs(ctx, cb);
            let upload_to_compute = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[upload_to_compute],
                &[],
                &[],
            );
            ctx.device.cmd_reset_query_pool(cb, queries, 0, 2);
            ctx.device
                .cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, queries, 0);
            let shader_raw = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);

            pipelines.cmd_ragged_mxfp4_matvec(ctx, cb, &routed_w1);
            pipelines.cmd_ragged_mxfp4_matvec(ctx, cb, &routed_w3);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[shader_raw],
                &[],
                &[],
            );
            pipelines.cmd_batched_official_expert_prepare(ctx, cb, &routed_prepare);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[shader_raw],
                &[],
                &[],
            );
            pipelines.cmd_ragged_mxfp4_matvec(ctx, cb, &routed_w2);

            pipelines.cmd_ragged_fp8_matvec(ctx, cb, &shared_w1);
            pipelines.cmd_ragged_fp8_matvec(ctx, cb, &shared_w3);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[shader_raw],
                &[],
                &[],
            );
            pipelines.cmd_batched_official_expert_prepare(ctx, cb, &shared_prepare);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[shader_raw],
                &[],
                &[],
            );
            pipelines.cmd_ragged_fp8_matvec(ctx, cb, &shared_w2);
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[shader_raw],
                &[],
                &[],
            );
            pipelines.cmd_exact_order_block_reduce(ctx, cb, &reduce);
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
                &workspace.output,
                &workspace.readback,
                CAUSAL_BLOCK_BATCH4_POSITIONS as u64 * 4096 * 4,
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
    })();

    for binder in [
        &reduce.binder,
        &shared_w2.binder,
        &shared_prepare.binder,
        &shared_w3.binder,
        &shared_w1.binder,
        &routed_w2.binder,
        &routed_prepare.binder,
        &routed_w3.binder,
        &routed_w1.binder,
    ] {
        binder.destroy(ctx);
    }
    let gpu_kernel_ms = execution?;
    let outputs = workspace
        .output_rows()
        .iter()
        .map(|row| official_branch_bf16(row))
        .collect::<Result<Vec<_>>>()?;
    Ok((outputs, gpu_kernel_ms))
}

fn run_official_top6_shared_moe_batch_union_arena(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    timestamp_bits: u32,
    timestamp_period_ns: f64,
    x: &[f32],
    routed: &[MoePayload],
    shared: &MoePayload,
    arena: &CausalBlockUnionArena<'_>,
    workspace: &ReusableOfficialMoeSlot,
) -> Result<OfficialMoeResult> {
    if x.len() != 4096 || routed.len() != OFFICIAL_ROUTED_EXPERT_COUNT || shared.expert_id.is_some()
    {
        bail!("official causal-block union MoE shape/route contract drift");
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
            bail!("official causal-block routed payload contract drift");
        }
    }

    workspace.rewrite_and_upload_activation(ctx, x)?;
    let routed_dispatches = routed
        .iter()
        .map(|payload| {
            let view = arena.view(payload)?;
            Ok(RoutedOfficialDispatch {
                w1: pipelines.bind_mxfp4_weight_arena(
                    ctx,
                    up_shape,
                    &workspace.x.device,
                    &arena.buffers.device,
                    arena.plan.arena_bytes,
                    view.w1.offset,
                    view.s1.offset,
                    &workspace.gate,
                )?,
                w3: pipelines.bind_mxfp4_weight_arena(
                    ctx,
                    up_shape,
                    &workspace.x.device,
                    &arena.buffers.device,
                    arena.plan.arena_bytes,
                    view.w3.offset,
                    view.s3.offset,
                    &workspace.up,
                )?,
                prepare: pipelines.bind_official_expert_prepare(
                    ctx,
                    2048,
                    payload.mix_weight,
                    &workspace.gate,
                    &workspace.up,
                    &workspace.hidden,
                )?,
                w2: pipelines.bind_mxfp4_weight_arena(
                    ctx,
                    down_shape,
                    &workspace.hidden,
                    &arena.buffers.device,
                    arena.plan.arena_bytes,
                    view.w2.offset,
                    view.s2.offset,
                    &workspace.down,
                )?,
                accumulate: pipelines.bind_bf16_accumulate(
                    ctx,
                    4096,
                    &workspace.down,
                    &workspace.accumulator,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let shared_view = arena.view(shared)?;
    let shared_dispatch = SharedOfficialDispatch {
        w1: pipelines.bind_fp8_weight_arena(
            ctx,
            shared_up_shape,
            &workspace.x.device,
            &arena.buffers.device,
            arena.plan.arena_bytes,
            shared_view.w1.offset,
            shared_view.s1.offset,
            &workspace.gate,
        )?,
        w3: pipelines.bind_fp8_weight_arena(
            ctx,
            shared_up_shape,
            &workspace.x.device,
            &arena.buffers.device,
            arena.plan.arena_bytes,
            shared_view.w3.offset,
            shared_view.s3.offset,
            &workspace.up,
        )?,
        prepare: pipelines.bind_official_expert_prepare(
            ctx,
            2048,
            1.0,
            &workspace.gate,
            &workspace.up,
            &workspace.hidden,
        )?,
        w2: pipelines.bind_fp8_weight_arena(
            ctx,
            shared_down_shape,
            &workspace.hidden,
            &arena.buffers.device,
            arena.plan.arena_bytes,
            shared_view.w2.offset,
            shared_view.s2.offset,
            &workspace.down,
        )?,
        accumulate: pipelines.bind_bf16_accumulate(
            ctx,
            4096,
            &workspace.down,
            &workspace.accumulator,
        )?,
    };

    let gpu_kernel_ms = record_official_moe_once(
        ctx,
        pipelines,
        &workspace.accumulator,
        &workspace.readback,
        timestamp_bits,
        timestamp_period_ns,
        &routed_dispatches,
        &shared_dispatch,
    )?;
    let output = workspace.output();
    let branch_bf16 = official_branch_bf16(&output)?;

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
        diagnostics: None,
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
    resident_shared_weights: Option<&GpuMoeWeights>,
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
    buffers.rewrite_and_upload(ctx, x, routed, shared, resident_shared_weights.is_none())?;
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
    let (shared_w1, shared_s1, shared_w3, shared_s3, shared_w2, shared_s2) =
        match resident_shared_weights {
            Some(weights) => (
                &weights.w1.device,
                &weights.s1.device,
                &weights.w3.device,
                &weights.s3.device,
                &weights.w2.device,
                &weights.s2.device,
            ),
            None => (
                &buffers.shared.w1.device,
                &buffers.shared.s1.device,
                &buffers.shared.w3.device,
                &buffers.shared.s3.device,
                &buffers.shared.w2.device,
                &buffers.shared.s2.device,
            ),
        };
    let shared_dispatch = SharedOfficialDispatch {
        w1: pipelines.bind_fp8(
            ctx,
            shared_up_shape,
            &buffers.x.device,
            shared_w1,
            shared_s1,
            &buffers.gate,
        )?,
        w3: pipelines.bind_fp8(
            ctx,
            shared_up_shape,
            &buffers.x.device,
            shared_w3,
            shared_s3,
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
            shared_w2,
            shared_s2,
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

    fn fake_bridge_manifest(payloads: Vec<FullDepthBridgePayload>) -> FullDepthBridgeManifest {
        FullDepthBridgeManifest {
            format: "test".to_string(),
            revision: "test".to_string(),
            profile: "test".to_string(),
            layer: 0,
            position: 0,
            input_token_id: 0,
            completed_layers_before_capture: Vec::new(),
            route_source: "test".to_string(),
            expert_ids: Vec::new(),
            route_weights: Vec::new(),
            route_weight_sum: 0.0,
            source_ffn_input_f32_le_sha256: "0".repeat(64),
            input: CaptureInput {
                name: "test".to_string(),
                file: PathBuf::from("input.bin"),
                shape: vec![1],
                bytes: 4,
                f32_le_sha256: "0".repeat(64),
            },
            payload_count: payloads.len(),
            payload_bytes: payloads.iter().map(|payload| payload.bytes).sum(),
            payloads,
            reference_semantics: "test".to_string(),
        }
    }

    fn fake_causal_block_manifest(layer: u32, position: u32) -> FullDepthBridgeManifest {
        let mut manifest = fake_bridge_manifest(Vec::new());
        manifest.layer = layer;
        manifest.position = position;
        manifest
    }

    fn fake_causal_block_request(size: usize) -> WritebackRequest {
        WritebackRequest {
            protocol: WRITEBACK_PROTOCOL.to_string(),
            op: "execute_causal_block_layer_replay".to_string(),
            request_id: "causal-block-test".to_string(),
            manifest: None,
            capture_root: None,
            manifest_json: None,
            manifest_sha256: None,
            manifests: Some(
                (0..size)
                    .map(|index| PathBuf::from(format!("capture-{index}/bridge_manifest.json")))
                    .collect(),
            ),
            batch_verify_payloads: true,
        }
    }

    fn fake_batch4_position(
        shared: &GpuMoeIdentity,
        routed_identities: &[GpuMoeIdentity],
    ) -> PreparedCausalBlockPosition {
        let layout = MoeWeightByteLayout {
            w1: 4,
            s1: 4,
            w3: 4,
            s3: 4,
            w2: 4,
            s2: 4,
        };
        let routed = routed_identities
            .iter()
            .enumerate()
            .map(|(slot, identity)| {
                let mut payload = fake_payload(layout, 0.1 + slot as f32 * 0.01);
                payload.expert_id = Some(slot as u32);
                payload.gpu_identity = Some(identity.clone());
                payload
            })
            .collect();
        let mut shared_payload = fake_payload(layout, 1.0);
        shared_payload.expert_id = None;
        shared_payload.gpu_identity = Some(shared.clone());
        PreparedCausalBlockPosition {
            input: vec![0.0; 4096],
            routed,
            shared: shared_payload,
        }
    }

    #[test]
    fn causal_block_batch4_groups_strict_identities_without_losing_route_slots() {
        let shared = fake_gpu_identity('f');
        let identities: Vec<GpuMoeIdentity> = ['0', '1', '2', '3', '4', '5']
            .into_iter()
            .map(fake_gpu_identity)
            .collect();
        let reversed: Vec<GpuMoeIdentity> = identities.iter().cloned().rev().collect();
        let prepared = vec![
            fake_batch4_position(&shared, &identities),
            fake_batch4_position(&shared, &reversed),
            fake_batch4_position(&shared, &identities),
            fake_batch4_position(&shared, &reversed),
        ];
        let schedule = plan_causal_block_batch4_schedule(&prepared).unwrap();
        assert_eq!(schedule.shared, shared);
        assert_eq!(schedule.routed.len(), 6);
        let first = schedule.routed.get(&identities[0]).unwrap();
        assert_eq!(first.rows[0].unwrap().route_slot, 0);
        assert_eq!(first.rows[1].unwrap().route_slot, 5);
        assert_eq!(first.rows[2].unwrap().route_slot, 0);
        assert_eq!(first.rows[3].unwrap().route_slot, 5);
    }

    #[test]
    fn causal_block_batch4_rejects_duplicate_identity_shared_drift_and_wrong_k() {
        let shared = fake_gpu_identity('f');
        let identities: Vec<GpuMoeIdentity> = ['0', '1', '2', '3', '4', '5']
            .into_iter()
            .map(fake_gpu_identity)
            .collect();
        let mut duplicate = identities.clone();
        duplicate[5] = duplicate[0].clone();
        let duplicated = (0..4)
            .map(|_| fake_batch4_position(&shared, &duplicate))
            .collect::<Vec<_>>();
        assert!(plan_causal_block_batch4_schedule(&duplicated).is_err());

        let mut shared_drift = (0..4)
            .map(|_| fake_batch4_position(&shared, &identities))
            .collect::<Vec<_>>();
        shared_drift[3].shared.gpu_identity = Some(fake_gpu_identity('e'));
        assert!(plan_causal_block_batch4_schedule(&shared_drift).is_err());
        assert!(plan_causal_block_batch4_schedule(&shared_drift[..3]).is_err());
    }

    #[test]
    fn causal_block_sequence_is_exactly_k4_or_k8_same_layer_and_contiguous() {
        let manifests: Vec<FullDepthBridgeManifest> = (0..8)
            .map(|position| fake_causal_block_manifest(17, position))
            .collect();
        let first_four: Vec<&FullDepthBridgeManifest> = manifests[..4].iter().collect();
        assert_eq!(
            validate_causal_block_layer_sequence(&first_four).unwrap(),
            (17, vec![0, 1, 2, 3])
        );
        let all_eight: Vec<&FullDepthBridgeManifest> = manifests.iter().collect();
        assert_eq!(
            validate_causal_block_layer_sequence(&all_eight).unwrap(),
            (17, (0..8).collect())
        );
        let first_three: Vec<&FullDepthBridgeManifest> = manifests[..3].iter().collect();
        assert!(validate_causal_block_layer_sequence(&first_three).is_err());

        let mut position_drift: Vec<FullDepthBridgeManifest> = (0..4)
            .map(|position| fake_causal_block_manifest(17, position))
            .collect();
        position_drift[2].position = 9;
        let refs: Vec<&FullDepthBridgeManifest> = position_drift.iter().collect();
        assert!(validate_causal_block_layer_sequence(&refs).is_err());

        let mut layer_drift: Vec<FullDepthBridgeManifest> = (0..4)
            .map(|position| fake_causal_block_manifest(17, position))
            .collect();
        layer_drift[3].layer = 18;
        let refs: Vec<&FullDepthBridgeManifest> = layer_drift.iter().collect();
        assert!(validate_causal_block_layer_sequence(&refs).is_err());
    }

    #[test]
    fn causal_block_request_rejects_mixed_fields_and_requires_batch_verification() {
        let valid = fake_causal_block_request(4);
        assert_eq!(
            validate_causal_block_request_contract(&valid)
                .unwrap()
                .len(),
            4
        );

        let mut disabled = fake_causal_block_request(4);
        disabled.batch_verify_payloads = false;
        assert!(validate_causal_block_request_contract(&disabled).is_err());

        let mut mixed = fake_causal_block_request(4);
        mixed.manifest = Some(PathBuf::from("bridge_manifest.json"));
        assert!(validate_causal_block_request_contract(&mixed).is_err());

        let mut missing = fake_causal_block_request(4);
        missing.manifests = None;
        assert!(validate_causal_block_request_contract(&missing).is_err());
        assert!(validate_causal_block_request_contract(&fake_causal_block_request(5)).is_err());
    }

    #[test]
    fn causal_block_manifest_paths_require_distinct_capture_roots() {
        let fixture = FixtureDir::new();
        let paths: Vec<PathBuf> = (0..4)
            .map(|index| {
                let root = fixture.0.join(format!("capture-{index}"));
                std::fs::create_dir(&root).unwrap();
                let path = root.join("bridge_manifest.json");
                std::fs::write(&path, b"{}").unwrap();
                path
            })
            .collect();
        let resolved = resolve_causal_block_manifest_paths(&paths).unwrap();
        assert_eq!(resolved.len(), 4);
        let duplicate = vec![paths[0].clone(); 4];
        assert!(resolve_causal_block_manifest_paths(&duplicate).is_err());

        let wrong_name = fixture.0.join("not-a-manifest.json");
        std::fs::write(&wrong_name, b"{}").unwrap();
        let mut wrong_paths = paths;
        wrong_paths[3] = wrong_name;
        assert!(resolve_causal_block_manifest_paths(&wrong_paths).is_err());
    }

    #[test]
    fn causal_block_duplicate_tensor_requires_full_identity_match() {
        let payload = FullDepthBridgePayload {
            tensor: "layers.17.ffn.experts.3.w1.weight".to_string(),
            kind: "routed".to_string(),
            expert_id: Some(3),
            dtype: "I8".to_string(),
            shape: vec![2048, 2048],
            bytes: 4_194_304,
            path: PathBuf::from("range_cache/expert.bin"),
            sha256: "a".repeat(64),
        };
        let mut unique = BTreeMap::new();
        merge_causal_block_payload_identity(&mut unique, &payload).unwrap();
        merge_causal_block_payload_identity(&mut unique, &payload).unwrap();
        assert_eq!(unique.len(), 1);

        let mut drifts = Vec::new();
        let mut drift = payload.clone();
        drift.kind = "shared".to_string();
        drifts.push(drift);
        let mut drift = payload.clone();
        drift.expert_id = Some(4);
        drifts.push(drift);
        let mut drift = payload.clone();
        drift.dtype = "F8_E4M3".to_string();
        drifts.push(drift);
        let mut drift = payload.clone();
        drift.shape = vec![4096, 2048];
        drifts.push(drift);
        let mut drift = payload.clone();
        drift.bytes += 4;
        drifts.push(drift);
        let mut drift = payload.clone();
        drift.path = PathBuf::from("range_cache/other.bin");
        drifts.push(drift);
        let mut drift = payload;
        drift.sha256 = "b".repeat(64);
        drifts.push(drift);
        for drift in drifts {
            assert!(merge_causal_block_payload_identity(&mut unique, &drift).is_err());
            assert_eq!(unique.len(), 1);
        }
    }

    fn tiny_arena_layout() -> MoeWeightByteLayout {
        MoeWeightByteLayout {
            w1: 4,
            s1: 8,
            w3: 12,
            s3: 16,
            w2: 20,
            s2: 24,
        }
    }

    #[test]
    fn causal_block_union_arena_plan_is_deterministic_disjoint_and_deduplicated() {
        let a = fake_gpu_identity('a');
        let b = fake_gpu_identity('b');
        let shared = fake_gpu_identity('c');
        let layout = tiny_arena_layout();
        let first = plan_causal_block_union_arena(
            vec![
                (b.clone(), layout, false),
                (a.clone(), layout, false),
                (shared.clone(), layout, true),
                (a.clone(), layout, false),
            ],
            256,
            1024 * 1024,
        )
        .unwrap();
        let second = plan_causal_block_union_arena(
            vec![
                (a, layout, false),
                (shared, layout, true),
                (b, layout, false),
            ],
            256,
            1024 * 1024,
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.entries.len(), 3);
        assert_eq!(first.unique_routed_identities, 2);
        assert_eq!(first.shared_identities, 1);
        assert_eq!(first.logical_payload_bytes, layout.total().unwrap() * 3);

        let mut spans = first
            .entries
            .values()
            .flat_map(|entry| {
                let view = &entry.view;
                [view.w1, view.s1, view.w3, view.s3, view.w2, view.s2]
            })
            .collect::<Vec<_>>();
        spans.sort_by_key(|span| span.offset);
        let mut previous_end = 0u64;
        for span in spans {
            assert_eq!(span.offset % 256, 0);
            assert!(span.offset >= previous_end);
            previous_end = span.offset + span.bytes;
            assert!(previous_end <= first.arena_bytes);
        }
    }

    #[test]
    fn causal_block_union_arena_route_lookup_uses_identity_not_route_slot() {
        let a = fake_gpu_identity('a');
        let b = fake_gpu_identity('b');
        let shared = fake_gpu_identity('c');
        let layout = tiny_arena_layout();
        let plan = plan_causal_block_union_arena(
            vec![
                (a.clone(), layout, false),
                (b.clone(), layout, false),
                (shared, layout, true),
            ],
            64,
            1024 * 1024,
        )
        .unwrap();
        let route = [(&b, 0.75f32), (&a, 0.20f32), (&b, 0.05f32)];
        let resolved = route
            .iter()
            .map(|(identity, weight)| {
                (plan.entries.get(*identity).unwrap().view.w1.offset, *weight)
            })
            .collect::<Vec<_>>();
        assert_eq!(resolved[0].0, resolved[2].0);
        assert_ne!(resolved[0].0, resolved[1].0);
        assert_eq!(
            resolved.iter().map(|row| row.1).collect::<Vec<_>>(),
            vec![0.75, 0.20, 0.05]
        );
    }

    #[test]
    fn causal_block_union_arena_enforces_checked_hard_bounds() {
        let eight_gib = 8u64 * 1024 * 1024 * 1024;
        let mut cursor = 0u64;
        let span = place_causal_block_arena_span(&mut cursor, eight_gib, 256, eight_gib).unwrap();
        assert_eq!(span.offset, 0);
        assert_eq!(cursor, eight_gib);
        let mut cursor = 0u64;
        assert!(place_causal_block_arena_span(&mut cursor, eight_gib + 4, 256, eight_gib).is_err());
        assert!(checked_align_up(u64::MAX, 256).is_err());
        let mut cursor = u64::MAX - 1;
        assert!(place_causal_block_arena_span(&mut cursor, 4, 4, u64::MAX).is_err());

        let layout = tiny_arena_layout();
        assert!(plan_causal_block_union_arena(
            vec![(fake_gpu_identity('s'), layout, true)],
            256,
            eight_gib + 1,
        )
        .is_err());
    }

    #[test]
    fn causal_block_union_arena_telemetry_proves_single_batch_upload() {
        let layout = tiny_arena_layout();
        let plan = plan_causal_block_union_arena(
            vec![
                (fake_gpu_identity('a'), layout, false),
                (fake_gpu_identity('s'), layout, true),
            ],
            256,
            1024 * 1024,
        )
        .unwrap();
        let telemetry = CausalBlockUnionArenaTelemetry::from_plan(
            &plan,
            5,
            1.0,
            2.0,
            3.0,
            true,
            CAUSAL_BLOCK_UNION_ARENA_MAX_BYTES,
        )
        .unwrap();
        assert!(telemetry.enabled);
        assert!(telemetry.persistent_worker_buffers);
        assert_eq!(
            telemetry.logical_capacity_bytes,
            CAUSAL_BLOCK_UNION_ARENA_MAX_BYTES
        );
        assert_eq!(telemetry.staging_allocations, 1);
        assert_eq!(telemetry.device_allocations, 1);
        assert_eq!(telemetry.upload_submissions, 1);
        assert_eq!(telemetry.copy_commands, 1);
        assert_eq!(telemetry.copy_regions, 1);
        assert_eq!(telemetry.actual_uploaded_bytes, plan.arena_bytes);
        assert_eq!(telemetry.unique_tensor_views, 12);
        assert_eq!(telemetry.unique_routed_identities, 1);
        assert_eq!(telemetry.shared_identities, 1);
        assert_eq!(telemetry.reused_routed_references, 5);
        assert!(telemetry.strict_sha_identity);

        let reused = CausalBlockUnionArenaTelemetry::from_plan(
            &plan,
            5,
            1.0,
            0.0,
            3.0,
            false,
            CAUSAL_BLOCK_UNION_ARENA_MAX_BYTES,
        )
        .unwrap();
        assert_eq!(reused.staging_allocations, 0);
        assert_eq!(reused.device_allocations, 0);
        assert_eq!(reused.arena_allocate_ms, 0.0);
    }

    #[test]
    fn writeback_batch_verify_switch_is_explicit_and_default_off() {
        use std::ffi::OsStr;

        assert!(!writeback_batch_verify_enabled_from(false, None).unwrap());
        assert!(!writeback_batch_verify_enabled_from(false, Some(OsStr::new("0"))).unwrap());
        assert!(writeback_batch_verify_enabled_from(true, None).unwrap());
        assert!(writeback_batch_verify_enabled_from(true, Some(OsStr::new("0"))).unwrap());
        assert!(writeback_batch_verify_enabled_from(false, Some(OsStr::new("1"))).unwrap());
        assert!(writeback_batch_verify_enabled_from(false, Some(OsStr::new("true"))).is_err());
    }

    #[test]
    fn writeback_batch_paths_are_canonical_and_range_cache_bounded() {
        let fixture = FixtureDir::new();
        let range_cache = fixture.0.join("range_cache");
        std::fs::create_dir(&range_cache).unwrap();
        let range_cache = range_cache.canonicalize().unwrap();
        let inside = range_cache.join("inside.bin");
        std::fs::write(&inside, b"safe").unwrap();
        let payload = FullDepthBridgePayload {
            tensor: "layers.0.ffn.experts.0.w1.weight".to_string(),
            kind: "routed".to_string(),
            expert_id: Some(0),
            dtype: "I8".to_string(),
            shape: vec![4],
            bytes: 4,
            path: inside.clone(),
            sha256: sha256_bytes(b"safe"),
        };
        let requests =
            prepare_writeback_batch_requests(std::slice::from_ref(&payload), 1, &range_cache)
                .unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].path, inside.canonicalize().unwrap());

        let outside = fixture.0.join("outside.bin");
        std::fs::write(&outside, b"safe").unwrap();
        let escaped = FullDepthBridgePayload {
            path: outside,
            ..payload.clone()
        };
        assert!(prepare_writeback_batch_requests(&[escaped], 1, &range_cache).is_err());

        let wrong_extension = range_cache.join("inside.dat");
        std::fs::write(&wrong_extension, b"safe").unwrap();
        let wrong_extension = FullDepthBridgePayload {
            path: wrong_extension,
            ..payload
        };
        assert!(prepare_writeback_batch_requests(&[wrong_extension], 1, &range_cache).is_err());
    }

    #[test]
    fn writeback_batch_preverification_makes_existing_loader_all_hits() {
        let fixture = FixtureDir::new();
        let capture_root = fixture.0.join("capture");
        let range_cache = fixture.0.join("range_cache");
        std::fs::create_dir(&capture_root).unwrap();
        std::fs::create_dir(&range_cache).unwrap();
        let capture_root = capture_root.canonicalize().unwrap();
        let range_cache = range_cache.canonicalize().unwrap();
        let path = range_cache.join("payload.bin");
        std::fs::write(&path, b"verified").unwrap();
        let tensor = "layers.0.ffn.experts.0.w1.weight";
        let sha256 = sha256_bytes(b"verified");
        let manifest = fake_bridge_manifest(vec![FullDepthBridgePayload {
            tensor: tensor.to_string(),
            kind: "routed".to_string(),
            expert_id: Some(0),
            dtype: "I8".to_string(),
            shape: vec![8],
            bytes: 8,
            path: path.clone(),
            sha256: sha256.clone(),
        }]);
        let mut cache = VerifiedPayloadCache::new(64).unwrap();

        let pending = preverify_writeback_payload_batch(
            &manifest,
            &capture_root,
            &range_cache,
            &mut cache,
            true,
        )
        .unwrap();
        let payload = read_verified_payload_cached_with_root(
            &mut cache,
            &range_cache,
            &path,
            8,
            &sha256,
            tensor,
        )
        .unwrap();
        assert_eq!(&*payload, b"verified");
        let receipt = pending.finish_before_compute(&cache).unwrap();

        assert!(receipt.enabled);
        assert_eq!(receipt.batch_entries, 1);
        assert_eq!(receipt.batch_hits, 0);
        assert_eq!(receipt.batch_misses, 1);
        assert_eq!(receipt.batch_disk_bytes_read, 8);
        assert_eq!(receipt.concurrency_limit, 8);
        assert_eq!(receipt.followup_cached_loader_hits, 1);
        assert!(receipt.all_verified_before_compute);
    }

    #[test]
    fn payload_verification_receipt_is_order_independent_and_strong() {
        let a_sha = "a".repeat(64);
        let b_sha = "b".repeat(64);
        let a = VerifiedPayloadIdentity {
            tensor: "layers.0.attn.wq_a.scale",
            bytes: 32,
            sha256: &a_sha,
        };
        let b = VerifiedPayloadIdentity {
            tensor: "layers.0.attn.wq_a.weight",
            bytes: 256,
            sha256: &b_sha,
        };
        let forward = payload_verification_receipt([a, b]).unwrap();
        let reverse = payload_verification_receipt([b, a]).unwrap();

        assert_eq!(forward, reverse);
        assert_eq!(forward.verification_owner, "rust_vulkan_worker");
        assert_eq!(forward.verified_count, 2);
        assert_eq!(forward.verified_bytes, 288);
        assert_eq!(forward.payload_identity_sha256.len(), 64);
        assert!(forward.verified_before_compute);
        assert_eq!(
            forward.verification_scope,
            "all_listed_payloads_before_corresponding_gpu_compute"
        );

        let json = serde_json::to_value(&forward).unwrap();
        assert_eq!(json["verified_count"], 2);
        assert_eq!(json["verified_bytes"], 288);
        assert_eq!(json["verified_before_compute"], true);
        assert_eq!(
            json["payload_identity_sha256"],
            forward.payload_identity_sha256
        );
    }

    #[test]
    fn payload_verification_receipt_binds_tensor_bytes_and_sha() {
        let sha = "c".repeat(64);
        let changed_sha = "d".repeat(64);
        let base = payload_verification_receipt([VerifiedPayloadIdentity {
            tensor: "layers.0.ffn.experts.1.w1.weight",
            bytes: 64,
            sha256: &sha,
        }])
        .unwrap();
        let changed_bytes = payload_verification_receipt([VerifiedPayloadIdentity {
            tensor: "layers.0.ffn.experts.1.w1.weight",
            bytes: 65,
            sha256: &sha,
        }])
        .unwrap();
        let changed_identity = payload_verification_receipt([VerifiedPayloadIdentity {
            tensor: "layers.0.ffn.experts.1.w1.weight",
            bytes: 64,
            sha256: &changed_sha,
        }])
        .unwrap();

        assert_ne!(
            base.payload_identity_sha256,
            changed_bytes.payload_identity_sha256
        );
        assert_ne!(
            base.payload_identity_sha256,
            changed_identity.payload_identity_sha256
        );
    }

    #[test]
    fn payload_verification_receipt_rejects_ambiguous_or_invalid_identity() {
        let sha = "e".repeat(64);
        let duplicate = VerifiedPayloadIdentity {
            tensor: "layers.0.ffn.shared_experts.w1.weight",
            bytes: 64,
            sha256: &sha,
        };
        assert!(payload_verification_receipt([duplicate, duplicate]).is_err());

        let uppercase_sha = "F".repeat(64);
        assert!(payload_verification_receipt([VerifiedPayloadIdentity {
            tensor: "layers.0.ffn.shared_experts.w1.scale",
            bytes: 64,
            sha256: &uppercase_sha,
        }])
        .is_err());
        assert!(payload_verification_receipt(std::iter::empty()).is_err());
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
            false,
        )
        .unwrap();
        assert!(reusable.enabled);
        assert_eq!(reusable.request_uploads, 1);
        assert_eq!(reusable.request_uploaded_bytes, 105_399_808);
        assert_eq!(reusable.weight_tensor_slots_reused, 42);
        assert!(reusable.workspace_reused);
        assert!(reusable.resident_cache_isolated);
        assert!(!reusable.shared_resident_cache_hybrid);

        let hybrid = WritebackReusableGpuSlotTelemetry::for_successful_request(
            Some((device_bytes, staging_bytes)),
            false,
            true,
        )
        .unwrap();
        assert!(hybrid.enabled);
        assert_eq!(hybrid.request_uploaded_bytes, 80_232_448);
        assert_eq!(hybrid.weight_tensor_slots_reused, 36);
        assert!(hybrid.shared_resident_cache_hybrid);

        let resident =
            WritebackReusableGpuSlotTelemetry::for_successful_request(None, true, false).unwrap();
        assert!(!resident.enabled);
        assert_eq!(resident.request_uploaded_bytes, 0);
        assert!(WritebackReusableGpuSlotTelemetry::for_successful_request(
            Some((device_bytes, staging_bytes)),
            true,
            false,
        )
        .is_err());
        assert!(
            WritebackReusableGpuSlotTelemetry::for_successful_request(None, false, false,).is_err()
        );
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
    fn shared_gpu_payload_cache_is_explicit_and_fixed_to_two_gib() {
        assert_eq!(DEFAULT_SHARED_GPU_PAYLOAD_CACHE_GIB, 0);
        assert_eq!(
            shared_gpu_payload_cache_capacity_bytes_from(None).unwrap(),
            None
        );
        assert_eq!(
            shared_gpu_payload_cache_capacity_bytes_from(Some("0")).unwrap(),
            None
        );
        assert_eq!(
            shared_gpu_payload_cache_capacity_bytes_from(Some("2")).unwrap(),
            Some(2_u64 * 1024 * 1024 * 1024)
        );
        assert!(shared_gpu_payload_cache_capacity_bytes_from(Some("1")).is_err());
        assert!(shared_gpu_payload_cache_capacity_bytes_from(Some("3")).is_err());
        assert!(shared_gpu_payload_cache_capacity_bytes_from(Some("two")).is_err());
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
