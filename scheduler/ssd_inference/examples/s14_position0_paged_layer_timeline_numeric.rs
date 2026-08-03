use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{Position0Asset, Position0WholeTokenManifest, FULL_DEPTH_LAYERS};
use ssd_inference::{
    compute::{ComputePipeline, DescriptorArena, VECTOR_ADD_SPV},
    s14_position0_mapped_assets::VerifiedMappedAssetStore,
    s14_position0_paged_layer_timeline::{
        S14Position0PagedLayerTimeline, S14Position0PagedLayerTimelineState,
    },
    s14_position0_paged_weight_arena::{
        S14Position0PagedWeightArena, S14Position0StaticLayerBinding,
    },
    s14_position0_weight_plan::S14Position0HybridWeightPlan,
    GpuBuffer, VulkanContext,
};
use std::{path::PathBuf, time::Instant};

const PROOF_ELEMENTS: usize = 4;
const PROOF_BYTES: u64 = (PROOF_ELEMENTS * std::mem::size_of::<f32>()) as u64;
const PROOF_STRIDE: u64 = 256;
const HEAD_WORD: u32 = 0x4845_4144;
const HC_WORD: u32 = 0x4843_4843;
const ARGMAX_WORD: u32 = 0x4152_474d;
const PROLOGUE_WORD: u32 = 0x5052_4f4c;
const TAIL_WORDS: u64 = 4;
const TAIL_BYTES: u64 = TAIL_WORDS * 4;
const PROLOGUE_COMPUTE_COMMAND: usize = FULL_DEPTH_LAYERS.len();
const TAIL_HC_COMMAND: usize = PROLOGUE_COMPUTE_COMMAND + 1;
const HEAD_COMPUTE_COMMAND: usize = TAIL_HC_COMMAND + 1;
const FINAL_ARGMAX_COMMAND: usize = HEAD_COMPUTE_COMMAND + 1;
const ORPHAN_PROLOGUE_COMMAND: usize = FINAL_ARGMAX_COMMAND + 1;
const HEAD_TRANSFER_COMMAND: usize = FULL_DEPTH_LAYERS.len();
const ORPHAN_TRANSFER_COMMAND: usize = HEAD_TRANSFER_COMMAND + 1;

#[derive(Debug)]
struct LayerStageEvidence {
    expected: [f32; PROOF_ELEMENTS],
    static_bytes: u64,
    routed_bytes: u64,
}

fn main() -> Result<()> {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
    );
    let manifest = Position0WholeTokenManifest::load(&manifest_path)?;
    let weight_plan = S14Position0HybridWeightPlan::build(&manifest)?;
    let ctx = VulkanContext::init()?;
    if !ctx.timeline_semaphore || !ctx.has_dedicated_transfer() {
        bail!("paged layer proof requires timeline semaphore and a dedicated transfer queue");
    }

    // 强制零静态缓存；全部 43 层都必须真实经过双页 transfer，不能用常驻命中跳过。
    let arena = S14Position0PagedWeightArena::new(&ctx, &weight_plan, Some(0))?;
    if arena.resident_static_layers() != 0 || arena.streamed_static_layers() != 43 {
        bail!("paged layer numeric did not force all 43 static layers through streaming");
    }
    let mut store = VerifiedMappedAssetStore::new(
        PathBuf::from("D:/models/Polaris-S14/range_cache").as_path(),
    )?;

    let static_staging = make_two_staging(&ctx, arena.plan().static_stream_bank_bytes)
        .context("allocate two paged static staging banks")?;
    let routed_staging = make_two_staging(&ctx, weight_plan.routed_bank_bytes)
        .context("allocate two paged routed staging banks")?;
    let head_staging = make_two_staging(&ctx, 4).context("allocate head staging banks")?;
    let head_device = make_two_vram(
        &ctx,
        4,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
    )
    .context("allocate head device banks")?;
    let proof_bytes = (FULL_DEPTH_LAYERS.len() as u64 - 1)
        .checked_mul(PROOF_STRIDE)
        .and_then(|bytes| bytes.checked_add(PROOF_BYTES))
        .ok_or_else(|| anyhow!("paged proof bytes overflow"))?;
    let proof = GpuBuffer::new_vram(
        &ctx,
        proof_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    let readback = GpuBuffer::new(
        &ctx,
        proof_bytes,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )?;
    let tail_marker = GpuBuffer::new_vram(
        &ctx,
        TAIL_BYTES,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    )?;
    let tail_readback = GpuBuffer::new(
        &ctx,
        TAIL_BYTES,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )?;

    let pipeline = ComputePipeline::new(&ctx, VECTOR_ADD_SPV, 3, 4)?;
    let mut descriptors = DescriptorArena::new(&ctx, FULL_DEPTH_LAYERS.len() as u32, 3)?;
    let (transfer_pool, compute_pool, transfer_commands, compute_commands) =
        allocate_commands(&ctx)?;
    record_all_commands(
        &ctx,
        &weight_plan,
        &arena,
        &static_staging,
        &routed_staging,
        &proof,
        &readback,
        &head_staging,
        &head_device,
        &tail_marker,
        &tail_readback,
        &pipeline,
        &mut descriptors,
        &transfer_commands,
        &compute_commands,
        proof_bytes,
    )?;

    // 注入发生在任何 queue submit 之前；一旦 staging 失败，timeline 必须 poison，
    // finish 不得产生 receipt，也不得执行 token host wait。
    let mut failure_probe = S14Position0PagedLayerTimeline::new(&ctx)?;
    let injected = unsafe {
        failure_probe.stage_and_submit_next(
            &ctx,
            FULL_DEPTH_LAYERS[0],
            transfer_commands[ORPHAN_TRANSFER_COMMAND],
            compute_commands[0],
            |_| Err(anyhow!("injected verified-staging failure")),
        )
    };
    if injected.is_ok()
        || failure_probe.finish_candidate(&ctx).is_ok()
        || failure_probe.stats().state != S14Position0PagedLayerTimelineState::Poisoned
        || failure_probe.stats().submitted_layers != 0
        || failure_probe.stats().token_host_waits != 0
    {
        bail!("paged layer failure probe did not fail closed");
    }
    failure_probe.destroy(&ctx);

    // transfer 已提交但 compute 尚未提交：drain_all 必须一次联合 wait 收敛 orphan。
    unsafe { head_staging[1].write_at(0, &0x4f52_5048u32.to_le_bytes()) };
    let mut orphan_probe = S14Position0PagedLayerTimeline::new(&ctx)?;
    let orphan_prologue = unsafe {
        orphan_probe
            .submit_prologue_compute_only(&ctx, compute_commands[ORPHAN_PROLOGUE_COMMAND])?
    };
    let pending = unsafe {
        orphan_probe.submit_next_layer_transfer(
            &ctx,
            FULL_DEPTH_LAYERS[0],
            transfer_commands[ORPHAN_TRANSFER_COMMAND],
            |_| Ok(()),
        )?
    };
    if orphan_prologue != 1 || pending.transfer_value != 1 {
        bail!("orphan probe prologue/transfer ticket drift");
    }
    orphan_probe.abort_pending()?;
    let orphan_receipt = orphan_probe.drain_all(&ctx)?;
    if !orphan_receipt.orphan_transfer
        || orphan_receipt.transfer_value != 1
        || orphan_receipt.compute_value != 1
        || orphan_receipt.host_wait_calls != 1
        || orphan_probe.stats().drain_host_waits != 1
        || orphan_probe.stats().token_host_waits != 0
        || orphan_probe.stats().state != S14Position0PagedLayerTimelineState::Drained
    {
        bail!("paged orphan drain receipt drift: {orphan_receipt:?}");
    }
    orphan_probe.destroy(&ctx);

    let started = Instant::now();
    let mut timeline = S14Position0PagedLayerTimeline::new(&ctx)?;
    let prologue_compute = unsafe {
        timeline.submit_prologue_compute_only(&ctx, compute_commands[PROLOGUE_COMPUTE_COMMAND])?
    };
    if prologue_compute != 1 {
        bail!("success candidate prologue ticket drift");
    }
    let mut evidence = Vec::<LayerStageEvidence>::with_capacity(FULL_DEPTH_LAYERS.len());
    for (index, &layer) in FULL_DEPTH_LAYERS.iter().enumerate() {
        let mut staged = None;
        unsafe {
            timeline.stage_and_submit_next(
                &ctx,
                layer,
                transfer_commands[index],
                compute_commands[index],
                |bank| {
                    if bank != index % 2 {
                        bail!("paged staging bank drift at L{layer}");
                    }
                    staged = Some(stage_real_layer(
                        index,
                        &manifest,
                        &weight_plan,
                        &arena,
                        &mut store,
                        &static_staging[bank],
                        &routed_staging[bank],
                    )?);
                    Ok(())
                },
            )?;
        }
        evidence.push(staged.ok_or_else(|| anyhow!("L{layer} staging evidence missing"))?);
    }
    let layer_final_compute = timeline.seal_layers()?;
    if layer_final_compute != 44
        || timeline.stats().token_host_waits != 0
        || timeline.stats().state != S14Position0PagedLayerTimelineState::TailOpen
    {
        bail!("L42 seal unexpectedly waited or returned wrong ticket");
    }
    let hc_compute =
        unsafe { timeline.submit_tail_compute_only(&ctx, compute_commands[TAIL_HC_COMMAND])? };
    unsafe { head_staging[0].write_at(0, &HEAD_WORD.to_le_bytes()) };
    let head_ticket = unsafe {
        timeline.stage_and_submit_head(
            &ctx,
            0,
            transfer_commands[HEAD_TRANSFER_COMMAND],
            compute_commands[HEAD_COMPUTE_COMMAND],
            |bank| {
                if bank != 0 {
                    bail!("head chunk0 staging bank drift");
                }
                Ok(())
            },
        )?
    };
    let final_compute =
        unsafe { timeline.submit_final_compute(&ctx, compute_commands[FINAL_ARGMAX_COMMAND])? };
    if hc_compute != 45 || head_ticket.compute_value != 46 || final_compute != 47 {
        bail!("tail/head/final compute ticket drift");
    }
    let receipt = timeline.finish_candidate(&ctx)?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;

    let distinct_reuse_pairs = evidence
        .iter()
        .enumerate()
        .skip(2)
        .filter(|(index, layer)| layer.expected != evidence[index - 2].expected)
        .count();
    if distinct_reuse_pairs != FULL_DEPTH_LAYERS.len() - 2 {
        bail!(
            "real proof vectors are not distinct across every bank reuse: distinct={distinct_reuse_pairs}"
        );
    }
    let max_abs_error = verify_gpu_proofs(&readback, &evidence)?;
    let tail_words = read_tail_words(&tail_readback)?;
    if tail_words != [PROLOGUE_WORD, HC_WORD, HEAD_WORD, ARGMAX_WORD] {
        bail!("tail/head/argmax marker drift: {tail_words:08x?}");
    }
    if receipt.layers != 43
        || receipt.prologue_compute_value != 1
        || receipt.layer_final_compute_value != 44
        || receipt.final_compute_value != 47
        || receipt.head_chunks != 1
        || receipt.tail_compute_segments != 2
        || receipt.device_bank_reuse_waits != 41
        || receipt.producer_transfer_waits != 41
        || receipt.token_host_waits != 1
        || receipt.completed_transfer_value != 44
        || receipt.completed_compute_value != 47
    {
        bail!("paged layer timeline receipt drift: {receipt:?}");
    }
    let static_bytes = evidence.iter().try_fold(0u64, |sum, layer| {
        sum.checked_add(layer.static_bytes)
            .ok_or_else(|| anyhow!("static byte counter overflow"))
    })?;
    let routed_bytes = evidence.iter().try_fold(0u64, |sum, layer| {
        sum.checked_add(layer.routed_bytes)
            .ok_or_else(|| anyhow!("routed byte counter overflow"))
    })?;
    let mapped = store.stats();
    println!(
        "status=pass gpu={:?} layers={} skipped_layers=0 static_banks=2 routed_banks=2 static_bytes={} routed_bytes={} total_transfer_payload_bytes={} transfer_submits={} compute_submits={} layer_final_compute={} final_compute={} head_chunks={} tail_compute_segments={} device_bank_reuse_waits={} producer_transfer_waits={} token_host_waits={} drain_host_waits=1 orphan_transfer_drained=true completed_transfer={} completed_compute={} proof_vectors={} distinct_reuse_pairs={} max_abs_error={:.9} tail_markers={:08x?} mapped_entries={} mapped_logical_bytes={} sha256_bytes={} fail_closed_probe=true token_input_used=false token_emitted=false wall_ms={wall_ms:.4}",
        ctx.gpu_name,
        receipt.layers,
        static_bytes,
        routed_bytes,
        static_bytes + routed_bytes,
        receipt.completed_transfer_value,
        receipt.completed_compute_value,
        receipt.layer_final_compute_value,
        receipt.final_compute_value,
        receipt.head_chunks,
        receipt.tail_compute_segments,
        receipt.device_bank_reuse_waits,
        receipt.producer_transfer_waits,
        receipt.token_host_waits,
        receipt.completed_transfer_value,
        receipt.completed_compute_value,
        evidence.len(),
        distinct_reuse_pairs,
        max_abs_error,
        tail_words,
        store.len(),
        mapped.mapped_logical_bytes,
        mapped.sha256_bytes,
    );

    timeline.destroy(&ctx);
    unsafe {
        ctx.device.destroy_command_pool(compute_pool, None);
        ctx.device.destroy_command_pool(transfer_pool, None);
    }
    descriptors.destroy(&ctx);
    pipeline.destroy(&ctx);
    tail_readback.destroy(&ctx);
    tail_marker.destroy(&ctx);
    readback.destroy(&ctx);
    proof.destroy(&ctx);
    for buffer in &head_device {
        buffer.destroy(&ctx);
    }
    for buffer in &head_staging {
        buffer.destroy(&ctx);
    }
    for buffer in &routed_staging {
        buffer.destroy(&ctx);
    }
    for buffer in &static_staging {
        buffer.destroy(&ctx);
    }
    arena.destroy(&ctx);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_all_commands(
    ctx: &VulkanContext,
    weight_plan: &S14Position0HybridWeightPlan,
    arena: &S14Position0PagedWeightArena,
    static_staging: &[GpuBuffer; 2],
    routed_staging: &[GpuBuffer; 2],
    proof: &GpuBuffer,
    readback: &GpuBuffer,
    head_staging: &[GpuBuffer; 2],
    head_device: &[GpuBuffer; 2],
    tail_marker: &GpuBuffer,
    tail_readback: &GpuBuffer,
    pipeline: &ComputePipeline,
    descriptors: &mut DescriptorArena,
    transfer_commands: &[vk::CommandBuffer],
    compute_commands: &[vk::CommandBuffer],
    proof_bytes: u64,
) -> Result<()> {
    for (index, &layer) in FULL_DEPTH_LAYERS.iter().enumerate() {
        let bank = index % 2;
        let (static_buffer, layout) = match arena.static_layer(layer)? {
            S14Position0StaticLayerBinding::Streamed {
                bank: actual_bank,
                buffer,
                layout,
            } if actual_bank == bank => (buffer, layout),
            _ => bail!("L{layer} did not bind to expected streamed bank {bank}"),
        };
        let routed_plan = &weight_plan.routed_layers[index];
        if routed_plan.layer != layer || routed_plan.bank != bank {
            bail!("L{layer} routed plan bank drift");
        }
        let input_a = find_proof_asset(layout, layer, "hc_attn_base")?;
        let input_b = find_proof_asset(layout, layer, "hc_ffn_base")?;
        let descriptor_set = descriptors.alloc_set(
            ctx,
            pipeline,
            &[
                (static_buffer, input_a.local_offset, PROOF_BYTES),
                (static_buffer, input_b.local_offset, PROOF_BYTES),
                (proof, index as u64 * PROOF_STRIDE, PROOF_BYTES),
            ],
        )?;

        unsafe {
            ctx.device.begin_command_buffer(
                transfer_commands[index],
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let static_copies = layout
                .assets
                .iter()
                .map(|asset| {
                    vk::BufferCopy::default()
                        .src_offset(asset.local_offset)
                        .dst_offset(asset.local_offset)
                        .size(asset.bytes)
                })
                .collect::<Vec<_>>();
            ctx.device.cmd_copy_buffer(
                transfer_commands[index],
                static_staging[bank].handle(),
                static_buffer.handle(),
                &static_copies,
            );
            let routed_copies = routed_plan
                .assets
                .iter()
                .map(|asset| {
                    vk::BufferCopy::default()
                        .src_offset(asset.offset)
                        .dst_offset(asset.offset)
                        .size(asset.bytes)
                })
                .collect::<Vec<_>>();
            ctx.device.cmd_copy_buffer(
                transfer_commands[index],
                routed_staging[bank].handle(),
                arena.routed(bank)?.handle(),
                &routed_copies,
            );
            ctx.device.end_command_buffer(transfer_commands[index])?;

            ctx.device.begin_command_buffer(
                compute_commands[index],
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            ctx.device.cmd_bind_pipeline(
                compute_commands[index],
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );
            ctx.device.cmd_bind_descriptor_sets(
                compute_commands[index],
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[descriptor_set],
                &[],
            );
            ctx.device.cmd_push_constants(
                compute_commands[index],
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &(PROOF_ELEMENTS as u32).to_le_bytes(),
            );
            ctx.device.cmd_dispatch(compute_commands[index], 1, 1, 1);
            ctx.device.end_command_buffer(compute_commands[index])?;
        }
    }
    unsafe {
        ctx.device.begin_command_buffer(
            compute_commands[PROLOGUE_COMPUTE_COMMAND],
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        ctx.device.cmd_fill_buffer(
            compute_commands[PROLOGUE_COMPUTE_COMMAND],
            tail_marker.handle(),
            0,
            4,
            PROLOGUE_WORD,
        );
        ctx.device
            .end_command_buffer(compute_commands[PROLOGUE_COMPUTE_COMMAND])?;

        ctx.device.begin_command_buffer(
            compute_commands[ORPHAN_PROLOGUE_COMMAND],
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        ctx.device.cmd_fill_buffer(
            compute_commands[ORPHAN_PROLOGUE_COMMAND],
            tail_marker.handle(),
            0,
            4,
            PROLOGUE_WORD,
        );
        ctx.device
            .end_command_buffer(compute_commands[ORPHAN_PROLOGUE_COMMAND])?;

        // final HC 占位段：写独立 marker，证明 L42 seal 后 compute timeline 仍可继续。
        ctx.device.begin_command_buffer(
            compute_commands[TAIL_HC_COMMAND],
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        ctx.device.cmd_fill_buffer(
            compute_commands[TAIL_HC_COMMAND],
            tail_marker.handle(),
            4,
            4,
            HC_WORD,
        );
        ctx.device
            .end_command_buffer(compute_commands[TAIL_HC_COMMAND])?;

        // head chunk0 的真实 transfer/compute 链；compute 把传入设备页的字复制到 marker[1]。
        ctx.device.begin_command_buffer(
            transfer_commands[HEAD_TRANSFER_COMMAND],
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        ctx.device.cmd_copy_buffer(
            transfer_commands[HEAD_TRANSFER_COMMAND],
            head_staging[0].handle(),
            head_device[0].handle(),
            &[vk::BufferCopy::default().size(4)],
        );
        ctx.device
            .end_command_buffer(transfer_commands[HEAD_TRANSFER_COMMAND])?;

        // orphan probe 使用独立 command，避免与成功候选重复提交同一 ONE_TIME command。
        ctx.device.begin_command_buffer(
            transfer_commands[ORPHAN_TRANSFER_COMMAND],
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        ctx.device.cmd_copy_buffer(
            transfer_commands[ORPHAN_TRANSFER_COMMAND],
            head_staging[1].handle(),
            head_device[1].handle(),
            &[vk::BufferCopy::default().size(4)],
        );
        ctx.device
            .end_command_buffer(transfer_commands[ORPHAN_TRANSFER_COMMAND])?;

        ctx.device.begin_command_buffer(
            compute_commands[HEAD_COMPUTE_COMMAND],
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        ctx.device.cmd_copy_buffer(
            compute_commands[HEAD_COMPUTE_COMMAND],
            head_device[0].handle(),
            tail_marker.handle(),
            &[vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(8)
                .size(4)],
        );
        ctx.device
            .end_command_buffer(compute_commands[HEAD_COMPUTE_COMMAND])?;

        // terminal argmax/readback 段是最终 compute ticket；proof 回读从 L42 移到这里。
        ctx.device.begin_command_buffer(
            compute_commands[FINAL_ARGMAX_COMMAND],
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        ctx.device.cmd_fill_buffer(
            compute_commands[FINAL_ARGMAX_COMMAND],
            tail_marker.handle(),
            12,
            4,
            ARGMAX_WORD,
        );
        let barriers = [
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .buffer(proof.handle())
                .offset(0)
                .size(proof_bytes),
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .buffer(tail_marker.handle())
                .offset(0)
                .size(TAIL_BYTES),
        ];
        ctx.device.cmd_pipeline_barrier(
            compute_commands[FINAL_ARGMAX_COMMAND],
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &barriers,
            &[],
        );
        ctx.device.cmd_copy_buffer(
            compute_commands[FINAL_ARGMAX_COMMAND],
            proof.handle(),
            readback.handle(),
            &[vk::BufferCopy::default().size(proof_bytes)],
        );
        ctx.device.cmd_copy_buffer(
            compute_commands[FINAL_ARGMAX_COMMAND],
            tail_marker.handle(),
            tail_readback.handle(),
            &[vk::BufferCopy::default().size(TAIL_BYTES)],
        );
        ctx.device
            .end_command_buffer(compute_commands[FINAL_ARGMAX_COMMAND])?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn stage_real_layer(
    index: usize,
    manifest: &Position0WholeTokenManifest,
    weight_plan: &S14Position0HybridWeightPlan,
    arena: &S14Position0PagedWeightArena,
    store: &mut VerifiedMappedAssetStore,
    static_staging: &GpuBuffer,
    routed_staging: &GpuBuffer,
) -> Result<LayerStageEvidence> {
    let layer = FULL_DEPTH_LAYERS[index];
    let manifest_layer = &manifest.layers[index];
    if manifest_layer.layer != layer {
        bail!("manifest layer order drift at L{layer}");
    }
    let physical = &arena.plan().physical.static_layers[index];
    let static_assets = manifest_layer
        .assets
        .non_expert
        .iter()
        .chain(&manifest_layer.assets.router)
        .chain(&manifest_layer.assets.shared)
        .cloned()
        .collect::<Vec<Position0Asset>>();
    let static_mapped = store.map_verified_batch(&static_assets)?;
    if static_mapped.len() != physical.assets.len() {
        bail!("L{layer} static mapped/physical count drift");
    }
    let mut proof_a = None;
    let mut proof_b = None;
    let mut static_bytes = 0u64;
    for asset in &physical.assets {
        let mapped = static_mapped
            .iter()
            .find(|mapped| mapped.tensor() == asset.tensor)
            .ok_or_else(|| anyhow!("L{layer} missing mapped static tensor {}", asset.tensor))?;
        if mapped.bytes().len() as u64 != asset.bytes
            || asset.local_offset + asset.bytes > static_staging.size()
        {
            bail!("L{layer} static payload range drift: {}", asset.tensor);
        }
        unsafe { static_staging.write_at(usize::try_from(asset.local_offset)?, mapped.bytes()) };
        static_bytes = static_bytes
            .checked_add(asset.bytes)
            .ok_or_else(|| anyhow!("L{layer} static bytes overflow"))?;
        if asset.tensor == format!("layers.{layer}.hc_attn_base") {
            proof_a = Some(read_f32x4(mapped.bytes(), &asset.tensor)?);
        }
        if asset.tensor == format!("layers.{layer}.hc_ffn_base") {
            proof_b = Some(read_f32x4(mapped.bytes(), &asset.tensor)?);
        }
    }

    let routed_plan = &weight_plan.routed_layers[index];
    let routed_mapped = store.map_verified_batch(&manifest_layer.assets.routed)?;
    if routed_mapped.len() != routed_plan.assets.len() {
        bail!("L{layer} routed mapped/plan count drift");
    }
    for (placement, mapped) in routed_plan.assets.iter().zip(&routed_mapped) {
        if mapped.tensor() != placement.tensor
            || mapped.bytes().len() as u64 != placement.bytes
            || placement.offset + placement.bytes > routed_staging.size()
        {
            bail!("L{layer} routed payload range drift: {}", placement.tensor);
        }
        unsafe { routed_staging.write_at(usize::try_from(placement.offset)?, mapped.bytes()) };
    }

    let proof_a = proof_a.ok_or_else(|| anyhow!("L{layer} hc_attn_base proof missing"))?;
    let proof_b = proof_b.ok_or_else(|| anyhow!("L{layer} hc_ffn_base proof missing"))?;
    let mut expected = [0.0f32; PROOF_ELEMENTS];
    for element in 0..PROOF_ELEMENTS {
        expected[element] = proof_a[element] + proof_b[element];
        if !expected[element].is_finite() {
            bail!("L{layer} real proof sum is non-finite");
        }
    }
    Ok(LayerStageEvidence {
        expected,
        static_bytes,
        routed_bytes: routed_plan.logical_bytes,
    })
}

fn find_proof_asset<'a>(
    layout: &'a ssd_inference::s14_position0_hybrid_weight_arena::S14Position0StaticLayerLayout,
    layer: u8,
    suffix: &str,
) -> Result<&'a ssd_inference::s14_position0_hybrid_weight_arena::S14Position0PhysicalAssetPlacement>
{
    let tensor = format!("layers.{layer}.{suffix}");
    let asset = layout
        .assets
        .iter()
        .find(|asset| asset.tensor == tensor)
        .ok_or_else(|| anyhow!("proof tensor missing: {tensor}"))?;
    if asset.bytes < PROOF_BYTES || asset.local_offset % 256 != 0 {
        bail!("proof tensor range/alignment drift: {tensor}");
    }
    Ok(asset)
}

fn read_f32x4(bytes: &[u8], tensor: &str) -> Result<[f32; PROOF_ELEMENTS]> {
    if bytes.len() < PROOF_BYTES as usize {
        bail!("proof tensor is too small: {tensor}");
    }
    let mut values = [0.0f32; PROOF_ELEMENTS];
    for (index, value) in values.iter_mut().enumerate() {
        let start = index * 4;
        *value = f32::from_le_bytes(bytes[start..start + 4].try_into()?);
        if !value.is_finite() {
            bail!("proof tensor has non-finite value: {tensor}");
        }
    }
    Ok(values)
}

fn verify_gpu_proofs(readback: &GpuBuffer, evidence: &[LayerStageEvidence]) -> Result<f32> {
    let bytes = unsafe {
        std::slice::from_raw_parts(readback.mapped() as *const u8, readback.size() as usize)
    };
    let mut max_abs_error = 0.0f32;
    for (index, layer) in evidence.iter().enumerate() {
        let offset = usize::try_from(index as u64 * PROOF_STRIDE)?;
        for element in 0..PROOF_ELEMENTS {
            let start = offset + element * 4;
            let actual = f32::from_le_bytes(bytes[start..start + 4].try_into()?);
            let expected = layer.expected[element];
            if !actual.is_finite() {
                bail!("GPU proof is non-finite at layer={index} element={element}");
            }
            let error = (actual - expected).abs();
            max_abs_error = max_abs_error.max(error);
            let tolerance = 1e-6f32.max(expected.abs() * 1e-6);
            if error > tolerance {
                bail!(
                    "paged bank overwrite proof mismatch: layer={index} element={element} expected={expected} actual={actual} error={error}"
                );
            }
        }
    }
    Ok(max_abs_error)
}

fn read_tail_words(readback: &GpuBuffer) -> Result<[u32; TAIL_WORDS as usize]> {
    if readback.size() != TAIL_BYTES {
        bail!("tail readback byte size drift");
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(readback.mapped() as *const u8, readback.size() as usize)
    };
    let mut words = [0u32; TAIL_WORDS as usize];
    for (index, word) in words.iter_mut().enumerate() {
        let start = index * 4;
        *word = u32::from_le_bytes(bytes[start..start + 4].try_into()?);
    }
    Ok(words)
}

fn make_two_staging(ctx: &VulkanContext, bytes: u64) -> Result<[GpuBuffer; 2]> {
    let first = GpuBuffer::new_staging(ctx, bytes)?;
    match GpuBuffer::new_staging(ctx, bytes) {
        Ok(second) => Ok([first, second]),
        Err(error) => {
            first.destroy(ctx);
            Err(error.into())
        }
    }
}

fn make_two_vram(
    ctx: &VulkanContext,
    bytes: u64,
    usage: vk::BufferUsageFlags,
) -> Result<[GpuBuffer; 2]> {
    let first = GpuBuffer::new_vram(ctx, bytes, usage)?;
    match GpuBuffer::new_vram(ctx, bytes, usage) {
        Ok(second) => Ok([first, second]),
        Err(error) => {
            first.destroy(ctx);
            Err(error.into())
        }
    }
}

type Commands = (
    vk::CommandPool,
    vk::CommandPool,
    Vec<vk::CommandBuffer>,
    Vec<vk::CommandBuffer>,
);

fn allocate_commands(ctx: &VulkanContext) -> Result<Commands> {
    unsafe {
        let transfer_pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.qf_transfer),
            None,
        )?;
        let compute_pool = match ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.qf_graphics),
            None,
        ) {
            Ok(pool) => pool,
            Err(error) => {
                ctx.device.destroy_command_pool(transfer_pool, None);
                return Err(error.into());
            }
        };
        let allocate = |pool, count| {
            ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(count),
            )
        };
        let transfer_commands = match allocate(transfer_pool, (ORPHAN_TRANSFER_COMMAND + 1) as u32)
        {
            Ok(commands) => commands,
            Err(error) => {
                ctx.device.destroy_command_pool(compute_pool, None);
                ctx.device.destroy_command_pool(transfer_pool, None);
                return Err(error.into());
            }
        };
        let compute_commands = match allocate(compute_pool, (ORPHAN_PROLOGUE_COMMAND + 1) as u32) {
            Ok(commands) => commands,
            Err(error) => {
                ctx.device.destroy_command_pool(compute_pool, None);
                ctx.device.destroy_command_pool(transfer_pool, None);
                return Err(error.into());
            }
        };
        Ok((
            transfer_pool,
            compute_pool,
            transfer_commands,
            compute_commands,
        ))
    }
}
