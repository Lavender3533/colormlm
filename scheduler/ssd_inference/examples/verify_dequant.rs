//! Verify Q2K/Q3K GPU dequant against CPU reference.

use anyhow::Result;
use ash::vk;
use gguf_reader::{ExpertKind, MultiGgufFile};
use ssd_inference::buffer::GpuBuffer;
use ssd_inference::compute::{ComputePipeline, DescriptorBinder, DEQUANT_Q2_K_SPV, DEQUANT_Q3_K_SPV};
use ssd_inference::device::VulkanContext;
use ssd_inference::expert_reader::ExpertReader;
use ssd_inference::model::ModelConfig;
use ssd_inference::streaming_weights::{cpu_dequant_q2_k, cpu_dequant_q3_k};

fn main() -> Result<()> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "../../models/Qwen3-235B-A22B-UD-Q2_K_XL-00001-of-00002.gguf".into());

    let mg = MultiGgufFile::open(&path)?;
    let cfg = ModelConfig::from_multi_gguf(&mg)?;
    let ctx = VulkanContext::init()?;
    let reader = ExpertReader::from_multi_gguf(&mg, cfg.n_experts)?;

    let d = cfg.d_model as usize;
    let inter = cfg.moe_intermediate as usize;

    // Test Q2K: gate_exps at layer 0, expert 0
    let gate_size = reader.expert_size(0, ExpertKind::GateExps, 0).unwrap();
    let mut gate_bytes = vec![0u8; gate_size];
    reader.read_into(0, ExpertKind::GateExps, 0, &mut gate_bytes)?;

    let n_weights = d * inter;
    let n_blocks = n_weights / 256;
    println!("Q2K gate: {} bytes, {} weights, {} blocks", gate_size, n_weights, n_blocks);
    println!("  expected Q2K: {} bytes", n_blocks * 84);

    // CPU reference
    let cpu_fp32 = cpu_dequant_q2_k(&gate_bytes);
    println!("  CPU dequant: {} floats, first 8: {:?}", cpu_fp32.len(), &cpu_fp32[..8]);

    // GPU dequant
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
    let q_buf = GpuBuffer::new(&ctx, gate_size as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::empty(), true)?;
    unsafe { q_buf.write_at(0, &gate_bytes); }

    let fp32_buf = GpuBuffer::new_vram(&ctx, (n_weights * 4) as u64, usage)?;
    let readback = GpuBuffer::new(&ctx, (n_weights * 4) as u64,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::empty(), true)?;

    let pipe = ComputePipeline::new(&ctx, DEQUANT_Q2_K_SPV, 2, 4)?;
    let binder = DescriptorBinder::new(&ctx, &pipe, &[
        (&q_buf, gate_size as u64),
        (&fp32_buf, (n_weights * 4) as u64),
    ])?;

    unsafe {
        let cmd_pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT), None)?;
        let cb = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1))?[0];
        ctx.device.begin_command_buffer(cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        ctx.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
        ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
            pipe.layout, 0, &[binder.set], &[]);
        let push = (n_blocks as u32).to_le_bytes();
        ctx.device.cmd_push_constants(cb, pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
        ctx.device.cmd_dispatch(cb, n_blocks as u32, 1, 1);

        // barrier + readback
        let bar = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        ctx.device.cmd_pipeline_barrier(cb,
            vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(), &[bar], &[], &[]);
        ctx.device.cmd_copy_buffer(cb, fp32_buf.handle(), readback.handle(),
            &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: (n_weights * 4) as u64 }]);
        ctx.device.end_command_buffer(cb)?;

        let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cb_arr = [cb];
        ctx.device.queue_submit(ctx.q_graphics, &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(cmd_pool, None);
    }

    let mut gpu_fp32 = vec![0f32; n_weights];
    unsafe { std::ptr::copy_nonoverlapping(readback.mapped() as *const f32, gpu_fp32.as_mut_ptr(), n_weights); }
    println!("  GPU dequant: {} floats, first 8: {:?}", gpu_fp32.len(), &gpu_fp32[..8]);

    // Compare
    let mut max_err = 0f32;
    let mut n_mismatch = 0;
    for i in 0..n_weights {
        let err = (cpu_fp32[i] - gpu_fp32[i]).abs();
        if err > max_err { max_err = err; }
        if err > 1e-4 { n_mismatch += 1; }
    }
    println!("  max_err={:.6} mismatches(>1e-4)={}/{}", max_err, n_mismatch, n_weights);

    if n_mismatch > 0 {
        // Show first few mismatches
        for i in 0..n_weights.min(512) {
            let err = (cpu_fp32[i] - gpu_fp32[i]).abs();
            if err > 1e-4 {
                println!("    [{}] cpu={:.6} gpu={:.6} err={:.6}", i, cpu_fp32[i], gpu_fp32[i], err);
            }
        }
    }

    // Test Q3K: down_exps at layer 0, expert 0
    println!("\n--- Q3K test ---");
    let down_size = reader.expert_size(0, ExpertKind::DownExps, 0).unwrap();
    let mut down_bytes = vec![0u8; down_size];
    reader.read_into(0, ExpertKind::DownExps, 0, &mut down_bytes)?;
    let n_w_down = inter * d;
    let n_blk_down = n_w_down / 256;
    println!("Q3K down: {} bytes, {} weights, {} blocks (expected {})", down_size, n_w_down, n_blk_down, n_blk_down * 110);

    let cpu_down = cpu_dequant_q3_k(&down_bytes);
    println!("  CPU: first 8: {:?}", &cpu_down[..8]);
    // Find first non-zero
    let first_nz = cpu_down.iter().position(|&x| x != 0.0 && x != -0.0);
    if let Some(idx) = first_nz {
        println!("  CPU first non-zero at [{}]: {}", idx, cpu_down[idx]);
        println!("  CPU [{}..{}]: {:?}", idx, idx+8, &cpu_down[idx..idx+8]);
    }

    // GPU Q3K dequant
    let q3_buf = GpuBuffer::new(&ctx, down_size as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::empty(), true)?;
    unsafe { q3_buf.write_at(0, &down_bytes); }

    let fp32_down = GpuBuffer::new_vram(&ctx, (n_w_down * 4) as u64, usage)?;
    let rb_down = GpuBuffer::new(&ctx, (n_w_down * 4) as u64,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::empty(), true)?;

    let pipe3 = ComputePipeline::new(&ctx, DEQUANT_Q3_K_SPV, 2, 8)?;
    let binder3 = DescriptorBinder::new(&ctx, &pipe3, &[
        (&q3_buf, down_size as u64),
        (&fp32_down, (n_w_down * 4) as u64),
    ])?;

    unsafe {
        let cmd_pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT), None)?;
        let cb = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1))?[0];
        ctx.device.begin_command_buffer(cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        ctx.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe3.pipeline);
        ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
            pipe3.layout, 0, &[binder3.set], &[]);
        let mut push3 = Vec::with_capacity(8);
        push3.extend_from_slice(&(n_blk_down as u32).to_le_bytes());
        push3.extend_from_slice(&28u32.to_le_bytes());
        ctx.device.cmd_push_constants(cb, pipe3.layout, vk::ShaderStageFlags::COMPUTE, 0, &push3);
        ctx.device.cmd_dispatch(cb, n_blk_down as u32, 1, 1);

        let bar = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        ctx.device.cmd_pipeline_barrier(cb,
            vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(), &[bar], &[], &[]);
        ctx.device.cmd_copy_buffer(cb, fp32_down.handle(), rb_down.handle(),
            &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: (n_w_down * 4) as u64 }]);
        ctx.device.end_command_buffer(cb)?;

        let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cb_arr = [cb];
        ctx.device.queue_submit(ctx.q_graphics, &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(cmd_pool, None);
    }

    let mut gpu_down = vec![0f32; n_w_down];
    unsafe { std::ptr::copy_nonoverlapping(rb_down.mapped() as *const f32, gpu_down.as_mut_ptr(), n_w_down); }
    println!("  GPU: first 8: {:?}", &gpu_down[..8]);
    if let Some(idx) = first_nz {
        println!("  GPU [{}..{}]: {:?}", idx, idx+8, &gpu_down[idx..idx+8]);
    }

    let mut max_err3 = 0f32;
    let mut n_mm3 = 0;
    for i in 0..n_w_down {
        let err = (cpu_down[i] - gpu_down[i]).abs();
        if err > max_err3 { max_err3 = err; }
        if err > 1e-4 { n_mm3 += 1; }
    }
    println!("  max_err={:.6} mismatches(>1e-4)={}/{}", max_err3, n_mm3, n_w_down);

    if n_mm3 > 0 {
        let mut shown = 0;
        for i in 0..n_w_down {
            let err = (cpu_down[i] - gpu_down[i]).abs();
            if err > 1e-4 && shown < 20 {
                println!("    [{}] cpu={:.6} gpu={:.6}", i, cpu_down[i], gpu_down[i]);
                shown += 1;
            }
        }
    }

    binder3.destroy(&ctx);
    pipe3.destroy(&ctx);
    q3_buf.destroy(&ctx);
    fp32_down.destroy(&ctx);
    rb_down.destroy(&ctx);

    binder.destroy(&ctx);
    pipe.destroy(&ctx);
    q_buf.destroy(&ctx);
    fp32_buf.destroy(&ctx);
    readback.destroy(&ctx);

    Ok(())
}
