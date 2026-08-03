//! FullDepth43 pre-compression window-KV attention Vulkan 路径。
//!
//! position1/2 均尚未形成 ratio4/128 compressed block。本核接收连续的已提交
//! window 前缀与当前 KV，position-aware 地旋转 query/current KV，并在输出端执行
//! inverse RoPE。position3 起还必须消费 compressed main/indexer cache，不属于本模块。

use crate::compute::{ComputePipeline, DescriptorBinder, StorageBufferSlice};
use crate::s14_position0_attention::{S14_POSITION0_HEADS, S14_POSITION0_HEAD_DIM};
use crate::VulkanContext;
use anyhow::{bail, Result};
use ash::vk;

pub const S14_POSITION1_ATTENTION_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_position1_attention.spv"));

pub struct S14Position1AttentionPipeline {
    pipeline: ComputePipeline,
}

pub struct S14Position1AttentionDispatch {
    pub binder: DescriptorBinder,
    previous_count: u32,
}

/// 固定 revision `precompute_freqs_cis` 在指定 position 的 32 组 `(cos,sin)`。
/// ratio4 与 ratio128 使用同一 YARN 频率合同；ratio0 使用原生 10k base。
pub fn position_rope_cos_sin(position: u32, compress_ratio: u32) -> Result<[f32; 64]> {
    if !matches!(compress_ratio, 0 | 4 | 128) {
        bail!("position RoPE compress_ratio 只允许 0/4/128");
    }
    // 这些位模式由固定 revision 的 PyTorch F32
    // `precompute_freqs_cis(...)[1]` 产生。直接冻结位模式可避免 Rust
    // libm 与 torch.polar 的 pow/cos/sin 归约差异改变 BF16 边界。
    const RATIO0_BITS: [u32; 64] = [
        0x3f0a5140, 0x3f576aa4, 0x3f3b54b0, 0x3f2e7ace, 0x3f58940e, 0x3f087dba, 0x3f6992c6,
        0x3ed190f1, 0x3f734e6a, 0x3e9f393e, 0x3f78d5f0, 0x3e708f26, 0x3f7bf683, 0x3e35233a,
        0x3f7dba28, 0x3e0825f3, 0x3f7eb898, 0x3dcc7577, 0x3f7f47d1, 0x3d996f36, 0x3f7f9868,
        0x3d6636af, 0x3f7fc5bd, 0x3d2cacfa, 0x3f7fdf3c, 0x3d01815c, 0x3f7fed93, 0x3cc23ea6,
        0x3f7ff5a3, 0x3c91ab42, 0x3f7ffa2c, 0x3c5a7a4a, 0x3f7ffcb9, 0x3c23d657, 0x3f7ffe28,
        0x3bf5b919, 0x3f7ffef7, 0x3bb8445c, 0x3f7fff6b, 0x3b8a2e5c, 0x3f7fffac, 0x3b4f3e21,
        0x3f7fffd1, 0x3b1b6903, 0x3f7fffe5, 0x3ae91520, 0x3f7ffff1, 0x3aaec98b, 0x3f7ffff8,
        0x3a83126e, 0x3f7ffffb, 0x3a44948b, 0x3f7ffffd, 0x3a136a15, 0x3f7fffff, 0x39dd1725,
        0x3f7fffff, 0x39a5cb60, 0x3f800000, 0x3978a815, 0x3f800000, 0x393a7753, 0x3f800000,
        0x390bd472,
    ];
    const YARN_BITS: [u32; 64] = [
        0x3f0a5140, 0x3f576aa4, 0x3f45d205, 0x3f227d83, 0x3f63e85f, 0x3ee92ff2, 0x3f7295a1,
        0x3ea391dc, 0x3f79a06a, 0x3e6311ec, 0x3f7cfac6, 0x3e1cd5ce, 0x3f7e91fc, 0x3dd82576,
        0x3f7f52d6, 0x3d94c7ce, 0x3f7fae19, 0x3d4cb6f5, 0x3f7fd944, 0x3d0ccde1, 0x3f7fedaf,
        0x3cc1ab7a, 0x3f7ff757, 0x3c852f4d, 0x3f7ffbe7, 0x3c372cc5, 0x3f7ffe10, 0x3bfbece5,
        0x3f7fff16, 0x3bad3d29, 0x3f7fff91, 0x3b6e4215, 0x3f7fffd5, 0x3b147ad9, 0x3f7ffff0,
        0x3ab714dc, 0x3f7ffffa, 0x3a5ebdb4, 0x3f7ffffe, 0x3a0530ce, 0x3f7fffff, 0x399bb3af,
        0x3f800000, 0x39305978, 0x3f800000, 0x38be904d, 0x3f800000, 0x383e9b5f, 0x3f800000,
        0x37a3d70c, 0x3f800000, 0x36b443d0, 0x3f800000, 0x3677eba6, 0x3f800000, 0x362a7be8,
        0x3f800000, 0x35ea77ff, 0x3f800000, 0x35a13bdc, 0x3f800000, 0x355dbf30, 0x3f800000,
        0x35187c4d,
    ];
    const POSITION2_RATIO0_BITS: [u32; 64] = [
        0xbed51133, 0x3f68c7b7, 0x3d914d53, 0x3f7f5ad9, 0x3edce8b2, 0x3f66f20a, 0x3f2a3902,
        0x3f3f3512, 0x3f4e7bec, 0x3f17541b, 0x3f63be69, 0x3ee9d3b6, 0x3f6ffaa5, 0x3eb247f6,
        0x3f76f2f5, 0x3e86f082, 0x3f7ae5a5, 0x3e4b6ff9, 0x3f7d204f, 0x3e1900d3, 0x3f7e61f3,
        0x3de5d987, 0x3f7f170e, 0x3dac85ae, 0x3f7f7cf9, 0x3d8170c9, 0x3f7fb64e, 0x3d4230ab,
        0x3f7fd68e, 0x3d11a55c, 0x3f7fe8b1, 0x3cda7551, 0x3f7ff2e5, 0x3ca3d43e, 0x3f7ff8a1,
        0x3c75b754, 0x3f7ffbdb, 0x3c38439d, 0x3f7ffdab, 0x3c0a2e0c, 0x3f7ffeb0, 0x3bcf3ddd,
        0x3f7fff43, 0x3b9b68e7, 0x3f7fff96, 0x3b691508, 0x3f7fffc4, 0x3b2ec980, 0x3f7fffde,
        0x3b031269, 0x3f7fffed, 0x3ac49487, 0x3f7ffff5, 0x3a936a14, 0x3f7ffffa, 0x3a5d1723,
        0x3f7ffffd, 0x3a25cb5f, 0x3f7ffffe, 0x39f8a814, 0x3f7fffff, 0x39ba7753, 0x3f7fffff,
        0x398bd472,
    ];
    const POSITION2_YARN_BITS: [u32; 64] = [
        0xbed51133, 0x3f68c7b7, 0x3e46e743, 0x3f7b1fc9, 0x3f15cbd8, 0x3f4f992e, 0x3f4bbe77,
        0x3f1aff7d, 0x3f66d2e7, 0x3edd6ab8, 0x3f73fd57, 0x3e9afc19, 0x3f7a4c06, 0x3e56f06d,
        0x3f7d4c43, 0x3e14632b, 0x3f7eb898, 0x3dcc7577, 0x3f7f651c, 0x3d8cb893, 0x3f7fb6be,
        0x3d419d9f, 0x3f7fdd5b, 0x3d052acc, 0x3f7fef9e, 0x3cb729d7, 0x3f7ff841, 0x3c7beafd,
        0x3f7ffc56, 0x3c2d3c8a, 0x3f7ffe45, 0x3bee41ad, 0x3f7fff54, 0x3b947ac0, 0x3f7fffbf,
        0x3b3714d0, 0x3f7fffe8, 0x3adebdaf, 0x3f7ffff7, 0x3a8530cc, 0x3f7ffffd, 0x3a1bb3ae,
        0x3f7fffff, 0x39b05978, 0x3f800000, 0x393e904d, 0x3f800000, 0x38be9b5f, 0x3f800000,
        0x3823d70c, 0x3f800000, 0x373443d0, 0x3f800000, 0x36f7eba6, 0x3f800000, 0x36aa7be8,
        0x3f800000, 0x366a77ff, 0x3f800000, 0x36213bdc, 0x3f800000, 0x35ddbf30, 0x3f800000,
        0x35987c4d,
    ];
    let bits = match (position, compress_ratio == 0) {
        (0, _) => {
            let mut identity = [0u32; 64];
            for pair in identity.chunks_exact_mut(2) {
                pair[0] = 1.0f32.to_bits();
            }
            identity
        }
        (1, true) => RATIO0_BITS,
        (1, false) => YARN_BITS,
        (2, true) => POSITION2_RATIO0_BITS,
        (2, false) => POSITION2_YARN_BITS,
        _ => return computed_position_rope(position, compress_ratio),
    };
    let mut output = [0.0f32; 64];
    for (value, bits) in output.iter_mut().zip(bits) {
        *value = f32::from_bits(bits);
    }
    Ok(output)
}

fn computed_position_rope(position: u32, compress_ratio: u32) -> Result<[f32; 64]> {
    let original_seq_len = if compress_ratio == 0 { 0.0 } else { 65_536.0 };
    let base: f32 = if compress_ratio == 0 {
        10_000.0
    } else {
        160_000.0
    };
    let (low, high) = if compress_ratio == 0 {
        (0.0f32, 0.0f32)
    } else {
        let correction = |rotations: f64| {
            64.0 * (65_536.0 / (rotations * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * 160_000.0f64.ln())
        };
        (
            correction(32.0).floor().max(0.0) as f32,
            correction(1.0).ceil().min(63.0) as f32,
        )
    };
    let mut output = [0.0f32; 64];
    for index in 0..32 {
        let exponent = (index * 2) as f32 / 64.0;
        let mut frequency = 1.0 / base.powf(exponent);
        if original_seq_len > 0.0 {
            let linear = ((index as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let smooth = 1.0 - linear;
            frequency = frequency / 16.0 * (1.0 - smooth) + frequency * smooth;
        }
        let angle = position as f32 * frequency;
        let (sin, cos) = angle.sin_cos();
        output[index * 2] = cos;
        output[index * 2 + 1] = sin;
    }
    Ok(output)
}

impl S14Position1AttentionPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_POSITION1_ATTENTION_SPV, 6, 12)?,
        })
    }

    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        query: StorageBufferSlice<'_>,
        previous_kv: StorageBufferSlice<'_>,
        current_kv: StorageBufferSlice<'_>,
        sink: StorageBufferSlice<'_>,
        rope_cos_sin: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        position: u32,
        previous_count: u32,
    ) -> Result<S14Position1AttentionDispatch> {
        const QUERY_BYTES: u64 = S14_POSITION0_HEADS as u64 * S14_POSITION0_HEAD_DIM as u64 * 2;
        const KV_ROW_BYTES: u64 = S14_POSITION0_HEAD_DIM as u64 * 2;
        const SINK_BYTES: u64 = S14_POSITION0_HEADS as u64 * 4;
        const ROPE_BYTES: u64 = 32 * 2 * 4;
        if position == 0 || position > 127 || previous_count != position {
            bail!("pre-compression window attention 要求 position=previous_count=1..127");
        }
        let previous_bytes = KV_ROW_BYTES * u64::from(previous_count);
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (query.buffer, query.offset, QUERY_BYTES),
                (previous_kv.buffer, previous_kv.offset, previous_bytes),
                (current_kv.buffer, current_kv.offset, KV_ROW_BYTES),
                (sink.buffer, sink.offset, SINK_BYTES),
                (rope_cos_sin.buffer, rope_cos_sin.offset, ROPE_BYTES),
                (output.buffer, output.offset, QUERY_BYTES),
            ],
        )?;
        Ok(S14Position1AttentionDispatch {
            binder,
            previous_count,
        })
    }

    /// # Safety
    /// 所有 descriptor 资源必须活到 `command` 完成；调用前后由上层插入 compute barrier。
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Position1AttentionDispatch,
    ) {
        ctx.device.cmd_bind_pipeline(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline.pipeline,
        );
        ctx.device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline.layout,
            0,
            &[dispatch.binder.set],
            &[],
        );
        let mut push = [0u8; 12];
        push[..4].copy_from_slice(&S14_POSITION0_HEADS.to_le_bytes());
        push[4..8].copy_from_slice(&S14_POSITION0_HEAD_DIM.to_le_bytes());
        push[8..].copy_from_slice(&dispatch.previous_count.to_le_bytes());
        ctx.device.cmd_push_constants(
            command,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push,
        );
        ctx.device.cmd_dispatch(command, S14_POSITION0_HEADS, 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn sha256_f32(values: &[f32]) -> String {
        let mut hasher = Sha256::new();
        for value in values {
            hasher.update(value.to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    #[test]
    fn position_rope_contract_is_finite_unit_complex_ratio_and_position_bound() {
        let no_compression = position_rope_cos_sin(1, 0).unwrap();
        let ratio4 = position_rope_cos_sin(1, 4).unwrap();
        let ratio128 = position_rope_cos_sin(1, 128).unwrap();
        assert_eq!(ratio4, ratio128);
        assert_ne!(no_compression, ratio4);
        for table in [no_compression, ratio4] {
            for pair in table.chunks_exact(2) {
                assert!(pair.iter().all(|value| value.is_finite()));
                let norm = pair[0] * pair[0] + pair[1] * pair[1];
                assert!((norm - 1.0).abs() <= 2.0e-7);
            }
        }
        assert_ne!(position_rope_cos_sin(2, 4).unwrap(), ratio4);
        assert_eq!(
            position_rope_cos_sin(2, 4).unwrap()[..4],
            [
                f32::from_bits(0xbed51133),
                f32::from_bits(0x3f68c7b7),
                f32::from_bits(0x3e46e743),
                f32::from_bits(0x3f7b1fc9),
            ]
        );
        assert!(position_rope_cos_sin(1, 1).is_err());
    }

    #[test]
    fn computed_arbitrary_position_rope_matches_frozen_pytorch_f32_hashes() {
        let cases = [
            (
                3,
                0,
                "220c69c4c9776a18d30c914f187a2cf0d39771261a680900d36da14c4237fa28",
            ),
            (
                17,
                0,
                "45d912eb7a8dbe0c5ccf366b4d9ca502ed01fcebe0a9152a652f9cfcd58ee8f9",
            ),
            (
                127,
                0,
                "77663fadd6395fb40bf199cdc6fadc4a3609591a46837d671e5b63eb37bec391",
            ),
            (
                128,
                0,
                "5ff6a325b9edfe2add066cea5bb393e43c232e00e2e0cf7302efe3b2baaae2ef",
            ),
            (
                4096,
                0,
                "42e8b2c01fc7ce3c9df9bb6ce9302ef9a07c3fd8c156318dbb521272ca937b85",
            ),
            (
                3,
                4,
                "7261d5dd2f0d45f71dcd66db0b98d0eeaa96e27a1b8c346bb9763fc5c7585245",
            ),
            (
                17,
                4,
                "2b545c8a470f3c8bbc141fcaec4ff70ae8424029c0afa42556747bc31ebe5f38",
            ),
            (
                127,
                4,
                "de78c7be9935f93cdde797ff3c227a91f6734341b1f9e86eafc5ae0700bc4cee",
            ),
            (
                128,
                4,
                "94e228bac5e9e08168f32f0a9273f36919b3e5a17af5893c7b459bb46dc403af",
            ),
            (
                4096,
                4,
                "53d12d656de53a0abdafe7f8c67fbd1852fc1ac5bf7d2a1ed15027625fee314c",
            ),
        ];
        for (position, ratio, expected) in cases {
            let actual = position_rope_cos_sin(position, ratio).unwrap();
            assert_eq!(
                sha256_f32(&actual),
                expected,
                "position={position} ratio={ratio}"
            );
        }
        assert_eq!(
            position_rope_cos_sin(4096, 4).unwrap(),
            position_rope_cos_sin(4096, 128).unwrap()
        );
    }
}
