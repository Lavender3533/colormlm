//! Debug: check tensor byte sizes for expert tensors at layer 6.
use gguf_reader::{ExpertKind, GgufFile};
fn main() {
    let g = GgufFile::open("D:/project/大模型ssd化/models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf").unwrap();
    for name in ["blk.6.ffn_gate_exps.weight", "blk.6.ffn_up_exps.weight", "blk.6.ffn_down_exps.weight"] {
        let info = g.tensor_info(name).unwrap();
        let bsz = g.tensor_byte_size(name).unwrap();
        println!("{name}: dtype={:?} shape={:?} byte_size={bsz} per_exp={}", info.ggml_dtype, info.shape.dims(), bsz / 128);
    }
    for kind in [ExpertKind::GateExps, ExpertKind::UpExps, ExpertKind::DownExps] {
        let sz = g.expert_slot_bytes(6, kind, 0, 128).unwrap().len();
        println!("expert_slot_bytes layer 6 {kind:?} slot 0 = {sz}");
    }
    // Also check ExpertReader sizes
    use ssd_inference::expert_reader::ExpertReader;
    let reader = ExpertReader::from_gguf(&g, "D:/project/大模型ssd化/models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf", 128).unwrap();
    for kind in [ExpertKind::GateExps, ExpertKind::UpExps, ExpertKind::DownExps] {
        let sz = reader.expert_size(6, kind, 0).unwrap();
        println!("ExpertReader layer 6 {kind:?} slot 0 = {sz}");
    }
}
