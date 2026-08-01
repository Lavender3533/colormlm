//! Dump GGUF metadata + a sample of tensor names so we can pin down
//! Qwen3 / Qwen3-MoE key naming (e.g. `qwen3.` vs `qwen3moe.`),
//! locate rope_theta / rms_eps / head_count, and check tensor dtypes.
//!
//! Usage:
//!   cargo run --release -p ssd_inference --example dump_metadata -- path/to/model.gguf

use anyhow::Result;
use gguf_reader::{GgufFile, MetaValue};

fn main() -> Result<()> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "../../models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf".to_string());

    println!("Opening {}", path);
    let g = GgufFile::open(&path)?;

    println!("\n=== ALL METADATA ({} keys) ===", g.metadata_keys().len());
    let mut keys: Vec<&String> = g.metadata_keys();
    keys.sort();
    for k in &keys {
        let v = g.metadata_value(k).unwrap();
        let summary = summarize_value(v);
        println!("  {:<60} {}", k, summary);
    }

    // Quick aggregate stats on tensor dtypes — finds the surprise Q6_K.
    println!("\n=== TENSOR DTYPE HISTOGRAM ===");
    let names = g.tensor_names();
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &names {
        let info = g.tensor_info(n).unwrap();
        let dt = format!("{:?}", info.ggml_dtype);
        *counts.entry(dt).or_insert(0) += 1;
    }
    for (dt, n) in &counts {
        println!("  {:>10}: {} tensors", dt, n);
    }

    // List a few representative tensors so we can lock down Qwen3-specific names
    println!("\n=== KEY TENSORS (Qwen3 attention + MoE landmarks) ===");
    let probes = [
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    ];
    for name in probes {
        match g.tensor_info(name) {
            Some(info) => {
                println!("  {:<40} dtype={:?} shape={:?}",
                    name, info.ggml_dtype, info.shape.dims());
            }
            None => println!("  {:<40} <missing>", name),
        }
    }

    Ok(())
}

fn summarize_value(v: &MetaValue) -> String {
    match v {
        MetaValue::U8(n)  => format!("U8={}", n),
        MetaValue::I8(n)  => format!("I8={}", n),
        MetaValue::U16(n) => format!("U16={}", n),
        MetaValue::I16(n) => format!("I16={}", n),
        MetaValue::U32(n) => format!("U32={}", n),
        MetaValue::I32(n) => format!("I32={}", n),
        MetaValue::U64(n) => format!("U64={}", n),
        MetaValue::I64(n) => format!("I64={}", n),
        MetaValue::F32(x) => format!("F32={}", x),
        MetaValue::F64(x) => format!("F64={}", x),
        MetaValue::Bool(b) => format!("Bool={}", b),
        MetaValue::String(s) => {
            if s.len() <= 80 { format!("Str=\"{}\"", s) }
            else { format!("Str(len={}) \"{}…\"", s.len(), &s[..77]) }
        }
        MetaValue::Array(arr) => {
            let head: Vec<String> = arr.iter().take(3).map(|x| match x {
                MetaValue::String(s) => format!("\"{}\"", truncate(s, 24)),
                other => short(other),
            }).collect();
            format!("Array[{}] [{}{}]",
                arr.len(),
                head.join(", "),
                if arr.len() > 3 { ", …" } else { "" })
        }
    }
}

fn short(v: &MetaValue) -> String {
    match v {
        MetaValue::U32(n) => n.to_string(),
        MetaValue::I32(n) => n.to_string(),
        MetaValue::U64(n) => n.to_string(),
        MetaValue::F32(x) => format!("{}", x),
        MetaValue::String(s) => format!("\"{}\"", truncate(s, 24)),
        other => format!("{:?}", other.value_type()),
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() }
    else { s.chars().take(n).collect::<String>() + "…" }
}
