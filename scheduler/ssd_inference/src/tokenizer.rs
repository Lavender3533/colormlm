//! Tokenizer for Qwen3-MoE — extracts BPE vocab + merges from GGUF metadata
//! and constructs a `tokenizers::Tokenizer` programmatically (no on-disk
//! merges file required).
//!
//! Verified against `tokenizer.ggml.model = "gpt2"` and `tokenizer.ggml.pre = "qwen2"`.

use anyhow::{anyhow, Context, Result};
use gguf_reader::{GgufFile, MultiGgufFile};
use tokenizers::{
    models::bpe::{Vocab, BPE},
    pre_tokenizers::byte_level::ByteLevel,
    AddedToken, Tokenizer,
};
use std::collections::HashMap;

pub struct Tok {
    inner: Tokenizer,
    pub eos_id: u32,
    pub bos_id: Option<u32>,
}

impl Tok {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let model = g.metadata_string("tokenizer.ggml.model")
            .unwrap_or_else(|_| "gpt2".to_string());
        if model != "gpt2" {
            return Err(anyhow!("only gpt2-style BPE supported (got {model})"));
        }

        let tokens = g.metadata_string_array("tokenizer.ggml.tokens")
            .context("missing tokenizer.ggml.tokens")?;
        let merges_raw = g.metadata_string_array("tokenizer.ggml.merges")
            .context("missing tokenizer.ggml.merges")?;

        // Build vocab map: token -> id
        let mut vocab: Vocab = HashMap::with_capacity(tokens.len());
        for (i, tok) in tokens.iter().enumerate() {
            vocab.insert(tok.clone(), i as u32);
        }

        // Merges: GGUF stores as "A B" strings; tokenizers expects (A, B) pairs.
        let merges: Vec<(String, String)> = merges_raw.iter()
            .filter_map(|line| {
                let mut it = line.splitn(2, ' ');
                let a = it.next()?.to_string();
                let b = it.next()?.to_string();
                Some((a, b))
            })
            .collect();
        if merges.len() != merges_raw.len() {
            return Err(anyhow!("malformed merge lines: parsed {} of {}",
                merges.len(), merges_raw.len()));
        }

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .ignore_merges(true)
            .build()
            .map_err(|e| anyhow!("BPE build failed: {e}"))?;

        // Qwen2/Qwen3 tokenizers use byte-level pre-tokenization (gpt2 family)
        // with `add_prefix_space = false` (matches HF Qwen tokenizer config).
        let mut tk = Tokenizer::new(bpe);
        tk.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
        tk.with_decoder(Some(ByteLevel::new(false, false, false)));

        // Special tokens — minimum set from GGUF metadata.
        let eos_id = g.metadata_u32("tokenizer.ggml.eos_token_id")?;
        let bos_id = g.metadata_u32("tokenizer.ggml.bos_token_id").ok();
        if let Some(eos_str) = tokens.get(eos_id as usize) {
            tk.add_special_tokens(&[AddedToken::from(eos_str.clone(), true)]);
        }

        Ok(Self { inner: tk, eos_id, bos_id })
    }

    pub fn from_multi_gguf(mg: &MultiGgufFile) -> Result<Self> {
        // Delegate to shard 0 which has all metadata
        Self::from_gguf(mg.shard(0))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.inner.encode(text, false)
            .map_err(|e| anyhow!("encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, false)
            .map_err(|e| anyhow!("decode failed: {e}"))
    }

    pub fn id_to_str(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}
