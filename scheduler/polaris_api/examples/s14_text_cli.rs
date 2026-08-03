use polaris_api::{
    DeepSeekV4ChatCodec, EngineChatMessage, EngineChatRequest, S14ChatCodec,
    VerifiedS14NumericalGate, DEFAULT_S14_N8_EVIDENCE_PATH, DEFAULT_S14_TOKENIZER_PATH,
    POLARIS_MODEL_ID,
};
use ssd_inference::s14_runtime::{S14Runtime, S14RuntimeConfig};
use std::{
    env, io,
    time::{Duration, Instant},
};

const DEFAULT_MAX_TOKENS: u32 = 4;
const MIN_MAX_TOKENS: u32 = 1;
const MAX_MAX_TOKENS: u32 = 16;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli_started = Instant::now();
    let Some(args) = parse_args()? else {
        print_usage();
        return Ok(());
    };

    // 在初始化 Vulkan 之前完成协议编码与长度校验，坏请求不会占用 GPU。
    let mut codec = DeepSeekV4ChatCodec::load_production(DEFAULT_S14_TOKENIZER_PATH)
        .map_err(|error| io::Error::other(error.message))?;
    let request = EngineChatRequest {
        model: POLARIS_MODEL_ID.to_owned(),
        messages: vec![EngineChatMessage {
            role: "user".to_owned(),
            content: args.input.clone(),
            name: None,
        }],
        max_tokens: Some(args.max_tokens),
        temperature: Some(0.0),
        stop: Vec::new(),
        tools: None,
        tool_choice: None,
    };
    let prompt = codec
        .encode_chat(&request)
        .map_err(|error| io::Error::other(error.message))?;
    let required_positions = prompt
        .len()
        .checked_sub(1)
        .and_then(|prefill| prefill.checked_add(args.max_tokens as usize))
        .ok_or_else(|| io::Error::other("prompt + max_tokens 长度溢出"))?;
    let max_seq_len = u32::try_from(required_positions)
        .map_err(|_| io::Error::other("prompt + max_tokens 超过 u32 position 上限"))?;

    // 文本入口与 HTTP production backend 共用同一个 fail-closed 数值门；
    // 旧证据不能通过 CLI 绕过已冻结的 position 覆盖范围。
    let evidence_path = env::var("POLARIS_S14_N8_EVIDENCE")
        .unwrap_or_else(|_| DEFAULT_S14_N8_EVIDENCE_PATH.to_owned());
    let numerical_gate = VerifiedS14NumericalGate::from_n8_evidence_file(&evidence_path)
        .map_err(|error| io::Error::other(error.message))?;
    if max_seq_len > numerical_gate.max_position_exclusive() {
        return Err(format!(
            "请求需要 {max_seq_len} 个 position，但冻结数值门只覆盖 [0,{})；evidence={}",
            numerical_gate.max_position_exclusive(),
            numerical_gate.evidence(),
        )
        .into());
    }

    let runtime_started = Instant::now();
    let config = S14RuntimeConfig::production_defaults().with_explicit_page_fetch(args.fetch);
    let mut runtime = S14Runtime::load(config)?;
    let runtime_load = runtime_started.elapsed();

    let run_result = run_text_generation(
        &mut runtime,
        &mut codec,
        &prompt,
        max_seq_len,
        args.max_tokens,
    );
    let runtime_cleanup = runtime.destroy();

    match (run_result, runtime_cleanup) {
        (Ok(result), Ok(())) => {
            println!("status=pass");
            println!("input={:?}", args.input);
            println!("prompt_tokens={} prompt_ids={prompt:?}", prompt.len());
            println!(
                "completion_tokens={} completion_ids={:?}",
                result.completion_ids.len(),
                result.completion_ids
            );
            println!("text={:?}", result.text);
            println!(
                "runtime_load_seconds={:.6} prefill_seconds={:.6} generation_seconds={:.6} model_seconds={:.6} total_seconds={:.6}",
                runtime_load.as_secs_f64(),
                result.prefill_wall.as_secs_f64(),
                result.generation_wall.as_secs_f64(),
                result.model_wall.as_secs_f64(),
                cli_started.elapsed().as_secs_f64(),
            );
            Ok(())
        }
        (Err(run_error), Ok(())) => Err(run_error),
        (Ok(_), Err(cleanup_error)) => {
            Err(format!("S14 runtime destroy 失败: {cleanup_error:#}").into())
        }
        (Err(run_error), Err(cleanup_error)) => {
            Err(format!("{run_error}；同时 S14 runtime destroy 失败: {cleanup_error:#}").into())
        }
    }
}

struct CliArgs {
    input: String,
    max_tokens: u32,
    fetch: bool,
}

fn parse_args() -> Result<Option<CliArgs>, Box<dyn std::error::Error>> {
    let mut positional = Vec::new();
    let mut fetch = false;
    let mut options = true;
    for arg in env::args().skip(1) {
        if options {
            match arg.as_str() {
                "--help" | "-h" => return Ok(None),
                "--fetch" => {
                    fetch = true;
                    continue;
                }
                "--" => {
                    options = false;
                    continue;
                }
                _ if arg.starts_with('-') => {
                    return Err(format!("未知参数 {arg:?}；用户文本以 '-' 开头时请先写 --").into())
                }
                _ => {}
            }
        }
        positional.push(arg);
    }

    if positional.is_empty() || positional.len() > 2 {
        return Err("需要一个用户文本，及可选的 max_tokens（1..=16）".into());
    }
    let input = positional.remove(0);
    if input.is_empty() {
        return Err("用户文本不能为空".into());
    }
    let max_tokens = match positional.pop() {
        Some(value) => value
            .parse::<u32>()
            .map_err(|error| format!("max_tokens={value:?} 不是整数: {error}"))?,
        None => DEFAULT_MAX_TOKENS,
    };
    if !(MIN_MAX_TOKENS..=MAX_MAX_TOKENS).contains(&max_tokens) {
        return Err(format!(
            "max_tokens 必须在 {MIN_MAX_TOKENS}..={MAX_MAX_TOKENS}，实际 {max_tokens}"
        )
        .into());
    }
    Ok(Some(CliArgs {
        input,
        max_tokens,
        fetch,
    }))
}

fn print_usage() {
    println!("用法: s14_text_cli [--fetch] <用户文本> [max_tokens]");
    println!("默认严格只读本地分页；只有显式 --fetch 才允许补齐缺页。");
}

struct CliResult {
    completion_ids: Vec<u32>,
    text: String,
    prefill_wall: Duration,
    generation_wall: Duration,
    model_wall: Duration,
}

fn run_text_generation(
    runtime: &mut S14Runtime,
    codec: &mut DeepSeekV4ChatCodec,
    prompt: &[u32],
    max_seq_len: u32,
    max_tokens: u32,
) -> Result<CliResult, Box<dyn std::error::Error>> {
    let first_token = *prompt
        .first()
        .ok_or_else(|| io::Error::other("S14 chat codec 生成了空 prompt"))?;
    let mut session = runtime.new_session(first_token, max_seq_len)?;
    let model_result = (|| {
        let model_started = Instant::now();
        let prefill_started = Instant::now();
        for &next_prompt_token in prompt.iter().skip(1) {
            runtime.step_with_next_input(&mut session, Some(next_prompt_token))?;
        }
        let prefill_wall = prefill_started.elapsed();

        let generation_started = Instant::now();
        let mut completion_ids = Vec::with_capacity(max_tokens as usize);
        for _ in 0..max_tokens {
            let output = runtime.step(&mut session)?;
            completion_ids.push(output.predicted_token_id);
            if codec.is_eos(output.predicted_token_id) {
                break;
            }
        }
        let generation_wall = generation_started.elapsed();
        let text = codec
            .decode_completion(&completion_ids)
            .map_err(|error| io::Error::other(error.message))?;
        Ok::<_, Box<dyn std::error::Error>>(CliResult {
            completion_ids,
            text,
            prefill_wall,
            generation_wall,
            model_wall: model_started.elapsed(),
        })
    })();
    let session_cleanup = session.destroy();

    match (model_result, session_cleanup) {
        (Ok(result), Ok(())) => Ok(result),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup_error)) => {
            Err(format!("S14 session destroy 失败: {cleanup_error:#}").into())
        }
        (Err(error), Err(cleanup_error)) => {
            Err(format!("{error}；同时 S14 session destroy 失败: {cleanup_error:#}").into())
        }
    }
}
