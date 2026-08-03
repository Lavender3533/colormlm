use polaris_api::{DeepSeekV4ChatCodec, S14ChatCodec, DEFAULT_S14_TOKENIZER_PATH};
use ssd_inference::s14_runtime::{S14Runtime, S14RuntimeConfig};
use std::{
    io,
    time::{Duration, Instant},
};

const FIRST_INPUT_TOKEN_ID: u32 = 0;
const MAX_SEQ_LEN: u32 = 16;
const STEPS: usize = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut codec = DeepSeekV4ChatCodec::load_production(DEFAULT_S14_TOKENIZER_PATH)
        .map_err(|error| io::Error::other(error.message))?;
    let mut runtime = S14Runtime::load(S14RuntimeConfig::production_defaults())?;

    let run_result = run_eight_steps(&mut runtime, &mut codec);
    let runtime_cleanup = runtime.destroy();

    match (run_result, runtime_cleanup) {
        (Ok(result), Ok(())) => {
            println!(
                "status=pass tokens={:?} text={:?} wall_seconds={:.6}",
                result.tokens,
                result.text,
                result.wall.as_secs_f64()
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

struct CliResult {
    tokens: Vec<u32>,
    text: String,
    wall: Duration,
}

fn run_eight_steps(
    runtime: &mut S14Runtime,
    codec: &mut DeepSeekV4ChatCodec,
) -> Result<CliResult, Box<dyn std::error::Error>> {
    let mut session = runtime.new_session(FIRST_INPUT_TOKEN_ID, MAX_SEQ_LEN)?;
    let started = Instant::now();
    let model_result = (|| {
        let mut tokens = Vec::with_capacity(STEPS);
        for _ in 0..STEPS {
            tokens.push(runtime.step(&mut session)?.predicted_token_id);
        }
        let text = codec
            .decode_completion(&tokens)
            .map_err(|error| io::Error::other(error.message))?;
        Ok::<_, Box<dyn std::error::Error>>(CliResult {
            tokens,
            text,
            wall: started.elapsed(),
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
