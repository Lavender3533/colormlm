use polaris_api::{
    serve_with_request_deadline, spawn_s14_starfold_ssd_root_worker, ChatEngine,
    DeepSeekV4ChatCodec, S14StarfoldSsdAdapterError, S14StarfoldSsdAdapterErrorKind,
    S14StarfoldSsdAdapterStage, DEFAULT_REQUEST_DEADLINE, DEFAULT_S14_TOKENIZER_PATH,
};
use ssd_inference::{
    s14_runtime::S14RuntimeConfig, s14_starfold_concrete_factory::S14StarfoldConcreteFactory,
};
use std::{env, io, net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};

const DEFAULT_STARFOLD_MICROTILE_MIB: u32 = 16;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let address: SocketAddr = env::var("POLARIS_API_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:11435".to_owned())
        .parse()?;
    let tokenizer_path = PathBuf::from(
        env::var("POLARIS_S14_TOKENIZER").unwrap_or_else(|_| DEFAULT_S14_TOKENIZER_PATH.to_owned()),
    );
    let max_seq_len = env_u32("POLARIS_S14_MAX_SEQ_LEN", 4096)?;
    let default_max_tokens = env_u32("POLARIS_S14_DEFAULT_MAX_TOKENS", 16)?;
    let queue_capacity = env_usize("POLARIS_S14_QUEUE_CAPACITY", 1)?;
    let request_deadline_secs = env_u32(
        "POLARIS_S14_REQUEST_DEADLINE_SECS",
        DEFAULT_REQUEST_DEADLINE.as_secs() as u32,
    )?;
    let explicit_page_fetch = env_bool("POLARIS_S14_EXPLICIT_PAGE_FETCH", false)?;
    let starfold_microtile_mib = env_u32(
        "POLARIS_S14_STARFOLD_MICROTILE_MIB",
        DEFAULT_STARFOLD_MICROTILE_MIB,
    )?;
    if explicit_page_fetch {
        require_proxy_policy()?;
    }
    if default_max_tokens == 0 || default_max_tokens > max_seq_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "POLARIS_S14_DEFAULT_MAX_TOKENS 必须位于 [1, MAX_SEQ_LEN]",
        )
        .into());
    }
    if request_deadline_secs == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "POLARIS_S14_REQUEST_DEADLINE_SECS 必须大于 0",
        )
        .into());
    }

    let starfold_microtile_bytes =
        starfold_microtile_mib
            .checked_mul(1024 * 1024)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "POLARIS_S14_STARFOLD_MICROTILE_MIB 换算字节溢出",
                )
            })?;
    let runtime_config = S14RuntimeConfig::production_defaults()
        .with_explicit_page_fetch(explicit_page_fetch)
        .with_starfold_microtile_bytes(starfold_microtile_bytes)?;
    let engine = spawn_s14_starfold_ssd_root_worker(
        queue_capacity,
        max_seq_len,
        default_max_tokens,
        move || {
            // loader 在 resident worker 内执行；Vulkan/context/runtime/session 不跨线程移动。
            let codec = DeepSeekV4ChatCodec::load_production(&tokenizer_path).map_err(|error| {
                worker_load_error(format!(
                    "加载官方 S14 tokenizer/codec 失败: {}",
                    error.message
                ))
            })?;
            let root = S14StarfoldConcreteFactory::load_production_root(runtime_config).map_err(
                |error| {
                    worker_load_error(format!(
                        "加载唯一 S14 StarFold production root 失败: {error:#}"
                    ))
                },
            )?;
            Ok((codec, root))
        },
    )
    .map_err(|error| io::Error::other(error.message))?;
    let engine: Arc<dyn ChatEngine> = Arc::new(engine);
    let listener = tokio::net::TcpListener::bind(address).await?;

    eprintln!("Polaris S14 StarFold API 正在监听 http://{address}");
    eprintln!(
        "StarFold resident windows=2x{starfold_microtile_mib} MiB, routed_path={}; \
         POLARIS_S14_STARFOLD_MICROTILE_MIB=8 可显式回退 microtile",
        if starfold_microtile_mib >= 16 {
            "constellation"
        } else {
            "microtile"
        }
    );
    eprintln!(
        "resident worker 正在加载官方 codec 与唯一 S14 StarFold root；ready 前 health/模型发现/生成均返回 HTTP 503。"
    );
    eprintln!(
        "request deadline={}s；到期后等待当前不可中断 block 原子提交，但不再启动下一 FullDepth43 block。",
        request_deadline_secs
    );
    serve_with_request_deadline(
        listener,
        engine,
        Duration::from_secs(u64::from(request_deadline_secs)),
    )
    .await?;
    Ok(())
}

fn worker_load_error(message: impl Into<String>) -> S14StarfoldSsdAdapterError {
    S14StarfoldSsdAdapterError::new(
        S14StarfoldSsdAdapterErrorKind::Internal,
        S14StarfoldSsdAdapterStage::WorkerLoad,
        message,
    )
}

fn env_u32(name: &str, default: u32) -> io::Result<u32> {
    match env::var(name) {
        Ok(value) => value.parse::<u32>().map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{name} 必须是 u32: {error}"),
            )
        }),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("读取 {name} 失败: {error}"),
        )),
    }
}

fn env_usize(name: &str, default: usize) -> io::Result<usize> {
    match env::var(name) {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{name} 必须是 usize: {error}"),
            )
        }),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("读取 {name} 失败: {error}"),
        )),
    }
}

fn env_bool(name: &str, default: bool) -> io::Result<bool> {
    match env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{name} 必须是 true/false 或 1/0"),
            )),
        },
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("读取 {name} 失败: {error}"),
        )),
    }
}

fn require_proxy_policy() -> io::Result<()> {
    const REQUIRED_PROXY: &str = "http://127.0.0.1:7897";
    for name in ["HTTP_PROXY", "HTTPS_PROXY"] {
        let value = env::var(name).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("启用外部 Range fetch 时必须设置 {name}={REQUIRED_PROXY}"),
            )
        })?;
        if value.trim_end_matches('/') != REQUIRED_PROXY {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{name} 必须统一为 {REQUIRED_PROXY}，实际为 {value}"),
            ));
        }
    }
    Ok(())
}
