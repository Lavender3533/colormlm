use polaris_api::{
    serve, ChatEngine, DeepSeekV4ChatCodec, EngineError, ResidentChatEngine, S14RuntimeChatBackend,
    S14RuntimeChatConfig, VerifiedS14NumericalGate, DEFAULT_S14_N8_EVIDENCE_PATH,
    DEFAULT_S14_TOKENIZER_PATH,
};
use ssd_inference::s14_runtime::{S14Runtime, S14RuntimeConfig};
use std::{env, io, net::SocketAddr, path::PathBuf, sync::Arc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let address: SocketAddr = env::var("POLARIS_API_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:11435".to_owned())
        .parse()?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    let evidence_path = PathBuf::from(
        env::var("POLARIS_S14_N8_EVIDENCE")
            .unwrap_or_else(|_| DEFAULT_S14_N8_EVIDENCE_PATH.to_owned()),
    );
    let tokenizer_path = PathBuf::from(
        env::var("POLARIS_S14_TOKENIZER").unwrap_or_else(|_| DEFAULT_S14_TOKENIZER_PATH.to_owned()),
    );
    let max_seq_len = env_u32("POLARIS_S14_MAX_SEQ_LEN", 4096)?;
    let default_max_tokens = env_u32("POLARIS_S14_DEFAULT_MAX_TOKENS", 3)?;
    let queue_capacity = env_usize("POLARIS_S14_QUEUE_CAPACITY", 1)?;
    let explicit_page_fetch = env_bool("POLARIS_S14_EXPLICIT_PAGE_FETCH", false)?;
    let live_n12_probe = env_bool("POLARIS_S14_LIVE_N12_EVIDENCE_PROBE", false)?;
    let live_n26_second_turn_probe =
        env_bool("POLARIS_S14_LIVE_N26_SECOND_TURN_EVIDENCE_PROBE", false)?;
    if explicit_page_fetch {
        require_proxy_policy()?;
    }
    if live_n12_probe && live_n26_second_turn_probe {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "S14 live N12 与 live N26 second-turn probe 不能同时启用",
        )
        .into());
    }
    if live_n12_probe
        && (!address.ip().is_loopback() || default_max_tokens != 8 || queue_capacity != 1)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "S14 live N12 evidence probe 只允许 loopback、default_max_tokens=8、queue_capacity=1",
        )
        .into());
    }
    if live_n26_second_turn_probe
        && (!address.ip().is_loopback() || default_max_tokens != 16 || queue_capacity != 1)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "S14 live N26 second-turn probe 只允许 loopback、default_max_tokens=16、queue_capacity=1",
        )
        .into());
    }
    let engine = ResidentChatEngine::spawn(queue_capacity, move || {
        // 顺序有意固定：先验 N=8 日志与官方 codec；二者失败时绝不初始化 Vulkan/模型资产。
        let numerical_gate = if live_n26_second_turn_probe {
            VerifiedS14NumericalGate::live_n26_second_turn_evidence_probe()
        } else if live_n12_probe {
            VerifiedS14NumericalGate::live_n12_evidence_probe()
        } else {
            VerifiedS14NumericalGate::from_n8_evidence_file(&evidence_path)?
        };
        let codec = DeepSeekV4ChatCodec::load_production(&tokenizer_path)?;
        let runtime_config =
            S14RuntimeConfig::production_defaults().with_explicit_page_fetch(explicit_page_fetch);
        let runtime = S14Runtime::load(runtime_config).map_err(|error| {
            EngineError::runtime_unavailable(format!("加载 Polaris S14 runtime 失败: {error:#}"))
        })?;
        let backend = S14RuntimeChatBackend::new(
            runtime,
            codec,
            S14RuntimeChatConfig {
                max_seq_len,
                default_max_tokens,
                numerical_gate,
            },
        )?;
        Ok(Box::new(backend))
    })
    .map_err(|error| io::Error::other(error.message))?;
    let engine: Arc<dyn ChatEngine> = Arc::new(engine);

    eprintln!("Polaris API 正在监听 http://{address}");
    if live_n12_probe {
        eprintln!("S14 live N12 evidence probe 已启用：只允许本机单请求，请求后必须关闭服务。");
    }
    if live_n26_second_turn_probe {
        eprintln!(
            "S14 live N26 second-turn evidence probe 已启用：只允许本机单请求，请求后必须关闭服务。"
        );
    }
    eprintln!(
        "resident loader 正在依次核验 N=8 证据、官方 codec 与 S14 runtime；完成前 health/模型发现/生成均返回 HTTP 503。"
    );
    serve(listener, engine).await?;
    Ok(())
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
