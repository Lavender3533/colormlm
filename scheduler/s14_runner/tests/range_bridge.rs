use polaris_s14_runner::{
    router_kind_for_layer, RangeBridgeConfig, RouteDecision, RouteFirstProvider,
    SubprocessRangeProvider,
};
use std::process::Command;
use std::time::{Duration, Instant};

fn spawn(mode: &str, timeout_ms: u64) -> Result<SubprocessRangeProvider, String> {
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python".into());
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("jsonl_fixture.py");
    let mut command = Command::new(python);
    command.arg("-X").arg("utf8").arg(fixture).arg(mode);
    SubprocessRangeProvider::spawn(
        command,
        RangeBridgeConfig {
            response_timeout: Duration::from_millis(timeout_ms),
            download_authorized: false,
        },
    )
    .map_err(|error| error.to_string())
}

#[test]
fn persistent_jsonl_closes_base_route_routed_release_loop() {
    let mut provider = spawn("normal", 1_000).unwrap();
    let ticket = provider.begin_base_load(0, 17).unwrap();
    let base = provider.wait_base_ready(ticket).unwrap();
    assert_eq!(base.layer, 0);
    assert_eq!(base.artifacts.len(), 1);

    let route = RouteDecision {
        layer: 0,
        kind: router_kind_for_layer(0).unwrap(),
        expert_ids: vec![126, 12, 205, 149, 227, 174],
        weights: vec![0.25; 6],
    };
    let ticket = provider.begin_routed_load(&base, &route).unwrap();
    let routed = provider.wait_routed_ready(ticket).unwrap();
    assert_eq!(routed.expert_ids, route.expert_ids);
    assert_eq!(routed.artifacts.len(), 6);
    assert_eq!(routed.observation.expert_cache_hits, 6);
    provider.release_layer(0).unwrap();
    assert!(!provider.is_poisoned());
}

#[test]
fn release_before_route_uses_abort_cleanup() {
    let mut provider = spawn("normal", 1_000).unwrap();
    let ticket = provider.begin_base_load(0, 17).unwrap();
    let base = provider.wait_base_ready(ticket).unwrap();
    assert_eq!(base.layer, 0);
    provider.release_layer(0).unwrap();
    assert!(!provider.is_poisoned());
}

#[test]
fn download_authorization_mismatch_is_hard_refused_at_hello() {
    let error = spawn("auth_mismatch", 1_000).err().unwrap();
    assert!(error.contains("download_authorized"), "{error}");
}

#[test]
fn worker_error_poisoned_bridge() {
    let mut provider = spawn("reject", 1_000).unwrap();
    let ticket = provider.begin_base_load(0, 17).unwrap();
    let error = provider.wait_base_ready(ticket).unwrap_err().to_string();
    assert!(error.contains("synthetic rejection"), "{error}");
    assert!(provider.is_poisoned());
}

#[test]
fn worker_timeout_is_bounded_and_poisoned() {
    let mut provider = spawn("timeout", 500).unwrap();
    let ticket = provider.begin_base_load(0, 17).unwrap();
    let started = Instant::now();
    let error = provider.wait_base_ready(ticket).unwrap_err().to_string();
    assert!(error.contains("超时"), "{error}");
    assert!(started.elapsed() < Duration::from_secs(2));
    assert!(provider.is_poisoned());
}

#[test]
fn worker_process_exit_is_not_misreported_as_ready() {
    let mut provider = spawn("exit", 1_000).unwrap();
    let ticket = provider.begin_base_load(0, 17).unwrap();
    let error = provider.wait_base_ready(ticket).unwrap_err().to_string();
    assert!(error.contains("EOF") || error.contains("退出"), "{error}");
    assert!(provider.is_poisoned());
}

#[test]
fn malformed_stdout_is_hard_rejected() {
    let mut provider = spawn("malformed", 1_000).unwrap();
    let ticket = provider.begin_base_load(0, 17).unwrap();
    let error = provider.wait_base_ready(ticket).unwrap_err().to_string();
    assert!(error.contains("非 JSONL"), "{error}");
    assert!(provider.is_poisoned());
}
