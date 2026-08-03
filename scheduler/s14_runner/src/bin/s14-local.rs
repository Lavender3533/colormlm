use polaris_s14_runner::{
    CapabilityManifest, GateError, LongContextMemoryPlan, MemoryLedger,
    CURRENT_FULL_DEPTH_CAPABILITIES_JSON, CURRENT_VULKAN_CAPABILITIES_JSON,
    DEPRECATED_FULL_DEPTH_TOP1_NEGATIVE_CONTRACT_JSON, EXACT_CASCADE_CONTRACT_JSON,
    INTEROP_CONTRACT_JSON,
};
use serde_json::json;
use std::path::Path;

fn read_manifest(path: Option<&String>, full_depth: bool) -> Result<CapabilityManifest, String> {
    let encoded = match path {
        Some(path) => std::fs::read_to_string(Path::new(path))
            .map_err(|error| format!("read {path}: {error}"))?,
        None if full_depth => CURRENT_FULL_DEPTH_CAPABILITIES_JSON.to_string(),
        None => CURRENT_VULKAN_CAPABILITIES_JSON.to_string(),
    };
    serde_json::from_str(&encoded).map_err(|error| format!("manifest JSON: {error}"))
}

fn print_json(value: &serde_json::Value) {
    println!("{}", serde_json::to_string_pretty(value).unwrap());
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("audit");
    let exit_code = match command {
        "contract" => {
            print!("{INTEROP_CONTRACT_JSON}");
            0
        }
        "cascade-contract" => {
            print!("{EXACT_CASCADE_CONTRACT_JSON}");
            0
        }
        "deprecated-top1" => {
            print!("{DEPRECATED_FULL_DEPTH_TOP1_NEGATIVE_CONTRACT_JSON}");
            0
        }
        "memory" => {
            let ledgers = vec![
                MemoryLedger::correctness_cold_stream(),
                MemoryLedger::steady_state_bf16_head(),
                MemoryLedger::steady_state_fp8_head_candidate(),
                MemoryLedger::full_depth43_native_top6_causal_block(),
                MemoryLedger::host_ram(),
            ];
            println!("{}", serde_json::to_string_pretty(&ledgers).unwrap());
            0
        }
        "long-context" => match LongContextMemoryPlan::target_200k() {
            Ok(plan) => {
                println!("{}", serde_json::to_string_pretty(&plan).unwrap());
                0
            }
            Err(error) => {
                print_json(&json!({"ok": false, "error": error.to_string()}));
                2
            }
        },
        "audit" | "gate" | "audit-full" | "gate-full" => {
            let full_depth = command.ends_with("-full");
            match read_manifest(args.get(1), full_depth) {
                Ok(manifest) => {
                    let missing = manifest.missing_capabilities();
                    let gate = manifest.gate_production();
                    print_json(&json!({
                        "format": "polaris-local-s14-gate-report-v1",
                        "native_forward_ready": gate.is_ok(),
                        "missing_capabilities": missing,
                        "token_emitted": false,
                        "speed_measurement": null,
                        "error": gate.as_ref().err().map(ToString::to_string),
                        "claim_limit": "static audit/capacity plan is not a model run"
                    }));
                    if command.starts_with("gate") && gate.is_err() {
                        2
                    } else {
                        0
                    }
                }
                Err(error) => {
                    print_json(&json!({"ok": false, "token_emitted": false, "error": error}));
                    2
                }
            }
        }
        _ => {
            let error = GateError::Parse(format!(
                "usage: s14-local [contract|cascade-contract|deprecated-top1|memory|long-context|audit|gate|audit-full|gate-full] [manifest]; got {command}"
            ));
            eprintln!("{error}");
            2
        }
    };
    std::process::exit(exit_code);
}
