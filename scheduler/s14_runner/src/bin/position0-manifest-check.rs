use polaris_s14_runner::Position0WholeTokenManifest;
use std::env;
use std::path::Path;

fn main() {
    let path = env::args_os().nth(1).unwrap_or_else(|| {
        eprintln!("用法: position0-manifest-check <position0_whole_token_manifest.json>");
        std::process::exit(2);
    });
    let path = Path::new(&path);
    let manifest = Position0WholeTokenManifest::load(path).unwrap_or_else(|error| {
        eprintln!("status=reject path={} error={error}", path.display());
        std::process::exit(1);
    });
    println!(
        "status=pass layers={} assets={} bytes={} moe_assets={} moe_bytes={} input_token={} output_token={}",
        manifest.summary.layer_count,
        manifest.summary.asset_unique_count,
        manifest.summary.asset_bytes,
        manifest.summary.moe_payload_references,
        manifest.summary.moe_payload_bytes,
        manifest.input_token_id,
        manifest.expected_output_token_id,
    );
}
