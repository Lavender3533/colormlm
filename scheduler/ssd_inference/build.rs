use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // ggml lib search paths
    println!("cargo:rustc-link-search=native=D:/project/大模型ssd化/build/ggml/src/Release");
    println!("cargo:rustc-link-search=native=D:/project/大模型ssd化/build/bin/Release");

    // Compile ggml_bridge.c with AVX2+FMA
    let csrc = manifest.join("csrc").join("ggml_bridge.c");
    if csrc.exists() {
        let mut build = cc::Build::new();
        build.file(&csrc).opt_level(3);
        if cfg!(target_env = "msvc") {
            build.flag("/arch:AVX2");
        } else {
            build.flag("-mavx2").flag("-mfma");
        }
        build.compile("ggml_bridge");
        println!("cargo:rerun-if-changed={}", csrc.display());
    }

    let shader_dir = manifest.join("shaders");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", shader_dir.display());

    let Ok(entries) = std::fs::read_dir(&shader_dir) else {
        // Shaders dir absent → nothing to compile; allow build to proceed.
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else { continue };
        if !matches!(ext, "comp" | "vert" | "frag") { continue; }

        let stem = path.file_stem().unwrap().to_str().unwrap();
        let out_path = out_dir.join(format!("{stem}.spv"));

        println!("cargo:rerun-if-changed={}", path.display());

        let status = Command::new("glslc")
            .args(["--target-env=vulkan1.2", "-O"])
            .arg(&path)
            .arg("-o").arg(&out_path)
            .status()
            .unwrap_or_else(|_| {
                panic!("`glslc` not found in PATH. Install Vulkan SDK (e.g. `scoop install vulkan`).");
            });

        if !status.success() {
            panic!("glslc failed for {}", path.display());
        }
    }
}
