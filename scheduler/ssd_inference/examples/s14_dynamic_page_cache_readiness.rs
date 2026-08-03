use anyhow::{bail, Context, Result};
use ssd_inference::{
    s14_dynamic_page_cache_readiness::{
        inspect_dynamic_page_cache, materialize_dynamic_routed_arena_with_transport,
        DynamicPageFetchMode, DynamicPageRangeTransport,
    },
    s14_dynamic_routed_page_plan::{FullDepthExpertCatalog, OnlineTop6},
};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

const DEFAULT_CATALOG: &str = "D:/models/Polaris-S14/fulldepth43_native_top6_catalog.json";
const DEFAULT_CACHE_ROOT: &str = "D:/models/Polaris-S14/range_cache";

struct Args {
    route: PathBuf,
    catalog: PathBuf,
    cache_root: PathBuf,
    fetch: bool,
    python: String,
    fetch_script: PathBuf,
}

fn default_fetch_script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../fast16/research/polaris_meridian_v1/s14_range_pack/fetch_dynamic_range_pages.py",
    )
}

fn usage() {
    eprintln!(
        "用法: s14_dynamic_page_cache_readiness --route ROUTE.json \
         [--catalog CATALOG.json] [--cache-root DIR] [--fetch] \
         [--python python] [--fetch-script SCRIPT.py]\n\
         默认仅检查本地 cache；只有显式 --fetch 才允许 Range 下载。"
    );
}

fn parse_args() -> Result<Option<Args>> {
    let mut route = None;
    let mut catalog = PathBuf::from(DEFAULT_CATALOG);
    let mut cache_root = PathBuf::from(DEFAULT_CACHE_ROOT);
    let mut fetch = false;
    let mut python = "python".to_owned();
    let mut fetch_script = default_fetch_script();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--route" => route = Some(PathBuf::from(next_value(&mut args, "--route")?)),
            "--catalog" => catalog = PathBuf::from(next_value(&mut args, "--catalog")?),
            "--cache-root" => cache_root = PathBuf::from(next_value(&mut args, "--cache-root")?),
            "--python" => python = next_value(&mut args, "--python")?,
            "--fetch-script" => {
                fetch_script = PathBuf::from(next_value(&mut args, "--fetch-script")?)
            }
            "--fetch" => fetch = true,
            "--help" | "-h" => {
                usage();
                return Ok(None);
            }
            _ => bail!("未知参数 {arg}"),
        }
    }
    let route = route.context("缺少必需参数 --route ROUTE.json")?;
    Ok(Some(Args {
        route,
        catalog,
        cache_root,
        fetch,
        python,
        fetch_script,
    }))
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next().with_context(|| format!("{flag} 缺少值"))
}

fn print_report(
    report: &ssd_inference::s14_dynamic_page_cache_readiness::DynamicPageCacheReadinessReport,
) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(report).context("encode readiness report")?
    );
    io::stdout().flush().context("flush readiness report")
}

fn main() -> Result<()> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let route_file = fs::File::open(&args.route)
        .with_context(|| format!("open OnlineTop6 JSON {}", args.route.display()))?;
    let route: OnlineTop6 = serde_json::from_reader(route_file)
        .with_context(|| format!("parse OnlineTop6 JSON {}", args.route.display()))?;
    let catalog = FullDepthExpertCatalog::load(&args.catalog)?;
    let plan = catalog.plan(route)?;
    let before = inspect_dynamic_page_cache(&plan, &args.cache_root)?;
    print_report(&before)?;

    if !args.fetch {
        return Ok(());
    }
    let transport = DynamicPageRangeTransport::new(args.python, args.fetch_script);
    let arena = materialize_dynamic_routed_arena_with_transport(
        &plan,
        &args.cache_root,
        DynamicPageFetchMode::ExplicitFetch,
        &transport,
    )
    .map_err(anyhow::Error::new)?;

    let after = inspect_dynamic_page_cache(&plan, &args.cache_root)?;
    print_report(&after)?;
    if after.unready_count != 0 {
        bail!(
            "Range fetch 后仍有 {} 个未就绪物理 Range",
            after.unready_count
        );
    }
    eprintln!(
        "canonical dynamic routed arena ready: layer={} position={} ranges={} bytes={}",
        plan.layer,
        plan.position,
        arena.pages.assets.len(),
        arena.arena_logical_bytes()
    );
    Ok(())
}
