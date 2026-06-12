// Enables the Metal/NAX backend automatically on macOS — no feature flag — so
// the emulator dispatches large matmuls to the M5 tensor engine by default,
// exactly as Apple Accelerate is linked by default on macOS. The `objc2-*`
// crates are macOS-only, so on macOS they are always present (target deps);
// elsewhere the optional `metal` feature can still force it on for cross builds.
//
// Emits `cfg(metal)`; all backend code gates on `cfg(metal)`.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(metal)");
    let is_macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    let feature_on = std::env::var("CARGO_FEATURE_METAL").is_ok();
    if is_macos || feature_on {
        println!("cargo::rustc-cfg=metal");
    }
}
