use meikiocr::{Backend, Config, MeikiOcr, ModelPaths, OcrOptions};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let models = args
        .next()
        .ok_or("usage: ocr MODEL_DIR IMAGE [cpu|auto|coreml|cuda|directml]")?;
    let image = args.next().ok_or("missing image path")?;
    let backend = match args.next().as_deref().unwrap_or("auto") {
        "cpu" => Backend::Cpu,
        "auto" => Backend::Auto,
        "coreml" => Backend::CoreMl,
        "cuda" => Backend::Cuda,
        "directml" => Backend::DirectMl,
        _ => return Err("unknown backend".into()),
    };
    let mut ocr = MeikiOcr::with_config(
        ModelPaths::from_dir(models),
        Config {
            backend,
            ..Default::default()
        },
    )?;
    eprintln!("{:?}", ocr.backend_info());
    println!(
        "{}",
        serde_json::to_string_pretty(&ocr.ocr_file(image, OcrOptions::default())?)?
    );
    Ok(())
}
