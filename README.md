# meikiocr (Rust)

A Rust port of [rtr46/meikiocr](https://github.com/rtr46/meikiocr) (Apache-2.0): local OCR for Japanese game text, running the original detector and recognizer models through ONNX Runtime, with no Python or OpenCV.

Used by [Kimochi](https://github.com/hamstu/kimochi-tauri).

```toml
[dependencies]
meikiocr = { git = "https://github.com/hamstu/kimochi-meikiocr" }
```

```rust,no_run
use meikiocr::{MeikiOcr, ModelPaths, OcrOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ocr = MeikiOcr::new(ModelPaths::from_dir("models"))?;
    let image = image::open("screenshot.png")?.to_rgb8();
    for line in ocr.ocr(&image, OcrOptions::default())? {
        println!("{} {:?}", line.text, line.bbox);
    }
    Ok(())
}
```

Models are not included. Download them (about 44 MiB, pinned and hash-checked) with `python3 scripts/download_models.py models`. They are published by rtr46 under LGPL-3.0: [detection](https://huggingface.co/rtr46/meiki.text.detect.v0), [recognition](https://huggingface.co/rtr46/meiki.txt.recognition.v0). At run time, point `ORT_DYLIB_PATH` at an ONNX Runtime 1.22+ shared library. Cargo features `coreml`, `cuda` and `directml` (all on by default) enable those execution providers, with automatic fallback to CPU.

```sh
cargo test
cargo run --release --example ocr -- models screenshot.png
```

## Licence

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)), matching upstream meikiocr.
