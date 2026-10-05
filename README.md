# meikiocr (Rust)

A reusable Rust port of [rtr46/meikiocr](https://github.com/rtr46/meikiocr) for local Japanese game text OCR. No Python or OpenCV is needed by the library. It runs the original detector, horizontal recognizer, and vertical recognizer through ONNX Runtime, including batching, overlapping vertical segments, character overlap suppression, punctuation weighting, and upstream swapped-pair corrections.

```toml
[dependencies]
meikiocr = { path = "../kimochi-meikiocr" }
image = { version = "0.25", default-features = false, features = ["png", "jpeg"] }
```

```rust,no_run
use meikiocr::{MeikiOcr, ModelPaths, OcrOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ocr = MeikiOcr::new(ModelPaths::from_dir("models"))?;
    println!("{:?}", ocr.backend_info());
    let image = image::open("screenshot.png")?.to_rgb8();
    for line in ocr.ocr(&image, OcrOptions::default())? {
        println!("{} {:?}", line.text, line.bbox);
    }
    Ok(())
}
```

Load the engine once and reuse it. `detect` runs only detection; `recognize` accepts a slice of cropped RGB images. Results implement Serde and contain line boxes, detection confidence, orientation, and individual character boxes/confidence. All boxes use original image pixels with exclusive right/bottom edges. `recognize` returns crop-relative coordinates and no detection confidence. RGB/RGBA screenshots must be converted to `RgbImage`; the library handles BGR model input internally.

## Models and runtime

Download the three models (about 44 MiB) once:

```sh
python3 scripts/download_models.py models
```

This optional setup script pins model repository revisions, verifies Hugging Face LFS SHA-256 hashes, and atomically installs files. The library never downloads files or calls a network service. Model files are ignored by Git. See the model cards for model licensing: [detection](https://huggingface.co/rtr46/meiki.text.detect.v0), [recognition](https://huggingface.co/rtr46/meiki.txt.recognition.v0).

Supply a **shared ONNX Runtime library with API version 22 or newer** (ONNX Runtime 1.22+). The pinned `ort` binding uses runtime dynamic loading: builds and unit tests need no installed native runtime. At execution, set `ORT_DYLIB_PATH` to an absolute path before launching the app, or configure `ort::init_from(path).commit()` at application startup before creating an OCR engine. The runtime is process-global. An absent/incompatible native library can panic inside `ort`; it is a deployment prerequisite, not a GPU fallback condition.

For local macOS development, a Python wheel is one convenient source of the shared library (Python is only used for setup):

```sh
python3 -m venv .venv
.venv/bin/pip install onnxruntime
export ORT_DYLIB_PATH="$(find "$PWD/.venv" -name 'libonnxruntime.*dylib' -print -quit)"
cargo run --release --example ocr -- models screenshot.png
```

For shipped applications, obtain the matching architecture's C/C++ shared runtime from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases), or build it with the required execution provider. Bundle the shared library, its dependencies, model files, and their licenses with your application. Python is unnecessary in the distributed app. Rust Cargo features enable provider bindings; they do **not** install native GPU drivers or a provider-enabled runtime.

## Hardware acceleration

| Platform | Automatic preference | Native runtime requirement |
| --- | --- | --- |
| Apple Silicon macOS | Core ML, then CPU | Core ML enabled arm64 runtime |
| Windows | CUDA, then DirectML, then CPU | CUDA or DirectML enabled runtime |
| Linux | CUDA, then CPU | CUDA enabled runtime for NVIDIA GPUs |
| Other / Intel macOS | CPU | Matching CPU runtime |

[Core ML](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) can use Apple's GPU and Neural Engine. [CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) requires compatible NVIDIA drivers, CUDA, and cuDNN versions for your runtime. [DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) supports compatible DirectX 12 GPUs on Windows, including AMD, Intel, and NVIDIA. AMD/Intel GPU acceleration on Linux is not currently implemented.

`Config::default()` selects `Backend::Auto`. All three sessions must load successfully for an accelerator to be selected. `backend_info()` reports the selected provider and any fallback reasons. Successful registration does not mean every operation is accelerated: ONNX Runtime may place unsupported operations on CPU. Core ML uses the NeuralNetwork format and only static-shape partitions: MLProgram with these model graphs aborted inside Apple's compiler during validation, so it is deliberately avoided. Dynamic recognition operations may run on CPU. First initialization can take longer due to model compilation. Benchmark your actual workload before choosing an explicit backend.

Set `Config { backend: Backend::Cpu, ..Default::default() }` for predictable CPU execution. Explicit `CoreMl`, `Cuda`, or `DirectMl` selection errors if unavailable rather than silently changing provider. For a CPU-only build use `default-features = false`; selectively enable `coreml`, `cuda`, or `directml` as needed. DirectML sessions disable memory patterns and parallel execution as required by that provider.

## Tauri integration

Keep `Arc<Mutex<MeikiOcr>>` in managed state and run image decoding/inference inside `spawn_blocking`. The async command returns serializable results directly:

```rust,ignore
use std::sync::{Arc, Mutex};
use meikiocr::{MeikiOcr, OcrOptions, TextLine};

struct OcrState(Arc<Mutex<MeikiOcr>>);

#[tauri::command]
async fn read_text(
    state: tauri::State<'_, OcrState>,
    png: Vec<u8>,
) -> Result<Vec<TextLine>, String> {
    let engine = Arc::clone(&state.0);
    tauri::async_runtime::spawn_blocking(move || {
        let image = image::load_from_memory(&png)
            .map_err(|e| e.to_string())?.to_rgb8();
        engine.lock().map_err(|e| e.to_string())?
            .ocr(&image, OcrOptions::default())
            .map_err(|e| e.to_string())
    }).await.map_err(|e| e.to_string())?
}
```

Initialize the runtime and engine during application setup, resolve bundled resources using Tauri's resource directory, then call `app.manage(OcrState(Arc::new(Mutex::new(engine))))`. Add the models and runtime to Tauri's bundled resources; configure the runtime path before the first session. With `ort::init_from`, the host app should depend on the same pinned `ort` version and `load-dynamic` feature. Code signing must include the native runtime on macOS. A dedicated worker is another option for frequent screenshot processing; avoid rebuilding sessions on every frame.

## Validation and differences

```sh
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo check --no-default-features
MEIKIOCR_MODEL_DIR="$PWD/models" MEIKIOCR_TEST_IMAGE=/path/to/japanese-screenshot.png \
    cargo test --test inference -- --ignored
```

The opt-in integration test exercises real model inference, crop recognition, serialization, and batch-size consistency. Unit tests cover BGR preprocessing, black padding, vertical segment coverage, overlap suppression, and pair correction.

This port validates thresholds, rejects empty images and zero batch sizes, skips invalid/degenerate predictions, clamps character coordinates to their crop, and fixes vertical coordinate mapping for horizontally compressed long lines. Upstream uses OpenCV's optimized bilinear resize; this port uses half-pixel bilinear sampling with byte rounding, so small numerical differences can occur. Reading order follows upstream's top-to-bottom detection sort; vertical columns are not reordered right-to-left. Model limits and domain constraints remain those of upstream: game/rendered Japanese text, up to 64 detection boxes and 48 horizontal or 24 vertical recognition predictions per segment. Vertical recognition remains experimental.

Code license: Apache-2.0. Upstream provenance is recorded in `NOTICE`.

Validated locally on Apple Silicon macOS with ONNX Runtime 1.30.0: the upstream example's five recognized lines match Python on CPU; Core ML selection and end-to-end inference succeed. Real-model tests cover recognition batching, vertical splitting, and compressed horizontal crops. Windows CUDA/DirectML and Linux CUDA are implemented but require validation on those devices; no cross-platform performance claim is made.
