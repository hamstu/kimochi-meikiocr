# Windows MeikiOCR bring-up follow-up

2026-10-05. Windows 11 ARM64 Parallels VM, running the x64 MSVC target. Adapter: Parallels Display Adapter (WDDM), driver 20.18.2640.57456. Rust stable 1.99.0; ONNX Runtime DirectML 1.22.0 and the application's SHA-256-pinned models. Work branch: windows-bringup. No macOS execution path changed.

## Failure captured before editing

The original `cargo run --example ocr --no-default-features --features directml -- MODEL_DIR GAME_CROP directml` exited 1. Session creation succeeded and reported `selected: DirectMl`, then detection failed:

```
Non-zero status code returned while running Add node. Name:'/model/backbone/model/Add' Status Message: D:\a\_work\1\s\onnxruntime\core\providers\dml\DmlExecutionProvider\src\MLOperatorAuthorImpl.cpp(2506)\onnxruntime.dll!00007FFF6E323285: (caller: 00007FFF6E3228EF) Exception(3) tid(d60) 80004005 Unspecified error
```

A temporary example tested detection, horizontal recognition and vertical recognition independently with graph optimizations disabled, Level1 and Level3. All nine cases failed at the same Add node with 80004005. The diagnostic example initially failed compilation with `E0382: use of moved value: level` because GraphOptimizationLevel is not Copy; the helper was corrected to construct the enum per session. The diagnostic source and both logs are retained outside the repository. This identifies an unusable DirectML execution path on this adapter; the precise upstream/driver cause is not established. CPU inference on the same model files and image succeeds.

## Minimal fix and change log

1. Created windows-bringup from the clean local main checkout; no dependency or model changes.
2. Added `src/runtime.rs::probe_directml`, compiled only for Windows + directml. Before selecting DirectML, execute a zero-valued batch-one tensor and the corresponding target-size tensor through all three model sessions. A failure preserves its original error in fallback diagnostics. Auto continues to CPU; explicit DirectMl returns the error and never silently switches backend. Other backends and platforms retain existing initialization behavior.
3. Added the opt-in Windows real-model test in `tests/inference.rs`: automatic OCR must produce text equal to CPU OCR; CPU fallback must explain DirectML failure; an explicit failed DirectML backend must return an error. This requires actual models/runtime and is ignored in ordinary source-only tests.
4. Updated README to document the Windows probe, startup cost and batch-one coverage limit. Probe success does not guarantee all operators run on GPU or every recognition batch size works.
5. Removed the temporary optimization diagnostic example from the checkout after copying it to the external evidence directory. No native runtime or model file was modified.

## Validation

- PASS: cargo test --no-default-features --features directml -- --include-ignored: 4 unit tests and 3 real-model integration tests, zero failures. Includes existing batching, vertical segmentation and invalid-input cases.
- PASS: final Windows automatic-selection regression after making it tolerate session-creation failures as well as inference failures.
- PASS: cargo clippy --all-targets --no-default-features --features directml -- -D warnings, with the existing bring-up Clippy MSRV 1.87 configuration.
- PASS: standalone automatic OCR selects Cpu and records `DirectML detection inference probe failed` plus the original Add/80004005 error. Reads the captured game dialogue exactly:
  - ゴロゴロとスーツケースを引きながら、
  - あたしは空港を歩いていた。
- PASS: real Kimochi application strict Clippy using the sibling path dependency.
- PASS: rebuilt application bundle, then actual Scan now UI reads both lines and publishes two overlay text regions. Cold OCR 2450ms; warm automatic settled OCR 448ms.
- PASS with cold-start retry: actual Auto OCR UI observer reads two lines/32 characters, 457.687583ms inference; later unchanged-frame polls use cached results in approximately 6–8ms. Initial 1500ms busy timeout recovered automatically after 2000ms backoff. Full app evidence and observations are in kimochi-tauri/WINDOWS_BRINGUP.md.

## Remaining limitation

FAIL: DirectML acceleration on this Parallels VM. This change restores automatic local OCR via CPU and accurately reports the selected backend; it does not repair the virtual GPU driver. Physical Windows GPU validation remains NOT TESTED. Automatic fallback happens at initialization, not after arbitrary later device loss or batch-specific failures.