//! Opt-in real model validation: MEIKIOCR_MODEL_DIR + MEIKIOCR_TEST_IMAGE + ORT_DYLIB_PATH.
use meikiocr::{Backend, Config, MeikiOcr, ModelPaths, OcrOptions};

#[cfg(all(target_os = "windows", feature = "directml"))]
#[test]
#[ignore = "requires downloaded models, text fixture and DirectML ONNX Runtime"]
fn windows_directml_matches_cpu_without_fallback() {
    let models = ModelPaths::from_dir(std::env::var("MEIKIOCR_MODEL_DIR").unwrap());
    let source = image::open(std::env::var("MEIKIOCR_TEST_IMAGE").unwrap())
        .unwrap()
        .to_rgb8();
    let mut cpu = MeikiOcr::with_config(
        models.clone(),
        Config {
            backend: Backend::Cpu,
            ..Default::default()
        },
    )
    .unwrap();
    let mut dml = MeikiOcr::with_config(
        models,
        Config {
            backend: Backend::DirectMl,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(dml.backend_info().selected, Backend::DirectMl);
    for image in [
        source.clone(),
        image::imageops::resize(
            &source,
            source.width() * 2,
            source.height() * 2,
            image::imageops::FilterType::Triangle,
        ),
    ] {
        let expected = cpu.detect(&image, 0.5).unwrap();
        let actual = dml.detect(&image, 0.5).unwrap();
        assert!(!actual.is_empty());
        assert_eq!(expected.len(), actual.len());
        for (a, b) in actual.iter().zip(&expected) {
            assert!((a.confidence - b.confidence).abs() < 0.001);
            assert!(a.bbox.iter().zip(b.bbox).all(|(a, b)| a.abs_diff(b) <= 2));
        }
        let expected = cpu.ocr(&image, OcrOptions::default()).unwrap();
        let actual = dml.ocr(&image, OcrOptions::default()).unwrap();
        assert!(actual.iter().all(|line| !line.text.is_empty()));
        assert_eq!(
            actual.iter().map(|l| &l.text).collect::<Vec<_>>(),
            expected.iter().map(|l| &l.text).collect::<Vec<_>>()
        );
        println!(
            "DirectML matches CPU: {}x{}, {} lines",
            image.width(),
            image.height(),
            actual.len()
        );
    }
    let crops = [
        image::RgbImage::new(32, 480),
        image::RgbImage::new(32, 1000),
        image::RgbImage::new(2000, 32),
    ];
    let expected = cpu.recognize(&crops, OcrOptions::default()).unwrap();
    let actual = dml.recognize(&crops, OcrOptions::default()).unwrap();
    assert_eq!(
        actual.iter().map(|l| &l.text).collect::<Vec<_>>(),
        expected.iter().map(|l| &l.text).collect::<Vec<_>>()
    );
    assert!(actual[0].is_vertical && actual[1].is_vertical && !actual[2].is_vertical);
}

#[cfg(all(target_os = "windows", feature = "directml"))]
#[test]
#[ignore = "requires downloaded models and DirectML ONNX Runtime"]
fn windows_auto_backend_runs_real_ocr() {
    let models = ModelPaths::from_dir(std::env::var("MEIKIOCR_MODEL_DIR").unwrap());
    let image = image::open(std::env::var("MEIKIOCR_TEST_IMAGE").unwrap())
        .unwrap()
        .to_rgb8();
    let mut automatic = MeikiOcr::new(models.clone()).unwrap();
    let actual = automatic.ocr(&image, OcrOptions::default()).unwrap();
    assert!(actual.iter().any(|line| !line.text.is_empty()));

    if automatic.backend_info().selected == Backend::Cpu {
        let reason = automatic
            .backend_info()
            .fallback_reasons
            .iter()
            .find(|reason| reason.starts_with("DirectMl:"))
            .expect("CPU selection must explain the DirectML failure");
        let explicit = MeikiOcr::with_config(
            models.clone(),
            Config {
                backend: Backend::DirectMl,
                ..Default::default()
            },
        );
        let error = explicit
            .err()
            .expect("explicit DirectML must not silently fall back");
        if reason.contains("inference probe failed") {
            assert!(error.to_string().contains("inference probe failed"));
        }
    }
    let mut cpu = MeikiOcr::with_config(
        models,
        Config {
            backend: Backend::Cpu,
            ..Default::default()
        },
    )
    .unwrap();
    let expected = cpu.ocr(&image, OcrOptions::default()).unwrap();
    assert_eq!(
        actual.iter().map(|line| &line.text).collect::<Vec<_>>(),
        expected.iter().map(|line| &line.text).collect::<Vec<_>>()
    );
}
#[test]
#[ignore = "requires downloaded models and ONNX Runtime"]
fn real_models_and_batching() {
    let models = std::env::var("MEIKIOCR_MODEL_DIR").expect("MEIKIOCR_MODEL_DIR");
    let image = image::open(std::env::var("MEIKIOCR_TEST_IMAGE").expect("MEIKIOCR_TEST_IMAGE"))
        .unwrap()
        .to_rgb8();
    let mut engine = MeikiOcr::with_config(
        ModelPaths::from_dir(models),
        Config {
            backend: Backend::Cpu,
            max_batch_size: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let lines = engine.ocr(&image, OcrOptions::default()).unwrap();
    assert!(!lines.is_empty());
    assert!(lines.iter().any(|l| !l.text.is_empty()));
    let json = serde_json::to_string(&lines).unwrap();
    let decoded: Vec<meikiocr::TextLine> = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.len(), lines.len());
    let crops: Vec<_> = lines
        .iter()
        .map(|l| {
            let [x1, y1, x2, y2] = l.bbox;
            image::imageops::crop_imm(&image, x1, y1, x2 - x1, y2 - y1).to_image()
        })
        .collect();
    let a = engine.recognize(&crops, OcrOptions::default()).unwrap();
    assert_eq!(a.len(), crops.len());
    assert!(a.iter().all(|l| l.detection_confidence.is_none()));
    let mut batched = MeikiOcr::with_config(
        ModelPaths::from_dir(std::env::var("MEIKIOCR_MODEL_DIR").unwrap()),
        Config {
            backend: Backend::Cpu,
            max_batch_size: 8,
            ..Default::default()
        },
    )
    .unwrap();
    let b = batched.ocr(&image, OcrOptions::default()).unwrap();
    assert_eq!(
        lines.iter().map(|l| &l.text).collect::<Vec<_>>(),
        b.iter().map(|l| &l.text).collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires downloaded models and ONNX Runtime"]
fn vertical_segments_and_empty_inputs() {
    let models = std::env::var("MEIKIOCR_MODEL_DIR").expect("MEIKIOCR_MODEL_DIR");
    let mut engine = MeikiOcr::with_config(
        ModelPaths::from_dir(models),
        Config {
            backend: Backend::Cpu,
            max_batch_size: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let crops = vec![
        image::RgbImage::from_pixel(32, 480, image::Rgb([0, 0, 0])),
        image::RgbImage::from_pixel(32, 1000, image::Rgb([0, 0, 0])),
        image::RgbImage::from_pixel(2000, 32, image::Rgb([0, 0, 0])),
    ];
    let results = engine.recognize(&crops, OcrOptions::default()).unwrap();
    assert_eq!(results.len(), 3);
    assert!(results[0].is_vertical && results[1].is_vertical);
    assert!(!results[2].is_vertical);
    assert!(
        engine
            .ocr(&image::RgbImage::new(0, 0), OcrOptions::default())
            .is_err()
    );
    assert!(engine.detect(&crops[0], f32::NAN).is_err());
    assert!(
        engine
            .recognize(&[], OcrOptions::default())
            .unwrap()
            .is_empty()
    );
}
