//! Opt-in real model validation: MEIKIOCR_MODEL_DIR + MEIKIOCR_TEST_IMAGE + ORT_DYLIB_PATH.
use meikiocr::{Backend, Config, MeikiOcr, ModelPaths, OcrOptions};
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
