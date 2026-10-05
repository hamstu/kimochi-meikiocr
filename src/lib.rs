//! Local Japanese OCR. Load models once, then reuse [`MeikiOcr`] on a worker thread.
//! Images use RGB channel order; the model's BGR conversion is handled internally.
mod pipeline;
mod runtime;
use image::RgbImage;
use ort::session::Session;
pub use runtime::{Backend, BackendInfo};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Errors from configuration, model contracts, image decoding, or inference.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    InvalidInput(String),
    #[error("unexpected model output: {0}")]
    ModelContract(String),
    #[error(transparent)]
    Runtime(#[from] ort::Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
}
pub type Result<T> = std::result::Result<T, Error>;

/// Model files are supplied by the host application; inference never uses the network.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub detection: PathBuf,
    pub recognition: PathBuf,
    pub vertical_recognition: PathBuf,
}
impl ModelPaths {
    pub fn from_dir(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        Self {
            detection: dir.join("meiki.text.detect.v0.1.960x544.onnx"),
            recognition: dir.join("meiki.text.rec.v0.960x32.onnx"),
            vertical_recognition: dir.join("meiki.text.rec.v0.vertical.32x480.onnx"),
        }
    }
}
#[derive(Debug, Clone)]
pub struct Config {
    pub backend: Backend,
    /// Maximum recognition segments per inference call. Must be positive.
    pub max_batch_size: usize,
    /// CPU worker threads; None uses ONNX Runtime's default.
    pub intra_threads: Option<usize>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            max_batch_size: 8,
            intra_threads: None,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct OcrOptions {
    pub detection_threshold: f32,
    pub recognition_threshold: f32,
    pub punctuation_confidence_factor: f32,
}
impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            detection_threshold: 0.5,
            recognition_threshold: 0.1,
            punctuation_confidence_factor: 1.0,
        }
    }
}
impl OcrOptions {
    fn validate(self) -> Result<()> {
        for v in [self.detection_threshold, self.recognition_threshold] {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err(Error::InvalidInput(
                    "thresholds must be finite and in [0, 1]".into(),
                ));
            }
        }
        if !self.punctuation_confidence_factor.is_finite()
            || self.punctuation_confidence_factor < 0.0
        {
            return Err(Error::InvalidInput(
                "punctuation factor must be finite and nonnegative".into(),
            ));
        }
        Ok(())
    }
}
/// Pixel coordinates [left, top, right, bottom], with exclusive right/bottom edges.
pub type BoundingBox = [u32; 4];
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub confidence: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Character {
    pub character: char,
    pub bbox: BoundingBox,
    pub confidence: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextLine {
    pub text: String,
    pub bbox: BoundingBox,
    pub detection_confidence: Option<f32>,
    pub characters: Vec<Character>,
    pub is_vertical: bool,
}
/// Reusable sessions. Methods take &mut self; share through a Mutex or dedicated worker.
pub struct MeikiOcr {
    detection: Session,
    recognition: Session,
    vertical: Session,
    config: Config,
    backend: BackendInfo,
}
impl MeikiOcr {
    pub fn new(models: ModelPaths) -> Result<Self> {
        Self::with_config(models, Config::default())
    }
    pub fn with_config(models: ModelPaths, config: Config) -> Result<Self> {
        if config.max_batch_size == 0 || config.intra_threads == Some(0) {
            return Err(Error::InvalidInput(
                "batch size and thread count must be positive".into(),
            ));
        }
        for path in [
            &models.detection,
            &models.recognition,
            &models.vertical_recognition,
        ] {
            if !path.is_file() {
                return Err(Error::InvalidInput(format!(
                    "model file missing: {}",
                    path.display()
                )));
            }
        }
        let (detection, recognition, vertical, backend) = runtime::load(&models, &config)?;
        Ok(Self {
            detection,
            recognition,
            vertical,
            config,
            backend,
        })
    }
    /// Provider successfully registered for all sessions; individual unsupported ops can use CPU.
    pub fn backend_info(&self) -> &BackendInfo {
        &self.backend
    }
    pub fn ocr_file(
        &mut self,
        path: impl AsRef<Path>,
        options: OcrOptions,
    ) -> Result<Vec<TextLine>> {
        self.ocr(&image::open(path)?.to_rgb8(), options)
    }
    pub fn ocr(&mut self, image: &RgbImage, options: OcrOptions) -> Result<Vec<TextLine>> {
        options.validate()?;
        let boxes = self.detect(image, options.detection_threshold)?;
        self.recognize_boxes(image, &boxes, options)
    }
    /// Detect text lines, sorted top to bottom.
    pub fn detect(&mut self, image: &RgbImage, threshold: f32) -> Result<Vec<Detection>> {
        OcrOptions {
            detection_threshold: threshold,
            ..Default::default()
        }
        .validate()?;
        pipeline::detect(&mut self.detection, image, threshold)
    }
    /// Recognize already cropped lines. Coordinates are relative to each crop.
    pub fn recognize(&mut self, images: &[RgbImage], options: OcrOptions) -> Result<Vec<TextLine>> {
        options.validate()?;
        images
            .iter()
            .map(|img| {
                pipeline::validate_image(img)?;
                let boxes = [Detection {
                    bbox: [0, 0, img.width(), img.height()],
                    confidence: 1.0,
                }];
                let mut lines = self.recognize_boxes(img, &boxes, options)?;
                lines[0].detection_confidence = None;
                Ok(lines.remove(0))
            })
            .collect()
    }
    fn recognize_boxes(
        &mut self,
        image: &RgbImage,
        boxes: &[Detection],
        options: OcrOptions,
    ) -> Result<Vec<TextLine>> {
        pipeline::recognize(
            &mut self.recognition,
            &mut self.vertical,
            image,
            boxes,
            options,
            self.config.max_batch_size,
        )
    }
}
