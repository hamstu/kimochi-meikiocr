use crate::{BoundingBox, Character, Detection, Error, OcrOptions, Result, TextLine};
use image::RgbImage;
use ndarray::{Array2, Array4};
use ort::{
    session::{Session, SessionOutputs},
    value::Tensor,
};
use unicode_general_category::{GeneralCategory as G, get_general_category};

pub(crate) fn validate_image(image: &RgbImage) -> Result<()> {
    if image.width() == 0 || image.height() == 0 {
        return Err(Error::InvalidInput(
            "image dimensions must be nonzero".into(),
        ));
    }
    Ok(())
}
// OpenCV INTER_LINEAR uses half-pixel sampling. Preserve BGR order and uint8
// rounding before normalization rather than applying a different RGB transform.
fn tensor(
    image: &RgbImage,
    bbox: BoundingBox,
    size: (usize, usize),
    effective: (usize, usize),
) -> Array4<f32> {
    let (tw, th) = size;
    let (ew, eh) = effective;
    let [x1, y1, x2, y2] = bbox;
    let w = (x2 - x1) as usize;
    let h = (y2 - y1) as usize;
    let mut out = Array4::zeros((1, 3, th, tw));
    for y in 0..eh {
        let fy = ((y as f64 + 0.5) * h as f64 / eh as f64 - 0.5).max(0.0);
        let ya = (fy.floor() as usize).min(h - 1);
        let yb = (ya + 1).min(h - 1);
        let dy = fy - ya as f64;
        for x in 0..ew {
            let fx = ((x as f64 + 0.5) * w as f64 / ew as f64 - 0.5).max(0.0);
            let xa = (fx.floor() as usize).min(w - 1);
            let xb = (xa + 1).min(w - 1);
            let dx = fx - xa as f64;
            for c in 0..3 {
                let p = |xx, yy| image.get_pixel(x1 + xx as u32, y1 + yy as u32)[2 - c] as f64;
                let value = (p(xa, ya) * (1.0 - dx) + p(xb, ya) * dx) * (1.0 - dy)
                    + (p(xa, yb) * (1.0 - dx) + p(xb, yb) * dx) * dy;
                out[[0, c, y, x]] = value.round() as f32 / 255.0;
            }
        }
    }
    out
}
struct Raw {
    labels: Vec<i64>,
    boxes: Vec<f32>,
    scores: Vec<f32>,
    batch: usize,
    count: usize,
}
fn decode(outputs: &SessionOutputs<'_>, batch: usize) -> Result<Raw> {
    if outputs.len() != 3 {
        return Err(Error::ModelContract(
            "expected labels, boxes, scores".into(),
        ));
    }
    let (ls, labels) = match outputs[0].try_extract_tensor::<i64>() {
        Ok((shape, data)) => (shape, data.to_vec()),
        Err(_) => {
            let (shape, data) = outputs[0].try_extract_tensor::<i32>()?;
            (shape, data.iter().map(|&v| i64::from(v)).collect())
        }
    };
    let (bs, boxes) = outputs[1].try_extract_tensor::<f32>()?;
    let (ss, scores) = outputs[2].try_extract_tensor::<f32>()?;
    if ls.len() != 2
        || ls[0] != batch as i64
        || ls[1] <= 0
        || ss != ls
        || bs.len() != 3
        || bs[0] != ls[0]
        || bs[1] != ls[1]
        || bs[2] != 4
    {
        return Err(Error::ModelContract(format!(
            "invalid output shapes: {ls:?}, {bs:?}, {ss:?}"
        )));
    }
    Ok(Raw {
        labels,
        boxes: boxes.to_vec(),
        scores: scores.to_vec(),
        batch,
        count: ls[1] as usize,
    })
}
pub(crate) fn detect(
    session: &mut Session,
    image: &RgbImage,
    threshold: f32,
) -> Result<Vec<Detection>> {
    validate_image(image)?;
    let scale = (960.0 / image.width() as f64).min(544.0 / image.height() as f64);
    let effective = (
        ((image.width() as f64 * scale) as usize).clamp(1, 960),
        ((image.height() as f64 * scale) as usize).clamp(1, 544),
    );
    let input = tensor(
        image,
        [0, 0, image.width(), image.height()],
        (960, 544),
        effective,
    );
    let sizes =
        Array2::from_shape_vec((1, 2), vec![(960.0 / scale) as i64, (544.0 / scale) as i64])
            .map_err(|e| Error::ModelContract(e.to_string()))?;
    if session.inputs.len() != 2 {
        return Err(Error::ModelContract("detection requires two inputs".into()));
    }
    let names = [
        session.inputs[0].name.clone(),
        session.inputs[1].name.clone(),
    ];
    let outputs = session.run(ort::inputs![names[0].as_str() => Tensor::from_array(input)?, names[1].as_str() => Tensor::from_array(sizes)?])?;
    let raw = decode(&outputs, 1)?;
    let bounds = [image.width(), image.height(), image.width(), image.height()];
    let mut detections = Vec::new();
    for i in 0..raw.count {
        let score = raw.scores[i];
        if !score.is_finite() || score <= threshold {
            continue;
        }
        let b = &raw.boxes[i * 4..i * 4 + 4];
        if b.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let bbox = std::array::from_fn(|j| b[j].clamp(0.0, bounds[j] as f32) as u32);
        if bbox[2] > bbox[0] && bbox[3] > bbox[1] {
            detections.push(Detection {
                bbox,
                confidence: score,
            });
        }
    }
    detections.sort_by_key(|d| d.bbox[1]);
    Ok(detections)
}
#[derive(Clone, Debug)]
struct Segment {
    line: usize,
    bbox: BoundingBox,
    effective: (usize, usize),
}
fn segments(boxes: &[Detection], vertical: bool) -> Vec<Segment> {
    let mut out = Vec::new();
    for (line, d) in boxes.iter().enumerate() {
        let [x1, y1, x2, y2] = d.bbox;
        let w = x2 - x1;
        let h = y2 - y1;
        if w == 0 || h == 0 || (h > w) != vertical {
            continue;
        }
        if !vertical {
            let mut nw = (w as f64 * 32.0 / h as f64).round_ties_even().max(1.0) as usize;
            let mut nh = 32;
            if nw > 960 {
                nh = (32.0 * 960.0 / nw as f64).round_ties_even().max(1.0) as usize;
                nw = 960;
            }
            out.push(Segment {
                line,
                bbox: d.bbox,
                effective: (nw, nh),
            });
        } else {
            let scale = 32.0 / w as f64;
            let split = h as f64 * scale > 480.0;
            let length = if split { 420.0 / scale } else { h as f64 };
            let mut starts = Vec::new();
            if split {
                let mut y = y1 as f64;
                while y + length < y2 as f64 {
                    starts.push(y);
                    y += 356.0 / scale;
                }
                let last = y2 as f64 - length;
                if starts.last().is_none_or(|prev| last > prev + 1.0) {
                    starts.push(last);
                }
            } else {
                starts.push(y1 as f64);
            }
            for start in starts {
                let sy = start.round_ties_even() as u32;
                let ey = ((start + length).round_ties_even() as u32).min(y2);
                if ey <= sy {
                    continue;
                }
                let nh = (((ey - sy) as f64 * scale).round_ties_even() as usize)
                    .clamp(1, if split { 420 } else { 480 });
                out.push(Segment {
                    line,
                    bbox: [x1, sy, x2, ey],
                    effective: (32, nh),
                });
            }
        }
    }
    out
}
fn punctuation(c: char) -> bool {
    matches!(
        get_general_category(c),
        G::ClosePunctuation
            | G::ConnectorPunctuation
            | G::DashPunctuation
            | G::FinalPunctuation
            | G::InitialPunctuation
            | G::OpenPunctuation
            | G::OtherPunctuation
    )
}
fn suppress(mut candidates: Vec<Character>, vertical: bool, factor: f32) -> Vec<Character> {
    let axis = usize::from(vertical);
    for c in &mut candidates {
        if punctuation(c.character) {
            c.confidence *= factor;
        }
    }
    candidates.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut accepted: Vec<Character> = Vec::new();
    for c in candidates {
        let start = c.bbox[axis] as f64;
        let end = c.bbox[axis + 2] as f64;
        if accepted.iter().any(|a| {
            let a1 = a.bbox[axis] as f64;
            let a2 = a.bbox[axis + 2] as f64;
            let overlap = (end.min(a2) - start.max(a1)).max(0.0);
            overlap / ((end - start).min(a2 - a1) + 1e-6) > 0.3
        }) {
            continue;
        }
        accepted.push(c);
    }
    accepted.sort_by_key(|c| c.bbox[axis]);
    for (wrong, _) in [
        ("儡傀", "傀儡"),
        ("談冗", "冗談"),
        ("汰淘", "淘汰"),
        ("沱滂", "滂沱"),
        ("攣痙", "痙攣"),
        ("酊酩", "酩酊"),
        ("麭麺", "麺麭"),
        ("哭慟", "慟哭"),
    ] {
        let pair: Vec<_> = wrong.chars().collect();
        if let Some(i) = accepted
            .windows(2)
            .position(|cs| cs[0].character == pair[0] && cs[1].character == pair[1])
        {
            let c = accepted[i].character;
            accepted[i].character = accepted[i + 1].character;
            accepted[i + 1].character = c;
        }
    }
    accepted
}
pub(crate) fn recognize(
    horizontal: &mut Session,
    vertical_session: &mut Session,
    image: &RgbImage,
    boxes: &[Detection],
    options: OcrOptions,
    batch_size: usize,
) -> Result<Vec<TextLine>> {
    let mut candidates = vec![Vec::new(); boxes.len()];
    for (vertical, session) in [(false, horizontal), (true, vertical_session)] {
        let size = if vertical { (32, 480) } else { (960, 32) };
        for chunk in segments(boxes, vertical).chunks(batch_size) {
            let mut batch = Array4::<f32>::zeros((chunk.len(), 3, size.1, size.0));
            for (i, s) in chunk.iter().enumerate() {
                batch
                    .slice_mut(ndarray::s![i..i + 1, .., .., ..])
                    .assign(&tensor(image, s.bbox, size, s.effective));
            }
            let sizes = Array2::from_shape_vec((1, 2), vec![size.0 as i64, size.1 as i64])
                .map_err(|e| Error::ModelContract(e.to_string()))?;
            let outputs = session.run(ort::inputs!["images" => Tensor::from_array(batch)?, "orig_target_sizes" => Tensor::from_array(sizes)?])?;
            let raw = decode(&outputs, chunk.len())?;
            for (i, s) in chunk.iter().enumerate().take(raw.batch) {
                for j in 0..raw.count {
                    let k = i * raw.count + j;
                    let score = raw.scores[k];
                    if !score.is_finite() || score < options.recognition_threshold {
                        continue;
                    }
                    let Some(character) =
                        u32::try_from(raw.labels[k]).ok().and_then(char::from_u32)
                    else {
                        continue;
                    };
                    let mut b: [f32; 4] = raw.boxes[k * 4..k * 4 + 4]
                        .try_into()
                        .map_err(|_| Error::ModelContract("invalid character box".into()))?;
                    if b.iter().any(|v| !v.is_finite()) {
                        continue;
                    }
                    let axis = usize::from(vertical);
                    let effective_axis = if vertical {
                        s.effective.1
                    } else {
                        s.effective.0
                    } as f32;
                    if b[axis] >= effective_axis {
                        continue;
                    }
                    b[axis] = b[axis].min(effective_axis);
                    b[axis + 2] = b[axis + 2].min(effective_axis);
                    // Use effective height for compressed horizontal crops too, so coordinates
                    // stay accurate when a line exceeds the horizontal model's width.
                    let ew = s.effective.0 as f32;
                    let eh = s.effective.1 as f32;
                    let [x1, y1, x2, y2] = s.bbox;
                    let mapped: [u32; 4] = std::array::from_fn(|n| {
                        let (origin, extent, denominator) = if n % 2 == 0 {
                            (x1, x2 - x1, ew)
                        } else {
                            (y1, y2 - y1, eh)
                        };
                        (origin as f32 + (b[n] / denominator * extent as f32).trunc())
                            .clamp(origin as f32, (origin + extent) as f32)
                            as u32
                    });
                    if mapped[2] <= mapped[0] || mapped[3] <= mapped[1] {
                        continue;
                    }
                    candidates[s.line].push(Character {
                        character,
                        bbox: mapped,
                        confidence: score,
                    });
                }
            }
        }
    }
    Ok(boxes
        .iter()
        .zip(candidates)
        .map(|(d, cs)| {
            let is_vertical = d.bbox[3] - d.bbox[1] > d.bbox[2] - d.bbox[0];
            let characters = suppress(cs, is_vertical, options.punctuation_confidence_factor);
            TextLine {
                text: characters.iter().map(|c| c.character).collect(),
                bbox: d.bbox,
                detection_confidence: Some(d.confidence),
                characters,
                is_vertical,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rgb_is_bgr_and_padding_is_zero() {
        let img = RgbImage::from_pixel(1, 1, image::Rgb([255, 128, 0]));
        let t = tensor(&img, [0, 0, 1, 1], (2, 2), (1, 1));
        assert_eq!(t[[0, 0, 0, 0]], 0.0);
        assert_eq!(t[[0, 2, 0, 0]], 1.0);
        assert_eq!(t[[0, 1, 1, 1]], 0.0);
    }
    #[test]
    fn vertical_splits_cover_end_with_overlap() {
        let boxes = [Detection {
            bbox: [10, 20, 42, 1020],
            confidence: 1.0,
        }];
        let ss = segments(&boxes, true);
        assert_eq!(ss.len(), 3);
        assert_eq!(ss[0].bbox[1], 20);
        assert_eq!(ss.last().unwrap().bbox[3], 1020);
        assert!(ss.windows(2).all(|s| s[1].bbox[1] < s[0].bbox[3]));
    }
    #[test]
    fn punctuation_factor_changes_overlap_winner() {
        let cs = vec![
            Character {
                character: '。',
                bbox: [0, 0, 10, 10],
                confidence: 0.9,
            },
            Character {
                character: 'あ',
                bbox: [1, 0, 11, 10],
                confidence: 0.8,
            },
        ];
        assert_eq!(suppress(cs, false, 0.5)[0].character, 'あ');
    }
    #[test]
    fn correction_keeps_coordinate_order() {
        let cs = vec![
            Character {
                character: '談',
                bbox: [0, 0, 10, 10],
                confidence: 0.9,
            },
            Character {
                character: '冗',
                bbox: [10, 0, 20, 10],
                confidence: 0.8,
            },
        ];
        let fixed = suppress(cs, false, 1.0);
        assert_eq!(
            fixed.iter().map(|c| c.character).collect::<String>(),
            "冗談"
        );
        assert_eq!(fixed[0].bbox, [0, 0, 10, 10]);
    }
}
