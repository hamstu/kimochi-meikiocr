//! DirectML's matrix-vector MatMul broadcasts the first vector value on ORT 1.22.
//! Promote constant RHS vectors to columns, then squeeze the result. Preserve all
//! other protobuf fields verbatim, including model metadata and tensor storage.
use crate::{Error, Result};
use std::collections::HashMap;

struct Field<'a> {
    number: u64,
    wire: u64,
    value: &'a [u8],
    encoded: &'a [u8],
}

fn invalid() -> Error {
    Error::ModelContract("invalid ONNX protobuf".into())
}

fn varint(bytes: &[u8], offset: &mut usize) -> Result<u64> {
    let mut value = 0;
    for shift in (0..70).step_by(7) {
        let byte = *bytes.get(*offset).ok_or_else(invalid)?;
        *offset += 1;
        if shift == 63 && byte > 1 {
            return Err(invalid());
        }
        value |= u64::from(byte & 127) << shift;
        if byte < 128 {
            return Ok(value);
        }
    }
    Err(invalid())
}

fn fields(bytes: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut offset = 0;
    let mut result = Vec::new();
    while offset < bytes.len() {
        let start = offset;
        let key = varint(bytes, &mut offset)?;
        let wire = key & 7;
        if key >> 3 == 0 {
            return Err(invalid());
        }
        let value_start = match wire {
            0 => {
                let pos = offset;
                varint(bytes, &mut offset)?;
                pos
            }
            1 | 5 => {
                let pos = offset;
                offset = offset
                    .checked_add(if wire == 1 { 8 } else { 4 })
                    .ok_or_else(invalid)?;
                pos
            }
            2 => {
                let size = usize::try_from(varint(bytes, &mut offset)?).map_err(|_| invalid())?;
                let pos = offset;
                offset = offset.checked_add(size).ok_or_else(invalid)?;
                pos
            }
            _ => return Err(invalid()),
        };
        result.push(Field {
            number: key >> 3,
            wire,
            value: bytes.get(value_start..offset).ok_or_else(invalid)?,
            encoded: &bytes[start..offset],
        });
    }
    Ok(result)
}

fn encode(mut value: u64, out: &mut Vec<u8>) {
    while value >= 128 {
        out.push(value as u8 | 128);
        value >>= 7;
    }
    out.push(value as u8);
}

fn data(number: u64, value: &[u8], out: &mut Vec<u8>) {
    encode(number << 3 | 2, out);
    encode(value.len() as u64, out);
    out.extend_from_slice(value);
}

fn integer(number: u64, value: u64, out: &mut Vec<u8>) {
    encode(number << 3, out);
    encode(value, out);
}

fn named<'a>(fields: &[Field<'a>], number: u64) -> Option<&'a [u8]> {
    fields
        .iter()
        .find(|f| f.number == number && f.wire == 2)
        .map(|f| f.value)
}

fn vector(tensor: &[u8]) -> Result<bool> {
    let mut dims = Vec::new();
    for field in fields(tensor)? {
        if field.number == 1 {
            let mut offset = 0;
            match field.wire {
                0 | 2 => {
                    while offset < field.value.len() {
                        dims.push(varint(field.value, &mut offset)?);
                    }
                }
                _ => return Err(invalid()),
            }
        }
    }
    Ok(dims.len() == 1 && dims[0] > 0)
}

fn node(op: &[u8], inputs: &[&[u8]], output: &[u8], out: &mut Vec<u8>) {
    let mut bytes = Vec::new();
    for input in inputs {
        data(1, input, &mut bytes);
    }
    data(2, output, &mut bytes);
    data(4, op, &mut bytes);
    data(1, &bytes, out);
}

fn graph(bytes: &[u8]) -> Result<(Vec<u8>, bool)> {
    let source = fields(bytes)?;
    let mut initializers = HashMap::new();
    for field in &source {
        if field.number == 5 && field.wire == 2 && vector(field.value)? {
            let tensor = fields(field.value)?;
            if let Some(name) = named(&tensor, 8) {
                initializers.insert(name, field.value);
            }
        }
    }
    let mut out = Vec::with_capacity(bytes.len());
    let mut additions = Vec::new();
    let mut changed = false;
    for (index, field) in source.iter().enumerate() {
        if field.number != 1 || field.wire != 2 {
            out.extend_from_slice(field.encoded);
            continue;
        }
        let original = fields(field.value)?;
        let inputs: Vec<_> = original
            .iter()
            .filter(|f| f.number == 1 && f.wire == 2)
            .collect();
        let output = named(&original, 2);
        let rhs = inputs.get(1).and_then(|f| initializers.get(f.value));
        let standard =
            named(&original, 7).is_none_or(|domain| domain.is_empty() || domain == b"ai.onnx");
        if !standard
            || named(&original, 4) != Some(b"MatMul")
            || inputs.len() != 2
            || rhs.is_none()
            || output.is_none()
        {
            out.extend_from_slice(field.encoded);
            continue;
        }
        let prefix = format!("__meiki_dml_matvec_{index}");
        // Refuse collisions rather than silently change a graph's existing names.
        if bytes
            .windows(prefix.len())
            .any(|window| window == prefix.as_bytes())
        {
            return Err(Error::ModelContract(
                "DirectML MatMul rewrite name collision".into(),
            ));
        }
        let column = format!("{prefix}_weight");
        let product = format!("{prefix}_product");
        let axes = format!("{prefix}_axes");
        let mut weight = Vec::new();
        for part in fields(rhs.unwrap())? {
            if part.number != 8 {
                weight.extend_from_slice(part.encoded);
            }
        }
        integer(1, 1, &mut weight);
        data(8, column.as_bytes(), &mut weight);
        data(5, &weight, &mut additions);
        let mut axis = Vec::new();
        integer(1, 1, &mut axis);
        integer(2, 7, &mut axis); // TensorProto.INT64
        data(8, axes.as_bytes(), &mut axis);
        data(9, &(-1_i64).to_le_bytes(), &mut axis);
        data(5, &axis, &mut additions);
        let mut rewritten = Vec::new();
        let mut input_index = 0;
        for part in original {
            match part.number {
                1 => {
                    data(
                        1,
                        if input_index == 1 {
                            column.as_bytes()
                        } else {
                            part.value
                        },
                        &mut rewritten,
                    );
                    input_index += 1;
                }
                2 => data(2, product.as_bytes(), &mut rewritten),
                _ => rewritten.extend_from_slice(part.encoded),
            }
        }
        data(1, &rewritten, &mut out);
        node(
            b"Squeeze",
            &[product.as_bytes(), axes.as_bytes()],
            output.unwrap(),
            &mut out,
        );
        changed = true;
    }
    out.extend(additions);
    Ok((out, changed))
}

pub(crate) fn directml_model(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut changed = false;
    let mut modern_opset = false;
    for field in fields(bytes)? {
        if field.number == 7 && field.wire == 2 {
            let (graph, patched) = graph(field.value)?;
            data(7, &graph, &mut out);
            changed |= patched;
        } else {
            if field.number == 8 && field.wire == 2 {
                let opset = fields(field.value)?;
                if named(&opset, 1).is_none_or(|d| d.is_empty() || d == b"ai.onnx") {
                    for version in opset.iter().filter(|f| f.number == 2 && f.wire == 0) {
                        modern_opset |= varint(version.value, &mut 0)? >= 13;
                    }
                }
            }
            out.extend_from_slice(field.encoded);
        }
    }
    if changed && !modern_opset {
        return Err(Error::ModelContract(
            "DirectML MatMul rewrite requires ONNX opset 13+".into(),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MATVEC: &[u8] = include_bytes!("../tests/fixtures/matvec.onnx");

    #[test]
    fn column_rewrite_preserves_metadata_storage_and_output_contract() {
        let mut source = MATVEC.to_vec();
        data(100, b"unknown future model metadata", &mut source);
        let patched = directml_model(&source).unwrap();
        let model = fields(&patched).unwrap();
        assert_eq!(
            named(&model, 100),
            Some(b"unknown future model metadata".as_slice())
        );
        let graph = fields(named(&model, 7).unwrap()).unwrap();
        let nodes: Vec<_> = graph.iter().filter(|f| f.number == 1).collect();
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            named(&fields(nodes[0].value).unwrap(), 4),
            Some(b"MatMul".as_slice())
        );
        let squeeze = fields(nodes[1].value).unwrap();
        assert_eq!(named(&squeeze, 4), Some(b"Squeeze".as_slice()));
        assert_eq!(named(&squeeze, 2), Some(b"y".as_slice()));
        let original_model = fields(MATVEC).unwrap();
        let original_graph = fields(named(&original_model, 7).unwrap()).unwrap();
        let original_weight = original_graph.iter().find(|f| f.number == 5).unwrap().value;
        let weights: Vec<_> = graph.iter().filter(|f| f.number == 5).collect();
        assert_eq!(weights[0].value, original_weight);
        let column = fields(weights[1].value).unwrap();
        assert!(!vector(weights[1].value).unwrap());
        assert_eq!(
            named(&column, 9),
            named(&fields(original_weight).unwrap(), 9)
        );
        assert_eq!(directml_model(&patched).unwrap(), patched);
    }

    #[test]
    fn rejects_truncated_and_overflowing_wire_fields() {
        for bytes in [&[0][..], &[0x3a, 0xff], &[0x3a, 4, 0], &[0xff; 12]] {
            assert!(directml_model(bytes).is_err());
        }
    }

    #[test]
    fn accepts_packed_tensor_dimensions_and_leaves_matrices_alone() {
        let mut tensor = Vec::new();
        data(1, &[33], &mut tensor);
        assert!(vector(&tensor).unwrap());
        integer(1, 1, &mut tensor);
        assert!(!vector(&tensor).unwrap());
    }

    #[test]
    fn refuses_an_opset_without_squeeze_axes_input() {
        let mut source = Vec::new();
        for field in fields(MATVEC).unwrap() {
            if field.number == 8 {
                let mut opset = Vec::new();
                integer(2, 12, &mut opset);
                data(8, &opset, &mut source);
            } else {
                source.extend_from_slice(field.encoded);
            }
        }
        assert!(
            directml_model(&source)
                .unwrap_err()
                .to_string()
                .contains("opset 13+")
        );
    }

    #[cfg(all(target_os = "windows", feature = "directml"))]
    #[test]
    #[ignore = "requires DirectML ONNX Runtime via ORT_DYLIB_PATH"]
    fn directml_matrix_vector_matches_cpu() {
        use ort::{
            execution_providers::DirectMLExecutionProvider, session::Session, value::Tensor,
        };
        let input: Vec<_> = (0..66).map(|v| v as f32 / 66.0).collect();
        let mut cpu = Session::builder()
            .unwrap()
            .commit_from_memory(MATVEC)
            .unwrap();
        let cpu_values = cpu
            .run(ort::inputs!["a" => Tensor::from_array(([2, 33], input.clone())).unwrap()])
            .unwrap();
        let (_, expected) = cpu_values[0].try_extract_tensor::<f32>().unwrap();
        let patched = directml_model(MATVEC).unwrap();
        let mut dml = Session::builder()
            .unwrap()
            .with_parallel_execution(false)
            .unwrap()
            .with_memory_pattern(false)
            .unwrap()
            .with_execution_providers([DirectMLExecutionProvider::default()
                .build()
                .error_on_failure()])
            .unwrap()
            .commit_from_memory(&patched)
            .unwrap();
        let actual = dml
            .run(ort::inputs!["a" => Tensor::from_array(([2, 33], input)).unwrap()])
            .unwrap();
        let (shape, values) = actual[0].try_extract_tensor::<f32>().unwrap();
        assert_eq!(shape.as_ref(), &[2]);
        for (a, b) in values.iter().zip(expected) {
            assert!((a - b).abs() < 1e-4, "DirectML {a} differs from CPU {b}");
        }
    }
}
