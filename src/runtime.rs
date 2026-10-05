use crate::{Config, ModelPaths, Result};
use ort::{
    execution_providers::ExecutionProviderDispatch,
    session::{Session, builder::GraphOptimizationLevel},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Backend {
    /// Try platform accelerators, then CPU. Failures are recorded in BackendInfo.
    #[default]
    Auto,
    Cpu,
    CoreMl,
    Cuda,
    DirectMl,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    pub requested: Backend,
    pub selected: Backend,
    pub fallback_reasons: Vec<String>,
}
fn provider(backend: Backend) -> Result<Option<ExecutionProviderDispatch>> {
    match backend {
        Backend::Cpu => Ok(None),
        #[cfg(feature = "coreml")]
        Backend::CoreMl => Ok(Some(
            ort::execution_providers::CoreMLExecutionProvider::default()
                .with_static_input_shapes(true)
                .with_model_format(
                    ort::execution_providers::coreml::CoreMLModelFormat::NeuralNetwork,
                )
                .with_compute_units(ort::execution_providers::coreml::CoreMLComputeUnits::All)
                .build()
                .error_on_failure(),
        )),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(Some(
            ort::execution_providers::CUDAExecutionProvider::default()
                .build()
                .error_on_failure(),
        )),
        #[cfg(feature = "directml")]
        Backend::DirectMl => Ok(Some(
            ort::execution_providers::DirectMLExecutionProvider::default()
                .build()
                .error_on_failure(),
        )),
        _ => Err(crate::Error::InvalidInput(format!(
            "backend {backend:?} is not compiled in"
        ))),
    }
}
fn session(path: &std::path::Path, backend: Backend, config: &Config) -> Result<Session> {
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_op_spinning(false)?
        .with_config_entry("session.inter_op.allow_spinning", "0")?;
    if let Some(n) = config.intra_threads {
        builder = builder.with_intra_threads(n)?;
    }
    if backend == Backend::DirectMl {
        builder = builder
            .with_parallel_execution(false)?
            .with_memory_pattern(false)?;
    }
    if let Some(ep) = provider(backend)? {
        builder = builder.with_execution_providers([ep])?;
    }
    Ok(builder.commit_from_file(path)?)
}

// Some Windows adapters accept DirectML sessions but fail on the first inference.
// Exercise every model before reporting the backend as usable. Keep this limited
// to DirectML: other platforms retain their existing initialization behavior.
#[cfg(all(target_os = "windows", feature = "directml"))]
fn probe_directml(session: &mut Session, width: usize, height: usize) -> Result<()> {
    use ndarray::{Array4, arr2};
    use ort::value::Tensor;

    if session.inputs.len() != 2 {
        return Err(crate::Error::ModelContract(
            "DirectML probe requires two model inputs".into(),
        ));
    }
    let names = [
        session.inputs[0].name.clone(),
        session.inputs[1].name.clone(),
    ];
    session.run(ort::inputs![
        names[0].as_str() => Tensor::from_array(Array4::<f32>::zeros((1, 3, height, width)))?,
        names[1].as_str() => Tensor::from_array(arr2(&[[width as i64, height as i64]]))?
    ])?;
    Ok(())
}

pub(crate) fn load(
    models: &ModelPaths,
    config: &Config,
) -> Result<(Session, Session, Session, BackendInfo)> {
    let mut candidates = Vec::new();
    if config.backend == Backend::Auto {
        if cfg!(all(
            target_os = "macos",
            target_arch = "aarch64",
            feature = "coreml"
        )) {
            candidates.push(Backend::CoreMl);
        }
        if cfg!(all(
            any(target_os = "windows", target_os = "linux"),
            feature = "cuda"
        )) {
            candidates.push(Backend::Cuda);
        }
        if cfg!(all(target_os = "windows", feature = "directml")) {
            candidates.push(Backend::DirectMl);
        }
        candidates.push(Backend::Cpu);
    } else {
        candidates.push(config.backend);
    }
    let mut reasons = Vec::new();
    for backend in candidates {
        let attempt = (|| {
            let sessions = (
                session(&models.detection, backend, config)?,
                session(&models.recognition, backend, config)?,
                session(&models.vertical_recognition, backend, config)?,
            );
            #[cfg(all(target_os = "windows", feature = "directml"))]
            let sessions = {
                let (mut det, mut rec, mut vert) = sessions;
                if backend == Backend::DirectMl {
                    for (name, model, width, height) in [
                        ("detection", &mut det, 960, 544),
                        ("recognition", &mut rec, 960, 32),
                        ("vertical recognition", &mut vert, 32, 480),
                    ] {
                        probe_directml(model, width, height).map_err(|err| {
                            crate::Error::ModelContract(format!(
                                "DirectML {name} inference probe failed: {err}"
                            ))
                        })?;
                    }
                }
                (det, rec, vert)
            };
            Ok(sessions)
        })();
        match attempt {
            Ok((det, rec, vert)) => {
                return Ok((
                    det,
                    rec,
                    vert,
                    BackendInfo {
                        requested: config.backend,
                        selected: backend,
                        fallback_reasons: reasons,
                    },
                ));
            }
            Err(err) => {
                if config.backend != Backend::Auto || backend == Backend::Cpu {
                    return Err(err);
                }
                reasons.push(format!("{backend:?}: {err}"));
            }
        }
    }
    unreachable!("CPU is always a candidate")
}
