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
            Ok((
                session(&models.detection, backend, config)?,
                session(&models.recognition, backend, config)?,
                session(&models.vertical_recognition, backend, config)?,
            ))
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
