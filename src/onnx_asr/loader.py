"""Loader for ASR models."""

import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

import onnxruntime as rt

from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.models.gigaam import GigaamV2Ctc, GigaamV2Rnnt, GigaamV3E2eCtc, GigaamV3E2eRnnt
from onnx_asr.models.kaldi import KaldiTransducer
from onnx_asr.models.nemo import NemoConformerAED, NemoConformerCtc, NemoConformerRnnt, NemoConformerTdt
from onnx_asr.models.pyannote import PyAnnoteVad
from onnx_asr.models.silero import SileroVad
from onnx_asr.models.tone import TOneCtc
from onnx_asr.models.whisper import WhisperHf, WhisperOrt
from onnx_asr.onnx import OnnxSessionOptions, get_onnx_providers, update_onnx_providers
from onnx_asr.preprocessors.preprocessor import (
    ConcurrentPreprocessor,
    IdentityPreprocessor,
    OnnxPreprocessor,
    Preprocessor,
)
from onnx_asr.preprocessors.resampler import Resampler
from onnx_asr.resolver import ModelResolver
from onnx_asr.vad import Vad

ModelNames = Literal[
    "gigaam-v2-ctc",
    "gigaam-v2-rnnt",
    "gigaam-v3-ctc",
    "gigaam-v3-rnnt",
    "gigaam-v3-e2e-ctc",
    "gigaam-v3-e2e-rnnt",
    "nemo-fastconformer-ru-ctc",
    "nemo-fastconformer-ru-rnnt",
    "nemo-parakeet-ctc-0.6b",
    "nemo-parakeet-rnnt-0.6b",
    "nemo-parakeet-tdt-0.6b-v2",
    "nemo-parakeet-tdt-0.6b-v3",
    "nemo-canary-1b-v2",
    "alphacep/vosk-model-ru",
    "alphacep/vosk-model-small-ru",
    "t-tech/t-one",
    "whisper-base",
]
"""Supported ASR model names (can be automatically downloaded from the Hugging Face)."""

ModelTypes = Literal[
    "kaldi-rnnt",
    "nemo-conformer-ctc",
    "nemo-conformer-rnnt",
    "nemo-conformer-tdt",
    "nemo-conformer-aed",
    "t-one-ctc",
    "vosk",
    "whisper-ort",
    "whisper",
]
"""Supported ASR model types."""

VadNames = Literal["silero"]
"""Supported VAD model names (can be automatically downloaded from the Hugging Face)."""


class PreprocessorRuntimeConfig(OnnxSessionOptions, total=False):
    """Preprocessor runtime config."""

    max_concurrent_workers: int | None
    """Max parallel preprocessing threads (None - auto, 1 - without parallel processing)."""


def _create_preprocessor(name: str, config: PreprocessorRuntimeConfig) -> Preprocessor:
    if name == "identity":
        return IdentityPreprocessor()

    providers = get_onnx_providers(config)
    if name == "kaldi" and providers and providers != ["CPUExecutionProvider"]:
        name = "kaldi_fast"

    max_concurrent_workers = config.pop("max_concurrent_workers", 1)
    preprocessor = OnnxPreprocessor(name, config)
    if max_concurrent_workers == 1:
        return preprocessor
    return ConcurrentPreprocessor(preprocessor, max_concurrent_workers)


AsrTypes: TypeAlias = (
    GigaamV2Ctc
    | GigaamV2Rnnt
    | KaldiTransducer
    | NemoConformerCtc
    | NemoConformerRnnt
    | NemoConformerAED
    | TOneCtc
    | WhisperHf
    | WhisperOrt
)


def create_asr_resolver(
    model_name: str, local_dir: str | Path | None = None, *, offline: bool | None = None
) -> ModelResolver[AsrTypes]:
    """Create resolver for ASR models."""
    model_types: dict[str, type[AsrTypes]] = {
        "gigaam-v2-ctc": GigaamV2Ctc,
        "gigaam-v2-rnnt": GigaamV2Rnnt,
        "gigaam-v3-ctc": GigaamV2Ctc,
        "gigaam-v3-rnnt": GigaamV2Rnnt,
        "gigaam-v3-e2e-ctc": GigaamV3E2eCtc,
        "gigaam-v3-e2e-rnnt": GigaamV3E2eRnnt,
        "nemo-fastconformer-ru-ctc": NemoConformerCtc,
        "nemo-fastconformer-ru-rnnt": NemoConformerRnnt,
        "nemo-parakeet-ctc-0.6b": NemoConformerCtc,
        "nemo-parakeet-rnnt-0.6b": NemoConformerRnnt,
        "nemo-parakeet-tdt-0.6b-v2": NemoConformerTdt,
        "nemo-parakeet-tdt-0.6b-v3": NemoConformerTdt,
        "nemo-canary-1b-v2": NemoConformerAED,
        "whisper-base": WhisperOrt,
        "kaldi-rnnt": KaldiTransducer,
        "nemo-conformer-ctc": NemoConformerCtc,
        "nemo-conformer-rnnt": NemoConformerRnnt,
        "nemo-conformer-tdt": NemoConformerTdt,
        "nemo-conformer-aed": NemoConformerAED,
        "t-one-ctc": TOneCtc,
        "vosk": KaldiTransducer,
        "whisper-ort": WhisperOrt,
        "whisper": WhisperHf,
        "alphacep/vosk-model-ru": KaldiTransducer,
        "alphacep/vosk-model-small-ru": KaldiTransducer,
        "t-tech/t-one": TOneCtc,
    }
    return ModelResolver[AsrTypes](model_types, model_name, local_dir, offline=offline)


VadTypes: TypeAlias = SileroVad | PyAnnoteVad


def create_vad_resolver(
    model_name: str, local_dir: str | Path | None = None, *, offline: bool | None = None
) -> ModelResolver[VadTypes]:
    """Create resolver for VAD models."""
    model_types: dict[str, type[VadTypes]] = {"silero": SileroVad, "pyannote": PyAnnoteVad}
    return ModelResolver[VadTypes](model_types, model_name, local_dir, offline=offline)


def load_model(
    model: str | ModelNames | ModelTypes,
    path: str | Path | None = None,
    *,
    quantization: str | None = None,
    sess_options: rt.SessionOptions | None = None,
    providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
    provider_options: Sequence[dict[Any, Any]] | None = None,
    cpu_preprocessing: bool | None = None,
    asr_config: OnnxSessionOptions | None = None,
    preprocessor_config: PreprocessorRuntimeConfig | None = None,
    resampler_config: OnnxSessionOptions | None = None,
) -> TextResultsAsrAdapter:
    """Load ASR model.

    Args:
        model: Model name or type (download from Hugging Face supported if full model name is provided):

                GigaAM v2 (`gigaam-v2-ctc` | `gigaam-v2-rnnt`)
                GigaAM v3 (`gigaam-v3-ctc` | `gigaam-v3-rnnt` |
                           `gigaam-v3-e2e-ctc` | `gigaam-v3-e2e-rnnt`)
                Kaldi Transducer (`kaldi-rnnt`)
                NeMo Conformer (`nemo-conformer-ctc` | `nemo-conformer-rnnt` | `nemo-conformer-tdt` |
                                `nemo-conformer-aed`)
                NeMo FastConformer Hybrid Large Ru P&C (`nemo-fastconformer-ru-ctc` |
                                                        `nemo-fastconformer-ru-rnnt`)
                NeMo Parakeet 0.6B En (`nemo-parakeet-ctc-0.6b` | `nemo-parakeet-rnnt-0.6b` |
                                       `nemo-parakeet-tdt-0.6b-v2`)
                NeMo Parakeet 0.6B Multilingual (`nemo-parakeet-tdt-0.6b-v3`)
                NeMo Canary (`nemo-canary-1b-v2`)
                T-One (`t-one-ctc` | `t-tech/t-one`)
                Vosk (`vosk` | `alphacep/vosk-model-ru` | `alphacep/vosk-model-small-ru`)
                Whisper Base exported with onnxruntime (`whisper-ort` | `whisper-base-ort`)
                Whisper from onnx-community (`whisper` | `onnx-community/whisper-large-v3-turbo` |
                                             `onnx-community/*whisper*`)
        path: Path to directory with model files.
        quantization: Model quantization (`None` | `int8` | ... ).
        sess_options: Default SessionOptions for onnxruntime.
        providers: Default providers for onnxruntime.
        provider_options: Default provider_options for onnxruntime.
        cpu_preprocessing: Deprecated and ignored, use `preprocessor_config` and `resampler_config` instead.
        asr_config: ASR ONNX config.
        preprocessor_config: Preprocessor ONNX and concurrency config.
        resampler_config: Resampler ONNX config.

    Returns:
        ASR model class.

    Raises:
        utils.ModelLoadingError: Model loading error (onnx-asr specific).

    """
    if cpu_preprocessing is not None:
        warnings.warn(
            "The cpu_preprocessing argument is deprecated and ignored (use preprocessor_config and resampler_config).",
            stacklevel=2,
        )

    resolver = create_asr_resolver(model, path)

    default_onnx_config: OnnxSessionOptions = {
        "sess_options": sess_options,
        "providers": providers or rt.get_available_providers(),
        "provider_options": provider_options,
    }

    if asr_config is None:
        asr_config = update_onnx_providers(
            default_onnx_config, excluded_providers=resolver.model_type._get_excluded_providers()
        )

    if preprocessor_config is None:
        preprocessor_config = {
            **update_onnx_providers(
                default_onnx_config,
                new_options={"TensorrtExecutionProvider": {"trt_fp16_enable": False, "trt_int8_enable": False}},
                excluded_providers=OnnxPreprocessor._get_excluded_providers(),
            ),
            "max_concurrent_workers": 1,
        }

    if resampler_config is None:
        resampler_config = update_onnx_providers(
            default_onnx_config, excluded_providers=Resampler._get_excluded_providers()
        )

    model_files = resolver.resolve_model(quantization=quantization)
    if config_path := model_files.get("config"):
        with config_path.open("rt", encoding="utf-8") as f:
            config: dict[str, object] = json.load(f)
    else:
        config = {}

    preprocessor = _create_preprocessor(resolver.model_type._get_preprocessor_name(config), preprocessor_config)
    resampler = Resampler(resolver.model_type._get_sample_rate(), resampler_config)
    asr = resolver.model_type(config, model_files, preprocessor, asr_config)
    return TextResultsAsrAdapter(asr, resampler)


def load_vad(
    model: VadNames = "silero",
    path: str | Path | None = None,
    *,
    quantization: str | None = None,
    sess_options: rt.SessionOptions | None = None,
    providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
    provider_options: Sequence[dict[Any, Any]] | None = None,
) -> Vad:
    """Load VAD model.

    Args:
        model: VAD model name (supports download from Hugging Face).
        path: Path to directory with model files.
        quantization: Model quantization (`None` | `int8` | ... ).
        sess_options: Optional SessionOptions for onnxruntime.
        providers: Optional providers for onnxruntime.
        provider_options: Optional provider_options for onnxruntime.

    Returns:
        VAD model class.

    Raises:
        utils.ModelLoadingError: Model loading error (onnx-asr specific).

    """
    resolver = create_vad_resolver(model, path)

    onnx_options = update_onnx_providers(
        {"providers": rt.get_available_providers()}, excluded_providers=resolver.model_type._get_excluded_providers()
    ) | {
        "sess_options": sess_options,
        "providers": providers,
        "provider_options": provider_options,
    }

    return resolver.model_type(resolver.resolve_model(quantization=quantization), onnx_options)
