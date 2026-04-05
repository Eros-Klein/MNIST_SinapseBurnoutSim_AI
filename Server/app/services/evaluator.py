import asyncio
import importlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.models import EvaluationResult

SERVER_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = SERVER_DIR.parent
DEFAULT_ARTIFACT_DIRS = [
    SERVER_DIR / "model_artifacts",
    WORKSPACE_DIR / "model_artifacts",
    WORKSPACE_DIR / "Notebook" / "model_artifacts",
]
METADATA_FILENAME = "synapsemnist_3d_cnn_metadata.json"
MODEL_FILENAME = "synapsemnist_3d_cnn.keras"

_MODEL_CACHE: Any | None = None
_METADATA_CACHE: dict[str, Any] | None = None
_MODEL_PATH_CACHE: Path | None = None
_LOAD_LOCK = threading.Lock()


def _find_metadata_path() -> Path:
    for artifact_dir in DEFAULT_ARTIFACT_DIRS:
        candidate = artifact_dir / METADATA_FILENAME
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path) for path in DEFAULT_ARTIFACT_DIRS)
    raise FileNotFoundError(
        f"Could not find {METADATA_FILENAME}. Searched:\n{searched}"
    )


def _resolve_model_path(metadata_path: Path, metadata: dict[str, Any]) -> Path:
    configured_model_path = str(metadata.get("model_path", "")).strip()
    model_path = Path(configured_model_path) if configured_model_path else Path(MODEL_FILENAME)

    candidates: list[Path] = []
    if model_path.is_absolute():
        candidates.append(model_path)
    else:
        candidates.extend(
            [
                metadata_path.parent / model_path,
                WORKSPACE_DIR / model_path,
                SERVER_DIR / model_path,
                metadata_path.parent / MODEL_FILENAME,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find model file for metadata {metadata_path}. Searched:\n{searched}"
    )


def _get_model_bundle() -> tuple[Any, dict[str, Any], Path]:
    global _MODEL_CACHE, _METADATA_CACHE, _MODEL_PATH_CACHE

    if _MODEL_CACHE is not None and _METADATA_CACHE is not None and _MODEL_PATH_CACHE is not None:
        return _MODEL_CACHE, _METADATA_CACHE, _MODEL_PATH_CACHE

    with _LOAD_LOCK:
        if _MODEL_CACHE is not None and _METADATA_CACHE is not None and _MODEL_PATH_CACHE is not None:
            return _MODEL_CACHE, _METADATA_CACHE, _MODEL_PATH_CACHE

        # Imported lazily so server can start even if TensorFlow is not installed yet.
        keras_models = importlib.import_module("tensorflow.keras.models")
        load_model = getattr(keras_models, "load_model")

        metadata_path = _find_metadata_path()
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        model_path = _resolve_model_path(metadata_path=metadata_path, metadata=metadata)
        model = load_model(model_path)

        _MODEL_CACHE = model
        _METADATA_CACHE = metadata
        _MODEL_PATH_CACHE = model_path

    assert _MODEL_CACHE is not None
    assert _METADATA_CACHE is not None
    assert _MODEL_PATH_CACHE is not None
    return _MODEL_CACHE, _METADATA_CACHE, _MODEL_PATH_CACHE


def _run_inference(volume: np.ndarray) -> EvaluationResult:
    model, metadata, model_path = _get_model_bundle()

    expected_shape = tuple(metadata.get("expected_volume_shape", [28, 28, 28]))
    normalized = np.asarray(volume, dtype=np.float32)
    if normalized.shape != expected_shape:
        raise ValueError(f"Expected volume shape {expected_shape}, got {normalized.shape}")

    normalization = metadata.get("normalization", {})
    method = str(normalization.get("method", "divide")).lower()
    value = float(normalization.get("value", 255.0))
    if method == "divide" and value > 0 and float(np.max(normalized)) > 1.0:
        normalized = normalized / value

    batch = normalized[np.newaxis, ..., np.newaxis]
    probability = float(model.predict(batch, verbose=0)[0][0])

    threshold = float(metadata.get("decision_threshold", 0.5))
    predicted_index = 1 if probability >= threshold else 0

    class_names = metadata.get("class_names", {})
    predicted_label = class_names.get(str(predicted_index), f"class_{predicted_index}")

    return EvaluationResult(
        label=predicted_label,
        score=probability,
        details={
            "predicted_class_index": predicted_index,
            "probability_has_synapse": probability,
            "decision_threshold": threshold,
            "model_path": str(model_path),
        },
    )


async def evaluate_volume(volume: np.ndarray) -> EvaluationResult:
    return await asyncio.to_thread(_run_inference, volume)
