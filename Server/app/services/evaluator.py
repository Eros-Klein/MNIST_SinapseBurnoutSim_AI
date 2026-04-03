import asyncio

import numpy as np

from app.models import EvaluationResult


async def evaluate_volume(volume: np.ndarray) -> EvaluationResult:
    """Placeholder evaluator separated from API and persistence logic."""

    def _evaluate_sync() -> EvaluationResult:
        # Deterministic placeholder score based on average intensity.
        mean_val = float(np.mean(volume))
        scale = 255.0 if mean_val > 1.0 else 1.0
        score = max(0.0, min(1.0, mean_val / scale))
        label = "placeholder-positive" if score >= 0.5 else "placeholder-negative"

        return EvaluationResult(
            label=label,
            score=score,
            details={
                "note": "Placeholder evaluation. Replace this service with real model inference.",
                "mean_intensity": mean_val,
            },
        )

    return await asyncio.to_thread(_evaluate_sync)
