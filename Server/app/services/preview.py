import asyncio

import numpy as np
from PIL import Image


async def build_preview_png(volume: np.ndarray) -> bytes:
    """Create a human-friendly 2D preview via max-intensity projection."""

    def _build_sync() -> bytes:
        projection = np.max(volume, axis=0).astype(np.float32)

        min_val = float(np.min(projection))
        max_val = float(np.max(projection))
        if max_val > min_val:
            normalized = (projection - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(projection)

        image_array = (normalized * 255.0).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image_array, mode="L")

        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    return await asyncio.to_thread(_build_sync)
