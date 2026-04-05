import asyncio
import io
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.models import ImageEntry, ImageListResponse
from app.services.evaluator import evaluate_volume
from app.services.storage import (
    DATA_DIR,
    list_entries,
    npy_path_for_id,
    store_image_result,
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(DATA_DIR.mkdir, parents=True, exist_ok=True)
    yield


app = FastAPI(title="SynapseMNIST API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def _load_volume_from_npy_upload(image_file: UploadFile) -> np.ndarray:
    content = await image_file.read()

    def _decode_sync() -> np.ndarray:
        with io.BytesIO(content) as buffer:
            arr = np.load(buffer, allow_pickle=False)
        return np.asarray(arr)

    array = await asyncio.to_thread(_decode_sync)

    if array.ndim != 3 or array.shape != (28, 28, 28):
        raise HTTPException(status_code=400, detail="Input image must have shape (28, 28, 28).")

    return array.astype(np.float32)


@app.post("/images", response_model=ImageEntry, status_code=201)
async def upload_image(image: UploadFile = File(..., description="Single 28x28x28 .npy volume")) -> ImageEntry:
    if image.content_type not in {"application/octet-stream", "application/x-npy", "binary/octet-stream"}:
        # Some clients omit correct MIME types. We still try decoding if extension looks valid.
        if not image.filename or not image.filename.endswith(".npy"):
            raise HTTPException(status_code=400, detail="Upload must be a .npy file.")

    try:
        volume = await _load_volume_from_npy_upload(image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Could not decode .npy upload.") from exc

    evaluation = await evaluate_volume(volume)
    entry = await store_image_result(volume=volume, evaluation=evaluation)

    return entry


@app.get("/images", response_model=ImageListResponse)
async def get_images(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
) -> ImageListResponse:
    items = await list_entries(offset=offset, limit=limit)
    return ImageListResponse(offset=offset, limit=limit, count=len(items), items=items)


@app.get("/images/{image_id}/npy")
async def get_image_npy(image_id: str) -> FileResponse:
    npy_path = await npy_path_for_id(image_id)
    if npy_path is None:
        raise HTTPException(status_code=404, detail="Image data not found.")

    return FileResponse(path=npy_path, media_type="application/octet-stream")
