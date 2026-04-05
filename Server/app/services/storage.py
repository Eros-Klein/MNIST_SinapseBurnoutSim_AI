import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np

from app.models import EvaluationResult, ImageEntry

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "images"


def _entry_dir(image_id: str) -> Path:
    return DATA_DIR / image_id


def _is_valid_image_id(image_id: str) -> bool:
    try:
        UUID(image_id)
        return True
    except ValueError:
        return False


async def store_image_result(
    volume: np.ndarray,
    evaluation: EvaluationResult,
) -> ImageEntry:
    image_id = str(uuid4())
    created_at = datetime.now(timezone.utc)
    entry_dir = _entry_dir(image_id)

    payload = {
        "id": image_id,
        "created_at": created_at.isoformat(),
        "shape": [int(x) for x in volume.shape],
        "evaluation": evaluation.model_dump(),
    }

    def _write_sync() -> None:
        entry_dir.mkdir(parents=True, exist_ok=False)
        np.save(entry_dir / "input.npy", volume)
        with (entry_dir / "meta.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=True)

    await asyncio.to_thread(_write_sync)

    return ImageEntry(
        id=image_id,
        created_at=created_at,
        shape=(int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2])),
        evaluation=evaluation,
        npy_url=f"/images/{image_id}/npy",
    )


async def list_entries(offset: int, limit: int) -> list[ImageEntry]:
    def _read_sync() -> list[ImageEntry]:
        if not DATA_DIR.exists():
            return []

        rows: list[dict] = []
        for child in DATA_DIR.iterdir():
            if not child.is_dir():
                continue

            meta_file = child / "meta.json"
            if not meta_file.exists():
                continue

            with meta_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            rows.append(data)

        rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        selected = rows[offset : offset + limit]

        result: list[ImageEntry] = []
        for row in selected:
            shape = row.get("shape", [28, 28, 28])
            result.append(
                ImageEntry(
                    id=row["id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    shape=(int(shape[0]), int(shape[1]), int(shape[2])),
                    evaluation=EvaluationResult(**row["evaluation"]),
                    npy_url=f"/images/{row['id']}/npy",
                )
            )

        return result

    return await asyncio.to_thread(_read_sync)


async def npy_path_for_id(image_id: str) -> Path | None:
    if not _is_valid_image_id(image_id):
        return None

    file_path = _entry_dir(image_id) / "input.npy"

    def _resolve_sync() -> Path | None:
        if file_path.exists() and file_path.is_file():
            return file_path
        return None

    return await asyncio.to_thread(_resolve_sync)
