from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
REGISTRY_PATH = APP_DIR / "model_registry.json"
DATA_DIR = Path("/var/lib/trash-bin-ai")
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
RESULT_TTL_SECONDS = 24 * 60 * 60
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
FORMAT_EXTENSION = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}

Image.MAX_IMAGE_PIXELS = 30_000_000

model_cache: dict[str, YOLO] = {}
model_lock = asyncio.Lock()


def read_registry() -> list[dict[str, Any]]:
    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Не удалось прочитать реестр моделей") from error
    if not isinstance(registry, list):
        raise RuntimeError("Некорректный формат реестра моделей")
    return registry


def model_entry(model_id: str) -> dict[str, Any]:
    for entry in read_registry():
        if entry.get("id") == model_id:
            return entry
    raise HTTPException(status_code=404, detail="Выбранная модель не найдена")


def model_path(entry: dict[str, Any]) -> Path:
    filename = entry.get("filename", "")
    candidate = (MODELS_DIR / filename).resolve()
    if candidate.parent != MODELS_DIR.resolve() or candidate.suffix != ".pt":
        raise HTTPException(status_code=500, detail="Некорректная запись модели")
    return candidate


def get_model(entry: dict[str, Any]) -> YOLO:
    model_id = entry["id"]
    path = model_path(entry)
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Файл выбранной модели временно недоступен")
    if model_id not in model_cache:
        model_cache[model_id] = YOLO(str(path))
    return model_cache[model_id]


def class_names(entry: dict[str, Any]) -> list[str]:
    model = get_model(entry)
    names = model.names
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def public_model(entry: dict[str, Any], *, load_names: bool = True) -> dict[str, Any]:
    path = model_path(entry)
    available = path.is_file()
    payload = {
        "id": entry["id"],
        "name": entry["name"],
        "filename": entry["filename"],
        "description": entry["description"],
        "warning": str(entry.get("warning", "")),
        "added_at": entry["added_at"],
        "status": "available" if available else "unavailable",
        "classes": [],
    }
    if available and load_names:
        try:
            payload["classes"] = class_names(entry)
        except Exception:
            payload["status"] = "unavailable"
    return payload


async def read_and_validate_image(upload: UploadFile) -> tuple[bytes, str, str]:
    if upload.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Поддерживаются только JPG, PNG и WEBP")
    content = bytearray()
    while chunk := await upload.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Размер изображения не должен превышать 15 МБ")
    if not content:
        raise HTTPException(status_code=422, detail="Файл изображения пуст")
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
        with Image.open(BytesIO(content)) as image:
            image.load()
            image_format = image.format
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=422, detail="Не удалось безопасно прочитать изображение")
    if image_format not in ALLOWED_FORMATS:
        raise HTTPException(status_code=415, detail="Формат файла не поддерживается")
    return bytes(content), image_format, FORMAT_EXTENSION[image_format]


def predict_sync(
    entry: dict[str, Any], input_path: Path, output_path: Path, confidence: float, iou: float
) -> dict[str, Any]:
    model = get_model(entry)
    started = time.perf_counter()
    result = model(str(input_path), conf=confidence, iou=iou, verbose=False)[0]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    names = result.names
    detections: list[dict[str, Any]] = []
    summary: Counter[str] = Counter()
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = str(names[class_id])
            x1, y1, x2, y2 = (round(float(value), 2) for value in box.xyxy[0].tolist())
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )
            summary[class_name] += 1
    rendered = Image.fromarray(result.plot()[..., ::-1])
    rendered.save(output_path, format="JPEG", quality=92, optimize=True)
    with Image.open(input_path) as source:
        width, height = source.size
    return {
        "width": width,
        "height": height,
        "inference_ms": elapsed_ms,
        "detections": detections,
        "summary": dict(sorted(summary.items())),
    }


def cleanup_expired_files() -> None:
    threshold = time.time() - RESULT_TTL_SECONDS
    for directory in (UPLOADS_DIR, RESULTS_DIR):
        for item in directory.glob("*"):
            try:
                if item.is_file() and item.stat().st_mtime < threshold:
                    item.unlink()
            except OSError:
                continue


async def cleanup_loop() -> None:
    while True:
        cleanup_expired_files()
        await asyncio.sleep(60 * 60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    for directory in (UPLOADS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    cleanup_expired_files()
    task = asyncio.create_task(cleanup_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Trash Bin AI API", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/api/health")
def health() -> dict[str, Any]:
    models = [public_model(entry, load_names=False) for entry in read_registry()]
    return {"status": "ok", "models_available": sum(item["status"] == "available" for item in models)}


@app.get("/api/models")
def models() -> dict[str, list[dict[str, Any]]]:
    return {"models": [public_model(entry) for entry in read_registry()]}


@app.post("/api/predict")
async def predict(
    model_id: str = Form(...),
    image: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
) -> dict[str, Any]:
    if not 0.01 <= confidence <= 0.99 or not 0.01 <= iou <= 0.99:
        raise HTTPException(status_code=422, detail="Confidence и IoU должны быть в диапазоне от 0.01 до 0.99")
    entry = model_entry(model_id)
    content, _, extension = await read_and_validate_image(image)
    result_id = uuid.uuid4().hex
    input_path = UPLOADS_DIR / f"{result_id}{extension}"
    output_path = RESULTS_DIR / f"{result_id}.jpg"
    input_path.write_bytes(content)
    try:
        async with model_lock:
            result = await run_in_threadpool(predict_sync, entry, input_path, output_path, confidence, iou)
    except HTTPException:
        input_path.unlink(missing_ok=True)
        raise
    except Exception as error:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Не удалось выполнить распознавание. Попробуйте ещё раз.") from error
    return {
        "result_id": result_id,
        "width": result["width"],
        "height": result["height"],
        "inference_ms": result["inference_ms"],
        "objects_count": len(result["detections"]),
        "detections": result["detections"],
        "summary": result["summary"],
        "original_image_url": f"/api/results/{result_id}/original",
        "processed_image_url": f"/api/results/{result_id}/image",
        "download_url": f"/api/results/{result_id}/download",
    }


def result_file(result_id: str, kind: str) -> Path:
    if not result_id.isalnum() or len(result_id) != 32:
        raise HTTPException(status_code=404, detail="Результат не найден")
    candidates = list(UPLOADS_DIR.glob(f"{result_id}.*")) if kind == "original" else [RESULTS_DIR / f"{result_id}.jpg"]
    file_path = candidates[0] if candidates else None
    if file_path is None or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Результат не найден или уже очищен")
    return file_path


@app.get("/api/results/{result_id}/original")
def original_image(result_id: str):
    return FileResponse(result_file(result_id, "original"))


@app.get("/api/results/{result_id}/image")
def processed_image(result_id: str):
    return FileResponse(result_file(result_id, "processed"), media_type="image/jpeg")


@app.get("/api/results/{result_id}/download")
def download_image(result_id: str):
    return FileResponse(
        result_file(result_id, "processed"), media_type="image/jpeg", filename=f"trash-bin-ai-{result_id}.jpg"
    )
