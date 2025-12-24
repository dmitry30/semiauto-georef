from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from pathlib import Path
from .georeferencer import Georeferencer
from .tile_fetcher import TileFetcher
from PIL import Image
import json
import uuid

app = FastAPI(title="Simple Georeferencer")

# Создаем временную директорию
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# Монтируем статику
app.mount("/static", StaticFiles(directory="static"), name="static")

# Инициализируем компоненты
georef = Georeferencer(temp_dir="temp")
tile_fetcher = TileFetcher()

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    zoom: int = Form(18)
):
    """Загрузить изображение и привязать его"""
    
    # Создаем уникальный ID для этой сессии
    session_id = str(uuid.uuid4())[:8]
    
    # Сохраняем загруженный файл
    upload_path = TEMP_DIR / f"{session_id}_{file.filename}"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Открываем изображение
        source_image = Image.open(upload_path)
        print(f"Loaded image: {file.filename}, size: {source_image.size}")
        
        # Получаем эталонные тайлы
        print(f"Fetching tiles for lat={lat}, lon={lon}, zoom={zoom}")
        reference_image, bounds = tile_fetcher.get_area_tiles(
            lat, lon, 
            source_image.width, 
            source_image.height, 
            zoom
        )
        
        print(f"Reference image size: {reference_image.size}")
        print(f"Reference bounds: {bounds}")
        
        # Сохраняем референс для отладки (опционально)
        reference_path = TEMP_DIR / f"{session_id}_reference.png"
        reference_image.save(reference_path)
        print(f"Saved reference image to: {reference_path}")
        
        # Привязываем изображение
        print("Starting georeferencing...")
        result_path, message = georef.apply_georeference(
            source_image, reference_image, bounds
        )
        
        if result_path is None:
            raise HTTPException(status_code=400, detail=message)
        
        print(f"Georeferencing completed: {result_path}")
        
        # Получаем имя файла из пути
        result_filename = Path(result_path).name
        
        # Сохраняем метаданные
        metadata = {
            "original_filename": file.filename,
            "anchor_point": {"lat": lat, "lon": lon},
            "zoom": zoom,
            "message": message,
            "result_file": result_filename,
            "session_id": session_id,
            "bounds": bounds
        }
        
        metadata_path = TEMP_DIR / f"{session_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "status": "success",
            "message": message,
            "download_url": f"/download/{result_filename}",
            "metadata_url": f"/download/{metadata_path.name}",
            "reference_url": f"/download/{reference_path.name}"  # для отладки
        }
        
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\n{traceback.format_exc()}"
        print(f"Error: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Удаляем загруженный файл
        if upload_path.exists():
            upload_path.unlink()

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Скачать результат"""
    file_path = TEMP_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {filename}")
    
    # Определяем content-type
    if filename.endswith(".tif") or filename.endswith(".tiff"):
        media_type = "image/tiff"
    elif filename.endswith(".png"):
        media_type = "image/png"
    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
        media_type = "image/jpeg"
    elif filename.endswith(".json"):
        media_type = "application/json"
    else:
        media_type = "application/octet-stream"
    
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename
    )

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/cleanup")
async def cleanup():
    """Очистить временные файлы (для отладки)"""
    import time
    import glob
    
    deleted = 0
    for file_path in TEMP_DIR.glob("*"):
        # Удаляем файлы старше 1 часа
        if file_path.is_file() and time.time() - file_path.stat().st_mtime > 3600:
            file_path.unlink()
            deleted += 1
    
    return {"status": "ok", "deleted_files": deleted}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)