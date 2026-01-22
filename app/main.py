from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import numpy as np
from typing import List, Dict, Any
import time
import uvicorn

from app.config import settings
from app.schemas import (
    FitRequest, PredictRequest, ModelConfig,
    FitResponse, PredictResponse, StatusResponse
)
from app.model_manager import ModelManager
from app.process_manager import ProcessManager

app = FastAPI(
    title="ML Model Training Server",
    description="Веб-сервер для обучения и использования ML моделей",
    version="1.0.0"
)

model_manager = ModelManager(settings.models_dir)
process_manager = ProcessManager(
    max_processes=settings.cpu_cores - 1,  
    models_dir=settings.models_dir
)

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "ML Model Training Server",
        "status": "running",
        "config": {
            "models_dir": settings.models_dir,
            "cpu_cores": settings.cpu_cores,
            "max_loaded_models": settings.max_loaded_models
        }
    }

@app.get("/status")
async def get_status():
    """Получение статуса сервера"""
    return {
        "active_training_processes": process_manager.get_active_processes_count(),
        "loaded_models": model_manager.get_loaded_models_count(),
        "max_loaded_models": settings.max_loaded_models,
        "can_start_training": process_manager.can_start_training()
    }

@app.post("/fit", response_model=FitResponse)
async def fit_model(request: FitRequest):
    """Обучение модели"""
    if not process_manager.can_start_training():
        raise HTTPException(
            status_code=429,
            detail="Нет свободных ядер для обучения. Максимум процессов: "
                   f"{settings.cpu_cores - 1}"
        )
    
    if model_manager.model_exists(request.model_name):
        raise HTTPException(
            status_code=400,
            detail=f"Модель с именем '{request.model_name}' уже существует"
        )
    
    if request.model_type not in ['logreg', 'rf', 'svm']:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый тип модели. Доступные: logreg, rf, svm"
        )
    
    try:
        X = np.array(request.features)
        y = np.array(request.labels)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка преобразования данных: {str(e)}"
        )
    
    success = process_manager.start_training(
        request.model_name,
        request.model_type,
        X, y,
        request.params
    )
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Не удалось запустить процесс обучения"
        )
    
    return FitResponse(
        status="started",
        model_name=request.model_name,
        message=f"Обучение модели '{request.model_name}' запущено"
    )

@app.get("/fit/{model_name}/status")
async def get_fit_status(model_name: str):
    """Проверка статуса обучения модели"""
    status = process_manager.check_training_status(model_name)
    
    if status is None:
        if model_manager.model_exists(model_name):
            return {
                "status": "completed",
                "model_name": model_name,
                "message": "Модель уже обучена"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Обучение модели '{model_name}' не найдено"
            )
    
    if status['status'] == 'error':
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обучении: {status.get('error', 'Unknown error')}"
        )
    
    return status

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Предсказание с помощью обученной модели"""
    if request.model_name not in model_manager.loaded_models:
        if not model_manager.load_model(request.model_name):
            raise HTTPException(
                status_code=404,
                detail=f"Модель '{request.model_name}' не найдена"
            )
    
    if model_manager.get_loaded_models_count() > settings.max_loaded_models:
        raise HTTPException(
            status_code=429,
            detail=f"Превышен лимит загруженных моделей: {settings.max_loaded_models}"
        )
    
    try:
        X = np.array(request.features)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка преобразования данных: {str(e)}"
        )
    
    try:
        predictions, probabilities = model_manager.predict(request.model_name, X)
        
        prob_list = None
        if probabilities is not None:
            prob_list = probabilities.tolist()
        
        return PredictResponse(
            status="success",
            predictions=predictions.tolist(),
            probabilities=prob_list
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при предсказании: {str(e)}"
        )

@app.post("/load", response_model=StatusResponse)
async def load_model(config: ModelConfig):
    """Загрузка модели в память"""
    if not model_manager.model_exists(config.model_name):
        raise HTTPException(
            status_code=404,
            detail=f"Модель '{config.model_name}' не найдена"
        )
    
    if model_manager.get_loaded_models_count() >= settings.max_loaded_models:
        raise HTTPException(
            status_code=429,
            detail=f"Превышен лимит загруженных моделей: {settings.max_loaded_models}"
        )
    
    success = model_manager.load_model(config.model_name)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось загрузить модель '{config.model_name}'"
        )
    
    return StatusResponse(
        status="success",
        message=f"Модель '{config.model_name}' загружена в память"
    )

@app.post("/unload", response_model=StatusResponse)
async def unload_model(config: ModelConfig):
    """Выгрузка модели из памяти"""
    success = model_manager.unload_model(config.model_name)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Модель '{config.model_name}' не загружена в память"
        )
    
    return StatusResponse(
        status="success",
        message=f"Модель '{config.model_name}' выгружена из памяти"
    )

@app.delete("/remove", response_model=StatusResponse)
async def remove_model(config: ModelConfig):
    """Удаление модели с диска"""
    model_manager.unload_model(config.model_name)
    
    success = model_manager.delete_model(config.model_name)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Модель '{config.model_name}' не найдена на диске"
        )
    
    return StatusResponse(
        status="success",
        message=f"Модель '{config.model_name}' удалена с диска"
    )

@app.delete("/remove_all", response_model=StatusResponse)
async def remove_all_models():
    """Удаление всех моделей"""
    count = model_manager.delete_all_models()
    
    return StatusResponse(
        status="success",
        message=f"Удалено {count} моделей",
        details={"deleted_count": count}
    )

@app.get("/models")
async def list_models():
    """Список всех моделей"""
    import os
    models = []
    
    for filename in os.listdir(settings.models_dir):
        if filename.endswith('.pkl'):
            model_name = filename[:-4]  
            models.append({
                "name": model_name,
                "loaded": model_name in model_manager.loaded_models
            })
    
    return {
        "models": models,
        "count": len(models)
    }

@app.on_event("startup")
async def startup_event():
    """Действия при запуске сервера"""
    import os
    os.makedirs(settings.models_dir, exist_ok=True)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )