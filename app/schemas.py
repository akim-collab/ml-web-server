from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np

class FitRequest(BaseModel):
    """Схема для запроса обучения модели"""
    model_name: str = Field(..., description="Название модели")
    model_type: str = Field(..., description="Тип модели (logreg, rf, svm)")
    features: List[List[float]] = Field(..., description="Признаки")
    labels: List[int] = Field(..., description="Метки")
    params: Optional[Dict[str, Any]] = Field(default={}, description="Параметры модели")

class PredictRequest(BaseModel):
    """Схема для запроса предсказания"""
    model_name: str = Field(..., description="Название модели")
    features: List[List[float]] = Field(..., description="Признаки для предсказания")

class ModelConfig(BaseModel):
    """Схема для операций с моделями"""
    model_name: str = Field(..., description="Название модели")

class FitResponse(BaseModel):
    """Ответ на обучение модели"""
    status: str
    model_name: str
    message: str
    training_time: Optional[float] = None

class PredictResponse(BaseModel):
    """Ответ на предсказание"""
    status: str
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None

class StatusResponse(BaseModel):
    """Общий ответ"""
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None