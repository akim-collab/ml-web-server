import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Настройки приложения из переменных окружения"""
    models_dir: str = "./saved_models"
    cpu_cores: int = 4
    max_loaded_models: int = 3
    
    class Config:
        env_file = ".env"

settings = Settings()