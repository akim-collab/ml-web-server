import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Dict, Any, Tuple
import time

class ModelManager:
    """Менеджер для работы с ML моделями"""
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.model_types = {
            'logreg': LogisticRegression,
            'rf': RandomForestClassifier,
            'svm': SVC
        }
        
        self.loaded_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        self.default_params = {
            'logreg': {'max_iter': 1000, 'random_state': 42},
            'rf': {'n_estimators': 100, 'random_state': 42},
            'svm': {'kernel': 'rbf', 'random_state': 42, 'probability': True}
        }
    
    def create_model(self, model_type: str, params: Dict[str, Any]) -> Any:
        """Создание модели по типу и параметрам"""
        if model_type not in self.model_types:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
        
        default = self.default_params.get(model_type, {})
        merged_params = {**default, **params}
        
        return self.model_types[model_type](**merged_params)
    
    def train_model(self, model_name: str, model_type: str, 
                   X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> float:
        """Обучение модели и сохранение на диск"""
        start_time = time.time()
        
        model = self.create_model(model_type, params)
        
        if model_type == 'svm':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[model_name] = scaler
        else:
            X_scaled = X
        
        model.fit(X_scaled, y)
        
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'model_type': model_type,
                'params': params
            }, f)
        
        training_time = time.time() - start_time
        return training_time
    
    def load_model(self, model_name: str) -> bool:
        """Загрузка модели в память"""
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            return False
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.loaded_models[model_name] = model_data
        return True
    
    def unload_model(self, model_name: str) -> bool:
        """Выгрузка модели из памяти"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if model_name in self.scalers:
                del self.scalers[model_name]
            return True
        return False
    
    def predict(self, model_name: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Предсказание с помощью загруженной модели"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Модель {model_name} не загружена")
        
        model_data = self.loaded_models[model_name]
        model = model_data['model']
        
        if model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X)
        else:
            X_scaled = X
        
        predictions = model.predict(X_scaled)
        
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def delete_model(self, model_name: str) -> bool:
        """Удаление модели с диска"""
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        
        self.unload_model(model_name)
        
        if os.path.exists(model_path):
            os.remove(model_path)
            return True
        return False
    
    def delete_all_models(self) -> int:
        """Удаление всех моделей"""
        count = 0
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pkl'):
                os.remove(os.path.join(self.models_dir, filename))
                count += 1
        
        # Очищаем кэш
        self.loaded_models.clear()
        self.scalers.clear()
        
        return count
    
    def get_loaded_models_count(self) -> int:
        """Количество загруженных моделей"""
        return len(self.loaded_models)
    
    def model_exists(self, model_name: str) -> bool:
        """Проверка существования модели"""
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        return os.path.exists(model_path)