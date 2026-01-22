# app/process_manager.py - Управление процессами обучения с ограничением по CPU
import multiprocessing as mp
from multiprocessing import Process, Queue
import time
from typing import Dict, Any, Optional
import numpy as np

class ProcessManager:
    """Менеджер для управления процессами обучения"""
    
    def __init__(self, max_processes: int, models_dir: str):
        from app.model_manager import ModelManager
        
        self.max_processes = max_processes
        self.model_manager = ModelManager(models_dir)
        self.active_processes: Dict[str, Process] = {}
        self.process_queues: Dict[str, Queue] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def _train_worker(self, queue: Queue, model_name: str, model_type: str,
                     X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> None:
        """Рабочая функция для процесса обучения"""
        try:
            training_time = self.model_manager.train_model(
                model_name, model_type, X, y, params
            )
            queue.put({
                'status': 'success',
                'model_name': model_name,
                'training_time': training_time
            })
        except Exception as e:
            queue.put({
                'status': 'error',
                'model_name': model_name,
                'error': str(e)
            })
    
    def start_training(self, model_name: str, model_type: str,
                      X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> bool:
        """Запуск процесса обучения"""
        # Проверка доступных процессов
        if len(self.active_processes) >= self.max_processes:
            return False
        
        # Проверка существования модели
        if self.model_manager.model_exists(model_name):
            return False
        
        # Создаем очередь для обмена данными
        queue = Queue()
        
        # Создаем и запускаем процесс
        process = Process(
            target=self._train_worker,
            args=(queue, model_name, model_type, X, y, params)
        )
        
        process.start()
        self.active_processes[model_name] = process
        self.process_queues[model_name] = queue
        
        return True
    
    def check_training_status(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Проверка статуса обучения"""
        if model_name not in self.active_processes:
            return None
        
        process = self.active_processes[model_name]
        queue = self.process_queues[model_name]
        
        # Проверяем, завершен ли процесс
        if not process.is_alive():
            # Получаем результат
            if not queue.empty():
                result = queue.get()
                self.results[model_name] = result
            
            # Очищаем
            process.join()
            del self.active_processes[model_name]
            del self.process_queues[model_name]
            
            if model_name in self.results:
                return self.results[model_name]
        
        return {'status': 'training', 'model_name': model_name}
    
    def get_active_processes_count(self) -> int:
        """Количество активных процессов"""
        return len(self.active_processes)
    
    def can_start_training(self) -> bool:
        """Можно ли запустить новый процесс"""
        return len(self.active_processes) < self.max_processes
    
    def cleanup_finished_processes(self):
        """Очистка завершенных процессов"""
        finished = []
        for model_name, process in self.active_processes.items():
            if not process.is_alive():
                finished.append(model_name)
        
        for model_name in finished:
            process = self.active_processes[model_name]
            process.join()
            del self.active_processes[model_name]
            del self.process_queues[model_name]