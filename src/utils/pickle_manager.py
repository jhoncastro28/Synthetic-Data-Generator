"""
Módulo para manejo de pickle en el sistema de generación de datos sintéticos.
Permite guardar y cargar modelos, preprocesadores, configuraciones y resultados.
"""

import pickle
import os
import json
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


class PickleManager:
    """
    Gestor de pickle para el sistema de GAN.
    Maneja el guardado y carga de todos los componentes del sistema.
    """
    
    def __init__(self, base_dir: str = "results"):
        """
        Inicializa el gestor de pickle.
        
        Args:
            base_dir: Directorio base para guardar archivos
        """
        self.base_dir = Path(base_dir)
        self._create_directories()
    
    def _create_directories(self):
        """Crea los directorios necesarios."""
        directories = [
            "models",
            "preprocessors", 
            "configs",
            "training_state",
            "results",
            "plots",
            "metrics"
        ]
        
        for dir_name in directories:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def save_model(self, 
                   model: Any, 
                   filename: str, 
                   epoch: Optional[int] = None,
                   metadata: Optional[Dict] = None) -> str:
        """
        Guarda un modelo en pickle.
        
        Args:
            model: Modelo a guardar
            filename: Nombre del archivo
            epoch: Época del entrenamiento (opcional)
            metadata: Metadatos adicionales
            
        Returns:
            Ruta del archivo guardado
        """
        if epoch is not None:
            filename = f"{filename}_epoch_{epoch}"
        
        filepath = self.base_dir / "models" / f"{filename}.pkl"
        
        # Preparar datos para guardar
        save_data = {
            'model': model,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        if epoch is not None:
            save_data['epoch'] = epoch
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Modelo guardado en: {filepath}")
        return str(filepath)
    
    def load_model(self, filename: str, epoch: Optional[int] = None) -> Tuple[Any, Dict]:
        """
        Carga un modelo desde pickle.
        
        Args:
            filename: Nombre del archivo
            epoch: Época específica (opcional)
            
        Returns:
            Tupla con (modelo, metadatos)
        """
        if epoch is not None:
            filename = f"{filename}_epoch_{epoch}"
        
        filepath = self.base_dir / "models" / f"{filename}.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        print(f"Modelo cargado desde: {filepath}")
        return save_data['model'], save_data.get('metadata', {})
    
    def save_preprocessor(self, 
                         preprocessor: Any, 
                         name: str,
                         metadata: Optional[Dict] = None) -> str:
        """
        Guarda un preprocesador (scaler, encoder, etc.).
        
        Args:
            preprocessor: Preprocesador a guardar
            name: Nombre del preprocesador
            metadata: Metadatos adicionales
            
        Returns:
            Ruta del archivo guardado
        """
        filepath = self.base_dir / "preprocessors" / f"{name}.pkl"
        
        save_data = {
            'preprocessor': preprocessor,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Preprocesador guardado en: {filepath}")
        return str(filepath)
    
    def load_preprocessor(self, name: str) -> Tuple[Any, Dict]:
        """
        Carga un preprocesador desde pickle.
        
        Args:
            name: Nombre del preprocesador
            
        Returns:
            Tupla con (preprocesador, metadatos)
        """
        filepath = self.base_dir / "preprocessors" / f"{name}.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocesador no encontrado: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        print(f"Preprocesador cargado desde: {filepath}")
        return save_data['preprocessor'], save_data.get('metadata', {})
    
    def save_training_state(self, 
                           trainer_state: Dict,
                           epoch: int,
                           config: Dict) -> str:
        """
        Guarda el estado completo del entrenamiento.
        
        Args:
            trainer_state: Estado del entrenador
            epoch: Época actual
            config: Configuración del entrenamiento
            
        Returns:
            Ruta del archivo guardado
        """
        filepath = self.base_dir / "training_state" / f"training_state_epoch_{epoch}.pkl"
        
        save_data = {
            'trainer_state': trainer_state,
            'epoch': epoch,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Estado de entrenamiento guardado en: {filepath}")
        return str(filepath)
    
    def load_training_state(self, epoch: Optional[int] = None) -> Tuple[Dict, int, Dict]:
        """
        Carga el estado del entrenamiento.
        
        Args:
            epoch: Época específica (opcional, carga la más reciente)
            
        Returns:
            Tupla con (estado, época, configuración)
        """
        if epoch is not None:
            filepath = self.base_dir / "training_state" / f"training_state_epoch_{epoch}.pkl"
        else:
            # Buscar el estado más reciente
            state_dir = self.base_dir / "training_state"
            state_files = list(state_dir.glob("training_state_epoch_*.pkl"))
            
            if not state_files:
                raise FileNotFoundError("No se encontraron estados de entrenamiento")
            
            # Ordenar por época y tomar el más reciente
            epochs = []
            for file in state_files:
                try:
                    epoch_num = int(file.stem.split('_')[-1])
                    epochs.append((epoch_num, file))
                except:
                    continue
            
            epochs.sort(key=lambda x: x[0], reverse=True)
            filepath = epochs[0][1]
        
        if not filepath.exists():
            raise FileNotFoundError(f"Estado de entrenamiento no encontrado: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        print(f"Estado de entrenamiento cargado desde: {filepath}")
        return (save_data['trainer_state'], 
                save_data['epoch'], 
                save_data['config'])
    
    def save_results(self, 
                    results: Dict,
                    experiment_name: str,
                    metadata: Optional[Dict] = None) -> str:
        """
        Guarda resultados de evaluación y experimentos.
        
        Args:
            results: Resultados a guardar
            experiment_name: Nombre del experimento
            metadata: Metadatos adicionales
            
        Returns:
            Ruta del archivo guardado
        """
        filepath = self.base_dir / "results" / f"{experiment_name}_results.pkl"
        
        save_data = {
            'results': results,
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Resultados guardados en: {filepath}")
        return str(filepath)
    
    def load_results(self, experiment_name: str) -> Tuple[Dict, Dict]:
        """
        Carga resultados de un experimento.
        
        Args:
            experiment_name: Nombre del experimento
            
        Returns:
            Tupla con (resultados, metadatos)
        """
        filepath = self.base_dir / "results" / f"{experiment_name}_results.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Resultados no encontrados: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        print(f"Resultados cargados desde: {filepath}")
        return save_data['results'], save_data.get('metadata', {})
    
    def save_metrics(self, 
                    metrics: Dict,
                    experiment_name: str,
                    epoch: Optional[int] = None) -> str:
        """
        Guarda métricas de evaluación.
        
        Args:
            metrics: Métricas a guardar
            experiment_name: Nombre del experimento
            epoch: Época específica (opcional)
            
        Returns:
            Ruta del archivo guardado
        """
        filename = f"{experiment_name}_metrics"
        if epoch is not None:
            filename += f"_epoch_{epoch}"
        
        filepath = self.base_dir / "metrics" / f"{filename}.pkl"
        
        save_data = {
            'metrics': metrics,
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Métricas guardadas en: {filepath}")
        return str(filepath)
    
    def load_metrics(self, experiment_name: str, epoch: Optional[int] = None) -> Dict:
        """
        Carga métricas de un experimento.
        
        Args:
            experiment_name: Nombre del experimento
            epoch: Época específica (opcional)
            
        Returns:
            Métricas cargadas
        """
        filename = f"{experiment_name}_metrics"
        if epoch is not None:
            filename += f"_epoch_{epoch}"
        
        filepath = self.base_dir / "metrics" / f"{filename}.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Métricas no encontradas: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        print(f"Métricas cargadas desde: {filepath}")
        return save_data['metrics']
    
    def save_synthetic_data(self, 
                           data: np.ndarray,
                           feature_names: list,
                           experiment_name: str,
                           epoch: Optional[int] = None) -> str:
        """
        Guarda datos sintéticos generados.
        
        Args:
            data: Datos sintéticos
            feature_names: Nombres de las características
            experiment_name: Nombre del experimento
            epoch: Época específica (opcional)
            
        Returns:
            Ruta del archivo guardado
        """
        filename = f"{experiment_name}_synthetic_data"
        if epoch is not None:
            filename += f"_epoch_{epoch}"
        
        filepath = self.base_dir / "results" / f"{filename}.pkl"
        
        save_data = {
            'data': data,
            'feature_names': feature_names,
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'shape': data.shape
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        # También guardar como CSV para facilidad de uso
        csv_path = self.base_dir / "results" / f"{filename}.csv"
        df = pd.DataFrame(data, columns=feature_names)
        df.to_csv(csv_path, index=False)
        
        print(f"Datos sintéticos guardados en: {filepath} y {csv_path}")
        return str(filepath)
    
    def load_synthetic_data(self, experiment_name: str, epoch: Optional[int] = None) -> Tuple[np.ndarray, list]:
        """
        Carga datos sintéticos generados.
        
        Args:
            experiment_name: Nombre del experimento
            epoch: Época específica (opcional)
            
        Returns:
            Tupla con (datos, nombres_de_características)
        """
        filename = f"{experiment_name}_synthetic_data"
        if epoch is not None:
            filename += f"_epoch_{epoch}"
        
        filepath = self.base_dir / "results" / f"{filename}.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Datos sintéticos no encontrados: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.dump(f)
        
        print(f"Datos sintéticos cargados desde: {filepath}")
        return save_data['data'], save_data['feature_names']
    
    def list_saved_models(self) -> list:
        """Lista todos los modelos guardados."""
        models_dir = self.base_dir / "models"
        return [f.stem for f in models_dir.glob("*.pkl")]
    
    def list_saved_experiments(self) -> list:
        """Lista todos los experimentos guardados."""
        results_dir = self.base_dir / "results"
        experiments = set()
        
        for file in results_dir.glob("*_results.pkl"):
            name = file.stem.replace("_results", "")
            experiments.add(name)
        
        return list(experiments)
    
    def cleanup_old_files(self, keep_last_n: int = 5):
        """
        Limpia archivos antiguos, manteniendo solo los últimos N.
        
        Args:
            keep_last_n: Número de archivos a mantener
        """
        # Limpiar modelos por época
        models_dir = self.base_dir / "models"
        model_files = list(models_dir.glob("*_epoch_*.pkl"))
        
        # Agrupar por tipo de modelo
        model_groups = {}
        for file in model_files:
            base_name = file.stem.split('_epoch_')[0]
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append(file)
        
        # Mantener solo los últimos N de cada grupo
        for base_name, files in model_groups.items():
            files.sort(key=lambda x: int(x.stem.split('_epoch_')[1]))
            files_to_delete = files[:-keep_last_n]
            
            for file in files_to_delete:
                file.unlink()
                print(f"Archivo eliminado: {file}")
    
    def get_experiment_summary(self, experiment_name: str) -> Dict:
        """
        Obtiene un resumen de un experimento.
        
        Args:
            experiment_name: Nombre del experimento
            
        Returns:
            Resumen del experimento
        """
        summary = {
            'experiment_name': experiment_name,
            'files': {},
            'latest_epoch': None
        }
        
        # Buscar archivos relacionados
        for subdir in ['models', 'results', 'metrics', 'training_state']:
            dir_path = self.base_dir / subdir
            files = list(dir_path.glob(f"{experiment_name}*"))
            summary['files'][subdir] = [f.name for f in files]
        
        # Encontrar la época más reciente
        state_files = list((self.base_dir / "training_state").glob(f"{experiment_name}*"))
        if state_files:
            epochs = []
            for file in state_files:
                try:
                    epoch = int(file.stem.split('_epoch_')[1])
                    epochs.append(epoch)
                except:
                    continue
            if epochs:
                summary['latest_epoch'] = max(epochs)
        
        return summary
