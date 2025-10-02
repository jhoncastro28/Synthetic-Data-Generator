"""
Módulo para carga y preprocesamiento de datos para el generador sintético.
Incluye división automática 80/20 y balanceo de clases.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.utils import resample
import yaml
import os
from typing import Tuple, Optional, Union


class DataLoader:
    """
    Clase para cargar y preprocesar datos para entrenamiento del GAN.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Inicializa el cargador de datos con configuración.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.scaler = None
        self.encoder = None
        self.feature_names = None
        self.target_column = None
        
    def load_data(self, 
                  file_path: str, 
                  target_column: Optional[str] = None,
                  file_type: str = 'csv') -> pd.DataFrame:
        """
        Carga datos desde archivo.
        
        Args:
            file_path: Ruta al archivo de datos
            target_column: Nombre de la columna objetivo (opcional)
            file_type: Tipo de archivo ('csv', 'excel', 'json')
            
        Returns:
            DataFrame con los datos cargados
        """
        if file_type == 'csv':
            data = pd.read_csv(file_path)
        elif file_type == 'excel':
            data = pd.read_excel(file_path)
        elif file_type == 'json':
            data = pd.read_json(file_path)
        else:
            raise ValueError(f"Tipo de archivo no soportado: {file_type}")
        
        self.target_column = target_column
        self.feature_names = data.columns.tolist()
        
        print(f"Datos cargados: {data.shape}")
        print(f"Columnas: {list(data.columns)}")
        
        return data
    
    def preprocess_data(self, 
                       data: pd.DataFrame,
                       normalize: bool = True,
                       balance_classes: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesa los datos: normalización, balanceo y división.
        
        Args:
            data: DataFrame con los datos
            normalize: Si normalizar los datos
            balance_classes: Si balancear las clases
            
        Returns:
            Tupla con (datos_entrenamiento, datos_validacion)
        """
        # Limpiar datos
        data_clean = self._clean_data(data)
        
        # Codificar categóricas antes de balancear/normalizar
        data_clean = self._encode_categoricals(data_clean)

        # Balancear clases si es necesario
        if balance_classes and self.target_column:
            data_clean = self._balance_classes(data_clean)
        
        # Normalizar datos
        if normalize:
            data_clean = self._normalize_data(data_clean)
        
        # Dividir datos 80/20
        train_data, val_data = self._split_data(data_clean)
        
        print(f"Datos de entrenamiento: {train_data.shape}")
        print(f"Datos de validación: {val_data.shape}")
        
        return train_data, val_data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Limpia los datos eliminando valores faltantes y duplicados."""
        # Eliminar filas con valores faltantes
        data_clean = data.dropna()
        
        # Eliminar duplicados
        data_clean = data_clean.drop_duplicates()
        
        print(f"Datos después de limpieza: {data_clean.shape}")
        return data_clean

    def _encode_categoricals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Codifica variables categóricas con OneHotEncoder y guarda el encoder."""
        # Excluir columna objetivo de la codificación
        features_df = data.drop(columns=[self.target_column]) if self.target_column and self.target_column in data.columns else data.copy()

        categorical_columns = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not categorical_columns:
            return data

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_array = self.encoder.fit_transform(features_df[categorical_columns])
        encoded_cols = self.encoder.get_feature_names_out(categorical_columns)

        # Construir nuevo DataFrame con numéricas + codificadas + target (si aplica)
        numeric_df = features_df.drop(columns=categorical_columns)
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=features_df.index)
        merged_features = pd.concat([numeric_df, encoded_df], axis=1)

        if self.target_column and self.target_column in data.columns:
            merged = pd.concat([merged_features, data[[self.target_column]]], axis=1)
        else:
            merged = merged_features

        # Actualizar feature names
        self.feature_names = [c for c in merged.columns if c != self.target_column]
        return merged
    
    def _balance_classes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Balancea las clases usando oversampling."""
        if not self.target_column or self.target_column not in data.columns:
            return data
        
        # Obtener conteo de clases
        class_counts = data[self.target_column].value_counts()
        max_class_size = class_counts.max()
        
        balanced_data = []
        
        for class_name in class_counts.index:
            class_data = data[data[self.target_column] == class_name]
            
            if len(class_data) < max_class_size:
                # Oversampling
                oversampled = resample(
                    class_data,
                    replace=True,
                    n_samples=max_class_size,
                    random_state=self.config['data']['random_state']
                )
                balanced_data.append(oversampled)
            else:
                balanced_data.append(class_data)
        
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        print(f"Datos balanceados: {balanced_df.shape}")
        return balanced_df
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normaliza los datos usando StandardScaler."""
        # Separar características numéricas
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            self.scaler = StandardScaler()
            data_normalized = data.copy()
            data_normalized[numeric_columns] = self.scaler.fit_transform(data[numeric_columns])
            return data_normalized
        
        return data
    
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide los datos en entrenamiento (80%) y validación (20%)."""
        train_split = self.config['data']['train_split']
        random_state = self.config['data']['random_state']
        
        train_data, val_data = train_test_split(
            data,
            train_size=train_split,
            random_state=random_state,
            stratify=data[self.target_column] if self.target_column and self.target_column in data.columns else None
        )
        
        return train_data, val_data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Aplica transformación inversa a los datos normalizados.
        
        Args:
            data: Datos normalizados
            
        Returns:
            Datos en escala original
        """
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data
    
    def get_feature_info(self) -> dict:
        """Retorna información sobre las características."""
        return {
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'scaler': self.scaler
        }
