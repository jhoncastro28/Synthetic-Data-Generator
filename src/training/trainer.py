"""
Clase Trainer para el entrenamiento del GAN con logging y checkpoints.
Incluye monitoreo de métricas, early stopping y visualizaciones.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, callbacks

from ..models.gan import GAN, ConditionalGAN
from ..utils.metrics import SyntheticDataEvaluator
from ..utils.data_loader import DataLoader


class GANTrainer:
    """
    Entrenador para GANs con logging avanzado y monitoreo.
    """
    
    def __init__(self, 
                 gan_model: GAN,
                 config_path: str = "configs/config.yaml",
                 results_dir: str = "results",
                 use_tensorboard: bool = True,
                 use_wandb: bool = False):
        """
        Inicializa el entrenador.
        
        Args:
            gan_model: Modelo GAN a entrenar
            config_path: Ruta al archivo de configuración
            results_dir: Directorio para guardar resultados
            use_tensorboard: Si usar TensorBoard
            use_wandb: Si usar Weights & Biases
        """
        self.gan_model = gan_model
        self.config_path = config_path
        self.results_dir = results_dir
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        # Crear directorios
        self._create_directories()
        
        # Cargar configuración
        self._load_config()
        
        # Inicializar logging
        self._setup_logging()
        
        # Historial de entrenamiento
        self.training_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'discriminator_real': [],
            'discriminator_fake': [],
            'crc1rs_score': [],
            'epochs': []
        }
        
        # Mejor modelo
        self.best_crc1rs_score = 0.0
        self.best_model_path = None
        
    def _create_directories(self):
        """Crea los directorios necesarios."""
        directories = [
            self.results_dir,
            os.path.join(self.results_dir, 'models'),
            os.path.join(self.results_dir, 'plots'),
            os.path.join(self.results_dir, 'logs'),
            os.path.join(self.results_dir, 'synthetic_data')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _load_config(self):
        """Carga la configuración desde archivo."""
        import yaml
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def _setup_logging(self):
        """Configura el sistema de logging."""
        # TensorBoard
        if self.use_tensorboard:
            self.tensorboard_dir = os.path.join(self.results_dir, 'logs', 'tensorboard')
            os.makedirs(self.tensorboard_dir, exist_ok=True)
            
            self.tensorboard_callback = callbacks.TensorBoard(
                log_dir=self.tensorboard_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        
        # Weights & Biases
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config['logging']['project_name'],
                    config=self.config,
                    dir=self.results_dir
                )
                self.wandb = wandb
            except ImportError:
                print("Weights & Biases no está instalado. Usando solo TensorBoard.")
                self.use_wandb = False
    
    def train(self, 
              train_data: np.ndarray,
              validation_data: np.ndarray,
              feature_names: Optional[List[str]] = None,
              class_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Entrena el modelo GAN.
        
        Args:
            train_data: Datos de entrenamiento
            validation_data: Datos de validación
            feature_names: Nombres de las características
            class_labels: Etiquetas de clase (para GAN condicional)
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        print("Iniciando entrenamiento del GAN...")
        print(f"Datos de entrenamiento: {train_data.shape}")
        print(f"Datos de validación: {validation_data.shape}")
        
        # Configurar optimizadores
        self._setup_optimizers()
        
        # Compilar modelo
        self._compile_model()
        
        # Configurar callbacks
        callbacks_list = self._setup_callbacks()
        
        # Preparar datos
        if isinstance(self.gan_model, ConditionalGAN) and class_labels is not None:
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data, class_labels))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        
        train_dataset = train_dataset.batch(self.config['training']['batch_size'])
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Entrenamiento
        start_time = datetime.now()
        
        try:
            # Loop de entrenamiento manual para mayor control
            self._manual_training_loop(
                train_dataset, 
                validation_data, 
                feature_names,
                class_labels
            )
            
        except KeyboardInterrupt:
            print("\nEntrenamiento interrumpido por el usuario.")
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"Entrenamiento completado en {training_time}")
        
        # Generar reporte final
        results = self._generate_final_report(validation_data, feature_names)
        results['training_time'] = str(training_time)
        
        return results
    
    def _setup_optimizers(self):
        """Configura los optimizadores."""
        lr_g = self.config['training']['learning_rate_g']
        lr_d = self.config['training']['learning_rate_d']
        beta_1 = self.config['training']['beta_1']
        beta_2 = self.config['training']['beta_2']
        
        self.generator_optimizer = optimizers.Adam(
            learning_rate=lr_g, 
            beta_1=beta_1, 
            beta_2=beta_2
        )
        self.discriminator_optimizer = optimizers.Adam(
            learning_rate=lr_d, 
            beta_1=beta_1, 
            beta_2=beta_2
        )
    
    def _compile_model(self):
        """Compila el modelo GAN."""
        self.gan_model.compile(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer
        )
    
    def _setup_callbacks(self) -> List[callbacks.Callback]:
        """Configura los callbacks de entrenamiento."""
        callbacks_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='generator_loss',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            mode='min'
        )
        callbacks_list.append(early_stopping)
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.results_dir, 'models', 'best_model.weights.h5')
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='generator_loss',
            save_best_only=True,
            mode='min',
            save_weights_only=True
        )
        callbacks_list.append(model_checkpoint)
        
        # TensorBoard
        if self.use_tensorboard:
            callbacks_list.append(self.tensorboard_callback)
        
        return callbacks_list
    
    def _manual_training_loop(self, 
                              train_dataset: tf.data.Dataset,
                              validation_data: np.ndarray,
                              feature_names: Optional[List[str]],
                              class_labels: Optional[np.ndarray]):
        """Loop de entrenamiento manual con evaluación personalizada."""
        epochs = self.config['training']['epochs']
        save_interval = self.config['training']['save_interval']
        log_interval = self.config['logging']['log_interval']
        
        for epoch in range(epochs):
            # Entrenar una época
            epoch_metrics = self._train_epoch(train_dataset)
            
            # Evaluar en datos de validación solo cada log_interval
            if epoch % log_interval == 0:
                try:
                    validation_metrics = self._evaluate_on_validation(
                        validation_data, feature_names, class_labels
                    )
                    epoch_metrics.update(validation_metrics)
                except Exception as e:
                    print(f"Error en evaluación: {e}")
                    # Continuar sin evaluación si hay error
            
            # Actualizar historial
            self._update_training_history(epoch, epoch_metrics)
            
            # Logging
            if epoch % log_interval == 0:
                self._log_epoch(epoch, epoch_metrics)
            
            # Guardar modelo periódicamente
            if epoch % save_interval == 0 and epoch > 0:
                try:
                    self._save_checkpoint(epoch)
                except Exception as e:
                    print(f"Error guardando checkpoint: {e}")
            
            # Early stopping check
            if self._check_early_stopping():
                print(f"Early stopping en época {epoch}")
                break
    
    def _train_epoch(self, train_dataset: tf.data.Dataset) -> Dict[str, float]:
        """Entrena una época completa."""
        epoch_metrics = {
            'generator_loss': 0.0,
            'discriminator_loss': 0.0,
            'discriminator_real': 0.0,
            'discriminator_fake': 0.0
        }
        
        num_batches = 0
        
        for batch in train_dataset:
            if isinstance(self.gan_model, ConditionalGAN):
                # Para GAN condicional, necesitamos etiquetas
                # Por ahora, generamos etiquetas aleatorias
                batch_data, batch_labels = batch
            else:
                batch_data = batch
                batch_labels = None
            
            # Entrenar el modelo
            if batch_labels is not None:
                metrics = self.gan_model.train_step((batch_data, batch_labels))
            else:
                metrics = self.gan_model.train_step(batch_data)
            
            # Acumular métricas
            for key, value in metrics.items():
                epoch_metrics[key] += value.numpy()
            
            num_batches += 1
        
        # Promediar métricas
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def _evaluate_on_validation(self, 
                                validation_data: np.ndarray,
                                feature_names: Optional[List[str]],
                                class_labels: Optional[np.ndarray]) -> Dict[str, float]:
        """Evalúa el modelo en datos de validación."""
        # Generar datos sintéticos
        num_samples = min(len(validation_data), 1000)  # Limitar para eficiencia
        
        if isinstance(self.gan_model, ConditionalGAN) and class_labels is not None:
            synthetic_data = self.gan_model.generate_synthetic_data(
                num_samples, class_labels[:num_samples]
            )
        else:
            synthetic_data = self.gan_model.generate_synthetic_data(num_samples)
        
        # Evaluar calidad
        evaluator = SyntheticDataEvaluator(
            validation_data[:num_samples], 
            synthetic_data
        )
        
        metrics = evaluator.calculate_crc1rs_metric()
        
        return metrics
    
    def _update_training_history(self, epoch: int, metrics: Dict[str, float]):
        """Actualiza el historial de entrenamiento."""
        self.training_history['epochs'].append(epoch)
        
        for key, value in metrics.items():
            if key in self.training_history:
                self.training_history[key].append(value)
        
        # Actualizar mejor modelo
        if 'crc1rs_score' in metrics:
            if metrics['crc1rs_score'] > self.best_crc1rs_score:
                self.best_crc1rs_score = metrics['crc1rs_score']
                self.best_model_path = os.path.join(
                    self.results_dir, 'models', f'best_model_epoch_{epoch}'
                )
    
    def _log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Registra información de la época."""
        print(f"\nÉpoca {epoch}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
        
        # TensorBoard
        if self.use_tensorboard:
            with self.tensorboard_callback.writer.as_default():
                for key, value in metrics.items():
                    tf.summary.scalar(key, value, step=epoch)
                self.tensorboard_callback.writer.flush()
        
        # Weights & Biases
        if self.use_wandb:
            self.wandb.log(metrics, step=epoch)
    
    def _save_checkpoint(self, epoch: int):
        """Guarda un checkpoint del modelo."""
        checkpoint_path = os.path.join(
            self.results_dir, 'models', f'checkpoint_epoch_{epoch}'
        )
        self.gan_model.save_model(checkpoint_path)
        print(f"Checkpoint guardado en época {epoch}")
    
    def _generate_visualizations(self, 
                                 validation_data: np.ndarray,
                                 feature_names: Optional[List[str]],
                                 epoch: int):
        """Genera visualizaciones del progreso."""
        # Generar datos sintéticos para visualización
        num_samples = min(len(validation_data), 500)
        synthetic_data = self.gan_model.generate_synthetic_data(num_samples)
        
        # Crear evaluador
        evaluator = SyntheticDataEvaluator(
            validation_data[:num_samples], 
            synthetic_data
        )
        
        # Generar gráficas
        plot_path = os.path.join(
            self.results_dir, 'plots', f'comparison_epoch_{epoch}.png'
        )
        evaluator.plot_comparison(feature_names, plot_path)
        
        # Guardar datos sintéticos
        synthetic_df = pd.DataFrame(
            synthetic_data, 
            columns=feature_names or [f'Feature_{i}' for i in range(synthetic_data.shape[1])]
        )
        synthetic_path = os.path.join(
            self.results_dir, 'synthetic_data', f'synthetic_epoch_{epoch}.csv'
        )
        synthetic_df.to_csv(synthetic_path, index=False)
    
    def _check_early_stopping(self) -> bool:
        """Verifica si se debe aplicar early stopping."""
        if len(self.training_history['crc1rs_score']) < 10:
            return False
        
        # Verificar si no hay mejora en las últimas 10 épocas
        recent_scores = self.training_history['crc1rs_score'][-10:]
        if len(recent_scores) >= 10:
            best_recent = max(recent_scores)
            if best_recent <= self.best_crc1rs_score * 0.95:  # 5% de tolerancia
                return True
        
        return False
    
    def _generate_final_report(self, 
                              validation_data: np.ndarray,
                              feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """Genera el reporte final del entrenamiento."""
        # Generar datos sintéticos finales
        synthetic_data = self.gan_model.generate_synthetic_data(len(validation_data))
        
        # Evaluación completa
        evaluator = SyntheticDataEvaluator(validation_data, synthetic_data)
        all_metrics = evaluator.calculate_all_metrics()
        
        # Generar reporte
        report_path = os.path.join(self.results_dir, 'evaluation_report.txt')
        report = evaluator.generate_evaluation_report(report_path)
        
        # Gráficas finales
        final_plot_path = os.path.join(self.results_dir, 'plots', 'final_comparison.png')
        evaluator.plot_comparison(feature_names, final_plot_path)
        
        # Guardar datos sintéticos finales
        synthetic_df = pd.DataFrame(
            synthetic_data,
            columns=feature_names or [f'Feature_{i}' for i in range(synthetic_data.shape[1])]
        )
        final_synthetic_path = os.path.join(self.results_dir, 'synthetic_data', 'final_synthetic.csv')
        synthetic_df.to_csv(final_synthetic_path, index=False)
        
        # Guardar historial de entrenamiento
        history_df = pd.DataFrame(self.training_history)
        history_path = os.path.join(self.results_dir, 'training_history.csv')
        history_df.to_csv(history_path, index=False)
        
        # Gráfica de evolución de métricas
        self._plot_training_evolution()
        
        return {
            'final_metrics': all_metrics,
            'training_history': self.training_history,
            'best_crc1rs_score': self.best_crc1rs_score,
            'report_path': report_path,
            'synthetic_data_path': final_synthetic_path,
            'training_history_path': history_path
        }
    
    def _plot_training_evolution(self):
        """Genera gráficas de evolución del entrenamiento."""
        if not self.training_history['epochs']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss del generador
        axes[0, 0].plot(self.training_history['epochs'], self.training_history['generator_loss'])
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Loss del discriminador
        axes[0, 1].plot(self.training_history['epochs'], self.training_history['discriminator_loss'])
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Score del discriminador
        axes[1, 0].plot(self.training_history['epochs'], self.training_history['discriminator_real'], label='Real')
        axes[1, 0].plot(self.training_history['epochs'], self.training_history['discriminator_fake'], label='Fake')
        axes[1, 0].set_title('Discriminator Scores')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # CrC1RS Score
        if self.training_history['crc1rs_score']:
            axes[1, 1].plot(self.training_history['epochs'], self.training_history['crc1rs_score'])
            axes[1, 1].set_title('CrC1RS Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Guardar gráfica
        evolution_path = os.path.join(self.results_dir, 'plots', 'training_evolution.png')
        plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def load_best_model(self):
        """Carga el mejor modelo guardado."""
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.gan_model.load_model(self.best_model_path)
            print(f"Mejor modelo cargado desde {self.best_model_path}")
        else:
            print("No se encontró el mejor modelo guardado.")
    
    def generate_synthetic_data(self, num_samples: int) -> np.ndarray:
        """Genera datos sintéticos usando el modelo entrenado."""
        return self.gan_model.generate_synthetic_data(num_samples)
