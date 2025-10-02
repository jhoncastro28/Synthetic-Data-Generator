"""
Clase Trainer para el entrenamiento del GAN con logging y checkpoints.
Incluye monitoreo de métricas, early stopping y visualizaciones.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # CRÍTICO: Configurar backend no-interactivo ANTES de importar pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, callbacks

from ..utils.pickle_manager import PickleManager
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
        
        # Inicializar PickleManager
        self.pickle_manager = PickleManager(results_dir)
        
        # Crear directorios
        self._create_directories()
        
        # Cargar configuración
        self._load_config()
        
        # Inicializar logging
        self._setup_logging()
        
        # Historial de entrenamiento - CORREGIDO: Inicializar todas las listas
        self.training_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'discriminator_real': [],
            'discriminator_fake': [],
            'correlation_score': [],
            'consistency_score': [],
            'robustness_score': [],
            'similarity_score': [],
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
              class_labels: Optional[np.ndarray] = None,
              start_epoch: int = 0,
              experiment_name: str = None) -> Dict[str, Any]:
        """
        Entrena el modelo GAN.
        
        Args:
            train_data: Datos de entrenamiento
            validation_data: Datos de validación
            feature_names: Nombres de las características
            class_labels: Etiquetas de clase (para GAN condicional)
            start_epoch: Época inicial
            experiment_name: Nombre del experimento
            
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
                class_labels,
                start_epoch,
                experiment_name
            )
            
        except KeyboardInterrupt:
            print("\nEntrenamiento interrumpido por el usuario.")
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"Entrenamiento completado en {training_time}")
        
        # Generar reporte final
        results = self._generate_final_report(validation_data, feature_names)
        results['training_time'] = str(training_time)
        results['final_epoch'] = len(self.training_history['epochs']) - 1 if self.training_history['epochs'] else 0
        
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
    
    def _manual_training_loop(self, 
                              train_dataset: tf.data.Dataset,
                              validation_data: np.ndarray,
                              feature_names: Optional[List[str]],
                              class_labels: Optional[np.ndarray],
                              start_epoch: int = 0,
                              experiment_name: str = None):
        """Loop de entrenamiento manual con evaluación personalizada."""
        epochs = self.config['training']['epochs']
        save_interval = self.config['training']['save_interval']
        log_interval = self.config['logging']['log_interval']
        
        for epoch in range(start_epoch, epochs):
            # Entrenar una época
            epoch_metrics = self._train_epoch(train_dataset)
            
            # CORREGIDO: Siempre actualizar historial básico
            self._update_basic_history(epoch, epoch_metrics)
            
            # Evaluar en datos de validación solo cada log_interval
            if epoch % log_interval == 0:
                try:
                    validation_metrics = self._evaluate_on_validation(
                        validation_data, feature_names, class_labels
                    )
                    epoch_metrics.update(validation_metrics)
                    
                    # Actualizar métricas de validación en el historial
                    for key in ['correlation_score', 'consistency_score', 'robustness_score', 
                               'similarity_score', 'crc1rs_score']:
                        if key in validation_metrics:
                            self.training_history[key].append(validation_metrics[key])
                    
                except Exception as e:
                    print(f"Error en evaluación: {e}")
                    # Agregar valores por defecto si hay error
                    for key in ['correlation_score', 'consistency_score', 'robustness_score', 
                               'similarity_score', 'crc1rs_score']:
                        self.training_history[key].append(0.0)
            
            # Logging
            if epoch % log_interval == 0:
                self._log_epoch(epoch, epoch_metrics)
            
            # Guardar estado completo con pickle
            if epoch % save_interval == 0 and epoch > 0:
                try:
                    self.save_training_state(epoch, experiment_name)
                except Exception as e:
                    print(f"Error guardando estado: {e}")
            
            # Early stopping check
            if self._check_early_stopping():
                print(f"Early stopping en época {epoch}")
                break
    
    def _update_basic_history(self, epoch: int, metrics: Dict[str, float]):
        """NUEVO: Actualiza solo el historial básico (sin métricas de validación)."""
        self.training_history['epochs'].append(epoch)
        
        for key in ['generator_loss', 'discriminator_loss', 'discriminator_real', 'discriminator_fake']:
            if key in metrics:
                self.training_history[key].append(float(metrics[key]))
    
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
                if key in epoch_metrics:
                    epoch_metrics[key] += float(value.numpy())
            
            num_batches += 1
        
        # Promediar métricas
        for key in epoch_metrics:
            if num_batches > 0:
                epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def _evaluate_on_validation(self, 
                                validation_data: np.ndarray,
                                feature_names: Optional[List[str]],
                                class_labels: Optional[np.ndarray]) -> Dict[str, float]:
        """Evalúa el modelo en datos de validación."""
        # Generar datos sintéticos
        num_samples = min(len(validation_data), 1000)
        
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
        
        # Actualizar mejor modelo
        if metrics['crc1rs_score'] > self.best_crc1rs_score:
            self.best_crc1rs_score = metrics['crc1rs_score']
        
        return metrics
    
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
    
    def _check_early_stopping(self) -> bool:
        """Verifica si se debe aplicar early stopping."""
        crc1rs_scores = self.training_history.get('crc1rs_score', [])
        
        if len(crc1rs_scores) < 10:
            return False
        
        # Verificar si no hay mejora en las últimas 10 mediciones
        recent_scores = crc1rs_scores[-10:]
        best_recent = max(recent_scores) if recent_scores else 0
        
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
        
        # CORREGIDO: Guardar historial de entrenamiento con longitudes verificadas
        try:
            # Asegurar que todas las listas tienen la misma longitud
            min_length = min(len(v) for v in self.training_history.values() if isinstance(v, list))
            
            # Truncar todas las listas a la longitud mínima
            truncated_history = {}
            for key, value in self.training_history.items():
                if isinstance(value, list):
                    truncated_history[key] = value[:min_length]
                else:
                    truncated_history[key] = value
            
            history_df = pd.DataFrame(truncated_history)
            history_path = os.path.join(self.results_dir, 'training_history.csv')
            history_df.to_csv(history_path, index=False)
        except Exception as e:
            print(f"Advertencia: No se pudo guardar el historial de entrenamiento: {e}")
            history_path = None
        
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
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss del generador
            if self.training_history['generator_loss']:
                axes[0, 0].plot(self.training_history['epochs'], 
                               self.training_history['generator_loss'])
                axes[0, 0].set_title('Generator Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].grid(True)
            
            # Loss del discriminador
            if self.training_history['discriminator_loss']:
                axes[0, 1].plot(self.training_history['epochs'], 
                               self.training_history['discriminator_loss'])
                axes[0, 1].set_title('Discriminator Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].grid(True)
            
            # Score del discriminador
            if self.training_history['discriminator_real'] and self.training_history['discriminator_fake']:
                axes[1, 0].plot(self.training_history['epochs'], 
                               self.training_history['discriminator_real'], label='Real')
                axes[1, 0].plot(self.training_history['epochs'], 
                               self.training_history['discriminator_fake'], label='Fake')
                axes[1, 0].set_title('Discriminator Scores')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # CrC1RS Score
            if self.training_history.get('crc1rs_score'):
                # Crear eje x solo para los puntos donde hay medición
                log_interval = self.config['logging']['log_interval']
                epochs_with_crc1rs = [e for e in self.training_history['epochs'] 
                                     if e % log_interval == 0][:len(self.training_history['crc1rs_score'])]
                
                axes[1, 1].plot(epochs_with_crc1rs, self.training_history['crc1rs_score'])
                axes[1, 1].set_title('CrC1RS Score')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Guardar gráfica
            evolution_path = os.path.join(self.results_dir, 'plots', 'training_evolution.png')
            plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # CRÍTICO: Cerrar la figura explícitamente
            print(f"Gráfica de evolución guardada: {evolution_path}")
            
        except Exception as e:
            print(f"Error generando gráfica de evolución: {e}")
            plt.close('all')  # Cerrar todas las figuras en caso de error
    
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
    
    def save_training_state(self, epoch: int, experiment_name: str = None):
        """
        Guarda el estado completo del entrenamiento.
        
        Args:
            epoch: Época actual
            experiment_name: Nombre del experimento
        """
        if experiment_name is None:
            experiment_name = f"gan_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Preparar estado del entrenador
        trainer_state = {
            'training_history': self.training_history,
            'best_crc1rs_score': self.best_crc1rs_score,
            'best_model_path': self.best_model_path,
            'gan_model_state': {
                'generator_weights': self.gan_model.generator.get_weights(),
                'discriminator_weights': self.gan_model.discriminator.get_weights(),
                'latent_dim': self.gan_model.latent_dim,
                'use_wasserstein': self.gan_model.use_wasserstein
            }
        }
        
        # Guardar estado usando PickleManager
        self.pickle_manager.save_training_state(
            trainer_state, epoch, self.config
        )
        
        # Guardar modelos individuales
        self.pickle_manager.save_model(
            self.gan_model.generator,
            f"{experiment_name}_generator",
            epoch,
            {'latent_dim': self.gan_model.latent_dim}
        )
        
        self.pickle_manager.save_model(
            self.gan_model.discriminator,
            f"{experiment_name}_discriminator", 
            epoch,
            {'input_dim': getattr(self.gan_model.discriminator, 'input_dim', None)}
        )
        
        print(f"Estado de entrenamiento guardado para época {epoch}")
    
    def load_training_state(self, epoch: Optional[int] = None, experiment_name: str = None):
        """
        Carga el estado del entrenamiento.
        
        Args:
            epoch: Época específica (opcional)
            experiment_name: Nombre del experimento
        """
        try:
            # Cargar estado usando PickleManager
            trainer_state, loaded_epoch, config = self.pickle_manager.load_training_state(epoch)
            
            # Restaurar estado del entrenador
            self.training_history = trainer_state['training_history']
            self.best_crc1rs_score = trainer_state['best_crc1rs_score']
            self.best_model_path = trainer_state['best_model_path']
            
            # Restaurar pesos de los modelos
            gan_state = trainer_state['gan_model_state']
            self.gan_model.generator.set_weights(gan_state['generator_weights'])
            self.gan_model.discriminator.set_weights(gan_state['discriminator_weights'])
            
            print(f"Estado de entrenamiento cargado desde época {loaded_epoch}")
            return loaded_epoch, config
            
        except FileNotFoundError as e:
            print(f"No se pudo cargar el estado: {e}")
            return None, None
    
    def save_preprocessors(self, data_loader: DataLoader, experiment_name: str):
        """
        Guarda los preprocesadores usados.
        
        Args:
            data_loader: DataLoader con preprocesadores
            experiment_name: Nombre del experimento
        """
        try:
            # Guardar scaler si existe
            if hasattr(data_loader, 'scaler') and data_loader.scaler is not None:
                self.pickle_manager.save_preprocessor(
                    data_loader.scaler,
                    f"{experiment_name}_scaler",
                    {'feature_names': getattr(data_loader, 'feature_names', [])}
                )
            
            # Guardar encoder si existe
            if hasattr(data_loader, 'encoder') and data_loader.encoder is not None:
                self.pickle_manager.save_preprocessor(
                    data_loader.encoder,
                    f"{experiment_name}_encoder",
                    {'target_column': getattr(data_loader, 'target_column', None)}
                )
            
            print("Preprocesadores guardados")
            
        except Exception as e:
            print(f"Error guardando preprocesadores: {e}")
    
    def load_preprocessors(self, experiment_name: str) -> Tuple[Any, Any]:
        """
        Carga los preprocesadores guardados.
        
        Args:
            experiment_name: Nombre del experimento
            
        Returns:
            Tupla con (scaler, encoder)
        """
        scaler = None
        encoder = None
        
        try:
            scaler, _ = self.pickle_manager.load_preprocessor(f"{experiment_name}_scaler")
        except FileNotFoundError:
            print("Scaler no encontrado")
        
        try:
            encoder, _ = self.pickle_manager.load_preprocessor(f"{experiment_name}_encoder")
        except FileNotFoundError:
            print("Encoder no encontrado")
        
        return scaler, encoder
    
    def save_experiment_results(self, 
                               synthetic_data: np.ndarray,
                               feature_names: list,
                               metrics: Dict,
                               epoch: int,
                               experiment_name: str):
        """
        Guarda los resultados completos del experimento.
        
        Args:
            synthetic_data: Datos sintéticos generados
            feature_names: Nombres de las características
            metrics: Métricas de evaluación
            epoch: Época del entrenamiento
            experiment_name: Nombre del experimento
        """
        # Guardar datos sintéticos
        self.pickle_manager.save_synthetic_data(
            synthetic_data, feature_names, experiment_name, epoch
        )
        
        # Guardar métricas
        self.pickle_manager.save_metrics(
            metrics, experiment_name, epoch
        )
        
        # Guardar resultados completos
        results = {
            'synthetic_data_shape': synthetic_data.shape,
            'feature_names': feature_names,
            'metrics': metrics,
            'epoch': epoch,
            'best_crc1rs_score': self.best_crc1rs_score
        }
        
        self.pickle_manager.save_results(
            results, f"{experiment_name}_complete", 
            {'epoch': epoch, 'timestamp': datetime.now().isoformat()}
        )
        
        print(f"Resultados del experimento guardados para época {epoch}")
    
    def resume_training(self, 
                      train_data: np.ndarray,
                      validation_data: np.ndarray,
                      feature_names: Optional[List[str]] = None,
                      experiment_name: str = None) -> Dict:
        """
        Continúa un entrenamiento interrumpido.
        
        Args:
            train_data: Datos de entrenamiento
            validation_data: Datos de validación
            feature_names: Nombres de las características
            experiment_name: Nombre del experimento
            
        Returns:
            Resultados del entrenamiento
        """
        if experiment_name is None:
            experiment_name = f"gan_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Intentar cargar estado previo
        loaded_epoch, config = self.load_training_state(experiment_name=experiment_name)
        
        if loaded_epoch is not None:
            print(f"Continuando entrenamiento desde época {loaded_epoch}")
            start_epoch = loaded_epoch + 1
        else:
            print("Iniciando nuevo entrenamiento")
            start_epoch = 0
        
        # Continuar entrenamiento
        return self.train(
            train_data, validation_data, feature_names,
            start_epoch=start_epoch, experiment_name=experiment_name
        )