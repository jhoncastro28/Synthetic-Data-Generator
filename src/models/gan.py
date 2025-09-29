"""
Clase principal del GAN que coordina generador y discriminador.
Implementa el entrenamiento adversarial y la generación de datos sintéticos.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import os
import json

from .generator import Generator, ConditionalGenerator
from .discriminator import Discriminator, ConditionalDiscriminator, WassersteinDiscriminator


class GAN(keras.Model):
    """
    GAN principal que coordina generador y discriminador.
    """
    
    def __init__(self, 
                 generator: Generator,
                 discriminator: Discriminator,
                 latent_dim: int = 100,
                 use_wasserstein: bool = False,
                 gradient_penalty_weight: float = 10.0,
                 **kwargs):
        """
        Inicializa el GAN.
        
        Args:
            generator: Instancia del generador
            discriminator: Instancia del discriminador
            latent_dim: Dimensión del vector latente
            use_wasserstein: Si usar WGAN
            gradient_penalty_weight: Peso del gradient penalty
        """
        super(GAN, self).__init__(**kwargs)
        
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.use_wasserstein = use_wasserstein
        self.gradient_penalty_weight = gradient_penalty_weight
        
        # Métricas de seguimiento
        self.generator_loss_metric = keras.metrics.Mean(name='generator_loss')
        self.discriminator_loss_metric = keras.metrics.Mean(name='discriminator_loss')
        self.discriminator_real_metric = keras.metrics.Mean(name='discriminator_real')
        self.discriminator_fake_metric = keras.metrics.Mean(name='discriminator_fake')
    
    def compile(self, 
                generator_optimizer: optimizers.Optimizer,
                discriminator_optimizer: optimizers.Optimizer,
                **kwargs):
        """
        Compila el GAN con optimizadores.
        
        Args:
            generator_optimizer: Optimizador para el generador
            discriminator_optimizer: Optimizador para el discriminador
        """
        super().compile(**kwargs)
        
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        
        # Loss functions
        if self.use_wasserstein:
            self.generator_loss_fn = self._wasserstein_generator_loss
            self.discriminator_loss_fn = self._wasserstein_discriminator_loss
        else:
            self.generator_loss_fn = self._standard_generator_loss
            self.discriminator_loss_fn = self._standard_discriminator_loss
    
    def _standard_generator_loss(self, fake_output):
        """Loss estándar para el generador."""
        return tf.keras.losses.binary_crossentropy(
            tf.ones_like(fake_output), fake_output
        )
    
    def _standard_discriminator_loss(self, real_output, fake_output):
        """Loss estándar para el discriminador."""
        real_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(real_output), real_output
        )
        fake_loss = tf.keras.losses.binary_crossentropy(
            tf.zeros_like(fake_output), fake_output
        )
        return real_loss + fake_loss
    
    def _wasserstein_generator_loss(self, fake_output):
        """Loss de Wasserstein para el generador."""
        return -tf.reduce_mean(fake_output)
    
    def _wasserstein_discriminator_loss(self, real_output, fake_output):
        """Loss de Wasserstein para el discriminador."""
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
    
    def train_step(self, data):
        """
        Un paso de entrenamiento del GAN.
        
        Args:
            data: Datos reales
            
        Returns:
            Diccionario con métricas
        """
        batch_size = tf.shape(data)[0]
        
        # Generar ruido latente
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        # Entrenar discriminador
        with tf.GradientTape() as d_tape:
            fake_data = self.generator(noise, training=True)
            
            real_output = self.discriminator(data, training=True)
            fake_output = self.discriminator(fake_data, training=True)
            
            d_loss = self.discriminator_loss_fn(real_output, fake_output)
            
            # Gradient penalty para WGAN
            if self.use_wasserstein and hasattr(self.discriminator, 'gradient_penalty'):
                penalty = self.discriminator.gradient_penalty(
                    data, fake_data, self.gradient_penalty_weight
                )
                d_loss += penalty
        
        # Aplicar gradientes al discriminador
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )
        
        # Entrenar generador
        with tf.GradientTape() as g_tape:
            fake_data = self.generator(noise, training=True)
            fake_output = self.discriminator(fake_data, training=True)
            
            g_loss = self.generator_loss_fn(fake_output)
        
        # Aplicar gradientes al generador
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        
        # Actualizar métricas
        self.generator_loss_metric.update_state(g_loss)
        self.discriminator_loss_metric.update_state(d_loss)
        self.discriminator_real_metric.update_state(tf.reduce_mean(real_output))
        self.discriminator_fake_metric.update_state(tf.reduce_mean(fake_output))
        
        return {
            'generator_loss': self.generator_loss_metric.result(),
            'discriminator_loss': self.discriminator_loss_metric.result(),
            'discriminator_real': self.discriminator_real_metric.result(),
            'discriminator_fake': self.discriminator_fake_metric.result()
        }
    
    def generate_synthetic_data(self, 
                               num_samples: int,
                               random_seed: Optional[int] = None) -> np.ndarray:
        """
        Genera datos sintéticos.
        
        Args:
            num_samples: Número de muestras a generar
            random_seed: Semilla para reproducibilidad
            
        Returns:
            Datos sintéticos generados
        """
        return self.generator.generate_samples(num_samples, random_seed)
    
    def evaluate_discriminator(self, real_data: np.ndarray, 
                              synthetic_data: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el rendimiento del discriminador.
        
        Args:
            real_data: Datos reales
            synthetic_data: Datos sintéticos
            
        Returns:
            Diccionario con métricas del discriminador
        """
        real_predictions = self.discriminator.predict_realism(real_data)
        synthetic_predictions = self.discriminator.predict_realism(synthetic_data)
        
        return {
            'real_accuracy': np.mean(real_predictions > 0.5),
            'synthetic_accuracy': np.mean(synthetic_predictions < 0.5),
            'real_mean_score': np.mean(real_predictions),
            'synthetic_mean_score': np.mean(synthetic_predictions)
        }
    
    def save_model(self, filepath: str):
        """
        Guarda el modelo completo.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        os.makedirs(filepath, exist_ok=True)
        
        # Guardar pesos
        self.generator.save_weights(os.path.join(filepath, 'generator_weights'))
        self.discriminator.save_weights(os.path.join(filepath, 'discriminator_weights'))
        
        # Guardar configuración
        config = {
            'latent_dim': self.latent_dim,
            'use_wasserstein': self.use_wasserstein,
            'gradient_penalty_weight': self.gradient_penalty_weight
        }
        
        with open(os.path.join(filepath, 'config.json'), 'w') as f:
            json.dump(config, f)
    
    def load_model(self, filepath: str):
        """
        Carga el modelo completo.
        
        Args:
            filepath: Ruta donde está guardado el modelo
        """
        # Cargar pesos
        self.generator.load_weights(os.path.join(filepath, 'generator_weights'))
        self.discriminator.load_weights(os.path.join(filepath, 'discriminator_weights'))
        
        # Cargar configuración
        with open(os.path.join(filepath, 'config.json'), 'r') as f:
            config = json.load(f)
        
        self.latent_dim = config['latent_dim']
        self.use_wasserstein = config['use_wasserstein']
        self.gradient_penalty_weight = config['gradient_penalty_weight']


class ConditionalGAN(GAN):
    """
    GAN condicional que puede generar datos para clases específicas.
    """
    
    def __init__(self, 
                 generator: ConditionalGenerator,
                 discriminator: ConditionalDiscriminator,
                 num_classes: int,
                 **kwargs):
        """
        Inicializa el GAN condicional.
        
        Args:
            generator: Generador condicional
            discriminator: Discriminador condicional
            num_classes: Número de clases
        """
        super().__init__(generator, discriminator, **kwargs)
        self.num_classes = num_classes
    
    def train_step(self, data):
        """
        Paso de entrenamiento para GAN condicional.
        
        Args:
            data: Tupla (datos_reales, etiquetas_clase)
        """
        real_data, class_labels = data
        batch_size = tf.shape(real_data)[0]
        
        # Generar ruido latente
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        # Generar etiquetas aleatorias para datos sintéticos
        fake_class_labels = tf.random.uniform(
            [batch_size], 0, self.num_classes, dtype=tf.int32
        )
        
        # Entrenar discriminador
        with tf.GradientTape() as d_tape:
            fake_data = self.generator((noise, fake_class_labels), training=True)
            
            real_output = self.discriminator((real_data, class_labels), training=True)
            fake_output = self.discriminator((fake_data, fake_class_labels), training=True)
            
            d_loss = self.discriminator_loss_fn(real_output, fake_output)
        
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )
        
        # Entrenar generador
        with tf.GradientTape() as g_tape:
            fake_data = self.generator((noise, fake_class_labels), training=True)
            fake_output = self.discriminator((fake_data, fake_class_labels), training=True)
            
            g_loss = self.generator_loss_fn(fake_output)
        
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        
        # Actualizar métricas
        self.generator_loss_metric.update_state(g_loss)
        self.discriminator_loss_metric.update_state(d_loss)
        self.discriminator_real_metric.update_state(tf.reduce_mean(real_output))
        self.discriminator_fake_metric.update_state(tf.reduce_mean(fake_output))
        
        return {
            'generator_loss': self.generator_loss_metric.result(),
            'discriminator_loss': self.discriminator_loss_metric.result(),
            'discriminator_real': self.discriminator_real_metric.result(),
            'discriminator_fake': self.discriminator_fake_metric.result()
        }
    
    def generate_synthetic_data(self, 
                               num_samples: int,
                               class_labels: Optional[np.ndarray] = None,
                               random_seed: Optional[int] = None) -> np.ndarray:
        """
        Genera datos sintéticos condicionales.
        
        Args:
            num_samples: Número de muestras a generar
            class_labels: Etiquetas de clase específicas
            random_seed: Semilla para reproducibilidad
            
        Returns:
            Datos sintéticos generados
        """
        return self.generator.generate_samples(
            num_samples, class_labels, random_seed
        )
