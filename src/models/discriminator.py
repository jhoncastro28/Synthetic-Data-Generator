"""
Implementación del Discriminador para el GAN.
Distingue entre datos reales y sintéticos.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import List, Optional, Dict, Any


class Discriminator(keras.Model):
    """
    Discriminador que aprende a distinguir datos reales de sintéticos.
    """
    
    def __init__(self, 
                 input_dim: int = 10,
                 hidden_layers: List[int] = [1024, 512, 256],
                 dropout_rate: float = 0.3,
                 use_batch_norm: bool = True,
                 activation: str = "leaky_relu",
                 use_spectral_norm: bool = False,
                 **kwargs):
        """
        Inicializa el discriminador.
        
        Args:
            input_dim: Dimensión de entrada (número de características)
            hidden_layers: Lista con número de neuronas por capa oculta
            dropout_rate: Tasa de dropout
            use_batch_norm: Si usar batch normalization
            activation: Función de activación
            use_spectral_norm: Si usar normalización espectral
        """
        super(Discriminator, self).__init__(**kwargs)
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_spectral_norm = use_spectral_norm
        
        # Construir arquitectura
        self._build_network(activation)
        
    def _build_network(self, activation: str):
        """Construye la red neuronal del discriminador."""
        self.layers_list = []
        
        # Primera capa: entrada
        self.layers_list.append(
            layers.Dense(
                self.hidden_layers[0],
                activation=activation,
                input_shape=(self.input_dim,),
                name='discriminator_dense_0'
            )
        )
        
        if self.use_batch_norm:
            self.layers_list.append(
                layers.BatchNormalization(name='discriminator_bn_0')
            )
        
        self.layers_list.append(
            layers.Dropout(self.dropout_rate, name='discriminator_dropout_0')
        )
        
        # Capas ocultas
        for i, units in enumerate(self.hidden_layers[1:], 1):
            self.layers_list.append(
                layers.Dense(
                    units,
                    activation=activation,
                    name=f'discriminator_dense_{i}'
                )
            )
            
            if self.use_batch_norm:
                self.layers_list.append(
                    layers.BatchNormalization(name=f'discriminator_bn_{i}')
                )
            
            self.layers_list.append(
                layers.Dropout(self.dropout_rate, name=f'discriminator_dropout_{i}')
            )
        
        # Capa de salida (probabilidad de autenticidad)
        self.layers_list.append(
            layers.Dense(
                1,
                activation='sigmoid',
                name='discriminator_output'
            )
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass del discriminador.
        
        Args:
            inputs: Tensor de datos (reales o sintéticos)
            training: Si está en modo entrenamiento
            
        Returns:
            Probabilidad de que los datos sean reales
        """
        x = inputs
        
        for layer in self.layers_list:
            x = layer(x, training=training)
        
        return x
    
    def predict_realism(self, data: np.ndarray) -> np.ndarray:
        """
        Predice la probabilidad de que los datos sean reales.
        
        Args:
            data: Datos a evaluar
            
        Returns:
            Probabilidades de autenticidad
        """
        predictions = self(data, training=False)
        return predictions.numpy()
    
    def get_model_summary(self) -> str:
        """Retorna un resumen de la arquitectura del modelo."""
        return self.summary()
    
    def save_weights(self, filepath: str):
        """Guarda los pesos del modelo."""
        self.save_weights(filepath)
    
    def load_weights(self, filepath: str):
        """Carga los pesos del modelo."""
        self.load_weights(filepath)


class ConditionalDiscriminator(Discriminator):
    """
    Discriminador condicional que considera las etiquetas de clase.
    """
    
    def __init__(self, 
                 num_classes: int,
                 class_embedding_dim: int = 50,
                 **kwargs):
        """
        Inicializa el discriminador condicional.
        
        Args:
            num_classes: Número de clases
            class_embedding_dim: Dimensión del embedding de clase
            **kwargs: Argumentos del discriminador base
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_embedding_dim = class_embedding_dim
        
        # Embedding para las clases
        self.class_embedding = layers.Embedding(
            num_classes,
            class_embedding_dim,
            name='class_embedding'
        )
        
        # Ajustar dimensión de entrada (datos + embedding de clase)
        self.adjusted_input_dim = self.input_dim + class_embedding_dim
    
    def call(self, inputs, training=None):
        """
        Forward pass del discriminador condicional.
        
        Args:
            inputs: Tupla (data, class_labels)
            training: Si está en modo entrenamiento
        """
        data, class_labels = inputs
        
        # Obtener embeddings de clase
        class_embeddings = self.class_embedding(class_labels)
        
        # Concatenar datos con embeddings de clase
        combined_input = tf.concat([data, class_embeddings], axis=1)
        
        # Pasar por la red
        x = combined_input
        for layer in self.layers_list:
            x = layer(x, training=training)
        
        return x


class WassersteinDiscriminator(Discriminator):
    """
    Discriminador optimizado para Wasserstein GAN (WGAN).
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa el discriminador de Wasserstein.
        """
        super().__init__(**kwargs)
        
        # Cambiar activación de salida para WGAN
        self.layers_list[-1] = layers.Dense(
            1,
            activation='linear',  # Sin activación para WGAN
            name='discriminator_output'
        )
    
    def gradient_penalty(self, 
                        real_data: tf.Tensor, 
                        fake_data: tf.Tensor,
                        penalty_weight: float = 10.0) -> tf.Tensor:
        """
        Calcula el gradient penalty para WGAN-GP.
        
        Args:
            real_data: Datos reales
            fake_data: Datos sintéticos
            penalty_weight: Peso del penalty
            
        Returns:
            Tensor con el gradient penalty
        """
        batch_size = tf.shape(real_data)[0]
        
        # Interpolación entre datos reales y sintéticos
        alpha = tf.random.uniform([batch_size, 1], 0., 1.)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            interpolated_output = self(interpolated, training=True)
        
        gradients = tape.gradient(interpolated_output, interpolated)
        gradient_norm = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=1))
        penalty = tf.reduce_mean((gradient_norm - 1) ** 2)
        
        return penalty_weight * penalty
    
    def call(self, inputs, training=None):
        """
        Forward pass del discriminador de Wasserstein.
        
        Args:
            inputs: Tensor de datos
            training: Si está en modo entrenamiento
            
        Returns:
            Score de critic (sin activación sigmoidal)
        """
        x = inputs
        
        for layer in self.layers_list:
            x = layer(x, training=training)
        
        return x
