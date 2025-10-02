"""
Implementación del Generador para el GAN.
Genera datos sintéticos a partir de ruido aleatorio.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import List, Optional, Dict, Any


class Generator(keras.Model):
    """
    Generador que aprende a crear datos sintéticos realistas.
    """
    
    def __init__(self, 
                 latent_dim: int = 100,
                 output_dim: int = 10,
                 hidden_layers: List[int] = [256, 512, 1024],
                 dropout_rate: float = 0.3,
                 use_batch_norm: bool = True,
                 activation: str = "relu",
                 output_activation: str = "tanh",
                 **kwargs):
        """
        Inicializa el generador.
        
        Args:
            latent_dim: Dimensión del vector latente (ruido)
            output_dim: Dimensión de salida (número de características)
            hidden_layers: Lista con número de neuronas por capa oculta
            dropout_rate: Tasa de dropout
            use_batch_norm: Si usar batch normalization
            activation: Función de activación
            output_activation: Función de activación de salida
        """
        super(Generator, self).__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Construir arquitectura
        self._build_network(activation, output_activation)
        
    def _build_network(self, activation: str, output_activation: str):
        """Construye la red neuronal del generador."""
        self.layers_list = []
        
        # Primera capa: de latente a primera capa oculta
        self.layers_list.append(
            layers.Dense(
                self.hidden_layers[0],
                activation=activation,
                input_shape=(self.latent_dim,),
                name='generator_dense_0'
            )
        )
        
        if self.use_batch_norm:
            self.layers_list.append(
                layers.BatchNormalization(name='generator_bn_0')
            )
        
        self.layers_list.append(
            layers.Dropout(self.dropout_rate, name='generator_dropout_0')
        )
        
        # Capas ocultas
        for i, units in enumerate(self.hidden_layers[1:], 1):
            self.layers_list.append(
                layers.Dense(
                    units,
                    activation=activation,
                    name=f'generator_dense_{i}'
                )
            )
            
            if self.use_batch_norm:
                self.layers_list.append(
                    layers.BatchNormalization(name=f'generator_bn_{i}')
                )
            
            self.layers_list.append(
                layers.Dropout(self.dropout_rate, name=f'generator_dropout_{i}')
            )
        
        # Capa de salida
        self.layers_list.append(
            layers.Dense(
                self.output_dim,
                activation=output_activation,
                name='generator_output'
            )
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass del generador.
        
        Args:
            inputs: Tensor de ruido latente
            training: Si está en modo entrenamiento
            
        Returns:
            Datos sintéticos generados
        """
        x = inputs
        
        for layer in self.layers_list:
            x = layer(x, training=training)
        
        return x
    
    def generate_samples(self, 
                        num_samples: int, 
                        random_seed: Optional[int] = None) -> np.ndarray:
        """
        Genera muestras sintéticas.
        
        Args:
            num_samples: Número de muestras a generar
            random_seed: Semilla para reproducibilidad
            
        Returns:
            Array con muestras sintéticas
        """
        try:
            if random_seed is not None:
                tf.random.set_seed(random_seed)
                np.random.seed(random_seed)
            
            # Generar ruido latente (usar 100 como default)
            latent_dim = getattr(self, 'latent_dim', 100)
            noise = tf.random.normal((num_samples, latent_dim))
            
            # DEBUG: Verificar dimensiones
            print(f"   DEBUG GENERATOR - num_samples: {num_samples}")
            print(f"   DEBUG GENERATOR - latent_dim: {latent_dim}")
            print(f"   DEBUG GENERATOR - noise.shape: {noise.shape}")
            print(f"   DEBUG GENERATOR - output_dim: {self.output_dim}")
            
            # Generar muestras
            synthetic_data = self(noise, training=False)
            
            print(f"   DEBUG GENERATOR - synthetic_data.shape: {synthetic_data.shape}")
            return synthetic_data.numpy()
        except Exception as e:
            print(f"Error generando muestras: {e}")
            print(f"   DEBUG GENERATOR - Exception details: {type(e).__name__}: {str(e)}")
            # Fallback: generar datos aleatorios
            output_dim = self.output_dim
            return np.random.normal(0, 1, (num_samples, output_dim))
    
    def get_model_summary(self) -> str:
        """Retorna un resumen de la arquitectura del modelo."""
        return self.summary()
    
    def save_weights(self, filepath: str):
        """Guarda los pesos del modelo."""
        self.save_weights(filepath)
    
    def load_weights(self, filepath: str):
        """Carga los pesos del modelo."""
        self.load_weights(filepath)


class ConditionalGenerator(Generator):
    """
    Generador condicional que puede generar datos para clases específicas.
    """
    
    def __init__(self, 
                 num_classes: int,
                 class_embedding_dim: int = 50,
                 **kwargs):
        """
        Inicializa el generador condicional.
        
        Args:
            num_classes: Número de clases
            class_embedding_dim: Dimensión del embedding de clase
            **kwargs: Argumentos del generador base
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
        
        # Ajustar dimensión de entrada (latente + embedding de clase)
        self.adjusted_latent_dim = self.latent_dim + class_embedding_dim
    
    def call(self, inputs, training=None):
        """
        Forward pass del generador condicional.
        
        Args:
            inputs: Tupla (noise, class_labels)
            training: Si está en modo entrenamiento
        """
        noise, class_labels = inputs
        
        # Obtener embeddings de clase
        class_embeddings = self.class_embedding(class_labels)
        
        # Concatenar ruido con embeddings de clase
        combined_input = tf.concat([noise, class_embeddings], axis=1)
        
        # Pasar por la red
        x = combined_input
        for layer in self.layers_list:
            x = layer(x, training=training)
        
        return x
    
    def generate_samples(self, 
                        num_samples: int,
                        class_labels: Optional[np.ndarray] = None,
                        random_seed: Optional[int] = None) -> np.ndarray:
        """
        Genera muestras sintéticas condicionales.
        
        Args:
            num_samples: Número de muestras a generar
            class_labels: Etiquetas de clase (opcional)
            random_seed: Semilla para reproducibilidad
        """
        if random_seed is not None:
            tf.random.set_seed(random_seed)
            np.random.seed(random_seed)
        
        # Generar ruido latente
        noise = tf.random.normal((num_samples, self.latent_dim))
        
        # Generar etiquetas aleatorias si no se proporcionan
        if class_labels is None:
            class_labels = tf.random.uniform(
                (num_samples,), 
                0, 
                self.num_classes, 
                dtype=tf.int32
            )
        else:
            class_labels = tf.constant(class_labels, dtype=tf.int32)
        
        # Generar muestras
        synthetic_data = self((noise, class_labels), training=False)
        
        return synthetic_data.numpy()
