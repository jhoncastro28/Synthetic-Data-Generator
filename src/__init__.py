"""
Paquete principal del Generador de Datos Sintéticos con GAN.
"""

__version__ = "1.0.0"
__author__ = "Synthetic Data Generator Team"
__description__ = "Sistema completo para generación de datos sintéticos usando GANs"

# Importaciones principales
from .models.generator import Generator, ConditionalGenerator
from .models.discriminator import Discriminator, ConditionalDiscriminator, WassersteinDiscriminator
from .models.gan import GAN, ConditionalGAN
from .utils.data_loader import DataLoader
from .utils.metrics import SyntheticDataEvaluator
from .training.trainer import GANTrainer

__all__ = [
    'Generator',
    'ConditionalGenerator', 
    'Discriminator',
    'ConditionalDiscriminator',
    'WassersteinDiscriminator',
    'GAN',
    'ConditionalGAN',
    'DataLoader',
    'SyntheticDataEvaluator',
    'GANTrainer'
]
