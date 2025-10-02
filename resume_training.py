"""
Script para continuar entrenamientos interrumpidos usando pickle.
Permite reanudar entrenamientos desde el último checkpoint guardado.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from src.training.trainer import GANTrainer
from src.models.gan import GAN
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.utils.data_loader import DataLoader
from src.utils.pickle_manager import PickleManager


def resume_training(experiment_name: str, 
                   data_file: str,
                   target_column: str = None,
                   config_file: str = "configs/config.yaml",
                   results_dir: str = "results",
                   additional_epochs: int = 50):
    """
    Continúa un entrenamiento interrumpido.
    
    Args:
        experiment_name: Nombre del experimento a continuar
        data_file: Archivo de datos original
        target_column: Columna objetivo
        config_file: Archivo de configuración
        results_dir: Directorio de resultados
        additional_epochs: Épocas adicionales a entrenar
    """
    
    print(f"=== CONTINUANDO ENTRENAMIENTO: {experiment_name} ===")
    
    try:
        # Inicializar PickleManager
        pickle_manager = PickleManager(results_dir)
        
        # Cargar estado del entrenamiento
        print("Cargando estado del entrenamiento...")
        trainer_state, current_epoch, config = pickle_manager.load_training_state()
        
        print(f"Estado cargado desde época {current_epoch}")
        print(f"Mejor CrC1RS score: {trainer_state['best_crc1rs_score']:.4f}")
        
        # Cargar datos
        print("Cargando datos...")
        data_loader = DataLoader(config_file)
        data = data_loader.load_data(data_file, target_column=target_column)
        train_data, val_data = data_loader.preprocess_data(data, normalize=True, balance_classes=True)
        
        # Obtener características
        feature_names = [col for col in train_data.columns if col != target_column] if target_column else train_data.columns.tolist()
        X_train = train_data[feature_names].values
        X_val = val_data[feature_names].values
        
        # Recrear modelos
        print("Recreando modelos...")
        n_features = len(feature_names)
        
        generator = Generator(
            latent_dim=100,
            output_dim=n_features,
            hidden_layers=[256, 512, 1024],
            dropout_rate=0.2,
            use_batch_norm=True,
            activation="relu",
            output_activation="tanh"
        )
        
        discriminator = Discriminator(
            input_dim=n_features,
            hidden_layers=[1024, 512, 256],
            dropout_rate=0.3,
            use_batch_norm=True,
            activation="leaky_relu"
        )
        
        gan = GAN(
            generator=generator,
            discriminator=discriminator,
            latent_dim=100,
            use_wasserstein=True
        )
        
        # Crear trainer
        trainer = GANTrainer(
            gan_model=gan,
            results_dir=results_dir,
            use_tensorboard=False,
            use_wandb=False
        )
        
        # Restaurar estado del trainer
        trainer.training_history = trainer_state['training_history']
        trainer.best_crc1rs_score = trainer_state['best_crc1rs_score']
        trainer.best_model_path = trainer_state['best_model_path']
        
        # Restaurar pesos de los modelos
        gan_state = trainer_state['gan_model_state']
        trainer.gan_model.generator.set_weights(gan_state['generator_weights'])
        trainer.gan_model.discriminator.set_weights(gan_state['discriminator_weights'])
        
        print("Estado restaurado exitosamente")
        
        # Continuar entrenamiento
        print(f"Continuando entrenamiento por {additional_epochs} épocas adicionales...")
        
        # Modificar configuración para épocas adicionales
        trainer.config['training']['epochs'] = current_epoch + additional_epochs
        
        results = trainer.train(
            train_data=X_train,
            validation_data=X_val,
            feature_names=feature_names,
            start_epoch=current_epoch + 1,
            experiment_name=experiment_name
        )
        
        print(f"Entrenamiento continuado completado")
        print(f"Nuevo mejor CrC1RS score: {results['best_crc1rs_score']:.4f}")
        
        # Generar datos sintéticos con el modelo mejorado
        print("Generando datos sintéticos...")
        synthetic_data = trainer.generate_synthetic_data(1000)
        
        # Guardar resultados finales
        trainer.save_experiment_results(
            synthetic_data, feature_names, results, 
            results.get('final_epoch', current_epoch + additional_epochs), 
            experiment_name
        )
        
        print("Entrenamiento continuado exitosamente")
        
    except Exception as e:
        print(f"Error continuando entrenamiento: {e}")
        return False
    
    return True


def list_available_experiments(results_dir: str = "results"):
    """
    Lista experimentos disponibles para continuar.
    
    Args:
        results_dir: Directorio de resultados
    """
    try:
        pickle_manager = PickleManager(results_dir)
        experiments = pickle_manager.list_saved_experiments()
        
        print(f"\n=== EXPERIMENTOS DISPONIBLES ({len(experiments)}) ===")
        
        for exp_name in experiments:
            try:
                summary = pickle_manager.get_experiment_summary(exp_name)
                latest_epoch = summary.get('latest_epoch', 'N/A')
                print(f"- {exp_name} (época: {latest_epoch})")
            except:
                print(f"- {exp_name} (información no disponible)")
        
        if not experiments:
            print("No se encontraron experimentos guardados")
            
    except Exception as e:
        print(f"Error listando experimentos: {e}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Continuar entrenamientos interrumpidos")
    parser.add_argument('--experiment', help='Nombre del experimento a continuar')
    parser.add_argument('--data', help='Archivo de datos original')
    parser.add_argument('--target', help='Columna objetivo')
    parser.add_argument('--config', default='configs/config.yaml', help='Archivo de configuración')
    parser.add_argument('--results_dir', default='results', help='Directorio de resultados')
    parser.add_argument('--epochs', type=int, default=50, help='Épocas adicionales a entrenar')
    parser.add_argument('--list', action='store_true', help='Listar experimentos disponibles')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_experiments(args.results_dir)
        return
    
    if not args.experiment:
        print("Error: Se requiere --experiment para continuar entrenamiento")
        print("Usa --list para ver experimentos disponibles")
        return
    
    if not args.data:
        print("Error: Se requiere --data para continuar entrenamiento")
        return
    
    success = resume_training(
        experiment_name=args.experiment,
        data_file=args.data,
        target_column=args.target,
        config_file=args.config,
        results_dir=args.results_dir,
        additional_epochs=args.epochs
    )
    
    if success:
        print("\n✅ Entrenamiento continuado exitosamente")
    else:
        print("\n❌ Error continuando entrenamiento")


if __name__ == "__main__":
    main()
