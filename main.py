#!/usr/bin/env python3
"""
Generador de Datos Sintéticos con GAN
Aplicación principal - Lista para usar
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Agregar src al path
sys.path.append('src')

from src.utils.data_loader import DataLoader
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.models.gan import GAN
from src.training.trainer import GANTrainer
from src.utils.metrics import SyntheticDataEvaluator


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Generador de Datos Sintéticos con GAN')
    parser.add_argument('--data', type=str, required=True, help='Archivo CSV con los datos')
    parser.add_argument('--target', type=str, help='Columna objetivo (opcional)')
    parser.add_argument('--output', type=str, default='results', help='Directorio de salida')
    parser.add_argument('--samples', type=int, default=1000, help='Número de muestras sintéticas')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Archivo de configuración')
    
    args = parser.parse_args()
    
    print("GENERADOR DE DATOS SINTETICOS CON GAN")
    print("=" * 50)
    
    # Crear directorios
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Cargar datos
        print(f"\nCargando datos desde: {args.data}")
        data_loader = DataLoader(args.config)
        data = data_loader.load_data(args.data, target_column=args.target)
        print(f"   Datos cargados: {data.shape}")
        
        # 2. Preprocesar datos
        print("\nPreprocesando datos...")
        train_data, val_data = data_loader.preprocess_data(data, normalize=True, balance_classes=True)
        print(f"   Entrenamiento: {train_data.shape}")
        print(f"   Validacion: {val_data.shape}")
        
        # 3. Crear modelos
        print("\nCreando modelos GAN...")
        feature_names = [col for col in train_data.columns if col != args.target] if args.target else train_data.columns
        n_features = len(feature_names)
        
        generator = Generator(
            latent_dim=100,
            output_dim=n_features,
            hidden_layers=[256, 512, 1024],
            dropout_rate=0.3,
            use_batch_norm=True
        )
        
        discriminator = Discriminator(
            input_dim=n_features,
            hidden_layers=[1024, 512, 256],
            dropout_rate=0.3,
            use_batch_norm=True
        )
        
        gan = GAN(
            generator=generator,
            discriminator=discriminator,
            latent_dim=100,
            use_wasserstein=True
        )
        
        print("   Modelos creados")
        
        # 4. Entrenar
        print("\nEntrenando modelo...")
        X_train = train_data[feature_names].values
        X_val = val_data[feature_names].values
        
        trainer = GANTrainer(
            gan_model=gan,
            results_dir=args.output,
            use_tensorboard=False,
            use_wandb=False
        )
        
        results = trainer.train(
            train_data=X_train,
            validation_data=X_val,
            feature_names=feature_names
        )
        
        print(f"   Entrenamiento completado")
        print(f"   Mejor score CrC1RS: {results['best_crc1rs_score']:.4f}")
        
        # 5. Generar datos sintéticos
        print(f"\nGenerando {args.samples} muestras sinteticas...")
        synthetic_data = trainer.generate_synthetic_data(num_samples=args.samples)
        
        # Guardar datos sintéticos
        synthetic_df = pd.DataFrame(synthetic_data, columns=feature_names)
        output_file = f'{args.output}/synthetic_data.csv'
        synthetic_df.to_csv(output_file, index=False)
        print(f"   Datos sinteticos guardados en: {output_file}")
        
        # 6. Evaluar calidad
        print("\nEvaluando calidad...")
        evaluator = SyntheticDataEvaluator(X_val[:args.samples], synthetic_data)
        metrics = evaluator.calculate_all_metrics()
        
        print(f"   CrC1RS Score: {metrics['crc1rs_score']:.4f}")
        print(f"   Correlacion: {metrics['correlation_score']:.4f}")
        print(f"   Similitud: {metrics['similarity_score']:.4f}")
        
        # Generar reporte
        report_path = f'{args.output}/evaluation_report.txt'
        report = evaluator.generate_evaluation_report(report_path)
        print(f"   Reporte guardado en: {report_path}")
        
        # Generar visualizaciones
        plot_path = f'{args.output}/comparison_plots.png'
        evaluator.plot_comparison(feature_names, plot_path)
        print(f"   Visualizaciones guardadas en: {plot_path}")
        
        print("\n" + "=" * 50)
        print("PROCESO COMPLETADO EXITOSAMENTE")
        print("=" * 50)
        print(f"\nResultados en: {args.output}/")
        print(f"Datos sinteticos: {output_file}")
        print(f"Reporte: {report_path}")
        print(f"Graficas: {plot_path}")
        
        # Interpretación
        crc1rs_score = metrics['crc1rs_score']
        if crc1rs_score >= 0.8:
            print(f"\nRESULTADO: EXCELENTE (CrC1RS: {crc1rs_score:.4f})")
        elif crc1rs_score >= 0.6:
            print(f"\nRESULTADO: BUENO (CrC1RS: {crc1rs_score:.4f})")
        elif crc1rs_score >= 0.4:
            print(f"\nRESULTADO: REGULAR (CrC1RS: {crc1rs_score:.4f})")
        else:
            print(f"\nRESULTADO: NECESITA MEJORAS (CrC1RS: {crc1rs_score:.4f})")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("   Revisa que el archivo de datos existe y tiene el formato correcto")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())