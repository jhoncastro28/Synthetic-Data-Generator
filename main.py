#!/usr/bin/env python3
"""
Generador de Datos Sintéticos con GAN
Aplicación principal - Completamente funcional
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib ANTES de cualquier otra importación relacionada
import matplotlib
matplotlib.use('Agg')  # Backend no-interactivo

# Agregar src al path
sys.path.append('src')

from src.utils.data_loader import DataLoader
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.models.gan import GAN
from src.training.trainer import GANTrainer
from src.utils.metrics import SyntheticDataEvaluator


def print_header(text):
    """Imprime un encabezado formateado."""
    print("\n" + "=" * 60)
    print(text.center(60))
    print("=" * 60 + "\n")


def print_step(step_num, title):
    """Imprime el número de paso."""
    print(f"\n{'=' * 60}")
    print(f"PASO {step_num}: {title}")
    print("=" * 60 + "\n")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Generador de Datos Sintéticos con GAN')
    parser.add_argument('--data', type=str, required=True, help='Archivo CSV con los datos')
    parser.add_argument('--target', type=str, help='Columna objetivo (opcional)')
    parser.add_argument('--output', type=str, default='results', help='Directorio de salida')
    parser.add_argument('--samples', type=int, default=1000, help='Número de muestras sintéticas')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Archivo de configuración')
    parser.add_argument('--seed', type=int, default=42, help='Semilla para reproducibilidad')
    
    args = parser.parse_args()
    
    # Encabezado
    print_header("GENERADOR DE DATOS SINTÉTICOS CON GAN")
    
    # Semillas globales
    try:
        import tensorflow as tf
        tf.random.set_seed(args.seed)
    except Exception as e:
        print(f"Advertencia: No se pudo configurar seed de TensorFlow: {e}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Crear directorios
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    try:
        # ========== PASO 1: CARGAR DATOS ==========
        print_step(1, "CARGA DE DATOS")
        
        print(f"Cargando datos desde: {args.data}")
        data_loader = DataLoader(args.config)
        data = data_loader.load_data(args.data, target_column=args.target)
        print(f"✓ Datos cargados: {data.shape}")
        print(f"  Columnas: {list(data.columns)}\n")
        
        # ========== PASO 2: PREPROCESAR DATOS ==========
        print_step(2, "PREPROCESAMIENTO DE DATOS")
        
        print("Preprocesando datos...")
        train_data, val_data = data_loader.preprocess_data(data, normalize=True, balance_classes=True)
        
        print(f"✓ Preprocesamiento completado:")
        print(f"  • Entrenamiento: {train_data.shape}")
        print(f"  • Validación: {val_data.shape}")
        
        # Guardar conjunto de validación
        val_path = os.path.join(args.output, 'validation_set.csv')
        val_data.to_csv(val_path, index=False)
        print(f"  • Validación guardada: {val_path}\n")
        
        # ========== PASO 3: CREAR MODELOS ==========
        print_step(3, "CREACIÓN DEL MODELO GAN")
        
        print("Creando arquitectura GAN...")
        feature_names = [col for col in train_data.columns if col != args.target] if args.target else train_data.columns.tolist()
        n_features = len(feature_names)
        
        print(f"  • Número de características: {n_features}")
        
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
        
        print("✓ Modelos creados:")
        print("  • Generador: Listo")
        print("  • Discriminador: Listo")
        print("  • GAN: Compilado\n")
        
        # ========== PASO 4: ENTRENAR ==========
        print_step(4, "ENTRENAMIENTO")
        
        X_train = train_data[feature_names].values
        X_val = val_data[feature_names].values
        
        # Crear nombre de experimento
        experiment_name = f"gan_{Path(args.data).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trainer = GANTrainer(
            gan_model=gan,
            config_path=args.config,
            results_dir=args.output,
            use_tensorboard=False,
            use_wandb=False
        )
        
        # Guardar preprocesadores
        trainer.save_preprocessors(data_loader, experiment_name)
        
        # Entrenar
        results = trainer.train(
            train_data=X_train,
            validation_data=X_val,
            feature_names=feature_names,
            experiment_name=experiment_name
        )
        
        print(f"\n✓ Entrenamiento completado")
        print(f"  • Tiempo: {results.get('training_time', 'N/A')}")
        print(f"  • Mejor CrC1RS: {results['best_crc1rs_score']:.4f}\n")
        
        # ========== PASO 5: GENERAR DATOS SINTÉTICOS ==========
        print_step(5, "GENERACIÓN DE DATOS SINTÉTICOS")
        
        print(f"Generando {args.samples} muestras sintéticas...")
        synthetic_data = trainer.generate_synthetic_data(num_samples=args.samples)
        
        # Guardar datos sintéticos
        synthetic_df = pd.DataFrame(synthetic_data, columns=feature_names)
        output_file = os.path.join(args.output, 'synthetic_data.csv')
        synthetic_df.to_csv(output_file, index=False)
        print(f"✓ Datos sintéticos guardados: {output_file}\n")
        
        # ========== PASO 6: EVALUAR CALIDAD ==========
        print_step(6, "EVALUACIÓN DE CALIDAD")
        
        print("Evaluando calidad de datos sintéticos...")
        
        # Alinear tamaños para evaluación
        n = min(X_val.shape[0], synthetic_data.shape[0])
        evaluator = SyntheticDataEvaluator(X_val[:n], synthetic_data[:n])
        metrics = evaluator.calculate_all_metrics()
        
        print(f"✓ Métricas calculadas:")
        print(f"  • CrC1RS Score: {metrics.get('crc1rs_score', 0):.4f}")
        print(f"  • Correlación: {metrics.get('correlation_score', 0):.4f}")
        print(f"  • Similitud: {metrics.get('similarity_score', 0):.4f}\n")
        
        # Guardar resultados completos
        final_epoch = results.get('final_epoch', 0)
        trainer.save_experiment_results(
            synthetic_data, feature_names, metrics, 
            final_epoch, experiment_name
        )
        
        # Generar reporte
        report_path = os.path.join(args.output, 'evaluation_report.txt')
        evaluator.generate_evaluation_report(report_path)
        print(f"✓ Reporte guardado: {report_path}")
        
        # Generar visualizaciones
        plot_path = os.path.join(args.output, 'plots', 'comparison_plots.png')
        evaluator.plot_comparison(feature_names, plot_path)
        print(f"✓ Visualizaciones guardadas: {plot_path}\n")
        
        # ========== RESUMEN FINAL ==========
        print_header("PROCESO COMPLETADO EXITOSAMENTE")
        
        print("ARCHIVOS GENERADOS:")
        print(f"  • Datos sintéticos: {output_file}")
        print(f"  • Reporte: {report_path}")
        print(f"  • Gráficas: {plot_path}")
        print(f"  • Modelos: {args.output}/models/")
        print(f"  • Preprocesadores: {args.output}/preprocessors/\n")
        
        # Interpretación
        crc1rs_score = metrics.get('crc1rs_score', 0)
        print("INTERPRETACIÓN DE RESULTADOS:")
        if crc1rs_score >= 0.8:
            print(f"  ★★★★★ EXCELENTE (CrC1RS: {crc1rs_score:.4f})")
            print("  Los datos sintéticos son de muy alta calidad.")
        elif crc1rs_score >= 0.6:
            print(f"  ★★★★☆ BUENO (CrC1RS: {crc1rs_score:.4f})")
            print("  Los datos sintéticos son de buena calidad.")
        elif crc1rs_score >= 0.4:
            print(f"  ★★★☆☆ REGULAR (CrC1RS: {crc1rs_score:.4f})")
            print("  Los datos sintéticos necesitan mejoras.")
        else:
            print(f"  ★★☆☆☆ MEJORABLE (CrC1RS: {crc1rs_score:.4f})")
            print("  Los datos sintéticos requieren más entrenamiento.")
        
        print("\n" + "=" * 60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nRevisa que:")
        print("  • El archivo de datos existe y es accesible")
        print("  • El archivo de configuración es válido")
        print("  • Hay suficiente espacio en disco")
        print("  • Las dependencias están instaladas correctamente")
        return 1


if __name__ == "__main__":
    exit(main())