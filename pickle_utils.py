"""
Utilidades para manejo de pickles en el sistema de GAN.
Permite gestionar experimentos guardados, continuar entrenamientos y analizar resultados.
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from src.utils.pickle_manager import PickleManager


class PickleUtils:
    """Utilidades para manejo de pickles del sistema GAN."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Inicializa las utilidades.
        
        Args:
            results_dir: Directorio de resultados
        """
        self.pickle_manager = PickleManager(results_dir)
    
    def list_experiments(self) -> List[Dict]:
        """
        Lista todos los experimentos disponibles.
        
        Returns:
            Lista de experimentos con información
        """
        experiments = self.pickle_manager.list_saved_experiments()
        experiment_info = []
        
        for exp_name in experiments:
            summary = self.pickle_manager.get_experiment_summary(exp_name)
            experiment_info.append(summary)
        
        return experiment_info
    
    def show_experiment_details(self, experiment_name: str):
        """
        Muestra detalles de un experimento.
        
        Args:
            experiment_name: Nombre del experimento
        """
        try:
            summary = self.pickle_manager.get_experiment_summary(experiment_name)
            
            print(f"\n=== DETALLES DEL EXPERIMENTO: {experiment_name} ===")
            print(f"Época más reciente: {summary.get('latest_epoch', 'N/A')}")
            
            print("\nArchivos disponibles:")
            for category, files in summary['files'].items():
                if files:
                    print(f"  {category}:")
                    for file in files:
                        print(f"    - {file}")
            
            # Intentar cargar métricas más recientes
            try:
                metrics = self.pickle_manager.load_metrics(experiment_name)
                print(f"\nMétricas más recientes:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
            except:
                print("\nNo se encontraron métricas")
                
        except Exception as e:
            print(f"Error mostrando detalles: {e}")
    
    def compare_experiments(self, experiment_names: List[str]):
        """
        Compara múltiples experimentos.
        
        Args:
            experiment_names: Lista de nombres de experimentos
        """
        print(f"\n=== COMPARACIÓN DE EXPERIMENTOS ===")
        
        comparison_data = []
        
        for exp_name in experiment_names:
            try:
                summary = self.pickle_manager.get_experiment_summary(exp_name)
                metrics = self.pickle_manager.load_metrics(exp_name)
                
                comparison_data.append({
                    'experiment': exp_name,
                    'latest_epoch': summary.get('latest_epoch', 0),
                    'best_crc1rs': metrics.get('crc1rs_score', 0),
                    'correlation': metrics.get('correlation_score', 0),
                    'similarity': metrics.get('similarity_score', 0)
                })
            except Exception as e:
                print(f"Error cargando {exp_name}: {e}")
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))
        else:
            print("No se pudieron cargar datos para comparación")
    
    def load_synthetic_data(self, experiment_name: str, epoch: int = None) -> np.ndarray:
        """
        Carga datos sintéticos de un experimento.
        
        Args:
            experiment_name: Nombre del experimento
            epoch: Época específica (opcional)
            
        Returns:
            Datos sintéticos
        """
        try:
            data, feature_names = self.pickle_manager.load_synthetic_data(experiment_name, epoch)
            print(f"Datos sintéticos cargados: {data.shape}")
            print(f"Características: {feature_names}")
            return data
        except Exception as e:
            print(f"Error cargando datos sintéticos: {e}")
            return None
    
    def export_experiment(self, experiment_name: str, output_dir: str = "exported_experiments"):
        """
        Exporta un experimento completo a un directorio.
        
        Args:
            experiment_name: Nombre del experimento
            output_dir: Directorio de salida
        """
        try:
            output_path = Path(output_dir) / experiment_name
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Cargar y exportar datos sintéticos
            data, feature_names = self.pickle_manager.load_synthetic_data(experiment_name)
            if data is not None:
                df = pd.DataFrame(data, columns=feature_names)
                df.to_csv(output_path / "synthetic_data.csv", index=False)
            
            # Cargar y exportar métricas
            metrics = self.pickle_manager.load_metrics(experiment_name)
            if metrics:
                metrics_df = pd.DataFrame([metrics])
                metrics_df.to_csv(output_path / "metrics.csv", index=False)
            
            # Exportar configuración
            try:
                _, config = self.pickle_manager.load_training_state()
                with open(output_path / "config.json", 'w') as f:
                    import json
                    json.dump(config, f, indent=2)
            except:
                pass
            
            print(f"Experimento exportado a: {output_path}")
            
        except Exception as e:
            print(f"Error exportando experimento: {e}")
    
    def cleanup_old_experiments(self, keep_last_n: int = 5):
        """
        Limpia experimentos antiguos.
        
        Args:
            keep_last_n: Número de experimentos a mantener
        """
        try:
            self.pickle_manager.cleanup_old_files(keep_last_n)
            print(f"Limpieza completada. Se mantuvieron los últimos {keep_last_n} archivos.")
        except Exception as e:
            print(f"Error en limpieza: {e}")
    
    def resume_training(self, experiment_name: str, new_epochs: int = 50):
        """
        Continúa un entrenamiento interrumpido.
        
        Args:
            experiment_name: Nombre del experimento
            new_epochs: Número de épocas adicionales
        """
        try:
            # Cargar estado del experimento
            trainer_state, current_epoch, config = self.pickle_manager.load_training_state()
            
            print(f"Continuando entrenamiento desde época {current_epoch}")
            print(f"Configuración: {config}")
            
            # Aquí se podría integrar con el trainer para continuar
            print("Para continuar el entrenamiento, usa el método resume_training del GANTrainer")
            
        except Exception as e:
            print(f"Error resumiendo entrenamiento: {e}")


def main():
    """Función principal para la línea de comandos."""
    parser = argparse.ArgumentParser(description="Utilidades para manejo de pickles del sistema GAN")
    parser.add_argument('--results_dir', default='results', help='Directorio de resultados')
    parser.add_argument('--action', choices=['list', 'details', 'compare', 'export', 'cleanup'], 
                       required=True, help='Acción a realizar')
    parser.add_argument('--experiment', help='Nombre del experimento')
    parser.add_argument('--experiments', nargs='+', help='Lista de experimentos para comparar')
    parser.add_argument('--output_dir', default='exported_experiments', help='Directorio de salida')
    parser.add_argument('--keep_last', type=int, default=5, help='Número de archivos a mantener')
    
    args = parser.parse_args()
    
    utils = PickleUtils(args.results_dir)
    
    if args.action == 'list':
        experiments = utils.list_experiments()
        print(f"\n=== EXPERIMENTOS DISPONIBLES ({len(experiments)}) ===")
        for exp in experiments:
            print(f"- {exp['experiment_name']} (época: {exp.get('latest_epoch', 'N/A')})")
    
    elif args.action == 'details':
        if not args.experiment:
            print("Error: Se requiere --experiment para mostrar detalles")
            return
        utils.show_experiment_details(args.experiment)
    
    elif args.action == 'compare':
        if not args.experiments:
            print("Error: Se requiere --experiments para comparar")
            return
        utils.compare_experiments(args.experiments)
    
    elif args.action == 'export':
        if not args.experiment:
            print("Error: Se requiere --experiment para exportar")
            return
        utils.export_experiment(args.experiment, args.output_dir)
    
    elif args.action == 'cleanup':
        utils.cleanup_old_experiments(args.keep_last)


if __name__ == "__main__":
    main()
