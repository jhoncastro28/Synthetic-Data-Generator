"""
Sistema de métricas para evaluar la calidad de datos sintéticos.
Incluye métricas estadísticas, distribucionales y CrC1RS personalizada.
"""

import numpy as np
import pandas as pd
from scipy import stats
try:
    from scipy.spatial.distance import wasserstein_distance
except ImportError:
    from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class SyntheticDataEvaluator:
    """
    Evaluador de calidad de datos sintéticos con múltiples métricas.
    """
    
    def __init__(self, real_data: np.ndarray, synthetic_data: np.ndarray):
        """
        Inicializa el evaluador.
        
        Args:
            real_data: Datos reales
            synthetic_data: Datos sintéticos
        """
        self.real_data = np.array(real_data)
        self.synthetic_data = np.array(synthetic_data)
        
        # Validar dimensiones
        if self.real_data.shape[1] != self.synthetic_data.shape[1]:
            raise ValueError("Los datos reales y sintéticos deben tener el mismo número de características")
        
        self.n_features = self.real_data.shape[1]
        self.n_real_samples = self.real_data.shape[0]
        self.n_synthetic_samples = self.synthetic_data.shape[0]
    
    def calculate_statistical_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas estadísticas básicas.
        
        Returns:
            Diccionario con métricas estadísticas
        """
        metrics = {}
        
        # Métricas por característica
        for i in range(self.n_features):
            real_feature = self.real_data[:, i]
            synthetic_feature = self.synthetic_data[:, i]
            
            # Media y desviación estándar
            real_mean = np.mean(real_feature)
            synthetic_mean = np.mean(synthetic_feature)
            real_std = np.std(real_feature)
            synthetic_std = np.std(synthetic_feature)
            
            metrics[f'feature_{i}_mean_diff'] = abs(real_mean - synthetic_mean)
            metrics[f'feature_{i}_std_diff'] = abs(real_std - synthetic_std)
            metrics[f'feature_{i}_mean_ratio'] = synthetic_mean / real_mean if real_mean != 0 else 0
            metrics[f'feature_{i}_std_ratio'] = synthetic_std / real_std if real_std != 0 else 0
        
        # Métricas globales
        metrics['overall_mean_mse'] = mean_squared_error(
            np.mean(self.real_data, axis=0), 
            np.mean(self.synthetic_data, axis=0)
        )
        metrics['overall_std_mse'] = mean_squared_error(
            np.std(self.real_data, axis=0), 
            np.std(self.synthetic_data, axis=0)
        )
        
        return metrics
    
    def calculate_distributional_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de distancia entre distribuciones.
        
        Returns:
            Diccionario con métricas distribucionales
        """
        metrics = {}
        
        # Wasserstein distance por característica
        wasserstein_distances = []
        for i in range(self.n_features):
            wd = wasserstein_distance(
                self.real_data[:, i], 
                self.synthetic_data[:, i]
            )
            wasserstein_distances.append(wd)
            metrics[f'feature_{i}_wasserstein'] = wd
        
        metrics['mean_wasserstein'] = np.mean(wasserstein_distances)
        metrics['max_wasserstein'] = np.max(wasserstein_distances)
        
        # Kolmogorov-Smirnov test
        ks_statistics = []
        for i in range(self.n_features):
            ks_stat, _ = stats.ks_2samp(
                self.real_data[:, i], 
                self.synthetic_data[:, i]
            )
            ks_statistics.append(ks_stat)
            metrics[f'feature_{i}_ks_statistic'] = ks_stat
        
        metrics['mean_ks_statistic'] = np.mean(ks_statistics)
        metrics['max_ks_statistic'] = np.max(ks_statistics)
        
        # Jensen-Shannon divergence
        js_divergences = []
        for i in range(self.n_features):
            js_div = self._jensen_shannon_divergence(
                self.real_data[:, i], 
                self.synthetic_data[:, i]
            )
            js_divergences.append(js_div)
            metrics[f'feature_{i}_js_divergence'] = js_div
        
        metrics['mean_js_divergence'] = np.mean(js_divergences)
        metrics['max_js_divergence'] = np.max(js_divergences)
        
        return metrics
    
    def calculate_crc1rs_metric(self, 
                                alpha: float = 0.1,
                                beta: float = 0.3,
                                gamma: float = 0.4,
                                delta: float = 0.2) -> Dict[str, float]:
        """
        Calcula la métrica CrC1RS personalizada.
        
        Args:
            alpha: Peso para correlaciones
            beta: Peso para consistencia estadística
            gamma: Peso para robustez
            delta: Peso para similitud
            
        Returns:
            Diccionario con métricas CrC1RS
        """
        metrics = {}
        
        # 1. Correlaciones (Correlation)
        correlation_score = self._calculate_correlation_score()
        metrics['correlation_score'] = correlation_score
        
        # 2. Consistencia estadística (Consistency)
        consistency_score = self._calculate_consistency_score()
        metrics['consistency_score'] = consistency_score
        
        # 3. Robustez (Robustness)
        robustness_score = self._calculate_robustness_score()
        metrics['robustness_score'] = robustness_score
        
        # 4. Similitud (Similarity)
        similarity_score = self._calculate_similarity_score()
        metrics['similarity_score'] = similarity_score
        
        # 5. CrC1RS combinada
        crc1rs_score = (alpha * correlation_score + 
                       beta * consistency_score + 
                       gamma * robustness_score + 
                       delta * similarity_score)
        
        metrics['crc1rs_score'] = crc1rs_score
        
        return metrics
    
    def _calculate_correlation_score(self) -> float:
        """Calcula el score de correlaciones."""
        real_corr = np.corrcoef(self.real_data.T)
        synthetic_corr = np.corrcoef(self.synthetic_data.T)
        
        # Eliminar diagonal
        mask = ~np.eye(real_corr.shape[0], dtype=bool)
        real_corr_flat = real_corr[mask]
        synthetic_corr_flat = synthetic_corr[mask]
        
        # Correlación entre matrices de correlación
        if len(real_corr_flat) > 1:
            corr_correlation = np.corrcoef(real_corr_flat, synthetic_corr_flat)[0, 1]
            return max(0, corr_correlation) if not np.isnan(corr_correlation) else 0
        return 0
    
    def _calculate_consistency_score(self) -> float:
        """Calcula el score de consistencia estadística."""
        # Comparar estadísticas descriptivas
        real_stats = self._calculate_descriptive_stats(self.real_data)
        synthetic_stats = self._calculate_descriptive_stats(self.synthetic_data)
        
        # Calcular similitud en estadísticas
        consistency_scores = []
        for stat_name in real_stats.keys():
            real_values = real_stats[stat_name]
            synthetic_values = synthetic_stats[stat_name]
            
            # Normalizar para comparar
            max_val = max(np.max(np.abs(real_values)), np.max(np.abs(synthetic_values)))
            if max_val > 0:
                normalized_diff = np.mean(np.abs(real_values - synthetic_values)) / max_val
                consistency_scores.append(1 - normalized_diff)
        
        return np.mean(consistency_scores) if consistency_scores else 0
    
    def _calculate_robustness_score(self) -> float:
        """Calcula el score de robustez."""
        # Evaluar estabilidad ante pequeñas perturbaciones
        noise_levels = [0.01, 0.05, 0.1]
        robustness_scores = []
        
        for noise_level in noise_levels:
            # Añadir ruido a datos sintéticos
            noisy_synthetic = self.synthetic_data + np.random.normal(0, noise_level, self.synthetic_data.shape)
            
            # Calcular similitud con datos reales
            similarity = self._calculate_similarity_score(self.real_data, noisy_synthetic)
            robustness_scores.append(similarity)
        
        return np.mean(robustness_scores)
    
    def _calculate_similarity_score(self, data1: Optional[np.ndarray] = None, 
                                   data2: Optional[np.ndarray] = None) -> float:
        """Calcula el score de similitud."""
        if data1 is None:
            data1 = self.real_data
        if data2 is None:
            data2 = self.synthetic_data
        
        # Usar distancia de Wasserstein promedio
        wasserstein_distances = []
        for i in range(data1.shape[1]):
            wd = wasserstein_distance(data1[:, i], data2[:, i])
            wasserstein_distances.append(wd)
        
        mean_wasserstein = np.mean(wasserstein_distances)
        
        # Convertir a score de similitud (0-1, donde 1 es más similar)
        similarity_score = 1 / (1 + mean_wasserstein)
        return similarity_score
    
    def _calculate_descriptive_stats(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Calcula estadísticas descriptivas."""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'median': np.median(data, axis=0),
            'q25': np.percentile(data, 25, axis=0),
            'q75': np.percentile(data, 75, axis=0)
        }
    
    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray, bins: int = 50) -> float:
        """Calcula la divergencia de Jensen-Shannon."""
        # Crear histogramas
        p_hist, _ = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bins, density=True)
        
        # Normalizar
        p_hist = p_hist / np.sum(p_hist)
        q_hist = q_hist / np.sum(q_hist)
        
        # Evitar ceros
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10
        
        # Calcular JS divergence
        m = 0.5 * (p_hist + q_hist)
        js_div = 0.5 * stats.entropy(p_hist, m) + 0.5 * stats.entropy(q_hist, m)
        
        return js_div
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calcula todas las métricas disponibles.
        
        Returns:
            Diccionario con todas las métricas
        """
        all_metrics = {}
        
        try:
            # Métricas estadísticas
            all_metrics.update(self.calculate_statistical_metrics())
        except Exception as e:
            print(f"Error calculando métricas estadísticas: {e}")
            all_metrics['correlation_score'] = 0.0
            all_metrics['similarity_score'] = 0.0
        
        try:
            # Métricas distribucionales
            all_metrics.update(self.calculate_distributional_metrics())
        except Exception as e:
            print(f"Error calculando métricas distribucionales: {e}")
            all_metrics['wasserstein_distance'] = 1.0
            all_metrics['ks_statistic'] = 1.0
            all_metrics['js_divergence'] = 1.0
        
        try:
            # Métricas CrC1RS
            all_metrics.update(self.calculate_crc1rs_metric())
        except Exception as e:
            print(f"Error calculando CrC1RS: {e}")
            all_metrics['crc1rs_score'] = 0.0
        
        return all_metrics
    
    def generate_evaluation_report(self, save_path: Optional[str] = None) -> str:
        """
        Genera un reporte completo de evaluación.
        
        Args:
            save_path: Ruta para guardar el reporte
            
        Returns:
            Reporte como string
        """
        metrics = self.calculate_all_metrics()
        
        report = "=" * 60 + "\n"
        report += "REPORTE DE EVALUACIÓN DE DATOS SINTÉTICOS\n"
        report += "=" * 60 + "\n\n"
        
        # Resumen ejecutivo
        report += "RESUMEN EJECUTIVO:\n"
        report += f"- Datos reales: {self.n_real_samples} muestras, {self.n_features} características\n"
        report += f"- Datos sintéticos: {self.n_synthetic_samples} muestras, {self.n_features} características\n"
        report += f"- Score CrC1RS: {metrics['crc1rs_score']:.4f}\n"
        report += f"- Correlación promedio: {metrics['correlation_score']:.4f}\n"
        report += f"- Similitud promedio: {metrics['similarity_score']:.4f}\n\n"
        
        # Métricas estadísticas
        report += "MÉTRICAS ESTADÍSTICAS:\n"
        report += f"- MSE de medias: {metrics['overall_mean_mse']:.6f}\n"
        report += f"- MSE de desviaciones estándar: {metrics['overall_std_mse']:.6f}\n\n"
        
        # Métricas distribucionales
        report += "MÉTRICAS DISTRIBUCIONALES:\n"
        report += f"- Distancia Wasserstein promedio: {metrics['mean_wasserstein']:.6f}\n"
        report += f"- Estadístico KS promedio: {metrics['mean_ks_statistic']:.6f}\n"
        report += f"- Divergencia JS promedio: {metrics['mean_js_divergence']:.6f}\n\n"
        
        # Métricas CrC1RS detalladas
        report += "MÉTRICAS CrC1RS DETALLADAS:\n"
        report += f"- Score de correlación: {metrics['correlation_score']:.4f}\n"
        report += f"- Score de consistencia: {metrics['consistency_score']:.4f}\n"
        report += f"- Score de robustez: {metrics['robustness_score']:.4f}\n"
        report += f"- Score de similitud: {metrics['similarity_score']:.4f}\n"
        report += f"- Score CrC1RS total: {metrics['crc1rs_score']:.4f}\n\n"
        
        # Interpretación
        report += "INTERPRETACIÓN:\n"
        if metrics['crc1rs_score'] >= 0.8:
            report += "EXCELENTE: Los datos sintéticos son de muy alta calidad.\n"
        elif metrics['crc1rs_score'] >= 0.6:
            report += "BUENO: Los datos sintéticos son de buena calidad.\n"
        elif metrics['crc1rs_score'] >= 0.4:
            report += "REGULAR: Los datos sintéticos necesitan mejoras.\n"
        else:
            report += "DEFICIENTE: Los datos sintéticos requieren entrenamiento adicional.\n"
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def plot_comparison(self, feature_names: Optional[List[str]] = None, 
                       save_path: Optional[str] = None):
        """
        Genera visualizaciones comparativas.
        
        Args:
            feature_names: Nombres de las características
            save_path: Ruta para guardar las gráficas
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(self.n_features)]
        
        # Crear figura con subplots
        n_cols = min(3, self.n_features)
        n_rows = (self.n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(self.n_features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Histogramas superpuestos
            ax.hist(self.real_data[:, i], alpha=0.7, label='Real', bins=30, density=True)
            ax.hist(self.synthetic_data[:, i], alpha=0.7, label='Sintético', bins=30, density=True)
            ax.set_title(f'{feature_names[i]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Ocultar subplots vacíos
        for i in range(self.n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
