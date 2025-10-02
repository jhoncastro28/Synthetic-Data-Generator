# Generador de Datos Sintéticos con GAN

Sistema completo para la generación de datos sintéticos utilizando Redes Generativas Adversarias (GANs) con TensorFlow/Keras.

## 📋 Descripción

Este proyecto implementa un sistema robusto de generación de datos sintéticos que:

- Entrena modelos GAN (estándar y Wasserstein) para aprender la distribución de datos reales
- Genera datos sintéticos con alta fidelidad estadística
- Evalúa la calidad mediante múltiples métricas personalizadas (CrC1RS)
- Soporta preprocesamiento automático de datos categóricos y numéricos
- Guarda modelos, preprocesadores y resultados para reproducibilidad
- Genera visualizaciones y reportes detallados

## 🚀 Características

- **Arquitectura GAN Flexible**: Soporta GAN estándar, Wasserstein GAN y GAN condicional
- **Preprocesamiento Automático**: Maneja datos numéricos y categóricos, normalización y balanceo de clases
- **Métricas Avanzadas**: CrC1RS personalizado con correlación, consistencia, robustez y similitud
- **Gestión de Experimentos**: Sistema de pickle para guardar y cargar modelos, preprocesadores y estados de entrenamiento
- **Visualizaciones**: Gráficas comparativas y evolución del entrenamiento
- **Sin Interfaz Gráfica**: Backend no-interactivo para ejecución en servidores

## 📦 Instalación

### Requisitos

- Python 3.8+
- pip

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone <repository-url>
cd Synthetic-Data-Generator

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias Principales

```
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
pyyaml>=5.4.0
```

## 🏗️ Estructura del Proyecto

```
Synthetic-Data-Generator/
├── configs/                    # Archivos de configuración
│   ├── config.yaml            # Configuración principal
│   ├── config_simple.yaml     # Configuración simplificada
│   └── ...
├── data/                      # Datos (no incluidos en repo)
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── results/                   # Resultados generados
│   ├── models/               # Modelos entrenados (.pkl)
│   ├── plots/                # Visualizaciones
│   ├── synthetic_data/       # Datos sintéticos generados
│   ├── preprocessors/        # Scalers y encoders
│   ├── training_state/       # Estados de entrenamiento
│   └── metrics/              # Métricas guardadas
├── src/                      # Código fuente
│   ├── models/
│   │   ├── generator.py      # Modelo generador
│   │   ├── discriminator.py  # Modelo discriminador
│   │   └── gan.py           # Modelo GAN principal
│   ├── training/
│   │   └── trainer.py        # Entrenador del GAN
│   └── utils/
│       ├── data_loader.py    # Carga y preprocesamiento
│       ├── metrics.py        # Sistema de métricas
│       └── pickle_manager.py # Gestión de persistencia
├── main.py                   # Punto de entrada principal
├── resume_training.py        # Continuar entrenamientos
├── pickle_utils.py          # Utilidades para pickles
└── requirements.txt          # Dependencias
```

## 💻 Uso

### Uso Básico

```bash
python main.py --data "datos.csv" --config configs/config.yaml --samples 1000
```

### Parámetros

- `--data`: Archivo CSV con los datos de entrenamiento (requerido)
- `--target`: Columna objetivo para clasificación (opcional)
- `--output`: Directorio para resultados (default: `results`)
- `--samples`: Número de muestras sintéticas a generar (default: 1000)
- `--config`: Archivo de configuración (default: `configs/config.yaml`)
- `--seed`: Semilla para reproducibilidad (default: 42)

### Ejemplos

#### Ejemplo 1: Generación Básica

```bash
python main.py \
  --data "Online Sales Data.csv" \
  --config configs/config_simple.yaml \
  --samples 1000 \
  --seed 42
```

#### Ejemplo 2: Con Columna Objetivo

```bash
python main.py \
  --data "medical_data.csv" \
  --target "diagnosis" \
  --config configs/config_medical.yaml \
  --samples 5000
```

#### Ejemplo 3: Continuar Entrenamiento

```bash
python resume_training.py \
  --experiment "gan_Online_Sales_Data_20251001_191424" \
  --data "Online Sales Data.csv" \
  --epochs 50
```

## ⚙️ Configuración

Edita `configs/config.yaml` para ajustar hiperparámetros:

```yaml
# Configuración de datos
data:
  train_split: 0.8
  validation_split: 0.2
  random_state: 42
  normalize: true
  balance_classes: true

# Configuración del modelo
model:
  latent_dim: 100
  generator_layers: [256, 512, 1024]
  discriminator_layers: [1024, 512, 256]
  dropout_rate: 0.3
  use_batch_norm: true

# Configuración de entrenamiento
training:
  batch_size: 64
  epochs: 1000
  learning_rate_g: 0.0002
  learning_rate_d: 0.0002
  beta_1: 0.5
  beta_2: 0.999
  save_interval: 50
  early_stopping_patience: 100

# Configuración de logging
logging:
  log_interval: 10
  save_plots: true
  use_tensorboard: false
```

## 📊 Métricas de Evaluación

### CrC1RS (Correlation, Consistency, Robustness, Similarity)

Métrica personalizada que combina:

- **Correlación (α=0.1)**: Similitud entre matrices de correlación
- **Consistencia (β=0.3)**: Comparación de estadísticas descriptivas
- **Robustez (γ=0.4)**: Estabilidad ante perturbaciones
- **Similitud (δ=0.2)**: Distancia de Wasserstein entre distribuciones

**Interpretación:**
- ★★★★★ **0.80 - 1.00**: Excelente calidad
- ★★★★☆ **0.60 - 0.79**: Buena calidad
- ★★★☆☆ **0.40 - 0.59**: Calidad regular
- ★★☆☆☆ **0.00 - 0.39**: Necesita mejoras

### Métricas Adicionales

- Distancia de Wasserstein
- Test Kolmogorov-Smirnov
- Divergencia de Jensen-Shannon
- MSE de medias y desviaciones estándar

## 📈 Resultados

Después de ejecutar el programa, encontrarás:

### Datos Sintéticos
- `results/synthetic_data.csv` - Datos generados en formato CSV
- `results/synthetic_data/final_synthetic.csv` - Datos finales del entrenamiento

### Visualizaciones
- `results/plots/final_comparison.png` - Comparación de distribuciones
- `results/plots/training_evolution.png` - Evolución de métricas
- `results/plots/comparison_plots.png` - Gráficas por característica

### Reportes
- `results/evaluation_report.txt` - Reporte completo de evaluación
- `results/training_history.csv` - Historial de entrenamiento

### Modelos
- `results/models/*_generator_epoch_*.pkl` - Generadores guardados
- `results/models/*_discriminator_epoch_*.pkl` - Discriminadores guardados

### Preprocesadores
- `results/preprocessors/*_scaler.pkl` - Escaladores
- `results/preprocessors/*_encoder.pkl` - Codificadores

## 🔧 Utilidades

### Listar Experimentos

```bash
python pickle_utils.py --action list
```

### Ver Detalles de Experimento

```bash
python pickle_utils.py --action details --experiment "gan_experiment_20251001"
```

### Comparar Experimentos

```bash
python pickle_utils.py --action compare --experiments exp1 exp2 exp3
```

### Exportar Experimento

```bash
python pickle_utils.py --action export --experiment "gan_experiment_20251001"
```

### Limpiar Archivos Antiguos

```bash
python pickle_utils.py --action cleanup --keep_last 5
```

## 🐛 Solución de Problemas

### Error: "All arrays must be of the same length"
- **Solución**: Asegúrate de usar la versión corregida de `trainer.py`

### Error: Ventanas de matplotlib se abren
- **Solución**: Verifica que `matplotlib.use('Agg')` esté al inicio de `trainer.py` y `metrics.py`

### Error: Memoria insuficiente
- **Solución**: Reduce `batch_size` en la configuración

### Bajo score CrC1RS
- **Solución**: Aumenta el número de épocas o ajusta hiperparámetros

## 📝 Notas Técnicas

- **Backend de matplotlib**: Configurado como 'Agg' (no-interactivo) para evitar ventanas emergentes
- **Gestión de memoria**: Los modelos se guardan en formato pickle para eficiencia
- **Sincronización de métricas**: Las métricas básicas se registran cada época, las de validación según `log_interval`
- **Early stopping**: Implementado con paciencia configurable

## 🤝 Contribuciones

Este proyecto fue desarrollado como parte del curso de Inteligencia Computacional en la Universidad Pedagógica y Tecnológica de Colombia (UPTC).

---