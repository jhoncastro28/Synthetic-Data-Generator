# Generador de Datos Sintéticos con GAN

Sistema completo para la generación de datos sintéticos de alta calidad usando GANs (Generative Adversarial Networks) con división automática 80/20 y métricas de evaluación avanzadas.

## 🚀 Características Principales

- **División Automática 80/20**: División automática de datos para entrenamiento y validación
- **Balanceo de Clases**: Balanceo automático de clases desbalanceadas
- **Métricas CrC1RS**: Sistema de evaluación personalizada con métrica CrC1RS
- **Logging Completo**: Monitoreo con TensorBoard y Weights & Biases
- **Visualizaciones**: Gráficas comparativas y de evolución del entrenamiento
- **Checkpoints**: Sistema de guardado automático y early stopping
- **Sistema de Pickle**: Guardado completo de modelos, preprocesadores y resultados
- **Continuación de Entrenamientos**: Reanudar entrenamientos interrumpidos
- **Gestión de Experimentos**: Herramientas para manejar múltiples experimentos
- **Arquitectura Modular**: Código organizado y reutilizable

## 📁 Estructura del Proyecto

```
Synthetic-Data-Generator/
├── main.py                    # Aplicación principal
├── instalar.py               # Instalación automática
├── datos_ejemplo.csv         # Datos de ejemplo
├── requirements.txt           # Dependencias
├── README.md                 # Documentación
├── configs/                  # Configuraciones
│   ├── config.yaml          # Configuración general
│   ├── config_simple.yaml   # Para principiantes
│   ├── config_financial.yaml # Para datos financieros
│   └── config_medical.yaml  # Para datos médicos
├── src/                      # Código fuente
│   ├── models/              # Modelos GAN
│   │   ├── generator.py     # Generador
│   │   ├── discriminator.py # Discriminador
│   │   └── gan.py          # GAN principal
│   ├── utils/               # Utilidades
│   │   ├── data_loader.py   # Carga y preprocesamiento
│   │   └── metrics.py       # Sistema de métricas
│   └── training/            # Entrenamiento
│       └── trainer.py       # Entrenador principal
├── data/                    # Directorios de datos
│   ├── raw/                # Datos originales
│   ├── processed/          # Datos preprocesados
│   └── synthetic/          # Datos sintéticos
└── results/                 # Resultados
    ├── models/             # Modelos entrenados
    ├── plots/              # Visualizaciones
    ├── logs/               # Logs de entrenamiento
    └── synthetic_data/     # Datos sintéticos por época
```

## 🚀 Uso

### 1. Instalación
```bash
# Instalación automática
python instalar.py

# O instalación manual
pip install -r requirements.txt
```

### 2. Uso Básico
```bash
# Con datos de ejemplo
python main.py --data datos_ejemplo.csv --target target

# Con tus propios datos
python main.py --data tu_archivo.csv --target columna_objetivo
```

### 3. Sistema de Pickle - Gestión de Experimentos

#### Listar Experimentos Guardados
```bash
# Listar todos los experimentos
python pickle_utils.py --action list

# Ver detalles de un experimento
python pickle_utils.py --action details --experiment gan_datos_20241201_143022
```

#### Continuar Entrenamientos Interrumpidos
```bash
# Listar experimentos disponibles
python resume_training.py --list

# Continuar un entrenamiento específico
python resume_training.py --experiment gan_datos_20241201_143022 --data datos_ejemplo.csv --target target --epochs 100
```

#### Comparar Experimentos
```bash
# Comparar múltiples experimentos
python pickle_utils.py --action compare --experiments exp1 exp2 exp3

# Exportar experimento completo
python pickle_utils.py --action export --experiment gan_datos_20241201_143022
```

#### Limpiar Archivos Antiguos
```bash
# Mantener solo los últimos 5 archivos de cada tipo
python pickle_utils.py --action cleanup --keep_last 5
```

### 3. Opciones Avanzadas
```bash
# Más muestras sintéticas
python main.py --data datos.csv --target target --samples 5000

# Configuración específica
python main.py --data datos.csv --target target --config configs/config_simple.yaml

# Directorio de salida personalizado
python main.py --data datos.csv --target target --output mi_resultado
```

### 4. Parámetros Disponibles
- `--data`: Archivo CSV con los datos (requerido)
- `--target`: Columna objetivo para balanceo de clases (opcional)
- `--output`: Directorio de salida (default: results)
- `--samples`: Número de muestras sintéticas a generar (default: 1000)
- `--config`: Archivo de configuración (default: configs/config.yaml)

## ⚙️ Configuraciones Disponibles

### Configuración General (`configs/config.yaml`)
- Para datos tabulares estándar
- Arquitectura balanceada
- Entrenamiento estándar

### Configuración Simple (`configs/config_simple.yaml`)
- Para principiantes
- Entrenamiento rápido (50 épocas)
- Sin TensorBoard
- Arquitectura simplificada

### Configuración Financiera (`configs/config_financial.yaml`)
- Optimizada para datos financieros
- Arquitectura más profunda
- Mayor regularización
- Entrenamiento más conservador

### Configuración Médica (`configs/config_medical.yaml`)
- Para datos médicos sensibles
- Balanceo de clases automático
- Evaluación más estricta
- Entrenamiento muy conservador

## 📊 Métricas de Evaluación

### Métrica CrC1RS Personalizada
- **Correlación (C)**: Preservación de correlaciones entre variables
- **Consistencia (C)**: Consistencia estadística con datos reales
- **Robustez (R)**: Estabilidad ante perturbaciones
- **Similitud (S)**: Similitud general con datos originales

### Métricas Complementarias
- **Wasserstein Distance**: Distancia entre distribuciones
- **Kolmogorov-Smirnov**: Test de igualdad de distribuciones
- **Jensen-Shannon Divergence**: Medida de similitud distribucional
- **MSE Estadístico**: Comparación de medias y varianzas

## 📈 Interpretación de Resultados

### CrC1RS Score
- **≥ 0.8**: ✅ Excelente calidad
- **0.6 - 0.8**: ✅ Buena calidad
- **0.4 - 0.6**: ⚠️ Calidad regular
- **< 0.4**: ❌ Necesita mejoras

### Métricas Complementarias
- **Wasserstein < 0.1**: Distribuciones muy similares
- **KS Statistic < 0.1**: Distribuciones estadísticamente similares
- **Correlación > 0.9**: Correlaciones bien preservadas

## 📊 Resultados Generados

El sistema genera automáticamente:

### Archivos de Salida
- **`results/synthetic_data.csv`**: Datos sintéticos generados
- **`results/evaluation_report.txt`**: Reporte detallado de evaluación
- **`results/comparison_plots.png`**: Visualizaciones comparativas

### Sistema de Pickle - Archivos Guardados
- **`results/models/`**: Modelos entrenados (generador, discriminador)
- **`results/preprocessors/`**: Scaler, encoders y preprocesadores
- **`results/training_state/`**: Estado completo del entrenamiento
- **`results/results/`**: Resultados y métricas de experimentos
- **`results/metrics/`**: Métricas de evaluación por época
- **`results/plots/`**: Gráficas de evolución del entrenamiento

### Logs de Entrenamiento
- **`results/logs/`**: Logs de TensorBoard (si está habilitado)
- **`results/models/`**: Modelos entrenados y checkpoints
- **`results/plots/`**: Gráficas de evolución del entrenamiento

## 🎯 Beneficios del Sistema de Pickle

### ✅ **Ventajas Principales:**

1. **Ahorro de Tiempo**: No re-entrenar desde cero
2. **Reproducibilidad**: Resultados consistentes entre sesiones
3. **Experimentos Iterativos**: Probar diferentes configuraciones
4. **Colaboración**: Compartir modelos entrenados fácilmente
5. **Recuperación**: Continuar entrenamientos interrumpidos
6. **Análisis**: Comparar múltiples experimentos
7. **Portabilidad**: Mover experimentos entre máquinas

### 🔧 **Funcionalidades del Sistema:**

- **Guardado Automático**: Estado completo cada N épocas
- **Carga Inteligente**: Restaurar desde cualquier punto
- **Gestión de Versiones**: Múltiples versiones por experimento
- **Limpieza Automática**: Mantener solo archivos relevantes
- **Exportación**: Extraer experimentos completos
- **Comparación**: Analizar múltiples experimentos

---
