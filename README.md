# Generador de Datos Sintéticos con GAN

Sistema completo para la generación de datos sintéticos de alta calidad usando GANs (Generative Adversarial Networks) con división automática 80/20 y métricas de evaluación avanzadas.

## 🚀 Características Principales

- **División Automática 80/20**: División automática de datos para entrenamiento y validación
- **Balanceo de Clases**: Balanceo automático de clases desbalanceadas
- **Métricas CrC1RS**: Sistema de evaluación personalizada con métrica CrC1RS
- **Logging Completo**: Monitoreo con TensorBoard y Weights & Biases
- **Visualizaciones**: Gráficas comparativas y de evolución del entrenamiento
- **Checkpoints**: Sistema de guardado automático y early stopping
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

### Logs de Entrenamiento
- **`results/logs/`**: Logs de TensorBoard (si está habilitado)
- **`results/models/`**: Modelos entrenados y checkpoints
- **`results/plots/`**: Gráficas de evolución del entrenamiento

## 🎯 Casos de Uso

- Datos financieros
- Datos médicos
- Datos de marketing
- Datos de IoT
- Datos demográficos

---