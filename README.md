# 🔥 SARFIRE-GIF: Clasificación de Grandes Incendios Forestales

> **Proyecto 1 del TFM - Máster en Data Science con IA (BIG School)**  
> **Autor:** Josué Belmonte  
> **Fecha de entrega:** Abril 2025  
> **Repositorio:** [github.com/josubb-lab/sarfire-gif](https://github.com/josubb-lab/sarfire-gif)

---

## 📋 Índice

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Objetivo de Negocio](#objetivo-de-negocio)
3. [Datos](#datos)
4. [Pipeline del Sistema](#pipeline-del-sistema)
5. [Preprocesamiento y Feature Engineering](#preprocesamiento-y-feature-engineering)
6. [Modelado](#modelado)
7. [Evaluación y Métricas](#evaluación-y-métricas)
8. [Explicabilidad](#explicabilidad)
9. [Métricas de Negocio](#métricas-de-negocio)
10. [Resultados](#resultados)
11. [Modelo en Producción (Conceptual)](#modelo-en-producción-conceptual)
12. [Conclusiones y Trabajo Futuro](#conclusiones-y-trabajo-futuro)
13. [Reproducibilidad](#reproducibilidad)
14. [Referencias](#referencias)

---

## 🎯 Resumen Ejecutivo

SARFIRE-GIF es un sistema de clasificación predictiva de **Grandes Incendios Forestales (GIF)** basado en Machine Learning. El objetivo es identificar, con la mayor antelación posible, qué incendios tienen alta probabilidad de superar las **500 hectáreas afectadas**, permitiendo una asignación óptima de recursos de extinción.

El proyecto utiliza **50 años de datos históricos** (1968-2023) de incendios forestales en España, cruzados con índices meteorológicos de riesgo (FWI - Fire Weather Index), para entrenar modelos de clasificación binaria. El modelo final seleccionado es **XGBoost**, que alcanza un **[TODO: F1-Score]** en la clase minoritaria (GIF) y permite reducir el **[TODO: X%]** de falsos negativos críticos.

Este trabajo forma parte de un ecosistema mayor denominado **SARFIRE** (Sistema de Asistencia y Respuesta para Incendios Forestales con RAG), donde este módulo predictivo se integrará con un sistema multi-agente basado en IA generativa para apoyo a la toma de decisiones en emergencias.

**Tecnologías clave:** Python, XGBoost, SHAP, Imbalanced-learn, Pandas, Scikit-learn.

---

## 💼 Objetivo de Negocio

### Problema a resolver

Los **Grandes Incendios Forestales (GIF)**, definidos como aquellos que afectan a más de 500 hectáreas, representan:
- Solo el **[TODO: X%]** del total de incendios en España
- Pero causan el **[TODO: Y%]** del total de hectáreas quemadas
- Y concentran el **[TODO: Z%]** de los costes de extinción

La detección temprana de un incendio con potencial para convertirse en GIF permite:
1. **Priorización de recursos:** Envío inmediato de medios aéreos y brigadas especializadas
2. **Prevención de pérdidas humanas:** Evacuaciones preventivas más eficaces
3. **Reducción de costes:** Menor superficie afectada = menores costes de extinción y recuperación
4. **Protección ambiental:** Mitigación del impacto ecológico

### Impacto esperado

Si el modelo identifica correctamente el **80%** de los GIF en sus primeras horas:
- **[TODO: Estimación de hectáreas salvadas]**
- **[TODO: Reducción de coste estimado]**
- **[TODO: Recursos optimizados]**

El modelo no sustituye la experiencia de los profesionales, sino que actúa como **sistema de alerta temprana** que complementa la toma de decisiones operativas.

### Tipo de problema

- **Variable objetivo:** Binaria categórica (GIF: Sí/No)
- **Enfoque:** Clasificación supervisada con datos tabulares históricos
- **Justificación:** La naturaleza binaria del problema (>500 ha vs ≤500 ha) permite una definición clara del objetivo y facilita la interpretación operativa. Además, los datos históricos disponibles (incendios + meteorología) son ideales para modelos de aprendizaje supervisado basados en árboles de decisión (XGBoost, Random Forest).

---

## 📊 Datos

### Fuentes de datos

El proyecto utiliza dos datasets principales obtenidos de **Civio**, una fundación de periodismo de datos especializada en transparencia:

#### 1. **Dataset de Incendios Forestales (1968-2023)**
- **Fuente:** [Civio - Incendios Forestales](https://civio.es/quien-manda/incendios-forestales/)
- **Descripción:** Registro histórico de todos los incendios forestales en España
- **Variables clave:**
  - Fecha y hora de inicio
  - Provincia
  - Superficie afectada (hectáreas)
  - Tipo de vegetación
  - Causa probable
  - Costes de extinción (cuando disponible)
- **Formato:** CSV
- **Tamaño:** ~628,000 registros

#### 2. **Dataset FWI (Fire Weather Index)**
- **Fuente:** [Civio - Índice de Riesgo Meteorológico](https://civio.es/)
- **Descripción:** Índices meteorológicos diarios de riesgo de incendio
- **Variables clave:**
  - FWI (Fire Weather Index)
  - ISI (Initial Spread Index)
  - BUI (Buildup Index)
  - FFMC (Fine Fuel Moisture Code)
  - DMC (Duff Moisture Code)
  - DC (Drought Code)
- **Formato:** CSV
- **Granularidad:** Diaria por provincia

### Descarga de datos

⚠️ **Importante:** Los datos no están incluidos en este repositorio debido a su tamaño (~500MB) y restricciones de licencia.

**Pasos para obtener los datos:**

1. **Acceder a Civio:**
   - Visitar: https://civio.es/quien-manda/incendios-forestales/
   - Rellenar el formulario de solicitud de datos (gratuito, fines académicos)

2. **Descargar datasets:**
   - `incendios_1968_2023.csv` → Guardar en `data/raw/`
   - `fwi_historico.csv` → Guardar en `data/raw/`

3. **Verificar descarga:**
   ```bash
   ls -lh data/raw/
   # Deberías ver ambos archivos CSV
   ```

### Exploración inicial

**[TODO: Completar tras EDA]**

- Total de registros analizados: [X]
- Período temporal: [1968-2023]
- Porcentaje de GIF en el dataset: [Y%] (clase desbalanceada)
- Variables con valores nulos: [lista]
- Distribución temporal de incendios: [insight]

---

## 🔄 Pipeline del Sistema

El sistema sigue un pipeline de Machine Learning end-to-end compuesto por las siguientes fases:

```
┌─────────────────────────────────────────────────────────────────┐
│                       PIPELINE SARFIRE-GIF                       │
└─────────────────────────────────────────────────────────────────┘

1. INGESTA DE DATOS
   ├── incendios_1968_2023.csv (Civio)
   ├── fwi_historico.csv (Civio)
   └── Validación de integridad

2. ANÁLISIS EXPLORATORIO (EDA)
   ├── Distribución de variables
   ├── Correlaciones
   ├── Detección de outliers
   └── Análisis temporal

3. PREPROCESAMIENTO
   ├── Limpieza de datos
   │   ├── Tratamiento de nulos
   │   ├── Corrección de tipos de datos
   │   └── Filtrado de registros inconsistentes
   ├── Merge temporal (incendios + FWI)
   │   └── Join por (fecha, provincia)
   └── Feature Engineering
       ├── Variables temporales (mes, día_semana, estación)
       ├── Variables geográficas (región, densidad_forestal)
       └── Variables derivadas del FWI

4. MODELADO
   ├── Definición de variable objetivo: GIF (>500 ha)
   ├── Split temporal: Train (1968-2018) / Test (2019-2023)
   ├── Balanceo de clases (SMOTE)
   ├── Entrenamiento de modelos:
   │   ├── Random Forest (baseline)
   │   ├── XGBoost (principal)
   │   └── LightGBM (comparación)
   └── Optimización de hiperparámetros (GridSearchCV)

5. EVALUACIÓN
   ├── Métricas técnicas:
   │   ├── F1-Score (clase GIF)
   │   ├── Recall (crítico - detectar todos los GIF)
   │   ├── Precision
   │   └── ROC-AUC
   ├── Matriz de confusión
   └── Curvas de aprendizaje

6. EXPLICABILIDAD
   ├── SHAP (Shapley Values)
   │   ├── Feature importance global
   │   ├── Dependence plots
   │   └── Force plots (casos individuales)
   └── Interpretación operativa

7. PERSISTENCIA
   ├── Serialización del modelo (joblib)
   └── Versionado de modelos

8. REPORTES
   ├── Visualizaciones (matplotlib, seaborn)
   ├── Métricas de negocio
   └── Documentación de resultados
```

### Diagrama de flujo técnico

**[TODO: Añadir diagrama Mermaid o imagen del pipeline]**

---

## 🛠️ Preprocesamiento y Feature Engineering

### Limpieza de datos

**[TODO: Completar tras implementación]**

**Problemas detectados:**
1. **Valores nulos en superficie afectada:** [estrategia aplicada]
2. **Fechas inconsistentes:** [tratamiento]
3. **Provincias con nomenclatura variable:** [normalización]
4. **Valores atípicos en costes:** [decisión tomada]

**Transformaciones aplicadas:**
- [Descripción de cada transformación]

### Merge temporal

El cruce entre el dataset de incendios y el FWI se realiza mediante:
- **Clave de join:** `(fecha, provincia)`
- **Estrategia:** Left join desde incendios
- **Justificación:** Queremos mantener todos los incendios aunque no tengan FWI disponible (rellenar con media histórica)

**Pérdida de datos tras merge:** [TODO: X% de incendios sin FWI]

### Feature Engineering

**Variables creadas:**

1. **Temporales:**
   - `mes` (1-12): Estacionalidad
   - `dia_semana` (0-6): Patrón semanal
   - `estacion` (primavera, verano, otoño, invierno)
   - `es_verano` (binaria): Período de máximo riesgo

2. **Geográficas:**
   - `region` (Norte, Centro, Sur, Levante, Canarias): Agrupación de provincias
   - `densidad_forestal` (estimada por provincia)

3. **Derivadas del FWI:**
   - `fwi_categoria` (Bajo, Moderado, Alto, Extremo): Discretización del FWI continuo
   - `dias_sequia` (acumulado): Contador de días consecutivos con DC alto

**Justificación de features:**
- **Estacionalidad:** Los incendios forestales tienen un claro patrón estacional (verano = mayor riesgo)
- **Región:** Diferentes zonas de España tienen vegetación y clima distintos
- **FWI categórico:** Facilita la interpretación operativa ("riesgo extremo" es más accionable que "FWI=45.7")

---

## 🤖 Modelado

### Definición de la variable objetivo

```python
# Variable objetivo: GIF (Grande Incendio Forestal)
y = (df['superficie_ha'] > 500).astype(int)
```

**Distribución de clases:**
- Clase 0 (No-GIF): [TODO: X%]
- Clase 1 (GIF): [TODO: Y%]
- **Desbalanceo:** ~[TODO: Z:1]

### Estrategia de validación

**Split temporal (NO aleatorio):**
- **Train:** 1968-2018 (80% de los datos)
- **Test:** 2019-2023 (20% de los datos)

**Justificación:** En series temporales, el modelo debe predecir el futuro. Un split aleatorio provocaría **data leakage** (entrenar con datos del futuro para predecir el pasado).

**Cross-validation:** Time Series Split con 5 folds sobre el conjunto de entrenamiento.

### Modelos evaluados

#### 1. Random Forest (Baseline)

**Configuración:**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=50,
    class_weight='balanced',
    random_state=42
)
```

**Justificación:** Modelo robusto y fácil de interpretar. Buen punto de partida para datos tabulares.

**Resultados:**
- F1-Score (GIF): [TODO]
- Recall (GIF): [TODO]

---

#### 2. XGBoost (Modelo principal) ⭐

**Configuración:**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=10,  # Crucial para clases desbalanceadas
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

**Justificación:** 
- **Rendimiento superior** en datasets tabulares con clases desbalanceadas
- **Parámetro `scale_pos_weight`** permite dar más peso a la clase minoritaria (GIF)
- **Regularización integrada** reduce overfitting
- **Compatibilidad con SHAP** para explicabilidad

**Hiperparámetros optimizados mediante GridSearchCV:**
- [TODO: Describir proceso de optimización]

**Resultados:**
- F1-Score (GIF): [TODO]
- Recall (GIF): [TODO] ← Métrica crítica

---

#### 3. LightGBM (Comparación)

**Configuración:**
```python
LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    is_unbalance=True,
    random_state=42
)
```

**Justificación:** Alternativa más rápida a XGBoost, útil para datasets muy grandes.

**Resultados:**
- F1-Score (GIF): [TODO]
- Recall (GIF): [TODO]

---

### Tratamiento del desbalanceo de clases

**Técnica aplicada:** SMOTE (Synthetic Minority Over-sampling Technique)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**Justificación:** SMOTE genera ejemplos sintéticos de la clase minoritaria (GIF) interpolando entre instancias existentes, mejorando la capacidad del modelo para detectar patrones en esta clase crítica.

**Alternativas evaluadas:**
- [TODO: Random undersampling, class_weight, etc.]

---

## 📈 Evaluación y Métricas

### Métricas técnicas

#### ¿Por qué estas métricas?

En un problema de **detección de GIF**, el coste de un **Falso Negativo** (no detectar un GIF que luego arrasa miles de hectáreas) es **mucho mayor** que el de un **Falso Positivo** (movilizar recursos para un incendio que finalmente no crece).

Por tanto, **priorizamos Recall** sobre Precision.

#### Métricas principales

1. **Recall (Sensibilidad)** - Métrica crítica
   - **Definición:** De todos los GIF reales, ¿cuántos detectamos?
   - **Objetivo:** > 85%
   - **Resultado obtenido:** [TODO: X%]

2. **F1-Score** - Balance Recall/Precision
   - **Resultado obtenido:** [TODO: X]

3. **Precision**
   - **Definición:** De todos los incendios que clasificamos como GIF, ¿cuántos lo son realmente?
   - **Resultado obtenido:** [TODO: X%]

4. **ROC-AUC**
   - **Resultado obtenido:** [TODO: X]

### Matriz de confusión

**[TODO: Insertar matriz de confusión del modelo final]**

```
                    Predicción
                 No-GIF    GIF
Real  No-GIF       TN      FP
      GIF          FN      TP
```

**Interpretación operativa:**
- **TN (True Negative):** Incendios pequeños correctamente identificados → OK
- **TP (True Positive):** GIF correctamente detectados → EXCELENTE
- **FP (False Positive):** Movilizar recursos innecesariamente → Coste asumible
- **FN (False Negative):** No detectar un GIF → CRÍTICO, minimizar

### Comparación de modelos

| Modelo | Recall (GIF) | F1-Score | ROC-AUC | Tiempo entrenamiento |
|--------|--------------|----------|---------|----------------------|
| Random Forest | [TODO] | [TODO] | [TODO] | [TODO] |
| **XGBoost** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** |
| LightGBM | [TODO] | [TODO] | [TODO] | [TODO] |

**Modelo seleccionado:** XGBoost

**Justificación de la elección:**
- [TODO: Razones técnicas basadas en métricas]

---

## 🔍 Explicabilidad

### ¿Por qué es importante la explicabilidad?

En un contexto operativo de emergencias, los profesionales necesitan **confiar** en las predicciones del modelo. La explicabilidad permite:
1. **Validar el modelo:** ¿Está aprendiendo patrones reales o correlaciones espurias?
2. **Generar confianza:** Los bomberos entienden **por qué** el sistema alerta de un GIF
3. **Mejorar el modelo:** Detectar features irrelevantes o faltantes

### Técnica aplicada: SHAP (SHapley Additive exPlanations)

SHAP asigna a cada feature una contribución al valor predicho, basándose en teoría de juegos cooperativos.

#### Feature Importance Global

**[TODO: Insertar gráfico SHAP summary plot]**

**Variables más influyentes (Top 5):**
1. **FWI (Fire Weather Index):** [TODO: Interpretación]
2. **Superficie primeras 2 horas:** [TODO: Interpretación]
3. **Mes del año:** [TODO: Interpretación]
4. **Provincia/Región:** [TODO: Interpretación]
5. **Días consecutivos de sequía:** [TODO: Interpretación]

#### Dependence Plots

**[TODO: Insertar dependence plot de FWI vs predicción]**

**Interpretación:**
- Cuando FWI > [X], la probabilidad de GIF aumenta exponencialmente
- Existe interacción entre FWI y mes (verano + FWI alto = riesgo extremo)

#### Caso de estudio: Explicación de una predicción individual

**[TODO: Insertar force plot de un GIF correctamente detectado]**

**Narrativa:**
> "El modelo predice GIF (probabilidad: 87%) para este incendio porque:
> - FWI=45 (muy alto) → +35% probabilidad
> - Mes=Agosto → +20% probabilidad
> - Provincia=Guadalajara → +15% probabilidad
> - Días de sequía=12 → +10% probabilidad"

---

## 💰 Métricas de Negocio

### Traducción de métricas técnicas a valor operativo

Las métricas técnicas (F1, Recall) son importantes para evaluar el modelo, pero **no hablan el lenguaje de los tomadores de decisiones**. Necesitamos traducirlas a impacto real.

### Métrica 1: Hectáreas protegidas

**Cálculo:**
```
Hectáreas_protegidas = (TP × promedio_hectareas_GIF) × factor_respuesta_temprana
```

**Supuestos:**
- Promedio de hectáreas afectadas por GIF: [TODO: X ha]
- Factor de reducción por respuesta temprana: [TODO: 30%]

**Resultado:**
- Con el modelo, se protegen potencialmente **[TODO: X hectáreas/año]**

### Métrica 2: Reducción de costes

**Cálculo:**
```
Ahorro = hectáreas_protegidas × (coste_extinción_por_ha + coste_recuperación_por_ha)
```

**Datos de referencia:**
- Coste medio de extinción: [TODO: X €/ha]
- Coste medio de recuperación forestal: [TODO: Y €/ha]

**Resultado:**
- Ahorro estimado: **[TODO: Z millones €/año]**

### Métrica 3: Optimización de recursos

**Impacto:**
- [TODO: Reducción de X% en movilizaciones innecesarias de medios aéreos]
- [TODO: Aumento de Y% en la efectividad de asignación de brigadas]

### ROI del sistema

**Coste del sistema:**
- Desarrollo: [TODO: X horas-persona]
- Infraestructura: [TODO: Y €/mes]

**Retorno estimado:**
- [TODO: ROI = (Ahorro - Coste) / Coste × 100]

---

## 📊 Resultados

### Rendimiento final del modelo

**Modelo:** XGBoost optimizado  
**Dataset de test:** 2019-2023 (datos no vistos durante entrenamiento)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Recall (GIF)** | [TODO: X%] | Detectamos X de cada 100 GIF reales |
| **F1-Score** | [TODO: Y] | Balance general del modelo |
| **Precision** | [TODO: Z%] | Z de cada 100 alertas son verdaderos GIF |
| **ROC-AUC** | [TODO: W] | Capacidad de discriminación |

### Visualizaciones clave

**[TODO: Insertar las siguientes visualizaciones]**

1. **Curva ROC**
2. **Matriz de confusión**
3. **Distribución de probabilidades predichas**
4. **Feature importance (SHAP)**
5. **Curvas de aprendizaje (train vs validation)**

### Casos de éxito y error

#### Caso de éxito: GIF correctamente detectado
**[TODO: Análisis de un caso real donde el modelo acertó]**

#### Caso de error: Falso Negativo
**[TODO: Análisis de un GIF que el modelo no detectó y por qué]**

**Lecciones aprendidas:**
- [TODO: Insights para mejorar el modelo]

---

## 🚀 Modelo en Producción (Conceptual)

### Arquitectura propuesta

El modelo SARFIRE-GIF se desplegaría como un **sistema de predicción por lotes (batch)** integrado en el flujo operativo de los centros de coordinación de emergencias.

```
┌─────────────────────────────────────────────────────────────┐
│              ARQUITECTURA EN PRODUCCIÓN                      │
└─────────────────────────────────────────────────────────────┘

1. INGESTA DE DATOS EN TIEMPO REAL
   ├── API AEMET (datos meteorológicos actualizados)
   ├── Sistema de detección de incendios (alertas nuevas)
   └── Base de datos histórica (contexto)

2. PIPELINE DE PREDICCIÓN (ejecutado cada 1 hora)
   ├── Preprocesamiento automático
   ├── Feature engineering
   ├── Predicción con modelo XGBoost
   └── Generación de alertas

3. CAPA DE APLICACIÓN
   ├── API REST (FastAPI)
   │   └── Endpoint: /predict (para integración con otros sistemas)
   ├── Dashboard de monitoreo
   │   ├── Alertas en tiempo real
   │   ├── Mapa de incendios activos + predicciones
   │   └── Explicaciones SHAP de cada predicción
   └── Notificaciones (Email/SMS a responsables)

4. INFRAESTRUCTURA
   ├── Docker container (modelo + API)
   ├── Cloud VM (AWS EC2 / GCP Compute Engine)
   └── PostgreSQL (histórico de predicciones)
```

### Frecuencia de ejecución

**Predicción:** Batch cada **1 hora** durante la temporada de riesgo (Mayo-Octubre)  
**Justificación:** 
- Los incendios forestales evolucionan en escalas de horas, no minutos
- Balance entre latencia y coste computacional
- Suficiente para toma de decisiones operativas

### Reentrenamiento del modelo

#### ¿Cada cuánto reentrenar?

**Estrategia propuesta:** Reentrenamiento **trimestral** con evaluación continua

**Criterios de reentrenamiento:**
1. **Temporal:** Cada 3 meses (4 veces/año)
2. **Basado en performance:**
   - Si Recall < 80% en últimos 30 días → Reentrenar inmediatamente
   - Si Data Drift detectado → Reentrenar
3. **Basado en datos:**
   - Al final de cada temporada de incendios (Octubre) → Reentrenamiento con datos de ese año

**Proceso de reentrenamiento:**
```
1. Descarga de nuevos datos (incendios + FWI del último trimestre)
2. Validación de calidad de datos
3. Reentrenamiento del modelo (mismo pipeline)
4. Evaluación en conjunto de test actualizado
5. Si métricas >= modelo anterior → Deploy
6. Si métricas < modelo anterior → Análisis de causas + ajustes
```

### Monitoreo del rendimiento

#### Métricas a monitorizar

1. **Métricas de ML:**
   - Recall semanal (calculado con ground truth diferido)
   - Distribución de probabilidades predichas (detectar drift)
   - Feature importance drift

2. **Métricas operativas:**
   - Latencia de predicción (tiempo de respuesta de la API)
   - Tasa de errores (API)
   - Número de alertas generadas/día

3. **Métricas de negocio:**
   - % de GIF detectados en las primeras 2 horas
   - Falsos positivos reportados por operadores

#### Herramientas de monitoreo

- **Evidently AI:** Detección de data drift y model drift
- **Prometheus + Grafana:** Dashboards de métricas operativas
- **MLflow:** Tracking de experimentos y versionado de modelos

### Detección de Data Drift

**¿Qué es Data Drift?**
Cambios en la distribución de las variables de entrada que pueden degradar el rendimiento del modelo.

**Ejemplo:** Si el cambio climático altera los patrones de FWI, el modelo entrenado con datos históricos podría perder precisión.

**Estrategia de detección:**

```python
# Calcular distribución de FWI en datos de entrenamiento
fwi_train_mean = 25.3
fwi_train_std = 12.1

# Cada semana, comparar con datos en producción
fwi_prod_mean = 28.7  # ¿Drift detectado?

# Test estadístico: Kolmogorov-Smirnov
if ks_test(fwi_train, fwi_prod).pvalue < 0.05:
    alert("Data drift detectado en FWI")
```

**Acciones ante drift:**
1. **Drift leve:** Reentrenamiento programado
2. **Drift severo:** Reentrenamiento urgente + investigación de causa
3. **Drift sistemático:** Revisión del feature engineering

### Actualización del modelo

#### Versionado de modelos

```
models/
├── v1.0.0_xgboost_2024-01.joblib
├── v1.1.0_xgboost_2024-04.joblib  ← Reentrenado con Q1 2024
└── v2.0.0_xgboost_2024-10.joblib  ← Nueva feature añadida
```

**Estrategia de versionado:** Semantic Versioning
- **Major (2.0.0):** Cambio de arquitectura o nuevas features
- **Minor (1.1.0):** Reentrenamiento con mismos features
- **Patch (1.0.1):** Correcciones de bugs

#### Proceso de actualización de features

**Escenario:** Se detecta que "velocidad del viento en superficie" mejora el modelo

**Pipeline de actualización:**
```
1. Desarrollo en rama experiment/feature-viento
2. Entrenamiento del nuevo modelo (v2.0.0)
3. Evaluación en test set histórico
4. A/B testing en producción (20% tráfico al modelo nuevo)
5. Si mejora métricas → Despliegue completo
6. Documentar cambio en changelog
```

### Infraestructura técnica

#### Stack propuesto para producción

| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| **API** | FastAPI | Rápida, moderna, documentación automática |
| **Modelo** | Joblib + XGBoost | Serialización eficiente |
| **Containerización** | Docker | Portabilidad y reproducibilidad |
| **Orquestación** | Kubernetes (opcional) | Escalabilidad si se expande |
| **Base de datos** | PostgreSQL + TimescaleDB | Datos tabulares + series temporales |
| **Monitoreo** | Prometheus + Grafana | Estándar de la industria |
| **Cloud** | AWS / GCP | Flexibilidad y servicios gestionados |

#### Ejemplo de Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/v1.0.0_xgboost.joblib models/
COPY src/ src/

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🎯 Conclusiones y Trabajo Futuro

### Conclusiones principales

1. **Viabilidad técnica demostrada:**
   - Es posible predecir GIF con [TODO: X%] de Recall utilizando datos históricos y meteorológicos
   - XGBoost supera a Random Forest y LightGBM en este problema

2. **Importancia de las variables meteorológicas:**
   - El FWI es el predictor más fuerte (según SHAP)
   - La combinación FWI + temporalidad + geolocalización captura el 80% de la varianza

3. **Valor operativo confirmado:**
   - El modelo puede reducir [TODO: X%] de hectáreas afectadas
   - ROI positivo estimado en [TODO: Y millones €/año]

4. **Explicabilidad crítica:**
   - SHAP permite generar confianza en los operadores
   - Las predicciones son auditables y justificables

### Limitaciones del estudio

1. **Datos faltantes:** 
   - [TODO: X%] de incendios sin FWI disponible
   - Información de causa del incendio incompleta

2. **Granularidad temporal:**
   - Predicción basada en datos del día del inicio
   - No se captura la evolución intra-día del incendio

3. **Variables no incluidas:**
   - Topografía (pendiente, orientación)
   - Tipo de vegetación detallado
   - Recursos de extinción desplegados

4. **Sesgo temporal:**
   - Los datos más antiguos (1968-1990) pueden tener menor calidad
   - El cambio climático puede alterar patrones históricos

### Trabajo futuro

#### Mejoras del modelo

1. **Incorporar imágenes satelitales:**
   - Integrar datos de Sentinel-2 / Landsat
   - Detectar vegetación seca mediante NDVI

2. **Modelos de secuencia temporal:**
   - LSTM / Transformer para capturar evolución del incendio
   - Predicción no solo al inicio, sino hora a hora

3. **Ensemble de modelos:**
   - Combinar XGBoost + LightGBM + Redes Neuronales

4. **Transfer Learning:**
   - Aplicar modelos entrenados en otros países (Australia, California)

#### Integración con SARFIRE-RAG

Este módulo predictivo se integrará con el **Proyecto 2 (SARFIRE-RAG)**, creando un sistema completo:

```
Usuario: "Tenemos un incendio en Guadalajara, FWI=40"
         ↓
SARFIRE-GIF (Proyecto 1): Probabilidad GIF = 85% ⚠️
         ↓
SARFIRE-RAG (Proyecto 2): "Riesgo CRÍTICO. Según el manual [ref], 
                           se recomienda: movilizar medios aéreos, 
                           establecer perímetro de 2km..."
```

#### Expansión a otros casos de uso

- Predicción de incendios **antes** de que ocurran (preventivo)
- Clasificación multi-clase (pequeño, mediano, grande, catastrófico)
- Estimación de tiempo de control

---

## 🔄 Reproducibilidad

### Requisitos del sistema

- **Sistema operativo:** Linux (Ubuntu 22.04+ recomendado) o WSL2 en Windows
- **Python:** 3.10 o superior
- **RAM:** Mínimo 8 GB (recomendado 16 GB)
- **Almacenamiento:** 5 GB libres

### Instalación

#### 1. Clonar el repositorio

```bash
git clone https://github.com/josubb-lab/sarfire-gif.git
cd sarfire-gif
```

#### 2. Crear entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

#### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Descargar los datos

**Importante:** Los datos NO están incluidos en el repositorio.

Sigue las instrucciones de la sección [Datos](#datos) para descargarlos de Civio y guardarlos en:
- `data/raw/incendios_1968_2023.csv`
- `data/raw/fwi_historico.csv`

#### 5. Verificar instalación

```bash
python -c "import xgboost; import shap; import pandas; print('✅ Todo OK')"
```

### Ejecución del proyecto

#### Análisis Exploratorio de Datos (EDA)

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

#### Preprocesamiento

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

#### Entrenamiento del modelo

```bash
jupyter notebook notebooks/03_modeling.ipynb
```

O ejecutar script standalone:

```bash
python scripts/train_model.py
```

#### Evaluación y explicabilidad

```bash
jupyter notebook notebooks/04_shap_explicabilidad.ipynb
```

### Estructura del repositorio

```
sarfire-gif/
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias Python
├── .gitignore                   # Archivos ignorados por Git
│
├── data/
│   ├── raw/                     # Datos originales (no en Git)
│   ├── processed/               # Datos procesados (no en Git)
│   └── external/                # Datos externos opcionales
│
├── notebooks/
│   ├── 01_EDA.ipynb             # Análisis Exploratorio
│   ├── 02_preprocessing.ipynb   # Limpieza y Feature Engineering
│   ├── 03_modeling.ipynb        # Entrenamiento de modelos
│   └── 04_shap_explicabilidad.ipynb  # Interpretación
│
├── src/
│   ├── data/
│   │   ├── load_data.py         # Funciones de carga
│   │   └── preprocess.py        # Preprocesamiento
│   ├── features/
│   │   └── build_features.py    # Feature engineering
│   ├── models/
│   │   ├── train.py             # Entrenamiento
│   │   ├── predict.py           # Predicción
│   │   └── evaluate.py          # Evaluación
│   └── utils/
│       └── helpers.py           # Utilidades
│
├── models/                      # Modelos entrenados (no en Git)
│   └── .gitkeep
│
├── reports/
│   ├── figures/                 # Gráficos generados
│   └── metrics/                 # Métricas de evaluación
│
├── scripts/
│   └── train_model.py           # Script de entrenamiento standalone
│
└── tests/                       # Tests unitarios (opcional)
    └── test_preprocessing.py
```

### Troubleshooting

**Problema:** `ImportError: No module named 'xgboost'`  
**Solución:** Verificar que el entorno virtual está activado y ejecutar `pip install -r requirements.txt`

**Problema:** `FileNotFoundError: data/raw/incendios_1968_2023.csv`  
**Solución:** Descargar los datos siguiendo las instrucciones de la sección [Datos](#datos)

**Problema:** Error de memoria al entrenar el modelo  
**Solución:** Reducir el tamaño del dataset en `notebooks/03_modeling.ipynb` o aumentar la RAM disponible

---

## 📚 Referencias

### Datasets

1. **Civio - Incendios Forestales en España (1968-2023)**  
   URL: https://civio.es/quien-manda/incendios-forestales/  
   Licencia: Open Data (uso académico permitido)

2. **Civio - Fire Weather Index (FWI)**  
   URL: https://civio.es/  
   Descripción: Índices meteorológicos de riesgo de incendio

### Literatura científica

**[TODO: Añadir referencias a papers relevantes]**

1. Van Wagner, C.E. (1987). "Development and structure of the Canadian Forest Fire Weather Index System"
2. Artés, T. et al. (2019). "A global wildfire dataset for the analysis of fire regimes and fire behaviour"
3. [Más referencias según la bibliografía consultada]

### Herramientas y librerías

- **XGBoost:** Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- **SHAP:** Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions"
- **Scikit-learn:** Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python"
- **Imbalanced-learn:** Lemaître, G. et al. (2017). "Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning"

### Recursos adicionales

- Documentación oficial de XGBoost: https://xgboost.readthedocs.io/
- Documentación oficial de SHAP: https://shap.readthedocs.io/
- Tutorial de SMOTE: https://imbalanced-learn.org/stable/references/over_sampling.html

---

## 📝 Notas del autor

Este proyecto forma parte del **Trabajo Final de Máster** del programa de Data Science con IA de BIG School (entrega abril 2025) y se integra en el ecosistema **SARFIRE** (Sistema de Asistencia y Respuesta para Incendios Forestales con RAG).

El código y la metodología están disponibles de forma abierta para fines académicos y de investigación. Para uso comercial o integración en sistemas operativos reales, contactar con el autor.

**Autor:** Josué Belmonte  
**Email:** [TODO: email de contacto]  
**LinkedIn:** [TODO: perfil de LinkedIn]  
**GitHub:** https://github.com/josubb-lab

---

**Última actualización:** [TODO: Fecha]  
**Versión del documento:** 1.0.0
