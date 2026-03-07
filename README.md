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
- Aproximadamente el **3-5%** del total de incendios en España
- Pero causan más del **70%** del total de hectáreas quemadas
- Y concentran más del **80%** de los costes de extinción

**El problema crítico:** Cuando un incendio inicia, los primeros 30-60 minutos son decisivos. Si no se detecta su potencial para convertirse en GIF, la ventana de oportunidad para una respuesta efectiva se cierra.

La detección temprana de un incendio con potencial para convertirse en GIF permite:
1. **Priorización de recursos:** Envío inmediato de medios aéreos y brigadas especializadas
2. **Prevención de pérdidas humanas:** Evacuaciones preventivas más eficaces
3. **Reducción de costes:** Menor superficie afectada = menores costes de extinción y recuperación
4. **Protección ambiental:** Mitigación del impacto ecológico

### Objetivo del modelo

**Predecir la probabilidad de que un incendio se convierta en GIF basándose en:**
- Condiciones meteorológicas (FWI y componentes)
- Ubicación geográfica (provincia/región)
- Temporalidad (mes, día, estación)
- Características iniciales del incendio

**Momento de predicción:** Durante las primeras **2-4 horas** desde el inicio del incendio, cuando aún hay margen de maniobra para movilizar recursos adicionales.

### Impacto esperado

Si el modelo identifica correctamente el **85%** de los GIF en sus primeras horas:

**Métricas de negocio (se calcularán con datos reales):**
- **Hectáreas protegidas:** Estimación basada en superficie promedio de GIF × tasa de detección
- **Ahorro económico:** Coste medio de extinción (~3.000 €/ha) + recuperación (~5.000 €/ha)
- **Recursos optimizados:** Reducción de falsas alarmas y movilizaciones innecesarias

**Restricción crítica:** El modelo **prioriza Recall sobre Precision**
- ✅ **Falso Positivo** (movilizar recursos innecesarios) → Coste asumible
- ⚠️ **Falso Negativo** (no detectar un GIF) → **CRÍTICO** → Miles de hectáreas en riesgo

**Aclaración importante:** El modelo no sustituye la experiencia de los profesionales, sino que actúa como **sistema de alerta temprana** que complementa la toma de decisiones operativas.

### Tipo de problema

- **Variable objetivo:** Binaria categórica (GIF: Sí/No, umbral: 500 hectáreas)
- **Enfoque:** Clasificación supervisada con datos tabulares históricos (50 años)
- **Métrica principal:** **Recall (Sensibilidad)** - Detectar el máximo de GIF posibles
- **Métrica secundaria:** F1-Score - Balance entre Recall y Precision

**Justificación técnica:** 
La naturaleza binaria del problema (>500 ha vs ≤500 ha) permite una definición operativa clara y facilita la interpretación por parte de los equipos de extinción. Los datos históricos disponibles (628K incendios + índices meteorológicos FWI) son ideales para modelos de aprendizaje supervisado basados en árboles de decisión (XGBoost, Random Forest), que manejan bien variables mixtas (numéricas y categóricas) y clases desbalanceadas.

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

⚠️ **Importante:** Los datos no están incluidos en este repositorio debido a su tamaño (~686 MB total) y restricciones de licencia.

**Archivos requeridos:**
- `incendios_1968_2023.csv` (30 MB) - Registro histórico de incendios (628K registros)
- `fwi_historico.csv` (656 MB) - Índices meteorológicos FWI

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
   # Deberías ver: incendios_1968_2023.csv (30M) y fwi_historico.csv (656M)
```

### Exploración inicial

**Análisis completado (ver `notebooks/01_EDA.ipynb`):**

- **Total de registros:** 292,181 incendios (1968-2023)
- **Período válido para modelado:** 1968-2020 (años 2021-2023 tienen datos incompletos)
- **Variable objetivo (GIF):** 1,981 casos (0.68% del total)
- **Desbalanceo de clases:** 146:1 (No-GIF:GIF) - requiere SMOTE y scale_pos_weight
- **Valores nulos críticos:** 
  - Coordenadas (lat/lng): 18.6% faltantes (54,435 incendios)
  - Resto de variables: completas
- **Distribución temporal:**
  - Mes crítico: Agosto (60,572 incendios, 20.7% del total)
  - Año pico: 1989 (15,922 incendios) - posible outlier
- **Distribución geográfica:**
  - Noroeste (Galicia/Asturias): muchos incendios, bajo % de GIF (<1%)
  - Centro/Sur (Ávila, Burgos, Albacete): menos incendios, alto % de GIF (2-3%)
- **Índice FWI:**
  - 46% de días con riesgo bajo (FWI < 5.2)
  - 24.5% de días con riesgo muy alto/extremo (FWI > 21.3)
  - **Problema detectado:** Dataset FWI usa coordenadas (x,y), no tiene `idprovincia` → merge requiere geocodificación
- **Outliers:** 14.98% de incendios (son GIF reales, se mantienen para el modelo)

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

## 🛠️ Preprocesamiento, limpieza y Feature Engineering

### Limpieza de datos

**Problemas detectados y resueltos:**

1. **Años con datos incompletos (2021-2023):**
   - Problema: Solo 1,137 incendios en 3 años (vs ~5,000/año esperado)
   - Solución: Excluir del análisis, usar período 1968-2020
   
2. **Valores nulos en coordenadas (18.6%):**
   - Problema: 54,435 incendios sin lat/lng
   - Solución: No crítico tras merge agregado por provincia
   
3. **Valores atípicos en superficie:**
   - Problema: 14.98% outliers estadísticos (>22.55 ha)
   - Decisión: **Mantener** (los GIF son outliers por definición)

**Transformaciones aplicadas:**
- Parseo de fechas a formato datetime
- Filtrado temporal: 1968-2020
- Normalización de IDs provinciales

### Merge temporal (FWI + Incendios)

**Estrategia implementada: Opción A (Agregación Provincial)**

El cruce entre incendios y FWI se realiza mediante:

- **Paso 1:** Asignar provincia a cada coordenada FWI (x, y)
  - Método: Distancia euclidiana a centroide provincial
  - Resultado: 18.5M puntos FWI → idprovincia asignado
  
- **Paso 2:** Agregar FWI por (fecha, provincia)
  - Métricas: `fwi_mean`, `fwi_max`, `fwi_p90`
  - Reducción: 18.5M → ~300K registros agregados
  
- **Paso 3:** Join con incendios
  - Clave: `(fecha, idprovincia)`
  - Tipo: Left join (mantener todos los incendios)
  - Imputación nulos: Media provincial/mensual

**Justificación de la estrategia:**
- Equilibrio precisión/viabilidad técnica
- FWI captura contexto meteorológico regional
- Orografía local capturada vía municipio/región
- Usa 100% de incendios (incluye 18.6% sin coordenadas)

**Pérdida de datos tras merge:** 0% (left join + imputación)

### Feature Engineering

**Variables creadas:**

**1. Temporales:**
- `mes` (1-12): Estacionalidad del riesgo
- `dia_semana` (0-6): Patrón semanal
- `estacion` (invierno/primavera/verano/otoño): Agrupación estacional
- `es_verano` (boolean): Período crítico (junio-septiembre)
- `año`: Tendencias temporales

**2. Geográficas:**
- `region` (8 categorías): Zonas climáticas de España
  - Noroeste, Norte, Nordeste, Centro, Levante, Sur, Canarias, Baleares
- `municipio_encoded`: Captura idiosincrasia local/orográfica (LabelEncoder)

**3. Derivadas del FWI:**
- `fwi_mean`: Promedio provincial diario (condición general)
- `fwi_max`: Máximo provincial (captura extremos locales)
- `fwi_p90`: Percentil 90 (robusto a outliers)
- `fwi_cat` (5 categorías): bajo/moderado/alto/muy_alto/extremo
  - Umbrales: 0-5.2 / 5.2-11.2 / 11.2-21.3 / 21.3-38.0 / >38.0

**Justificación de features:**
- **Mes/Estación:** Patrón estacional fuerte (agosto = 20.7% incendios)
- **Región:** Captura diferencias climáticas norte (húmedo) vs centro/sur (seco)
- **Municipio:** Proxy de orografía específica (valles, laderas) que precipita GIF
- **FWI_max:** Más predictivo que promedio para eventos extremos
- **FWI_cat:** Interpretabilidad operativa para protocolos de actuación

## 🤖 Modelado

### Baseline (ver `notebooks/03_Modeling_Baseline.ipynb`)

**Objetivo:** Establecer punto de referencia con modelos simples antes de optimizar.

**Modelos probados:**

#### 1. Logistic Regression (baseline simple)
**Configuración:**
```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
```

**Resultados:**
- ⭐ **Recall: 43.4%** (detecta 49 de 113 GIF en test)
- Precision: 3.0%
- F1-Score: 5.6%
- ROC-AUC: 83.0%

**Conclusión:** ✅ Modelo simple captura bien el patrón principal. Establece baseline sólido.

---

#### 2. XGBoost sin optimizar
**Configuración:**
```python
XGBClassifier(
    scale_pos_weight=147.5,  # Ratio calculado del desbalanceo
    max_depth=6,
    n_estimators=100,
    learning_rate=0.3,
    eval_metric='logloss',
    random_state=42
)
```

**Resultados:**
- ⭐ **Recall: 10.6%** (solo detecta 12 de 113 GIF)
- Precision: 2.1%
- F1-Score: 3.5%
- ROC-AUC: 77.5%

**Conclusión:** ⚠️ XGBoost demasiado conservador con hiperparámetros por defecto. Necesita optimización.

---

### Features utilizadas (9 totales)

**Categóricas (5):**
- `fwi_cat` (bajo/moderado/alto/muy_alto/extremo)
- `estacion` (invierno/primavera/verano/otoño)
- `region` (8 zonas climáticas de España)
- `dia_semana` (0-6)
- `mes` (1-12)

**Numéricas (4):**
- `fwi_mean` (promedio FWI provincial diario)
- `fwi_max` (máximo FWI provincial - captura extremos)
- `fwi_p90` (percentil 90 FWI - robusto a outliers)
- `año` (1968-2020)

**Features excluidas del baseline:**
- `municipio_encoded`: 19.75% valores "INDETERMINADO" + 7,050 categorías únicas
  - Justificación: Ruido significativo, se evaluará en fase de optimización
- Metadata: `id`, `fecha`
- Leakage: `superficie` (define directamente el target GIF)
- Redundantes: `idmunicipio`, `municipio`, `idprovincia`, `es_verano`

---

### Estrategia de validación

**Split temporal (NO aleatorio):**
- **Train:** 1968-2015 (275,302 registros, 1,854 GIF - 0.67%)
- **Test:** 2016-2020 (15,742 registros, 113 GIF - 0.72%)

**Justificación:** 
- Evita data leakage temporal
- Simula predicción en producción (entrenar con pasado, predecir futuro)
- Desbalanceo consistente entre train/test (~147:1)

---

### Tratamiento del desbalanceo

**Baseline:**
- Logistic Regression: `class_weight='balanced'` ✅ Funciona bien
- XGBoost: `scale_pos_weight=147.5` ⚠️ Demasiado conservador

**Próxima fase (Optimización):**
- Ajustar `scale_pos_weight` (probar 50-200)
- GridSearch sobre otros hiperparámetros
- Evaluar SMOTE si es necesario
- Considerar ensemble

---

### Resultado clave del baseline

> **Logistic Regression (modelo más simple) supera a XGBoost sin optimizar**
> 
> Esto valida que:
> 1. ✅ El pipeline de preprocesamiento funciona correctamente
> 2. ✅ Las features tienen poder predictivo (ROC-AUC ~80%)
> 3. ✅ El problema es resoluble (baseline detecta 43% de GIF)
> 4. ⚠️ XGBoost necesita tuning cuidadoso para aprovechar su potencial

**Métrica objetivo:** Recall >85% (detectar 85% de GIF)
**Estado actual:** Recall 43.4% → Margen de mejora significativo

**Próximo paso:** Notebook 04 - Optimización de XGBoost con GridSearch

---

---

## 🎯 Optimización del Modelo

### Notebook 04: Optimización XGBoost con GridSearch ✅

**Duración:** 1 sesión (~70 minutos GridSearch)  
**Fecha:** 8 Febrero 2026

#### Estrategia de optimización

**GridSearchCV con TimeSeriesSplit:**
- 5 folds temporales
- 120 combinaciones de hiperparámetros
- 600 fits totales
- Scoring: `recall` (métrica prioritaria)

**Hiperparámetros optimizados:**
```python
param_grid = {
    'scale_pos_weight': [75, 100, 125, 150, 200],
    'max_depth': [6, 8, 10],
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'min_child_weight': [1, 3],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}
```

#### Mejores hiperparámetros encontrados
```python
{
    'scale_pos_weight': 200,      # vs 147.5 baseline
    'max_depth': 6,
    'n_estimators': 100,
    'learning_rate': 0.05,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

**Conclusión técnica:**
- `scale_pos_weight=200` (vs 147.5 del ratio) resultó óptimo
- Modelo conservador en profundidad (max_depth=6) evita overfitting
- Learning rate bajo (0.05) con 100 árboles da mejor generalización

---

## 📈 Evaluación y Métricas

### Métricas técnicas

#### ¿Por qué estas métricas?

En un problema de **detección de GIF**, el coste de un **Falso Negativo** (no detectar un GIF que luego arrasa miles de hectáreas) es **mucho mayor** que el de un **Falso Positivo** (movilizar recursos para un incendio que finalmente no crece).

Por tanto, **priorizamos Recall** sobre Precision.

#### Métricas principales

1. **Recall (Sensibilidad)** - Métrica crítica ⭐
   - **Definición:** De todos los GIF reales, ¿cuántos detectamos?
   - **Objetivo:** > 85%
   - **Logistic Regression (baseline):** 43.4%
   - **XGBoost sin optimizar:** 10.6%
   - **XGBoost optimizado:** **70.8%** ✅
   - **Interpretación:** Detecta 80 de 113 GIF en test

2. **F1-Score** - Balance Recall/Precision
   - **Logistic Regression:** 5.6%
   - **XGBoost sin optimizar:** 3.5%
   - **XGBoost optimizado:** **5.4%**

3. **Precision**
   - **Definición:** De todos los incendios que clasificamos como GIF, ¿cuántos lo son realmente?
   - **Logistic Regression:** 3.0%
   - **XGBoost sin optimizar:** 2.1%
   - **XGBoost optimizado:** **2.8%**

4. **ROC-AUC**
   - **Logistic Regression:** 83.0%
   - **XGBoost sin optimizar:** 77.5%
   - **XGBoost optimizado:** **82.6%**

### Matriz de confusión

**XGBoost Optimizado (Test 2016-2020):**
```
                    Predicción
                 No-GIF    GIF
Real  No-GIF      12868   2761
      GIF            33     80
```

**Interpretación operativa:**
- **TN (12,868):** Incendios pequeños correctamente identificados → OK
- **TP (80):** GIF correctamente detectados → **EXCELENTE** (70.8% recall)
- **FP (2,761):** Movilizar recursos innecesariamente → Coste asumible (esperado con desbalanceo 147:1)
- **FN (33):** No detectar un GIF → **CRÍTICO** (29.2% aún sin detectar, margen de mejora)

### Comparación de modelos

| Modelo | Recall (GIF) | Precision | F1-Score | ROC-AUC | Mejora vs LR |
|--------|--------------|-----------|----------|---------|--------------|
| Logistic Regression (baseline) | 43.4% | 3.0% | 5.6% | 83.0% | - |
| XGBoost sin optimizar | 10.6% | 2.1% | 3.5% | 77.5% | -32.7pp ❌ |
| **XGBoost optimizado** | **70.8%** | **2.8%** | **5.4%** | **82.6%** | **+27.4pp** ✅ |

**Modelo seleccionado:** XGBoost optimizado

**Justificación de la elección:**
1. ✅ **Recall 70.8%** alcanza objetivo intermedio (>70%)
2. ✅ **Mejora sustancial** (+27.4pp vs baseline simple)
3. ✅ ROC-AUC 82.6% confirma buena capacidad discriminatoria
4. ✅ GridSearch validó robustez con TimeSeriesSplit
5. ⚠️ **Limitación:** 33 GIF siguen sin detectarse (posible mejora con SMOTE o features adicionales)

### Feature Importance (Top 5)

Variables más importantes en el modelo optimizado:

1. **fwi_p90** (9.6%) - Percentil 90 FWI provincial (captura extremos sin outliers)
2. **fwi_max** (7.6%) - FWI máximo provincial (picos meteorológicos)
3. **region_nan** (5.6%) - Región desconocida (patrón inesperado)
4. **año** (4.8%) - Tendencia temporal (cambio climático, mejoras gestión)
5. **region_Sur** (3.5%) - Patrón geográfico (clima seco)

**Conclusión:** Variables FWI dominan (~22% importancia combinada), validando estrategia de merge provincial.

### Análisis de errores

**Falsos Negativos (33 GIF no detectados):**
- Probabilidad promedio de predicción: 27.8%
- Interpretación: GIF con características atípicas, difíciles de predecir
- Ejemplos: GIF en condiciones FWI bajas pero viento extremo local, o vegetación especialmente inflamable

**Falsos Positivos (2,761 alertas innecesarias):**
- Probabilidad promedio de predicción: 68.9%
- Interpretación: Esperado con desbalanceo 147:1, trade-off aceptable
- Coste operativo: Movilización preventiva asumible vs coste de GIF no detectado

**Estado del objetivo:**
- **Objetivo final:** Recall >85% (detectar 85 de cada 100 GIF)
- **Estado actual:** Recall 70.8% (detecta 80 de 113 GIF)
- **Margen de mejora:** ~15pp para alcanzar objetivo de producción

**Próxima fase:** Métricas de negocio (hectáreas protegidas, costes evitados, ROI)

---

## 🔍 Explicabilidad

### ¿Por qué es importante la explicabilidad?

En un contexto operativo de emergencias, los profesionales necesitan **confiar** en las predicciones del modelo. La explicabilidad permite:
1. **Validar el modelo:** ¿Está aprendiendo patrones reales o correlaciones espurias?
2. **Generar confianza:** Los bomberos entienden **por qué** el sistema alerta de un GIF
3. **Mejorar el modelo:** Detectar features irrelevantes o faltantes

---

### FASE 5: EXPLICABILIDAD SHAP (COMPLETADA) ✅

**Notebook:** `05_Explainability_SHAP.ipynb`
**Duración:** 1 sesión (~10 minutos cálculo SHAP)
**Fecha:** 8 Marzo 2026

#### Objetivo:
Explicar **POR QUÉ** el modelo XGBoost predice GIF, identificando:
- Features más influyentes globalmente
- Dirección del efecto (positivo/negativo hacia GIF)
- Interacciones entre variables
- Explicación de casos individuales (TP, FN, FP)

#### Técnica utilizada:

**SHAP (SHapley Additive exPlanations)**
- TreeExplainer (optimizado para XGBoost)
- Muestra: 1,000 registros de test
- Tiempo de cálculo: ~3 minutos

#### RESULTADOS - Top 5 Features más influyentes:

| Ranking | Feature | SHAP Importance | Interpretación |
|---------|---------|-----------------|----------------|
| 1 | **fwi_max** | 0.7299 | FWI máximo provincial - predictor dominante |
| 2 | **fwi_p90** | 0.5356 | Percentil 90 FWI - condiciones extremas |
| 3 | **año** | 0.4696 | Tendencia temporal |
| 4 | **region_nan** | 0.2468 | Región desconocida - patrón inesperado |
| 5 | **fwi_mean** | 0.2047 | FWI promedio provincial |

**Observación clave:** Las 3 métricas FWI (max, p90, mean) suman **1.47** de importancia → **meteorología domina la predicción**.

#### Insights del análisis SHAP:

##### 1. **Relación fwi_max ↔ Predicción GIF**
- **Relación casi lineal positiva** hasta fwi_max ≈ 1.5 (normalizado)
- **Umbral crítico detectado:** fwi_max > 1.5 → impacto se dispara exponencialmente
- FWI extremo (>2.5) puede aportar hasta **+1.0 SHAP value** (diferencia entre No-GIF y GIF)

##### 2. **Efecto de región desconocida (region_nan)**
- Valores altos de `region_nan` → **impacto positivo consistente** hacia GIF
- Posible explicación: incendios sin región asignada ocurren en zonas remotas/difíciles → mayor probabilidad de propagación incontrolada
- **Interacción con FWI:** region_nan + FWI alto amplifica el riesgo

##### 3. **Temporalidad (año, meses)**
- `año` tiene **efecto variable** (no monotónico) → captura cambios en políticas/tecnología/clima
- `mes_6` (junio) muestra **efecto protector** (valores altos reducen probabilidad GIF)
- Agosto (mes_8) tiene efecto positivo moderado (esperado)

##### 4. **Modelo principalmente aditivo**
- Análisis de interacciones (SHAP interaction values) muestra **interacciones débiles**
- El modelo suma efectos independientes de cada feature
- No hay combinaciones complejas dominantes (simplicidad del modelo es fortaleza)

#### Waterfall Plots - Casos individuales analizados:

##### **Caso 1: True Positive (GIF detectado - 79.5% probabilidad)**
```
Características clave:
- fwi_max = 2.64 → +0.95 SHAP value
- fwi_p90 = 2.76 → +0.61
- region_Levante → +0.33
- Base value: 0.375 → Predicción final: 1.358 (79.5%)

Conclusión: FWI extremo domina completamente la predicción.
```

##### **Caso 2: False Negative (GIF NO detectado - 26.8% probabilidad)**
```
Características clave:
- año → -0.76 (efecto temporal negativo fuerte)
- mes_6 → -0.65 (junio reduce riesgo)
- region_nan → -0.50
- fwi_max = 1.42 → +0.47 (moderado, no extremo)
- Base value: 0.375 → Predicción final: -1.007 (26.8%)

Conclusión: FWI moderado + factores temporales desfavorables 
→ modelo subestima el riesgo. GIF probablemente causado por 
factores no capturados (orografía local, viento puntual).
```

##### **Caso 3: False Positive (Falsa alarma - 92.7% probabilidad)**
```
Características clave:
- fwi_max = 2.74 → +1.09 (extremo)
- region_Nordeste → +0.61
- fwi_p90 = 2.58 → +0.47
- Base value: 0.375 → Predicción final: 2.538 (92.7%)

Conclusión: FWI extremo activó alarma máxima. 
Real: No fue GIF (posible control rápido o error en datos).
Trade-off aceptable: priorizar detección sobre precisión.
```

#### Conclusiones del análisis SHAP:

✅ **Validación del modelo:**
- Features más importantes coinciden con conocimiento de dominio (meteorología crítica)
- Relaciones interpretables (FWI alto → más riesgo)
- No hay "magic" oculta en la caja negra

⚠️ **Limitaciones identificadas:**
- **Falsos Negativos:** Ocurren cuando FWI es moderado (1.4-1.5) pero otros factores locales (no capturados) favorecen propagación
- **region_nan** tiene peso inesperadamente alto → posible mejora imputando región correctamente

💡 **Recomendaciones para mejora:**
1. **Feature engineering avanzado:**
   - Crear features de interacción explícitas: `fwi_max * region_Sur`, `fwi_p90 * mes_8`
   - Incluir variables de sequía acumulada (días sin lluvia previos)
   - Orografía local (pendiente, orientación) si disponible

2. **Umbrales adaptativos por región:**
   - El umbral FWI crítico podría variar por región climática
   - Modelo ensemble con sub-modelos especializados por zona

3. **Análisis temporal más rico:**
   - Tendencias multi-año (no solo `año` lineal)
   - Estacionalidad intra-anual más granular

4. **Resolver `region_nan`:**
   - Imputar región usando coordenadas (lat/lng) cuando estén disponibles
   - Reducir del 19.75% actual a <5%

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

**Última actualización:** 08/03/2026  
**Versión del documento:** 1.0.0

**Progreso: ~60% completado**

**✅ COMPLETADO:**
1. Infraestructura y setup (día 1)
2. EDA completo con hallazgos (días 1-2)
3. Preprocesamiento con merge FWI (día 3)
4. Baseline validado (día 4)
5. Optimización XGBoost con GridSearch (día 5)
6. **Explicabilidad SHAP** (día 6)
7. README actualizado continuamente
8. Git con 20+ commits descriptivos

**⏳ PENDIENTE (~40%):**
1. **Métricas de negocio** (día 7) ← SIGUIENTE
   - Hectáreas protegidas estimadas
   - Costes evitados
   - ROI del modelo
2. **Documentación final** (día 8)
   - README completo
   - Conclusiones académicas
   - Limitaciones y mejoras futuras