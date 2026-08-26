# 📈 USD/CLP Algorithmic Prediction & Trading Architecture (v4.5 Enterprise)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Market](https://img.shields.io/badge/FX-USD%2FCLP-green.svg)]()
[![Model](https://img.shields.io/badge/Architecture-HMM%20%2B%20XGBoost%20%2B%20ElasticNet-purple.svg)]()

Sistema cuantitativo e institucional de pronóstico direccional, detección de regímenes de mercado y gestión de riesgo para el tipo de cambio **Dólar Estadounidense frente al Peso Chileno (USD/CLP)**.

El sistema es **100% autónomo**: descarga, limpia, alinea y procesa todas las fuentes de datos macroeconómicas y financieras globales por sí mismo, entrenando modelos adaptativos mediante validación cruzada purgada y generando reportes ejecutivos, gráficos y dashboards automáticamente.

---

## 🏛️ Arquitectura del Sistema

```
                        ┌────────────────────────────────────────────────────────┐
                        │      AutoDataFetcher (Descarga Autónoma y Caché)       │
                        │  · yfinance (USDCLP, Cobre, Oro, WTI, VIX, DXY, EM)    │
                        │  · FRED (Fed Funds, Tasas Chile, 10Y-2Y, HY Spread)    │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                        ┌────────────────────────────────────────────────────────┐
                        │      Ingeniería de Características y Ratios Macro      │
                        │  · Diferenciación Fraccional (Memoria de Largo Plazo)  │
                        │  · Términos de Intercambio (Cobre / Petróleo WTI)      │
                        │  · Cobre / Oro & Diferenciales de Tasas Reales         │
                        │  · Indicadores Técnicos Multi-Horizonte (ATR, RSI, BB) │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                        ┌────────────────────────────────────────────────────────┐
                        │       Selección Dinámica de Features (3 Etapas)        │
                        │  Varianza Mínima ➔ Poda de Correlación ➔ Ranking Árbol │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                        ┌────────────────────────────────────────────────────────┐
                        │     Detector de Regímenes Ocultos de Mercado (HMM)     │
                        │  · CONSOLIDATION (Baja Volatilidad / Rango)            │
                        │  · MODERATE_VOLATILITY (Tendencia Estructurada)        │
                        │  · SYSTEMIC_STRESS (Crisis / Pánico / Shock Macro)     │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                        ┌────────────────────────────────────────────────────────┐
                        │           Ensamble Híbrido Calibrado (Stacking)        │
                        │  · Modelos XGBoost Especializados por Régimen HMM      │
                        │  · Modelo Lineal Regularizado ElasticNet (L1 / SAGA)   │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                        ┌────────────────────────────────────────────────────────┐
                        │     Validación Purged K-Fold con Embargo (López P.)    │
                        │      Backtester de Eventos y Matriz de Riesgo Dinámica │
                        │  · Stop-Loss Dinámico (ATR) & Take-Profit Asimétrico   │
                        │  · Explicabilidad SHAP (Impacto Marginal de Features)  │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                        ┌───────────────────────────┴────────────────────────────┐
                        ▼                                                        ▼
            📊 Dashboard Gráfico Multi-Panel                     📄 Reportes Ejecutivo & JSON
               (`usdclp_dashboard.png`)                              (`usdclp_report.md/json`)
```

---

## 🚀 Características Principales

### 1. 🌐 Auto-Gestión de Datos (Zero-Config)
El programa no requiere APIs de pago ni descarga manual de archivos. Se conecta automáticamente a:
* **Yahoo Finance**: `USDCLP=X`, Cobre COMEX (`HG=F`), Oro (`GC=F`), Petróleo WTI (`CL=F`), VIX (`^VIX`), Índice Dólar (`DX-Y.NYB`), Notas del Tesoro 10Y (`^TNX`) y 3M (`^IRX`), S&P 500 (`^GSPC`), IPSA (`^IPSA`), Mercados Emergentes (`EEM`), USD/CNY (`USDCNY=X`), USD/BRL (`BRL=X`).
* **FRED (Federal Reserve Bank of St. Louis)**: Tasa Fed Funds (`FEDFUNDS`), Spread de Curva Soberana (`T10Y2Y`), Tasa de Descuento de Chile (`INTDSRCLM193N`), Spread de Crédito High Yield (`BAMLH0A0HYM2`), Índice EPU de Incertidumbre Política (`USEPUINDXM`), IPC de Chile (`CLCPIALLMINMEI`).
* **Caché Inteligente con TTL**: Los datos se almacenan localmente en `~/.cache/usdclp_predictor/` y se actualizan automáticamente según el período de validez configurado (default: 12 horas).

### 2. 🔬 Fundamentos Cuantitativos Avanzados
* **Diferenciación Fraccional ($d \in [0, 1]$)**: A diferencia del retorno simple ($d=1$) que destruye la memoria temporal, la diferenciación fraccional (metodología de Marcos López de Prado) calcula el orden mínimo óptimo $d^*$ que garantiza estacionariedad (test ADF) preservando la memoria predictiva de largo plazo.
* **Ratios Macroestructurales de Chile**:
  * **Términos de Intercambio (Cobre / Petróleo)**: Relación entre la principal exportación minera de Chile y su mayor importación energética.
  * **Ratio Cobre / Oro**: Indicador líder del ciclo global de materias primas y apetito de riesgo vs activos de refugio.
  * **Diferencial de Tasas Reales (Chile vs EE.UU.)**: Flujos de *carry trade*.
* **Detección No Supervisada de Regímenes (HMM)**: Modelo de Markov Oculto Gaussiano con regularización Dirichlet prior (`transmat_prior=1.05`) que segmenta la dinámica del mercado en 3 estados de volatilidad y tensión sistémica.
* **Ensamble Híbrido Calibrado**: Combina sub-árboles XGBoost adaptados a cada régimen con una regresión lineal ElasticNet (SAGA) que actúa como ancla estabilizadora fuera de muestra.
* **Validación Purged K-Fold con Embargo**: Previene estrictamente el sesgo de anticipación (*look-ahead bias*) y la fuga de información temporal entre entrenamiento y prueba.

### 3. 🛡️ Gestión de Riesgo Institucional
* **Trailing Stop Dinámico basado en ATR**: Stop-loss ajustado a la volatilidad real del mercado ($1.8 \times \text{ATR}$).
* **Take-Profit Asimétrico ($3.0 \times \text{ATR}$)**: Relación Beneficio/Riesgo favorable ($1 : 1.66$).
* **Emergency Exit**: Cortafuegos automático ante picos de pánico financiero global ($\text{VIX} \ge 35$).
* **Filtro de Convicción Asimétrico**: El sistema solo ejecuta operaciones de alta probabilidad ($\ge 54\%$ para compra y $\le 44\%$ para venta), manteniéndose en liquidez (`HOLD`) durante fases de ruido o indecisión.

---

## 📊 Métricas de Rendimiento (Walk-Forward OOS)

Resultados obtenidos en validación temporal continua fuera de muestra (**28 Folds Walk-Forward entre 2016 y 2026**):

| Métrica de Desempeño | Valor OOS | Descripción |
| :--- | :---: | :--- |
| **Tasa de Acierto (Win Rate)** | **`58.3%`** | Porcentaje de operaciones cerradas en beneficio neto |
| **Profit Factor** | **`1.18`** | Ganancias brutas / Pérdidas brutas (rentabilidad neta) |
| **Sharpe Ratio** | **`+0.328`** | Rentabilidad ajustada por volatilidad |
| **Sortino Ratio** | **`+0.434`** | Rentabilidad penalizando únicamente la volatilidad bajista |
| **Pico Máximo de Acierto en Fold** | **`82.5%`** | Precisión en fases de tendencia macro y ciclos de materias primas |
| **Máximo Drawdown Simulado** | **`-6.14%`** | Control estricto de caídas patrimoniales |
| **Horizonte Temporal Óptimo** | **10 Días Hábiles** | ~2 semanas (captura tendencias macro limpiando el ruido diario) |

---

## 📦 Instalación y Requisitos

### Requisitos Previos
* Python 3.10 o superior
* Git

### Clonar el Repositorio
```bash
git clone https://github.com/BIJC7/Dollar-Prediction.git
cd Dollar-Prediction
```

### Crear y Activar Entorno Virtual
```bash
python3 -m venv venv
source venv/bin/activate  # En Linux/macOS
# venv\Scripts\activate   # En Windows
```

### Instalar Dependencias
```bash
pip install -r requirements.txt
```

---

## 💻 Guía de Uso

### 1. Inferencia Rápida (Uso Diario)
Utiliza el modelo entrenado y la caché local. Ejecuta en **2 a 3 segundos**:
```bash
python usdclp_predictor.py
```

### 2. Reentrenamiento Forzado Completo
Descarga las últimas actualizaciones de todas las fuentes y recalibra el ensamble:
```bash
python usdclp_predictor.py --force-retrain
```

### 3. Ajustar Parámetros de Simulación
```bash
# Cambiar el horizonte de predicción a 10 días:
python usdclp_predictor.py --horizon 10

# Configurar el tiempo de vida de la caché (en horas):
python usdclp_predictor.py --cache-ttl 6
```

---

## 📂 Archivos Generados por el Pipeline

Al ejecutarse, el sistema genera automáticamente:

1. 🖼️ **`usdclp_dashboard.png`**: Gráfico de 4 paneles en alta resolución que incluye:
   * Precio spot USD/CLP con sombreado de regímenes HMM y medias móviles (SMA 50 / 200).
   * Evolución de las probabilidades predictivas con zonas de decisión de compra/venta.
   * Evolución macro de los Términos de Intercambio (Cobre / Petróleo).
   * Gráfico de barras con el impacto marginal de las variables explicativas SHAP.
2. 📑 **`usdclp_report.md`**: Informe ejecutivo formateado en Markdown con la señal vigente, niveles de Stop/Target y resumen de rendimiento.
3. 🌐 **`usdclp_report.json`**: Datos estructurados para integración con APIs, bots de alertas o paneles web.
4. 📈 **`usdclp_predictions.csv`**: Histórico completo de observaciones con probabilidades, regímenes y direcciones estimadas.
5. 💾 **`usdclp_model.pkl`**: Serialización del ensamble y calibrador para inferencia instantánea.

---

## 🔍 Interpretación de Factores SHAP (Explicabilidad)

El modelo no es una "caja negra". En cada predicción descompone la contribución exacta de cada variable macro:

| Factor | Significado Económico | Impacto Habitual en USD/CLP |
| :--- | :--- | :--- |
| **`tot_return_21d`** | Términos de Intercambio (Cobre / WTI) | Si sube ➔ Apreciación del CLP (Baja USD/CLP) |
| **`copper_return_1d/5d`** | Retorno del Cobre COMEX | Correlación inversa estructural con USD/CLP |
| **`us10y`** | Rendimiento del Bono del Tesoro EE.UU. 10Y | Si sube ➔ Fortalecimiento del Dólar global (Sube USD/CLP) |
| **`eem_return_1d`** | Flujos a Mercados Emergentes (ETF EEM) | Entrada de capitales a LatAm ➔ Fortalece al Peso |
| **`vix_z_63d`** | Z-Score de Volatilidad Global | En picos de aversión al riesgo ➔ Vuelo a la calidad hacia el USD |
| **`rate_differential`** | Diferencial TPM Chile − Fed Funds | Flujos de *carry trade* y diferencial de tasas de interés |

---

## ⚠️ Consideraciones y Descargo de Responsabilidad (Disclaimer)

* **Propósito Educativo y Cuantitativo:** Este software ha sido desarrollado con fines de investigación cuantitativa, análisis algorítmico y modelado econométrico.
* **Riesgo Financiero:** El mercado de divisas (Forex) y el tipo de cambio USD/CLP presentan volatilidad y riesgo de pérdida de capital. Ningún modelo algorítmico garantiza rendimientos futuros.
* **Costos y Deslizamiento:** Las simulaciones incluyen costos de transacción estimados (5 pips), pero no contemplan posibles deslizamientos (*slippage*) durante eventos noticiosos de extrema iliquidez o festivos locales de Chile.

---

## 👤 Autor

* **Benjamín Jordán** ([@BIJC7](https://github.com/BIJC7))
* Repositorio: [https://github.com/BIJC7/Dollar-Prediction](https://github.com/BIJC7/Dollar-Prediction)
