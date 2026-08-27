# 📈 USD/CLP Quantitative Research & Algorithmic Trading Architecture

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Market](https://img.shields.io/badge/FX-USD%2FCLP-green.svg)]()
[![Notifications](https://img.shields.io/badge/Alerts-Discord%20%7C%20Telegram%20%7C%20Desktop-blueviolet.svg)]()
[![Architecture](https://img.shields.io/badge/Model-HMM%20%2B%20XGBoost%20%2B%20ElasticNet-purple.svg)]()

Arquitectura modular de investigación cuantitativa, pronóstico direccional, detección de regímenes de mercado y gestión de riesgo para el tipo de cambio **Dólar Estadounidense frente al Peso Chileno (USD/CLP)**.

El sistema opera de forma autónoma: descarga, alinea e imputa series macroeconómicas y financieras globales, entrena ensambles adaptativos mediante validación cruzada purgada con embargo (*Marcos López de Prado*) y distribuye análisis ejecutivos, explicabilidad SHAP y alertas programadas en la nube.

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
                        │   Selección Dinámica por Fold (FeatureSelector)        │
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
       ┌──────────────────────────────┬─────────────┴────────────────┬──────────────────────────────┐
       ▼                              ▼                              ▼                              ▼
📊 Dashboard Gráfico          📄 Reporte Markdown            🌐 Dashboard JSON            🔔 Alertas Automáticas
(`usdclp_dashboard.png`)      (`usdclp_report.md`)           (`usdclp_report.json`)        (Discord, Telegram, OS)
```

---

## 🔬 Fundamentos Metodológicos

1. **Diferenciación Fraccional ($d \in [0, 1]$):** Metodología de Marcos López de Prado para alcanzar estacionariedad matemática (test ADF, $p < 0.05$) preservando la máxima memoria histórica de la serie temporal.
2. **Modelado de Regímenes con HMM:** Detección no supervisada de estados de mercado (*Gaussian Hidden Markov Model*) con regularización Dirichlet prior (`transmat_prior=1.05`), entrenado de forma aislada dentro de cada fold para garantizar cero fuga de datos hacia el futuro.
3. **Selección Dinámica de Variables por Fold:** El conjunto de características se reevalúa y poda en cada ventana temporal expansiva evitando sesgos estáticos.
4. **Validación Purged K-Fold con Embargo:** Purga las observaciones dentro del horizonte de la etiqueta ($H=10$ días) y añade una ventana de embargo porcentual para eliminar el sesgo de anticipación (*look-ahead bias*).
5. **Backtest de Eventos Realista:** Ejecuta operaciones no solapadas con descuento de costos de transacción (5 pips).

---

## 📊 Métricas de Validación Temporal (Walk-Forward OOS)

Resultados en validación continua fuera de muestra (**Horizonte objetivo: 10 Días Hábiles**):

| Métrica de Desempeño | Valor OOS | Descripción y Contexto Metodológico |
| :--- | :---: | :--- |
| **Profit Factor Neto** | **`1.36`** | Ganancias brutas / Pérdidas brutas (costos de fricción incluidos) |
| **Sharpe Ratio** | **`+0.547`** | Retorno anualizado ajustado por volatilidad |
| **Sortino Ratio** | **`+1.313`** | Retorno penalizando únicamente la volatilidad bajista |
| **Retorno Acumulado Simulado** | **`+6.41%`** | Retorno neto durante el período de validación temporal |
| **Tasa de Acierto (Win Rate)** | **`48.4%`** | 15 operaciones ganadoras de 31 trades totales |
| **Total de Operaciones OOS** | **31 trades** | Simulación de eventos con selección de features dinámica por fold |
| **Máximo Drawdown Simulado** | **`-6.17%`** | Control estricto de caídas patrimoniales |
| **Precisión Direccional Promedio** | **`47.1%`** | Mediana: `47.6%` \| Rango por fold: `[11.9%, 66.7%]` |
| **Horizonte Temporal** | **10 Días Hábiles** | ~2 semanas (captura tendencias macro limpiando el ruido intradía) |

> **Nota Metodológica:** Al operar con un horizonte de 10 días y filtros estrictos de convicción, el volumen de operaciones es bajo. Esto introduce varianza muestral natural en períodos cortos, requiriendo meses de ejecución continua para converger a la media estadística.

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
python3 -m venv venv_usdclp
source venv_usdclp/bin/activate  # En Linux/macOS
# venv_usdclp\Scripts\activate   # En Windows
```

### Instalar Dependencias
```bash
pip install -r requirements.txt
```

---

## 💻 Guía de Comandos CLI

```bash
# 1. Inferencia Rápida con Modelo Persistente (2 a 3 segundos)
python usdclp_predictor.py

# 2. Inferencia con Despacho de Alertas (Discord / Telegram / Desktop)
python usdclp_predictor.py --notify

# 3. Reentrenamiento Completo y Calibración Walk-Forward
python usdclp_predictor.py --force-retrain

# 4. Modo Servicio / Bucle Desatendido Local
python usdclp_predictor.py --loop-hours 4 --notify
```

---

## ☁️ Automatización en la Nube (GitHub Actions)

El proyecto incluye 3 flujos de trabajo en `.github/workflows/` para operar sin depender del hardware local:

* 🟢 **Diario (Días hábiles):** Inferencia y alertas a las **09:00, 13:30 y 16:30 CLT** (`usdclp_bot.yml`).
* 🔄 **Mensual (Día 1):** Reentrenamiento completo con los datos más recientes (`usdclp_monthly_retrain.yml`).
* 🩺 **Semestral y Anual:** Diagnóstico de salud de APIs y recordatorio preventivo (`usdclp_maintenance.yml`).

---

## 📂 Archivos Generados

1. 🖼️ **`usdclp_dashboard.png`**: Gráfico de 4 paneles con sombreado de regímenes HMM, probabilidades predictivas, términos de intercambio y barras SHAP.
2. 📑 **`usdclp_report.md`**: Informe ejecutivo en formato Markdown.
3. 🌐 **`usdclp_report.json`**: Datos estructurados para consumo en APIs.
4. 📈 **`usdclp_predictions.csv`**: Histórico completo de probabilidades y regímenes.
5. 💾 **`usdclp_model.pkl`**: Serialización binaria del ensamble calibrado.

---

## ⚠️ Descargo de Responsabilidad (Disclaimer)

* **Propósito Académico y de Investigación:** Este software ha sido desarrollado con fines de investigación cuantitativa, análisis econométrico y modelado de series de tiempo financieras.
* **Riesgo en Mercado FX:** El mercado de divisas (Forex) y el tipo de cambio USD/CLP presentan volatilidad y riesgo de pérdida de capital. Los rendimientos pasados no garantizan rendimientos futuros.
* **Condiciones de Mercado:** Las simulaciones incorporan un costo estándar de 5 pips, pero en eventos de iliquidez severa o noticias macroeconómicas extraordinarias los spreads bancarios reales pueden ampliarse.

---

## 👤 Autor

* **Benjamín Jordán** ([@BIJC7](https://github.com/BIJC7))
* Repositorio: [https://github.com/BIJC7/Dollar-Prediction](https://github.com/BIJC7/Dollar-Prediction)
