# 📈 USD/CLP Algorithmic Prediction & Trading Architecture (v5.0 Enterprise)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Market](https://img.shields.io/badge/FX-USD%2FCLP-green.svg)]()
[![Notifications](https://img.shields.io/badge/Alerts-Telegram%20%7C%20Discord%20%7C%20Desktop-blueviolet.svg)]()
[![Architecture](https://img.shields.io/badge/Model-HMM%20%2B%20XGBoost%20%2B%20ElasticNet-purple.svg)]()

Sistema cuantitativo e institucional de pronóstico direccional, detección de regímenes de mercado, gestión de riesgo y **alertas automatizadas** para el tipo de cambio **Dólar Estadounidense frente al Peso Chileno (USD/CLP)**.

El sistema es **100% autónomo**: descarga, limpia, alinea y procesa todas las fuentes de datos macroeconómicas y financieras globales por sí mismo, entrena modelos adaptativos mediante validación cruzada purgada y distribuye reportes ejecutivos, dashboards gráficos y notificaciones en tiempo real.

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
       ┌──────────────────────────────┬─────────────┴────────────────┬──────────────────────────────┐
       ▼                              ▼                              ▼                              ▼
📊 Dashboard Gráfico          📄 Reporte Markdown            🌐 Dashboard JSON            🔔 Alertas Automáticas
(`usdclp_dashboard.png`)      (`usdclp_report.md`)           (`usdclp_report.json`)        (Telegram, Discord, OS)
```

---

## 🚀 Características Principales

### 1. 🌐 Auto-Gestión de Datos (Zero-Config)
El programa no requiere APIs de pago ni descarga manual de archivos. Se conecta automáticamente a:
* **Yahoo Finance**: `USDCLP=X`, Cobre COMEX (`HG=F`), Oro (`GC=F`), Petróleo WTI (`CL=F`), VIX (`^VIX`), Índice Dólar (`DX-Y.NYB`), Notas del Tesoro 10Y (`^TNX`) y 3M (`^IRX`), S&P 500 (`^GSPC`), IPSA (`^IPSA`), Mercados Emergentes (`EEM`), USD/CNY (`USDCNY=X`), USD/BRL (`BRL=X`).
* **FRED (Federal Reserve Bank of St. Louis)**: Tasa Fed Funds (`FEDFUNDS`), Spread de Curva Soberana (`T10Y2Y`), Tasa de Descuento de Chile (`INTDSRCLM193N`), Spread de Crédito High Yield (`BAMLH0A0HYM2`), Índice EPU de Incertidumbre Política (`USEPUINDXM`), IPC de Chile (`CLCPIALLMINMEI`).
* **Caché Inteligente con TTL**: Los datos se almacenan localmente en `~/.cache/usdclp_predictor/` y se actualizan automáticamente según el período de validez configurado (default: 12 horas).

### 2. 🔬 Fundamentos Cuantitativos Avanzados
* **Diferenciación Fraccional ($d \in [0, 1]$)**: Metodología de Marcos López de Prado para lograr estacionariedad preservando memoria de largo plazo.
* **Ratios Macroestructurales de Chile**:
  * **Términos de Intercambio (Cobre / Petróleo)**: Relación exportación minera / importación energética.
  * **Ratio Cobre / Oro**: Indicador líder de ciclo global y apetito de riesgo.
  * **Diferencial de Tasas Reales**: Chile vs EE.UU.
* **Detección No Supervisada de Regímenes (Gaussian HMM)**: Con regularización Dirichlet prior (`transmat_prior=1.05`) para evitar artefactos numéricos.
* **Ensamble Híbrido Calibrado**: XGBoost condicionado al régimen + ancla lineal ElasticNet (SAGA).
* **Validación Purged K-Fold con Embargo**: Elimina el sesgo de anticipación (*look-ahead bias*).

### 3. 🔔 Sistema de Notificaciones Automáticas
Soporte nativo sin librerías externas pesadas (usando HTTP estándar):
* **Telegram Bot**: Mensajes enriquecidos con formato Markdown, señal, niveles de precio, stop-loss y probabilidades.
* **Discord Webhook**: Rich Embeds con código de color dinámico (Verde = Compra, Rojo = Venta, Gris = Hold).
* **Notificaciones de Escritorio**: Integración con el sistema de notificaciones del SO (`notify-send` en Linux).
* **Modo Demonio / Bucle**: Ejecución programada periódica cada $N$ horas (`--loop-hours 4`).

---

## 📊 Métricas de Rendimiento (Walk-Forward OOS)

Resultados en validación temporal continua fuera de muestra (**28 Folds Walk-Forward entre 2016 y 2026, horizonte 10 días**):

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

## 💻 Guía de Uso y Comandos

### 1. Inferencia Rápida (Uso Diario)
Utiliza el modelo entrenado y la caché local. Ejecuta en **2 a 3 segundos**:
```bash
python usdclp_predictor.py
```

### 2. Inferencia con Notificación Automática
```bash
# Envía alertas a los canales configurados (Telegram / Discord / Escritorio)
python usdclp_predictor.py --notify
```

### 3. Reentrenamiento Forzado Completo
Descarga las últimas actualizaciones de todas las fuentes y recalibra el ensamble:
```bash
python usdclp_predictor.py --force-retrain
```

### 4. Modo Servicio / Bucle Desatendido
Ejecuta el pipeline de forma continua cada 4 horas y envía notificaciones:
```bash
python usdclp_predictor.py --loop-hours 4 --notify
```

---

## ⚙️ Configuración de Alertas (Variables de Entorno)

Puedes configurar las credenciales de alerta exportando las variables en tu entorno o en tu archivo `~/.bashrc` / `.env`:

### Telegram
1. Habla con `@BotFather` en Telegram para crear un bot y obtener tu `TELEGRAM_BOT_TOKEN`.
2. Obtén tu `TELEGRAM_CHAT_ID` (por ejemplo, mediante `@userinfobot`).
3. Exporta las variables:
```bash
export TELEGRAM_BOT_TOKEN="123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ"
export TELEGRAM_CHAT_ID="987654321"
```

### Discord
1. En tu servidor de Discord, ve a **Ajustes de canal ➔ Integraciones ➔ Webhooks ➔ Crear Webhook**.
2. Copia la URL del Webhook y expórtala:
```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

---

## 📂 Archivos Generados por el Pipeline

1. 🖼️ **`usdclp_dashboard.png`**: Gráfico de 4 paneles en alta resolución con sombreado de regímenes, probabilidades, términos de intercambio y barras SHAP.
2. 📑 **`usdclp_report.md`**: Informe ejecutivo formateado en Markdown.
3. 🌐 **`usdclp_report.json`**: Datos estructurados para APIs y dashboards web.
4. 📈 **`usdclp_predictions.csv`**: Histórico completo de observaciones con probabilidades y regímenes.
5. 💾 **`usdclp_model.pkl`**: Serialización del modelo para inferencia instantánea.

---

## ⚠️ Consideraciones y Descargo de Responsabilidad (Disclaimer)

* **Propósito Cuantitativo y Educativo:** Este software ha sido desarrollado con fines de investigación cuantitativa, análisis algorítmico y modelado econométrico.
* **Riesgo Financiero:** El mercado de divisas (Forex) y el tipo de cambio USD/CLP presentan volatilidad y riesgo de pérdida de capital. Ningún modelo algorítmico garantiza rendimientos futuros.
* **Costos y Deslizamiento:** Las simulaciones incluyen costos de transacción estimados (5 pips), pero no contemplan posibles deslizamientos (*slippage*) durante eventos noticiosos de extrema iliquidez o festivos locales de Chile.

---

## 👤 Autor

* **Benjamín Jordán** ([@BIJC7](https://github.com/BIJC7))
* Repositorio: [https://github.com/BIJC7/Dollar-Prediction](https://github.com/BIJC7/Dollar-Prediction)
