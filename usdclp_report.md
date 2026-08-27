# Informe Cuantitativo de Posicionamiento USD/CLP (v5.0)

<<<<<<< HEAD
**Fecha de Ejecución:** `2026-08-27 02:22:50`  
**Fecha de Datos:** `2026-08-13`
=======
**Fecha de Ejecución:** `2026-08-26 22:35:26`  
**Fecha de Datos:** `2026-08-25`
>>>>>>> a4e8291 (fix: Alinear inferencia en tiempo real a la última observación de mercado y ampliar TTL del modelo a 30 días)

---

## 🎯 Señal de Mercado

| Métrica | Valor |
| :--- | :--- |
| **Señal de Trading** | **`BUY_USD`** |
| **Probabilidad Estimada Alza USD** | **`94.3%`** |
| **Régimen de Mercado (HMM)** | `SYSTEMIC_STRESS` |
| **Precio Actual USD/CLP** | **$912.19 CLP** |
| **Stop-Loss Dinámico (2.0x ATR)** | **$897.25 CLP** |
| **Take-Profit Sugerido (3.5x ATR)** | **$938.33 CLP** |
| **Nivel de VIX** | `15.4` |

---

## 📈 Rendimiento Histórico de Simulación (Walk-Forward OOS)

| Métrica de Desempeño | Valor |
| :--- | :--- |
| **Precisión Direccional Promedio** | **`51.1%`** |
| **Rango de DA (Mín / Máx)** | `[11.9%, 100.0%]` |
| **Retorno Acumulado Simulado** | **`+0.99%`** |
| **Retorno Anualizado (CAGR)** | **`+0.73%`** |
| **Sharpe Ratio** | **`0.382`** |
| **Sortino Ratio** | **`0.771`** |
| **Profit Factor** | **`1.21`** |
| **Win Rate en Operaciones** | **`50.0%`** (8 trades) |
| **Máximo Drawdown** | **`-3.46%`** |

---

## 🔍 Factores de Mayor Impacto (Explicabilidad SHAP)

```
ema_12                              : +0.959070
ema_50                              : +0.871342
realized_vol_63                     : +0.630414
brl_return_5d                       : +0.432688
price_z_52w                         : +0.387858
terms_of_trade_proxy                : +0.233428
hy_spread                           : +0.219903
copper                              : +0.155868
oil                                 : +0.137193
vol_ratio_21_63                     : +0.100547
```

---
*Generado automáticamente por el Pipeline Predictor USD/CLP v5.0 Enterprise.*
