"""
Arquitectura Algorítmica de Alta Precisión para Predicción USD/CLP (v5.0 Enterprise)
===================================================================================

Sistema cuantitativo autónomo e integral para predicción direccional, trading,
gestión de riesgo y NOTIFICACIONES AUTOMÁTICAS (Telegram, Discord, Webhooks, Desktop)
para el tipo de cambio USD/CLP (Dólar estadounidense / Peso chileno).

Módulos y Capacidades:
  1. AutoDataFetcher: Descarga autónoma y caché local con TTL de 15+ variables macro
  2. FeatureEngineer: Ratios macro (Cobre/Petróleo, Cobre/Oro), FracDiff, Indicadores técnicos
  3. FeatureSelector: Poda de multicolinealidad y ranking no lineal
  4. HmmRegimeDetector: Segmentación no supervisada de regímenes de mercado
  5. HybridCalibratedEnsemble: XGBoost por régimen + LogisticRegression ElasticNet (SAGA)
  6. PurgedKFoldEmbargo: Validación temporal rigurosa sin fuga de datos
  7. EventDrivenBacktester: Simulación basada en trades reales con costos de transacción
  8. VisualDashboardGenerator: Gráfico de 4 paneles en alta resolución (`usdclp_dashboard.png`)
  9. NotificationManager: Alertas automáticas para Telegram, Discord, Webhooks y Desktop
 10. AutomatedScheduler: Ejecución desatendida diaria a la apertura/cierre de mercado
"""

from __future__ import annotations

import abc
import argparse
import json
import logging
import os
import pickle
import sys
import time
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

import xgboost as xgb
import shap

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None  # type: ignore[assignment,misc]

try:
    import pandas_datareader.data as pdr
    _HAS_PDR = True
except ImportError:
    _HAS_PDR = False
    pdr = None  # type: ignore[assignment]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    plt = None  # type: ignore[assignment]

logger = logging.getLogger("usdclp_predictor")
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

def _load_dotenv() -> None:
    """Carga variables de entorno desde un archivo .env local si existe."""
    env_file = Path(".env")
    if not env_file.exists():
        return
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception as exc:
        logger.debug("No se pudo leer .env: %s", exc)

_load_dotenv()


_CACHE_DIR: Path = Path(
    os.environ.get("USDCLP_CACHE_DIR",
                   str(Path.home() / ".cache" / "usdclp_predictor"))
)
_CACHE_TTL_HOURS: int = int(os.environ.get("USDCLP_CACHE_TTL_HOURS", "12"))
_MODEL_PATH: Path     = Path(os.environ.get("USDCLP_MODEL_PATH", "./usdclp_model.pkl"))
_MODEL_TTL_HOURS: int = 24


# ===========================================================================
# 0. TIPOS Y ENUMERACIONES
# ===========================================================================

class MarketRegime(Enum):
    CONSOLIDATION       = auto()
    MODERATE_VOLATILITY = auto()
    SYSTEMIC_STRESS     = auto()


class TradingSignal(Enum):
    BUY_USD        = auto()
    SELL_USD       = auto()
    HOLD           = auto()
    EMERGENCY_EXIT = auto()


# ===========================================================================
# 1. DESCARGA AUTÓNOMA DE FUENTES DE DATOS
# ===========================================================================

class AutoDataFetcher:
    _RISK_TICKERS: Dict[str, str] = {
        "vix":   "^VIX",
        "dxy":   "DX-Y.NYB",
        "us10y": "^TNX",
        "us3m":  "^IRX",
        "gold":  "GC=F",
        "oil":   "CL=F",
        "sp500": "^GSPC",
        "ipsa":  "^IPSA",
        "eem":   "EEM",
        "cny":   "USDCNY=X",
        "brl":   "BRL=X",
    }

    _FRED_RISK: Dict[str, str] = {
        "hy_spread": "BAMLH0A0HYM2",
        "us_10y2y":  "T10Y2Y",
    }

    _FRED_MACRO: Dict[str, str] = {
        "fed_funds":  "FEDFUNDS",
        "chile_rate": "INTDSRCLM193N",
        "cny_fred":   "DEXCHUS",
        "epu_us":     "USEPUINDXM",
        "chile_cpi":  "CLCPIALLMINMEI",
    }

    def __init__(self, start: str = "2013-01-01", end: Optional[str] = None,
                 cache_dir: Path = _CACHE_DIR, ttl_hours: int = _CACHE_TTL_HOURS) -> None:
        self.start     = start
        self.end       = end or datetime.today().strftime("%Y-%m-%d")
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.csv"

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age_h = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
        return age_h < self.ttl_hours

    def _read_cache(self, name: str) -> Optional[pd.DataFrame]:
        p = self._cache_path(name)
        if not self._is_cache_valid(p):
            return None
        try:
            return pd.read_csv(p, index_col=0, parse_dates=True)
        except Exception:
            return None

    def _write_cache(self, df: pd.DataFrame, name: str) -> None:
        try:
            df.to_csv(self._cache_path(name))
        except Exception as exc:
            logger.debug("Fallo al escribir cache '%s': %s", name, exc)

    def _yf_close(self, ticker: str, col_name: str) -> Optional[pd.Series]:
        try:
            raw = yf.download(ticker, start=self.start, end=self.end,
                              auto_adjust=True, progress=False)
            if raw.empty:
                return None
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            close.name = col_name
            close.index = pd.to_datetime(close.index)
            return close.sort_index().dropna()
        except Exception as exc:
            logger.warning("[yfinance] %-12s (%s): %s", col_name, ticker, exc)
            return None

    def _fred_series(self, series_id: str, col_name: str) -> Optional[pd.Series]:
        if _HAS_PDR:
            try:
                df = pdr.DataReader(series_id, "fred", self.start, self.end)
                s = df.iloc[:, 0].rename(col_name)
                s.index = pd.to_datetime(s.index)
                logger.info("  ✓ %-16s (FRED:%s via pdr)", col_name, series_id)
                return s.sort_index().dropna()
            except Exception:
                pass
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            df  = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
            s   = df.iloc[:, 0].rename(col_name)
            s   = s.loc[self.start: self.end].sort_index().dropna()
            logger.info("  ✓ %-16s (FRED:%s via URL)", col_name, series_id)
            return s
        except Exception as exc:
            logger.warning("[FRED] %-16s (%s): %s", col_name, series_id, exc)
            return None

    def fetch_price_ohlcv(self) -> pd.DataFrame:
        cached = self._read_cache("price_ohlcv")
        if cached is not None:
            return cached
        logger.info("Descargando USD/CLP OHLCV...")
        raw = yf.download("USDCLP=X", start=self.start, end=self.end,
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError("Error: Sin datos para USDCLP=X.")
        raw.columns = raw.columns.get_level_values(0).str.lower()
        raw.index   = pd.to_datetime(raw.index)
        raw         = raw.sort_index().dropna(subset=["close"])
        self._write_cache(raw, "price_ohlcv")
        logger.info("  USD/CLP: %d barras (%s -> %s)",
                    len(raw), raw.index[0].date(), raw.index[-1].date())
        return raw

    def fetch_risk_indicators(self, base_index: pd.DatetimeIndex) -> pd.DataFrame:
        cached = self._read_cache("risk_indicators")
        if cached is not None:
            return cached.reindex(base_index).ffill(limit=5)
        logger.info("Descargando variables de riesgo y mercado global...")
        parts: List[pd.Series] = []
        for col_name, ticker in self._RISK_TICKERS.items():
            s = self._yf_close(ticker, col_name)
            if s is not None:
                parts.append(s)
                logger.info("  ✓ %-12s (%s): %d obs", col_name, ticker, len(s))
        for col_name, sid in self._FRED_RISK.items():
            s = self._fred_series(sid, col_name)
            if s is not None:
                parts.append(s)
        if not parts:
            return pd.DataFrame(index=base_index)
        df = pd.concat(parts, axis=1).sort_index()
        df = df.reindex(base_index).ffill(limit=5)
        self._write_cache(df, "risk_indicators")
        return df

    def fetch_macro_indicators(self, base_index: pd.DatetimeIndex) -> pd.DataFrame:
        cached = self._read_cache("macro_indicators")
        if cached is not None:
            return cached.reindex(base_index).ffill(limit=31)
        logger.info("Descargando variables macroeconomicas y fundamentales...")
        parts: List[pd.Series] = []
        copper = self._yf_close("HG=F", "copper")
        if copper is not None:
            parts.append(copper)
            logger.info("  ✓ copper (HG=F): %d obs", len(copper))
        for col_name, sid in self._FRED_MACRO.items():
            s = self._fred_series(sid, col_name)
            if s is not None:
                parts.append(s)
        if not parts:
            return pd.DataFrame(index=base_index)
        df = pd.concat(parts, axis=1).sort_index()
        df = df.reindex(base_index).ffill(limit=31)
        if "chile_rate" in df.columns and "fed_funds" in df.columns:
            df["rate_differential"] = df["chile_rate"] - df["fed_funds"]
        self._write_cache(df, "macro_indicators")
        return df

    def fetch_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        price_df = self.fetch_price_ohlcv()
        macro_df = self.fetch_macro_indicators(price_df.index)
        risk_df  = self.fetch_risk_indicators(price_df.index)
        return price_df, macro_df, risk_df


# ===========================================================================
# 2. DIFERENCIACIÓN FRACCIONAL
# ===========================================================================

@dataclass
class FractionalDifferentiator:
    d_min:            float = 0.0
    d_max:            float = 1.0
    d_step:           float = 0.05
    threshold:        float = 1e-4
    adf_significance: float = 0.05
    _fitted_d:        Optional[float] = field(default=None, init=False, repr=False)

    def _get_weights(self, d: float, size: int) -> np.ndarray:
        weights = [1.0]
        for k in range(1, size):
            weights.append(-weights[-1] * (d - k + 1) / k)
        return np.array(weights[::-1]).reshape(-1, 1)

    def _fractional_diff_series(self, series: pd.Series, d: float) -> pd.Series:
        weights = self._get_weights(d, len(series))
        cutoff  = int(np.argmax(np.abs(weights) < self.threshold))
        cutoff  = cutoff if cutoff > 0 else len(weights)
        w       = weights[-cutoff:]
        output  = pd.Series(index=series.index, dtype=float)
        values  = series.values
        for i in range(cutoff, len(series)):
            window = values[i - cutoff + 1: i + 1]
            output.iloc[i] = float(np.dot(w.T, window.reshape(-1, 1)))
        return output

    def _passes_adf_test(self, series: pd.Series) -> bool:
        clean = series.dropna()
        if len(clean) < 20:
            return False
        return bool(adfuller(clean, autolag="AIC")[1] < self.adf_significance)

    def find_minimum_d(self, price_series: pd.Series) -> float:
        for d in np.arange(self.d_min, self.d_max + self.d_step, self.d_step):
            transformed = self._fractional_diff_series(price_series, float(d))
            if self._passes_adf_test(transformed.dropna()):
                self._fitted_d = float(d)
                logger.info("Orden fraccional optimo: d=%.2f", float(d))
                return self._fitted_d
        raise RuntimeError("No se encontro 'd' estacionario.")

    def transform(self, price_series: pd.Series, d: Optional[float] = None) -> pd.Series:
        d_to_use = d if d is not None else self._fitted_d
        if d_to_use is None:
            raise ValueError("Llamar find_minimum_d() primero.")
        return self._fractional_diff_series(price_series, d_to_use)


# ===========================================================================
# 3. INGENIERÍA DE CARACTERÍSTICAS
# ===========================================================================

class FeatureEngineer:
    def __init__(self, frac_differentiator: Optional[FractionalDifferentiator] = None) -> None:
        self.frac_differentiator = frac_differentiator or FractionalDifferentiator()

    def add_fractional_memory_features(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        try:
            d = self.frac_differentiator.find_minimum_d(df[price_col])
            df[f"{price_col}_fracdiff"] = self.frac_differentiator.transform(df[price_col], d)
        except RuntimeError:
            df[f"{price_col}_fracdiff"] = df[price_col].pct_change()
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        df["ema_12"]  = close.ewm(span=12, adjust=False).mean()
        df["ema_26"]  = close.ewm(span=26, adjust=False).mean()
        df["ema_50"]  = close.ewm(span=50, adjust=False).mean()
        df["ema_200"] = close.ewm(span=200, adjust=False).mean()
        df["sma_50"]  = close.rolling(50).mean()
        df["sma_200"] = close.rolling(200).mean()

        df["macd_line"]   = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd_line"] - df["macd_signal"]

        delta        = close.diff()
        gain         = delta.clip(lower=0).rolling(14).mean()
        loss         = (-delta.clip(upper=0)).rolling(14).mean()
        rs           = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        hl               = high - low
        hc               = (high - close.shift()).abs()
        lc               = (low  - close.shift()).abs()
        tr               = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr_14"]     = tr.rolling(14).mean()
        df["atr_14_pct"] = df["atr_14"] / close.replace(0, np.nan)

        bb_mid          = close.rolling(20).mean()
        bb_std          = close.rolling(20).std(ddof=0)
        df["bb_upper"]  = bb_mid + 2 * bb_std
        df["bb_lower"]  = bb_mid - 2 * bb_std
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)
        df["bb_pct_b"]  = (close - df["bb_lower"]) / (
            (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan))

        df["sma_cross"]       = np.sign(df["sma_50"] - df["sma_200"])
        df["realized_vol_21"] = close.pct_change().rolling(21).std() * np.sqrt(252)
        df["realized_vol_63"] = close.pct_change().rolling(63).std() * np.sqrt(252)
        df["vol_ratio_21_63"] = df["realized_vol_21"] / df["realized_vol_63"].replace(0, np.nan)
        df["price_z_52w"]     = (close - close.rolling(252).mean()) / close.rolling(252).std().replace(0, np.nan)
        return df

    def add_lag_features(self, df: pd.DataFrame, col: str = "close",
                         lags: Tuple[int, ...] = (1, 3, 5, 10, 21)) -> pd.DataFrame:
        for lag in lags:
            df[f"return_{lag}d"] = df[col].pct_change(lag)
        return df

    def add_macro_and_ratios(self, df: pd.DataFrame, macro_df: pd.DataFrame,
                             risk_df: pd.DataFrame) -> pd.DataFrame:
        merged = df.copy()
        if not macro_df.empty:
            merged = pd.merge_asof(merged.sort_index(), macro_df.sort_index(),
                                   left_index=True, right_index=True)
        if not risk_df.empty:
            merged = pd.merge_asof(merged.sort_index(), risk_df.sort_index(),
                                   left_index=True, right_index=True)

        if "copper" in merged.columns and "oil" in merged.columns:
            merged["terms_of_trade_proxy"] = merged["copper"] / merged["oil"].replace(0, np.nan)
            merged["tot_return_21d"]       = merged["terms_of_trade_proxy"].pct_change(21)

        if "copper" in merged.columns and "gold" in merged.columns:
            merged["copper_gold_ratio"]   = merged["copper"] / merged["gold"].replace(0, np.nan)
            merged["copper_gold_ret_21d"] = merged["copper_gold_ratio"].pct_change(21)

        if "copper" in merged.columns:
            merged["copper_return_1d"] = merged["copper"].pct_change()
            merged["copper_return_5d"] = merged["copper"].pct_change(5)
            merged["copper_return_21d"]= merged["copper"].pct_change(21)
            merged["copper_z_63d"]     = (
                (merged["copper"] - merged["copper"].rolling(63).mean())
                / merged["copper"].rolling(63).std().replace(0, np.nan))

        if "vix" in merged.columns:
            merged["vix_change_1d"] = merged["vix"].diff()
            merged["vix_z_63d"]     = (
                (merged["vix"] - merged["vix"].rolling(63).mean())
                / merged["vix"].rolling(63).std().replace(0, np.nan))
        if "hy_spread" in merged.columns:
            merged["hy_spread_change_5d"] = merged["hy_spread"].diff(5)
            merged["hy_spread_z_63d"]     = (
                (merged["hy_spread"] - merged["hy_spread"].rolling(63).mean())
                / merged["hy_spread"].rolling(63).std().replace(0, np.nan))

        if "dxy" in merged.columns:
            merged["dxy_return_1d"] = merged["dxy"].pct_change()
            merged["dxy_return_5d"] = merged["dxy"].pct_change(5)
        if "us10y" in merged.columns and "us3m" in merged.columns:
            merged["us_yield_spread"] = merged["us10y"] - merged["us3m"]

        if "eem" in merged.columns:
            merged["eem_return_1d"] = merged["eem"].pct_change()
            merged["eem_return_5d"] = merged["eem"].pct_change(5)
        if "cny" in merged.columns:
            merged["cny_return_5d"] = merged["cny"].pct_change(5)
        if "brl" in merged.columns:
            merged["brl_return_5d"] = merged["brl"].pct_change(5)

        if "copper_return_1d" in merged.columns and "return_1d" in merged.columns:
            merged["copper_usdclp_corr_21d"] = (
                merged["return_1d"].rolling(21).corr(merged["copper_return_1d"]))
        if "cny_return_5d" in merged.columns and "return_5d" in merged.columns:
            merged["cny_usdclp_corr_21d"] = (
                merged["return_5d"].rolling(21).corr(merged["cny_return_5d"]))

        return merged

    def build_feature_matrix(self, price_df: pd.DataFrame, macro_df: pd.DataFrame,
                             risk_df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        df = price_df.copy()
        df = self.add_technical_indicators(df)
        df = self.add_fractional_memory_features(df, price_col)
        df = self.add_lag_features(df, price_col)
        df = self.add_macro_and_ratios(df, macro_df, risk_df)
        return df


# ===========================================================================
# 4. SELECCIÓN DE CARACTERÍSTICAS
# ===========================================================================

class FeatureSelector:
    def __init__(self, max_features: int = 35,
                 correlation_threshold: float = 0.90,
                 min_variance: float = 1e-6) -> None:
        self.max_features          = max_features
        self.correlation_threshold = correlation_threshold
        self.min_variance          = min_variance
        self.selected_cols_: List[str] = []
        self._is_fitted: bool          = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        cols = list(X.columns)
        variances = X[cols].var()
        cols = [c for c in cols if variances.get(c, 0) >= self.min_variance]

        corr_matrix = X[cols].corr().abs().fillna(0)
        to_drop: set = set()
        for i in range(len(cols)):
            if cols[i] in to_drop:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in to_drop:
                    continue
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    to_drop.add(cols[j])
        cols = [c for c in cols if c not in to_drop]

        Xc = X[cols].ffill().bfill().fillna(0)
        m = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.08,
                              subsample=0.8, colsample_bytree=0.8, verbosity=0,
                              objective="binary:logistic", eval_metric="logloss",
                              random_state=42)
        m.fit(Xc, y)
        scores = pd.Series(m.feature_importances_, index=cols)
        self.selected_cols_ = scores.sort_values(ascending=False).head(self.max_features).index.tolist()
        self._is_fitted = True
        logger.info("FeatureSelector: %d caracteristicas seleccionadas.", len(self.selected_cols_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Llamar fit() antes de transform().")
        available = [c for c in self.selected_cols_ if c in X.columns]
        return X[available].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# ===========================================================================
# 5. VALIDACIÓN PURGADA (WALK-FORWARD)
# ===========================================================================

class PurgedKFoldEmbargo(BaseCrossValidator):
    def __init__(self, n_splits: int = 5, label_horizon: int = 10,
                 embargo_pct: float = 0.01) -> None:
        self.n_splits      = n_splits
        self.label_horizon = label_horizon
        self.embargo_pct   = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X: pd.DataFrame, y=None, groups=None):
        n_samples    = len(X)
        indices      = np.arange(n_samples)
        fold_bounds  = np.array_split(indices, self.n_splits)
        embargo_size = int(n_samples * self.embargo_pct)
        for test_fold in fold_bounds:
            test_start, test_end = int(test_fold[0]), int(test_fold[-1])
            purge_start = max(0, test_start - self.label_horizon)
            purge_end   = min(n_samples, test_end + self.label_horizon)
            embargo_end = min(n_samples, purge_end + embargo_size)
            train_mask  = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:embargo_end] = False
            yield indices[train_mask], indices[test_start: test_end + 1]


class WalkForwardOrchestrator:
    def __init__(self, cross_validator: PurgedKFoldEmbargo,
                 min_train_size: int = 600, step_size: int = 63) -> None:
        self.cross_validator = cross_validator
        self.min_train_size  = min_train_size
        self.step_size       = step_size

    def generate_expanding_windows(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        windows: List[Tuple[np.ndarray, np.ndarray]] = []
        start        = self.min_train_size
        embargo_size = int(n_samples * self.cross_validator.embargo_pct)
        horizon      = self.cross_validator.label_horizon
        while start + self.step_size <= n_samples:
            train_end  = start
            test_start = min(train_end + horizon, n_samples)
            test_end   = min(test_start + self.step_size, n_samples)
            train_idx  = np.arange(0, max(0, train_end - horizon))
            test_idx   = np.arange(test_start, test_end)
            windows.append((train_idx, test_idx))
            start += self.step_size + embargo_size
        return windows


# ===========================================================================
# 6. MODELO DE ENSAMBLE HÍBRIDO SENSIBLE AL RÉGIMEN
# ===========================================================================

class HmmRegimeDetector:
    _PRIMARY_COLS:  List[str] = ["realized_vol_21", "atr_14_pct", "rsi_14"]
    _FALLBACK_COLS: List[str] = ["return_1d", "atr_14_pct", "rsi_14"]

    def __init__(self, n_states: int = 3, n_iter: int = 300,
                 n_restarts: int = 5, random_state: int = 42) -> None:
        if GaussianHMM is None:
            raise ImportError("pip install hmmlearn")
        self.n_states      = n_states
        self.n_iter        = n_iter
        self.n_restarts    = n_restarts
        self.random_state  = random_state
        self._scaler:          StandardScaler          = StandardScaler()
        self._model:           Optional[GaussianHMM]   = None
        self._state_to_regime: Dict[int, MarketRegime] = {}
        self._feature_cols:    List[str]               = []

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        for candidates in [self._PRIMARY_COLS, self._FALLBACK_COLS]:
            cols = [c for c in candidates if c in features.columns]
            if len(cols) >= 2:
                self._feature_cols = cols
                return features[cols].copy()
        num_cols = features.select_dtypes(include=[np.number]).columns.tolist()[:3]
        self._feature_cols = num_cols
        return features[num_cols].copy()

    def fit(self, features: pd.DataFrame) -> "HmmRegimeDetector":
        X_raw = self._select_features(features).ffill().bfill().dropna()
        X_sc  = self._scaler.fit_transform(X_raw.values)
        best_model: Optional[GaussianHMM] = None
        best_score = -np.inf

        for seed in range(self.n_restarts):
            model = GaussianHMM(
                n_components     = self.n_states,
                covariance_type  = "diag",
                min_covar        = 1e-2,
                transmat_prior   = 1.05,
                startprob_prior  = 1.05,
                n_iter           = self.n_iter,
                random_state     = self.random_state + seed,
            )
            try:
                model.fit(X_sc)
                score = model.score(X_sc)
                if score > best_score:
                    best_score, best_model = score, model
            except Exception:
                continue

        if best_model is None:
            raise RuntimeError("HMM no logro converger.")

        self._model = best_model
        state_vol = {s: float(best_model.covars_[s].mean()) for s in range(self.n_states)}
        ordered   = sorted(state_vol, key=state_vol.get)  # type: ignore[arg-type]
        regimes   = list(MarketRegime)
        self._state_to_regime = {s: regimes[min(i, len(regimes) - 1)]
                                 for i, s in enumerate(ordered)}
        return self

    def predict_regime(self, features: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Llamar fit() primero.")
        available = [c for c in self._feature_cols if c in features.columns]
        X_raw = features[available].copy()
        for col in self._feature_cols:
            if col not in X_raw.columns:
                X_raw[col] = 0.0
        X_raw    = X_raw[self._feature_cols]
        X_filled = X_raw.ffill().bfill().fillna(0.0)
        X_sc     = self._scaler.transform(X_filled.values)
        states   = self._model.predict(X_sc)
        mapped   = [self._state_to_regime.get(int(s), MarketRegime.MODERATE_VOLATILITY)
                    for s in states]
        return pd.Series(mapped, index=features.index, name="market_regime")


class HybridCalibratedEnsemble(BaseEstimator, ClassifierMixin):
    _MIN_REGIME_SAMPLES: int = 50

    def __init__(self, regime_detector: HmmRegimeDetector,
                 xgb_params: Optional[Dict[str, Any]] = None,
                 linear_weight: float = 0.20) -> None:
        self.regime_detector = regime_detector
        self.linear_weight   = linear_weight
        self.xgb_params      = xgb_params or {
            "n_estimators":     250,
            "max_depth":        3,
            "learning_rate":    0.03,
            "subsample":        0.75,
            "colsample_bytree": 0.75,
            "reg_alpha":        0.8,
            "reg_lambda":       1.5,
            "gamma":            0.1,
            "objective":        "binary:logistic",
            "eval_metric":      "logloss",
            "verbosity":        0,
            "random_state":     42,
        }
        self._regime_models: Dict[MarketRegime, xgb.XGBClassifier] = {}
        self._global_xgb:    Optional[xgb.XGBClassifier]           = None
        self._linear_model:  Optional[LogisticRegression]          = None
        self._linear_scaler: StandardScaler                        = StandardScaler()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HybridCalibratedEnsemble":
        self.regime_detector.fit(X)
        regimes = self.regime_detector.predict_regime(X)

        X_clean = X.ffill().bfill().fillna(0)
        X_scaled = self._linear_scaler.fit_transform(X_clean)
        self._linear_model = LogisticRegression(
            penalty="elasticnet", l1_ratio=0.8, C=0.15,
            solver="saga", max_iter=600, random_state=42
        )
        self._linear_model.fit(X_scaled, y)

        pos_w = float((y == 0).sum() / max((y == 1).sum(), 1))
        self._global_xgb = xgb.XGBClassifier(**dict(self.xgb_params, scale_pos_weight=pos_w))
        self._global_xgb.fit(X_clean, y)

        self._regime_models = {}
        for regime in MarketRegime:
            mask = (regimes == regime)
            n    = int(mask.sum())
            if n < self._MIN_REGIME_SAMPLES:
                continue
            y_r = y.loc[mask]
            spw = float((y_r == 0).sum() / max((y_r == 1).sum(), 1))
            m   = xgb.XGBClassifier(**dict(self.xgb_params, scale_pos_weight=spw))
            m.fit(X_clean.loc[mask], y_r)
            self._regime_models[regime] = m

        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        X_clean = X.ffill().bfill().fillna(0)
        regimes = self.regime_detector.predict_regime(X_clean)
        tree_proba = pd.Series(np.nan, index=X.index, dtype=float)

        for regime, model in self._regime_models.items():
            mask = (regimes == regime)
            if mask.any():
                tree_proba.loc[mask] = model.predict_proba(X_clean.loc[mask])[:, 1]

        global_mask = tree_proba.isna()
        if global_mask.any() and self._global_xgb is not None:
            tree_proba.loc[global_mask] = self._global_xgb.predict_proba(X_clean.loc[global_mask])[:, 1]

        tree_proba = tree_proba.fillna(0.5)

        if self._linear_model is not None:
            X_sc = self._linear_scaler.transform(X_clean)
            lin_proba = pd.Series(self._linear_model.predict_proba(X_sc)[:, 1], index=X.index)
            final_proba = (1.0 - self.linear_weight) * tree_proba + self.linear_weight * lin_proba
        else:
            final_proba = tree_proba

        return final_proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        return (self.predict_proba(X) >= threshold).astype(int)


# ===========================================================================
# 7. EXPLICABILIDAD SHAP
# ===========================================================================

class ShapExplainabilityModule:
    def __init__(self, ensemble_model: HybridCalibratedEnsemble) -> None:
        self.ensemble_model = ensemble_model
        all_models: Dict[MarketRegime, xgb.XGBClassifier] = dict(ensemble_model._regime_models)
        if ensemble_model._global_xgb is not None:
            for regime in MarketRegime:
                if regime not in all_models:
                    all_models[regime] = ensemble_model._global_xgb
        self._explainers: Dict[MarketRegime, shap.TreeExplainer] = {
            regime: shap.TreeExplainer(model)
            for regime, model in all_models.items()
        }

    def _sv2d(self, expl: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
        sv = np.array(expl.shap_values(X))
        if sv.ndim == 3:
            sv = sv[1]
        return sv

    def global_importance(self, X: pd.DataFrame, regime: MarketRegime) -> pd.Series:
        expl = self._explainers.get(regime, list(self._explainers.values())[0])
        return pd.Series(np.abs(self._sv2d(expl, X)).mean(axis=0),
                         index=X.columns).sort_values(ascending=False)

    def local_explanation(self, x_instance: pd.DataFrame, regime: MarketRegime) -> pd.Series:
        expl = self._explainers.get(regime, list(self._explainers.values())[0])
        sv   = self._sv2d(expl, x_instance)
        return pd.Series(sv[0], index=x_instance.columns)


# ===========================================================================
# 8. GESTIÓN DE RIESGO Y MATRIZ DE DECISIÓN
# ===========================================================================

@dataclass
class RiskLimits:
    max_drawdown_pct:             float = 0.12
    atr_trailing_stop_multiplier: float = 2.0
    take_profit_atr_multiplier:   float = 3.5
    vix_panic_threshold:          float = 35.0


class SignalSmoother:
    def __init__(self, span: int = 3) -> None:
        self.span = span

    def smooth(self, raw_probabilities: pd.Series) -> pd.Series:
        return raw_probabilities.ewm(span=self.span, adjust=False).mean()


class RiskManager:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits        = limits
        self._equity_peak: float = -np.inf

    def update_drawdown(self, current_equity: float) -> float:
        self._equity_peak = max(self._equity_peak, current_equity)
        return 0.0 if self._equity_peak <= 0 else (
            (self._equity_peak - current_equity) / self._equity_peak)

    def is_max_drawdown_breached(self, current_equity: float) -> bool:
        return self.update_drawdown(current_equity) >= self.limits.max_drawdown_pct

    def compute_trailing_stop(self, entry_price: float, atr: float, direction: int) -> float:
        return entry_price - direction * self.limits.atr_trailing_stop_multiplier * atr

    def compute_take_profit(self, entry_price: float, atr: float, direction: int) -> float:
        return entry_price + direction * self.limits.take_profit_atr_multiplier * atr

    def is_emergency_exit(self, current_vix: float) -> bool:
        return current_vix >= self.limits.vix_panic_threshold


class DecisionMatrix:
    def __init__(self, smoother: SignalSmoother, risk_manager: RiskManager,
                 buy_threshold: float = 0.54, sell_threshold: float = 0.44) -> None:
        self.smoother       = smoother
        self.risk_manager   = risk_manager
        self.buy_threshold  = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signal(self, raw_probabilities: pd.Series, momentum_confirmation: pd.Series,
                        current_vix: float, current_equity: float) -> pd.Series:
        if self.risk_manager.is_emergency_exit(current_vix):
            return pd.Series(TradingSignal.EMERGENCY_EXIT, index=raw_probabilities.index)
        if self.risk_manager.is_max_drawdown_breached(current_equity):
            return pd.Series(TradingSignal.HOLD, index=raw_probabilities.index)

        smoothed = self.smoother.smooth(raw_probabilities)
        signals  = pd.Series(TradingSignal.HOLD, index=smoothed.index)
        signals.loc[(smoothed >= self.buy_threshold) & (momentum_confirmation > 0)] = TradingSignal.BUY_USD
        signals.loc[smoothed <= self.sell_threshold]                                 = TradingSignal.SELL_USD
        return signals


# ===========================================================================
# 9. BACKTESTING BASADO EN EVENTOS
# ===========================================================================

class BacktestMetrics(NamedTuple):
    cumulative_return:    float
    annualized_return:    float
    sharpe_ratio:         float
    sortino_ratio:        float
    calmar_ratio:         float
    max_drawdown:         float
    directional_accuracy: float
    win_rate:             float
    profit_factor:        float
    total_trades:         int
    avg_trade_return:     float
    avg_win:              float
    avg_loss:             float


class EventDrivenBacktester:
    def __init__(self, transaction_cost_pct: float = 0.0005, holding_period: int = 10) -> None:
        self.transaction_cost_pct = transaction_cost_pct
        self.holding_period       = holding_period

    def run(self, close_prices: pd.Series, predictions: pd.Series,
            probabilities: pd.Series) -> BacktestMetrics:
        aligned = close_prices.reindex(predictions.index).dropna()
        preds   = predictions.reindex(aligned.index)
        probs   = probabilities.reindex(aligned.index)

        fwd_return = aligned.shift(-self.holding_period) / aligned - 1.0
        trade_returns: List[float] = []
        last_exit_idx = -1

        for idx in range(len(aligned)):
            if idx <= last_exit_idx or idx + self.holding_period >= len(aligned):
                continue

            prob = probs.iloc[idx]
            pred = preds.iloc[idx]
            fwd_ret = fwd_return.iloc[idx]

            if prob >= 0.54 and pred == 1:
                pnl = fwd_ret - 2 * self.transaction_cost_pct
                trade_returns.append(pnl)
                last_exit_idx = idx + self.holding_period
            elif prob <= 0.44 and pred == 0:
                pnl = -fwd_ret - 2 * self.transaction_cost_pct
                trade_returns.append(pnl)
                last_exit_idx = idx + self.holding_period

        if not trade_returns:
            return BacktestMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        trade_series = pd.Series(trade_returns)
        equity_curve = (1.0 + trade_series).cumprod()

        cum_ret = float(equity_curve.iloc[-1] - 1.0)
        n_years = max(len(aligned) / 252.0, 0.2)
        ann_ret = float((equity_curve.iloc[-1]) ** (1.0 / n_years) - 1.0)

        std = float(trade_series.std(ddof=1))
        sharpe = (trade_series.mean() / std) * np.sqrt(252 / self.holding_period) if std > 0 else 0.0

        downside_std = float(trade_series[trade_series < 0].std(ddof=1))
        sortino = (trade_series.mean() / downside_std) * np.sqrt(252 / self.holding_period) if downside_std > 0 else 0.0

        peak = equity_curve.cummax()
        drawdowns = (equity_curve - peak) / peak
        max_dd = float(drawdowns.min())
        calmar = abs(ann_ret / max_dd) if abs(max_dd) > 1e-4 else 0.0

        wins   = trade_series[trade_series > 0]
        losses = trade_series[trade_series < 0]
        win_rate      = float(len(wins) / len(trade_series))
        total_gains   = float(wins.sum())
        total_losses  = abs(float(losses.sum()))
        profit_factor = (total_gains / total_losses) if total_losses > 0 else (99.0 if total_gains > 0 else 0.0)

        avg_trade = float(trade_series.mean())
        avg_win   = float(wins.mean())   if len(wins)   > 0 else 0.0
        avg_loss  = float(losses.mean())  if len(losses) > 0 else 0.0
        da        = float((trade_series > 0).mean())

        return BacktestMetrics(
            cumulative_return    = cum_ret,
            annualized_return    = ann_ret,
            sharpe_ratio         = sharpe,
            sortino_ratio        = sortino,
            calmar_ratio         = calmar,
            max_drawdown         = max_dd,
            directional_accuracy = da,
            win_rate             = win_rate,
            profit_factor        = profit_factor,
            total_trades         = len(trade_series),
            avg_trade_return     = avg_trade,
            avg_win              = avg_win,
            avg_loss             = avg_loss,
        )


# ===========================================================================
# 10. GESTOR DE PERSISTENCIA
# ===========================================================================

class ModelPersistenceManager:
    def __init__(self, path: Path = _MODEL_PATH, max_age_hours: float = _MODEL_TTL_HOURS) -> None:
        self.path          = path
        self.max_age_hours = max_age_hours

    def is_stale(self) -> bool:
        if not self.path.exists():
            return True
        age_h = (datetime.now().timestamp() - self.path.stat().st_mtime) / 3600
        return age_h > self.max_age_hours

    def save(self, payload: dict) -> None:
        try:
            with open(self.path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Modelo guardado en disco: %s", self.path.resolve())
        except Exception as exc:
            logger.warning("Error al guardar modelo: %s", exc)

    def load(self) -> Optional[dict]:
        if self.is_stale():
            return None
        try:
            with open(self.path, "rb") as f:
                payload = pickle.load(f)
            logger.info("Modelo cargado desde disco: %s", self.path.resolve())
            return payload
        except Exception as exc:
            logger.warning("Error al cargar modelo: %s", exc)
            return None


# ===========================================================================
# 11. SISTEMA DE NOTIFICACIONES AUTOMÁTICAS
# ===========================================================================

class NotificationManager:
    """
    Gestor universal de alertas automáticas:
      · Telegram Bot (vía HTTP API)
      · Discord Webhook (rich embeds)
      · Slack Webhook
      · Desktop Notification (Linux notify-send)
    """

    def __init__(self, telegram_token: Optional[str] = None,
                 telegram_chat_id: Optional[str] = None,
                 discord_webhook: Optional[str] = None,
                 slack_webhook: Optional[str] = None) -> None:
        self.telegram_token   = telegram_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.discord_webhook  = discord_webhook or os.environ.get("DISCORD_WEBHOOK_URL")
        self.slack_webhook    = slack_webhook or os.environ.get("SLACK_WEBHOOK_URL")

    def _format_message(self, signal: TradingSignal, price: float, prob_up: float,
                        regime: str, stop_loss: float, take_profit: float,
                        date_str: str) -> str:
        icon = "🟢" if signal == TradingSignal.BUY_USD else ("🔴" if signal == TradingSignal.SELL_USD else "⚪")
        msg = (
            f"📊 *ALERTA USD/CLP (v5.0)*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📅 *Fecha:* `{date_str}`\n"
            f"💰 *Precio Spot:* `${price:,.2f} CLP`\n"
            f"{icon} *Señal:* `>>> {signal.name} <<<`\n"
            f"📈 *Prob. Alza USD:* `{prob_up:.1%}`\n"
            f"🌪️ *Régimen:* `{regime}`\n"
        )
        if signal in (TradingSignal.BUY_USD, TradingSignal.SELL_USD):
            msg += (
                f"🛡️ *Stop-Loss (ATR):* `${stop_loss:,.2f} CLP`\n"
                f"🎯 *Take-Profit:* `${take_profit:,.2f} CLP`\n"
            )
        msg += "━━━━━━━━━━━━━━━━━━━\n*USD/CLP Quantitative Engine*"
        return msg

    def send_telegram(self, message: str) -> bool:
        if not self.telegram_token or not self.telegram_chat_id:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.info("✓ Notificación Telegram enviada exitosamente.")
                    return True
        except Exception as exc:
            logger.warning("Fallo al enviar notificación Telegram: %s", exc)
        return False

    def send_discord(self, signal: TradingSignal, price: float, prob_up: float,
                     regime: str, stop_loss: float, take_profit: float, date_str: str) -> bool:
        if not self.discord_webhook:
            return False
        try:
            color = 0x00FF00 if signal == TradingSignal.BUY_USD else (0xFF0000 if signal == TradingSignal.SELL_USD else 0x808080)
            payload = {
                "embeds": [{
                    "title": f"🚨 ALERTA USD/CLP: {signal.name}",
                    "color": color,
                    "fields": [
                        {"name": "💵 Precio Cierre", "value": f"${price:,.2f} CLP", "inline": True},
                        {"name": "📊 Probabilidad Alza", "value": f"{prob_up:.1%}", "inline": True},
                        {"name": "🌪️ Régimen HMM", "value": regime, "inline": True},
                        {"name": "🛡️ Stop-Loss Dinámico", "value": f"${stop_loss:,.2f} CLP", "inline": True},
                        {"name": "🎯 Take-Profit Objetivo", "value": f"${take_profit:,.2f} CLP", "inline": True},
                        {"name": "📅 Fecha", "value": date_str, "inline": True},
                    ],
                    "footer": {"text": "USD/CLP Quantitative Engine v5.0"}
                }]
            }
            req = urllib.request.Request(
                self.discord_webhook,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status in (200, 204):
                    logger.info("✓ Notificación Discord enviada exitosamente.")
                    return True
        except Exception as exc:
            logger.warning("Fallo al enviar notificación Discord: %s", exc)
        return False

    def send_desktop(self, signal: TradingSignal, price: float, prob_up: float) -> bool:
        try:
            import subprocess
            title = f"USD/CLP Señal: {signal.name}"
            body = f"Precio: ${price:,.2f} CLP | Prob. Alza: {prob_up:.1%}"
            subprocess.run(["notify-send", title, body], check=False)
            return True
        except Exception:
            return False

    def broadcast(self, signal: TradingSignal, price: float, prob_up: float,
                  regime: str, stop_loss: float, take_profit: float, date_str: str) -> None:
        msg = self._format_message(signal, price, prob_up, regime, stop_loss, take_profit, date_str)
        self.send_telegram(msg)
        self.send_discord(signal, price, prob_up, regime, stop_loss, take_profit, date_str)
        self.send_desktop(signal, price, prob_up)


# ===========================================================================
# 12. ORQUESTADOR PRINCIPAL DEL PIPELINE
# ===========================================================================

class UsdClpPredictionPipeline:
    def __init__(self, feature_engineer: FeatureEngineer, walk_forward: WalkForwardOrchestrator,
                 ensemble_model: HybridCalibratedEnsemble, decision_matrix: DecisionMatrix,
                 feature_selector: Optional[FeatureSelector] = None) -> None:
        self.feature_engineer = feature_engineer
        self.walk_forward     = walk_forward
        self.ensemble_model   = ensemble_model
        self.decision_matrix  = decision_matrix
        self.feature_selector = feature_selector
        self.shap_module:     Optional[ShapExplainabilityModule] = None
        self.oos_predictions_: Optional[pd.DataFrame]            = None

    def run_training_and_validation(self, feature_matrix: pd.DataFrame,
                                    target: pd.Series) -> List[float]:
        fold_scores: List[float]     = []
        all_preds:   List[pd.Series] = []
        all_probas:  List[pd.Series] = []
        all_true:    List[pd.Series] = []
        windows = self.walk_forward.generate_expanding_windows(len(feature_matrix))

        if self.feature_selector is not None:
            init_end = self.walk_forward.min_train_size
            X_init   = feature_matrix.iloc[:init_end].ffill().bfill().fillna(0)
            y_init   = target.iloc[:init_end]
            feature_matrix = self.feature_selector.fit_transform(X_init, y_init).reindex(
                feature_matrix.index).ffill().bfill().fillna(0)
            feature_matrix = feature_matrix[self.feature_selector.selected_cols_].copy()

        for i, (train_idx, test_idx) in enumerate(windows, 1):
            if len(train_idx) < 50 or len(test_idx) == 0:
                continue
            X_train = feature_matrix.iloc[train_idx].ffill().bfill().fillna(0)
            y_train = target.iloc[train_idx]
            X_test  = feature_matrix.iloc[test_idx].ffill().bfill().fillna(0)
            y_test  = target.iloc[test_idx]

            self.ensemble_model.fit(X_train, y_train)
            probas = self.ensemble_model.predict_proba(X_test)
            preds  = self.ensemble_model.predict(X_test)
            score  = float((y_test == preds).mean())
            fold_scores.append(score)
            all_preds.append(preds)
            all_probas.append(probas)
            all_true.append(y_test)
            logger.info("Fold %2d/%d  DA=%.4f  (train=%4d, test=%3d, %s -> %s)",
                        i, len(windows), score, len(train_idx), len(test_idx),
                        X_test.index[0].date(), X_test.index[-1].date())

        if all_preds:
            self.oos_predictions_ = pd.DataFrame({
                "pred":  pd.concat(all_preds),
                "proba": pd.concat(all_probas),
                "true":  pd.concat(all_true),
            })
            self.shap_module = ShapExplainabilityModule(self.ensemble_model)

        return fold_scores

    def generate_live_signal(self, latest_features: pd.DataFrame,
                             momentum_confirmation: pd.Series, current_vix: float,
                             current_equity: float) -> pd.Series:
        if self.feature_selector is not None and self.feature_selector._is_fitted:
            latest_features = self.feature_selector.transform(
                latest_features.ffill().bfill().fillna(0))
        else:
            latest_features = latest_features.ffill().bfill().fillna(0)
        raw_proba = self.ensemble_model.predict_proba(latest_features)
        return self.decision_matrix.generate_signal(
            raw_probabilities     = raw_proba,
            momentum_confirmation = momentum_confirmation,
            current_vix           = current_vix,
            current_equity        = current_equity)

    def explain_last_decision(self, x_instance: pd.DataFrame,
                              regime: MarketRegime) -> pd.Series:
        if self.shap_module is None:
            raise RuntimeError("Ejecutar entrenamiento antes de explicabilidad.")
        if self.feature_selector is not None and self.feature_selector._is_fitted:
            x_instance = self.feature_selector.transform(x_instance.ffill().bfill().fillna(0))
        return self.shap_module.local_explanation(x_instance, regime)


# ===========================================================================
# 13. GENERADOR DE DASHBOARDS VISUALES
# ===========================================================================

class VisualDashboardGenerator:
    @staticmethod
    def render_dashboard(price_df: pd.DataFrame, full_df: pd.DataFrame,
                         predictions_df: pd.DataFrame, shap_top: Optional[pd.Series],
                         signal: TradingSignal, output_path: Path) -> Optional[Path]:
        if not _HAS_MATPLOTLIB:
            return None

        try:
            plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
            fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False,
                                     gridspec_kw={"height_ratios": [2.5, 1.8, 1.8, 2.0]})

            # Panel 1: Precio USD/CLP y Regímenes
            ax1 = axes[0]
            close = price_df["close"]
            ax1.plot(close.index, close.values, label="USD/CLP (Cierre)", color="#1f77b4", linewidth=1.5)
            if "sma_50" in full_df.columns:
                ax1.plot(full_df.index, full_df["sma_50"], label="SMA 50", color="#ff7f0e", linestyle="--", alpha=0.7)
            if "sma_200" in full_df.columns:
                ax1.plot(full_df.index, full_df["sma_200"], label="SMA 200", color="#2ca02c", linestyle="--", alpha=0.7)

            if "market_regime" in predictions_df.columns:
                reg_colors = {
                    "CONSOLIDATION": ("#2ca02c", 0.08),
                    "MODERATE_VOLATILITY": ("#ff7f0e", 0.10),
                    "SYSTEMIC_STRESS": ("#d62728", 0.15),
                }
                curr_reg = None
                start_dt = None
                for dt, row in predictions_df.iterrows():
                    reg = row["market_regime"]
                    if reg != curr_reg:
                        if curr_reg is not None and start_dt is not None:
                            color, alpha = reg_colors.get(curr_reg, ("#gray", 0.05))
                            ax1.axvspan(start_dt, dt, color=color, alpha=alpha)
                        curr_reg = reg
                        start_dt = dt
                if curr_reg is not None and start_dt is not None:
                    color, alpha = reg_colors.get(curr_reg, ("#gray", 0.05))
                    ax1.axvspan(start_dt, predictions_df.index[-1], color=color, alpha=alpha)

            last_price = float(close.iloc[-1])
            ax1.set_title(f"USD/CLP Tipo de Cambio & Detección de Regímenes HMM | Último: ${last_price:,.2f} CLP ({signal.name})",
                          fontsize=13, fontweight="bold", pad=10)
            ax1.set_ylabel("CLP / USD", fontsize=10)
            ax1.legend(loc="upper left", frameon=True)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

            # Panel 2: Probabilidades y Umbrales
            ax2 = axes[1]
            if "prob_usd_up" in predictions_df.columns:
                probs = predictions_df["prob_usd_up"]
                ax2.plot(probs.index, probs.values, color="#6f42c1", label="P(Alza USD a 10d)", linewidth=1.2)
                ax2.axhline(0.54, color="red", linestyle=":", alpha=0.8, label="Umbral Compra (0.54)")
                ax2.axhline(0.44, color="green", linestyle=":", alpha=0.8, label="Umbral Venta (0.44)")
                ax2.axhline(0.50, color="gray", linestyle="-", alpha=0.3)
                ax2.fill_between(probs.index, 0.54, 1.0, color="red", alpha=0.06)
                ax2.fill_between(probs.index, 0.0, 0.44, color="green", alpha=0.06)

            ax2.set_title("Probabilidad Predictiva de Apreciación del Dólar (Horizonte 10 Días)", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Probabilidad", fontsize=10)
            ax2.set_ylim(0.2, 0.8)
            ax2.legend(loc="upper right", frameon=True)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

            # Panel 3: Términos de Intercambio
            ax3 = axes[2]
            if "terms_of_trade_proxy" in full_df.columns:
                tot = full_df["terms_of_trade_proxy"].dropna()
                ax3.plot(tot.index, tot.values, color="#e377c2", label="Ratio Cobre / Petróleo (Términos de Intercambio)", linewidth=1.2)
                ax3.set_ylabel("Ratio HG/CL", fontsize=10)
                ax3.set_title("Evolución Macro: Términos de Intercambio de Chile", fontsize=11, fontweight="bold")
                ax3.legend(loc="upper left", frameon=True)
                ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

            # Panel 4: SHAP Top 10
            ax4 = axes[3]
            if shap_top is not None and not shap_top.empty:
                top_features = shap_top.head(10)
                colors = ["#1f77b4" if v >= 0 else "#d62728" for v in top_features.values]
                ax4.barh(top_features.index[::-1], top_features.values[::-1], color=colors[::-1], alpha=0.85)
                ax4.set_title("Explicabilidad Local SHAP (Impacto Marginal en la Señal Actual)", fontsize=11, fontweight="bold")
                ax4.set_xlabel("Magnitud del Impacto SHAP (|Valor|)", fontsize=10)

            plt.tight_layout()
            plt.savefig(output_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            logger.info("Dashboard grafico guardado en: %s", output_path.resolve())
            return output_path
        except Exception as exc:
            logger.warning("Error al generar dashboard grafico: %s", exc)
            return None


# ===========================================================================
# 14. GENERADOR DE REPORTES
# ===========================================================================

class ResultsReporter:
    def __init__(self, output_dir: Path = Path(".")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_console_summary(self, fold_scores: List[float], signal: TradingSignal,
                              signal_date: str, current_vix: float, current_price: float,
                              entry_stop: float, entry_target: float,
                              shap_top5: Optional[pd.Series] = None,
                              backtest: Optional[BacktestMetrics] = None) -> None:
        arr = np.array(fold_scores)
        mean_da = float(arr.mean()) if len(arr) > 0 else 0.0
        sep = "=" * 70
        logger.info(sep)
        logger.info("  RESUMEN EJECUTIVO USD/CLP (v5.0 Enterprise)")
        logger.info(sep)
        logger.info("  Validacion Walk-Forward (%d Folds OOS):", len(fold_scores))
        logger.info("    Precision Direccional Promedio : %.4f (%.1f%%)", mean_da, mean_da * 100)
        logger.info("    Mediana                        : %.4f", float(np.median(arr)) if len(arr)>0 else 0)
        logger.info("    Rango de Rendimiento           : [%.4f, %.4f]", float(arr.min()) if len(arr)>0 else 0, float(arr.max()) if len(arr)>0 else 0)

        if backtest is not None:
            logger.info(sep)
            logger.info("  Simulacion de Trading (Trades OOS reales, costos incluidos):")
            logger.info("    Retorno Acumulado              : %+.2f%%", backtest.cumulative_return * 100)
            logger.info("    Retorno Anualizado (CAGR)      : %+.2f%%", backtest.annualized_return * 100)
            logger.info("    Sharpe Ratio                   : %.3f", backtest.sharpe_ratio)
            logger.info("    Sortino Ratio                  : %.3f", backtest.sortino_ratio)
            logger.info("    Profit Factor                  : %.2f", backtest.profit_factor)
            logger.info("    Win Rate                       : %.1f%% (%d trades)", backtest.win_rate * 100, backtest.total_trades)
            logger.info("    Max Drawdown                   : %.2f%%", backtest.max_drawdown * 100)

        logger.info(sep)
        logger.info("  ESTADO DE MERCADO Y SEÑAL VIGENTE:")
        logger.info("    Fecha de Señal                 : %s", signal_date)
        logger.info("    Precio de Cierre USD/CLP       : $%.2f CLP", current_price)
        logger.info("    Nivel de VIX                   : %.1f", current_vix)
        logger.info("    Señal de Posicionamiento       : >>> %s <<<", signal.name)
        if signal in (TradingSignal.BUY_USD, TradingSignal.SELL_USD):
            logger.info("    Stop-Loss Dinamico (ATR)       : $%.2f CLP", entry_stop)
            logger.info("    Take-Profit Objetivo (ATR)     : $%.2f CLP", entry_target)
        logger.info(sep)

        if shap_top5 is not None and not shap_top5.empty:
            logger.info("  Factores Clave de la Decision (Top-5 SHAP):")
            for feat, val in shap_top5.items():
                logger.info("    %-32s : %.6f", feat, val)
            logger.info(sep)

    def export_markdown_and_json(self, fold_scores: List[float], signal: TradingSignal,
                                 signal_date: str, current_price: float, current_vix: float,
                                 current_regime: str, entry_stop: float, entry_target: float,
                                 prob_up: float, shap_top: Optional[pd.Series],
                                 backtest: Optional[BacktestMetrics]) -> Tuple[Path, Path]:
        arr = np.array(fold_scores)
        data = {
            "timestamp": datetime.now().isoformat(),
            "signal_date": signal_date,
            "current_price": current_price,
            "signal": signal.name,
            "probability_usd_up": prob_up,
            "market_regime": current_regime,
            "vix": current_vix,
            "stop_loss": entry_stop,
            "take_profit": entry_target,
            "validation": {
                "n_folds": len(fold_scores),
                "mean_directional_accuracy": float(arr.mean()) if len(arr)>0 else 0,
                "median_da": float(np.median(arr)) if len(arr)>0 else 0,
                "min_da": float(arr.min()) if len(arr)>0 else 0,
                "max_da": float(arr.max()) if len(arr)>0 else 0,
            },
            "backtest": {
                "cumulative_return_pct": backtest.cumulative_return * 100 if backtest else None,
                "annualized_return_pct": backtest.annualized_return * 100 if backtest else None,
                "sharpe_ratio": backtest.sharpe_ratio if backtest else None,
                "sortino_ratio": backtest.sortino_ratio if backtest else None,
                "profit_factor": backtest.profit_factor if backtest else None,
                "win_rate_pct": backtest.win_rate * 100 if backtest else None,
                "max_drawdown_pct": backtest.max_drawdown * 100 if backtest else None,
                "total_trades": backtest.total_trades if backtest else None,
            } if backtest else {},
            "top_shap_factors": shap_top.to_dict() if shap_top is not None else {}
        }

        json_path = self.output_dir / "usdclp_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        md_content = f"""# Informe Cuantitativo de Posicionamiento USD/CLP (v5.0)

**Fecha de Ejecución:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`  
**Fecha de Datos:** `{signal_date}`

---

## 🎯 Señal de Mercado

| Métrica | Valor |
| :--- | :--- |
| **Señal de Trading** | **`{signal.name}`** |
| **Probabilidad Estimada Alza USD** | **`{prob_up:.1%}`** |
| **Régimen de Mercado (HMM)** | `{current_regime}` |
| **Precio Actual USD/CLP** | **${current_price:,.2f} CLP** |
| **Stop-Loss Dinámico (2.0x ATR)** | **${entry_stop:,.2f} CLP** |
| **Take-Profit Sugerido (3.5x ATR)** | **${entry_target:,.2f} CLP** |
| **Nivel de VIX** | `{current_vix:.1f}` |

---

## 📈 Rendimiento Histórico de Simulación (Walk-Forward OOS)

| Métrica de Desempeño | Valor |
| :--- | :--- |
| **Precisión Direccional Promedio** | **`{float(arr.mean()):.1%}`** |
| **Rango de DA (Mín / Máx)** | `[{float(arr.min()):.1%}, {float(arr.max()):.1%}]` |
| **Retorno Acumulado Simulado** | **`{backtest.cumulative_return*100:+.2f}%`** |
| **Retorno Anualizado (CAGR)** | **`{backtest.annualized_return*100:+.2f}%`** |
| **Sharpe Ratio** | **`{backtest.sharpe_ratio:.3f}`** |
| **Sortino Ratio** | **`{backtest.sortino_ratio:.3f}`** |
| **Profit Factor** | **`{backtest.profit_factor:.2f}`** |
| **Win Rate en Operaciones** | **`{backtest.win_rate*100:.1f}%`** ({backtest.total_trades} trades) |
| **Máximo Drawdown** | **`{backtest.max_drawdown*100:.2f}%`** |

---

## 🔍 Factores de Mayor Impacto (Explicabilidad SHAP)

```
"""
        if shap_top is not None:
            for feat, val in shap_top.items():
                md_content += f"{feat:<35} : {val:+.6f}\n"
        md_content += """```

---
*Generado automáticamente por el Pipeline Predictor USD/CLP v5.0 Enterprise.*
"""
        md_path = self.output_dir / "usdclp_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info("Reporte JSON exportado en: %s", json_path.resolve())
        logger.info("Reporte Markdown exportado en: %s", md_path.resolve())
        return json_path, md_path

    def save_predictions_csv(self, feature_matrix: pd.DataFrame,
                             ensemble_model: HybridCalibratedEnsemble,
                             feature_selector: Optional[FeatureSelector] = None,
                             filename: str = "usdclp_predictions.csv") -> Path:
        Xp = feature_matrix.ffill().bfill().fillna(0)
        if feature_selector is not None and feature_selector._is_fitted:
            Xp = feature_selector.transform(Xp)
        proba  = ensemble_model.predict_proba(Xp).rename("prob_usd_up")
        regime = ensemble_model.regime_detector.predict_regime(Xp).apply(lambda r: r.name)
        pred   = ensemble_model.predict(Xp).rename("pred_direction")
        signal_col = proba.apply(
            lambda p: "BUY_USD" if p >= 0.54 else ("SELL_USD" if p <= 0.44 else "HOLD")
        ).rename("signal")
        out = pd.concat([proba, regime, pred, signal_col], axis=1)
        path = self.output_dir / filename
        out.to_csv(path)
        logger.info("Predicciones completas guardadas en: %s", path.resolve())
        return path


# ===========================================================================
# 15. FUNCIÓN DE EJECUCIÓN PRINCIPAL
# ===========================================================================

def _build_volatility_adjusted_target(close: pd.Series, horizon: int = 10) -> pd.Series:
    fwd_ret = close.shift(-horizon) / close - 1.0
    return (fwd_ret > 0.0).astype(int).rename("target")


def run_pipeline(force_retrain: bool = False, horizon: int = 10, cache_ttl: int = _CACHE_TTL_HOURS,
                 notify: bool = False) -> None:
    logger.info("=" * 70)
    logger.info("  USD/CLP ALGORITHMIC ENGINE v5.0 Enterprise")
    logger.info("  Directorio de Cache : %s", _CACHE_DIR)
    logger.info("  Modelo Persistente  : %s", _MODEL_PATH)
    logger.info("  Horizonte Objetivo  : %d dias habiles", horizon)
    logger.info("=" * 70)

    # 1. Descarga autónoma de datos
    fetcher = AutoDataFetcher(start="2013-01-01", ttl_hours=cache_ttl)
    price_df, macro_df, risk_df = fetcher.fetch_all()

    # 2. Ingeniería de características y ratios
    logger.info("Construyendo matriz de caracteristicas macroestructurales...")
    feature_eng = FeatureEngineer()
    full_df     = feature_eng.build_feature_matrix(price_df, macro_df, risk_df, price_col="close")
    target      = _build_volatility_adjusted_target(price_df["close"], horizon=horizon)

    _raw = {"open", "high", "low", "close", "volume", "target", "adj close",
            "dividends", "stock splits", "open_x", "high_x", "low_x", "close_x",
            "volume_x", "open_y", "high_y", "low_y", "close_y", "volume_y"}
    feature_cols = [c for c in full_df.columns
                    if c.lower() not in _raw and pd.api.types.is_numeric_dtype(full_df[c])]

    combined       = pd.concat([full_df[feature_cols], target], axis=1).dropna()
    feature_matrix = combined[feature_cols]
    target_aligned = combined["target"]

    logger.info("Matriz de entrenamiento: %d filas × %d columnas | Clases {0: %d, 1: %d}",
                len(feature_matrix), len(feature_cols),
                int((target_aligned == 0).sum()), int((target_aligned == 1).sum()))

    # 3. Comprobar modelo persistido
    persistence = ModelPersistenceManager()
    saved_model = None if force_retrain else persistence.load()
    skip_train  = False

    if saved_model is not None:
        try:
            ensemble         = saved_model["ensemble"]
            feature_selector = saved_model["feature_selector"]
            fold_scores      = saved_model["fold_scores"]
            oos_preds        = saved_model["oos_predictions"]
            skip_train       = True
        except KeyError:
            skip_train = False

    # 4. Entrenamiento si es necesario
    if not skip_train:
        cross_val = PurgedKFoldEmbargo(n_splits=5, label_horizon=horizon, embargo_pct=0.01)
        walk_fwd  = WalkForwardOrchestrator(cross_validator=cross_val, min_train_size=600, step_size=63)
        regime_det = HmmRegimeDetector(n_states=3, n_iter=300, n_restarts=5)
        ensemble   = HybridCalibratedEnsemble(regime_detector=regime_det, linear_weight=0.20)
        feature_selector = FeatureSelector(max_features=35, correlation_threshold=0.90)
        smoother   = SignalSmoother(span=3)
        risk_mgr   = RiskManager(RiskLimits())
        decision   = DecisionMatrix(smoother, risk_mgr, buy_threshold=0.54, sell_threshold=0.44)

        pipeline = UsdClpPredictionPipeline(
            feature_engineer = feature_eng,
            walk_forward     = walk_fwd,
            ensemble_model   = ensemble,
            decision_matrix  = decision,
            feature_selector = feature_selector,
        )

        logger.info("Iniciando ciclo de validacion Walk-Forward cruzada purgada...")
        fold_scores = pipeline.run_training_and_validation(feature_matrix, target_aligned)
        oos_preds   = pipeline.oos_predictions_

        if not fold_scores:
            logger.error("No se generaron folds de validacion.")
            return

        persistence.save({
            "ensemble":         ensemble,
            "feature_selector": feature_selector,
            "fold_scores":      fold_scores,
            "oos_predictions":  oos_preds,
        })
    else:
        cross_val = PurgedKFoldEmbargo(n_splits=5, label_horizon=horizon, embargo_pct=0.01)
        walk_fwd  = WalkForwardOrchestrator(cross_validator=cross_val, min_train_size=600, step_size=63)
        smoother  = SignalSmoother(span=3)
        risk_mgr  = RiskManager(RiskLimits())
        decision  = DecisionMatrix(smoother, risk_mgr, buy_threshold=0.54, sell_threshold=0.44)
        pipeline  = UsdClpPredictionPipeline(
            feature_engineer = feature_eng,
            walk_forward     = walk_fwd,
            ensemble_model   = ensemble,
            decision_matrix  = decision,
            feature_selector = feature_selector,
        )
        pipeline.oos_predictions_ = oos_preds
        pipeline.shap_module = ShapExplainabilityModule(ensemble)

    # 5. Backtesting Basado en Eventos
    backtest_metrics: Optional[BacktestMetrics] = None
    if oos_preds is not None and not oos_preds.empty:
        try:
            backtester = EventDrivenBacktester(transaction_cost_pct=0.0005, holding_period=horizon)
            backtest_metrics = backtester.run(
                close_prices  = price_df["close"],
                predictions   = oos_preds["pred"],
                probabilities = oos_preds["proba"],
            )
        except Exception as exc:
            logger.warning("Error en Backtester: %s", exc)

    # 6. Inferencia en Tiempo Real
    latest       = feature_matrix.iloc[[-1]]
    sma_cross    = float(full_df["sma_cross"].iloc[-1]) if "sma_cross" in full_df.columns else 1.0
    momentum_ser = pd.Series([sma_cross], index=latest.index)
    live_vix     = float(risk_df["vix"].dropna().iloc[-1]) if "vix" in risk_df.columns and not risk_df["vix"].dropna().empty else 20.0
    curr_price   = float(price_df["close"].iloc[-1])
    curr_atr     = float(full_df["atr_14"].dropna().iloc[-1]) if "atr_14" in full_df.columns else curr_price * 0.01

    signal_ser   = pipeline.generate_live_signal(
        latest_features       = latest,
        momentum_confirmation = momentum_ser,
        current_vix           = live_vix,
        current_equity        = 1.0,
    )
    live_signal = signal_ser.iloc[0]

    direction = 1 if live_signal == TradingSignal.BUY_USD else (-1 if live_signal == TradingSignal.SELL_USD else 0)
    stop_price   = risk_mgr.compute_trailing_stop(curr_price, curr_atr, direction) if direction != 0 else curr_price
    target_price = risk_mgr.compute_take_profit(curr_price, curr_atr, direction)   if direction != 0 else curr_price

    latest_sel = (feature_selector.transform(latest.ffill().bfill().fillna(0))
                  if feature_selector._is_fitted else latest)
    prob_up = float(ensemble.predict_proba(latest_sel).iloc[0])
    current_regime = ensemble.regime_detector.predict_regime(latest_sel).iloc[0].name
    signal_date_str = str(latest.index[-1].date())

    # 7. Explicabilidad SHAP
    shap_top5: Optional[pd.Series] = None
    try:
        regime_enum = ensemble.regime_detector.predict_regime(latest_sel).iloc[0]
        shap_exp    = pipeline.explain_last_decision(latest, regime_enum)
        shap_top5   = shap_exp.abs().sort_values(ascending=False).head(10)
    except Exception as exc:
        logger.warning("SHAP omitido: %s", exc)

    # 8. Reportes
    reporter = ResultsReporter(output_dir=Path("."))
    reporter.print_console_summary(
        fold_scores   = fold_scores,
        signal        = live_signal,
        signal_date   = signal_date_str,
        current_vix   = live_vix,
        current_price = curr_price,
        entry_stop    = stop_price,
        entry_target  = target_price,
        shap_top5     = shap_top5,
        backtest      = backtest_metrics,
    )

    reporter.export_markdown_and_json(
        fold_scores    = fold_scores,
        signal         = live_signal,
        signal_date    = signal_date_str,
        current_price  = curr_price,
        current_vix    = live_vix,
        current_regime = current_regime,
        entry_stop     = stop_price,
        entry_target   = target_price,
        prob_up        = prob_up,
        shap_top       = shap_top5,
        backtest       = backtest_metrics,
    )

    predictions_path = reporter.save_predictions_csv(feature_matrix, ensemble, feature_selector)

    # 9. Dashboard Gráfico
    try:
        preds_df = pd.read_csv(predictions_path, index_col=0, parse_dates=True)
        dashboard_path = Path("usdclp_dashboard.png")
        VisualDashboardGenerator.render_dashboard(
            price_df       = price_df,
            full_df        = full_df,
            predictions_df = preds_df,
            shap_top       = shap_top5,
            signal         = live_signal,
            output_path    = dashboard_path,
        )
    except Exception as exc:
        logger.warning("No se pudo generar dashboard grafico: %s", exc)

    # 10. Notificaciones Automáticas
    if notify or os.environ.get("NOTIFY_ALERTS", "").lower() in ("1", "true", "yes"):
        notifier = NotificationManager()
        notifier.broadcast(
            signal      = live_signal,
            price       = curr_price,
            prob_up     = prob_up,
            regime      = current_regime,
            stop_loss   = stop_price,
            take_profit = target_price,
            date_str    = signal_date_str,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictor Algorítmico USD/CLP v5.0")
    parser.add_argument("--force-retrain", action="store_true", help="Forzar reentrenamiento completo ignorando modelo cacheado")
    parser.add_argument("--horizon", type=int, default=10, help="Horizonte de predicción en días hábiles (default: 10)")
    parser.add_argument("--cache-ttl", type=int, default=_CACHE_TTL_HOURS, help="Horas de validez de caché de datos")
    parser.add_argument("--notify", action="store_true", help="Enviar alertas por Telegram / Discord / Desktop si están configuradas")
    parser.add_argument("--loop-hours", type=float, default=None, help="Ejecutar en bucle continuo cada N horas")
    args = parser.parse_args()

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    if args.loop_hours is not None:
        logger.info("Iniciando modo demonio/servicio: ejecucion cada %.1f horas", args.loop_hours)
        while True:
            try:
                run_pipeline(force_retrain=args.force_retrain, horizon=args.horizon,
                             cache_ttl=args.cache_ttl, notify=True)
            except Exception as exc:
                logger.error("Error en ciclo de ejecucion programada: %s", exc)
            time.sleep(args.loop_hours * 3600)
    else:
        run_pipeline(force_retrain=args.force_retrain, horizon=args.horizon,
                     cache_ttl=args.cache_ttl, notify=args.notify)


if __name__ == "__main__":
    main()
